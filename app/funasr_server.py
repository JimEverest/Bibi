#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR模型服务器
保持模型在内存中，通过stdin/stdout进行通信，同时提供最小CLI用于本地文件转写测试。
"""

import argparse
import json
import logging
import traceback
import signal
import os
import sys
import warnings
import time

import numpy as np

# 过滤掉 jieba 的 pkg_resources 弃用警告
warnings.filterwarnings("ignore", category=UserWarning, module="jieba._compat")

# 在导入任何深度学习库之前设置环境变量
os.environ.setdefault("OMP_NUM_THREADS", "8")  # ONNX 推理并行线程数，可提升速度
# 默认使用 CPU 进行推理；如需使用 GPU，可在外部设置环境变量 FUNASR_DEVICE=cuda:0
os.environ.setdefault("FUNASR_DEVICE", "cpu")

from app.funasr_config import ASR_ENGINE, MODEL_REVISION, MODELS, SENSEVOICE_MODEL_NAME
from app.download_models import get_model_cache_path
from app.logging_config import setup_logging


logger = logging.getLogger(__name__)


def build_streaming_speech_detector(vad_cfg: dict, sample_rate: int):
    """构建基于能量阈值的流式语音活动检测器。

    `Fsmn_vad_online` 返回的是边界事件而非逐帧语音标志，因此这里使用有状态的
    RMS 能量阈值检测器，由现有（此前未使用的）`vad.*` 配置项驱动。
    """
    start_threshold = float(vad_cfg.get("start_threshold", 0.02))
    stop_threshold = float(vad_cfg.get("stop_threshold", 0.01))
    min_speech_ms = float(vad_cfg.get("min_speech_ms", 300))
    min_silence_ms = float(vad_cfg.get("min_silence_ms", 200))

    state = {"active": False, "speech_ms": 0.0, "silence_ms": 0.0}

    def detect(samples: np.ndarray, is_final: bool = False) -> bool:
        if samples.size == 0:
            return state["active"]
        rms = float(np.sqrt(np.mean((samples.astype("float32") / 32768.0) ** 2)))
        chunk_ms = samples.size / sample_rate * 1000.0
        if rms >= start_threshold:
            state["speech_ms"] += chunk_ms
            state["silence_ms"] = 0.0
            if state["speech_ms"] >= min_speech_ms:
                state["active"] = True
        elif rms <= stop_threshold:
            state["silence_ms"] += chunk_ms
            state["speech_ms"] = 0.0
            if state["silence_ms"] >= min_silence_ms:
                state["active"] = False
        return state["active"]

    return detect


class FunASRServer:
    def __init__(self):
        self.asr_model = None
        self.vad_model = None
        self.punc_model = None
        self.initialized = False
        self.running = True
        self.transcription_count = 0  # 转录计数器
        self.total_audio_duration = 0.0  # 总音频时长
        self.engine = ASR_ENGINE  # sensevoice 或 paraformer

        # 使用统一配置
        self.model_revision = MODEL_REVISION
        self.model_names = {
            "asr": MODELS["asr"]["name"],
            "vad": MODELS["vad"]["name"],
            "punc": MODELS["punc"]["name"],
        }
        if self.engine == "sensevoice":
            # SenseVoice 自带标点/ITN，无需外挂标点模型
            self.model_names["asr"] = SENSEVOICE_MODEL_NAME
            self.model_names.pop("punc", None)

        self.device = self._select_device()
        logger.info(
            "FunASR服务器初始化，引擎=%s，模型版本=%s，设备=%s",
            self.engine,
            self.model_revision,
            self.device,
        )

        # 设置信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def __del__(self):
        """析构函数，确保释放模型资源"""
        try:
            self.cleanup()
        except Exception as e:
            logger.debug(f"析构函数清理时出错: {str(e)}")

    def cleanup(self):
        """清理所有模型和资源"""
        logger.info("开始清理 FunASR 服务器资源")
        try:
            # 清理模型引用（ONNX 的 InferenceSession 会在对象销毁时自动释放）
            if self.asr_model is not None:
                logger.debug("释放 ASR 模型")
                self.asr_model = None
            
            if self.vad_model is not None:
                logger.debug("释放 VAD 模型")
                self.vad_model = None
            
            if self.punc_model is not None:
                logger.debug("释放标点模型")
                self.punc_model = None
            
            # 执行最后一次内存清理（包括 gc.collect 强制回收）
            self._cleanup_memory()
            
            logger.info("FunASR 服务器资源清理完成")
        except Exception as e:
            logger.error(f"清理 FunASR 资源时出错: {str(e)}")

    def _signal_handler(self, signum, frame):
        """处理退出信号，清理资源后正常退出"""
        logger.info(f"收到信号 {signum}，准备退出...")
        self.running = False
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"信号处理中清理资源失败: {str(e)}")
        # 正常退出
        sys.exit(0)

    def _select_device(self):
        """自动选择推理设备"""
        env_device = os.environ.get("FUNASR_DEVICE")
        if env_device:
            logger.info("使用环境变量指定的设备: %s", env_device)
            return env_device

        return "cpu"

    def _load_asr_model(self):
        """加载ASR模型"""
        try:
            model_name_lower = str(self.model_names["asr"]).lower()

            if self.engine == "sensevoice":
                return self._load_sensevoice_model()

            # 如果是 ONNX 模型，使用 funasr_onnx 专用加载器
            if "onnx" in model_name_lower:
                from funasr_onnx.paraformer_bin import Paraformer

                logger.info("开始加载ASR ONNX模型: %s", self.model_names["asr"])
                try:
                    model_dir = get_model_cache_path(
                        self.model_names["asr"],
                        self.model_revision
                    )
                except Exception as e:
                    logger.error("下载 ASR ONNX 模型失败: %s", e)
                    return False

                # 基本完整性校验，优先使用量化模型
                quant_file = os.path.join(model_dir, "model_quant.onnx")
                base_file = os.path.join(model_dir, "model.onnx")
                use_quantize = False
                if os.path.exists(quant_file):
                    use_quantize = True
                elif not os.path.exists(base_file):
                    logger.error("ASR 模型目录缺少 model.onnx: %s", model_dir)
                    return False

                device_id = -1  # CPU
                if self.device and "cuda" in self.device:
                    try:
                        device_id = int(self.device.split(":")[-1])
                    except Exception:
                        device_id = 0
                
                # 性能优化参数
                num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))

                self.asr_model = Paraformer(
                    str(model_dir),
                    batch_size=1,
                    device_id=device_id,
                    quantize=use_quantize,
                    intra_op_num_threads=num_threads,  # 线程并行加速
                )
                logger.info("ASR ONNX模型加载完成")
                return True
            else:
                logger.error("仅支持 ONNX 模型加载，当前模型名称: %s", self.model_names["asr"]) 
                return False
                
        except Exception as e:
            logger.error(f"ASR模型加载失败: {str(e)}")
            logger.debug(traceback.format_exc())
            self.asr_model = None
            return False

    def _load_sensevoice_model(self):
        """加载 SenseVoice ONNX 模型（中英混说，自带标点）"""
        try:
            from funasr_onnx.sensevoice_bin import SenseVoiceSmall

            logger.info("开始加载 SenseVoice ONNX 模型: %s", self.model_names["asr"])
            model_dir = os.path.join(
                os.path.expanduser("~"),
                ".cache", "modelscope", "hub", "models",
                self.model_names["asr"].replace("/", os.sep),
            )
            if not os.path.isdir(model_dir):
                # 本地不存在则下载（内部会调 modelscope）
                model_dir = self.model_names["asr"]

            quant_file = os.path.join(model_dir, "model_quant.onnx")
            use_quantize = os.path.exists(quant_file)
            if not use_quantize and not os.path.exists(os.path.join(model_dir, "model.onnx")):
                logger.error("SenseVoice 模型目录缺少 onnx 文件: %s", model_dir)
                return False

            num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
            self.asr_model = SenseVoiceSmall(
                str(model_dir),
                batch_size=1,
                device_id=-1,
                quantize=use_quantize,
                intra_op_num_threads=num_threads,
            )
            logger.info("SenseVoice ONNX模型加载完成 (quantize=%s)", use_quantize)
            return True
        except Exception as e:
            logger.error(f"SenseVoice模型加载失败: {str(e)}")
            logger.debug(traceback.format_exc())
            self.asr_model = None
            return False

    def _load_vad_model(self):
        """加载VAD模型（使用 funasr_onnx 专用加载器）"""
        try:
            from funasr_onnx.vad_bin import Fsmn_vad

            logger.info("开始加载VAD ONNX模型: %s", self.model_names["vad"])
            try:
                model_dir = get_model_cache_path(
                    self.model_names["vad"],
                    self.model_revision
                )
            except Exception as e:
                logger.error("下载 VAD ONNX 模型失败: %s", e)
                return False

            quant_file = os.path.join(model_dir, "model_quant.onnx")
            base_file = os.path.join(model_dir, "model.onnx")
            use_quantize = False
            if os.path.exists(quant_file):
                use_quantize = True
            elif not os.path.exists(base_file):
                logger.error("VAD 模型目录缺少 model.onnx: %s", model_dir)
                return False

            device_id = -1  # CPU
            if self.device and "cuda" in self.device:
                try:
                    device_id = int(self.device.split(":")[-1])
                except Exception:
                    device_id = 0
            
            num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))

            self.vad_model = Fsmn_vad(
                str(model_dir),
                batch_size=1,
                device_id=device_id,
                quantize=use_quantize,
                intra_op_num_threads=num_threads,
            )
            logger.info("VAD ONNX模型加载完成")
            return True
        except Exception as e:
            logger.error(f"VAD模型加载失败: {str(e)}")
            logger.debug(traceback.format_exc())
            self.vad_model = None
            return False

    def _load_punc_model(self):
        """加载标点恢复模型（使用 funasr_onnx 专用加载器）"""
        try:
            from funasr_onnx.punc_bin import CT_Transformer

            logger.info("开始加载标点恢复 ONNX模型: %s", self.model_names["punc"])
            try:
                model_dir = get_model_cache_path(
                    self.model_names["punc"],
                    self.model_revision
                )
            except Exception as e:
                logger.error("下载 标点 ONNX 模型失败: %s", e)
                return False

            quant_file = os.path.join(model_dir, "model_quant.onnx")
            base_file = os.path.join(model_dir, "model.onnx")
            use_quantize = False
            if os.path.exists(quant_file):
                use_quantize = True
            elif not os.path.exists(base_file):
                logger.error("标点模型目录缺少 model.onnx: %s", model_dir)
                return False

            device_id = -1  # CPU
            if self.device and "cuda" in self.device:
                try:
                    device_id = int(self.device.split(":")[-1])
                except Exception:
                    device_id = 0
            
            num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))

            self.punc_model = CT_Transformer(
                str(model_dir),
                batch_size=1,
                device_id=device_id,
                quantize=use_quantize,
                intra_op_num_threads=num_threads,
            )
            logger.info("标点恢复 ONNX模型加载完成")
            return True
        except Exception as e:
            logger.error(f"标点恢复模型加载失败: {str(e)}")
            logger.debug(traceback.format_exc())
            self.punc_model = None
            return False

    def initialize(self):
        """并行初始化FunASR模型"""
        if self.initialized:
            return {"success": True, "message": "模型已初始化"}

        try:
            import threading

            logger.info("正在并行初始化FunASR模型...")
            start_time = time.time()

            # 预导入 funasr_onnx 子模块，避免多线程导入导致的 ModuleLock 死锁
            try:
                import importlib
                pre_modules = (
                    "funasr_onnx.utils.utils",
                    "funasr_onnx.utils.frontend",
                    "funasr_onnx.paraformer_bin",
                    "funasr_onnx.vad_bin",
                    "funasr_onnx.punc_bin",
                )
                for m in pre_modules:
                    importlib.import_module(m)
                logger.info("funasr_onnx 模块预导入完成")
            except Exception as pre_e:
                logger.warning("funasr_onnx 预导入失败: %s", str(pre_e))

            # 创建加载结果存储
            results = {}

            def load_model_thread(model_name, load_func):
                """模型加载线程包装函数"""
                thread_start = time.time()
                results[model_name] = load_func()
                thread_time = time.time() - thread_start
                logger.info(f"{model_name}模型加载线程耗时: {thread_time:.2f}秒")

            # 根据开关决定是否加载 VAD / PUNC（默认启用）
            # SenseVoice 引擎下 punc 模型仅用于中文文本补标点，仍加载但失败不阻塞
            load_vad = os.environ.get("FUNASR_USE_VAD", "false").lower() not in ("0", "false", "no")
            load_punc = (
                os.environ.get("FUNASR_USE_PUNC", "true").lower() not in ("0", "false", "no")
                and "punc" in self.model_names
            )

            # 创建并启动线程（ASR 必须，VAD/PUNC 可选）
            threads = [
                threading.Thread(
                    target=load_model_thread,
                    args=("asr", self._load_asr_model),
                    daemon=True,
                )
            ]
            if load_vad:
                threads.append(
                    threading.Thread(
                        target=load_model_thread,
                        args=("vad", self._load_vad_model),
                        daemon=True,
                    )
                )
            if load_punc:
                threads.append(
                    threading.Thread(
                        target=load_model_thread,
                        args=("punc", self._load_punc_model),
                        daemon=True,
                    )
                )

            # 启动所有线程
            for thread in threads:
                thread.start()

            # 等待所有线程完成，设置超时
            timeout_occurred = False
            for thread in threads:
                thread.join(timeout=300)  # 5分钟超时
                if thread.is_alive():
                    timeout_occurred = True
                    logger.error("模型加载线程超时，线程仍在运行")
            
            # 检查是否有超时
            if timeout_occurred:
                return {
                    "success": False,
                    "error": "模型加载超时（超过5分钟）",
                    "type": "timeout_error",
                }

            # 检查加载结果
            failed_models = [name for name, success in results.items() if not success]

            if failed_models:
                error_msg = f"以下模型加载失败: {', '.join(failed_models)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "type": "init_error"}

            total_time = time.time() - start_time
            self.initialized = True
            logger.info(
                f"所有FunASR模型并行初始化完成，总耗时: {total_time:.2f}秒"
            )
            
            # 预热librosa，避免首次load时的初始化延迟
            self._warmup_librosa()
            
            return {
                "success": True,
                "message": f"FunASR模型并行初始化成功，耗时: {total_time:.2f}秒",
            }

        except ImportError as e:
            error_msg = "FunASR未安装，请先安装FunASR: pip install funasr"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "type": "import_error"}

        except Exception as e:
            error_msg = f"FunASR模型初始化失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg, "type": "init_error"}

    def transcribe_audio(self, audio_path, options=None):
        """转录音频文件"""
        if not self.initialized:
            init_result = self.initialize()
            if not init_result["success"]:
                return init_result

        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                return {"success": False, "error": f"音频文件不存在: {audio_path}"}

            logger.info(f"开始转录音频文件: {audio_path}")

            # 设置默认选项
            default_options = {
                "batch_size_s": 60,
                "hotword": "",
                # 默认启用 VAD / PUNC，可在外部通过选项或环境变量关闭
                "use_vad": os.environ.get("FUNASR_USE_VAD", "false").lower() not in ("0", "false", "no"),
                "use_punc": os.environ.get("FUNASR_USE_PUNC", "true").lower() not in ("0", "false", "no"),
                "language": "zh",
            }

            if options:
                default_options.update(options)

            # 执行语音识别（VAD 处理）
            if default_options["use_vad"] and self.vad_model:
                # funasr_onnx.Fsmn_vad 直接调用，返回 segments [[start_ms, end_ms], ...]
                vad_result = self.vad_model(audio_path)
                logger.info("VAD处理完成，检测到 %s 个语音段", len(vad_result[0]) if vad_result else 0)
            elif default_options["use_vad"] and not self.vad_model:
                logger.warning("use_vad=True 但VAD模型未加载，跳过VAD处理")

            # 执行ASR识别（根据模型类型使用不同接口）
            if self.engine == "sensevoice":
                # SenseVoice: 直接调用，返回 [tagged_text]；
                # language: auto/zh/en/ja/ko/yue；use_itn 启用数字规范化
                asr_result = self.asr_model(
                    [audio_path],
                    language=default_options.get("language", "auto"),
                    use_itn=True,
                )
            elif hasattr(self.asr_model, "generate"):
                # PyTorch 模型使用 generate 方法
                asr_result = self.asr_model.generate(
                    input=audio_path,
                    batch_size_s=default_options["batch_size_s"],
                    hotword=default_options["hotword"],
                    cache={},
                )
            else:
                # ONNX 模型直接调用（funasr_onnx.Paraformer）
                asr_result = self.asr_model([audio_path])

            # 提取识别文本（兼容 PyTorch / ONNX / SenseVoice 三种格式）
            if isinstance(asr_result, list) and len(asr_result) > 0:
                first_item = asr_result[0]
                # SenseVoice 格式: ["<|zh|><|NEUTRAL|>...<|woitn|>文本"]
                if isinstance(first_item, str):
                    raw_text = first_item
                # PyTorch 格式: [{"text": "..."}]
                elif isinstance(first_item, dict) and "text" in first_item:
                    raw_text = first_item["text"]
                # ONNX 格式: [{"preds": (text_string, token_list)}]
                elif isinstance(first_item, dict) and "preds" in first_item:
                    preds = first_item["preds"]
                    if isinstance(preds, tuple) and len(preds) > 0:
                        raw_text = str(preds[0])
                    else:
                        raw_text = str(preds)
                else:
                    raw_text = str(first_item)
            else:
                raw_text = ""

            # SenseVoice 输出带 <|zh|><|HAPPY|><|Speech|><|woitn|> 标签，剥离
            if raw_text and raw_text.startswith("<|"):
                import re
                raw_text = re.sub(r"<\|[^|]*\|>", "", raw_text).strip()

            logger.info(f"ASR识别完成，原始文本: {raw_text[:100]}...")

            # 使用标点恢复（ONNX 的 CT_Transformer 直接调用）
            # SenseVoice 的 ITN 输出无标点（woitn 模式），中文场景补充标点模型
            final_text = raw_text
            if (
                self.engine == "sensevoice"
                and self.punc_model
                and raw_text.strip()
                and self._contains_cjk(raw_text)
            ):
                try:
                    punc_result = self.punc_model(raw_text)
                    if isinstance(punc_result, tuple) and len(punc_result) > 0:
                        final_text = str(punc_result[0])
                    else:
                        final_text = str(punc_result)
                    logger.info("标点恢复完成")
                except Exception as e:
                    logger.warning(f"标点恢复失败，使用原始文本: {str(e)}")
            elif default_options["use_punc"] and self.punc_model and raw_text.strip():
                try:
                    # funasr_onnx.CT_Transformer 返回 (text_with_punc, punc_list)
                    punc_result = self.punc_model(raw_text)
                    if isinstance(punc_result, tuple) and len(punc_result) > 0:
                        final_text = str(punc_result[0])
                    else:
                        final_text = str(punc_result)
                    logger.info("标点恢复完成")
                except Exception as e:
                    logger.warning(f"标点恢复失败，使用原始文本: {str(e)}")

            duration = self._get_audio_duration(audio_path)
            self.transcription_count += 1

            result = {
                "success": True,
                "text": final_text,
                "raw_text": raw_text,
                # 注意：静音/噪声音频时 funasr_onnx 返回空列表 []，
                # 直接取 asr_result[0] 会 IndexError，必须先判空
                "confidence": (
                    getattr(asr_result[0], "confidence", 0.0)
                    if isinstance(asr_result, list) and len(asr_result) > 0
                    else 0.0
                ),
                "duration": duration,
                "language": "zh-CN",
                "model_type": self.engine,
                "models": self.model_names,
            }

            # 生产环境：每10次转录后进行内存清理
            if self.transcription_count % 10 == 0:
                self._cleanup_memory()
                logger.info(f"已完成 {self.transcription_count} 次转录，执行内存清理")

            logger.info(f"转录完成，最终文本: {final_text[:100]}...")
            return result

        except Exception as e:
            error_msg = f"音频转录失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg, "type": "transcription_error"}

    def transcribe_samples(self, samples, sample_rate: int, options=None):
        """转录内存中的音频采样（用于流式分段识别），内部落地为临时 WAV 文件后复用 transcribe_audio"""
        import tempfile
        import wave

        fd, path = tempfile.mkstemp(prefix="stream_segment_", suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.astype("int16").tobytes())
            return self.transcribe_audio(path, options=options)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _contains_cjk(self, text: str) -> bool:
        """检查文本是否包含中日韩字符（SenseVoice 英文输出无需中文标点恢复）"""
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def _get_audio_duration(self, audio_path):
        """获取音频时长"""
        try:
            import librosa

            duration = librosa.get_duration(path=audio_path)
            self.total_audio_duration += duration  # 累计音频时长
            return duration
        except Exception as e:
            logger.debug(f"获取音频时长失败: {str(e)}")
            return 0.0

    def _warmup_librosa(self):
        """预热librosa库，避免首次load时的初始化延迟（这是真正的问题所在）"""
        try:
            logger.info("开始预热librosa，触发音频库初始化...")
            warmup_start = time.time()
            
            import tempfile
            import numpy as np
            import wave
            
            # 创建一个极短的测试音频（10ms）
            sample_rate = 16000
            samples = int(sample_rate * 0.01)
            audio_data = np.zeros(samples, dtype=np.int16)
            
            # 写入临时WAV文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with wave.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data.tobytes())
            
            try:
                # 调用librosa.load触发初始化（这是funasr_onnx内部使用的）
                import librosa
                _, _ = librosa.load(tmp_path, sr=16000)
                
                warmup_time = time.time() - warmup_start
                logger.info(f"librosa预热完成，耗时: {warmup_time:.2f}秒")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"librosa预热失败（不影响使用）: {str(e)}")
    
    def _cleanup_memory(self):
        """生产环境内存清理"""
        try:
            import gc
            # 执行垃圾回收
            gc.collect()
            logger.info("内存清理完成")
        except Exception as e:
            logger.warning(f"内存清理失败: {str(e)}")


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="FunASR 离线音频转写 CLI（基于 funasr_server.py）"
    )
    parser.add_argument(
        "--audio",
        "-a",
        required=True,
        help="需要转写的音频文件路径，支持 funasr 支持的格式",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="禁用 FunASR VAD 处理（默认启用）",
    )
    parser.add_argument(
        "--no-punc",
        action="store_true",
        help="禁用 FunASR 标点恢复（默认启用）",
    )
    parser.add_argument(
        "--language",
        "-l",
        help="识别语言代码，例如 zh、en、auto 等，默认使用服务器内置配置",
    )
    parser.add_argument(
        "--hotword",
        help="识别时使用的热词字符串",
    )
    parser.add_argument(
        "--batch-size-s",
        type=float,
        help="动态 batch 总时长（秒），默认 60",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="使用缩进格式输出 JSON 结果",
    )
    return parser


def main():
    # 配置日志（CLI模式，使用统一配置）
    # funasr_server作为独立脚本，日志保存到项目根目录的logs/
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, "logs")
    setup_logging("INFO", log_dir)
    
    parser = _build_cli_parser()
    args = parser.parse_args()

    server = FunASRServer()
    init_result = server.initialize()
    success = init_result.get("success", False)

    indent = 2 if args.pretty else None

    if not success:
        print(json.dumps(init_result, ensure_ascii=False, indent=indent))
        raise SystemExit(1)

    options = {}
    if args.no_vad:
        options["use_vad"] = False
    if args.no_punc:
        options["use_punc"] = False
    if args.language:
        options["language"] = args.language
    if args.hotword:
        options["hotword"] = args.hotword
    if args.batch_size_s is not None:
        options["batch_size_s"] = args.batch_size_s

    result = server.transcribe_audio(args.audio, options=options)
    print(json.dumps(result, ensure_ascii=False, indent=indent))

    if not result.get("success", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
