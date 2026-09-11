"""Configuration helpers for the speak-keyboard runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "hotkeys": {
        "toggle": "f2",
        "toggle_enabled": True,
        # 按住说话（push-to-talk）：组合键全部按下开始录音，任一键松开停止。
        # 设为 "none" 或空可禁用
        "push_to_talk": "win+ctrl+alt",
        "push_to_talk_enabled": True,
    },
    "audio": {
        "sample_rate": 16000,
        "block_ms": 20,
        # None = 系统默认输入设备；也可填 sounddevice 设备编号
        "device": None,
        # 单次录音的最大大小（字节）
        # 16kHz*2字节=32KB/s；约 60s ≈ 1.9MB。超过自动停止并转入识别，
        # 防止意外长录音导致分钟级推理卡顿（CPU 推理下 60s 已远超日常语句）
        "max_session_bytes": 2 * 1024 * 1024,
        # 输入增益（float），录完的音频在送识别前统一乘以该系数
        # 内置麦克风音量小时可调大（例如 8.0 ~ 15.0）
        "gain": 12.0,
        # 是否持久化保存录音 WAV 到 logs/（调试用）。默认关闭，避免磁盘膨胀
        "save_recordings": False,
    },
    "vad": {
        "start_threshold": 0.02,
        "stop_threshold": 0.01,
        "min_speech_ms": 300,
        "min_silence_ms": 200,
        "pad_ms": 200,
    },
    # 识别后端：funasr（本地离线）或 volcengine（火山引擎云端流式）
    "backend": "funasr",
    "asr": {
        "use_vad": False,
        "use_punc": True,
        "language": "zh",
        "hotword": "",
        "batch_size_s": 60.0,
    },
    # 火山引擎 BigASR 流式识别配置（仅当 backend == "volcengine" 时生效）
    # 文档：https://www.volcengine.com/docs/6561/1354869
    "volcengine": {
        # 在火山引擎控制台 https://console.volcengine.com/speech/app 创建应用后获取
        "app_key": "",
        "access_key": "",
        # 资源 ID，默认按时长计费
        "resource_id": "volc.bigasr.sauc.duration",
        # WebSocket 端点（一般无需修改）
        "url": "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
        # 识别模型名称
        "model_name": "bigmodel",
        # 每次发送的音频时长（毫秒），越小延迟越低
        "chunk_ms": 100,
        # 是否添加标点
        "enable_punc": True,
        # 是否启用数字/格式规范化（ITN）
        "enable_itn": True,
    },
    # LLM 二次润色（方案5）。默认关闭；开启后走 endpoint 调用，
    # 失败自动降级为词典处理后的文本，绝不阻塞语音输入主流程
    "llm": {
        "enabled": False,
        # "openai"（/chat/completions 或 /v1/responses）或 "anthropic"（/v1/messages）
        "schema": "openai",
        # OpenAI 经典端点填到 .../v1/chat/completions；
        # 新 responses 风格填到 .../v1/responses 并把 style 设为 "responses"；
        # Anthropic 填到 .../v1/messages
        "endpoint": "",
        "style": "chat",        # chat | responses（仅 openai schema 有效）
        "api_key": "",
        "model": "",
        "temperature": 0.3,
        "timeout_seconds": 15,
        # 留空用内置默认 prompt（最小化润色）；自定义时用 {TEXT} 占位原文
        "prompt": "",
        # ---- 网络设置（内网代理/自签名证书环境）----
        # HTTP/HTTPS 代理，如 http://proxy.corp.local:8080；留空 = 不使用代理
        "http_proxy": "",
        "https_proxy": "",
        # 跳过 SSL 证书校验（自签名证书/内网中间人劫持场景）。默认 false；
        # 开启可解决 certificate verify failed，但会降低传输安全性
        "disable_ssl_verify": False,
    },
    "output": {
        "dedupe": True,
        "max_history": 5,
        "min_chars": 1,
        "method": "auto",
        "append_newline": False,
    },
    "logging": {"dir": "logs", "level": "INFO"},
    "streaming": {
        "enabled": True,
        "segment_silence_ms": 450,
        "commit_silence_ms": 1200,
        "max_uncommitted_ms": 12000,
        # Onset-smoothing pad kept in the uncommitted buffer after a commit
        # (not a text-splice overlap): gives the next segment a little
        # audio-only context so ASR doesn't start "cold".
        "audio_overlap_ms": 150,
        "preview_context_chars": 30,
        "preview_max_chars": 120,
        "ui_update_debounce_ms": 80,
        "dedicated_model_instance": False,
    },
}


def _merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON file if provided, otherwise defaults."""

    config = dict(DEFAULT_CONFIG)
    if not path:
        return config

    expanded_path = os.path.expanduser(path)
    if not os.path.exists(expanded_path):
        raise FileNotFoundError(f"Config file not found: {expanded_path}")

    with open(expanded_path, "r", encoding="utf-8") as f:
        overrides = json.load(f)

    return _merge_dict(config, overrides)


def ensure_logging_dir(config: Dict[str, Any]) -> str:
    """Ensure the logging directory exists and return its absolute path.
    
    日志目录相对于项目根目录（main.py 所在目录），而不是当前工作目录。
    这样即使从其他目录运行脚本，日志也能正确保存到项目目录下。
    """
    log_dir = config["logging"].get("dir", "logs")
    
    # 如果已经是绝对路径，直接使用
    if os.path.isabs(log_dir):
        pass
    else:
        # 相对路径：基于项目根目录（向上两级到达项目根目录）
        # app/config.py -> app/ -> 项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(project_root, log_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


