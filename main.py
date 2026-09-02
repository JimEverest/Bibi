"""Command-line entry for the speak-keyboard prototype."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import keyboard

from app import HotkeyManager, TranscriptionResult, TranscriptionWorker, load_config, type_text
from app.plugins.dataset_recorder import wrap_result_handler
from app.logging_config import setup_logging


logger = logging.getLogger(__name__)


_TOGGLE_DEBOUNCE_SECONDS = 0.2
_toggle_lock = threading.Lock()
_last_toggle_time = 0.0

# LLM 润色器在 main() 中根据配置初始化（handler 工厂里引用）
_LLM_POLISHER = {"instance": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speak Keyboard prototype")
    parser.add_argument("--config", help="Path to config JSON")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single transcription cycle for debugging",
    )
    parser.add_argument("--save-dataset", action="store_true", help="Persist audio/text pairs")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset output directory")
    parser.add_argument(
        "--no-tray", action="store_true",
        help="不启动系统托盘（纯命令行模式，Ctrl+C 退出）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 单实例保护：多实例并发会导致同一句话被注入两遍（字符交错乱码）
    from app.singleton import acquire_single_instance_lock
    if not acquire_single_instance_lock():
        import sys
        sys.exit(1)

    # 配置文件：优先命令行 --config；否则自动加载项目根的 vocotype_config.json
    # （设置对话框保存的就是这个文件，保证"设置→保存→重启生效"闭环）
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_config_file = os.path.join(project_root, "vocotype_config.json")
    config_file = args.config or (
        default_config_file if os.path.exists(default_config_file) else None
    )
    if config_file:
        logger.info("加载配置文件: %s", config_file)
    config = load_config(config_file)
    
    # 配置日志系统（统一配置）
    from app.config import ensure_logging_dir
    log_dir_abs = ensure_logging_dir(config)
    setup_logging(
        level=config["logging"].get("level", "INFO"),
        log_dir=log_dir_abs
    )

    output_cfg = config.get("output", {})
    output_method = output_cfg.get("method", "auto")
    append_newline = output_cfg.get("append_newline", False)

    # UI 线程模型（关键）：全进程唯一 Tk root 必须在【主线程】创建并泵事件。
    # 浮窗/设置窗口都是它的 Toplevel，其他线程一律 after() 派发，
    # 否则触发 "RuntimeError: main thread is not in main loop"。
    if not args.once:
        from app.ui_hub import get_hub
        get_hub().create_root()

    # 先创建worker（没有回调）
    worker = TranscriptionWorker(
        config_path=config_file,
        on_result=None,  # 稍后设置
    )

    # 初始化 LLM 润色器（方案5，默认 disabled）
    from app.llm_polish import LLMPolisher
    _LLM_POLISHER["instance"] = LLMPolisher(config.get("llm", {}))
    if _LLM_POLISHER["instance"].enabled:
        logger.info(
            "LLM 润色已启用: schema=%s model=%s endpoint=%s",
            _LLM_POLISHER["instance"].schema,
            _LLM_POLISHER["instance"].model,
            _LLM_POLISHER["instance"].endpoint,
        )
    else:
        logger.info("LLM 润色未启用（llm.enabled=false）")

    # 创建result handler（需要worker引用）
    worker.on_result = _make_result_handler(output_method, append_newline, worker)
    if args.save_dataset:
        worker.on_result = wrap_result_handler(worker.on_result, worker, args.dataset_dir)
    
    hotkeys = HotkeyManager()

    toggle_combo = config["hotkeys"].get("toggle", "f2")
    hotkeys.register(toggle_combo, lambda: _toggle(worker))

    # 按住说话（PTT）：组合键全部按下开始录音，任一键松开停止。
    # 与 F2 开关模式并存；PTT 启动的录音只由 PTT 松开来停（避免和 F2 互相干扰）
    ptt_combo = str(config["hotkeys"].get("push_to_talk", "win+ctrl+alt")).strip()
    _ptt_owned = {"recording": False}

    def _ptt_start() -> None:
        if worker.is_running:
            return  # 已在录音（F2 开的或上次 PTT），不重复启动
        _ptt_owned["recording"] = True
        _toggle(worker)

    def _ptt_stop() -> None:
        if _ptt_owned["recording"] and worker.is_running:
            _toggle(worker)
        _ptt_owned["recording"] = False

    if ptt_combo and ptt_combo.lower() not in ("none", "off", "disabled"):
        hotkeys.register_push_to_talk(ptt_combo, _ptt_start, _ptt_stop)

    # 系统托盘（UI）：不干扰 CLI 用法；--no-tray 可关
    tray_app = None
    if not args.no_tray and not args.once:
        try:
            from app.tray import TrayApp

            # 设置对话框 / 托盘菜单共用的配置文件路径（与上面 load_config 一致）
            config_path = os.path.abspath(config_file) if config_file else default_config_file
            tray_app = TrayApp(
                toggle_callback=lambda: _toggle(worker),
                is_recording_fn=lambda: worker.is_running,
                llm_config_path=config_path,
                log_dir=log_dir_abs,
            )
            tray_app.run_detached()
        except Exception as exc:  # noqa: BLE001
            logger.warning("托盘启动失败（继续以纯 CLI 模式运行）: %s", exc)
            tray_app = None

    try:
        logger.info("Speak Keyboard 启动完成，按 %s 开始/停止录音（单击切换，说完再按一次），按 Ctrl+C 退出", toggle_combo)
        if args.once:
            _toggle(worker)
            input("按 Enter 停止并退出...")
            _toggle(worker)
        else:
            # 主线程事件泵：驱动浮窗/设置窗口等所有 Tk UI（Ctrl+C 可中断）
            from app.ui_hub import get_hub
            get_hub().run()
    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
    finally:
        # 清理所有资源
        try:
            worker.stop()
        except Exception as exc:
            logger.debug("停止 worker 时出错: %s", exc)
        
        try:
            worker.cleanup()
        except Exception as exc:
            logger.debug("清理 worker 时出错: %s", exc)
        
        try:
            hotkeys.cleanup()
        except Exception as exc:
            logger.debug("清理热键时出错: %s", exc)
        
        logger.info("所有资源已清理，正常退出")
        import sys
        sys.exit(0)


def _make_result_handler(output_method: str, append_newline: bool, worker: TranscriptionWorker):
    from app.dictionary import TermReplacer
    from app.indicator import get_indicator

    replacer = TermReplacer()
    indicator = get_indicator()

    def _handle_result(result: TranscriptionResult) -> None:
        if result.error:
            logger.error("转写失败: %s", result.error)
            return

        # 术语词典后处理（方案2）：最长优先替换识别错误
        corrected = replacer.replace(result.text)
        if corrected != result.text:
            logger.info("词典替换: %s -> %s", result.text, corrected)

        # LLM 二次润色（方案5）：失败自动降级，绝不阻塞
        final_text = corrected
        if _LLM_POLISHER["instance"] is not None:
            polisher = _LLM_POLISHER["instance"]
            if polisher.enabled:
                indicator.show_polishing()
                # 注入领域词汇提示：术语词典的正确写法，帮助 LLM 纠同音错字
                polished = polisher.polish(corrected, extra_vocab=replacer.vocab_hints())
                if polished:
                    logger.info("LLM润色: %s -> %s", corrected, polished)
                    final_text = polished
                else:
                    logger.info("LLM润色未生效（禁用/失败/超时），使用词典处理文本")

        # 获取转录统计信息
        stats = worker.transcription_stats

        logger.info(
            "转写成功: %s (推理 %.2fs) [已完成 %d/%d，队列剩余 %d]",
            final_text,
            result.inference_latency,
            stats["completed"],
            stats["submitted"],
            stats["pending"],
        )
        type_text(
            final_text,
            append_newline=append_newline,
            method=output_method,
        )
        # 输出完成后隐藏指示浮窗
        try:
            indicator.hide()
        except Exception:
            pass

    return _handle_result


def _toggle(worker: TranscriptionWorker) -> None:
    global _last_toggle_time
    now = time.monotonic()
    with _toggle_lock:
        if now - _last_toggle_time < _TOGGLE_DEBOUNCE_SECONDS:
            logger.debug("忽略快速重复的录音切换请求 (%.3fs)", now - _last_toggle_time)
            return
        _last_toggle_time = now

    from app.indicator import get_indicator
    indicator = get_indicator()

    if worker.is_running:
        # 停止录音，提交转录任务
        worker.stop()
        indicator.show_polishing()
        stats = worker.transcription_stats
        if stats["pending"] > 0:
            logger.info(
                "录音已停止并提交转录，队列中还有 %d 个任务等待处理",
                stats["pending"]
            )
    else:
        # 开始录音
        stats = worker.transcription_stats
        if stats["pending"] > 0:
            logger.info(
                "开始录音（后台还有 %d 个转录任务正在处理）",
                stats["pending"]
            )
        indicator.show_recording()
        worker.start()


if __name__ == "__main__":
    main()

