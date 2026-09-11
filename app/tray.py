# -*- coding: utf-8 -*-
"""
系统托盘（UI 步骤4）

- 托盘图标：待机=灰蓝麦克风、录音中=红、处理中=黄（纯 Pillow 绘制，无图标文件）
- 右键菜单：录音 开/关（等效 F2）、LLM 润色 勾选开关、打开日志目录、退出
- pystray 在独立线程运行；菜单项回调投递到主程序的热键线程安全执行
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

from app.appinfo import APP_NAME, app_version  # noqa: E402

# 状态色
_COLOR_IDLE = (90, 105, 135)      # 灰蓝
_COLOR_REC = (232, 72, 72)        # 红
_COLOR_BUSY = (240, 180, 40)      # 黄

# 兼容旧引用
_app_version = app_version


def _make_icon(color) -> "Image.Image":
    """画一个圆形麦克风风格图标。"""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 麦克风头
    d.rounded_rectangle([22, 8, 42, 34], radius=10, fill=color)
    # 支架弧
    d.arc([16, 18, 48, 46], start=0, end=180, fill=color, width=4)
    # 立杆与底座
    d.line([32, 46, 32, 54], fill=color, width=4)
    d.line([22, 56, 42, 56], fill=color, width=4)
    return img


class TrayApp:
    def __init__(
        self,
        toggle_callback,
        is_recording_fn,
        llm_config_path: str,
        log_dir: str,
        llm_enabled_callback: Callable[[bool], None] | None = None,
        force_stop_callback: Callable[[], None] | None = None,
    ):
        """
        toggle_callback: 线程安全的 F2 等效操作（切录音）
        is_recording_fn: 返回当前是否录音中
        llm_config_path: config json 路径（托盘菜单直接改 llm.enabled）
        log_dir: 日志目录
        """
        import pystray

        self._toggle_cb = toggle_callback
        self._is_recording_fn = is_recording_fn
        self._config_path = llm_config_path
        self._log_dir = log_dir
        self._icon = None
        self._llm_enabled_lock = threading.RLock()
        self._llm_enabled_callback = llm_enabled_callback
        self._force_stop_callback = force_stop_callback
        self._llm_enabled = self._read_llm_enabled()
        self._poll_thread: threading.Thread | None = None

        menu = pystray.Menu(
            pystray.MenuItem("开始/停止录音（托盘）", self._on_toggle, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("强制停止", self._on_force_stop),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: "LLM 润色：开" if self._is_llm_enabled() else "LLM 润色：关",
                self._on_toggle_llm,
            ),
            pystray.MenuItem("设置…", self._on_settings),
            pystray.MenuItem("打开日志目录", self._on_open_logs),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._on_quit),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(lambda item: f"{APP_NAME} v{app_version()}", None, enabled=False),
        )
        self._icon = pystray.Icon(
            "bibi", _make_icon(_COLOR_IDLE), f"{APP_NAME} 语音输入", menu
        )

    # ---------- lifecycle ----------

    def run_detached(self):
        t = threading.Thread(target=self._icon.run, daemon=True, name="Tray")
        t.start()
        self._poll_thread = threading.Thread(target=self._poll_state, daemon=True)
        self._poll_thread.start()
        logger.info("托盘图标已启动")

    def stop(self):
        try:
            self._icon.stop()
        except Exception:
            pass

    # ---------- state polling (icon color) ----------

    def _poll_state(self):
        import time

        last = None
        while True:
            try:
                rec = bool(self._is_recording_fn())
                color = _COLOR_REC if rec else _COLOR_IDLE
                key = (rec, self._is_llm_enabled())
                if key != last:
                    self._icon.icon = _make_icon(color)
                    self._icon.title = (
                        f"{APP_NAME} 语音输入 [录音中]" if rec else f"{APP_NAME} 语音输入"
                    )
                    last = key
            except Exception:
                pass
            time.sleep(0.5)

    # ---------- menu actions ----------

    def _on_toggle(self, icon, item):
        try:
            self._toggle_cb()
        except Exception as exc:  # noqa: BLE001
            logger.error("托盘触发录音切换失败: %s", exc)

    def _on_force_stop(self, icon, item):
        if self._force_stop_callback is None:
            return
        try:
            self._force_stop_callback()
        except Exception as exc:  # noqa: BLE001
            logger.error("强制停止触发失败: %s", exc)

    def _on_toggle_llm(self, icon, item):
        try:
            import json

            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            cfg = {}
        except Exception as exc:  # noqa: BLE001
            logger.error("读取配置失败: %s", exc)
            cfg = {}

        cfg.setdefault("llm", {})
        enabled = not self._is_llm_enabled()
        cfg["llm"]["enabled"] = enabled
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.error("写入配置失败: %s", exc)
            return

        self.set_llm_enabled(enabled)
        if self._llm_enabled_callback is not None:
            try:
                self._llm_enabled_callback(enabled)
            except Exception as exc:  # noqa: BLE001
                logger.error("同步 LLM 运行时开关失败: %s", exc)
        logger.info("LLM 润色开关已切换为 %s（立即生效）", enabled)

    def _on_settings(self, icon, item):
        """打开设置对话框（独立线程 + 独立 Tk root，不阻塞托盘）。"""
        try:
            from app.settings_dialog import open_settings_dialog

            open_settings_dialog(
                self._config_path,
                llm_enabled_callback=self._on_llm_enabled_from_settings,
            )
            logger.info("已打开设置对话框")
        except Exception as exc:  # noqa: BLE001
            logger.error("打开设置对话框失败: %s", exc)

    def show_context_menu(self, parent, x: int, y: int) -> None:
        import tkinter as tk

        menu = tk.Menu(parent, tearoff=0)
        menu.add_command(
            label="开始/停止录音（托盘）",
            command=lambda: self._on_toggle(None, None),
        )
        menu.add_separator()
        menu.add_command(
            label="强制停止",
            command=lambda: self._on_force_stop(None, None),
        )
        menu.add_separator()
        menu.add_command(
            label="LLM 润色：开" if self._is_llm_enabled() else "LLM 润色：关",
            command=lambda: self._on_toggle_llm(None, None),
        )
        menu.add_command(
            label="设置…",
            command=lambda: self._on_settings(None, None),
        )
        menu.add_command(
            label="打开日志目录",
            command=lambda: self._on_open_logs(None, None),
        )
        menu.add_separator()
        menu.add_command(
            label="退出",
            command=lambda: self._on_quit(None, None),
        )
        menu.add_separator()
        menu.add_command(
            label=f"{APP_NAME} v{app_version()}",
            state="disabled",
        )
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _on_open_logs(self, icon, item):
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            os.startfile(self._log_dir)  # Windows only
        except Exception as exc:  # noqa: BLE001
            logger.error("打开日志目录失败: %s", exc)

    def _on_quit(self, icon, item):
        logger.info("托盘菜单请求退出")
        self.stop()
        try:
            import keyboard

            keyboard.unhook_all()
        except Exception:
            pass
        os._exit(0)

    # ---------- helpers ----------

    def _is_llm_enabled(self) -> bool:
        with self._llm_enabled_lock:
            return self._llm_enabled

    def set_llm_enabled(self, enabled: bool) -> None:
        with self._llm_enabled_lock:
            self._llm_enabled = bool(enabled)

    def _on_llm_enabled_from_settings(self, enabled: bool) -> None:
        self.set_llm_enabled(enabled)
        if self._llm_enabled_callback is not None:
            self._llm_enabled_callback(bool(enabled))

    def _read_llm_enabled(self) -> bool:
        try:
            import json

            with open(self._config_path, "r", encoding="utf-8") as f:
                return bool(json.load(f).get("llm", {}).get("enabled", False))
        except Exception:
            return False
