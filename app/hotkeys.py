"""Global hotkey management for the application."""

from __future__ import annotations

import logging
import threading
from typing import Callable

import keyboard


logger = logging.getLogger(__name__)

_DISABLED_HOTKEY_VALUES = {"", "none", "off", "disabled"}


def is_hotkey_enabled(combo: str | None) -> bool:
    return bool(combo and str(combo).strip().lower() not in _DISABLED_HOTKEY_VALUES)


class HotkeyManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registrations = {}
        self._ptt_hooks = []

    def register(self, combo: str, callback: Callable[[], None]) -> None:
        if not is_hotkey_enabled(combo):
            logger.info("热键未配置或已禁用")
            return

        with self._lock:
            if combo in self._registrations:
                logger.warning("热键 %s 已注册，覆盖旧的回调", combo)
                keyboard.remove_hotkey(self._registrations[combo])

            # 过滤 Windows 按键自动重复（按住不放会连续触发 keydown）
            # trigger_on_release=False 时 add_hotkey 每次重复 keydown 都会回调；
            # 用 suppressed_events 无法区分，改为在回调层做"键仍被物理按住则忽略"
            def _dedup() -> None:
                try:
                    if keyboard.is_pressed(combo.split("+")[-1].strip()):
                        pass  # 键仍按住 —— 仍可能是首次触发，交由回调自行去抖
                    else:
                        return  # 键已松开（自动重复的迟到事件），忽略
                except Exception:
                    pass
                callback()

            try:
                hotkey_id = keyboard.add_hotkey(combo, _dedup)
            except Exception as exc:  # noqa: BLE001
                logger.error("注册热键 %s 失败: %s", combo, exc)
                raise

            self._registrations[combo] = hotkey_id
            logger.info("已注册热键 %s", combo)

    def register_push_to_talk(self, combo: str, on_start: Callable[[], None], on_stop: Callable[[], None]) -> None:
        """按住说话（PTT）模式：组合键全部按下 → on_start；任一键松开 → on_stop。

        与 toggle 模式互相独立；组合键为空字符串或 "none" 时视为禁用。
        """
        if not is_hotkey_enabled(combo):
            logger.info("PTT 按住说话未配置或已禁用")
            return

        keys = [k.strip().lower() for k in combo.split("+") if k.strip()]
        if not keys:
            logger.info("PTT 按住说话未配置或已禁用")
            return

        state = {"active": False}

        def _all_down() -> bool:
            try:
                return all(keyboard.is_pressed(k) for k in keys)
            except Exception:
                return False

        def _on_press() -> None:
            # 按住不放会产生大量 keydown 自动重复事件，用 active 标志去重
            if state["active"]:
                return
            if _all_down():
                state["active"] = True
                logger.info("PTT 开始录音: %s", combo)
                on_start()

        def _on_release() -> None:
            if state["active"]:
                state["active"] = False
                logger.info("PTT 停止录音（按键松开）")
                on_stop()

        with self._lock:
            try:
                for k in keys:
                    self._ptt_hooks.append(keyboard.on_press_key(k, lambda e: _on_press()))
                    self._ptt_hooks.append(keyboard.on_release_key(k, lambda e: _on_release()))
                logger.info("已注册按住说话热键 %s（按住录音，松开停止）", combo)
            except Exception as exc:  # noqa: BLE001
                logger.error("注册按住说话热键 %s 失败: %s", combo, exc)
                raise

    def unregister_all(self) -> None:
        with self._lock:
            for combo, hotkey_id in list(self._registrations.items()):
                keyboard.remove_hotkey(hotkey_id)
                logger.info("已移除热键 %s", combo)
            self._registrations.clear()
            for hook in list(self._ptt_hooks):
                try:
                    keyboard.unhook_key(hook)
                except Exception:
                    pass
            self._ptt_hooks.clear()

    def cleanup(self) -> None:
        self.unregister_all()
        # 彻底停止 keyboard 库的所有钩子和监听线程
        try:
            keyboard.unhook_all()
            logger.info("已停止 keyboard 监听线程")
        except Exception as exc:
            logger.warning("停止 keyboard 监听线程失败: %s", exc)


