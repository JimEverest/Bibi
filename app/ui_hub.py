# -*- coding: utf-8 -*-
"""
统一 Tk UI 中心（修复 "main thread is not in main loop"）

问题根源：录音浮窗与设置对话框曾在不同线程各自创建 tk.Tk() + mainloop。
Tcl/Tk 解释器不是线程安全的：CPython 的 _tkinter 对跨线程访问有严格限制，
实测连 root.after() 跨线程调用都会抛
RuntimeError: main thread is not in main loop。

修复架构（最严格模式）：
- 全进程唯一 Tk root，仅在进程主线程创建；
- 录音浮窗 / 设置窗口都是该 root 的 Toplevel；
- 其他线程（托盘/热键/转写回调）绝不直接调用任何 Tk API（包括 after），
  而是把 callable 投入线程安全队列；
- 主线程事件泵每帧：先清空队列执行 UI 任务，再 root.update() 泄事件；
  用 update() 轮询而非 mainloop，保证 Ctrl+C 可中断退出。
"""
from __future__ import annotations

import logging
import queue
import threading

logger = logging.getLogger(__name__)

_hub: "UIHub | None" = None


def get_hub() -> "UIHub":
    global _hub
    if _hub is None:
        _hub = UIHub()
    return _hub


class UIHub:
    def __init__(self):
        self._root = None
        self._lock = threading.Lock()
        self._q: "queue.Queue" = queue.Queue()

    @property
    def root(self):
        return self._root

    def create_root(self):
        """在主线程创建唯一 Tk root（隐藏宿主窗口，只显示 Toplevel）。

        注意：只能在主线程调用；其他线程调用会抛 RuntimeError（已兜底捕获）。
        """
        try:
            import tkinter as tk

            with self._lock:
                if self._root is None:
                    self._root = tk.Tk()
                    self._root.withdraw()
                    self._root.title("bibi-ui")
                    logger.info("UIHub: 主线程 Tk root 已创建")
            return self._root
        except RuntimeError as exc:
            # 在非创建线程调用 Tk API 时的典型错误
            logger.error("UIHub: Tk root 创建失败（必须在主线程）: %s", exc)
            return None

    def dispatch(self, fn) -> bool:
        """线程安全：把 UI 任务投入队列，由主线程事件泵执行。

        绝不在此直接触碰任何 Tk API（包括 root.after）。
        """
        try:
            self._q.put_nowait(fn)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("UIHub: 任务入队失败: %s", exc)
            return False

    def _drain(self):
        """在主线程执行队列中的所有 UI 任务。"""
        while True:
            try:
                fn = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.error("UIHub: UI 任务执行失败: %s", exc)

    def run(self):
        """主线程事件泵。update() 轮询 + 队列清空，Ctrl+C 可正常打断。"""
        import time

        root = self.create_root()
        if root is None:
            raise RuntimeError("UIHub: 主线程 Tk root 创建失败，无法启动 UI")
        logger.info("UIHub: 主线程事件泵启动")
        while True:
            try:
                self._drain()
                root.update()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # 窗口销毁等瞬态错误不终止程序
                logger.debug("UIHub: update 异常: %s", exc)
            time.sleep(0.02)
