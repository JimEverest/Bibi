# -*- coding: utf-8 -*-
"""
录音状态指示浮窗（UI/UX 改进）

- 屏幕右下角小型无边框置顶窗口，不抢键盘焦点
- 三种状态：录音中（红点+计时）/ 处理中（黄点+"处理中…"）/ 隐藏
- tkinter 实现；所有 UI 都挂在 UIHub 的主线程 root 下（Toplevel），
  通过 after() 派发操作，杜绝跨线程创建/访问 Tk 解释器
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_BG = "#1f2430"
_FG = "#ffffff"


class _IndicatorWindow:
    def __init__(self):
        self._root = None          # Toplevel，在主线程创建
        self._label = None
        self._dot = None
        self._lock = threading.Lock()
        self._state = "hidden"  # hidden|recording|polishing
        self._start_ts = 0.0
        self._timer_job = None

    # ---------- build (dispatched to main thread) ----------

    def _ensure_built(self):
        """请求主线程构建浮窗（幂等）。可在任意线程调用。"""
        from app.ui_hub import get_hub

        hub = get_hub()
        if hub.root is None:
            # 启动时 main() 已在主线程创建 root；这里兜底
            try:
                hub.create_root()
            except RuntimeError as exc:
                logger.error("指示浮窗: 无法创建 Tk root（需主线程）: %s", exc)
                return

        def _build():
            if self._root is not None:
                return
            try:
                import tkinter as tk

                root = tk.Toplevel(hub.root)
                root.title("bibi-indicator")
                root.withdraw()
                root.overrideredirect(True)   # 无边框
                root.attributes("-topmost", True)
                root.attributes("-alpha", 0.92)
                root.configure(bg=_BG)

                frame = tk.Frame(root, bg=_BG)
                frame.pack(fill="both", expand=True, padx=12, pady=8)

                dot = tk.Canvas(frame, width=14, height=14, bg=_BG, highlightthickness=0)
                dot.pack(side="left")
                label = tk.Label(
                    frame, text="", bg=_BG, fg=_FG,
                    font=("Segoe UI", 11), anchor="w",
                )
                label.pack(side="left", padx=(8, 0))

                self._root = root
                self._dot = dot
                self._label = label
                self._place(root)
            except Exception as exc:  # noqa: BLE001
                logger.warning("指示浮窗构建失败: %s", exc)

        hub.dispatch(_build)

    @staticmethod
    def _place(root):
        try:
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.geometry("148x44+{}+{}".format(w - 180, h - 110))
        except Exception:
            root.geometry("148x44+1200+800")

    # ---------- state ops (thread-safe via event scheduling) ----------

    def _dispatch(self, fn):
        """把 UI 操作派发到主线程执行。"""
        from app.ui_hub import get_hub

        if not get_hub().dispatch(fn):
            logger.debug("指示浮窗: 派发失败（root 未就绪）")

    def show_recording(self):
        self._ensure_built()
        with self._lock:
            self._state = "recording"
            self._start_ts = time.time()

        def _apply():
            if not self._root:
                return
            self._dot.delete("all")
            self._dot.create_oval(1, 1, 13, 13, fill="#ff5252", outline="")
            self._root.deiconify()
            self._tick()

        self._dispatch(_apply)

    def show_polishing(self):
        self._ensure_built()
        with self._lock:
            self._state = "polishing"

        def _apply():
            if not self._root:
                return
            if self._timer_job:
                try:
                    self._root.after_cancel(self._timer_job)
                except Exception:
                    pass
                self._timer_job = None
            self._dot.delete("all")
            self._dot.create_oval(1, 1, 13, 13, fill="#ffc107", outline="")
            self._label.config(text="处理中…")
            self._root.deiconify()

        self._dispatch(_apply)

    def hide(self):
        with self._lock:
            self._state = "hidden"

        def _apply():
            if not self._root:
                return
            if self._timer_job:
                try:
                    self._root.after_cancel(self._timer_job)
                except Exception:
                    pass
                self._timer_job = None
            self._root.withdraw()

        self._dispatch(_apply)

    def _tick(self):
        with self._lock:
            if self._state != "recording":
                return
            elapsed = int(time.time() - self._start_ts)
        mm, ss = divmod(elapsed, 60)
        if self._label is not None:
            try:
                self._label.config(text=f"录音中  {mm:02d}:{ss:02d}")
            except Exception:
                return
        if self._root is not None:
            try:
                self._timer_job = self._root.after(500, self._tick)
            except Exception:
                pass


_instance: _IndicatorWindow | None = None
_instance_lock = threading.Lock()


def get_indicator() -> _IndicatorWindow:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = _IndicatorWindow()
        return _instance
