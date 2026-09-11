"""Persistent desktop capsule for press-and-hold and click-to-toggle recording."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_BG = "#1f2430"
_BORDER = "#151923"
_IDLE_DOT = "#5a6987"
_RECORDING_BG = "#3a1f24"
_RECORDING_DOT = "#ff5252"
_PROCESSING_BG = "#3a3320"
_PROCESSING_DOT = "#ffc107"
_FG = "#ffffff"
_MUTED = "#8a94ad"
_BAR_BG = "#0473ea"
_EXPANDED_SIZE = (176, 54)
_COLLAPSED_SIZE = (68, 16)
_PREVIEW_SIZE = (280, 78)
_DRAG_THRESHOLD = 3


def render_preview_text(committed: str, tail: str, max_chars: int = 120) -> str:
    """Render a single-line preview, truncating from the left so the newest tail stays visible."""
    text = f"{committed}{tail}"
    if len(text) <= max_chars:
        return text
    ellipsis = "…"
    keep = max_chars - len(ellipsis)
    if keep <= 0:
        return ellipsis[:max_chars]
    return ellipsis + text[-keep:]


class FloatingButtonDragController:
    def __init__(self):
        self._press_data = None
        self._moved = False

    def press(self, pointer_x: int, pointer_y: int, window_x: int, window_y: int) -> None:
        self._press_data = (pointer_x, pointer_y, window_x, window_y)
        self._moved = False

    def move(self, pointer_x: int, pointer_y: int):
        if self._press_data is None:
            return None
        press_x, press_y, window_x, window_y = self._press_data
        delta_x = pointer_x - press_x
        delta_y = pointer_y - press_y
        if abs(delta_x) > _DRAG_THRESHOLD or abs(delta_y) > _DRAG_THRESHOLD:
            self._moved = True
        return window_x + delta_x, window_y + delta_y

    def release(self) -> bool:
        should_toggle = self._press_data is not None and not self._moved
        self._press_data = None
        self._moved = False
        return should_toggle


class FloatingButtonController:
    """Translate the two capsule gestures into idempotent recording operations."""

    def __init__(self, start_recording, stop_recording, is_recording):
        self._start_recording = start_recording
        self._stop_recording = stop_recording
        self._is_recording = is_recording
        self._hold_active = False

    def press_main(self) -> None:
        if self._hold_active or self._is_recording():
            return
        self._start_recording()
        self._hold_active = True

    def release_main(self) -> None:
        if not self._hold_active:
            return
        self._hold_active = False
        if self._is_recording():
            self._stop_recording()

    def click_toggle(self) -> None:
        if self._hold_active:
            return
        if self._is_recording():
            self._stop_recording()
        else:
            self._start_recording()


class FloatingButton:
    """Tk Toplevel capsule attached to the process-wide UIHub root."""

    def __init__(self, start_recording, stop_recording, is_recording):
        self._root = None
        self._bar = None
        self._body = None
        self._main_area = None
        self._main_label = None
        self._toggle_button = None
        self._preview_label = None
        self._state = "idle"
        self._collapsed = False
        self._preview_text = ""
        self._preview_max_chars = 120
        self._drag_controller = FloatingButtonDragController()
        self._context_menu_callback = None
        self._lock = threading.Lock()
        self._controller = FloatingButtonController(
            start_recording=start_recording,
            stop_recording=stop_recording,
            is_recording=is_recording,
        )
        self._ensure_built()
        self.show_idle()

    def _ensure_built(self) -> None:
        from app.ui_hub import get_hub

        hub = get_hub()
        if hub.root is None:
            logger.error("悬浮按钮无法创建：UIHub root 尚未创建")
            return

        def _build() -> None:
            if self._root is not None:
                return
            try:
                import tkinter as tk

                root = tk.Toplevel(hub.root)
                root.title("bibi-floating-button")
                root.overrideredirect(True)
                root.attributes("-topmost", True)
                root.attributes("-alpha", 0.96)
                root.configure(bg=_BORDER)

                bar = tk.Frame(root, bg=_BAR_BG, height=8, cursor="fleur")
                bar.pack(fill="x")
                bar.pack_propagate(False)

                shell = tk.Frame(root, bg=_BORDER, padx=1, pady=1)
                shell.pack(fill="both", expand=True)

                main = tk.Frame(shell, bg=_BG, cursor="hand2")
                main.pack(side="left", fill="both", expand=True)
                label = tk.Label(
                    main,
                    text="按住说话",
                    bg=_BG,
                    fg=_FG,
                    font=("Segoe UI", 10),
                    padx=12,
                    pady=8,
                    cursor="hand2",
                )
                label.pack(fill="both", expand=True)

                separator = tk.Frame(shell, bg="#3a4256", width=1)
                separator.pack(side="left", fill="y", pady=8)

                toggle = tk.Button(
                    shell,
                    text="○",
                    command=self._on_toggle_click,
                    bg=_BG,
                    activebackground=_BG,
                    fg=_MUTED,
                    activeforeground=_FG,
                    relief="flat",
                    borderwidth=0,
                    highlightthickness=0,
                    font=("Segoe UI Symbol", 13),
                    width=3,
                    cursor="hand2",
                )
                toggle.pack(side="left", fill="y", padx=(0, 2))

                preview = tk.Label(
                    root,
                    text="",
                    bg=_BG,
                    fg=_MUTED,
                    font=("Segoe UI", 9),
                    anchor="w",
                    padx=12,
                    pady=4,
                )
                # Not packed here on purpose — only shown when there is preview text.

                for widget in (main, label):
                    widget.bind("<ButtonPress-1>", self._on_main_press)
                    widget.bind("<ButtonRelease-1>", self._on_main_release)
                bar.bind("<ButtonPress-1>", self._on_bar_press)
                bar.bind("<B1-Motion>", self._on_bar_motion)
                bar.bind("<ButtonRelease-1>", self._on_bar_release)
                for widget in (main, label, toggle, bar):
                    widget.bind("<Button-3>", self._on_context_menu)

                self._root = root
                self._bar = bar
                self._body = shell
                self._main_area = main
                self._main_label = label
                self._toggle_button = toggle
                self._preview_label = preview
                self._place(root)
                self._apply_state()
            except Exception as exc:  # noqa: BLE001
                logger.warning("悬浮按钮构建失败: %s", exc)

        hub.dispatch(_build)

    @staticmethod
    def _place(root) -> None:
        try:
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            window_width, window_height = _EXPANDED_SIZE
            root.geometry(f"{window_width}x{window_height}+{width - 220}+{height - 130}")
        except Exception:
            window_width, window_height = _EXPANDED_SIZE
            root.geometry(f"{window_width}x{window_height}+1200+800")

    def _dispatch(self, fn) -> None:
        from app.ui_hub import get_hub

        if not get_hub().dispatch(fn):
            logger.debug("悬浮按钮 UI 操作派发失败")

    def set_context_menu_callback(self, callback) -> None:
        self._context_menu_callback = callback

    def _on_context_menu(self, event):
        if self._context_menu_callback is not None and self._root is not None:
            self._context_menu_callback(self._root, event.x_root, event.y_root)
        return "break"

    def _on_bar_press(self, event):
        if self._root is None:
            return "break"
        self._drag_controller.press(
            event.x_root,
            event.y_root,
            self._root.winfo_x(),
            self._root.winfo_y(),
        )
        return "break"

    def _on_bar_motion(self, event):
        if self._root is None:
            return "break"
        position = self._drag_controller.move(event.x_root, event.y_root)
        if position is not None:
            x, y = position
            width = self._root.winfo_width()
            height = self._root.winfo_height()
            self._root.geometry(f"{width}x{height}+{x}+{y}")
        return "break"

    def _on_bar_release(self, _event=None):
        if self._drag_controller.release():
            self._toggle_collapsed()
        return "break"

    def _toggle_collapsed(self):
        if self._root is None or self._body is None:
            return
        x = self._root.winfo_x()
        y = self._root.winfo_y()
        was_collapsed = self._collapsed
        if was_collapsed:
            self._body.pack(fill="both", expand=True)
        else:
            self._body.pack_forget()
            if self._preview_label is not None:
                self._preview_label.pack_forget()
        self._collapsed = not was_collapsed
        if self._collapsed:
            width, height = _COLLAPSED_SIZE
            self._root.geometry(f"{width}x{height}+{x}+{y}")
        else:
            # Let _apply_state decide expanded vs. preview size (and re-show
            # the preview label if there is preview text waiting).
            self._apply_state()

    def _on_main_press(self, _event=None):
        self._controller.press_main()
        self.show_recording()
        if self._root is not None:
            try:
                self._root.grab_set()
            except Exception:
                pass
        return "break"

    def _on_main_release(self, _event=None):
        self._controller.release_main()
        try:
            if self._root is not None:
                self._root.grab_release()
        except Exception:
            pass
        if self._controller._is_recording():
            self.show_recording()
        else:
            self.show_processing()
        return "break"

    def _on_toggle_click(self):
        was_recording = self._controller._is_recording()
        self._controller.click_toggle()
        if was_recording:
            self.show_processing()
        else:
            self.show_recording()

    def _apply_state(self) -> None:
        if self._root is None or self._main_area is None:
            return
        with self._lock:
            state = self._state
            preview_text = self._preview_text
        if state == "recording":
            bg, dot, text = _RECORDING_BG, _RECORDING_DOT, "录音中"
            symbol = "■"
        elif state == "processing":
            bg, dot, text = _PROCESSING_BG, _PROCESSING_DOT, "处理中…"
            symbol = "…"
        else:
            bg, dot, text = _BG, _IDLE_DOT, "按住说话"
            symbol = "○"
        self._main_area.configure(bg=bg)
        self._main_label.configure(bg=bg, text=text, fg=_FG)
        self._toggle_button.configure(
            bg=bg,
            activebackground=bg,
            fg=dot,
            activeforeground=dot,
            text=symbol,
        )
        if self._preview_label is not None:
            if self._collapsed or not preview_text:
                self._preview_label.pack_forget()
                target_size = _EXPANDED_SIZE
            else:
                self._preview_label.configure(bg=bg, fg=_MUTED, text=preview_text)
                self._preview_label.pack(side="top", fill="x", padx=0, pady=(0, 6))
                target_size = _PREVIEW_SIZE
            self._resize_to(target_size)

    def _resize_to(self, size) -> None:
        if self._root is None:
            return
        width, height = size
        if self._root.winfo_width() == width and self._root.winfo_height() == height:
            return
        x = self._root.winfo_x()
        y = self._root.winfo_y()
        self._root.geometry(f"{width}x{height}+{x}+{y}")

    def show_preview(self, committed: str, tail: str, state: str) -> None:
        with self._lock:
            max_chars = self._preview_max_chars
            self._preview_text = render_preview_text(committed, tail, max_chars=max_chars)
            self._state = state
        self._dispatch(self._apply_state)

    def clear_preview(self) -> None:
        with self._lock:
            self._preview_text = ""
        self._dispatch(self._apply_state)

    def _set_state(self, state: str) -> None:
        self._ensure_built()
        with self._lock:
            self._state = state
        self._dispatch(self._apply_state)

    def show_idle(self) -> None:
        self._set_state("idle")

    def show_recording(self) -> None:
        self._set_state("recording")

    def show_processing(self) -> None:
        self._set_state("processing")

    def destroy(self) -> None:
        def _destroy() -> None:
            if self._root is not None:
                self._root.destroy()
                self._root = None

        self._dispatch(_destroy)
