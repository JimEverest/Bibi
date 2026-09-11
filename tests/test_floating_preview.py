import threading
import unittest

from app.floating_button import render_preview_text, FloatingButton


class FloatingPreviewTests(unittest.TestCase):
    def test_render_preview_text_truncates_from_left(self):
        committed = "0123456789" * 15
        tail = "最新尾巴"
        rendered = render_preview_text(committed, tail, max_chars=20)
        self.assertTrue(rendered.startswith("…"))
        self.assertTrue(rendered.endswith("最新尾巴"))
        self.assertLessEqual(len(rendered), 20)

    def test_show_preview_updates_internal_state_and_dispatches(self):
        button = FloatingButton.__new__(FloatingButton)
        button._lock = threading.Lock()
        button._preview_max_chars = 40
        button._preview_text = ""
        button._state = "idle"
        dispatched = []
        button._dispatch = lambda fn: dispatched.append(fn)

        button.show_preview("稳定前缀", "新尾巴", state="recording")

        self.assertEqual(button._preview_text, "稳定前缀新尾巴")
        self.assertEqual(button._state, "recording")
        self.assertEqual(len(dispatched), 1)
