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

    def test_show_preview_processing_does_not_override_active_recording_state(self):
        # LLM 在录音仍在进行时对某一段尾巴做后台润色，state="processing" 只是
        # 描述这次预览更新的来源，不应把胶囊主态从"recording"翻转成
        # "processing"（那是留给录音真正停止后的状态）。
        button = FloatingButton.__new__(FloatingButton)
        button._lock = threading.Lock()
        button._preview_max_chars = 40
        button._preview_text = ""
        button._state = "recording"
        dispatched = []
        button._dispatch = lambda fn: dispatched.append(fn)

        button.show_preview("今天下午", "开会", state="processing")

        self.assertEqual(button._preview_text, "今天下午开会")
        self.assertEqual(button._state, "recording")
        self.assertEqual(len(dispatched), 1)

    def test_show_preview_processing_applies_when_not_actively_recording(self):
        # 一旦不处于"recording"主态（例如已经 idle，等待中），处理中的预览事件
        # 仍然应当照常生效，覆盖行为只针对"仍在录音"这一具体场景。
        button = FloatingButton.__new__(FloatingButton)
        button._lock = threading.Lock()
        button._preview_max_chars = 40
        button._preview_text = ""
        button._state = "idle"
        dispatched = []
        button._dispatch = lambda fn: dispatched.append(fn)

        button.show_preview("今天下午", "开会", state="processing")

        self.assertEqual(button._state, "processing")
