import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.llm_polish import LLMPolisher
from app.settings_dialog import _SettingsWindow
from app.tray import TrayApp
import main


class LLMRuntimeToggleTests(unittest.TestCase):
    def test_polisher_enablement_changes_without_recreating_instance(self):
        polisher = LLMPolisher(
            {
                "enabled": False,
                "endpoint": "http://localhost/v1/messages",
                "model": "test-model",
            }
        )
        with patch.object(polisher, "_call_openai", return_value="polished") as call:
            self.assertIsNone(polisher.polish("raw text"))
            call.assert_not_called()

            polisher.set_enabled(True)
            self.assertEqual(polisher.polish("raw text"), "polished")
            call.assert_called_once()

            polisher.set_enabled(False)
            self.assertIsNone(polisher.polish("raw text"))
            self.assertEqual(call.call_count, 1)

    def test_tray_toggle_updates_runtime_after_successful_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"llm": {"enabled": False}}), encoding="utf-8")
            events = []
            tray = TrayApp.__new__(TrayApp)
            tray._config_path = str(path)
            tray._llm_enabled_lock = threading.RLock()
            tray._llm_enabled = False
            tray._llm_enabled_callback = events.append

            tray._on_toggle_llm(None, None)

            self.assertEqual(events, [True])
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["llm"]["enabled"])

    def test_settings_save_updates_runtime_after_successful_persistence(self):
        events = []
        window = _SettingsWindow.__new__(_SettingsWindow)
        window._path = "unused.json"
        window._dirty = True
        window._collect = lambda: {"llm": {"enabled": True}}
        window._llm_enabled_callback = events.append
        window._msg = Mock()

        with patch("app.settings_dialog.save_settings", return_value=True):
            window._save()

        self.assertEqual(events, [True])
        window._msg.config.assert_called_once()

    def test_main_runtime_callback_updates_existing_polisher(self):
        original = main._LLM_POLISHER["instance"]
        try:
            polisher = LLMPolisher({"enabled": False})
            main._LLM_POLISHER["instance"] = polisher

            main._set_llm_enabled(True)
            self.assertIs(main._LLM_POLISHER["instance"], polisher)
            self.assertTrue(polisher.enabled)

            main._set_llm_enabled(False)
            self.assertFalse(polisher.enabled)
        finally:
            main._LLM_POLISHER["instance"] = original


if __name__ == "__main__":
    unittest.main()
