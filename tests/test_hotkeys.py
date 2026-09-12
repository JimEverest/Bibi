import unittest
from unittest.mock import patch

from app.hotkeys import HotkeyManager
from app.settings_dialog import collect_hotkey_settings
from main import _register_configured_hotkeys


class HotkeyManagerTests(unittest.TestCase):
    @patch("app.hotkeys.keyboard.add_hotkey")
    def test_disabled_toggle_hotkey_is_not_registered(self, add_hotkey):
        manager = HotkeyManager()

        manager.register("none", lambda: None)

        add_hotkey.assert_not_called()
        self.assertEqual(manager._registrations, {})

    def test_startup_skips_both_disabled_hotkeys(self):
        manager = unittest.mock.Mock()
        worker = unittest.mock.Mock()
        config = {
            "hotkeys": {
                "toggle": "f2",
                "toggle_enabled": False,
                "push_to_talk": "win+ctrl+alt",
                "push_to_talk_enabled": False,
            }
        }

        _register_configured_hotkeys(manager, config, worker)

        manager.register.assert_not_called()
        manager.register_push_to_talk.assert_not_called()

    def test_collect_hotkey_settings_persists_disabled_values(self):
        result = collect_hotkey_settings("f2", "win+ctrl+alt", False, False)

        self.assertEqual(
            result,
            {
                "toggle": "f2",
                "toggle_enabled": False,
                "push_to_talk": "win+ctrl+alt",
                "push_to_talk_enabled": False,
            },
        )

    @patch("app.hotkeys.keyboard.on_press_key")
    @patch("app.hotkeys.keyboard.on_release_key")
    def test_disabled_push_to_talk_is_not_registered(self, on_release, on_press):
        manager = HotkeyManager()

        manager.register_push_to_talk("off", lambda: None, lambda: None)

        on_press.assert_not_called()
        on_release.assert_not_called()
        self.assertEqual(manager._ptt_hooks, [])


if __name__ == "__main__":
    unittest.main()
