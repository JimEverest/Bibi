import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import load_config
from app.download_models import get_existing_model_cache_path


class StreamingConfigTests(unittest.TestCase):
    def test_streaming_defaults_are_present(self):
        cfg = load_config()
        self.assertTrue(cfg["streaming"]["enabled"])
        self.assertEqual(cfg["streaming"]["segment_silence_ms"], 450)
        self.assertEqual(cfg["streaming"]["audio_overlap_ms"], 500)
        self.assertEqual(cfg["streaming"]["preview_context_chars"], 30)
        self.assertEqual(cfg["streaming"]["preview_max_chars"], 120)
        self.assertEqual(cfg["streaming"]["ui_update_debounce_ms"], 80)
        self.assertFalse(cfg["streaming"]["dedicated_model_instance"])

    @patch("app.download_models.Path.home")
    def test_existing_model_cache_path_raises_when_missing(self, mock_home):
        mock_home.return_value = Path("/tmp/bibi-home")
        with self.assertRaises(FileNotFoundError):
            get_existing_model_cache_path("iic/not-downloaded")


if __name__ == "__main__":
    unittest.main()
