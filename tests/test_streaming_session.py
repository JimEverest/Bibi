import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import load_config
from app.download_models import get_existing_model_cache_path
from app.streaming_session import PreviewAssembler


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


class PreviewAssemblerTests(unittest.TestCase):
    def setUp(self):
        self.assembler = PreviewAssembler(context_chars=30)

    def test_tail_rewrite_replaces_overlap_without_duplication(self):
        snap = self.assembler.apply_asr_segment(1, "我们明天开")
        self.assertEqual(snap.visible_text, "我们明天开")

        snap = self.assembler.apply_asr_segment(2, "明天开会讨论 streaming")
        self.assertEqual(snap.visible_text, "我们明天开会讨论 streaming")

    def test_stale_llm_update_is_ignored(self):
        self.assembler.apply_asr_segment(2, "第二段文本")
        snap = self.assembler.apply_llm_update(1, "旧结果")
        self.assertEqual(snap.visible_text, "第二段文本")

    def test_promote_tail_moves_text_into_committed_prefix(self):
        self.assembler.apply_asr_segment(1, "第一句")
        snap = self.assembler.promote_tail()
        self.assertEqual(snap.committed, "第一句")
        self.assertEqual(snap.tail, "")


if __name__ == "__main__":
    unittest.main()
