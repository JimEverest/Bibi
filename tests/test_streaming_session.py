import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import load_config
from app.download_models import get_existing_model_cache_path
from app.streaming_session import PreviewAssembler, StreamingSegmenter, StreamingSession


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


class StreamingSegmenterTests(unittest.TestCase):
    def test_silence_emits_segment_after_threshold(self):
        segmenter = StreamingSegmenter(sample_rate=16000, silence_ms=200, overlap_ms=500)
        speech = np.ones(1600, dtype=np.int16)
        silence = np.zeros(1600, dtype=np.int16)

        emitted = []
        emitted.extend(segmenter.push_chunk(speech, is_speech=True))
        emitted.extend(segmenter.push_chunk(silence, is_speech=False))
        emitted.extend(segmenter.push_chunk(silence, is_speech=False))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].sequence, 1)

    def test_second_segment_contains_overlap_samples(self):
        # NOTE: silence_ms is set to 100 (not 200 as in the original plan draft) so that a
        # single 1600-sample/100ms silence chunk reaches the threshold and triggers emission,
        # matching this test's intent (verify overlap carries into the next segment). With
        # silence_ms=200 the reference StreamingSegmenter never emits after only one silence
        # chunk, so `first`/`second` would both be empty lists regardless of implementation.
        segmenter = StreamingSegmenter(sample_rate=16000, silence_ms=100, overlap_ms=100)
        speech = np.ones(1600, dtype=np.int16)
        silence = np.zeros(1600, dtype=np.int16)

        first = segmenter.push_chunk(speech, is_speech=True)
        first += segmenter.push_chunk(silence, is_speech=False)
        second = segmenter.push_chunk(speech * 2, is_speech=True)
        second += segmenter.push_chunk(silence, is_speech=False)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertGreater(len(second[0].samples), len(speech))


class StreamingSessionTests(unittest.TestCase):
    def test_streaming_session_emits_preview_from_stable_segment(self):
        events = []
        session = StreamingSession(
            sample_rate=16000,
            streaming_cfg={"segment_silence_ms": 200, "audio_overlap_ms": 100},
            detect_speech=lambda samples, is_final=False: bool(samples.max()),
            transcribe_segment=lambda samples: {"success": True, "text": "第一句"},
            on_preview=lambda committed, tail, state: events.append((committed, tail, state)),
        )
        session.start()
        session.push_chunk(np.ones(1600, dtype=np.int16))
        session.push_chunk(np.zeros(1600, dtype=np.int16))
        session.push_chunk(np.zeros(1600, dtype=np.int16))
        session.finish()

        self.assertTrue(events)
        self.assertIn(("", "第一句", "recording"), events)


class StreamingSessionLLMTests(unittest.TestCase):
    def test_llm_partial_updates_replace_only_tail(self):
        previews = []
        partials = ["今天下午", "今天下午开会"]

        def polish_tail(prefix_context, tail_text, on_update):
            for partial in partials:
                on_update(partial)
            return partials[-1]

        session = StreamingSession(
            sample_rate=16000,
            streaming_cfg={"segment_silence_ms": 200, "audio_overlap_ms": 100, "preview_context_chars": 20},
            detect_speech=lambda samples, is_final=False: bool(samples.max()),
            transcribe_segment=lambda samples: {"success": True, "text": "今天下午开会"},
            polish_tail=polish_tail,
            on_preview=lambda committed, tail, state: previews.append((committed, tail, state)),
        )
        session.start()
        session.push_chunk(__import__("numpy").ones(1600, dtype=__import__("numpy").int16))
        session.push_chunk(__import__("numpy").zeros(1600, dtype=__import__("numpy").int16))
        session.push_chunk(__import__("numpy").zeros(1600, dtype=__import__("numpy").int16))
        session.finish()

        self.assertIn(("", "今天下午", "processing"), previews)
        self.assertIn(("", "今天下午开会", "processing"), previews)


if __name__ == "__main__":
    unittest.main()
