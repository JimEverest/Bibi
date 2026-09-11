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
        self.assertEqual(cfg["streaming"]["commit_silence_ms"], 1200)
        self.assertEqual(cfg["streaming"]["max_uncommitted_ms"], 12000)
        self.assertEqual(cfg["streaming"]["audio_overlap_ms"], 150)
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

    def test_apply_asr_segment_wholesale_replaces_tail(self):
        # Every ASR segment result is a full re-decode of the growing
        # uncommitted buffer, so it must always fully replace the visible
        # tail -- no character-level splicing against the previous tail.
        snap = self.assembler.apply_asr_segment(1, "第一段")
        self.assertEqual(snap.visible_text, "第一段")

        snap = self.assembler.apply_asr_segment(2, "完全不相关的新内容")
        self.assertEqual(snap.visible_text, "完全不相关的新内容")

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
    def _segmenter(self, **overrides):
        params = dict(
            sample_rate=16000,
            silence_ms=200,
            commit_silence_ms=600,
            max_uncommitted_ms=100000,
            overlap_ms=100,
        )
        params.update(overrides)
        return StreamingSegmenter(**params)

    def test_short_pause_emits_single_preview_with_accumulated_speech(self):
        segmenter = self._segmenter()
        speech = np.ones(1600, dtype=np.int16)  # 100ms
        silence = np.zeros(1600, dtype=np.int16)  # 100ms

        emitted = []
        emitted.extend(segmenter.push_chunk(speech, is_speech=True))
        emitted.extend(segmenter.push_chunk(silence, is_speech=False))
        emitted.extend(segmenter.push_chunk(silence, is_speech=False))  # 200ms silence -> preview

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].kind, "preview")
        self.assertEqual(emitted[0].sequence, 1)
        # accumulated buffer = speech + 2 silence chunks
        self.assertEqual(len(emitted[0].samples), 1600 * 3)

    def test_long_pause_after_preview_emits_cumulative_commit(self):
        segmenter = self._segmenter()
        speech = np.ones(1600, dtype=np.int16)  # 100ms
        silence = np.zeros(1600, dtype=np.int16)  # 100ms

        emitted = []
        emitted.extend(segmenter.push_chunk(speech, is_speech=True))
        emitted.extend(segmenter.push_chunk(silence, is_speech=False))
        emitted.extend(segmenter.push_chunk(silence, is_speech=False))  # 200ms -> preview fires
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].kind, "preview")

        # keep talking (resets pause flags/silence), producing more cumulative speech
        emitted.extend(segmenter.push_chunk(speech, is_speech=True))
        # now pause long enough to cross commit_silence_ms (600ms of silence)
        for _ in range(6):
            emitted.extend(segmenter.push_chunk(silence, is_speech=False))

        commit_tasks = [t for t in emitted if t.kind == "commit"]
        self.assertEqual(len(commit_tasks), 1)
        commit_task = commit_tasks[0]
        # cumulative: 2 speech chunks (200ms) + all silence chunks since recording start
        # (2 from the first pause + 6 from the second pause = 8 silence chunks, 800ms)
        expected_len = 1600 * 2 + 1600 * 8
        self.assertEqual(len(commit_task.samples), expected_len)

        # Exactly one commit fires when the commit threshold is crossed --
        # it does not *also* emit a second preview at that same instant
        # (a preview may legitimately have already fired earlier in this same
        # pause, once silence passed silence_ms but before commit_silence_ms).

    def test_buffer_resets_after_commit(self):
        segmenter = self._segmenter(overlap_ms=100)  # 1600 samples pad at 16kHz
        speech = np.ones(1600, dtype=np.int16)
        silence = np.zeros(1600, dtype=np.int16)

        emitted = []
        emitted.extend(segmenter.push_chunk(speech, is_speech=True))
        for _ in range(6):  # 600ms silence -> crosses commit_silence_ms
            emitted.extend(segmenter.push_chunk(silence, is_speech=False))

        commit_tasks = [t for t in emitted if t.kind == "commit"]
        self.assertEqual(len(commit_tasks), 1)

        # Speak again after the commit; the buffer should only contain the
        # onset pad (from the just-committed audio) plus the new speech --
        # not the full pre-commit history.
        more_speech = np.ones(1600, dtype=np.int16) * 2
        emitted2 = segmenter.push_chunk(more_speech, is_speech=True)
        for _ in range(1):
            emitted2.extend(segmenter.push_chunk(silence, is_speech=False))
        emitted2.extend(segmenter.push_chunk(silence, is_speech=False))  # crosses silence_ms -> preview

        preview_tasks = [t for t in emitted2 if t.kind == "preview"]
        self.assertEqual(len(preview_tasks), 1)
        # onset pad (1600 samples) + new speech (1600) + 2 silence chunks (3200) = 6400
        self.assertEqual(len(preview_tasks[0].samples), 1600 + 1600 + 1600 * 2)

    def test_max_uncommitted_ms_forces_commit_without_pause(self):
        segmenter = self._segmenter(
            silence_ms=200, commit_silence_ms=600, max_uncommitted_ms=500, overlap_ms=50
        )
        speech = np.ones(1600, dtype=np.int16)  # 100ms per chunk

        emitted = []
        for _ in range(6):  # 600ms of continuous speech, never pausing
            emitted.extend(segmenter.push_chunk(speech, is_speech=True))

        commit_tasks = [t for t in emitted if t.kind == "commit"]
        self.assertGreaterEqual(len(commit_tasks), 1)
        preview_tasks = [t for t in emitted if t.kind == "preview"]
        self.assertEqual(len(preview_tasks), 0)

    def test_is_final_does_not_double_commit_after_pause_triggered_commit(self):
        # Regression: a commit fired mid-recording from a long pause, leaving
        # only the leftover onset pad in the buffer. If no further real
        # audio arrives before finish()/is_final, there must be NO second
        # spurious commit of just that pad.
        segmenter = self._segmenter(silence_ms=200, commit_silence_ms=600)
        speech = np.ones(1600, dtype=np.int16)
        silence = np.zeros(1600, dtype=np.int16)

        emitted = []
        emitted.extend(segmenter.push_chunk(speech, is_speech=True))
        for _ in range(6):  # 600ms silence -> crosses commit_silence_ms
            emitted.extend(segmenter.push_chunk(silence, is_speech=False))

        commit_tasks = [t for t in emitted if t.kind == "commit"]
        self.assertEqual(len(commit_tasks), 1)

        dummy = np.zeros(0, dtype=np.int16)
        final_emitted = segmenter.push_chunk(dummy, is_speech=False, is_final=True)
        self.assertEqual(final_emitted, [])

        total_commits = commit_tasks + [t for t in final_emitted if t.kind == "commit"]
        self.assertEqual(len(total_commits), 1)

    def test_is_final_flushes_new_content_arriving_after_a_commit(self):
        # A commit fires, then MORE real speech/silence arrives before
        # finish() -- that new bit (plus the leftover pad) must still be
        # flushed as a final commit.
        segmenter = self._segmenter(silence_ms=200, commit_silence_ms=600, overlap_ms=100)
        speech = np.ones(1600, dtype=np.int16)
        silence = np.zeros(1600, dtype=np.int16)

        emitted = []
        emitted.extend(segmenter.push_chunk(speech, is_speech=True))
        for _ in range(6):
            emitted.extend(segmenter.push_chunk(silence, is_speech=False))
        commit_tasks = [t for t in emitted if t.kind == "commit"]
        self.assertEqual(len(commit_tasks), 1)

        # a bit more speech arrives after the commit
        more_speech = np.ones(1600, dtype=np.int16) * 2
        segmenter.push_chunk(more_speech, is_speech=True)

        dummy = np.zeros(0, dtype=np.int16)
        final_emitted = segmenter.push_chunk(dummy, is_speech=False, is_final=True)
        self.assertEqual(len(final_emitted), 1)
        self.assertEqual(final_emitted[0].kind, "commit")
        # onset pad (1600) + new speech (1600)
        self.assertEqual(len(final_emitted[0].samples), 1600 + 1600)

    def test_is_final_always_flushes_remaining_buffer_as_commit(self):
        segmenter = self._segmenter(silence_ms=200, commit_silence_ms=600)
        speech = np.ones(1600, dtype=np.int16)

        emitted = list(segmenter.push_chunk(speech, is_speech=True))
        self.assertEqual(len(emitted), 0)

        dummy = np.zeros(0, dtype=np.int16)
        final_emitted = segmenter.push_chunk(dummy, is_speech=False, is_final=True)

        self.assertEqual(len(final_emitted), 1)
        self.assertEqual(final_emitted[0].kind, "commit")
        self.assertEqual(len(final_emitted[0].samples), 1600)


class StreamingSessionTests(unittest.TestCase):
    def _streaming_cfg(self, **overrides):
        cfg = {
            "segment_silence_ms": 200,
            "commit_silence_ms": 400,
            "max_uncommitted_ms": 100000,
            "audio_overlap_ms": 100,
        }
        cfg.update(overrides)
        return cfg

    def test_streaming_session_emits_preview_from_stable_segment(self):
        events = []
        session = StreamingSession(
            sample_rate=16000,
            streaming_cfg=self._streaming_cfg(),
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

    def test_commit_without_polish_promotes_tail_into_committed(self):
        # Regression: previously `promote_tail()` was never invoked anywhere
        # in StreamingSession, so `committed` stayed empty forever. This test
        # would have FAILED against the old code.
        #
        # A single speech chunk followed directly by finish() means the only
        # segment emitted is the final `is_final=True` forced commit flush
        # (no interim silence long enough to also trigger a mid-recording
        # commit), keeping this a clean single-commit scenario.
        events = []
        session = StreamingSession(
            sample_rate=16000,
            streaming_cfg=self._streaming_cfg(),
            detect_speech=lambda samples, is_final=False: bool(samples.max()),
            transcribe_segment=lambda samples: {"success": True, "text": "已提交文本"},
            on_preview=lambda committed, tail, state: events.append((committed, tail, state)),
        )
        session.start()
        session.push_chunk(np.ones(1600, dtype=np.int16))
        session.finish()

        committed_events = [e for e in events if e[0] and e[1] == ""]
        self.assertTrue(committed_events, f"expected a committed/cleared-tail event, got: {events}")
        self.assertEqual(committed_events[-1][0], "已提交文本")


    def test_asr_failure_on_commit_logs_warning_and_does_not_crash(self):
        events = []
        session = StreamingSession(
            sample_rate=16000,
            streaming_cfg=self._streaming_cfg(),
            detect_speech=lambda samples, is_final=False: bool(samples.max()),
            transcribe_segment=lambda samples: {"success": False},
            on_preview=lambda committed, tail, state: events.append((committed, tail, state)),
        )
        session.start()
        with self.assertLogs("app.streaming_session", level="WARNING") as cm:
            session.push_chunk(np.ones(1600, dtype=np.int16))
            session.finish()

        self.assertTrue(any("commit" in msg for msg in cm.output))
        # No promotion into committed should have happened.
        self.assertFalse(any(e[0] for e in events))


class StreamingSessionLLMTests(unittest.TestCase):
    def _streaming_cfg(self, **overrides):
        cfg = {
            "segment_silence_ms": 200,
            "commit_silence_ms": 400,
            "max_uncommitted_ms": 100000,
            "audio_overlap_ms": 100,
            "preview_context_chars": 20,
        }
        cfg.update(overrides)
        return cfg

    def test_llm_partial_updates_replace_only_tail(self):
        previews = []
        partials = ["今天下午", "今天下午开会"]

        def polish_tail(prefix_context, tail_text, on_update):
            for partial in partials:
                on_update(partial)
            return partials[-1]

        session = StreamingSession(
            sample_rate=16000,
            streaming_cfg=self._streaming_cfg(),
            detect_speech=lambda samples, is_final=False: bool(samples.max()),
            transcribe_segment=lambda samples: {"success": True, "text": "今天下午开会"},
            polish_tail=polish_tail,
            on_preview=lambda committed, tail, state: previews.append((committed, tail, state)),
        )
        session.start()
        session.push_chunk(np.ones(1600, dtype=np.int16))
        session.push_chunk(np.zeros(1600, dtype=np.int16))
        session.push_chunk(np.zeros(1600, dtype=np.int16))
        session.finish()

        self.assertIn(("", "今天下午", "processing"), previews)
        self.assertIn(("", "今天下午开会", "processing"), previews)

    def test_commit_with_polish_promotes_polished_text_into_committed(self):
        previews = []

        def polish_tail(prefix_context, tail_text, on_update):
            on_update("润色中")
            return "润色后的最终文本"

        session = StreamingSession(
            sample_rate=16000,
            streaming_cfg=self._streaming_cfg(),
            detect_speech=lambda samples, is_final=False: bool(samples.max()),
            transcribe_segment=lambda samples: {"success": True, "text": "原始文本"},
            polish_tail=polish_tail,
            on_preview=lambda committed, tail, state: previews.append((committed, tail, state)),
        )
        session.start()
        session.push_chunk(np.ones(1600, dtype=np.int16))
        session.finish()

        committed_events = [e for e in previews if e[0] and e[1] == "" and e[2] == "recording"]
        self.assertTrue(committed_events, f"expected a final committed event, got: {previews}")
        self.assertEqual(committed_events[-1][0], "润色后的最终文本")


class StreamingSessionReuseTests(unittest.TestCase):
    def test_session_can_be_started_and_finished_multiple_times(self):
        # Regression for Bug A: StreamingSession is constructed once at app
        # startup and reused across every recording. `finish()` permanently
        # shuts down the thread pools, so a second `start()` must rebuild
        # fresh ones -- otherwise `push_chunk()`/`finish()` raise
        # RuntimeError: cannot schedule new futures after shutdown.
        events = []
        session = StreamingSession(
            sample_rate=16000,
            streaming_cfg={
                "segment_silence_ms": 200,
                "commit_silence_ms": 400,
                "max_uncommitted_ms": 100000,
                "audio_overlap_ms": 100,
            },
            detect_speech=lambda samples, is_final=False: bool(samples.max()),
            transcribe_segment=lambda samples: {"success": True, "text": "文本"},
            on_preview=lambda committed, tail, state: events.append((committed, tail, state)),
        )

        # First recording.
        session.start()
        session.push_chunk(np.ones(1600, dtype=np.int16))
        session.push_chunk(np.zeros(1600, dtype=np.int16))
        session.push_chunk(np.zeros(1600, dtype=np.int16))
        session.finish()

        self.assertTrue(events, "first recording should have produced preview events")
        events.clear()

        # Second recording on the SAME StreamingSession instance -- must not
        # raise despite the pools having been shut down by the first finish().
        session.start()
        session.push_chunk(np.ones(1600, dtype=np.int16))
        session.push_chunk(np.zeros(1600, dtype=np.int16))
        session.push_chunk(np.zeros(1600, dtype=np.int16))
        session.finish()

        self.assertTrue(events, "second recording should also have produced preview events")


if __name__ == "__main__":
    unittest.main()
