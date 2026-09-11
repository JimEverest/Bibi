import threading
import unittest
from unittest.mock import Mock

import numpy as np

from app.transcribe import TranscriptionWorker


def _make_worker() -> TranscriptionWorker:
    worker = TranscriptionWorker.__new__(TranscriptionWorker)
    worker.audio = Mock()
    worker._streaming_session = None
    worker._state_lock = threading.RLock()
    worker._running = threading.Event()
    worker._running.set()
    worker._recording = threading.Event()
    worker._recording.set()
    worker._stop_requested = threading.Event()
    worker._stop_requested.set()
    worker._capture_thread = Mock()
    worker._current_session_id = 1
    worker._buffer_lock = threading.Lock()
    worker._buffer = [np.ones(4, dtype=np.int16)]
    return worker


class ForceStopTests(unittest.TestCase):
    def test_force_reset_swallows_runtime_error_from_streaming_session(self):
        worker = _make_worker()
        worker._streaming_session = Mock()
        worker._streaming_session.finish.side_effect = RuntimeError(
            "cannot schedule new futures after shutdown"
        )

        worker.force_reset()  # must not raise

        self.assertFalse(worker.is_running)

    def test_force_reset_swallows_audio_stop_exception(self):
        worker = _make_worker()
        worker.audio.stop.side_effect = Exception("boom")

        worker.force_reset()  # must not raise

        self.assertFalse(worker.is_running)

    def test_force_reset_clears_state_and_buffer(self):
        worker = _make_worker()

        worker.force_reset()

        self.assertFalse(worker.is_running)
        self.assertFalse(worker._recording.is_set())
        self.assertFalse(worker._stop_requested.is_set())
        self.assertIsNone(worker._capture_thread)
        self.assertIsNone(worker._current_session_id)
        with worker._buffer_lock:
            self.assertEqual(worker._buffer, [])

    def test_force_reset_handles_no_streaming_session(self):
        worker = _make_worker()
        worker._streaming_session = None

        worker.force_reset()  # must not raise

        self.assertFalse(worker.is_running)


if __name__ == "__main__":
    unittest.main()
