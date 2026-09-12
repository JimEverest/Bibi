import unittest
from unittest.mock import Mock

from app.transcribe import TranscriptionWorker


class TranscriptionWorkerStreamingTests(unittest.TestCase):
    def test_capture_loop_fans_out_chunks_to_streaming_session(self):
        worker = TranscriptionWorker.__new__(TranscriptionWorker)
        worker._streaming_session = Mock()
        worker._recording = Mock()
        worker._recording.is_set.side_effect = [True, False]
        worker.audio = Mock()
        worker.audio.queue.get.return_value = __import__("numpy").ones(8, dtype=__import__("numpy").int16)
        worker._buffer = []
        worker._buffer_lock = __import__("threading").Lock()
        worker._stop_requested = Mock()
        worker._stop_requested.is_set.return_value = False
        worker._session_bytes = 0
        worker._max_session_bytes = 999999
        worker.gain = 1.0

        worker._capture_loop()

        worker._streaming_session.push_chunk.assert_called_once()

    def test_stop_finishes_streaming_session_before_return(self):
        worker = TranscriptionWorker.__new__(TranscriptionWorker)
        worker._streaming_session = Mock()
        worker._state_lock = __import__("threading").RLock()
        worker._running = Mock()
        worker._running.is_set.return_value = True
        worker._stop_requested = Mock()
        worker._recording = Mock()
        worker._capture_thread = None
        worker.audio = Mock()
        worker._combine_buffer = Mock(return_value=__import__("numpy").ones(8, dtype=__import__("numpy").int16))
        worker._transcription_queue = Mock()
        worker._transcription_queue.put_nowait = Mock()
        worker._buffer_lock = __import__("threading").Lock()
        worker._current_session_id = 1
        worker._transcription_task_count = 0
        worker._session_bytes = 0
        worker._max_session_bytes = 999999

        worker.stop()

        worker._streaming_session.finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
