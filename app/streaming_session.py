from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading

import numpy as np


@dataclass(frozen=True)
class PreviewSnapshot:
    committed: str
    tail: str
    state: str = "recording"

    @property
    def visible_text(self) -> str:
        return f"{self.committed}{self.tail}"


class PreviewAssembler:
    def __init__(self, context_chars: int = 30) -> None:
        self._context_chars = context_chars
        self._committed = ""
        self._tail = ""
        self._latest_sequence = 0
        self._llm_sequence = 0

    def snapshot(self, state: str = "recording") -> PreviewSnapshot:
        return PreviewSnapshot(self._committed, self._tail, state=state)

    def apply_asr_segment(self, sequence: int, text: str) -> PreviewSnapshot:
        if sequence < self._latest_sequence:
            return self.snapshot()
        self._latest_sequence = sequence
        self._llm_sequence = sequence
        self._tail = self._merge_tail(self._tail, text)
        return self.snapshot()

    def apply_llm_update(self, sequence: int, text: str) -> PreviewSnapshot:
        if sequence != self._llm_sequence:
            return self.snapshot()
        self._tail = text
        return self.snapshot(state="processing")

    def promote_tail(self) -> PreviewSnapshot:
        self._committed += self._tail
        self._tail = ""
        return self.snapshot()

    def context_prefix(self) -> str:
        return self._committed[-self._context_chars :]

    @staticmethod
    def _merge_tail(existing_tail: str, new_text: str) -> str:
        max_overlap = min(len(existing_tail), len(new_text))
        for size in range(max_overlap, 0, -1):
            if existing_tail[-size:] == new_text[:size]:
                return existing_tail + new_text[size:]
        return new_text


@dataclass(frozen=True)
class SegmentTask:
    sequence: int
    samples: np.ndarray


class StreamingSegmenter:
    def __init__(self, sample_rate: int, silence_ms: int, overlap_ms: int) -> None:
        self._sample_rate = sample_rate
        self._silence_samples_limit = int(sample_rate * silence_ms / 1000)
        self._overlap_samples = int(sample_rate * overlap_ms / 1000)
        self._active_chunks: list[np.ndarray] = []
        self._silence_samples = 0
        self._previous_tail = np.zeros(0, dtype=np.int16)
        self._sequence = 0

    def push_chunk(self, samples: np.ndarray, is_speech: bool, is_final: bool = False) -> list[SegmentTask]:
        emitted: list[SegmentTask] = []
        if is_speech:
            self._active_chunks.append(samples)
            self._silence_samples = 0
        elif self._active_chunks:
            self._active_chunks.append(samples)
            self._silence_samples += samples.size
            if self._silence_samples >= self._silence_samples_limit:
                emitted.append(self._emit_segment())
        if is_final and self._active_chunks:
            emitted.append(self._emit_segment())
        return emitted

    def _emit_segment(self) -> SegmentTask:
        current = np.concatenate(self._active_chunks, axis=0)
        merged = np.concatenate([self._previous_tail, current], axis=0)
        self._previous_tail = current[-self._overlap_samples :].copy()
        self._active_chunks.clear()
        self._silence_samples = 0
        self._sequence += 1
        return SegmentTask(sequence=self._sequence, samples=merged)


class StreamingSession:
    def __init__(
        self,
        sample_rate: int,
        streaming_cfg: dict,
        detect_speech,
        transcribe_segment,
        on_preview,
        polish_tail=None,
    ) -> None:
        self._sample_rate = sample_rate
        self._detect_speech = detect_speech
        self._transcribe_segment = transcribe_segment
        self._polish_tail = polish_tail
        self._on_preview = on_preview
        self._preview = PreviewAssembler(
            context_chars=int(streaming_cfg.get("preview_context_chars", 30))
        )
        self._segmenter = StreamingSegmenter(
            sample_rate=sample_rate,
            silence_ms=int(streaming_cfg.get("segment_silence_ms", 450)),
            overlap_ms=int(streaming_cfg.get("audio_overlap_ms", 500)),
        )
        self._asr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-asr")
        self._llm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-llm")
        self._lock = threading.Lock()
        self._closed = True

    def start(self) -> None:
        with self._lock:
            self._closed = False

    def push_chunk(self, samples: np.ndarray) -> None:
        with self._lock:
            if self._closed:
                return
        is_speech = self._detect_speech(samples, is_final=False)
        for task in self._segmenter.push_chunk(samples, is_speech=is_speech, is_final=False):
            self._asr_pool.submit(self._run_asr_task, task)

    def finish(self) -> None:
        with self._lock:
            self._closed = True
        dummy = np.zeros(0, dtype=np.int16)
        for task in self._segmenter.push_chunk(dummy, is_speech=False, is_final=True):
            self._asr_pool.submit(self._run_asr_task, task)
        self._asr_pool.shutdown(wait=True)
        self._llm_pool.shutdown(wait=True)

    def _run_asr_task(self, task: SegmentTask) -> None:
        result = self._transcribe_segment(task.samples)
        if not result.get("success"):
            return
        snap = self._preview.apply_asr_segment(task.sequence, result.get("text", ""))
        self._on_preview(snap.committed, snap.tail, "recording")
        if self._polish_tail is not None and snap.tail:
            self._llm_pool.submit(
                self._run_llm_task,
                task.sequence,
                self._preview.context_prefix(),
                snap.tail,
            )

    def _run_llm_task(self, sequence: int, prefix_context: str, tail_text: str) -> None:
        def _on_update(partial: str) -> None:
            snap = self._preview.apply_llm_update(sequence, partial)
            self._on_preview(snap.committed, snap.tail, "processing")

        final_text = self._polish_tail(prefix_context, tail_text, _on_update)
        if final_text:
            snap = self._preview.apply_llm_update(sequence, final_text)
            self._on_preview(snap.committed, snap.tail, "processing")
