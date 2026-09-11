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
        # Every ASR segment is a fresh, full re-decode of the entire growing
        # uncommitted audio window, so its text is always the authoritative
        # current tail -- no character-level splicing needed or wanted.
        self._tail = text
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


@dataclass(frozen=True)
class SegmentTask:
    sequence: int
    samples: np.ndarray
    kind: str = "preview"


class StreamingSegmenter:
    """Accumulates audio chunks and decides when to emit preview/commit tasks.

    A single, ever-growing "uncommitted buffer" holds all speech (and any
    silence interleaved after speech has started) since the last COMMIT.
    Short pauses re-decode and re-emit the *entire* uncommitted buffer as a
    cheap "preview" (buffer keeps growing). Long pauses (or an unbounded
    uncommitted buffer) emit a "commit": the whole buffer is transcribed one
    final time and then the buffer is cleared down to a small onset pad so
    the next segment has a little audio-only context to smooth into.
    """

    def __init__(
        self,
        sample_rate: int,
        silence_ms: int,
        commit_silence_ms: int,
        max_uncommitted_ms: int,
        overlap_ms: int,
    ) -> None:
        self._sample_rate = sample_rate
        self._silence_samples_limit = int(sample_rate * silence_ms / 1000)
        self._commit_silence_samples_limit = int(sample_rate * commit_silence_ms / 1000)
        self._max_uncommitted_samples = int(sample_rate * max_uncommitted_ms / 1000)
        self._overlap_samples = int(sample_rate * overlap_ms / 1000)

        self._uncommitted_chunks: list[np.ndarray] = []
        self._uncommitted_total_samples = 0
        self._silence_samples = 0
        self._preview_emitted_this_pause = False
        self._commit_emitted_this_pause = False
        self._sequence = 0

    def push_chunk(self, samples: np.ndarray, is_speech: bool, is_final: bool = False) -> list[SegmentTask]:
        emitted: list[SegmentTask] = []

        if is_speech:
            self._uncommitted_chunks.append(samples)
            self._uncommitted_total_samples += samples.size
            self._silence_samples = 0
            self._preview_emitted_this_pause = False
            self._commit_emitted_this_pause = False
            if self._uncommitted_total_samples >= self._max_uncommitted_samples:
                emitted.append(self._emit_task(kind="commit", clear_to_pad=True))
        elif self._uncommitted_chunks:
            self._uncommitted_chunks.append(samples)
            self._uncommitted_total_samples += samples.size
            self._silence_samples += samples.size
            if not self._commit_emitted_this_pause and self._silence_samples >= self._commit_silence_samples_limit:
                emitted.append(self._emit_task(kind="commit", clear_to_pad=True))
                self._commit_emitted_this_pause = True
                self._preview_emitted_this_pause = True
            elif not self._preview_emitted_this_pause and self._silence_samples >= self._silence_samples_limit:
                emitted.append(self._emit_task(kind="preview", clear_to_pad=False))
                self._preview_emitted_this_pause = True

        if is_final and self._uncommitted_chunks:
            emitted.append(self._emit_task(kind="commit", clear_to_pad=True))

        return emitted

    def _emit_task(self, kind: str, clear_to_pad: bool) -> SegmentTask:
        merged = np.concatenate(self._uncommitted_chunks, axis=0)
        self._sequence += 1
        task = SegmentTask(sequence=self._sequence, samples=merged, kind=kind)

        if clear_to_pad:
            if self._overlap_samples > 0 and merged.size > 0:
                pad = merged[-self._overlap_samples :].copy()
            else:
                pad = np.zeros(0, dtype=merged.dtype)
            if pad.size > 0:
                self._uncommitted_chunks = [pad]
                self._uncommitted_total_samples = pad.size
            else:
                self._uncommitted_chunks = []
                self._uncommitted_total_samples = 0

        return task


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
        self._streaming_cfg = streaming_cfg
        self._detect_speech = detect_speech
        self._transcribe_segment = transcribe_segment
        self._polish_tail = polish_tail
        self._on_preview = on_preview
        self._lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._closed = True
        self._reset_for_new_recording()

    def _reset_for_new_recording(self) -> None:
        """(Re)build all per-recording state: pools, segmenter, assembler.

        Must be called both at construction time and at the start of every
        recording. `finish()` permanently shuts down the thread pools, so a
        stale `StreamingSession` reused across recordings needs brand-new
        pools each time `start()` is called -- otherwise `push_chunk()` would
        try to submit work to an already-shut-down executor.
        """
        streaming_cfg = self._streaming_cfg
        self._preview = PreviewAssembler(
            context_chars=int(streaming_cfg.get("preview_context_chars", 30))
        )
        self._segmenter = StreamingSegmenter(
            sample_rate=self._sample_rate,
            silence_ms=int(streaming_cfg.get("segment_silence_ms", 450)),
            commit_silence_ms=int(streaming_cfg.get("commit_silence_ms", 1200)),
            max_uncommitted_ms=int(streaming_cfg.get("max_uncommitted_ms", 12000)),
            overlap_ms=int(streaming_cfg.get("audio_overlap_ms", 150)),
        )
        self._asr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-asr")
        self._llm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-llm")

    def start(self) -> None:
        with self._lock:
            self._reset_for_new_recording()
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
        is_commit = task.kind == "commit"
        prefix_context = None
        with self._preview_lock:
            snap = self._preview.apply_asr_segment(task.sequence, result.get("text", ""))
            self._on_preview(snap.committed, snap.tail, "recording")
            if self._polish_tail is not None and snap.tail:
                prefix_context = self._preview.context_prefix()

        should_polish = self._polish_tail is not None and bool(snap.tail)
        if should_polish:
            self._llm_pool.submit(
                self._run_llm_task,
                task.sequence,
                prefix_context,
                snap.tail,
                is_commit,
            )
        elif is_commit:
            # No LLM polishing configured (or nothing to polish): the commit
            # is authoritative on its own, so promote it into `committed`
            # immediately instead of leaving it stranded in `tail` forever.
            with self._preview_lock:
                snap2 = self._preview.promote_tail()
                self._on_preview(snap2.committed, snap2.tail, "recording")

    def _run_llm_task(self, sequence: int, prefix_context: str, tail_text: str, is_commit: bool = False) -> None:
        def _on_update(partial: str) -> None:
            with self._preview_lock:
                snap = self._preview.apply_llm_update(sequence, partial)
                self._on_preview(snap.committed, snap.tail, "processing")

        final_text = self._polish_tail(prefix_context, tail_text, _on_update)
        if final_text:
            with self._preview_lock:
                snap = self._preview.apply_llm_update(sequence, final_text)
                self._on_preview(snap.committed, snap.tail, "processing")

        if is_commit:
            # The polished (or best-effort) text for this segment is now
            # final: promote it into `committed` and revert state back to
            # "recording" so the UI doesn't sit in "processing" forever.
            with self._preview_lock:
                snap = self._preview.promote_tail()
                self._on_preview(snap.committed, snap.tail, "recording")
