from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


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
    kind: Literal["preview", "commit"] = "preview"


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
        # Tracks whether the uncommitted buffer holds any genuinely new audio
        # (real speech, or silence appended after speech) since the buffer
        # was last cleared to its onset pad by a commit. This is distinct
        # from `bool(self._uncommitted_chunks)`, which stays True even when
        # the buffer contains *only* the leftover onset pad with nothing new
        # added -- that pad is a copy of already-committed audio, not new
        # content, and must not trigger a spurious extra commit on
        # `is_final`.
        self._has_new_content_since_commit = False

    def push_chunk(self, samples: np.ndarray, is_speech: bool, is_final: bool = False) -> list[SegmentTask]:
        emitted: list[SegmentTask] = []

        if is_speech:
            self._uncommitted_chunks.append(samples)
            self._uncommitted_total_samples += samples.size
            self._silence_samples = 0
            self._preview_emitted_this_pause = False
            self._commit_emitted_this_pause = False
            self._has_new_content_since_commit = True
            if self._uncommitted_total_samples >= self._max_uncommitted_samples:
                emitted.append(self._emit_task(kind="commit", clear_to_pad=True))
        elif self._uncommitted_chunks:
            self._uncommitted_chunks.append(samples)
            self._uncommitted_total_samples += samples.size
            self._silence_samples += samples.size
            if samples.size > 0:
                self._has_new_content_since_commit = True
            if not self._commit_emitted_this_pause and self._silence_samples >= self._commit_silence_samples_limit:
                emitted.append(self._emit_task(kind="commit", clear_to_pad=True))
                self._commit_emitted_this_pause = True
                self._preview_emitted_this_pause = True
            elif not self._preview_emitted_this_pause and self._silence_samples >= self._silence_samples_limit:
                emitted.append(self._emit_task(kind="preview", clear_to_pad=False))
                self._preview_emitted_this_pause = True

        if is_final and self._has_new_content_since_commit:
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
            self._has_new_content_since_commit = False

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
        # Guards the *identity* of `self._segmenter`/`self._asr_pool`/
        # `self._llm_pool` (and `self._closed`): serializes push_chunk/finish
        # segmenter access + task submission against start()'s swap-in of
        # brand-new objects for a new recording. It does NOT protect
        # `self._preview` state -- that's `self._preview_lock`'s job, since
        # preview mutation happens from the ASR/LLM worker threads and is an
        # unrelated concern. Do not nest acquisitions of this lock.
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
        is_speech = self._detect_speech(samples, is_final=False)
        with self._lock:
            if self._closed:
                return
            for task in self._segmenter.push_chunk(samples, is_speech=is_speech, is_final=False):
                self._asr_pool.submit(self._run_asr_task, task)

    def finish(self) -> None:
        with self._lock:
            self._closed = True
            dummy = np.zeros(0, dtype=np.int16)
            for task in self._segmenter.push_chunk(dummy, is_speech=False, is_final=True):
                self._asr_pool.submit(self._run_asr_task, task)
            asr_pool = self._asr_pool
            llm_pool = self._llm_pool
        # Shutdown can block for a while waiting on in-flight ASR/LLM work;
        # don't hold `_lock` here since it doesn't touch shared segmenter
        # state and would otherwise block a concurrent start() unnecessarily.
        asr_pool.shutdown(wait=True)
        llm_pool.shutdown(wait=True)

    def _promote_and_notify(self) -> None:
        with self._preview_lock:
            snap = self._preview.promote_tail()
            self._on_preview(snap.committed, snap.tail, "recording")

    def _run_asr_task(self, task: SegmentTask) -> None:
        result = self._transcribe_segment(task.samples)
        if not result.get("success"):
            if task.kind == "commit":
                logger.warning(
                    "ASR failed for commit segment (sequence=%s); audio already "
                    "cleared from buffer, transcript is lost",
                    task.sequence,
                )
            return
        is_commit = task.kind == "commit"
        prefix_context = None
        with self._preview_lock:
            snap = self._preview.apply_asr_segment(task.sequence, result.get("text", ""))
            self._on_preview(snap.committed, snap.tail, "recording")
            should_polish = self._polish_tail is not None and bool(snap.tail)
            if should_polish:
                prefix_context = self._preview.context_prefix()

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
            self._promote_and_notify()

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
            self._promote_and_notify()
