# Streaming Dictation v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fast local pseudo-streaming dictation preview by segmenting live microphone audio with VAD, transcribing stable local segments with the existing offline FunASR model, streaming LLM polish for the editable tail, and still injecting final text only once at session end.

**Architecture:** Keep the current batch transcription path as the source of truth, then add a side pipeline for preview. `TranscriptionWorker` fans out chunks into a new `StreamingSession`, which uses local-only VAD + segment ASR + tail-only LLM streaming to update the floating capsule preview without typing partial text into the target application.

**Tech Stack:** Python 3.13, `unittest`, `threading`, `concurrent.futures`, `numpy`, `urllib`, Tkinter, existing `funasr_onnx` models, existing `UIHub`, existing floating capsule.

---

## File Map

- Modify: `app/config.py`
  - Add conservative `streaming` config defaults.
- Modify: `app/download_models.py`
  - Add a local-only cache lookup helper that never downloads.
- Modify: `app/funasr_server.py`
  - Add `transcribe_samples(...)` and a local-only streaming-VAD detector factory.
- Create: `app/streaming_session.py`
  - Own `PreviewSnapshot`, `PreviewAssembler`, `StreamingSegmenter`, and `StreamingSession`.
- Modify: `app/floating_button.py`
  - Add preview rendering API and collapsed-state-safe preview behavior.
- Modify: `app/llm_polish.py`
  - Add SSE streaming support and a tail-only streaming polish API.
- Modify: `app/transcribe.py`
  - Fan out chunks to a streaming session and flush preview work on stop.
- Modify: `main.py`
  - Wire `StreamingSession` only for `backend == "funasr"` and preview-capable UI sessions.
- Create: `tests/test_streaming_session.py`
  - Test config defaults, local-only model guard, preview merge logic, segmentation, sequence handling.
- Create: `tests/test_floating_preview.py`
  - Test preview rendering and UI state transitions.
- Create: `tests/test_llm_streaming.py`
  - Test SSE parsing, incremental callbacks, and fallback behavior.
- Create: `tests/test_transcription_worker_streaming.py`
  - Test chunk fan-out and stop/finalize ordering.

## Task 1: Add local-only streaming config and model cache guard

**Files:**
- Modify: `app/config.py`
- Modify: `app/download_models.py`
- Test: `tests/test_streaming_session.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.test_streaming_session.StreamingConfigTests -v`

Expected: FAIL with `KeyError: 'streaming'` and `ImportError` for `get_existing_model_cache_path`.

- [ ] **Step 3: Write minimal implementation**

`app/config.py`

```python
DEFAULT_CONFIG: Dict[str, Any] = {
    "hotkeys": {
        "toggle": "f2",
        "toggle_enabled": True,
        "push_to_talk": "win+ctrl+alt",
        "push_to_talk_enabled": True,
    },
    "audio": {
        "sample_rate": 16000,
        "block_ms": 20,
        "device": None,
        "max_session_bytes": 2 * 1024 * 1024,
        "gain": 12.0,
        "save_recordings": False,
    },
    "vad": {
        "start_threshold": 0.02,
        "stop_threshold": 0.01,
        "min_speech_ms": 300,
        "min_silence_ms": 200,
        "pad_ms": 200,
    },
    "backend": "funasr",
    "asr": {
        "use_vad": False,
        "use_punc": True,
        "language": "zh",
        "hotword": "",
        "batch_size_s": 60.0,
    },
    "streaming": {
        "enabled": True,
        "segment_silence_ms": 450,
        "audio_overlap_ms": 500,
        "preview_context_chars": 30,
        "preview_max_chars": 120,
        "ui_update_debounce_ms": 80,
        "dedicated_model_instance": False,
    },
}
```

`app/download_models.py`

```python
from pathlib import Path


def get_existing_model_cache_path(
    model_name: str,
    required_files: tuple[str, ...] = ("model.onnx", "model_quant.onnx"),
) -> str:
    home = Path.home()
    cache_base = home / ".cache" / "modelscope" / "hub" / "models" / "iic"
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    model_dir = cache_base / short_name

    if not model_dir.exists():
        raise FileNotFoundError(f"Local model cache not found: {model_name}")

    if not any((model_dir / file_name).exists() for file_name in required_files):
        raise FileNotFoundError(
            f"Local model cache is incomplete for {model_name}: need one of {required_files}"
        )

    return str(model_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.test_streaming_session.StreamingConfigTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/download_models.py tests/test_streaming_session.py
git commit -m "feat: add local-only streaming config defaults"
```

## Task 2: Build preview snapshot and rollback-safe tail merge logic

**Files:**
- Create: `app/streaming_session.py`
- Modify: `tests/test_streaming_session.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from app.streaming_session import PreviewAssembler


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.test_streaming_session.PreviewAssemblerTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.streaming_session'`.

- [ ] **Step 3: Write minimal implementation**

`app/streaming_session.py`

```python
from __future__ import annotations

from dataclasses import dataclass


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

    @staticmethod
    def _merge_tail(existing_tail: str, new_text: str) -> str:
        max_overlap = min(len(existing_tail), len(new_text))
        for size in range(max_overlap, 0, -1):
            if existing_tail[-size:] == new_text[:size]:
                return existing_tail + new_text[size:]
        return new_text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.test_streaming_session.PreviewAssemblerTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/streaming_session.py tests/test_streaming_session.py
git commit -m "feat: add rollback-safe preview assembler"
```

## Task 3: Add floating capsule preview rendering API

**Files:**
- Modify: `app/floating_button.py`
- Create: `tests/test_floating_preview.py`

- [ ] **Step 1: Write the failing test**

```python
import threading
import unittest

from app.floating_button import render_preview_text, FloatingButton


class FloatingPreviewTests(unittest.TestCase):
    def test_render_preview_text_truncates_from_left(self):
        committed = "0123456789" * 15
        tail = "最新尾巴"
        rendered = render_preview_text(committed, tail, max_chars=20)
        self.assertTrue(rendered.startswith("…"))
        self.assertTrue(rendered.endswith("最新尾巴"))
        self.assertLessEqual(len(rendered), 20)

    def test_show_preview_updates_internal_state_and_dispatches(self):
        button = FloatingButton.__new__(FloatingButton)
        button._lock = threading.Lock()
        button._preview_max_chars = 40
        button._preview_text = ""
        button._state = "idle"
        dispatched = []
        button._dispatch = lambda fn: dispatched.append(fn)

        button.show_preview("稳定前缀", "新尾巴", state="recording")

        self.assertEqual(button._preview_text, "稳定前缀新尾巴")
        self.assertEqual(button._state, "recording")
        self.assertEqual(len(dispatched), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.test_floating_preview -v`

Expected: FAIL with `ImportError` for `render_preview_text` and `AttributeError` for `show_preview`.

- [ ] **Step 3: Write minimal implementation**

**Confirmed UI decision:** the existing capsule is a fixed `_EXPANDED_SIZE = (176, 54)` block with no room for growing text. Do not try to cram preview text into the existing `main` label. Instead:

- add a new `_PREVIEW_SIZE = (280, 78)` constant (wider + taller than `_EXPANDED_SIZE`);
- add a `preview` `tk.Label` as a child of `root` (a sibling of `bar`/`shell`, packed *after* `shell` with `side="top", fill="x"`), created in `_ensure_built()`'s `_build()` alongside the existing widgets, but **not packed by default** — only `.pack(...)` it when there's preview text to show, and `.pack_forget()` it when there isn't. This makes the resize purely geometry-driven: packing/unpacking the extra row plus calling `root.geometry(...)` is what grows/shrinks the capsule;
- `render_preview_text(committed, tail, max_chars=120)` is unchanged from the failing test above: single line, truncate-from-left with a leading `…`, keep the tail intact (this already matches the "scroll to show only the newest content" decision — no separate scrolling widget needed, truncation *is* the scroll effect);
- `show_preview(committed, tail, state)` stores `_preview_text`/`_state` under `self._lock` (as in the failing test) then dispatches `_apply_state`;
- `clear_preview()` clears `_preview_text` and dispatches `_apply_state`;
- `_apply_state()` (existing method) additionally:
  - if `self._collapsed` or `not self._preview_text`: `self._preview_label.pack_forget()` and resize to `_EXPANDED_SIZE` (reuse the same "keep current `x`/`y`, change `width`/`height`" pattern already used in `_toggle_collapsed()`);
  - else: configure `self._preview_label` text/colors, `.pack(side="top", fill="x", padx=0, pady=(0, 6))` it, and resize to `_PREVIEW_SIZE`;
  - only resize when the target size actually differs from the current one (avoid redundant `geometry()` calls / flicker on every state update).
- Collapsed capsule behavior must stay exactly as today: collapsed state never shows preview text and never triggers the wider/taller geometry, regardless of `_preview_text`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.test_floating_preview -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/floating_button.py tests/test_floating_preview.py
git commit -m "feat: render live preview in floating capsule"
```

## Task 4: Add local VAD segmentation and segment-level ASR preview session

**Files:**
- Modify: `app/funasr_server.py`
- Modify: `app/streaming_session.py`
- Modify: `tests/test_streaming_session.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
import numpy as np

from app.streaming_session import StreamingSegmenter, StreamingSession


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
        segmenter = StreamingSegmenter(sample_rate=16000, silence_ms=200, overlap_ms=100)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.test_streaming_session.StreamingSegmenterTests tests.test_streaming_session.StreamingSessionTests -v`

Expected: FAIL with missing `StreamingSegmenter` / `StreamingSession` / `start` / `push_chunk` / `finish`.

- [ ] **Step 3: Write minimal implementation**

**Confirmed VAD decision:** `Fsmn_vad_online` returns boundary *events*, not a
per-chunk speech flag (`bool(segments)` would misclassify most in-utterance
chunks as silence). Use a stateful RMS/energy threshold detector instead,
driven by the existing (currently unused) `vad.*` config keys. No new
model, no `app.download_models` import needed here.

`app/funasr_server.py`

```python
import numpy as np


def build_streaming_speech_detector(vad_cfg: dict, sample_rate: int):
    start_threshold = float(vad_cfg.get("start_threshold", 0.02))
    stop_threshold = float(vad_cfg.get("stop_threshold", 0.01))
    min_speech_ms = float(vad_cfg.get("min_speech_ms", 300))
    min_silence_ms = float(vad_cfg.get("min_silence_ms", 200))

    state = {"active": False, "speech_ms": 0.0, "silence_ms": 0.0}

    def detect(samples: np.ndarray, is_final: bool = False) -> bool:
        if samples.size == 0:
            return state["active"]
        rms = float(np.sqrt(np.mean((samples.astype("float32") / 32768.0) ** 2)))
        chunk_ms = samples.size / sample_rate * 1000.0
        if rms >= start_threshold:
            state["speech_ms"] += chunk_ms
            state["silence_ms"] = 0.0
            if state["speech_ms"] >= min_speech_ms:
                state["active"] = True
        elif rms <= stop_threshold:
            state["silence_ms"] += chunk_ms
            state["speech_ms"] = 0.0
            if state["silence_ms"] >= min_silence_ms:
                state["active"] = False
        return state["active"]

    return detect
```

Add this as a module-level function in `app/funasr_server.py` (it needs no
`FunASRServer` instance state — no model, no lock). `min_speech_ms` /
`min_silence_ms` gate against flicker on brief blips; `pad_ms` is not used
directly here because `StreamingSegmenter`'s `audio_overlap_ms` already
retains trailing context across segment boundaries — don't add a second,
overlapping padding mechanism.

```python
def transcribe_samples(self, samples, sample_rate: int, options=None):
    import os
    import tempfile
    import wave

    fd, path = tempfile.mkstemp(prefix="stream_segment_", suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.astype("int16").tobytes())
        return self.transcribe_audio(path, options=options)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
```

`app/streaming_session.py`

```python
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import numpy as np


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
```

```python
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

    def _run_asr_task(self, task: SegmentTask) -> None:
        result = self._transcribe_segment(task.samples)
        if not result.get("success"):
            return
        snap = self._preview.apply_asr_segment(task.sequence, result.get("text", ""))
        self._on_preview(snap.committed, snap.tail, "recording")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.test_streaming_session.StreamingSegmenterTests tests.test_streaming_session.StreamingSessionTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/funasr_server.py app/streaming_session.py tests/test_streaming_session.py
git commit -m "feat: add local streaming segment preview session"
```

## Task 5: Add tail-only streaming LLM polish

**Files:**
- Modify: `app/llm_polish.py`
- Create: `tests/test_llm_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from unittest.mock import patch

from app.llm_polish import LLMPolisher


class LLMStreamingTests(unittest.TestCase):
    def test_openai_stream_calls_on_update_incrementally(self):
        updates = []
        polisher = LLMPolisher(
            {
                "enabled": True,
                "schema": "openai",
                "endpoint": "http://localhost/v1/chat/completions",
                "model": "demo",
            }
        )
        events = [
            {"choices": [{"delta": {"content": "你好"}}]},
            {"choices": [{"delta": {"content": "世界"}}]},
        ]
        with patch.object(polisher, "_http_post_sse", return_value=iter(events)):
            result = polisher.polish_tail_stream("稳定前缀", "你好世界", on_update=updates.append)

        self.assertEqual(result, "你好世界")
        self.assertEqual(updates, ["你好", "你好世界"])

    def test_stream_failure_returns_none(self):
        polisher = LLMPolisher(
            {
                "enabled": True,
                "schema": "anthropic",
                "endpoint": "http://localhost/v1/messages",
                "model": "demo",
            }
        )
        with patch.object(polisher, "_http_post_sse", side_effect=TimeoutError("boom")):
            result = polisher.polish_tail_stream("稳定前缀", "尾巴")

        self.assertIsNone(result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.test_llm_streaming -v`

Expected: FAIL with `AttributeError: 'LLMPolisher' object has no attribute 'polish_tail_stream'`.

- [ ] **Step 3: Write minimal implementation**

`app/llm_polish.py`

```python
def _build_tail_stream_prompt(self, prefix_context: str, tail_text: str, extra_vocab: list[str] | None = None) -> str:
    vocab_section = self._build_vocab_section(extra_vocab)
    return (
        "你会收到不可改写的参考前文和一个可编辑尾巴。"
        "只能润色尾巴，禁止改写参考前文。\n\n"
        f"参考前文（不可改写）：\n{prefix_context}\n\n"
        f"可编辑尾巴：\n{tail_text}\n\n"
        f"{vocab_section}"
        "直接返回润色后的尾巴，不要解释。"
    )
```

```python
def _http_post_sse(self, url: str, headers: dict, body: dict):
    data = json.dumps(body).encode("utf-8")
    opener = urllib.request.build_opener()
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    with opener.open(req, timeout=self.timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            yield json.loads(payload)
```

```python
def polish_tail_stream(
    self,
    prefix_context: str,
    tail_text: str,
    extra_vocab: list[str] | None = None,
    on_update=None,
) -> str | None:
    if not self.is_enabled() or not tail_text.strip():
        return None

    prompt = self._build_tail_stream_prompt(prefix_context, tail_text, extra_vocab)
    pieces: list[str] = []

    try:
        if self.schema == "anthropic":
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": prompt}],
            }
            stream = self._http_post_sse(self.endpoint, headers, body)
            for event in stream:
                if event.get("type") != "content_block_delta":
                    continue
                delta = (event.get("delta") or {}).get("text", "")
                if delta:
                    pieces.append(delta)
                    if on_update is not None:
                        on_update(self._clean("".join(pieces)))
        else:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "stream": True,
            }
            stream = self._http_post_sse(self.endpoint, headers, body)
            for event in stream:
                choices = event.get("choices") or [{}]
                delta = (choices[0].get("delta") or {}).get("content", "")
                if delta:
                    pieces.append(delta)
                    if on_update is not None:
                        on_update(self._clean("".join(pieces)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 流式润色失败（降级为 ASR 文本）: %s", exc)
        return None

    text = self._clean("".join(pieces))
    return text or None
```

**Verified against the local proxy:** a live probe of `127.0.0.1:4141/v1/messages`
(anthropic schema) confirmed real SSE (`content_block_delta` events) and showed the
backend also emits a `thinking` content block (`delta={"thinking": "..."}`) and a
`signature_delta` before the `text` block. The `delta.get("text", "")` lookup above
already ignores those safely (they don't have a `"text"` key), and `"thinking": {"type": "disabled"}` in the request body keeps that block minimal. No extra
filtering logic needed — just don't remove the `.get("text", "")` narrowing.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.test_llm_streaming -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/llm_polish.py tests/test_llm_streaming.py
git commit -m "feat: stream llm tail polish updates"
```

## Task 6: Integrate streaming session with worker lifecycle and main UI wiring

**Files:**
- Modify: `app/transcribe.py`
- Modify: `main.py`
- Create: `tests/test_transcription_worker_streaming.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.test_transcription_worker_streaming -v`

Expected: FAIL because `TranscriptionWorker` has no `_streaming_session` handling.

- [ ] **Step 3: Write minimal implementation**

`app/transcribe.py`

```python
class TranscriptionWorker:
    def __init__(self, config_path: Optional[str] = None, on_result=None) -> None:
        self.config = load_config(config_path)
        self.on_result = on_result
        self.log_dir = ensure_logging_dir(self.config)
        self.last_segment_path = None
        self._streaming_session = None
        audio_cfg = self.config["audio"]
        self.gain = float(audio_cfg.get("gain", 1.0) or 1.0)
        self.audio = AudioCapture(
            sample_rate=audio_cfg["sample_rate"],
            block_ms=audio_cfg["block_ms"],
            device=audio_cfg.get("device"),
        )

    def attach_streaming_session(self, session) -> None:
        self._streaming_session = session
```

```python
# in start()
if self._streaming_session is not None:
    self._streaming_session.start()
```

```python
# in _capture_loop(), after arr is appended
if self._streaming_session is not None:
    self._streaming_session.push_chunk(arr.copy())
```

```python
# in stop(), after audio.stop() and before queueing final batch result
if self._streaming_session is not None:
    self._streaming_session.finish()
```

`main.py`

**Confirmed concurrency decision:** segment-level streaming ASR and the
final session-end batch ASR must not corrupt each other's state if they run
concurrently. Two parts:

1. **Always-on safety net in `app/funasr_server.py`:** add
   `self._infer_lock = threading.Lock()` to `FunASRServer.__init__`, and wrap
   the entire body of the existing `transcribe_audio()` method in
   `with self._infer_lock:`. This makes any two concurrent calls into the
   *same* `FunASRServer` instance safe by construction, regardless of which
   caller they come from. `transcribe_samples()` (Task 4) already calls
   `self.transcribe_audio(...)`, so it's covered automatically — no separate
   locking needed there.
2. **`streaming.dedicated_model_instance` config flag decides which instance
   streaming preview uses:**
   - `false` (default): streaming preview reuses `worker.fun_server` — safe
     now because of the lock above, but preview and final batch calls will
     serialize (briefly block each other).
   - `true`: construct a second, independent `FunASRServer()` purely for
     streaming preview, so it never contends with the final batch call at
     all (at the cost of loading a second copy of the ASR/VAD/punc models).

```python
from app.dictionary import TermReplacer
from app.streaming_session import StreamingSession
from app.funasr_server import FunASRServer, build_streaming_speech_detector
```

```python
replacer = TermReplacer()
```

```python
def _stream_transcribe_segment(streaming_fun_server):
    def _transcribe(samples):
        result = streaming_fun_server.transcribe_samples(
            samples,
            sample_rate=config["audio"]["sample_rate"],
            options=config.get("asr"),
        )
        if not result.get("success"):
            return {"success": False, "text": ""}
        result["text"] = replacer.replace(result.get("text", ""))
        return result

    return _transcribe


def _stream_polish_tail(prefix_context, tail_text, on_update):
    polisher = _LLM_POLISHER["instance"]
    if polisher is None or not polisher.is_enabled():
        return None
    return polisher.polish_tail_stream(
        prefix_context,
        tail_text,
        extra_vocab=replacer.vocab_hints(),
        on_update=on_update,
    )
```

```python
streaming_cfg = config.get("streaming", {})
if (
    not args.once
    and floating_button is not None
    and config.get("backend", "funasr").lower() == "funasr"
    and streaming_cfg.get("enabled", True)
):
    if streaming_cfg.get("dedicated_model_instance", False):
        streaming_fun_server = FunASRServer()
    else:
        streaming_fun_server = worker.fun_server

    streaming_session = StreamingSession(
        sample_rate=config["audio"]["sample_rate"],
        streaming_cfg=streaming_cfg,
        detect_speech=build_streaming_speech_detector(
            config.get("vad", {}),
            config["audio"]["sample_rate"],
        ),
        transcribe_segment=_stream_transcribe_segment(streaming_fun_server),
        polish_tail=_stream_polish_tail,
        on_preview=lambda committed, tail, state: floating_button.show_preview(
            committed,
            tail,
            state=state,
        ),
    )
    worker.attach_streaming_session(streaming_session)
```

```python
# in final result handling after successful final output
if floating_button is not None:
    floating_button.clear_preview()
    floating_button.show_idle()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.test_transcription_worker_streaming -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/transcribe.py main.py tests/test_transcription_worker_streaming.py
git commit -m "feat: wire streaming preview into worker lifecycle"
```

## Task 7: Finish LLM + preview session integration and verify end-to-end behavior

**Files:**
- Modify: `app/streaming_session.py`
- Modify: `app/floating_button.py`
- Modify: `tests/test_streaming_session.py`
- Modify: `tests/test_floating_preview.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from app.streaming_session import StreamingSession


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.test_streaming_session.StreamingSessionLLMTests -v`

Expected: FAIL because `StreamingSession` does not call `polish_tail` or emit incremental processing updates.

- [ ] **Step 3: Write minimal implementation**

`app/streaming_session.py`

```python
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
```

```python
class PreviewAssembler:
    def context_prefix(self) -> str:
        return self._committed[-self._context_chars :]
```

```python
class StreamingSession:
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

    def finish(self) -> None:
        with self._lock:
            self._closed = True
        dummy = np.zeros(0, dtype=np.int16)
        for task in self._segmenter.push_chunk(dummy, is_speech=False, is_final=True):
            self._asr_pool.submit(self._run_asr_task, task)
        self._asr_pool.shutdown(wait=True)
        self._llm_pool.shutdown(wait=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.test_streaming_session.StreamingSessionLLMTests -v`

Expected: PASS.

- [ ] **Step 5: Run the focused streaming suite**

Run:

```bash
.venv/Scripts/python.exe -m unittest \
  tests.test_streaming_session \
  tests.test_floating_preview \
  tests.test_llm_streaming \
  tests.test_transcription_worker_streaming -v
```

Expected: PASS for all streaming-specific tests.

- [ ] **Step 6: Commit**

```bash
git add app/streaming_session.py app/floating_button.py tests/test_streaming_session.py tests/test_floating_preview.py
git commit -m "feat: stream llm polish into capsule preview"
```

## Task 8: Full verification and user-engaged manual tests

**Files:**
- Modify: none unless a failing verification discovers a bug
- Test: existing and new streaming test suite

- [ ] **Step 1: Run static verification**

Run:

```bash
.venv/Scripts/python.exe -m py_compile \
  app/config.py \
  app/download_models.py \
  app/funasr_server.py \
  app/streaming_session.py \
  app/floating_button.py \
  app/llm_polish.py \
  app/transcribe.py \
  main.py
```

Expected: no output.

- [ ] **Step 2: Run the full test suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 3: Run local UI smoke test**

Run: `! .venv\Scripts\python.exe main.py`

Expected:
- floating capsule appears,
- recording still starts/stops,
- no Tk thread exceptions,
- preview area stays empty until recording begins.

- [ ] **Step 4: Engage the user for first real microphone validation**

Ask the user to speak three utterances:

1. `今天下午开会，讨论 streaming 方案。`
2. `这个 PR 先不要 merge，等我 review 完。`
3. `周三上线，不对，周四上午上线。`

Expected:
- preview appears after natural pauses,
- tail may revise locally,
- final injected output appears once at stop.

- [ ] **Step 5: Engage the user for threshold tuning if needed**

If preview is too eager or too sticky, test these changes one at a time:

- `segment_silence_ms`: `450 -> 350` for faster commits,
- `segment_silence_ms`: `450 -> 600` for fewer premature segments,
- `audio_overlap_ms`: `500 -> 700` for stronger cross-boundary repair,
- `audio_overlap_ms`: `500 -> 300` if duplication increases.

Expected: user confirms a better latency/accuracy balance.

- [ ] **Step 6: Engage the user for mixed-language acceptance**

Ask the user to test:

- `把这个 API route 改成 streaming 版本。`
- `今天 sync 一下 Claude Code 和 FunASR 的行为。`
- `这个 queue 先别清，等 retry 完再说。`

Expected:
- overlap reduces term-splitting mistakes,
- tail rollback corrects local mistakes without rewriting the full prefix.

- [ ] **Step 7: Engage the user for cross-app output safety**

Ask the user to test final output in at least two real targets they use daily:

- Teams / chat input,
- editor / note app.

Expected:
- no live partial typing into the target app,
- no duplicate final injection,
- no lost final segment.

- [ ] **Step 8: Final commit**

```bash
git add app/config.py app/download_models.py app/funasr_server.py app/streaming_session.py app/floating_button.py app/llm_polish.py app/transcribe.py main.py tests/
git commit -m "feat: add local streaming dictation preview"
```
