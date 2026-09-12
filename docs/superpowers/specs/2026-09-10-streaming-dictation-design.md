# Streaming Dictation & Transcription Design

## Context

Bibi currently uses a batch pipeline:

1. capture one full recording session,
2. stop recording,
3. run one blocking ASR pass,
4. optionally run one blocking LLM polish pass,
5. inject the final text once.

That flow is safe but feels slow compared with WeChat Input / Typeless style dictation. The goal of this change is to improve perceived responsiveness while preserving the current safe output model:

- **recording still starts/stops with the existing floating capsule / hotkey flow**;
- **final text is still injected only once at session end**;
- **during recording, the floating capsule shows continuously improving preview text**;
- **the first version stays fully local for ASR** and must not depend on downloading new ModelScope assets during development inside the corporate network.

The approved direction for v1 is:

- **ASR:** local pseudo-streaming via online VAD segmentation + existing offline FunASR ONNX models for fast small-segment re-transcription;
- **preview:** update only inside the floating capsule, not in the target application;
- **LLM:** run after an ASR segment is considered stable, and stream its incremental output into the preview;
- **session model:** keep the current single recording session, with real-time preview added on the side.

## Constraints

1. Do **not** depend on `modelscope.cn` network access during implementation or testing.
2. Do **not** require downloading a new online Paraformer model for v1.
3. Do **not** inject partial text into the focused application.
4. Do **not** rewrite already committed preview prefix aggressively.
5. Keep the single-Tk-root rule: all UI updates must go through `UIHub.dispatch()`.
6. Preserve the existing final batch result path as the authoritative fallback.

## Existing Reusable Building Blocks

- `app/audio_capture.py`
  - `AudioCapture.queue` already emits microphone frames continuously.
- `app/transcribe.py`
  - `TranscriptionWorker._capture_loop()` already sees audio frame-by-frame.
  - existing stop/final-transcribe path remains the authoritative end-of-session fallback.
- `app/funasr_server.py`
  - current local offline ASR entry point for small-segment recognition.
- `app/llm_polish.py`
  - current batch `polish()` logic and protocol configuration remain the fallback path.
- `app/floating_button.py`
  - existing floating capsule is already persistent and already uses `UIHub.dispatch()` safely.
- `app/ui_hub.py`
  - thread-safe bridge for all Tk updates.
- `app/output.py`
  - unchanged final one-shot text injection.

## Recommended Architecture

### 1. Keep the current final batch pipeline

The existing session-end path remains intact:

- audio capture buffer still accumulates the full session,
- stop still triggers the existing final batch transcription,
- final text injection still happens once.

This provides the main safety net if preview logic, VAD, or streaming LLM behavior is noisy.

### 2. Add a side pipeline for preview

Introduce a new module:

- `app/streaming_session.py`

This module owns the real-time preview workflow and is independent from final output injection.

Suggested internal responsibilities:

- `StreamingSession`
  - owns state for one recording session;
  - receives audio chunks during recording;
  - coordinates segmentation, segment ASR, LLM preview updates, and final flush.
- `StreamingSegmenter`
  - converts incoming chunks into candidate speech segments;
  - uses online VAD if available, otherwise conservative silence/energy heuristics.
- `PreviewAssembler`
  - manages `committed_prefix`, `editable_tail`, and the visible preview text;
  - handles rollback-safe replacement of the tail.
- `SegmentTask`
  - immutable representation of a stable candidate segment to send to ASR.

A single file is enough for v1 as long as these responsibilities remain clearly separated.

### 3. Data flow

During recording:

1. `AudioCapture` emits chunk arrays.
2. `TranscriptionWorker._capture_loop()` continues appending them to the full-session buffer.
3. The same loop also forwards each chunk into `StreamingSession.push_chunk(...)`.
4. `StreamingSegmenter` accumulates chunks and decides when a segment is stable enough to submit.
5. Each stable segment becomes a small local ASR task using the **existing offline model**.
6. Segment ASR output is merged into preview text via `PreviewAssembler`.
7. If LLM is enabled, the stable text tail is streamed through the LLM and the capsule preview is updated incrementally.
8. On session stop, `StreamingSession.finish()` flushes any remaining speech, waits for in-flight preview tasks, and hands control back to the existing final batch path.

## VAD / LLM Cooperation Strategy

VAD boundaries are not trusted as true sentence boundaries. The design therefore uses a **stable prefix + rollback-safe tail** model.

### Preview states

The preview is split into:

- `committed_prefix`
  - stable text that should almost never change.
- `editable_tail`
  - the newest one or two segments that may still be corrected.
- `live_preview`
  - the visible string shown in the floating capsule.

The preview shown to the user is always:

`committed_prefix + editable_tail + optional processing hint`

### VAD detection strategy (confirmed)

`funasr_onnx.vad_bin.Fsmn_vad_online` was evaluated and rejected: it returns
boundary *events* (empty list on most in-utterance chunks), not a per-chunk
"is this speech" flag, so a naive `bool(segments)` read would misclassify
ongoing speech as silence and over-segment. v1 instead uses a **simple
RMS/energy threshold detector**, reusing the existing (currently unused)
`vad.start_threshold` / `vad.stop_threshold` / `vad.min_speech_ms` /
`vad.min_silence_ms` / `vad.pad_ms` config keys:

- compute RMS of each incoming chunk,
- chunk counts as speech once RMS crosses `start_threshold` for at least
  `min_speech_ms`,
- speech ends once RMS stays below `stop_threshold` for at least
  `min_silence_ms`,
- `pad_ms` of audio on each side is kept so segment edges aren't clipped.

This avoids a new model dependency and sidesteps the online-VAD semantics
mismatch entirely. Accuracy is coarser than a neural VAD; if segmentation
feels too eager/sticky in manual testing, thresholds are tunable via config
(see Task 8 tuning checklist).

### Audio overlap

Each new ASR segment should include:

- the current detected speech segment,
- plus **300-800ms** of trailing audio overlap from the previous segment.

This reduces cross-boundary word breakage and gives the offline ASR enough acoustic context to repair cuts like:

- split English tokens,
- Chinese multi-character words split by pause detection,
- phrase continuation across short pauses.

### Tail rewrite instead of blind append

When a new ASR segment result arrives:

1. compare it against the current `editable_tail`,
2. find the largest reasonable suffix/prefix overlap,
3. replace only the tail region,
4. keep `committed_prefix` unchanged.

This allows local corrections without making the whole preview flicker.

### LLM only edits the tail

LLM input for a stable segment should include:

- last **20-40 chars** of `committed_prefix` as context,
- current `editable_tail`,
- an instruction that the prefix is immutable reference context and only the tail may be polished.

The LLM must not re-author the whole transcript preview.

Confirmed via a live probe against the configured local proxy
(`127.0.0.1:4141`, anthropic schema): SSE streaming works
(`content-type: text/event-stream`, real `content_block_delta` events), but
the backing model emits a `thinking` content block before the `text` block.
`polish_tail_stream` requests set `"thinking": {"type": "disabled"}` on
anthropic-schema calls to skip that block where the backend honors it; the
dominant latency is the ~1.7s time-to-first-byte of the underlying model
call itself, not the thinking block, so preview must keep showing ASR-only
text until the LLM update arrives rather than blocking on it.

### Commit policy

Part of `editable_tail` is promoted into `committed_prefix` when one of these conditions holds:

- a later stable segment arrives, making the earlier tail older and safer;
- silence remains longer than the configured threshold;
- session stop triggers final flush.

## Error Mitigation and Fallbacks

### 1. Segment-level ASR errors

Risk:
- short segments lack context;
- VAD cuts at awkward boundaries;
- mixed Chinese/English terminology is unstable.

Mitigation:
- audio overlap between segments;
- keep only the newest tail editable;
- final session-end batch ASR remains authoritative;
- preserve dictionary replacement after ASR for domain term correction.

### 2. Segment-level LLM errors

Risk:
- unstable small tail gets over-corrected;
- jitter from repeated partial rephrasing;
- network/timeout may stall polish.

Mitigation:
- start LLM only for stable segments, never every raw chunk;
- provide prefix context but forbid prefix rewriting;
- if streaming LLM fails, continue preview with ASR text immediately;
- final end-of-session polish may still run once over the final transcript.

### 3. UI jitter

Risk:
- preview changes too frequently;
- long text overflows the capsule;
- Tk thread violations.

Mitigation:
- coalesce UI updates on a short debounce window;
- show only a preview window of the latest content when the text becomes long;
- expose `FloatingButton.show_preview(committed, tail, state)` and route all updates through `UIHub.dispatch()`.

### 4. Concurrency hazards

Risk:
- capture thread, ASR worker, and LLM stream all update preview state concurrently;
- segment-level streaming ASR calls and the final session-end batch ASR call
  both hit the same `FunASRServer` instance (`self.model` / `self.vad_model` /
  `self.punc_model` are shared, unlocked singletons), so concurrent calls are
  a real hazard, not just a theoretical one.

Mitigation:
- `StreamingSession` owns all mutable preview state behind a lock;
- per-segment sequence IDs discard stale ASR/LLM completions;
- stop/finalize marks the session closed before final flush;
- **ASR concurrency is configurable** via `streaming.dedicated_model_instance`:
  - `false` (default): a single shared `FunASRServer` instance is reused, and
    calls into it (segment preview + final batch) are serialized behind a lock;
  - `true`: a second, fully independent `FunASRServer` instance is loaded
    solely for streaming segment preview, at the cost of extra memory/load
    time, so preview and final-batch inference never contend for the same
    model state.

## UI Behavior

### Expanded capsule (confirmed)

The current capsule is a fixed 176x54 (expanded) / 68x16 (collapsed) block
with a single short status label — there is no room for growing preview
text. Rather than force preview text into that fixed area:

- while recording, the capsule **resizes wider/taller** to make room for a
  single preview line;
- that single line shows **only the newest content**, scrolling/truncating
  from the left (`render_preview_text` truncates with a leading `…` and
  keeps the tail intact) so it always reads as "what I'm saying right now";
- once recording stops and preview is cleared, the capsule shrinks back to
  its normal idle size.

Collapsed capsule behavior is unaffected: collapsed state never shows
preview text, matching existing behavior.

### Collapsed capsule

Collapsed state remains compact and primarily communicates recording / processing state. It does not need to show full preview text.

### Stop behavior

On stop:

1. recording UI stops immediately,
2. preview enters finalizing/processing state,
3. any remaining preview tasks flush,
4. existing final batch transcription path produces final output,
5. one-shot injection occurs,
6. capsule resets to idle.

## Configuration

Add a `streaming` section to `app/config.py` with conservative defaults:

```json
"streaming": {
  "enabled": true,
  "segment_silence_ms": 450,
  "audio_overlap_ms": 500,
  "preview_context_chars": 30,
  "preview_max_chars": 120,
  "ui_update_debounce_ms": 80,
  "dedicated_model_instance": false
}
```

V1 should keep these as internal/advanced config only. No settings dialog work is required initially.

## File-Level Change Plan

### Modify
- `app/transcribe.py`
  - tap per-chunk audio flow and manage a `StreamingSession` lifecycle alongside the existing session buffer.
- `app/floating_button.py`
  - add preview rendering API and visual state for live text.
- `app/llm_polish.py`
  - add a streaming polish API for OpenAI/Anthropic style SSE responses while retaining the current blocking `polish()` fallback.
- `main.py`
  - wire preview callbacks from the worker to the floating capsule and keep final output behavior unchanged.
- `app/config.py`
  - add streaming configuration defaults.

### Add
- `app/streaming_session.py`
  - segmentation, overlap handling, preview merge policy, sequence guarding, and callbacks.
- `tests/test_streaming_session.py`
  - unit tests for segment merging and rollback logic.
- `tests/test_llm_streaming.py`
  - unit tests for streaming parse / fallback behavior.
- `tests/test_floating_preview.py`
  - unit tests for preview rendering state transitions.

## Non-Goals for V1

- real-time text injection into the target application;
- requiring a new online Paraformer model;
- replacing the final batch ASR path;
- fully semantic sentence segmentation;
- letting the LLM rewrite the whole preview history;
- adding a broad new settings surface before core behavior is stable.

## Test Strategy

### Automated unit tests

#### A. `StreamingSession` / segment assembly

1. **silence creates a stable segment**
   - feed voiced chunks followed by silence;
   - assert one ASR task is emitted.

2. **short silence does not over-segment**
   - feed voiced chunks with brief pause below threshold;
   - assert no premature segment submission.

3. **audio overlap is included**
   - emit two adjacent segments;
   - assert second segment contains configured overlap samples.

4. **tail rewrite replaces overlap instead of duplicating text**
   - current tail: `我们明天开`
   - next ASR result: `明天开会讨论 streaming`
   - assert merged preview does not duplicate `明天开`.

5. **stale segment result is discarded**
   - deliver result for older sequence after newer sequence already committed;
   - assert preview state does not regress.

6. **final flush emits remaining tail**
   - stop session with speech still buffered;
   - assert remaining pending segment is processed.

#### B. `LLMPolisher` streaming mode

1. **openai SSE tokens append incrementally**
   - feed mocked SSE chunks;
   - assert callback sees cumulative partial text in order.

2. **anthropic SSE text deltas append incrementally**
   - feed mocked SSE event stream;
   - assert callback receives partial polished tail.

3. **stream failure falls back cleanly**
   - simulate timeout/transport error mid-stream;
   - assert preview falls back to ASR text and session remains alive.

4. **prefix is not rewritten**
   - provide immutable prefix context + editable tail;
   - assert post-merge only the tail region changes.

#### C. `FloatingButton` preview UI

1. **preview updates route through dispatch**
   - assert background-triggered preview updates use the dispatch helper.

2. **recording → preview → processing → idle transitions**
   - verify text/state transitions in order.

3. **collapsed mode ignores long preview rendering**
   - assert collapsed view remains compact.

#### D. `TranscriptionWorker` integration seam

1. **chunk fan-out does not break final buffer**
   - feed chunks;
   - assert full session buffer still combines correctly.

2. **stop waits for preview flush before final reset**
   - assert stop/finalize order is deterministic.

### Automated integration tests

1. **ASR-only preview session**
   - mocked chunk stream → mocked segment ASR → final preview string.

2. **ASR + LLM preview session**
   - mocked ASR stable segments + mocked LLM token stream → final preview string.

3. **session-end fallback still works**
   - simulate preview instability/failure;
   - assert final batch result still becomes injected text.

## Manual test cases

### Manual tests I can drive locally

1. **capsule preview updates while recording**
   - start recording,
   - verify expanded capsule updates after a natural pause,
   - stop and verify final output still injects once.

2. **LLM disabled path**
   - verify preview uses ASR text only and no LLM processing delay appears.

3. **LLM enabled path**
   - verify stable segment enters polishing state and preview continues updating.

4. **collapse / expand / drag still work during preview**
   - verify the existing floating interactions remain stable under live updates.

### Manual tests that require user participation

These depend on real microphone quality, your corp machine environment, your speech style, and target apps. I should explicitly engage you for these.

1. **mixed Chinese + English technical dictation**
   - sample phrases with code terms, acronyms, and product names;
   - verify overlap + tail rewrite reduce cross-boundary mistakes.

2. **fast speech with short pauses**
   - verify segmentation is not too eager.

3. **longer hesitation / self-correction**
   - say phrases like `今天下午开会，呃不对，明天下午开会`;
   - verify the tail corrects locally instead of corrupting the full prefix.

4. **cross-app final injection safety**
   - test final output in Teams / chat / editor surfaces you actually use;
   - confirm no duplicate injection and no partial live typing into the target app.

5. **perceived latency acceptance**
   - confirm whether pause-to-preview latency feels acceptable on your machine.

## Explicit User Engagement Plan

When the implementation reaches a runnable point, I should ask you to help with these checkpoints:

1. **first real microphone validation**
   - after automated tests pass, before tuning thresholds.
2. **threshold tuning session**
   - if segmentation is too eager / too sticky, we tune `segment_silence_ms` and `audio_overlap_ms` together.
3. **mixed-language accuracy sweep**
   - collect 5-10 real utterances from your daily workflow.
4. **final UX acceptance**
   - confirm whether this is close enough to WeChat Input / Typeless feel for v1, or whether we need to escalate to a true online ASR model later.
