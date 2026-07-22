# OpenAI Audio Depth Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox syntax.

**Goal:** Ship `/v1/models`, STT `verbose_json` (segments), TTS `speed`, and Piper multi-voice preload.

**Architecture:** Enrich STT return type; extend TTS `synthesize_stream(text, voice, speed)`; preload Piper baked voices; add models route in `app.py`.

**Tech Stack:** FastAPI, faster-whisper, piper-tts, kokoro-onnx, pytest

## Global Constraints

- Em-dash-free copy
- Keep `json`/`text` transcription latency path (`without_timestamps=True`)
- Offline: no HF downloads at request time
- Piper baked set: `en_US-amy-medium`, `en_US-bryce-medium`, `en_US-lessac-medium`

---

### Task 1: STT TranscriptionResult + verbose timestamps

**Files:** Modify `src/voicebox/stt.py`, `tests/test_stt.py`, `tests/test_app.py`, `tests/test_latency.py`

- [ ] Add `TranscriptionSegment` / `TranscriptionResult` dataclasses
- [ ] `transcribe(..., timestamps: bool = False) -> TranscriptionResult`
- [ ] Unit tests for timestamp flag and result `.text`
- [ ] Update callers

### Task 2: TTS speed + Piper multi-voice

**Files:** Modify `src/voicebox/tts.py`, `src/voicebox/tts_piper.py`, tests

- [ ] `synthesize_stream(..., speed: float = 1.0)`
- [ ] Piper preload baked voices; resolve/reject voice; `list_voice_ids()`; `sample_rate_for(voice)`
- [ ] Kokoro pass speed; `list_voice_ids()` returns `[default_voice]`
- [ ] Unit tests with fakes/mocks where models not required

### Task 3: App routes

**Files:** Modify `src/voicebox/app.py`, `tests/test_app.py`, `README.md`

- [ ] `GET /v1/models`
- [ ] `verbose_json` response shape
- [ ] Parse/validate `speed`; pass to TTS
- [ ] Bump FastAPI version string to `0.2.7` (docs only until release bump)
- [ ] README API section
- [ ] `pytest -m "not model and not latency"`
