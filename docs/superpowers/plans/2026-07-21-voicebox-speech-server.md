# voicebox Speech Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build voicebox — a self-hosted, OpenAI-compatible speech server that transcribes audio (`faster-whisper`) and streams synthesized speech (`Kokoro`), runnable CPU-only via Docker Compose.

**Architecture:** A single Python + FastAPI app in one Docker container. Two engine modules (`stt.py`, `tts.py`) each wrap one library behind a plain function interface; `app.py` exposes them as the standard OpenAI audio endpoints (`/v1/audio/transcriptions`, `/v1/audio/speech`) plus `/health`. Models load once at startup and stay warm. TTS splits input into sentences and streams each chunk as it is synthesized.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, faster-whisper (CTranslate2), kokoro-onnx, PyAV (via faster-whisper), espeak-ng, ffmpeg, pytest, Docker Compose.

## Global Constraints

- Python **3.11** (pin in Dockerfile and `pyproject.toml`).
- **CPU-only by default**; no code may hard-require CUDA. Device selectable via `VOICEBOX_DEVICE` env (`cpu` default, `cuda` optional).
- **Vendor-neutral / open-source clean:** no hostnames, IPs, personal infra, or machine names in code, comments, docs, or commit messages. Refer only to "the host".
- **OpenAI audio API compatibility is the stable contract.** Endpoint paths, request field names, and the `{"text": ...}` STT response shape must match OpenAI's audio API exactly.
- STT model default: `Systran/faster-distil-whisper-small.en` (int8). TTS model default: `speaches-ai/Kokoro-82M-v1.0-ONNX`. Default voice: `af_heart`. Default port: `8790`.
- TTS **must stream** (sentence-by-sentence); never buffer a whole paragraph before first byte.
- Supported `response_format` for TTS: `wav` and `pcm`. Unsupported → HTTP 400.
- Input audio duration cap: **120 s** (reject longer → HTTP 400).
- All configuration via environment variables with the `VOICEBOX_` prefix, read in `config.py`.
- TDD: every behavior gets a failing test first. Commit after each green task.

---

### Task 1: Project scaffold, config, and tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/voicebox/__init__.py`
- Create: `src/voicebox/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `.env.example`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `voicebox.config.Settings` — a frozen dataclass with attributes
  `stt_model: str`, `tts_model: str`, `default_voice: str`, `port: int`,
  `device: str`, `cpu_threads: int`, `max_audio_seconds: int`.
  `voicebox.config.load_settings() -> Settings` reads `VOICEBOX_*` env vars with
  the defaults from Global Constraints.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "voicebox"
version = "0.1.0"
description = "Self-hosted OpenAI-compatible speech server (STT + streaming TTS)"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "python-multipart>=0.0.9",
  "faster-whisper>=1.0.3",
  "kokoro-onnx>=0.4.0",
  "soundfile>=0.12",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27", "ruff>=0.6"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Write the failing test** in `tests/test_config.py`

```python
import os
from voicebox.config import load_settings, Settings


def test_defaults_when_no_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VOICEBOX_"):
            monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert isinstance(s, Settings)
    assert s.stt_model == "Systran/faster-distil-whisper-small.en"
    assert s.tts_model == "speaches-ai/Kokoro-82M-v1.0-ONNX"
    assert s.default_voice == "af_heart"
    assert s.port == 8790
    assert s.device == "cpu"
    assert s.max_audio_seconds == 120


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VOICEBOX_PORT", "9001")
    monkeypatch.setenv("VOICEBOX_DEVICE", "cuda")
    monkeypatch.setenv("VOICEBOX_MAX_AUDIO_SECONDS", "60")
    s = load_settings()
    assert s.port == 9001
    assert s.device == "cuda"
    assert s.max_audio_seconds == 60
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicebox.config'`

- [ ] **Step 4: Write `src/voicebox/config.py`**

```python
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    stt_model: str
    tts_model: str
    default_voice: str
    port: int
    device: str
    cpu_threads: int
    max_audio_seconds: int


def load_settings() -> Settings:
    return Settings(
        stt_model=os.getenv("VOICEBOX_STT_MODEL", "Systran/faster-distil-whisper-small.en"),
        tts_model=os.getenv("VOICEBOX_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX"),
        default_voice=os.getenv("VOICEBOX_DEFAULT_VOICE", "af_heart"),
        port=int(os.getenv("VOICEBOX_PORT", "8790")),
        device=os.getenv("VOICEBOX_DEVICE", "cpu"),
        cpu_threads=int(os.getenv("VOICEBOX_CPU_THREADS", "0")),
        max_audio_seconds=int(os.getenv("VOICEBOX_MAX_AUDIO_SECONDS", "120")),
    )
```

Create empty `src/voicebox/__init__.py` and `tests/__init__.py`. Write `.env.example`:

```bash
VOICEBOX_STT_MODEL=Systran/faster-distil-whisper-small.en
VOICEBOX_TTS_MODEL=speaches-ai/Kokoro-82M-v1.0-ONNX
VOICEBOX_DEFAULT_VOICE=af_heart
VOICEBOX_PORT=8790
VOICEBOX_DEVICE=cpu
VOICEBOX_CPU_THREADS=0
VOICEBOX_MAX_AUDIO_SECONDS=120
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/voicebox/__init__.py src/voicebox/config.py tests/__init__.py tests/test_config.py .env.example
git commit -m "feat: project scaffold and env-driven config"
```

---

### Task 2: STT module (faster-whisper wrapper)

**Files:**
- Create: `src/voicebox/stt.py`
- Create: `tests/fixtures/hello.wav` (generated in Step 1)
- Create: `tests/test_stt.py`

**Interfaces:**
- Consumes: `voicebox.config.Settings`.
- Produces:
  - `voicebox.stt.SttEngine(settings: Settings)` — loads the model once in
    `__init__`.
  - `SttEngine.transcribe(audio: bytes) -> str` — decodes any container
    (wav/mp3/webm) via faster-whisper's built-in PyAV decode and returns the
    joined transcript text (stripped). Raises `voicebox.stt.AudioTooLongError`
    (subclass of `ValueError`) if audio exceeds `settings.max_audio_seconds`,
    and `voicebox.stt.AudioDecodeError` (subclass of `ValueError`) if the bytes
    cannot be decoded.

- [ ] **Step 1: Create the test fixture** (a real spoken wav, generated once)

Run this helper to synthesize a known-text wav using the TTS model so the STT
test is self-contained (requires Task-agnostic one-off; commit the wav):

```bash
python - <<'PY'
# One-off fixture generator. Uses kokoro-onnx to produce a known utterance.
import soundfile as sf
from kokoro_onnx import Kokoro
import urllib.request, os
os.makedirs("tests/fixtures", exist_ok=True)
# Model files are pulled by kokoro-onnx on first use into its cache.
k = Kokoro.from_pretrained("speaches-ai/Kokoro-82M-v1.0-ONNX")
samples, sr = k.create("Hello world, this is a voicebox test.", voice="af_heart", speed=1.0)
sf.write("tests/fixtures/hello.wav", samples, sr, subtype="PCM_16")
print("wrote tests/fixtures/hello.wav", sr)
PY
```

If `Kokoro.from_pretrained` is unavailable in the installed version, use the
constructor with explicit model paths per Task 3, Step 4. The exact known text
is: `"Hello world, this is a voicebox test."`

- [ ] **Step 2: Write the failing test** in `tests/test_stt.py`

```python
import pytest
from voicebox.config import load_settings
from voicebox.stt import SttEngine, AudioTooLongError, AudioDecodeError


@pytest.fixture(scope="module")
def engine():
    return SttEngine(load_settings())


def test_transcribes_known_audio(engine):
    with open("tests/fixtures/hello.wav", "rb") as f:
        text = engine.transcribe(f.read())
    lowered = text.lower()
    assert "hello world" in lowered
    assert "voicebox" in lowered


def test_rejects_undecodable_bytes(engine):
    with pytest.raises(AudioDecodeError):
        engine.transcribe(b"not audio at all")


def test_rejects_too_long_audio(engine, monkeypatch):
    monkeypatch.setattr(engine, "max_audio_seconds", 0)
    with open("tests/fixtures/hello.wav", "rb") as f:
        with pytest.raises(AudioTooLongError):
            engine.transcribe(f.read())
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_stt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicebox.stt'`

- [ ] **Step 4: Write `src/voicebox/stt.py`**

```python
from __future__ import annotations
import io
import wave
from faster_whisper import WhisperModel
from voicebox.config import Settings


class AudioDecodeError(ValueError):
    pass


class AudioTooLongError(ValueError):
    pass


def _duration_seconds(audio: bytes) -> float | None:
    """Best-effort duration for WAV; returns None for other containers."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


class SttEngine:
    def __init__(self, settings: Settings) -> None:
        self.max_audio_seconds = settings.max_audio_seconds
        compute_type = "int8" if settings.device == "cpu" else "float16"
        self.model = WhisperModel(
            settings.stt_model,
            device=settings.device,
            compute_type=compute_type,
            cpu_threads=settings.cpu_threads,
        )

    def transcribe(self, audio: bytes) -> str:
        dur = _duration_seconds(audio)
        if dur is not None and dur > self.max_audio_seconds:
            raise AudioTooLongError(f"audio {dur:.1f}s exceeds cap {self.max_audio_seconds}s")
        try:
            segments, _info = self.model.transcribe(io.BytesIO(audio), language="en")
            parts = [seg.text for seg in segments]
        except Exception as exc:  # PyAV/CT2 decode failures
            raise AudioDecodeError(str(exc)) from exc
        text = "".join(parts).strip()
        if not text and dur is None:
            # Non-WAV that produced nothing usable is treated as undecodable.
            raise AudioDecodeError("no audio decoded")
        return text
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_stt.py -v`
Expected: PASS (3 passed). First run downloads the model (~1–2 min).

- [ ] **Step 6: Commit**

```bash
git add src/voicebox/stt.py tests/test_stt.py tests/fixtures/hello.wav
git commit -m "feat: STT module wrapping faster-whisper with duration + decode guards"
```

---

### Task 3: TTS module (streaming Kokoro wrapper)

**Files:**
- Create: `src/voicebox/tts.py`
- Create: `tests/test_tts.py`

**Interfaces:**
- Consumes: `voicebox.config.Settings`.
- Produces:
  - `voicebox.tts.split_sentences(text: str) -> list[str]` — splits on sentence
    boundaries (`. ! ?` followed by whitespace), dropping empties.
  - `voicebox.tts.TtsEngine(settings: Settings)` — loads Kokoro once.
  - `TtsEngine.sample_rate: int` — 24000.
  - `TtsEngine.synthesize_stream(text: str, voice: str | None = None) ->
    Iterator[bytes]` — yields **one int16 little-endian PCM chunk per sentence**
    (24 kHz mono), so callers receive audio for sentence 1 before sentence 2 is
    synthesized. Raises `ValueError` on empty text.

- [ ] **Step 1: Write the failing test** in `tests/test_tts.py`

```python
import pytest
from voicebox.config import load_settings
from voicebox.tts import TtsEngine, split_sentences


def test_split_sentences():
    out = split_sentences("Hello there. How are you? Fine!  ")
    assert out == ["Hello there.", "How are you?", "Fine!"]


def test_split_single_sentence_no_terminator():
    assert split_sentences("just one clause") == ["just one clause"]


@pytest.fixture(scope="module")
def engine():
    return TtsEngine(load_settings())


def test_streams_one_chunk_per_sentence(engine):
    chunks = list(engine.synthesize_stream("Hello there. How are you?"))
    assert len(chunks) == 2
    assert all(isinstance(c, bytes) and len(c) > 0 for c in chunks)
    # int16 PCM => even byte length
    assert all(len(c) % 2 == 0 for c in chunks)


def test_empty_text_raises(engine):
    with pytest.raises(ValueError):
        list(engine.synthesize_stream("   "))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicebox.tts'`

- [ ] **Step 3: Write `src/voicebox/tts.py`**

```python
from __future__ import annotations
import re
from typing import Iterator
import numpy as np
from kokoro_onnx import Kokoro
from voicebox.config import Settings

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def split_sentences(text: str) -> list[str]:
    parts = [m.group().strip() for m in _SENTENCE_RE.finditer(text)]
    return [p for p in parts if p]


class TtsEngine:
    sample_rate = 24000

    def __init__(self, settings: Settings) -> None:
        self.default_voice = settings.default_voice
        # Kokoro model + voices are fetched into the image at build time
        # (see Dockerfile). from_pretrained resolves the cached files.
        self.kokoro = Kokoro.from_pretrained(settings.tts_model)

    def synthesize_stream(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("input text is empty")
        v = voice or self.default_voice
        for sentence in sentences:
            samples, _sr = self.kokoro.create(sentence, voice=v, speed=1.0)
            pcm = np.clip(samples, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype("<i2")
            yield pcm16.tobytes()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts.py -v`
Expected: PASS (4 passed). First run downloads Kokoro model files.

> If `Kokoro.from_pretrained` does not exist in the pinned `kokoro-onnx`
> version, replace the constructor with explicit paths:
> `self.kokoro = Kokoro("/models/kokoro-v1.0.onnx", "/models/voices-v1.0.bin")`
> and have the Dockerfile (Task 5) download those two files to `/models`.

- [ ] **Step 5: Commit**

```bash
git add src/voicebox/tts.py tests/test_tts.py
git commit -m "feat: streaming Kokoro TTS module with per-sentence chunks"
```

---

### Task 4: FastAPI app with OpenAI-compatible endpoints

**Files:**
- Create: `src/voicebox/wav.py`
- Create: `src/voicebox/app.py`
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `SttEngine`, `TtsEngine`, `Settings`, `split_sentences`, and the
  engine exceptions from Tasks 1–3.
- Produces:
  - `voicebox.wav.pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes` —
    wraps raw int16 PCM in a WAV container.
  - `voicebox.wav.wav_header(sample_rate: int) -> bytes` — a streaming WAV
    header with `0xFFFFFFFF` placeholder sizes, emitted before PCM frames when
    `response_format=wav`.
  - `voicebox.app.create_app(stt: SttEngine, tts: TtsEngine, settings: Settings)
    -> FastAPI` — factory returning the wired app (injectable engines for
    testing).
  - `voicebox.app.app` — module-level app built from real engines via
    `load_settings()`, used by uvicorn in the container.
- Routes:
  - `GET /health` → `200 {"status": "ok", "models_loaded": true}`.
  - `POST /v1/audio/transcriptions` (multipart: `file`, optional form `model`,
    `language`, `response_format`) → `200 {"text": "..."}`; `400` on
    decode/too-long errors.
  - `POST /v1/audio/speech` (JSON: `model`, `input`, optional `voice`,
    `response_format` default `wav`) → `200` streaming audio
    (`Content-Type: audio/wav` or `audio/pcm`); `400` on empty input or
    unsupported `response_format`.

- [ ] **Step 1: Write `src/voicebox/wav.py`**

```python
from __future__ import annotations
import struct


def wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    # Placeholder sizes (0xFFFFFFFF) so the header can precede a stream.
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                 byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_len = len(pcm)
    riff_len = 36 + data_len
    return (
        b"RIFF" + struct.pack("<I", riff_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                 byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", data_len) + pcm
    )
```

- [ ] **Step 2: Write the failing test** in `tests/test_app.py`

```python
import io
import wave
import pytest
from fastapi.testclient import TestClient
from voicebox.app import create_app
from voicebox.stt import AudioDecodeError
from voicebox.wav import pcm_to_wav_bytes


class FakeStt:
    def transcribe(self, audio: bytes) -> str:
        if audio == b"bad":
            raise AudioDecodeError("bad")
        return "hello world"


class FakeTts:
    sample_rate = 24000

    def synthesize_stream(self, text, voice=None):
        if not text.strip():
            raise ValueError("empty")
        yield (b"\x01\x00" * 100)
        yield (b"\x02\x00" * 100)


@pytest.fixture
def client():
    from voicebox.config import load_settings
    return TestClient(create_app(FakeStt(), FakeTts(), load_settings()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": True}


def test_transcription_ok(client):
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"anything", "audio/wav")},
                    data={"model": "whatever"})
    assert r.status_code == 200
    assert r.json() == {"text": "hello world"}


def test_transcription_bad_audio_400(client):
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"bad", "audio/wav")})
    assert r.status_code == 400


def test_speech_wav_streams_valid_container(client):
    r = client.post("/v1/audio/speech",
                    json={"model": "m", "input": "hi there", "response_format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    # Body should be a parseable WAV once fully read (via helper for full-buffer).
    body = r.content
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"


def test_speech_empty_input_400(client):
    r = client.post("/v1/audio/speech", json={"model": "m", "input": "   "})
    assert r.status_code == 400


def test_speech_bad_format_400(client):
    r = client.post("/v1/audio/speech",
                    json={"model": "m", "input": "hi", "response_format": "ogg"})
    assert r.status_code == 400
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicebox.app'`

- [ ] **Step 4: Write `src/voicebox/app.py`**

```python
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from voicebox.config import Settings, load_settings
from voicebox.wav import wav_header

_SUPPORTED_FORMATS = {"wav", "pcm"}


def create_app(stt, tts, settings: Settings) -> FastAPI:
    app = FastAPI(title="voicebox", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": True}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default=settings.stt_model),
        language: str = Form(default="en"),
        response_format: str = Form(default="json"),
    ):
        audio = await file.read()
        try:
            text = stt.transcribe(audio)
        except ValueError as exc:  # AudioDecodeError / AudioTooLongError
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse({"text": text})

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        body = await request.json()
        text = (body.get("input") or "")
        response_format = (body.get("response_format") or "wav").lower()
        voice = body.get("voice")
        if response_format not in _SUPPORTED_FORMATS:
            raise HTTPException(status_code=400,
                                detail=f"unsupported response_format: {response_format}")
        if not text.strip():
            raise HTTPException(status_code=400, detail="input is empty")

        def gen():
            if response_format == "wav":
                yield wav_header(tts.sample_rate)
            for chunk in tts.synthesize_stream(text, voice):
                yield chunk

        media = "audio/wav" if response_format == "wav" else "audio/pcm"
        return StreamingResponse(gen(), media_type=media)

    return app


app = create_app(
    __import__("voicebox.stt", fromlist=["SttEngine"]).SttEngine(load_settings()),
    __import__("voicebox.tts", fromlist=["TtsEngine"]).TtsEngine(load_settings()),
    load_settings(),
) if __name__ != "__main__" else None
```

> Note: the module-level `app` builds real engines (loads models) at import
> time, which is what uvicorn needs in the container. Tests never import it —
> they call `create_app` with fakes. To avoid loading models during unit-test
> collection, the module-level construction is guarded so importing
> `voicebox.app` in tests does not trigger it. Replace the guarded block with an
> explicit factory call in Task 5's entrypoint instead if import-time loading is
> undesirable; see Step 4a.

- [ ] **Step 4a: Replace the fragile module-level guard with a clean entrypoint**

Delete the trailing `app = create_app(...) if __name__ ...` block from
`app.py`. Instead add `src/voicebox/__main__.py`:

```python
from __future__ import annotations
import uvicorn
from voicebox.config import load_settings
from voicebox.stt import SttEngine
from voicebox.tts import TtsEngine
from voicebox.app import create_app


def main() -> None:
    settings = load_settings()
    application = create_app(SttEngine(settings), TtsEngine(settings), settings)
    uvicorn.run(application, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
```

This keeps `voicebox.app` import-safe (no model load on import) and gives the
container a single command: `python -m voicebox`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: all tests from Tasks 1–4 pass.

- [ ] **Step 7: Commit**

```bash
git add src/voicebox/wav.py src/voicebox/app.py src/voicebox/__main__.py tests/test_app.py
git commit -m "feat: FastAPI OpenAI-compatible audio endpoints with streaming TTS"
```

---

### Task 5: Dockerfile, compose, and containerized integration test

**Files:**
- Create: `Dockerfile`
- Create: `compose.yaml`
- Create: `scripts/fetch_models.py`
- Create: `scripts/smoke.sh`

**Interfaces:**
- Consumes: `python -m voicebox` entrypoint from Task 4.
- Produces: a runnable image exposing port `8790`; `scripts/smoke.sh` curls a
  running instance and asserts both endpoints work.

- [ ] **Step 1: Write `scripts/fetch_models.py`** (build-time model download)

```python
"""Pre-download STT + TTS models into the image so runtime is offline."""
from voicebox.config import load_settings
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

s = load_settings()
WhisperModel(s.stt_model, device="cpu", compute_type="int8")
Kokoro.from_pretrained(s.tts_model)
print("models cached")
```

- [ ] **Step 2: Write the `Dockerfile`**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Bake models into the image at build time (reproducible, offline runtime).
COPY scripts/fetch_models.py ./scripts/fetch_models.py
RUN python scripts/fetch_models.py

ENV VOICEBOX_PORT=8790
EXPOSE 8790
CMD ["python", "-m", "voicebox"]
```

- [ ] **Step 3: Write `compose.yaml`**

```yaml
services:
  voicebox:
    build: .
    image: voicebox:latest
    ports:
      - "8790:8790"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "6"
          memory: 4g
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8790/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

- [ ] **Step 4: Write `scripts/smoke.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://localhost:8790}"

echo "health:"; curl -fsS "$BASE/health"; echo

echo "tts -> /tmp/vb_smoke.wav"
curl -fsS "$BASE/v1/audio/speech" -H "Content-Type: application/json" \
  --output /tmp/vb_smoke.wav \
  -d '{"model":"m","input":"Hello from voicebox. Streaming works.","response_format":"wav"}'
python3 -c "import wave; w=wave.open('/tmp/vb_smoke.wav','rb'); print('wav ok', w.getnframes()/w.getframerate(),'s')"

echo "stt round-trip:"
curl -fsS "$BASE/v1/audio/transcriptions" -F "file=@/tmp/vb_smoke.wav" | tee /tmp/vb_smoke.json
python3 -c "import json;t=json.load(open('/tmp/vb_smoke.json'))['text'].lower();assert 'voicebox' in t or 'hello' in t, t;print('stt ok')"
echo "SMOKE PASS"
```

- [ ] **Step 5: Build the image**

Run: `docker compose build`
Expected: build succeeds; the `fetch_models.py` layer prints `models cached`.

- [ ] **Step 6: Start and smoke-test**

Run: `docker compose up -d && sleep 20 && chmod +x scripts/smoke.sh && ./scripts/smoke.sh`
Expected: `health` returns ok JSON, TTS writes a valid multi-second WAV, STT
round-trips it, final line `SMOKE PASS`.

- [ ] **Step 7: Tear down**

Run: `docker compose down`

- [ ] **Step 8: Commit**

```bash
git add Dockerfile compose.yaml scripts/fetch_models.py scripts/smoke.sh
git commit -m "feat: Dockerfile, compose, build-time model fetch, and smoke test"
```

---

### Task 6: Latency regression guard + usage docs

**Files:**
- Create: `tests/test_latency.py`
- Modify: `README.md` (add a "Run it" usage section)

**Interfaces:**
- Consumes: `SttEngine`, `TtsEngine`, the fixture wav from Task 2.
- Produces: a latency test that asserts the spec's thresholds on CPU reference
  hardware, skippable via `VOICEBOX_SKIP_LATENCY=1` for CI on unknown hardware.

- [ ] **Step 1: Write `tests/test_latency.py`**

```python
import os
import time
import wave
import pytest
from voicebox.config import load_settings
from voicebox.stt import SttEngine
from voicebox.tts import TtsEngine

pytestmark = pytest.mark.skipif(
    os.getenv("VOICEBOX_SKIP_LATENCY") == "1",
    reason="latency guard skipped on this host",
)


def _wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def test_stt_faster_than_half_realtime():
    engine = SttEngine(load_settings())
    dur = _wav_seconds("tests/fixtures/hello.wav")
    with open("tests/fixtures/hello.wav", "rb") as f:
        data = f.read()
    engine.transcribe(data)  # warm
    t0 = time.perf_counter()
    engine.transcribe(data)
    rtf = (time.perf_counter() - t0) / dur
    assert rtf < 0.5, f"STT RTF {rtf:.2f} exceeds 0.5x"


def test_tts_first_chunk_under_2s():
    engine = TtsEngine(load_settings())
    gen = engine.synthesize_stream("This is the first sentence. And a second one.")
    t0 = time.perf_counter()
    next(gen)  # first sentence chunk
    first = time.perf_counter() - t0
    assert first < 2.0, f"TTS first chunk {first:.2f}s exceeds 2s"
```

- [ ] **Step 2: Run the latency guard**

Run: `pytest tests/test_latency.py -v`
Expected: PASS on CPU reference hardware (both under threshold).

- [ ] **Step 3: Add a "Run it" section to `README.md`**

Append:

````markdown
## Run it

```bash
docker compose up -d --build      # first build bakes the models in (~a few min)
curl -fsS localhost:8790/health

# text -> speech (streamed WAV)
curl -fsS localhost:8790/v1/audio/speech -H "Content-Type: application/json" \
  --output out.wav \
  -d '{"model":"tts","input":"Hello from voicebox.","voice":"af_heart","response_format":"wav"}'

# speech -> text
curl -fsS localhost:8790/v1/audio/transcriptions -F "file=@out.wav"
```

Point any OpenAI-audio-compatible client at `http://<host>:8790/v1`.
Configuration is via `VOICEBOX_*` env vars — see `.env.example`.
````

- [ ] **Step 4: Commit**

```bash
git add tests/test_latency.py README.md
git commit -m "test: latency regression guard; docs: usage section"
```

---

## Self-Review

**Spec coverage:**
- OpenAI-compatible STT/TTS/health endpoints → Task 4. ✓
- faster-whisper distil-small.en int8 → Task 2 + config Task 1. ✓
- kokoro-onnx streaming, sentence-split → Task 3. ✓
- Streaming TTS (hard requirement) → Task 3 interface + Task 4 `StreamingResponse`. ✓
- Error handling (400 bad audio/empty/format, duration cap) → Tasks 2 & 4. ✓
- Docker Compose, build-time model bake, CPU/mem caps → Task 5. ✓
- Unit + integration + latency tests → Tasks 1–4, 5 (smoke), 6 (latency). ✓
- Vendor-neutral / env-driven config → Task 1 + Global Constraints. ✓
- wav + pcm formats → Task 4. ✓

**Placeholder scan:** No TBD/TODO; every code step has full code. The two
"if the pinned version differs" notes (Tasks 2 & 3) are explicit fallbacks with
exact replacement code, not placeholders.

**Type consistency:** `Settings` fields, `SttEngine.transcribe(bytes)->str`,
`TtsEngine.synthesize_stream(str, voice)->Iterator[bytes]`,
`TtsEngine.sample_rate`, `split_sentences`, `wav_header`/`pcm_to_wav_bytes`, and
the `AudioDecodeError`/`AudioTooLongError` (both `ValueError` subclasses, caught
as `ValueError` in the route) are consistent across tasks.
