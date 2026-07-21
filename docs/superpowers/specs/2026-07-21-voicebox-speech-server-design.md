# voicebox — Local OpenAI-compatible speech server

**Status:** Approved design (Piece A of 3)
**Date:** 2026-07-21

## Goal

Enable full voice interaction with local AI — speech-to-text in and
text-to-speech out — across any consumer: local LLM chats, browser UIs, custom
agent harnesses, and command-line tools. This document specifies **Piece A only:
the speech server**, the foundation every client depends on.

voicebox is a **single, universal, shared audio backend** — not a point solution
for one app. Any consumer that can speak the OpenAI audio API gets voice for
free:

- **Browser chat UIs** (e.g. Open WebUI) — native STT/TTS settings accept an
  OpenAI-compatible base URL; point them at voicebox, no glue code.
- **Custom agent harnesses** — call the endpoints directly.
- **CLIs / coding agents** — thin per-CLI clients.

The larger effort is decomposed into layered pieces, each with its own
spec → plan → build cycle. Pieces B and C are the **first two of an open-ended
set of consumers**, not the scope ceiling:

- **Piece A — voicebox (this spec):** the shared, self-hosted,
  OpenAI-compatible speech server wrapping `faster-whisper` (STT) and `Kokoro`
  (streaming TTS).
- **Piece B — voice chat client:** a turn-taking loop:
  mic → STT → local LLM → streaming TTS → speakers.
- **Piece C — coding-agent voice:** hooks for spoken replies + push-to-talk
  dictation (both directions).
- **Other consumers (no separate spec needed):** browser UIs and custom
  harnesses plug into the same server directly.

## Context & decisions

- **Host: any Linux host with Docker.** voicebox is designed to run well
  **CPU-only** on modest hardware (a mini PC or an old quad-core), and to scale
  up transparently on a host with an NVIDIA GPU. Nothing in the design assumes a
  particular machine.
- **CPU-first, GPU-optional.** `faster-whisper` (CTranslate2) runs on CPU or
  NVIDIA CUDA; the default target is CPU so the server runs anywhere. On a CUDA
  host it can use the GPU with no API change.
- **Deploy: plain Docker Compose**, standalone. Keeps a latency-sensitive
  service simple to run and isolate, and makes CPU/RAM capping trivial so it
  never starves other workloads on the host.
- **English only (default).** `Systran/faster-distil-whisper-small.en` for STT,
  a Kokoro English voice for TTS. Model IDs are swappable via config; a
  multilingual model can be substituted at the cost of CPU latency.
- **DIY, not a wrapper project.** Rather than depend on a third-party
  OpenAI-compatible speech wrapper (several are low-activity), voicebox wraps the
  well-maintained engines (`faster-whisper`, `Kokoro`) directly in a thin
  FastAPI shell. Because it exposes exactly the OpenAI audio API, any backend
  (LocalAI, openedai-speech, a future rewrite) is swappable behind the same
  contract.

## Reference benchmark (CPU-only)

Measured on representative low-power CPU hardware — a 2015-era quad-core
(Intel i7-6700T class: 4C/8T, AVX2, no AVX-512), CPU-only, host near-idle. These
numbers are a **lower bound**: a newer CPU or a GPU host will be faster.

| Stage | Model | Result | Notes |
|---|---|---|---|
| STT | `distil-whisper-small.en` (int8) | 15.2 s audio → ~4.0 s = **RTF 0.26×** | ~4× faster than real-time; a 5–8 s utterance ≈ 1.5–2 s |
| TTS | `Kokoro-82M` | 15.2 s audio → ~7.0 s = **RTF 0.46×** | ~2× faster than real-time; **must stream** for low first-audio latency |
| RAM | both loaded | **~2.2 GB** | Fits comfortably on a small host |

Transcript quality was near-perfect. Verdict: a **usable, natural turn-taking**
voice experience is achievable even on modest CPU-only hardware. Sub-second
OpenAI-Realtime-style barge-in is **not** a goal here, and end-to-end latency in
a full voice loop is dominated by LLM generation time, not the speech layer.

## Architecture

Single Python + FastAPI container, exposing exactly the OpenAI audio API shape
so it is a drop-in, swappable backend for any client.

```
client ──HTTP──▶  voicebox (host :8790)
                    ├── POST /v1/audio/transcriptions  → faster-whisper (distil-small.en, int8)
                    ├── POST /v1/audio/speech          → Kokoro (streaming)
                    └── GET  /health
```

Models load once at startup and stay warm in memory (~2.2 GB).

## Components (each independently testable)

- **`stt.py`** — wraps `faster-whisper`; `transcribe(audio_bytes) -> text`.
  PyAV decodes arbitrary input formats (wav / mp3 / webm-opus), so clients need
  not pre-convert.
- **`tts.py`** — wraps `kokoro-onnx` (ONNX runtime — the right choice for CPU
  over the PyTorch build); `synthesize_stream(text, voice) -> iterator[pcm]`.
  **Splits input into sentences and streams each as it is ready** (hard
  requirement, per benchmark: first audio ~1 s instead of ~7 s).
- **`app.py`** — FastAPI routes, request validation, streaming-response wiring.
- **`config.py`** — env-driven: model IDs, default voice, port, CT2/OMP thread
  count, optional CUDA device.
- **`Dockerfile` + `compose.yaml`** — CPU-only by default (GPU variant optional);
  installs `espeak-ng` (Kokoro phonemizer) + `ffmpeg`; Whisper + Kokoro model
  files fetched at **build time** into the image (reproducible, offline after
  build); configurable CPU/mem caps (default `cpus=6`, `mem=4g`) so it never
  starves co-located workloads.

## Data flow

- **STT:** `POST` audio → decode → faster-whisper → `{"text": "..."}`.
- **TTS:** `POST {input, voice, response_format}` → sentence-split → synth
  sentence 1 → stream PCM/WAV → sentence 2 → … Client plays as bytes arrive.

## Error handling

- Bad/undecodable audio → `400`. Empty `input` → `400`.
- Unsupported `response_format` → `400`. Supported: **wav + pcm** (mp3 optional
  later via ffmpeg).
- Model not loaded / OOM → `503`.
- Input audio duration capped (default 120 s) to prevent runaway CPU.

## Testing

- **Unit:** STT returns expected text for a committed fixture wav; TTS yields
  multiple PCM chunks at 24 kHz sample rate.
- **Integration:** `docker compose up` → curl both endpoints → assert.
- **Latency regression guard:** assert STT RTF < 0.5× and TTS first-chunk < 2 s
  on CPU-only reference hardware.

## Interface contract (stable; internals may change)

Standard OpenAI audio endpoints — anything speaking this contract can replace
voicebox:

- `POST /v1/audio/transcriptions` — multipart `file`, `model`, optional
  `language`, `response_format` → `{"text": "..."}`.
- `POST /v1/audio/speech` — `{model, input, voice, response_format, speed?}` →
  streamed audio bytes.
- `GET /health` → `{status, models_loaded}`.

## Defaults (changeable)

- Name `voicebox`; port `8790`.
- Access by `host:8790` initially; an optional reverse-proxy / hostname can be
  added later.
- Kokoro default voice `af_heart`.

## Out of scope (future pieces / follow-ups)

- Piece B (voice chat client) and Piece C (coding-agent voice) — separate specs.
- Multilingual STT + TTS (would need a larger, slower Whisper and a non-English
  voice model; re-benchmark first).
- A dedicated GPU build/profile (the CPU build already runs on GPU hosts; a
  tuned GPU image is a later optimization).
- mp3 output and browser-UI integration guides.
