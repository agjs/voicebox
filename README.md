# voicebox

A small, self-hosted, **OpenAI-compatible speech server** — speech-to-text
(`faster-whisper`) and streaming text-to-speech (`Kokoro`) — for full voice
interaction with local AI. It's a single, universal audio backend: any client
that speaks the OpenAI audio API (browser chat UIs, custom agent harnesses,
CLIs, coding agents) gets voice with no glue code.

Vendor-neutral by design. Runs **CPU-only on modest hardware** (a mini PC or an
old quad-core is enough) and scales up transparently on a host with an NVIDIA
GPU.

## Status

Piece A of a 3-part effort. Design approved — see
[`docs/superpowers/specs/2026-07-21-voicebox-speech-server-design.md`](docs/superpowers/specs/2026-07-21-voicebox-speech-server-design.md).

- **Piece A — voicebox (this repo):** BUILT and working. Server runs via `docker compose up` or `python -m voicebox`; unit, container smoke, and latency tests pass.
- **Piece B — voice chat client:** mic → STT → local LLM → streaming TTS → speakers. (upcoming)
- **Piece C — coding-agent voice:** spoken replies + push-to-talk dictation. (upcoming)

Pieces B and C are the first two of an open-ended set of consumers; browser UIs
and custom harnesses plug into the same server directly.

## Endpoints

- `POST /v1/audio/transcriptions` — multipart audio → `{"text": ...}`
- `POST /v1/audio/speech` — `{input, voice, ...}` → streamed audio
- `GET /health`

## Quick start

Start the server:

```bash
docker compose up -d --build      # first build bakes the models in (~a few min)
curl -fsS localhost:8790/health
```

Point any OpenAI-audio-compatible client at `http://<host>:8790/v1`. Configuration is via `VOICEBOX_*` env vars — see `.env.example`.

### Speech synthesis (TTS)

```bash
curl http://localhost:8790/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"hello world","response_format":"wav"}' \
  --output out.wav

# Play
ffplay out.wav
```

**Streaming WAV note:** The WAV response is streamed with placeholder size fields
in the RIFF and data chunks (standard for streaming; the actual size is unknown
before streaming completes). It plays correctly in ffmpeg-based players
(ffplay, afplay, mpv, browsers) and round-trips correctly through the
transcription endpoint. However, strict parsers that trust the declared sizes
(e.g., Python's `wave` module) may reject a saved file. For byte-exact or
random-access needs, request `response_format=pcm` instead (raw 24 kHz 16-bit
mono PCM, no header).

### Speech transcription (STT)

```bash
curl -X POST http://localhost:8790/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=base"
```

## Hardware

CPU-first, GPU-optional. English-only models by default (swappable via config).
See the spec for a reference CPU benchmark and rationale.
