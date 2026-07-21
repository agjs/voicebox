# voicebox

**A tiny, self-hosted, OpenAI-compatible speech server.** Speech-to-text
(`faster-whisper`) and text-to-speech (`Piper` or `Kokoro`) behind the exact
same HTTP API as OpenAI's audio endpoints — so anything that already speaks that
API (Open WebUI, your own agents, CLIs, coding assistants) gets a local voice
with zero glue code.

- 🗣️ **Talk to your local models.** One backend, many clients.
- 🔒 **Fully local & private.** No cloud, no API keys, no data leaving your box.
- ⚡ **Fast on plain CPUs.** Real-time on a mini PC or an old quad-core — no GPU required (but it'll use one if you have it).
- 🔌 **Drop-in OpenAI audio API.** `/v1/audio/transcriptions` and `/v1/audio/speech`.
- 🎚️ **Swappable voices & engines** via env vars — no rebuild.

---

## Quick start

You need Docker. That's it — the models are baked into the image.

```bash
git clone https://github.com/agjs/voicebox.git && cd voicebox
docker compose up -d --build      # first build downloads the models (~a few min)

curl -fsS localhost:8790/health   # {"status":"ok","models_loaded":true}
```

**Text → speech:**
```bash
curl -fsS localhost:8790/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts","input":"Hello from voicebox.","response_format":"wav"}' \
  --output hello.wav
```

**Speech → text:**
```bash
curl -fsS localhost:8790/v1/audio/transcriptions \
  -F "file=@hello.wav" -F "model=stt"
# {"text":"Hello from voicebox."}
```

Point any OpenAI-audio-compatible client at `http://<host>:8790/v1` and you're done.

> No Docker? `pip install -e ".[dev]" && python -m voicebox` also works
> (needs `ffmpeg` and `espeak-ng` on the host).

---

## Use it with your apps

### Open WebUI
Admin → Settings → **Audio**:

| Setting | Value |
|---|---|
| Speech-to-Text Engine | `OpenAI` |
| STT API Base URL | `http://<host>:8790/v1` |
| STT API Key | `sk-none` *(any non-empty string — voicebox has no auth)* |
| Text-to-Speech Engine | `OpenAI` |
| TTS API Base URL | `http://<host>:8790/v1` |
| TTS Voice | `af_heart` *(ignored by Piper; used by Kokoro)* |

### Included clients (`clients/`)
- **`clients/voice-chat/`** — a mic → STT → your LLM → streaming TTS turn-taking CLI.
- **`clients/claude-code/`** — a Stop-hook that reads Claude Code's replies aloud, plus a push-to-talk dictation helper.

Both are configured with `VOICEBOX_URL` (and, for voice-chat, your LLM endpoint).

---

## Voices

TTS defaults to **Piper** — fast, natural, CPU-friendly. Three voices ship baked
into the image and switch with no rebuild:

| `VOICEBOX_PIPER_VOICE` | character |
|---|---|
| `en_US-amy-medium` *(default)* | warm, female |
| `en_US-bryce-medium` | male |
| `en_US-lessac-medium` | clear, neutral |

**🎧 Audition every Piper voice here → https://rhasspy.github.io/piper-samples/**

Found one you like? Set `VOICEBOX_PIPER_VOICE=en_US-<voice>-<quality>`. Voices
not baked in are fetched on first use (needs network, or add it to
`scripts/fetch_models.py` and rebuild to keep the image offline).

**Speaking rate:** `VOICEBOX_PIPER_LENGTH_SCALE` — lower is faster
(`1.0` = natural, `0.8` = brisk, `1.2`+ = slow/narration).

### Two engines
- **`piper`** (default) — fastest; great quality; ~7× faster than Kokoro on CPU.
- **`kokoro`** — higher-fidelity neural voices (~50 of them, e.g. `af_bella`,
  `bf_emma`), slower on CPU. Switch with `VOICEBOX_TTS_ENGINE=kokoro` and pick a
  voice with the `voice` request field or `VOICEBOX_DEFAULT_VOICE`.

---

## API

| Endpoint | Description |
|---|---|
| `POST /v1/audio/transcriptions` | Multipart `file` (any common audio format) → `{"text": "..."}`. |
| `POST /v1/audio/speech` | JSON `{input, voice?, response_format?}` → audio. `wav` (default, complete file) or `pcm` (raw 16-bit mono, streamed). |
| `GET /health` | `{"status":"ok","models_loaded":true}` |

Shapes match OpenAI's audio API, so existing SDKs and clients work unmodified.

---

## Configuration

All via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `VOICEBOX_TTS_ENGINE` | `piper` | `piper` or `kokoro` |
| `VOICEBOX_PIPER_VOICE` | `en_US-amy-medium` | Piper voice id |
| `VOICEBOX_PIPER_LENGTH_SCALE` | `1.0` | Speaking rate (lower = faster) |
| `VOICEBOX_PIPER_NOISE_SCALE` | `0.667` | Prosody variability |
| `VOICEBOX_PIPER_NOISE_W` | `0.8` | Phoneme-duration variability |
| `VOICEBOX_STT_MODEL` | `Systran/faster-distil-whisper-small.en` | faster-whisper model |
| `VOICEBOX_TTS_MODEL` | `speaches-ai/Kokoro-82M-v1.0-ONNX` | Kokoro model (when engine=kokoro) |
| `VOICEBOX_DEFAULT_VOICE` | `af_heart` | Kokoro default voice |
| `VOICEBOX_DEVICE` | `cpu` | `cpu` or `cuda` |
| `VOICEBOX_PORT` | `8790` | Listen port |
| `VOICEBOX_MAX_AUDIO_SECONDS` | `120` | Reject longer STT input |
| `VOICEBOX_MAX_UPLOAD_MB` | `25` | Reject larger uploads |
| `VOICEBOX_MAX_INPUT_CHARS` | `4000` | Reject longer TTS text |

---

## Performance

CPU-only, single reference box (2015-era quad-core, no GPU):

| Stage | Model | Real-time factor |
|---|---|---|
| STT | `distil-whisper-small.en` (int8) | ~0.3× (≈3× faster than real-time) |
| TTS | Piper `amy-medium` | ~0.07× (≈14× faster than real-time) |
| TTS | Kokoro-82M | ~0.5× |

A newer CPU or an NVIDIA GPU (`VOICEBOX_DEVICE=cuda`) is faster still.

---

## Development

```bash
pip install -e ".[dev]"
pytest                    # unit + latency tests
docker compose up -d --build && ./scripts/smoke.sh   # end-to-end
```

Layout: `src/voicebox/` — `stt.py` (faster-whisper), `tts.py` (Kokoro),
`tts_piper.py` (Piper), `app.py` (FastAPI routes), `config.py` (env), `wav.py`.

---

## Licenses

voicebox itself is **MIT** (see [`LICENSE`](LICENSE)). It builds on:

| Component | License |
|---|---|
| faster-whisper, ONNX Runtime, kokoro-onnx | MIT |
| Kokoro-82M model | Apache-2.0 |
| Piper voices (amy/bryce/lessac) | public domain / MIT |
| **piper-tts** | **GPL-3.0** |

⚠️ **Note:** `piper-tts` is GPL-3.0. voicebox's own source is MIT and only
imports it (nothing is vendored), but the prebuilt **Docker image bundles it**,
so a distributed image carries GPL obligations for that component. If you need a
purely permissive image, set `VOICEBOX_TTS_ENGINE=kokoro` and drop `piper-tts`
from `pyproject.toml` / the Dockerfile.
