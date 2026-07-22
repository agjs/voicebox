# OpenAI audio API depth

Date: 2026-07-22  
Status: approved

## Purpose

Deepen OpenAI audio compatibility: `/v1/models`, STT `verbose_json` (segment timestamps), TTS `speed`, Piper request `voice` with all three baked voices preloaded.

## Behavior

- `GET /v1/models` — list object with STT model id plus current-engine TTS voice ids (Piper: amy/bryce/lessac; Kokoro: default voice). Auth like other non-health routes.
- `POST /v1/audio/transcriptions` — `response_format` in `{json,text,verbose_json}`. `verbose_json` returns task/language/duration/text/segments (segment start/end/text only). `json`/`text` keep current shapes and `without_timestamps=True`.
- `POST /v1/audio/speech` — honor `speed` in `[0.25, 4.0]` (default 1.0). Piper: `length_scale = piper_length_scale / speed`. Kokoro: pass `speed` into `create`. Invalid → 400.
- Piper preloads amy/bryce/lessac at startup; request `voice` selects among them (default env voice); unknown → 400.

## Not doing

Word timestamps, translations endpoint, mp3/opus, Realtime, multilingual model swap.

## Verify

`pytest -m "not model and not latency"`
