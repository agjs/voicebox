# Notes for Pieces B & C (client-side)

Distilled, vendor-neutral engineering notes to inform the Piece B (voice chat
client) and Piece C (coding-agent voice) specs. Piece A (this repo's server) is
unaffected — these are consumer-side concerns.

## Real-time voice loop (Piece B)

Target end-to-end latency budget: keep a turn under ~3 s (dominated by LLM
generation, not the speech layer). Run stages concurrently, not sequentially:

```
mic (continuous PCM) → VAD → STT → LLM (stream=true)
  → incremental sentence parser → streaming TTS → non-blocking playback
```

- **VAD (endpointing):** use a real VAD engine, not just an amplitude gate.
  Options: Silero VAD v5 (ONNX, ~10 ms) or WebRTC VAD. Silence cutoff ~1.5 s to
  mark end-of-utterance.
- **Sample rate:** system mics commonly capture 48 kHz; Whisper expects mono
  **16 kHz**. Downsample correctly or transcription degrades.
- **Sentence-level chunking:** accumulate streamed LLM tokens; on a sentence
  boundary (`. ? !` / newline) flush the phrase to TTS so playback of sentence 1
  starts while the LLM is still generating. (Piece A's `/v1/audio/speech` already
  streams per sentence, so the client can also just forward text and let the
  server chunk — measure which is lower latency.)
- **Barge-in:** keep the mic hot during playback; if VAD detects new speech,
  stop playback, flush the audio queue, and start a new turn.
- **LLM stream parsing:** parse the SSE/JSON stream properly (do not regex
  `"content":"..."`). Strip model reasoning tags (e.g. `<think>...</think>`)
  before sending text to TTS.
- **Playback:** non-blocking worker (e.g. `sounddevice`), 24 kHz to match Kokoro
  output.

Reference deps (client host): `ffmpeg`, PortAudio, `sounddevice`, `numpy`,
`soundfile`, an async HTTP client.

## Coding-agent voice (Piece C) — Claude Code, verified against docs

Two halves — dictation (speech → prompt) and spoken replies (response → speech).

### Spoken replies (TTS out) — recommended, fully local

The `Stop` hook fires after the assistant finishes a response and receives JSON
on stdin including `last_assistant_message` (event `hook_event_name: "Stop"`). A
hook script extracts that text and POSTs it to the Piece A `/v1/audio/speech`
endpoint, then plays the returned audio locally (`afplay` on macOS,
`mpv`/`ffplay`/`aplay` on Linux). No MCP required; fully local against Piece A.
Registered under `hooks.Stop` in the agent's settings.

### Dictation (STT in) — one real decision to make in the Piece C spec

- **Native dictation:** Claude Code has a built-in push-to-talk dictation
  (`hold` and `tap` modes), configured via a `voice` block in settings plus a
  top-level `language` setting, and optional keybindings. **Trade-off: it is
  cloud-based** (transcribes on the vendor's servers, needs a signed-in
  account) — trivial to enable but NOT local, so it bypasses Piece A.
- **Local dictation:** wrap Piece A's `/v1/audio/transcriptions` in a small MCP
  server (a `transcribe` tool the agent can call) or a push-to-talk client that
  injects transcribed text. More work, but fully local and vendor-neutral.

Decision for the Piece C spec: native cloud dictation (easy) vs. local MCP/
push-to-talk against Piece A (aligned with the local-first goal).

### MCP option (both directions)

A bidirectional voice MCP server is an alternative to the hook: expose
`transcribe` + `synthesize` tools that call Piece A. Community servers exist that
use the same faster-whisper + Kokoro stack and could be pointed at Piece A
instead of bundling their own engines.

## Prior art (patterns to borrow, not dependencies)

- Custom async Python S2S loop (faster-whisper + kokoro-onnx + sounddevice) —
  the canonical Piece B shape.
- Post-response TTS hooks and bidirectional voice MCP servers — the canonical
  Piece C shapes.
