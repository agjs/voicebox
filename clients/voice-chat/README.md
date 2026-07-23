# Voice Chat Client: Turn-Taking CLI

A conversational CLI that lets you talk to a local LLM and hear spoken replies, using the voicebox speech server for STT (speech-to-text) and streaming TTS (text-to-speech).

## Overview

The voice chat client implements a full end-to-end pipeline:

```
Mic (16 kHz mono) → VAD endpointing → voicebox STT → local LLM (streamed)
→ incremental sentence parser → background voicebox TTS → queued PCM playback
```

Key features:
- **Turn-taking loop**: record until silence (VAD), send to STT, stream from LLM, synthesize and play responses incrementally
- **Reasoning stripping**: removes `<think>...</think>` blocks from models like DeepSeek so they're never spoken
- **Sentence chunking**: parses streamed LLM output and emits complete sentences as they're detected (`.?!` or newline), so TTS can start while the LLM is still generating
- **Multiple modes**: interactive mic (optional `--barge-in`), `--wake` for hey-jarvis activation, `--text`, `--file`
- **Graceful degradation**: audio libraries are optional (lazy-imported) so `--text --no-audio` and tests work without audio dependencies

## Installation

Requires a running [voicebox](https://github.com/agjs/voicebox) server for STT/TTS
and an OpenAI-compatible LLM endpoint. The client package is separate and light
(httpx, numpy, sounddevice, webrtcvad, openwakeword) — it does not install the
speech server. First `--wake` run downloads openWakeWord model files once.

```bash
uv tool install "git+https://github.com/agjs/voicebox.git#subdirectory=clients/voice-chat"
# or: pip install "git+https://github.com/agjs/voicebox.git#subdirectory=clients/voice-chat"
```

From a clone (editable, for development):

```bash
pip install -e clients/voice-chat
```

## Configuration

All configuration is via environment variables (defaults are generic):

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOICEBOX_URL` | `http://localhost:8790` | voicebox server URL (STT/TTS) |
| `VOICEBOX_LLM_URL` | `http://localhost:8000/v1/chat/completions` | LLM endpoint (OpenAI-compatible) |
| `VOICEBOX_LLM_MODEL` | `local-model` | LLM model name |
| `VOICEBOX_VOICE` | `en_US-amy-medium` | TTS voice ID (Piper uses its server-configured voice) |
| `VOICEBOX_SILENCE_MS` | `700` | Silence required to end a turn |
| `VOICEBOX_PRE_ROLL_MS` | `300` | Audio retained before speech detection |
| `VOICEBOX_POST_ROLL_MS` | `200` | Silence retained after speech |
| `VOICEBOX_VAD_AGGRESSIVENESS` | `2` | WebRTC VAD mode, 0 through 3 |
| `VOICEBOX_MAX_HISTORY_TURNS` | `8` | Conversation turns retained for LLM latency control |
| `VOICEBOX_SHOW_TIMINGS` | `0` | Set to `1` for STT, first-token, and first-audio timings |
| `VOICEBOX_SYSTEM_PROMPT` | (see below) | System prompt for LLM |
| `VOICEBOX_WAKE_MODEL` | `hey_jarvis` | openWakeWord pretrained model id |
| `VOICEBOX_WAKE_THRESHOLD` | `0.5` | Minimum wake score to activate |
| `VOICEBOX_WAKE_IDLE_SECONDS` | `300` | Silence after last utterance before sleep |

Default system prompt asks for concise plain text suitable for speech.

## Usage

`voicebox-chat` is a long-running interactive session: start it when you want to
talk, leave it running through turns, press `Ctrl-C` to exit.

### Interactive Mic Mode (default)

```bash
export VOICEBOX_URL=http://localhost:8790
export VOICEBOX_LLM_URL=http://localhost:8000/v1/chat/completions
export VOICEBOX_LLM_MODEL=local-model
voicebox-chat
# or: voicebox-chat --barge-in
```

From a clone without installing the console script:

```bash
python clients/voice-chat/voice_chat.py --barge-in
```

Starts a loop:
1. **Record**: Waits for speech and ends after about 700 ms of silence
2. **Transcribe**: Sends audio to voicebox STT
3. **Stream LLM**: Sends your utterance + chat history to the LLM
4. **Synthesize & play**: A TTS worker and persistent PCM output stream run while LLM generation continues
5. **Repeat**: Ready for next turn

### Wake Mode

Always-on mic listen for the wake phrase, then enter conversation until you say
goodbye (or idle times out), then listen for wake again. Chat history is kept
across wake cycles for the life of the process.

```bash
voicebox-chat --wake
# or: voicebox-chat --wake --barge-in
```

- Default wake phrase: **hey jarvis** (`VOICEBOX_WAKE_MODEL`)
- End session: say **goodbye**, **stop listening**, or **that's all**; or wait
  `VOICEBOX_WAKE_IDLE_SECONDS` (default 5 minutes) with no speech
- Cannot combine with `--text` / `--file`
- Wake listening keeps the mic open continuously; room walking works better with
  a headset or a good far-field mic. Headphones are still recommended when using
  `--barge-in` during conversation (no AEC).

### Text Mode

Process text directly without mic input:

```bash
voicebox-chat --text "In one short sentence, what is a vector database?" --no-audio
```

- `--text "..."`: Send text straight through LLM → sentence parser → TTS (skips STT/mic)
- `--no-audio`: Print output only, skip TTS synthesis and playback

Useful for testing without audio hardware.

### File Mode

Transcribe a WAV file, then process the transcript through LLM:

```bash
voicebox-chat --file path/to/audio.wav
```

- Expects 16 kHz mono WAV format (or any format soundfile can decode)
- Transcribes the file via voicebox STT
- Then processes the transcript through LLM → TTS → playback

Combine with `--no-audio` to skip playback:

```bash
voicebox-chat --file path/to/audio.wav --no-audio
```

## Architecture

### `pipeline.py`: Pure, Testable Logic

All functions are vendor-neutral, network-free, and unit-testable:

- **`parse_sse_stream(chunks)`**: Parses OpenAI-style `data: {...}` SSE lines from streamed LLM responses. Yields text tokens, stops on `[DONE]`, ignores keep-alives.
  
- **`ReasoningFilter`**: Stateful removal of `<think>...</think>` blocks even when tags are split across network chunks.
  
- **`SentenceChunker`**: Incremental state machine. Call `.feed(text)` to process streamed text; emits complete sentences on `.?!` or newline boundaries. Includes a min-length guard to avoid over-splitting on abbreviations like "U.S." Flush with `.flush()` at end of stream to get any remainder.

### `voice_chat.py`: CLI & Integration

Main entry point combining the pipeline, voicebox APIs, and audio I/O:

- **`VoiceChat` class**: Manages chat history, HTTP requests, audio I/O
  - `record_from_mic()`: Records from the default device using VAD (webrtcvad if available, else energy-based fallback)
  - `transcribe(audio_bytes)`: POSTs to voicebox STT
  - `stream_llm_response(messages)`: Streams bytes from the LLM endpoint
  - `stream_speech(text)`: Streams PCM and reads its sample rate from response headers
  - `PcmPlayback`: Writes every sentence to one persistent `RawOutputStream`
  
- **Three run modes**:
  - `run_interactive()`: Mic loop
  - `run_text_mode(text, no_audio)`: Direct text
  - `run_file_mode(filepath, no_audio)`: File transcription

Lazy imports ensure audio libraries aren't required for text-only usage.

### `test_pipeline.py`: Unit Tests

Full coverage of the pure pipeline logic:
- SSE parsing (multiline, keep-alives, malformed JSON, [DONE] marker)
- Reasoning stripping (blocks, stray tags, preserving other XML)
- Sentence chunking (boundaries, abbreviation guards, incremental feeding, flushing)

Run with:
```bash
python -m pytest clients/voice-chat/test_pipeline.py -v
```

No network or audio required.

## Barge-In (Interrupt on Speech)

Opt-in for interactive mic mode only (not `--text` / `--file`):

```bash
voicebox-chat --barge-in
``` (LLM + TTS + playback), a parallel VAD monitor
listens on the mic. After ~200 ms of cumulative voiced frames, playback aborts,
TTS/LLM streams cancel, partial assistant text stays in history, and the loop
returns to the next listen turn. `Ctrl-C` uses the same cancel/abort path.

**Headphones required.** There is no acoustic echo cancellation in this release.
Open speakers will often self-interrupt when the TTS output reaches the mic.
Raising `VOICEBOX_VAD_AGGRESSIVENESS` only reduces noise false positives; it does
not replace AEC.

## Latency & Performance

Expected latency (turn-taking mode):
- **Mic endpoint delay**: ~0.7 seconds after speech (configurable)
- **STT roundtrip**: ~0.5–2 seconds (model + network)
- **LLM generation**: dominant factor, often 5–30 seconds depending on model and response length
- **TTS streaming**: begins immediately as sentences arrive; the server declares the PCM rate

Sentences are synthesized in parallel with LLM generation and written to a single
queued output stream. Playback is gapless. Without `--barge-in`, later sentences
cannot interrupt earlier ones, and the microphone does not reopen until playback
finishes. With `--barge-in`, speaking can abort the turn mid-playback (headphones).

## Error Handling

- **Network errors** (unreachable endpoints): Caught and logged; loop continues (user can retry)
- **Audio not available**: Gracefully degraded (falls back to text-only if sounddevice missing)
- **Empty transcripts**: Logged; loop continues
- **Malformed LLM responses**: SSE parser is resilient (skips bad JSON, stops on [DONE])

## Constraints & Design

- **Vendor-neutral**: No hardcoded IPs, hostnames, or personal infra. All endpoints via generic env vars with sensible defaults.
- **Connection reuse**: One thread-safe `httpx.Client` serves STT, LLM streaming, and background TTS.
- **Lazy loading**: Audio libraries imported only when needed (`--text --no-audio` works without sounddevice).
- **Testable**: Pure pipeline logic has no side effects; integration is in the CLI.

## Example Session

```bash
$ export VOICEBOX_SYSTEM_PROMPT="You are a pirate. Respond briefly."
$ voicebox-chat
Voice Chat CLI (Ctrl-C to exit)
Recording... (speak, then pause; Ctrl-C to cancel)
Silence detected.
User: What is machine learning?
Assistant: Arr, 'tis the art of teaching scurvy machines to learn without explicit orders, ye scallywag!

Recording... (Ctrl-C to stop)
Silence detected.
User: Tell me a joke.
Assistant: Why did the buccaneer go to the gym? To get his pirate ship in shape, har har!

^C
Exiting...
```

## Troubleshooting

### "sounddevice not available"
Reinstall the client so audio deps are present:
`uv tool install --force "git+https://github.com/agjs/voicebox.git#subdirectory=clients/voice-chat"`
(or `pip install -e clients/voice-chat` from a clone).

### "LLM endpoint unreachable"
Check that your local LLM server is running at the URL set in `VOICEBOX_LLM_URL`.

### "voicebox STT/TTS failing"
Verify the voicebox server is running: `VOICEBOX_URL` should be accessible.

### "No speech recognized"
- Check mic is unmuted and working
- Try `voicebox-chat --file path/to/audio.wav` to test STT with a known audio file
- Adjust `VOICEBOX_SILENCE_MS` if silence detection is too aggressive

### "Tests failing"
The unit tests don't require network or audio:
```bash
python -m pytest clients/voice-chat/test_pipeline.py -v
```

If you're seeing import errors, ensure you're in the venv and the pipeline module is importable from the test directory.

---

**Built for voicebox.** Designed to be simple, robust, and extensible.
