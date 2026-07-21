# Claude Code Voice Integration

Speak Claude Code's replies via local TTS and dictate locally via local STT, using voicebox as the speech engine.

## What You Get

- **Spoken replies**: Enable a Stop-hook that speaks a cleaned summary of Claude Code's assistant messages
- **Local dictation**: Record from your microphone or transcribe existing audio files without cloud services

## Requirements

- `jq` and `curl` (for the Stop-hook)
- An audio player: `afplay` (macOS), `mpv`, `ffplay`, or `aplay` (Linux)
- A reachable voicebox instance (default: `http://localhost:8790`)

For local dictation only:
- Python 3.11+
- `sounddevice` and `scipy` (if using microphone recording)

## Installation

### 1. Copy the scripts to your system

```bash
cp speak.sh ~/.local/bin/voicebox-speak.sh
chmod +x ~/.local/bin/voicebox-speak.sh

cp dictate.py ~/.local/bin/voicebox-dictate
chmod +x ~/.local/bin/voicebox-dictate
```

(Ensure `~/.local/bin` is in your `$PATH`.)

### 2. Enable spoken replies (optional)

To automatically speak Claude Code's assistant messages, add the `hooks.Stop` block from `settings.snippet.json` into your `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": {
      "command": "${HOME}/.local/bin/voicebox-speak.sh"
    }
  }
}
```

The hook speaks a cleaned ≤600-character summary of each reply:
- Fenced code blocks are removed entirely
- Markdown formatting is stripped (backticks, asterisks, headings)
- Whitespace is collapsed

If your voicebox is not at `http://localhost:8790`, set the environment variable:

```bash
export VOICEBOX_URL=http://<host>:8790
```

You can also customize:
- `VOICEBOX_VOICE` (default: `af_heart`) — voice identifier
- `VOICEBOX_SPEAK_MAX_CHARS` (default: `600`) — character limit for spoken text
- `VOICEBOX_LOG` (default: `/dev/null`) — optional log file for debugging

## Dictation

### Option A: Native `/voice` (built-in, cloud-based)

Claude Code includes a native `/voice` push-to-talk command that transcribes speech via Anthropic's cloud.

### Option B: Local dictation with voicebox

For fully local transcription via voicebox:

```bash
# Install dependencies (if using microphone)
pip install sounddevice scipy

# Record and transcribe from microphone
~/.local/bin/voicebox-dictate

# Or transcribe an existing file
~/.local/bin/voicebox-dictate --file /path/to/audio.wav
```

The transcript is printed to stdout and copied to your clipboard (pbcopy on macOS, xclip/wl-copy on Linux).

**Trade-off**: `/voice` is convenient and cloud-backed; `voicebox-dictate` is fully local, requires extra setup, and no external transcription service.

## Environment

All scripts use environment variables for configuration:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VOICEBOX_URL` | `http://localhost:8790` | voicebox endpoint |
| `VOICEBOX_VOICE` | `af_heart` | TTS voice identifier |
| `VOICEBOX_SPEAK_MAX_CHARS` | `600` | Max characters to speak per reply |
| `VOICEBOX_LOG` | `/dev/null` | Optional debug log for the speak hook |

Example:

```bash
export VOICEBOX_URL=http://192.168.1.100:8790
export VOICEBOX_VOICE=en_male
```

## Troubleshooting

**No audio is being spoken**
- Check that `VOICEBOX_URL` is reachable: `curl -I $VOICEBOX_URL/health`
- Verify that an audio player is installed (`afplay`, `mpv`, `ffplay`, or `aplay`)
- Check `VOICEBOX_LOG` if set: `tail -f /path/to/logfile`

**Dictation errors**
- Ensure sounddevice and scipy are installed: `pip install sounddevice scipy`
- Test the transcription endpoint: `curl -X POST http://localhost:8790/v1/audio/transcriptions -F file=@yourfile.wav`

**Clipboard not working**
- Install `pbcopy` (macOS, built-in), `xclip` (X11), or `wl-copy` (Wayland)
- Fallback: the transcript is always printed; use `voicebox-dictate | some-command` to pipe it
