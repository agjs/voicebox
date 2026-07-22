#!/usr/bin/env bash
# Canned README demo - no live voicebox required.
# Used by assets/demo.tape (vhs).
set -euo pipefail

green=$'\033[32m'
cyan=$'\033[36m'
dim=$'\033[2m'
bold=$'\033[1m'
reset=$'\033[0m'

pause() { sleep "${1:-0.45}"; }

printf '\033c' || true
printf '%s# voicebox 0.2.8 - self-hosted OpenAI-compatible speech%s\n' "$dim" "$reset"
pause 0.4

printf '%s$%s curl -fsS localhost:8790/v1/models\n' "$green" "$reset"
pause 0.3
printf '{"object":"list","data":[{"id":"Systran/faster-distil-whisper-small.en"},{"id":"en_US-amy-medium"},...]}\n'
pause 0.55

printf '%s$%s curl -fsS localhost:8790/health\n' "$green" "$reset"
pause 0.25
printf '{"status":"ok","models_loaded":true}\n'
pause 0.5

printf '%s$%s curl -fsS localhost:8790/v1/audio/speech \\\n' "$green" "$reset"
printf '  -H "Content-Type: application/json" \\\n'
printf '  -d '"'"'{"model":"tts","input":"Hello from voicebox.","voice":"en_US-amy-medium","speed":1.0,"response_format":"wav"}'"'"' \\\n'
printf '  --output hello.wav\n'
pause 0.4
printf '%s# wrote hello.wav  (Piper, ~14x faster than real-time on CPU)%s\n' "$dim" "$reset"
pause 0.65

printf '%s$%s curl -fsS localhost:8790/v1/audio/transcriptions \\\n' "$green" "$reset"
printf '  -F file=@hello.wav -F model=stt -F response_format=json\n'
pause 0.35
printf '{"text":"Hello from voicebox."}\n'
pause 0.55

printf '%s$%s %s# point any OpenAI-audio client at http://localhost:8790/v1%s\n' "$green" "$reset" "$cyan" "$reset"
pause 1.0

printf '%s%sready.%s\n' "$bold" "$green" "$reset"
pause 1.2
