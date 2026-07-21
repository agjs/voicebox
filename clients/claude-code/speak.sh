#!/bin/bash
# Claude Code Stop-hook: speak the last assistant message via voicebox TTS
set -euo pipefail

# Configuration from environment
VOICEBOX_URL="${VOICEBOX_URL:-http://localhost:8790}"
VOICEBOX_VOICE="${VOICEBOX_VOICE:-af_heart}"
VOICEBOX_SPEAK_MAX_CHARS="${VOICEBOX_SPEAK_MAX_CHARS:-600}"
VOICEBOX_LOG="${VOICEBOX_LOG:-/dev/null}"

# Read stdin and extract last_assistant_message
message=$(jq -r '.last_assistant_message // empty' 2>/dev/null || true)

# If message is missing, empty, or null, exit silently
if [[ -z "$message" ]]; then
    exit 0
fi

# Clean text for speech:
# 1. Remove fenced code blocks (triple backtick + content + triple backtick)
# 2. Remove inline code backticks and markdown formatting (*, #, etc.)
# 3. Collapse whitespace
# 4. Cap to max characters

cleaned=$(printf '%s\n' "$message" | \
    sed -e ':a' -e '$!N;$!ba' -e 's/```[^`]*```//g' | \
    sed \
    -e 's/`//g' \
    -e 's/\*//g' \
    -e 's/#//g' \
    -e 's/[[:space:]]\+/ /g' \
    -e 's/^ *//;s/ *$//' | \
    sed -e '/^$/d')

# If nothing remains after cleaning, exit silently
if [[ -z "$cleaned" ]]; then
    exit 0
fi

# Cap to max characters
cleaned="${cleaned:0:$VOICEBOX_SPEAK_MAX_CHARS}"

# Log the cleaned text (optional)
if [[ "$VOICEBOX_LOG" != "/dev/null" ]]; then
    echo "[speak.sh] cleaned text: $cleaned" >> "$VOICEBOX_LOG"
fi

# Create a temp file for the audio
tmpfile=$(mktemp -t voicebox.XXXXXX.wav)

# Build JSON safely with jq and POST to voicebox
if ! jq -n \
    --arg model "tts" \
    --arg input "$cleaned" \
    --arg voice "$VOICEBOX_VOICE" \
    --arg response_format "wav" \
    '{model: $model, input: $input, voice: $voice, response_format: $response_format}' \
    | curl -s -X POST \
        -H "Content-Type: application/json" \
        -d @- \
        "$VOICEBOX_URL/v1/audio/speech" \
        -o "$tmpfile" 2>/dev/null; then
    rm -f "$tmpfile"
    exit 0
fi

# Check if we got a valid response (should be RIFF WAVE header)
if [[ ! -s "$tmpfile" ]] || ! head -c 4 "$tmpfile" | grep -q "RIFF"; then
    rm -f "$tmpfile"
    exit 0
fi

# Play the audio in the background, detached
# Try available players in order: afplay (macOS), mpv, ffplay, aplay (Linux)
player_found=0
for player in afplay mpv ffplay aplay; do
    if command -v "$player" &>/dev/null; then
        {
            "$player" "$tmpfile" 2>/dev/null || true
            rm -f "$tmpfile"
        } &
        player_found=1
        break
    fi
done

# If no player found, just clean up
if [[ $player_found -eq 0 ]]; then
    rm -f "$tmpfile"
fi

# Hook must always exit 0
exit 0
