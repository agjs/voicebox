#!/bin/bash
# Claude Code Stop hook: synthesize the first sentence first, then queue the rest.
set -euo pipefail

VOICEBOX_URL="${VOICEBOX_URL:-http://localhost:8790}"
VOICEBOX_VOICE="${VOICEBOX_VOICE:-en_US-amy-medium}"
VOICEBOX_SPEAK_MAX_CHARS="${VOICEBOX_SPEAK_MAX_CHARS:-600}"
VOICEBOX_LOG="${VOICEBOX_LOG:-/dev/null}"
HERE="$(cd "$(dirname "$0")" && pwd)"

message=$(jq -r '.last_assistant_message // empty' 2>/dev/null || true)
if [[ -z "$message" ]]; then
    exit 0
fi

cleaned_and_split=$(printf '%s\n' "$message" | python3 "$HERE/speak_text.py" --max-chars "$VOICEBOX_SPEAK_MAX_CHARS" --split-first)
if [[ -z "$cleaned_and_split" ]]; then
    exit 0
fi

first="${cleaned_and_split%%$'\0'*}"
remaining="${cleaned_and_split#*$'\0'}"
if [[ "$remaining" == "$cleaned_and_split" ]]; then
    remaining=""
fi
cleaned="$first"
if [[ -n "$remaining" ]]; then
    cleaned="$first $remaining"
fi

if [[ "$VOICEBOX_LOG" != "/dev/null" ]]; then
    printf '[speak.sh] cleaned text: %s\n' "$cleaned" >> "$VOICEBOX_LOG"
fi

player=""
for candidate in afplay mpv ffplay aplay; do
    if command -v "$candidate" >/dev/null 2>&1; then
        player="$candidate"
        break
    fi
done
if [[ -z "$player" ]]; then
    exit 0
fi

auth_args=()
if [[ -n "${VOICEBOX_API_KEY:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${VOICEBOX_API_KEY}")
fi

post_wav() {
    local text="$1"
    local output="$2"
    jq -n \
        --arg model "tts" \
        --arg input "$text" \
        --arg voice "$VOICEBOX_VOICE" \
        '{model: $model, input: $input, voice: $voice, response_format: "wav"}' \
        | curl --fail --silent --show-error \
            --connect-timeout 3 --max-time 60 \
            -H "Content-Type: application/json" \
            "${auth_args[@]}" \
            -d @- "$VOICEBOX_URL/v1/audio/speech" -o "$output"
    [[ -s "$output" ]] && [[ "$(head -c 4 "$output")" == "RIFF" ]]
}

play_wav() {
    local input="$1"
    case "$player" in
        mpv) "$player" --no-video --really-quiet "$input" ;;
        ffplay) "$player" -nodisp -autoexit -loglevel quiet "$input" ;;
        aplay) "$player" -q "$input" ;;
        *) "$player" "$input" ;;
    esac
}

temp_dir=$(mktemp -d -t voicebox)
(
    trap 'rm -f "$temp_dir/first.wav" "$temp_dir/rest.wav"; rmdir "$temp_dir" 2>/dev/null || true' EXIT
    if ! post_wav "$first" "$temp_dir/first.wav"; then
        exit 0
    fi

    rest_pid=""
    if [[ -n "$remaining" ]]; then
        post_wav "$remaining" "$temp_dir/rest.wav" &
        rest_pid=$!
    fi

    play_wav "$temp_dir/first.wav" || true
    if [[ -n "$rest_pid" ]] && wait "$rest_pid"; then
        play_wav "$temp_dir/rest.wav" || true
    fi
) >/dev/null 2>&1 &

exit 0
