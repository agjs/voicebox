#!/bin/bash
# Claude Code Stop hook: synthesize the first sentence first, then queue the rest.
set -euo pipefail

VOICEBOX_URL="${VOICEBOX_URL:-http://localhost:8790}"
VOICEBOX_VOICE="${VOICEBOX_VOICE:-en_US-amy-medium}"
VOICEBOX_SPEAK_MAX_CHARS="${VOICEBOX_SPEAK_MAX_CHARS:-600}"
VOICEBOX_LOG="${VOICEBOX_LOG:-/dev/null}"

message=$(jq -r '.last_assistant_message // empty' 2>/dev/null || true)
if [[ -z "$message" ]]; then
    exit 0
fi

# The sed expression is intentionally single-quoted; "$!" is a sed address.
# shellcheck disable=SC2016
cleaned=$(printf '%s\n' "$message" | \
    sed -e ':a' -e '$!N;$!ba' -e 's/```[^`]*```//g' | \
    sed \
    -e 's#https\{0,1\}://[^ ]*##g' \
    -e 's/`//g' \
    -e 's/[*#]//g' \
    -e 's/[[:space:]]\+/ /g' \
    -e 's/^ *//;s/ *$//')

if [[ -z "$cleaned" ]]; then
    exit 0
fi

if (( ${#cleaned} > VOICEBOX_SPEAK_MAX_CHARS )); then
    cleaned="${cleaned:0:VOICEBOX_SPEAK_MAX_CHARS}"
    cleaned="${cleaned% *}"
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

first=$(printf '%s\n' "$cleaned" | awk '
    match($0, /[.!?]([[:space:]]|$)/) { print substr($0, 1, RSTART); next }
    { print }
')
remaining="${cleaned:${#first}}"
remaining=$(printf '%s' "$remaining" | sed -e 's/^ *//;s/ *$//')

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
