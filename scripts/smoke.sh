#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://localhost:8790}"

echo "health:"; /usr/bin/curl -fsS "$BASE/health"; echo

echo "tts -> /tmp/vb_smoke.wav"
/usr/bin/curl -fsS "$BASE/v1/audio/speech" -H "Content-Type: application/json" \
  --output /tmp/vb_smoke.wav \
  -d '{"model":"m","input":"Hello from voicebox. Streaming works.","response_format":"wav"}'

# Validate raw streaming WAV (no reconstruction)
python3 << 'EOF'
import struct
import os

def validate_streaming_wav(filename: str) -> float:
    """Validate raw streaming WAV with placeholder header sizes.

    The WAV header contains 0xFFFFFFFF placeholders for RIFF and data sizes,
    which is standard for streaming WAV. Compute duration from actual file size.
    Returns: duration in seconds
    """
    with open(filename, 'rb') as f:
        data = f.read()

    # Validate WAV header structure
    assert len(data) >= 44, f"WAV too short: {len(data)} bytes"
    assert data[0:4] == b"RIFF", f"Invalid RIFF magic: {data[0:4]}"
    assert data[8:12] == b"WAVE", f"Invalid WAVE magic: {data[8:12]}"

    # Compute duration from actual file size (44-byte header + PCM data)
    # 24 kHz sample rate, 16-bit (2 bytes) mono
    pcm_bytes = len(data) - 44
    sample_rate = 24000
    bytes_per_sample = 2  # 16-bit
    duration = pcm_bytes / (sample_rate * bytes_per_sample)

    assert duration > 1.0, f"Duration too short: {duration:.2f}s (expected > 1.0s)"
    print(f"wav ok {duration:.2f}s")
    return duration

validate_streaming_wav('/tmp/vb_smoke.wav')
EOF

echo "stt round-trip:"
/usr/bin/curl -fsS "$BASE/v1/audio/transcriptions" -F "file=@/tmp/vb_smoke.wav" | tee /tmp/vb_smoke.json
python3 -c "import json;t=json.load(open('/tmp/vb_smoke.json'))['text'].lower();assert 'voicebox' in t or 'hello' in t, t;print('stt ok')"
echo "SMOKE PASS"
