#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://localhost:8790}"

echo "health:"; /usr/bin/curl -fsS "$BASE/health"; echo

echo "tts -> /tmp/vb_smoke.wav"
/usr/bin/curl -fsS "$BASE/v1/audio/speech" -H "Content-Type: application/json" \
  --output /tmp/vb_smoke.wav \
  -d '{"model":"m","input":"Hello from voicebox. Streaming works.","response_format":"wav"}'

# Validate a complete WAV (real RIFF/data sizes; sample rate from header).
python3 << 'EOF'
import struct

def validate_complete_wav(filename: str) -> float:
    with open(filename, "rb") as f:
        data = f.read()

    assert len(data) >= 44, f"WAV too short: {len(data)} bytes"
    assert data[0:4] == b"RIFF", f"Invalid RIFF magic: {data[0:4]}"
    assert data[8:12] == b"WAVE", f"Invalid WAVE magic: {data[8:12]}"

    # RIFF size at offset 4 is file_size - 8 for a complete WAV.
    riff_size = struct.unpack_from("<I", data, 4)[0]
    assert riff_size == len(data) - 8, f"RIFF size {riff_size} != {len(data) - 8}"

    # Walk chunks to find fmt and data (header may include extra chunks).
    offset = 12
    sample_rate = None
    channels = None
    bits = None
    data_size = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        payload = offset + 8
        if chunk_id == b"fmt ":
            channels = struct.unpack_from("<H", data, payload + 2)[0]
            sample_rate = struct.unpack_from("<I", data, payload + 4)[0]
            bits = struct.unpack_from("<H", data, payload + 14)[0]
        elif chunk_id == b"data":
            data_size = chunk_size
            break
        offset = payload + chunk_size + (chunk_size % 2)

    assert sample_rate and channels and bits and data_size is not None, "missing fmt/data"
    bytes_per_sample = bits // 8
    duration = data_size / (sample_rate * channels * bytes_per_sample)
    assert duration > 1.0, f"Duration too short: {duration:.2f}s (expected > 1.0s)"
    print(f"wav ok {duration:.2f}s @ {sample_rate} Hz")
    return duration

validate_complete_wav("/tmp/vb_smoke.wav")
EOF

echo "stt round-trip:"
/usr/bin/curl -fsS "$BASE/v1/audio/transcriptions" -F "file=@/tmp/vb_smoke.wav" | tee /tmp/vb_smoke.json
python3 -c "import json;t=json.load(open('/tmp/vb_smoke.json'))['text'].lower();assert 'voicebox' in t or 'hello' in t, t;print('stt ok')"
echo "SMOKE PASS"
