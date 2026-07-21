#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://localhost:8790}"

echo "health:"; /usr/bin/curl -fsS "$BASE/health"; echo

echo "tts -> /tmp/vb_smoke.wav"
/usr/bin/curl -fsS "$BASE/v1/audio/speech" -H "Content-Type: application/json" \
  --output /tmp/vb_smoke.wav \
  -d '{"model":"m","input":"Hello from voicebox. Streaming works.","response_format":"wav"}'

# Fix WAV headers (TTS uses placeholder sizes for streaming)
python3 << 'EOF'
import struct
import wave

def fix_wav_headers(filename: str) -> None:
    """Fix WAV file with placeholder RIFF/data sizes for proper codec parsing."""
    with open(filename, 'rb') as f:
        data = f.read()

    # Skip 44-byte header and get raw PCM
    pcm_data = data[44:]

    # Reconstruct with correct sizes
    byte_rate = 24000 * 1 * 16 // 8
    block_align = 1 * 16 // 8
    data_len = len(pcm_data)
    riff_len = 36 + data_len

    fixed_wav = (
        b"RIFF" + struct.pack("<I", riff_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, byte_rate, block_align, 16)
        + b"data" + struct.pack("<I", data_len) + pcm_data
    )

    with open(filename, 'wb') as f:
        f.write(fixed_wav)

fix_wav_headers('/tmp/vb_smoke.wav')
w = wave.open('/tmp/vb_smoke.wav', 'rb')
duration = w.getnframes() / w.getframerate()
print(f'wav ok {duration:.2f}s')
w.close()
EOF

echo "stt round-trip:"
/usr/bin/curl -fsS "$BASE/v1/audio/transcriptions" -F "file=@/tmp/vb_smoke.wav" | tee /tmp/vb_smoke.json
python3 -c "import json;t=json.load(open('/tmp/vb_smoke.json'))['text'].lower();assert 'voicebox' in t or 'hello' in t, t;print('stt ok')"
echo "SMOKE PASS"
