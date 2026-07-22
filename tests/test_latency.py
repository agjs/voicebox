import os
import time
import wave
import pytest
from voicebox.config import load_settings
from voicebox.stt import SttEngine
from voicebox.tts import TtsEngine
from voicebox.tts_piper import PiperTtsEngine

pytestmark = [
    pytest.mark.model,
    pytest.mark.latency,
    pytest.mark.skipif(
        os.getenv("VOICEBOX_SKIP_LATENCY") == "1",
        reason="latency guard skipped on this host",
    ),
]


def _wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def test_stt_faster_than_half_realtime():
    engine = SttEngine(load_settings())
    dur = _wav_seconds("tests/fixtures/hello.wav")
    with open("tests/fixtures/hello.wav", "rb") as f:
        data = f.read()
    engine.transcribe(data)  # warm
    t0 = time.perf_counter()
    engine.transcribe(data)
    rtf = (time.perf_counter() - t0) / dur
    assert rtf < 0.5, f"STT RTF {rtf:.2f} exceeds 0.5x"


def test_tts_kokoro_first_chunk_under_2s():
    engine = TtsEngine(load_settings())
    gen = engine.synthesize_stream("This is the first sentence. And a second one.")
    t0 = time.perf_counter()
    next(gen)  # first sentence chunk
    first = time.perf_counter() - t0
    assert first < 2.0, f"Kokoro first chunk {first:.2f}s exceeds 2s"


def test_tts_piper_first_chunk_under_2s(settings_factory):
    settings = settings_factory(tts_engine="piper")
    try:
        engine = PiperTtsEngine(settings)
    except RuntimeError as exc:
        if "Failed to load Piper voice" in str(exc):
            pytest.skip(f"Piper voice not available: {exc}")
        raise
    gen = engine.synthesize_stream("This is the first sentence. And a second one.")
    t0 = time.perf_counter()
    next(gen)
    first = time.perf_counter() - t0
    assert first < 2.0, f"Piper first chunk {first:.2f}s exceeds 2s"
