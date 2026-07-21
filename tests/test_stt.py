import pytest
from voicebox.config import load_settings
from voicebox.stt import SttEngine, AudioTooLongError, AudioDecodeError


@pytest.fixture(scope="module")
def engine():
    return SttEngine(load_settings())


def test_transcribes_known_audio(engine):
    with open("tests/fixtures/hello.wav", "rb") as f:
        text = engine.transcribe(f.read())
    lowered = text.lower()
    assert "hello world" in lowered
    # Whisper may transcribe "voicebox" as "voice box" (two words)
    assert "voicebox" in lowered or "voice box" in lowered


def test_rejects_undecodable_bytes(engine):
    with pytest.raises(AudioDecodeError):
        engine.transcribe(b"not audio at all")


def test_rejects_too_long_audio(engine, monkeypatch):
    monkeypatch.setattr(engine, "max_audio_seconds", 0)
    with open("tests/fixtures/hello.wav", "rb") as f:
        with pytest.raises(AudioTooLongError):
            engine.transcribe(f.read())
