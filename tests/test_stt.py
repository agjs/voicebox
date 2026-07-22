import pytest
import numpy as np
from types import SimpleNamespace
from voicebox.config import load_settings
from voicebox.stt import (
    AudioDecodeError,
    AudioTooLongError,
    SttEngine,
    _decode_audio_limited,
)


@pytest.fixture(scope="module")
def engine():
    return SttEngine(load_settings())


@pytest.mark.model
def test_transcribes_known_audio(engine):
    with open("tests/fixtures/hello.wav", "rb") as f:
        text = engine.transcribe(f.read()).text
    lowered = text.lower()
    assert "hello world" in lowered
    # Whisper may transcribe "voicebox" as "voice box" (two words)
    assert "voicebox" in lowered or "voice box" in lowered


@pytest.mark.model
def test_rejects_undecodable_bytes(engine):
    with pytest.raises(AudioDecodeError):
        engine.transcribe(b"not audio at all")


@pytest.mark.model
def test_rejects_too_long_audio(engine, monkeypatch):
    monkeypatch.setattr(engine, "max_audio_seconds", 0)
    with open("tests/fixtures/hello.wav", "rb") as f:
        with pytest.raises(AudioTooLongError):
            engine.transcribe(f.read())


def test_init_with_bad_model_raises_runtime_error(monkeypatch, settings_factory):
    from voicebox.stt import SttEngine

    def mock_whisper_model(*args, **kwargs):
        raise ValueError("model not found")

    monkeypatch.setattr("voicebox.stt.WhisperModel", mock_whisper_model)
    settings = settings_factory(stt_model="invalid/model")

    with pytest.raises(RuntimeError) as exc_info:
        SttEngine(settings)
    assert "Failed to load STT model" in str(exc_info.value)
    assert "invalid/model" in str(exc_info.value)


def test_decoded_duration_limit_applies_to_audio_content():
    with open("tests/fixtures/hello.wav", "rb") as f:
        with pytest.raises(AudioTooLongError):
            _decode_audio_limited(f.read(), max_seconds=1)


def test_transcription_uses_low_latency_options(monkeypatch, settings_factory):
    captured = {}

    class FakeWhisperModel:
        supported_languages = ["en"]

        def __init__(self, *args, **kwargs):
            captured["init"] = kwargs

        def transcribe(self, audio, **kwargs):
            captured["transcribe"] = kwargs
            assert isinstance(audio, np.ndarray)
            return iter([SimpleNamespace(text=" hello")]), None

    monkeypatch.setattr("voicebox.stt.WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(
        "voicebox.stt._decode_audio_limited",
        lambda audio, max_seconds: np.zeros(1600, dtype=np.float32),
    )
    settings = settings_factory(stt_beam_size=1, stt_hotwords="Voicebox, ClickHouse")
    engine = SttEngine(settings)
    assert engine.transcribe(b"audio").text == "hello"
    assert captured["init"]["cpu_threads"] == 4
    assert captured["init"]["revision"] == settings.stt_model_revision
    assert captured["transcribe"]["beam_size"] == 1
    assert captured["transcribe"]["condition_on_previous_text"] is False
    assert captured["transcribe"]["without_timestamps"] is True
    assert captured["transcribe"]["hotwords"] == "Voicebox, ClickHouse"


def test_transcription_verbose_enables_timestamps(monkeypatch, settings_factory):
    captured = {}

    class FakeWhisperModel:
        supported_languages = ["en"]

        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            captured["transcribe"] = kwargs
            segments = [
                SimpleNamespace(text=" hello", start=0.0, end=0.5),
                SimpleNamespace(text=" world", start=0.5, end=1.0),
            ]
            info = SimpleNamespace(language="en", duration=1.0)
            return iter(segments), info

    monkeypatch.setattr("voicebox.stt.WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(
        "voicebox.stt._decode_audio_limited",
        lambda audio, max_seconds: np.zeros(1600, dtype=np.float32),
    )
    engine = SttEngine(settings_factory())
    result = engine.transcribe(b"audio", timestamps=True)
    assert captured["transcribe"]["without_timestamps"] is False
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.duration == 1.0
    assert len(result.segments) == 2
    assert result.segments[0].start == 0.0
    assert result.segments[1].text == " world"
