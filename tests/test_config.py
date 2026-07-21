import os
import pytest
from voicebox.config import load_settings, Settings


def test_defaults_when_no_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VOICEBOX_"):
            monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert isinstance(s, Settings)
    assert s.stt_model == "Systran/faster-distil-whisper-small.en"
    assert s.tts_model == "speaches-ai/Kokoro-82M-v1.0-ONNX"
    assert s.default_voice == "af_heart"
    assert s.port == 8790
    assert s.device == "cpu"
    assert s.cpu_threads == 4
    assert s.stt_beam_size == 1
    assert s.stt_vad_filter is True
    assert s.max_audio_seconds == 120


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VOICEBOX_PORT", "9001")
    monkeypatch.setenv("VOICEBOX_DEVICE", "cuda")
    monkeypatch.setenv("VOICEBOX_MAX_AUDIO_SECONDS", "60")
    s = load_settings()
    assert s.port == 9001
    assert s.device == "cuda"
    assert s.max_audio_seconds == 60


def test_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("VOICEBOX_STT_VAD_FILTER", "sometimes")
    with pytest.raises(ValueError, match="must be true or false"):
        load_settings()


def test_rejects_invalid_engine(monkeypatch):
    monkeypatch.setenv("VOICEBOX_TTS_ENGINE", "slowbox")
    with pytest.raises(ValueError, match="TTS_ENGINE"):
        load_settings()
