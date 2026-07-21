import os
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
    assert s.max_audio_seconds == 120


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VOICEBOX_PORT", "9001")
    monkeypatch.setenv("VOICEBOX_DEVICE", "cuda")
    monkeypatch.setenv("VOICEBOX_MAX_AUDIO_SECONDS", "60")
    s = load_settings()
    assert s.port == 9001
    assert s.device == "cuda"
    assert s.max_audio_seconds == 60
