import os
import pytest
from voicebox.config import load_settings, Settings, is_loopback_bind, validate_bind_auth


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
    assert s.bind_address == "127.0.0.1"
    assert s.allow_insecure_bind is False
    assert s.device == "cpu"
    assert s.cpu_threads == 4
    assert s.stt_beam_size == 1
    assert s.stt_vad_filter is True
    assert s.max_audio_seconds == 120


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VOICEBOX_PORT", "9001")
    monkeypatch.setenv("VOICEBOX_DEVICE", "cuda")
    monkeypatch.setenv("VOICEBOX_MAX_AUDIO_SECONDS", "60")
    monkeypatch.setenv("VOICEBOX_BIND_ADDRESS", "0.0.0.0")
    monkeypatch.setenv("VOICEBOX_ALLOW_INSECURE_BIND", "true")
    s = load_settings()
    assert s.port == 9001
    assert s.device == "cuda"
    assert s.max_audio_seconds == 60
    assert s.bind_address == "0.0.0.0"
    assert s.allow_insecure_bind is True


def test_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("VOICEBOX_STT_VAD_FILTER", "sometimes")
    with pytest.raises(ValueError, match="must be true or false"):
        load_settings()


def test_rejects_invalid_engine(monkeypatch):
    monkeypatch.setenv("VOICEBOX_TTS_ENGINE", "slowbox")
    with pytest.raises(ValueError, match="TTS_ENGINE"):
        load_settings()


def test_is_loopback_bind():
    assert is_loopback_bind("127.0.0.1")
    assert is_loopback_bind("::1")
    assert is_loopback_bind("localhost")
    assert not is_loopback_bind("0.0.0.0")
    assert not is_loopback_bind("192.168.1.10")


def test_validate_bind_auth_allows_loopback_without_key():
    validate_bind_auth("127.0.0.1", api_key=None, allow_insecure_bind=False)


def test_validate_bind_auth_requires_key_for_non_loopback():
    with pytest.raises(ValueError, match="VOICEBOX_API_KEY"):
        validate_bind_auth("0.0.0.0", api_key=None, allow_insecure_bind=False)
    with pytest.raises(ValueError, match="VOICEBOX_API_KEY"):
        validate_bind_auth("192.168.1.10", api_key=None, allow_insecure_bind=False)


def test_validate_bind_auth_allows_non_loopback_with_key():
    validate_bind_auth("0.0.0.0", api_key="secret", allow_insecure_bind=False)


def test_validate_bind_auth_allows_insecure_opt_in():
    validate_bind_auth("0.0.0.0", api_key=None, allow_insecure_bind=True)


def test_load_settings_rejects_insecure_non_loopback(monkeypatch):
    monkeypatch.setenv("VOICEBOX_BIND_ADDRESS", "0.0.0.0")
    monkeypatch.delenv("VOICEBOX_API_KEY", raising=False)
    monkeypatch.delenv("VOICEBOX_ALLOW_INSECURE_BIND", raising=False)
    with pytest.raises(ValueError, match="VOICEBOX_API_KEY"):
        load_settings()
