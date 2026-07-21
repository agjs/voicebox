from __future__ import annotations
import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


@dataclass(frozen=True)
class Settings:
    stt_model: str
    stt_model_revision: str
    stt_beam_size: int
    stt_vad_filter: bool
    stt_min_silence_ms: int
    stt_hotwords: str | None
    tts_model: str
    tts_model_revision: str
    tts_engine: str
    piper_voice: str
    piper_model_revision: str
    piper_length_scale: float
    piper_noise_scale: float
    piper_noise_w: float
    default_voice: str
    port: int
    api_key: str | None
    device: str
    cpu_threads: int
    max_audio_seconds: int
    max_upload_mb: int
    max_input_chars: int


def load_settings() -> Settings:
    settings = Settings(
        stt_model=os.getenv("VOICEBOX_STT_MODEL", "Systran/faster-distil-whisper-small.en"),
        stt_model_revision=os.getenv(
            "VOICEBOX_STT_MODEL_REVISION", "ef77d90526ccd62cde3808ee70626a01e5cf83e4"
        ),
        stt_beam_size=int(os.getenv("VOICEBOX_STT_BEAM_SIZE", "1")),
        stt_vad_filter=_env_bool("VOICEBOX_STT_VAD_FILTER", True),
        stt_min_silence_ms=int(os.getenv("VOICEBOX_STT_MIN_SILENCE_MS", "500")),
        stt_hotwords=os.getenv("VOICEBOX_STT_HOTWORDS") or None,
        tts_model=os.getenv("VOICEBOX_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX"),
        tts_model_revision=os.getenv(
            "VOICEBOX_TTS_MODEL_REVISION", "dc196c76d64fed9203906231372bcb98135815df"
        ),
        tts_engine=os.getenv("VOICEBOX_TTS_ENGINE", "piper"),
        piper_voice=os.getenv("VOICEBOX_PIPER_VOICE", "en_US-amy-medium"),
        piper_model_revision=os.getenv(
            "VOICEBOX_PIPER_MODEL_REVISION", "5b44ec7bab7c5822cfec48fbd5aa99db71a823d6"
        ),
        # Piper speaking rate: <1.0 = faster speech, >1.0 = slower. 1.0 = voice default.
        piper_length_scale=float(os.getenv("VOICEBOX_PIPER_LENGTH_SCALE", "1.0")),
        # Piper prosody randomness (0.667) and duration variability (0.8); Piper defaults.
        piper_noise_scale=float(os.getenv("VOICEBOX_PIPER_NOISE_SCALE", "0.667")),
        piper_noise_w=float(os.getenv("VOICEBOX_PIPER_NOISE_W", "0.8")),
        default_voice=os.getenv("VOICEBOX_DEFAULT_VOICE", "af_heart"),
        port=int(os.getenv("VOICEBOX_PORT", "8790")),
        api_key=os.getenv("VOICEBOX_API_KEY") or None,
        device=os.getenv("VOICEBOX_DEVICE", "cpu"),
        cpu_threads=int(os.getenv("VOICEBOX_CPU_THREADS", "4")),
        max_audio_seconds=int(os.getenv("VOICEBOX_MAX_AUDIO_SECONDS", "120")),
        max_upload_mb=int(os.getenv("VOICEBOX_MAX_UPLOAD_MB", "25")),
        max_input_chars=int(os.getenv("VOICEBOX_MAX_INPUT_CHARS", "4000")),
    )
    if settings.tts_engine not in {"piper", "kokoro"}:
        raise ValueError("VOICEBOX_TTS_ENGINE must be 'piper' or 'kokoro'")
    if settings.device not in {"cpu", "cuda"}:
        raise ValueError("VOICEBOX_DEVICE must be 'cpu' or 'cuda'")
    if not 1 <= settings.stt_beam_size <= 10:
        raise ValueError("VOICEBOX_STT_BEAM_SIZE must be between 1 and 10")
    if settings.cpu_threads < 1:
        raise ValueError("VOICEBOX_CPU_THREADS must be at least 1")
    if settings.stt_min_silence_ms < 0:
        raise ValueError("VOICEBOX_STT_MIN_SILENCE_MS must not be negative")
    if settings.max_audio_seconds < 1 or settings.max_upload_mb < 1:
        raise ValueError("audio limits must be positive")
    return settings
