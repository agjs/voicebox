from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    stt_model: str
    tts_model: str
    tts_engine: str
    piper_voice: str
    piper_length_scale: float
    piper_noise_scale: float
    piper_noise_w: float
    default_voice: str
    port: int
    device: str
    cpu_threads: int
    max_audio_seconds: int
    max_upload_mb: int
    max_input_chars: int


def load_settings() -> Settings:
    return Settings(
        stt_model=os.getenv("VOICEBOX_STT_MODEL", "Systran/faster-distil-whisper-small.en"),
        tts_model=os.getenv("VOICEBOX_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX"),
        tts_engine=os.getenv("VOICEBOX_TTS_ENGINE", "kokoro"),
        piper_voice=os.getenv("VOICEBOX_PIPER_VOICE", "en_US-bryce-medium"),
        # Piper speaking rate: <1.0 = faster speech, >1.0 = slower. 1.0 = voice default.
        piper_length_scale=float(os.getenv("VOICEBOX_PIPER_LENGTH_SCALE", "1.0")),
        # Piper prosody randomness (0.667) and duration variability (0.8) — Piper defaults.
        piper_noise_scale=float(os.getenv("VOICEBOX_PIPER_NOISE_SCALE", "0.667")),
        piper_noise_w=float(os.getenv("VOICEBOX_PIPER_NOISE_W", "0.8")),
        default_voice=os.getenv("VOICEBOX_DEFAULT_VOICE", "af_heart"),
        port=int(os.getenv("VOICEBOX_PORT", "8790")),
        device=os.getenv("VOICEBOX_DEVICE", "cpu"),
        cpu_threads=int(os.getenv("VOICEBOX_CPU_THREADS", "0")),
        max_audio_seconds=int(os.getenv("VOICEBOX_MAX_AUDIO_SECONDS", "120")),
        max_upload_mb=int(os.getenv("VOICEBOX_MAX_UPLOAD_MB", "25")),
        max_input_chars=int(os.getenv("VOICEBOX_MAX_INPUT_CHARS", "4000")),
    )
