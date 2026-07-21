from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    stt_model: str
    tts_model: str
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
        default_voice=os.getenv("VOICEBOX_DEFAULT_VOICE", "af_heart"),
        port=int(os.getenv("VOICEBOX_PORT", "8790")),
        device=os.getenv("VOICEBOX_DEVICE", "cpu"),
        cpu_threads=int(os.getenv("VOICEBOX_CPU_THREADS", "0")),
        max_audio_seconds=int(os.getenv("VOICEBOX_MAX_AUDIO_SECONDS", "120")),
        max_upload_mb=int(os.getenv("VOICEBOX_MAX_UPLOAD_MB", "25")),
        max_input_chars=int(os.getenv("VOICEBOX_MAX_INPUT_CHARS", "4000")),
    )
