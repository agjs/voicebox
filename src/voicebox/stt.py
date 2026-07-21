from __future__ import annotations
import io
import wave
from faster_whisper import WhisperModel
from voicebox.config import Settings


class AudioDecodeError(ValueError):
    pass


class AudioTooLongError(ValueError):
    pass


def _duration_seconds(audio: bytes) -> float | None:
    """Best-effort duration for WAV; returns None for other containers."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


class SttEngine:
    def __init__(self, settings: Settings) -> None:
        self.max_audio_seconds = settings.max_audio_seconds
        compute_type = "int8" if settings.device == "cpu" else "float16"
        self.model = WhisperModel(
            settings.stt_model,
            device=settings.device,
            compute_type=compute_type,
            cpu_threads=settings.cpu_threads,
        )

    def transcribe(self, audio: bytes) -> str:
        dur = _duration_seconds(audio)
        if dur is not None and dur > self.max_audio_seconds:
            raise AudioTooLongError(f"audio {dur:.1f}s exceeds cap {self.max_audio_seconds}s")
        try:
            segments, _info = self.model.transcribe(io.BytesIO(audio), language="en")
            parts = [seg.text for seg in segments]
        except Exception as exc:  # PyAV/CT2 decode failures
            raise AudioDecodeError(str(exc)) from exc
        text = "".join(parts).strip()
        if not text and dur is None:
            # Non-WAV that produced nothing usable is treated as undecodable.
            raise AudioDecodeError("no audio decoded")
        return text
