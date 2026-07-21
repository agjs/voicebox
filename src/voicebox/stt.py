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
    """Best-effort duration for WAV; returns None for other containers or streaming WAVs.

    Streaming WAVs have placeholder sizes (0xFFFFFFFF) in the RIFF/data chunks,
    which cause Python's wave module to miscompute duration. Detect and skip them.
    """
    try:
        # Check for streaming WAV with placeholder sizes (0xFFFFFFFF)
        if len(audio) >= 44:
            # RIFF size at bytes 4-7, data size at bytes 40-43
            riff_size = int.from_bytes(audio[4:8], "little")
            data_size = int.from_bytes(audio[40:44], "little")
            if riff_size == 0xFFFFFFFF or data_size == 0xFFFFFFFF:
                # Streaming WAV with placeholders; can't determine duration
                return None
        with wave.open(io.BytesIO(audio), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


class SttEngine:
    def __init__(self, settings: Settings) -> None:
        self.max_audio_seconds = settings.max_audio_seconds
        compute_type = "int8" if settings.device == "cpu" else "float16"
        try:
            self.model = WhisperModel(
                settings.stt_model,
                device=settings.device,
                compute_type=compute_type,
                cpu_threads=settings.cpu_threads,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load STT model '{settings.stt_model}'. "
                f"If offline, ensure models are baked into the image / present in the HF cache. "
                f"Original error: {exc}"
            ) from exc

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
