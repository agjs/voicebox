from __future__ import annotations
import io
from dataclasses import dataclass

import av
import numpy as np
from faster_whisper import WhisperModel
from voicebox.config import Settings


class AudioDecodeError(ValueError):
    pass


class AudioTooLongError(ValueError):
    pass


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: tuple[TranscriptionSegment, ...] = ()


def _decode_audio_limited(audio: bytes, max_seconds: int, sampling_rate: int = 16000) -> np.ndarray:
    """Decode any PyAV-supported container while enforcing a decoded-duration cap."""
    max_samples = max_seconds * sampling_rate
    chunks: list[np.ndarray] = []
    sample_count = 0
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)

    def append_frame(frame) -> None:
        nonlocal sample_count
        chunk = frame.to_ndarray().reshape(-1)
        sample_count += len(chunk)
        if sample_count > max_samples:
            raise AudioTooLongError(f"audio exceeds cap {max_seconds}s after decoding")
        chunks.append(chunk)

    try:
        with av.open(io.BytesIO(audio), mode="r", metadata_errors="ignore") as container:
            frames = iter(container.decode(audio=0))
            while True:
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                except av.error.InvalidDataError:
                    continue
                for resampled in resampler.resample(frame):
                    append_frame(resampled)
            for resampled in resampler.resample(None):
                append_frame(resampled)
    except AudioTooLongError:
        raise
    except Exception as exc:
        raise AudioDecodeError(str(exc)) from exc
    finally:
        del resampler

    if not chunks:
        raise AudioDecodeError("no audio decoded")
    pcm16 = np.concatenate(chunks)
    return pcm16.astype(np.float32) / 32768.0


class SttEngine:
    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.stt_model
        self.max_audio_seconds = settings.max_audio_seconds
        self.beam_size = settings.stt_beam_size
        self.vad_filter = settings.stt_vad_filter
        self.vad_parameters = {"min_silence_duration_ms": settings.stt_min_silence_ms}
        self.hotwords = settings.stt_hotwords
        compute_type = "int8" if settings.device == "cpu" else "float16"
        try:
            self.model = WhisperModel(
                settings.stt_model,
                device=settings.device,
                compute_type=compute_type,
                cpu_threads=settings.cpu_threads,
                revision=settings.stt_model_revision,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load STT model '{settings.stt_model}'. "
                f"If offline, ensure models are baked into the image / present in the HF cache. "
                f"Original error: {exc}"
            ) from exc

    def transcribe(
        self, audio: bytes, language: str = "en", *, timestamps: bool = False
    ) -> TranscriptionResult:
        # Validate language before decoding so an unsupported language fails fast
        # without spending CPU on audio decode.
        if language not in self.model.supported_languages:
            raise AudioDecodeError(
                f"language {language!r} is not supported by model; "
                f"supported: {', '.join(self.model.supported_languages)}"
            )
        try:
            decoded = _decode_audio_limited(audio, self.max_audio_seconds)
            segments_iter, info = self.model.transcribe(
                decoded,
                language=language,
                beam_size=self.beam_size,
                condition_on_previous_text=False,
                without_timestamps=not timestamps,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters if self.vad_filter else None,
                hotwords=self.hotwords,
            )
            raw_segments = list(segments_iter)
        except (AudioDecodeError, AudioTooLongError):
            raise
        except Exception as exc:  # PyAV/CT2 decode failures
            raise AudioDecodeError(str(exc)) from exc

        text = "".join(seg.text for seg in raw_segments).strip()
        if not timestamps:
            return TranscriptionResult(text=text, language=language)

        segments = tuple(
            TranscriptionSegment(
                id=index,
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text,
            )
            for index, seg in enumerate(raw_segments)
        )
        duration = getattr(info, "duration", None)
        detected = getattr(info, "language", None) or language
        return TranscriptionResult(
            text=text,
            language=detected,
            duration=float(duration) if duration is not None else None,
            segments=segments,
        )
