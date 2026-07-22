from __future__ import annotations
import re
from typing import Iterator
import numpy as np
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
from voicebox.config import Settings

_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“‘])')


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for paragraph in re.split(r"\n+", text):
        parts.extend(_SENTENCE_BOUNDARY_RE.split(paragraph.strip()))
    return [part.strip() for part in parts if part.strip()]


class TtsEngine:
    sample_rate = 24000

    def __init__(self, settings: Settings) -> None:
        self.default_voice = settings.default_voice
        try:
            # Download model and voices files from HuggingFace
            onnx_path = hf_hub_download(
                settings.tts_model, "model.onnx", revision=settings.tts_model_revision
            )
            voices_path = hf_hub_download(
                settings.tts_model, "voices.bin", revision=settings.tts_model_revision
            )
            self.kokoro = Kokoro(onnx_path, voices_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TTS model '{settings.tts_model}'. "
                f"If offline, ensure models are baked into the image / present in the HF cache. "
                f"Original error: {exc}"
            ) from exc

    def list_voice_ids(self) -> list[str]:
        return [self.default_voice]

    def sample_rate_for(self, voice: str | None = None) -> int:
        return self.sample_rate

    def synthesize_stream(
        self, text: str, voice: str | None = None, speed: float = 1.0
    ) -> Iterator[bytes]:
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("input text is empty")
        v = voice or self.default_voice
        for sentence in sentences:
            samples, _sr = self.kokoro.create(sentence, voice=v, speed=speed)
            pcm = np.clip(samples, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype("<i2")
            yield pcm16.tobytes()
