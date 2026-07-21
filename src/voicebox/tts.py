from __future__ import annotations
import re
from typing import Iterator
import numpy as np
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
from voicebox.config import Settings

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def split_sentences(text: str) -> list[str]:
    parts = [m.group().strip() for m in _SENTENCE_RE.finditer(text)]
    return [p for p in parts if p]


class TtsEngine:
    sample_rate = 24000

    def __init__(self, settings: Settings) -> None:
        self.default_voice = settings.default_voice
        # Download model and voices files from HuggingFace
        onnx_path = hf_hub_download(settings.tts_model, "model.onnx")
        voices_path = hf_hub_download(settings.tts_model, "voices.bin")
        self.kokoro = Kokoro(onnx_path, voices_path)

    def synthesize_stream(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("input text is empty")
        v = voice or self.default_voice
        for sentence in sentences:
            samples, _sr = self.kokoro.create(sentence, voice=v, speed=1.0)
            pcm = np.clip(samples, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype("<i2")
            yield pcm16.tobytes()
