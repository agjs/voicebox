from __future__ import annotations
from typing import Iterator
import io
import json
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from voicebox.config import Settings
from voicebox.tts import split_sentences


class PiperTtsEngine:
    def __init__(self, settings: Settings) -> None:
        self.default_voice = settings.default_voice
        self.piper_voice_name = settings.piper_voice

        try:
            from piper import PiperVoice

            # Resolve voice model path from HF cache (respects HF_HUB_OFFLINE)
            voice_onnx = hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=f"en/{settings.piper_voice}/{settings.piper_voice}.onnx"
            )
            voice_json = hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=f"en/{settings.piper_voice}/{settings.piper_voice}.onnx.json"
            )

            # Load the voice
            self.piper_voice = PiperVoice.load(voice_onnx)

            # Read sample rate from the voice config JSON
            with open(voice_json, "r") as f:
                config = json.load(f)
            self.sample_rate = config.get("sample_rate", 22050)

        except ImportError as exc:
            raise RuntimeError(
                "piper-tts package not installed. "
                "Install it with: pip install piper-tts"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Piper voice '{settings.piper_voice}'. "
                f"If offline, ensure models are baked into the image / present in the HF cache. "
                f"Original error: {exc}"
            ) from exc

    def synthesize_stream(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("input text is empty")

        # Piper doesn't use per-sentence voice selection like Kokoro,
        # but we accept the parameter for interface compatibility
        for sentence in sentences:
            audio_bytes = io.BytesIO()
            self.piper_voice.synthesize(sentence, audio_bytes)
            audio_bytes.seek(0)
            pcm_data = audio_bytes.read()
            yield pcm_data
