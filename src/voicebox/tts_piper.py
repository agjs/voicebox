from __future__ import annotations
from typing import Iterator
import json
from huggingface_hub import hf_hub_download
from voicebox.config import Settings
from voicebox.tts import split_sentences

_PIPER_REPO = "rhasspy/piper-voices"


def piper_voice_relpaths(voice_name: str) -> tuple[str, str]:
    """Map a Piper voice name to its files in the rhasspy/piper-voices repo.

    e.g. 'en_US-lessac-high' -> ('en/en_US/lessac/high/en_US-lessac-high.onnx',
                                 'en/en_US/lessac/high/en_US-lessac-high.onnx.json')
    """
    lang, name, quality = voice_name.split("-", 2)
    family = lang.split("_")[0]
    base = f"{family}/{lang}/{name}/{quality}/{voice_name}"
    return base + ".onnx", base + ".onnx.json"


class PiperTtsEngine:
    """Fast CPU TTS via Piper. Same interface as TtsEngine (Kokoro)."""

    def __init__(self, settings: Settings) -> None:
        self.piper_voice_name = settings.piper_voice
        onnx_rel, json_rel = piper_voice_relpaths(settings.piper_voice)
        try:
            from piper import PiperVoice

            # Resolve baked voice files from the HF cache (respects HF_HUB_OFFLINE=1).
            voice_onnx = hf_hub_download(repo_id=_PIPER_REPO, filename=onnx_rel)
            voice_json = hf_hub_download(repo_id=_PIPER_REPO, filename=json_rel)
            self.piper_voice = PiperVoice.load(voice_onnx, config_path=voice_json)
            with open(voice_json) as f:
                cfg = json.load(f)
            self.sample_rate = int(cfg.get("audio", {}).get("sample_rate", 22050))
        except ImportError as exc:
            raise RuntimeError(
                "piper-tts is not installed. Install with: pip install piper-tts"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Piper voice '{settings.piper_voice}'. If offline, "
                f"ensure the voice is baked into the image / present in the HF cache. "
                f"Original error: {exc}"
            ) from exc

    def synthesize_stream(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("input text is empty")
        # One int16-LE mono PCM chunk per sentence (playback can start on sentence 1).
        for sentence in sentences:
            pcm = bytearray()
            for chunk in self.piper_voice.synthesize(sentence):
                pcm += chunk.audio_int16_bytes
            yield bytes(pcm)
