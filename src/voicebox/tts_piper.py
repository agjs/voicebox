from __future__ import annotations
from typing import Iterator
import json
from huggingface_hub import hf_hub_download
from voicebox.config import Settings
from voicebox.tts import split_sentences

_PIPER_REPO = "rhasspy/piper-voices"
BAKED_PIPER_VOICES = (
    "en_US-amy-medium",
    "en_US-bryce-medium",
    "en_US-lessac-medium",
)


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
        self.default_voice = settings.piper_voice
        self._base_length_scale = settings.piper_length_scale
        self._noise_scale = settings.piper_noise_scale
        self._noise_w = settings.piper_noise_w
        self._voices: dict = {}
        self._sample_rates: dict[str, int] = {}
        voice_names = set(BAKED_PIPER_VOICES) | {settings.piper_voice}
        try:
            from piper import PiperVoice

            for voice_name in sorted(voice_names):
                onnx_rel, json_rel = piper_voice_relpaths(voice_name)
                # Resolve baked voice files from the HF cache (respects HF_HUB_OFFLINE=1).
                voice_onnx = hf_hub_download(
                    repo_id=_PIPER_REPO,
                    filename=onnx_rel,
                    revision=settings.piper_model_revision,
                )
                voice_json = hf_hub_download(
                    repo_id=_PIPER_REPO,
                    filename=json_rel,
                    revision=settings.piper_model_revision,
                )
                self._voices[voice_name] = PiperVoice.load(voice_onnx, config_path=voice_json)
                with open(voice_json) as f:
                    cfg = json.load(f)
                self._sample_rates[voice_name] = int(cfg.get("audio", {}).get("sample_rate", 22050))
        except ImportError as exc:
            raise RuntimeError(
                "piper-tts is not installed. Install with: pip install piper-tts"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Piper voice(s) {sorted(voice_names)}. If offline, "
                f"ensure the voices are baked into the image / present in the HF cache. "
                f"Original error: {exc}"
            ) from exc
        self.sample_rate = self._sample_rates[self.default_voice]

    def list_voice_ids(self) -> list[str]:
        return sorted(self._voices)

    def sample_rate_for(self, voice: str | None = None) -> int:
        return self._sample_rates[self._resolve_voice(voice)]

    def _resolve_voice(self, voice: str | None) -> str:
        name = voice or self.default_voice
        if name not in self._voices:
            supported = ", ".join(self.list_voice_ids())
            raise ValueError(f"unsupported voice {name!r}; supported: {supported}")
        return name

    def synthesize_stream(
        self, text: str, voice: str | None = None, speed: float = 1.0
    ) -> Iterator[bytes]:
        from piper import SynthesisConfig

        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("input text is empty")
        voice_name = self._resolve_voice(voice)
        piper_voice = self._voices[voice_name]
        syn_config = SynthesisConfig(
            length_scale=self._base_length_scale / speed,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w,
        )
        # Forward Piper's int16-LE chunks immediately for true low-latency PCM playback.
        for sentence in sentences:
            for chunk in piper_voice.synthesize(sentence, syn_config=syn_config):
                yield chunk.audio_int16_bytes
