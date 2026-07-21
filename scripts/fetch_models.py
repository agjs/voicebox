"""Pre-download STT + TTS models into the image so runtime is offline."""
from voicebox.config import load_settings
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download

s = load_settings()

# STT: triggers HF download into the image's HF cache
print("Downloading STT model...")
WhisperModel(s.stt_model, device="cpu", compute_type="int8")

# TTS (Kokoro): the two files TtsEngine loads at runtime
print("Downloading TTS model files...")
hf_hub_download(s.tts_model, "model.onnx")
hf_hub_download(s.tts_model, "voices.bin")

# TTS (Piper): bake the configured voice plus a couple of good medium voices so
# they can be selected at runtime (VOICEBOX_PIPER_VOICE) with no rebuild.
from voicebox.tts_piper import piper_voice_relpaths

_piper_voices = {s.piper_voice, "en_US-amy-medium", "en_US-lessac-medium"}
for _voice in sorted(_piper_voices):
    print(f"Downloading Piper voice {_voice}...")
    _onnx_rel, _json_rel = piper_voice_relpaths(_voice)
    hf_hub_download(repo_id="rhasspy/piper-voices", filename=_onnx_rel)
    hf_hub_download(repo_id="rhasspy/piper-voices", filename=_json_rel)

print("models cached")
