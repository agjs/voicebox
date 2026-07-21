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

# TTS (Piper): the voice files PiperTtsEngine loads at runtime
print("Downloading Piper voice model...")
hf_hub_download(
    repo_id="rhasspy/piper-voices",
    filename=f"en/{s.piper_voice}/{s.piper_voice}.onnx"
)
hf_hub_download(
    repo_id="rhasspy/piper-voices",
    filename=f"en/{s.piper_voice}/{s.piper_voice}.onnx.json"
)

print("models cached")
