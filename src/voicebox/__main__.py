from __future__ import annotations
import uvicorn
from voicebox.config import load_settings
from voicebox.stt import SttEngine
from voicebox.tts import TtsEngine
from voicebox.tts_piper import PiperTtsEngine
from voicebox.app import create_app


def main() -> None:
    settings = load_settings()
    # Select TTS engine based on configuration
    if settings.tts_engine == "piper":
        tts = PiperTtsEngine(settings)
    else:
        tts = TtsEngine(settings)
    application = create_app(SttEngine(settings), tts, settings)
    uvicorn.run(application, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
