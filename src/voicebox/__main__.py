from __future__ import annotations
import uvicorn
from voicebox.config import load_settings
from voicebox.stt import SttEngine
from voicebox.tts import TtsEngine
from voicebox.app import create_app


def main() -> None:
    settings = load_settings()
    application = create_app(SttEngine(settings), TtsEngine(settings), settings)
    uvicorn.run(application, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
