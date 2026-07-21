from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from voicebox.config import Settings, load_settings
from voicebox.wav import wav_header

_SUPPORTED_FORMATS = {"wav", "pcm"}


def create_app(stt, tts, settings: Settings) -> FastAPI:
    app = FastAPI(title="voicebox", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": True}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default=settings.stt_model),
        language: str = Form(default="en"),
        response_format: str = Form(default="json"),
    ):
        audio = await file.read()
        try:
            text = stt.transcribe(audio)
        except ValueError as exc:  # AudioDecodeError / AudioTooLongError
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse({"text": text})

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        body = await request.json()
        text = (body.get("input") or "")
        response_format = (body.get("response_format") or "wav").lower()
        voice = body.get("voice")
        if response_format not in _SUPPORTED_FORMATS:
            raise HTTPException(status_code=400,
                                detail=f"unsupported response_format: {response_format}")
        if not text.strip():
            raise HTTPException(status_code=400, detail="input is empty")

        def gen():
            if response_format == "wav":
                yield wav_header(tts.sample_rate)
            for chunk in tts.synthesize_stream(text, voice):
                yield chunk

        media = "audio/wav" if response_format == "wav" else "audio/pcm"
        return StreamingResponse(gen(), media_type=media)

    return app
