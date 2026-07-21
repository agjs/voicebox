from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from voicebox.config import Settings, load_settings
from voicebox.wav import pcm_to_wav_bytes

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
        max_bytes = settings.max_upload_mb * 1024 * 1024
        audio = await file.read(max_bytes + 1)
        if len(audio) > max_bytes:
            raise HTTPException(status_code=413,
                                detail=f"file exceeds {settings.max_upload_mb} MB limit")
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
        _model = body.get("model")
        if response_format not in _SUPPORTED_FORMATS:
            raise HTTPException(status_code=400,
                                detail=f"unsupported response_format: {response_format}")
        if not text.strip():
            raise HTTPException(status_code=400, detail="input is empty")
        if len(text) > settings.max_input_chars:
            raise HTTPException(status_code=400,
                                detail=f"input exceeds {settings.max_input_chars} character limit")

        if response_format == "wav":
            # Buffer to a COMPLETE, correctly-sized WAV. Synthesis is fast, and a
            # proper header (real RIFF/data sizes) is what strict decoders like
            # Open WebUI's Web Audio path require — a streaming placeholder-size
            # header decodes to distorted audio there.
            pcm = b"".join(tts.synthesize_stream(text, voice))
            return Response(content=pcm_to_wav_bytes(pcm, tts.sample_rate),
                            media_type="audio/wav")

        # pcm: raw int16 stream for low-latency clients that play as bytes arrive.
        def gen():
            for chunk in tts.synthesize_stream(text, voice):
                yield chunk

        return StreamingResponse(gen(), media_type="audio/pcm")

    return app
