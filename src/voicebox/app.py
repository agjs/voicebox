from __future__ import annotations
import hmac
import threading
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, Response
from starlette.concurrency import run_in_threadpool
from voicebox.config import Settings
from voicebox.wav import pcm_to_wav_bytes

_SUPPORTED_FORMATS = {"wav", "pcm"}
_SUPPORTED_TRANSCRIPTION_FORMATS = {"json", "text"}


def create_app(stt, tts, settings: Settings) -> FastAPI:
    app = FastAPI(title="voicebox", version="0.2.5")
    # This CPU-oriented server favors predictable latency over throughput. Keeping
    # one model inference active prevents STT and TTS from fighting for the same cores.
    inference_lock = threading.Lock()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if settings.api_key and request.url.path != "/health":
            authorization = request.headers.get("authorization", "")
            supplied = (
                authorization.removeprefix("Bearer ")
                if authorization.startswith("Bearer ")
                else request.headers.get("x-api-key", "")
            )
            if not hmac.compare_digest(supplied, settings.api_key):
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

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
            raise HTTPException(
                status_code=413, detail=f"file exceeds {settings.max_upload_mb} MB limit"
            )
        if response_format not in _SUPPORTED_TRANSCRIPTION_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported response_format: {response_format}",
            )

        def run_transcription():
            with inference_lock:
                return stt.transcribe(audio, language=language)

        started = time.perf_counter()
        try:
            text = await run_in_threadpool(run_transcription)
        except ValueError as exc:  # AudioDecodeError / AudioTooLongError
            raise HTTPException(status_code=400, detail=str(exc))
        duration_ms = (time.perf_counter() - started) * 1000
        headers = {"Server-Timing": f"stt;dur={duration_ms:.1f}"}
        if response_format == "text":
            return PlainTextResponse(text, headers=headers)
        return JSONResponse({"text": text}, headers=headers)

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        body = await request.json()
        text = body.get("input") or ""
        response_format = (body.get("response_format") or "wav").lower()
        voice = body.get("voice")
        _model = body.get("model")
        if response_format not in _SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400, detail=f"unsupported response_format: {response_format}"
            )
        if not text.strip():
            raise HTTPException(status_code=400, detail="input is empty")
        if len(text) > settings.max_input_chars:
            raise HTTPException(
                status_code=400, detail=f"input exceeds {settings.max_input_chars} character limit"
            )

        if response_format == "wav":
            # Buffer to a COMPLETE, correctly-sized WAV. Synthesis is fast, and a
            # proper header (real RIFF/data sizes) is what strict decoders like
            # Open WebUI's Web Audio path require; a streaming placeholder-size
            # header decodes to distorted audio there.
            def synthesize_wav():
                with inference_lock:
                    return b"".join(tts.synthesize_stream(text, voice))

            started = time.perf_counter()
            pcm = await run_in_threadpool(synthesize_wav)
            duration_ms = (time.perf_counter() - started) * 1000
            return Response(
                content=pcm_to_wav_bytes(pcm, tts.sample_rate),
                media_type="audio/wav",
                headers={"Server-Timing": f"tts;dur={duration_ms:.1f}"},
            )

        # pcm: raw int16 stream for low-latency clients that play as bytes arrive.
        def gen():
            with inference_lock:
                yield from tts.synthesize_stream(text, voice)

        return StreamingResponse(
            gen(),
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": str(tts.sample_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Sample-Format": "s16le",
            },
        )

    return app
