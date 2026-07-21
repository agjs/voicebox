import io
import wave
import pytest
from fastapi.testclient import TestClient
from voicebox.app import create_app
from voicebox.stt import AudioDecodeError
from voicebox.wav import pcm_to_wav_bytes


class FakeStt:
    def transcribe(self, audio: bytes) -> str:
        if audio == b"bad":
            raise AudioDecodeError("bad")
        return "hello world"


class FakeTts:
    sample_rate = 24000

    def synthesize_stream(self, text, voice=None):
        if not text.strip():
            raise ValueError("empty")
        yield (b"\x01\x00" * 100)
        yield (b"\x02\x00" * 100)


@pytest.fixture
def client():
    from voicebox.config import load_settings
    return TestClient(create_app(FakeStt(), FakeTts(), load_settings()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": True}


def test_transcription_ok(client):
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"anything", "audio/wav")},
                    data={"model": "whatever"})
    assert r.status_code == 200
    assert r.json() == {"text": "hello world"}


def test_transcription_bad_audio_400(client):
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"bad", "audio/wav")})
    assert r.status_code == 400


def test_speech_wav_streams_valid_container(client):
    r = client.post("/v1/audio/speech",
                    json={"model": "m", "input": "hi there", "response_format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    # Body should be a parseable WAV once fully read (via helper for full-buffer).
    body = r.content
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"


def test_speech_empty_input_400(client):
    r = client.post("/v1/audio/speech", json={"model": "m", "input": "   "})
    assert r.status_code == 400


def test_speech_bad_format_400(client):
    r = client.post("/v1/audio/speech",
                    json={"model": "m", "input": "hi", "response_format": "ogg"})
    assert r.status_code == 400
