import pytest
from fastapi.testclient import TestClient
from voicebox.app import create_app
from voicebox.config import load_settings
from voicebox.stt import AudioDecodeError, TranscriptionResult, TranscriptionSegment


class FakeStt:
    model_id = "fake-stt"

    def __init__(self):
        self.last_language = None
        self.last_timestamps = None

    def transcribe(self, audio: bytes, language: str = "en", *, timestamps: bool = False):
        self.last_language = language
        self.last_timestamps = timestamps
        if audio == b"bad":
            raise AudioDecodeError("bad")
        if timestamps:
            return TranscriptionResult(
                text="hello world",
                language=language,
                duration=1.5,
                segments=(TranscriptionSegment(id=0, start=0.0, end=1.5, text=" hello world"),),
            )
        return TranscriptionResult(text="hello world", language=language)


class FakeTts:
    sample_rate = 24000

    def __init__(self):
        self.last_voice = None
        self.last_speed = None

    def list_voice_ids(self):
        return ["af_heart", "bf_emma"]

    def sample_rate_for(self, voice=None):
        if voice is not None and voice not in self.list_voice_ids():
            raise ValueError(f"unsupported voice {voice!r}")
        return self.sample_rate

    def synthesize_stream(self, text, voice=None, speed=1.0):
        self.last_voice = voice
        self.last_speed = speed
        if not text.strip():
            raise ValueError("empty")
        yield (b"\x01\x00" * 100)
        yield (b"\x02\x00" * 100)


@pytest.fixture
def client():
    return TestClient(create_app(FakeStt(), FakeTts(), load_settings()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": True}


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [item["id"] for item in body["data"]]
    assert ids == ["fake-stt", "af_heart", "bf_emma"]
    assert all(
        item["object"] == "model" and item["owned_by"] == "voicebox" for item in body["data"]
    )


def test_transcription_ok(client):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"anything", "audio/wav")},
        data={"model": "whatever"},
    )
    assert r.status_code == 200
    assert r.json() == {"text": "hello world"}
    assert r.headers["server-timing"].startswith("stt;dur=")


def test_transcription_verbose_json():
    stt = FakeStt()
    client = TestClient(create_app(stt, FakeTts(), load_settings()))
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"anything", "audio/wav")},
        data={"response_format": "verbose_json"},
    )
    assert r.status_code == 200
    assert stt.last_timestamps is True
    body = r.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["duration"] == 1.5
    assert body["text"] == "hello world"
    assert body["segments"] == [
        {"id": 0, "start": 0.0, "end": 1.5, "text": " hello world"},
    ]


def test_transcription_text_format_honors_language():
    stt = FakeStt()
    client = TestClient(create_app(stt, FakeTts(), load_settings()))
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"anything", "audio/wav")},
        data={"language": "fr", "response_format": "text"},
    )
    assert r.status_code == 200
    assert r.text == "hello world"
    assert stt.last_language == "fr"
    assert stt.last_timestamps is False


def test_transcription_bad_audio_400(client):
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"bad", "audio/wav")})
    assert r.status_code == 400


def test_speech_wav_streams_valid_container(client):
    r = client.post(
        "/v1/audio/speech", json={"model": "m", "input": "hi there", "response_format": "wav"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    # Body should be a parseable WAV once fully read (via helper for full-buffer).
    body = r.content
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"


def test_speech_pcm_declares_audio_format(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "m", "input": "hi there", "response_format": "pcm"},
    )
    assert r.status_code == 200
    assert r.headers["x-audio-sample-rate"] == "24000"
    assert r.headers["x-audio-channels"] == "1"
    assert r.headers["x-audio-sample-format"] == "s16le"


def test_speech_honors_speed_and_voice():
    tts = FakeTts()
    client = TestClient(create_app(FakeStt(), tts, load_settings()))
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "m",
            "input": "hi",
            "voice": "bf_emma",
            "speed": 1.5,
            "response_format": "pcm",
        },
    )
    assert r.status_code == 200
    assert tts.last_voice == "bf_emma"
    assert tts.last_speed == 1.5


def test_speech_rejects_bad_speed(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "m", "input": "hi", "speed": 9, "response_format": "wav"},
    )
    assert r.status_code == 400


def test_speech_rejects_unknown_voice(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "m", "input": "hi", "voice": "nope", "response_format": "wav"},
    )
    assert r.status_code == 400


def test_speech_empty_input_400(client):
    r = client.post("/v1/audio/speech", json={"model": "m", "input": "   "})
    assert r.status_code == 400


def test_speech_bad_format_400(client):
    r = client.post(
        "/v1/audio/speech", json={"model": "m", "input": "hi", "response_format": "ogg"}
    )
    assert r.status_code == 400


def test_transcription_oversized_upload_413(client):
    # Create fake upload larger than 25 MB limit (25 * 1024 * 1024 + 1 byte)
    oversized = b"x" * (25 * 1024 * 1024 + 1)
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", oversized, "audio/wav")})
    assert r.status_code == 413


def test_speech_overlong_input_400(client):
    # Input longer than 4000 character default limit
    long_input = "x" * 4001
    r = client.post(
        "/v1/audio/speech", json={"model": "m", "input": long_input, "response_format": "wav"}
    )
    assert r.status_code == 400


def test_optional_api_key_protects_audio_endpoints(settings_factory):
    settings = settings_factory(api_key="test-secret")
    client = TestClient(create_app(FakeStt(), FakeTts(), settings))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    denied = client.post(
        "/v1/audio/speech", json={"model": "m", "input": "hi", "response_format": "pcm"}
    )
    assert denied.status_code == 401
    ok = client.post(
        "/v1/audio/speech",
        json={"model": "m", "input": "hi", "response_format": "pcm"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert ok.status_code == 200
