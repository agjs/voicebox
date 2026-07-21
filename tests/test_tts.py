import pytest
from voicebox.config import load_settings
from voicebox.tts import TtsEngine, split_sentences


def test_split_sentences():
    out = split_sentences("Hello there. How are you? Fine!  ")
    assert out == ["Hello there.", "How are you?", "Fine!"]


def test_split_single_sentence_no_terminator():
    assert split_sentences("just one clause") == ["just one clause"]


@pytest.fixture(scope="module")
def engine():
    return TtsEngine(load_settings())


def test_streams_one_chunk_per_sentence(engine):
    chunks = list(engine.synthesize_stream("Hello there. How are you?"))
    assert len(chunks) == 2
    assert all(isinstance(c, bytes) and len(c) > 0 for c in chunks)
    # int16 PCM => even byte length
    assert all(len(c) % 2 == 0 for c in chunks)


def test_empty_text_raises(engine):
    with pytest.raises(ValueError):
        list(engine.synthesize_stream("   "))


def test_init_with_bad_model_raises_runtime_error(monkeypatch):
    from voicebox.tts import TtsEngine
    from voicebox.config import Settings

    def mock_hf_hub_download(*args, **kwargs):
        raise ValueError("model not found")

    monkeypatch.setattr("voicebox.tts.hf_hub_download", mock_hf_hub_download)
    settings = Settings(
        stt_model="dummy",
        tts_model="invalid/model",
        tts_engine="kokoro",
        piper_voice="en_US-lessac-high",
        default_voice="dummy",
        port=8790,
        device="cpu",
        cpu_threads=0,
        max_audio_seconds=120,
        max_upload_mb=25,
        max_input_chars=4000,
    )

    with pytest.raises(RuntimeError) as exc_info:
        TtsEngine(settings)
    assert "Failed to load TTS model" in str(exc_info.value)
    assert "invalid/model" in str(exc_info.value)
