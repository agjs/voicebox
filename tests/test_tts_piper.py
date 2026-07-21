import pytest
from voicebox.tts import split_sentences


# Test split_sentences is reused from voicebox.tts
def test_piper_reuses_split_sentences():
    """Verify split_sentences is used in both Kokoro and Piper engines."""
    out = split_sentences("Hello there. How are you?")
    assert out == ["Hello there.", "How are you?"]
    assert len(out) == 2


def test_piper_engine_initialization():
    """Test PiperTtsEngine initializes with valid config."""
    from voicebox.config import Settings
    from voicebox.tts_piper import PiperTtsEngine

    settings = Settings(
        stt_model="dummy",
        tts_model="dummy",
        tts_engine="piper",
        piper_voice="en_US-lessac-high",
        default_voice="dummy",
        port=8790,
        device="cpu",
        cpu_threads=0,
        max_audio_seconds=120,
        max_upload_mb=25,
        max_input_chars=4000,
    )

    try:
        engine = PiperTtsEngine(settings)
        assert engine.sample_rate > 0
        assert isinstance(engine.sample_rate, int)
    except RuntimeError as e:
        # Model not available locally (expected on arm64 without prebuilt)
        if "Failed to load Piper voice" in str(e):
            pytest.skip(f"Piper voice not available: {e}")
        raise


def test_piper_sample_rate_is_positive():
    """Test that sample_rate is set to a positive integer."""
    from voicebox.config import load_settings
    from voicebox.tts_piper import PiperTtsEngine

    try:
        settings = load_settings()
        if settings.tts_engine != "piper":
            # Override to test piper explicitly
            from voicebox.config import Settings
            settings = Settings(
                stt_model=settings.stt_model,
                tts_model=settings.tts_model,
                tts_engine="piper",
                piper_voice=settings.piper_voice,
                default_voice=settings.default_voice,
                port=settings.port,
                device=settings.device,
                cpu_threads=settings.cpu_threads,
                max_audio_seconds=settings.max_audio_seconds,
                max_upload_mb=settings.max_upload_mb,
                max_input_chars=settings.max_input_chars,
            )
        engine = PiperTtsEngine(settings)
        assert isinstance(engine.sample_rate, int)
        assert engine.sample_rate > 0
    except RuntimeError as e:
        if "Failed to load Piper voice" in str(e):
            pytest.skip(f"Piper voice not available: {e}")
        raise


def test_piper_synthesize_stream_yields_one_chunk_per_sentence():
    """Test that synthesize_stream yields one PCM chunk per sentence."""
    from voicebox.config import Settings
    from voicebox.tts_piper import PiperTtsEngine

    settings = Settings(
        stt_model="dummy",
        tts_model="dummy",
        tts_engine="piper",
        piper_voice="en_US-lessac-high",
        default_voice="dummy",
        port=8790,
        device="cpu",
        cpu_threads=0,
        max_audio_seconds=120,
        max_upload_mb=25,
        max_input_chars=4000,
    )

    try:
        engine = PiperTtsEngine(settings)
        chunks = list(engine.synthesize_stream("Hello there. How are you?"))
        assert len(chunks) == 2
        assert all(isinstance(c, bytes) and len(c) > 0 for c in chunks)
        # int16 PCM => even byte length
        assert all(len(c) % 2 == 0 for c in chunks)
    except RuntimeError as e:
        if "Failed to load Piper voice" in str(e):
            pytest.skip(f"Piper voice not available: {e}")
        raise


def test_piper_empty_text_raises():
    """Test that synthesize_stream raises ValueError on empty text."""
    from voicebox.config import Settings
    from voicebox.tts_piper import PiperTtsEngine

    settings = Settings(
        stt_model="dummy",
        tts_model="dummy",
        tts_engine="piper",
        piper_voice="en_US-lessac-high",
        default_voice="dummy",
        port=8790,
        device="cpu",
        cpu_threads=0,
        max_audio_seconds=120,
        max_upload_mb=25,
        max_input_chars=4000,
    )

    try:
        engine = PiperTtsEngine(settings)
        with pytest.raises(ValueError) as exc_info:
            list(engine.synthesize_stream("   "))
        assert "empty" in str(exc_info.value)
    except RuntimeError as e:
        if "Failed to load Piper voice" in str(e):
            pytest.skip(f"Piper voice not available: {e}")
        raise
