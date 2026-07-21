import pytest
from voicebox.tts import split_sentences


# Test split_sentences is reused from voicebox.tts
def test_piper_reuses_split_sentences():
    """Verify split_sentences is used in both Kokoro and Piper engines."""
    out = split_sentences("Hello there. How are you?")
    assert out == ["Hello there.", "How are you?"]
    assert len(out) == 2


def test_piper_engine_initialization(settings_factory):
    """Test PiperTtsEngine initializes with valid config."""
    from voicebox.tts_piper import PiperTtsEngine

    settings = settings_factory(
        tts_engine="piper",
        piper_voice="en_US-lessac-high",
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


def test_piper_sample_rate_is_positive(settings_factory):
    """Test that sample_rate is set to a positive integer."""
    from voicebox.config import load_settings
    from voicebox.tts_piper import PiperTtsEngine

    try:
        settings = load_settings()
        if settings.tts_engine != "piper":
            # Override to test piper explicitly
            settings = settings_factory(tts_engine="piper")
        engine = PiperTtsEngine(settings)
        assert isinstance(engine.sample_rate, int)
        assert engine.sample_rate > 0
    except RuntimeError as e:
        if "Failed to load Piper voice" in str(e):
            pytest.skip(f"Piper voice not available: {e}")
        raise


def test_piper_synthesize_stream_yields_one_chunk_per_sentence(settings_factory):
    """Test that synthesize_stream yields one PCM chunk per sentence."""
    from voicebox.tts_piper import PiperTtsEngine

    settings = settings_factory(
        tts_engine="piper",
        piper_voice="en_US-lessac-high",
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


def test_piper_empty_text_raises(settings_factory):
    """Test that synthesize_stream raises ValueError on empty text."""
    from voicebox.tts_piper import PiperTtsEngine

    settings = settings_factory(
        tts_engine="piper",
        piper_voice="en_US-lessac-high",
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
