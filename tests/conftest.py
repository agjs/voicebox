import dataclasses
import pytest
from voicebox.config import Settings, load_settings


def make_settings(**overrides) -> Settings:
    """Build a Settings from defaults, overriding only what a test needs.

    Resilient to new Settings fields (unlike full-kwargs construction).
    Example: make_settings(stt_model="invalid/model")
    """
    return dataclasses.replace(load_settings(), **overrides)


@pytest.fixture
def settings_factory():
    """Fixture that provides the make_settings factory function."""
    return make_settings
