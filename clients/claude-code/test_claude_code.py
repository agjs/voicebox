"""Tests for Claude Code speak/dictate helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


speak_text = _load("speak_text", "speak_text.py")
dictate = _load("dictate", "dictate.py")


def test_clean_strips_code_fences_and_markdown():
    raw = "Hello **world**. ```python\nprint(1)\n``` See https://example.com/a for more."
    cleaned = speak_text.clean_assistant_message(raw)
    assert "print" not in cleaned
    assert "https://" not in cleaned
    assert "*" not in cleaned
    assert "Hello world." in cleaned


def test_clean_truncates_on_word_boundary():
    raw = "one two three four five"
    cleaned = speak_text.clean_assistant_message(raw, max_chars=10)
    assert cleaned == "one two"
    assert len(cleaned) <= 10


def test_first_sentence_split():
    first, remaining = speak_text.first_sentence("Hello there. More words follow.")
    assert first == "Hello there."
    assert remaining == "More words follow."


def test_auth_headers_forward_bearer():
    assert speak_text.auth_headers(None) == {}
    assert speak_text.auth_headers("secret") == {"Authorization": "Bearer secret"}


def test_dictate_uses_bearer_when_key_set(monkeypatch):
    monkeypatch.setenv("VOICEBOX_URL", "http://vb.example:8790")
    monkeypatch.setenv("VOICEBOX_API_KEY", "test-key")
    assert dictate.get_voicebox_url() == "http://vb.example:8790"

    captured = {}

    class FakeResponse:
        def read(self):
            return b'{"text":"hi"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        captured["authorization"] = req.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr(dictate, "urlopen", fake_urlopen)
    assert dictate.transcribe_audio(b"RIFF....") == "hi"
    assert captured["url"] == "http://vb.example:8790/v1/audio/transcriptions"
    assert captured["authorization"] == "Bearer test-key"


def test_dictate_omits_auth_without_key(monkeypatch):
    monkeypatch.delenv("VOICEBOX_API_KEY", raising=False)
    monkeypatch.setenv("VOICEBOX_URL", "http://localhost:8790")
    captured = {}

    class FakeResponse:
        def read(self):
            return b'{"text":"ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=30):
        captured["headers"] = dict(req.header_items())
        return FakeResponse()

    monkeypatch.setattr(dictate, "urlopen", fake_urlopen)
    assert dictate.transcribe_audio(b"data") == "ok"
    assert "Authorization" not in captured["headers"]
