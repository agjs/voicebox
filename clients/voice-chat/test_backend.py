"""Tests for the pluggable voice-chat Backend protocol."""

from __future__ import annotations

import threading

from backend import OpenAIChatBackend


def _sse_chunks(tokens: list[str]):
    def stream(messages, cancel):
        for token in tokens:
            if cancel is not None and cancel.is_set():
                break
            payload = '{"choices":[{"delta":{"content":"%s"}}]}' % token
            yield f"data: {payload}\n".encode()
        yield b"data: [DONE]\n"

    return stream


def test_openai_backend_yields_speak_then_done():
    history = [{"role": "system", "content": "Be brief."}]
    backend = OpenAIChatBackend(
        stream_chunks=_sse_chunks(["Hello ", "there. ", "More words."]),
        history=history,
    )
    events = list(backend.run_turn("Hi"))
    speak = [e for e in events if e["type"] == "speak"]
    done = [e for e in events if e["type"] == "done"]
    assert [e["text"] for e in speak] == ["Hello there.", "More words."]
    assert len(done) == 1
    assert done[0]["text"] == "Hello there. More words."
    assert history[-1] == {"role": "assistant", "content": "Hello there. More words."}
    assert history[-2] == {"role": "user", "content": "Hi"}


def test_openai_backend_strips_reasoning_blocks():
    history = [{"role": "system", "content": "sys"}]
    backend = OpenAIChatBackend(
        stream_chunks=_sse_chunks(["<think>secret</think>", "Visible."]),
        history=history,
    )
    events = list(backend.run_turn("q"))
    speak_text = " ".join(e["text"] for e in events if e["type"] == "speak")
    assert "secret" not in speak_text
    assert "Visible." in speak_text


def test_openai_backend_respects_cancel():
    history = [{"role": "system", "content": "sys"}]
    cancel = threading.Event()

    def stream(messages, cancel_event):
        yield b'data: {"choices":[{"delta":{"content":"Hello. "}}]}\n'
        cancel_event.set()
        if cancel_event.is_set():
            return
        yield b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n'
        yield b"data: [DONE]\n"

    backend = OpenAIChatBackend(stream_chunks=stream, history=history)
    events = list(backend.run_turn("q", cancel=cancel))
    speak = [e for e in events if e["type"] == "speak"]
    assert speak == [{"type": "speak", "text": "Hello."}]
    assert events[-1]["type"] == "done"
    assert events[-1]["text"] == "Hello. "
    assert all(m["role"] != "assistant" for m in history)
