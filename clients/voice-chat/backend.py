"""Pluggable turn backend for voice-chat (vendor-neutral agent hook).

Any agent can implement ``Backend.run_turn`` and yield speak / approval / done
events. ``OpenAIChatBackend`` is the reference OpenAI-compatible implementation.
"""

from __future__ import annotations

import threading
from typing import Callable, Iterator, Literal, Protocol, TypedDict

from pipeline import ReasoningFilter, SentenceChunker, parse_sse_stream


class SpeakEvent(TypedDict):
    type: Literal["speak"]
    text: str


class ApprovalRequestEvent(TypedDict):
    type: Literal["approval_request"]
    prompt: str
    id: str


class DoneEvent(TypedDict):
    type: Literal["done"]
    text: str


TurnEvent = SpeakEvent | ApprovalRequestEvent | DoneEvent

StreamChunks = Callable[[list[dict[str, str]], threading.Event | None], Iterator[bytes]]


class Backend(Protocol):
    """Turn driver used by the voice-chat shell.

    Implementations must be free of product-specific agent branding. Private
    agents plug in their own Backend without forking this client.
    """

    def run_turn(self, text: str, *, cancel: threading.Event | None = None) -> Iterator[TurnEvent]:
        """Consume one user utterance; yield speak/approval events, then done."""


class OpenAIChatBackend:
    """Reference backend: OpenAI-compatible chat completions over SSE."""

    def __init__(
        self,
        *,
        stream_chunks: StreamChunks,
        history: list[dict[str, str]],
        max_history_turns: int = 8,
    ) -> None:
        self._stream_chunks = stream_chunks
        self.history = history
        self.max_history_turns = max_history_turns

    def _trim_history(self) -> None:
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages + 1:
            self.history[:] = [self.history[0], *self.history[-max_messages:]]

    def run_turn(self, text: str, *, cancel: threading.Event | None = None) -> Iterator[TurnEvent]:
        self.history.append({"role": "user", "content": text})
        self._trim_history()

        reasoning_filter = ReasoningFilter()
        chunker = SentenceChunker()
        assistant_parts: list[str] = []

        def emit_visible(visible: str) -> Iterator[SpeakEvent]:
            if not visible:
                return
            assistant_parts.append(visible)
            for sentence in chunker.feed(visible):
                yield {"type": "speak", "text": sentence}

        for token in parse_sse_stream(self._stream_chunks(self.history, cancel)):
            yield from emit_visible(reasoning_filter.feed(token))
            if cancel is not None and cancel.is_set():
                break

        # Flush pending speech even when cancelled so the last partial sentence plays.
        yield from emit_visible(reasoning_filter.flush())
        remainder = chunker.flush()
        if remainder:
            yield {"type": "speak", "text": remainder}

        assistant_text = "".join(assistant_parts)
        if assistant_text and (cancel is None or not cancel.is_set()):
            self.history.append({"role": "assistant", "content": assistant_text})
            self._trim_history()
        yield {"type": "done", "text": assistant_text}
