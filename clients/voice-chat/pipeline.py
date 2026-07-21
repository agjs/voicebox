"""
Pure, testable pipeline logic for voice chat client.
No network, no audio—just SSE parsing, reasoning stripping, and sentence chunking.
"""

import json
import re
from typing import Generator, Iterator


def parse_sse_stream(chunks: Iterator[bytes]) -> Generator[str, None, None]:
    """
    Parse raw OpenAI-style SSE lines; yield assistant text tokens.

    Handles `data: {...}` JSON lines, stops on `data: [DONE]`, ignores keep-alives.

    Args:
        chunks: Iterator of byte chunks from streaming response

    Yields:
        Text tokens from assistant.choices[0].delta.content
    """
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        lines = buffer.split(b"\n")
        # Keep the last incomplete line in the buffer
        buffer = lines[-1]

        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue

            # Stop on [DONE] marker (bare or with data: prefix)
            if line == b"[DONE]" or line == b"data: [DONE]":
                return

            # Parse `data: {...}` lines
            if line.startswith(b"data: "):
                data_str = line[6:].decode("utf-8", errors="replace")

                # Skip keep-alive comments
                if data_str.startswith(":"):
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract text from assistant.choices[0].delta.content
                try:
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except (KeyError, IndexError, TypeError):
                    pass

    # Process any remaining buffered data
    if buffer.strip():
        line = buffer.strip()
        if not line.startswith(b":") and line != b"[DONE]":
            if line.startswith(b"data: "):
                data_str = line[6:].decode("utf-8", errors="replace")
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    pass


def strip_reasoning(text: str) -> str:
    """
    Remove <think>...</think> blocks and stray tags from reasoning-enabled LLMs.

    DeepSeek and similar models emit reasoning tags; they must never be spoken.
    Uses DOTALL regex to handle multiline blocks.

    Args:
        text: Raw text with possible <think> blocks

    Returns:
        Text with reasoning blocks removed
    """
    # Remove full <think>...</think> blocks (DOTALL = . matches newlines)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove any stray opening or closing tags
    text = re.sub(r"</?think>", "", text)

    return text


class SentenceChunker:
    """
    Feed streamed text incrementally; emit complete sentences on boundaries.

    Splits on `.?!` or newline boundaries with minimum length guard
    to avoid over-splitting on abbreviations.

    Smart about abbreviations: only splits on a period if it looks like
    a sentence boundary (followed by space + capital letter or end of string).
    """

    _MIN_SENTENCE_LENGTH = 3  # Don't emit very short fragments

    def __init__(self, min_length: int = 3):
        """
        Initialize the chunker.

        Args:
            min_length: Minimum length of a sentence to emit (guard against abbreviations)
        """
        self._buffer = ""
        self._min_length = min_length

    def _is_sentence_boundary(self, pos: int) -> bool:
        """
        Check if position pos (pointing at punctuation) is a true sentence boundary.

        A period is a sentence boundary if:
        - It's an ! or ? followed by newline/space+capital/end-of-buffer, OR
        - It's a period followed by space + capital letter, OR
        - It's a period at the very end of the buffer (end-of-chunk),
          AND it's not an abbreviation like "U.S."
        """
        if pos >= len(self._buffer):
            return False

        char = self._buffer[pos]

        # ! and ? sentences are more flexible
        if char in "!?":
            next_pos = pos + 1
            if next_pos >= len(self._buffer):
                # End of buffer - yes, emit
                return True
            if self._buffer[next_pos] == "\n":
                return True
            while next_pos < len(self._buffer) and self._buffer[next_pos] == " ":
                next_pos += 1
            # If nothing after space, still emit (end of chunk)
            if next_pos >= len(self._buffer):
                return True
            # If capital letter, emit
            if self._buffer[next_pos].isupper():
                return True
            return False

        # Period: smarter handling
        if char == ".":
            # Check if this looks like an abbreviation (single letter before period)
            # e.g., "U.S." or "U.K." - detect the pattern: single letter + period + capital letter
            if pos > 0:
                before_period = self._buffer[pos - 1]
                # Single letter followed by period
                if before_period.isalpha() and (pos == 1 or self._buffer[pos - 2] in " \n"):
                    # Check what comes after the period
                    after_period = pos + 1
                    # If next char is another capital letter (without space), it's likely "U.S."
                    if after_period < len(self._buffer) and self._buffer[after_period].isupper():
                        return False  # "U.S" pattern - not a sentence boundary

            next_pos = pos + 1
            if next_pos >= len(self._buffer):
                # At end of buffer - don't emit for periods (might be streaming)
                # The buffered text will be handled by flush() if needed
                return False

            if self._buffer[next_pos] == "\n":
                return True

            # Skip spaces
            space_pos = next_pos
            while next_pos < len(self._buffer) and self._buffer[next_pos] == " ":
                next_pos += 1

            if next_pos >= len(self._buffer):
                # Period followed by only spaces at end of buffer - don't emit yet
                return False

            # Capital letter after period (with space) = sentence boundary
            if self._buffer[next_pos].isupper():
                return True

        return False

    def feed(self, text: str) -> Generator[str, None, None]:
        """
        Feed streamed text; yield complete sentences as they're detected.

        Args:
            text: New text chunk to process

        Yields:
            Complete sentences ending with .?! or newline
        """
        self._buffer += text

        # Look for sentence boundaries
        while True:
            # Check for newline first
            newline_idx = self._buffer.find("\n")

            # Check for sentence-ending punctuation
            sentence_idx = -1
            for i, char in enumerate(self._buffer):
                if char in ".!?" and self._is_sentence_boundary(i):
                    sentence_idx = i + 1
                    break

            # Determine which comes first
            if newline_idx != -1 and (sentence_idx == -1 or newline_idx < sentence_idx):
                # Newline comes first
                sentence = self._buffer[:newline_idx].strip()
                self._buffer = self._buffer[newline_idx + 1:]
                if len(sentence) >= self._min_length:
                    yield sentence
            elif sentence_idx != -1:
                # Sentence-ending punctuation comes first
                # Strip leading/trailing, and collapse extra spaces before punctuation
                sentence_raw = self._buffer[:sentence_idx].strip()
                # Also clean up extra spaces before punctuation (e.g., "Hello  ." -> "Hello.")
                sentence = re.sub(r'\s+([.!?]+)$', r'\1', sentence_raw)
                self._buffer = self._buffer[sentence_idx:].lstrip()
                if len(sentence) >= self._min_length:
                    yield sentence
            else:
                # No boundary found, but check if buffer ends with definitive punctuation
                # Only emit ! or ? at end-of-buffer (period could be followed by more text)
                if self._buffer and self._buffer[-1] in "!?":
                    sentence = self._buffer.strip()
                    sentence = re.sub(r'\s+([.!?]+)$', r'\1', sentence)
                    if len(sentence) >= self._min_length:
                        yield sentence
                        self._buffer = ""
                break

    def flush(self) -> str:
        """
        Return any remaining text at end of stream.

        Returns:
            Remaining buffered text, or empty string
        """
        remaining = self._buffer.strip()
        # Clean up extra spaces before punctuation
        remaining = re.sub(r'\s+([.!?]+)$', r'\1', remaining)
        self._buffer = ""
        return remaining if len(remaining) >= self._min_length else ""
