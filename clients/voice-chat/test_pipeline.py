"""
Unit tests for voice chat pipeline (pure logic, no network/audio).
"""

from pipeline import ReasoningFilter, SentenceChunker, parse_sse_stream, strip_reasoning


class TestParseSseStream:
    """Tests for OpenAI-style SSE parsing."""

    def test_parses_single_message(self):
        """Parse a simple single-chunk SSE message."""
        chunk = b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        result = list(parse_sse_stream(iter([chunk])))
        assert result == ["Hello"]

    def test_parses_multiple_chunks(self):
        """Parse multiple SSE messages streamed separately."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n',
        ]
        result = list(parse_sse_stream(iter(chunks)))
        assert result == ["Hello", " ", "world"]

    def test_stops_on_done_marker(self):
        """Stop parsing on [DONE] marker."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Done"}}]}\n',
            b"data: [DONE]\n",
            b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n',
        ]
        result = list(parse_sse_stream(iter(chunks)))
        assert result == ["Done"]

    def test_ignores_keep_alives(self):
        """Ignore keep-alive comment lines (`:`)."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b": keep-alive\n",
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n',
        ]
        result = list(parse_sse_stream(iter(chunks)))
        assert result == ["Hello", "world"]

    def test_handles_empty_lines(self):
        """Skip empty lines."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Test"}}]}\n',
            b"\n",
            b'data: {"choices":[{"delta":{"content":"works"}}]}\n',
        ]
        result = list(parse_sse_stream(iter(chunks)))
        assert result == ["Test", "works"]

    def test_handles_missing_delta(self):
        """Skip messages with no delta.content."""
        chunks = [
            b'data: {"choices":[{"delta":{}}]}\n',
            b'data: {"choices":[{"delta":{"content":"valid"}}]}\n',
        ]
        result = list(parse_sse_stream(iter(chunks)))
        assert result == ["valid"]

    def test_handles_malformed_json(self):
        """Skip malformed JSON lines."""
        chunks = [
            b"data: {bad json\n",
            b'data: {"choices":[{"delta":{"content":"valid"}}]}\n',
        ]
        result = list(parse_sse_stream(iter(chunks)))
        assert result == ["valid"]

    def test_handles_split_lines(self):
        """Handle chunks that don't align with line boundaries."""
        # Simulate a chunk split across boundaries
        full_msg = b'data: {"choices":[{"delta":{"content":"Hello world"}}]}\n'
        chunks = [full_msg[:10], full_msg[10:20], full_msg[20:]]
        result = list(parse_sse_stream(iter(chunks)))
        assert result == ["Hello world"]


class TestStripReasoning:
    """Tests for <think> tag removal."""

    def test_removes_full_think_blocks(self):
        """Remove complete <think>...</think> blocks."""
        text = "Before <think>reasoning here</think> after"
        result = strip_reasoning(text)
        assert result == "Before  after"

    def test_removes_multiline_think_blocks(self):
        """Remove <think> blocks spanning multiple lines."""
        text = "Start <think>\nmultiline\nreasoning\n</think> end"
        result = strip_reasoning(text)
        assert result == "Start  end"

    def test_removes_stray_opening_tags(self):
        """Suppress content after an unmatched opening tag to avoid reasoning leaks."""
        text = "Before <think> stray tag"
        result = strip_reasoning(text)
        assert result == "Before "

    def test_removes_stray_closing_tags(self):
        """Remove stray </think> closing tags."""
        text = "Before </think> stray tag"
        result = strip_reasoning(text)
        assert result == "Before  stray tag"

    def test_preserves_normal_text(self):
        """Don't remove non-think content."""
        text = "This is normal text with no reasoning."
        result = strip_reasoning(text)
        assert result == text

    def test_handles_nested_angle_brackets(self):
        """Handle text with other <...> content (e.g., XML)."""
        text = "Response: <think>hidden</think> and <other>keep</other>"
        result = strip_reasoning(text)
        assert "<other>keep</other>" in result
        assert "<think>" not in result

    def test_streaming_filter_handles_tags_split_across_tokens(self):
        filter_ = ReasoningFilter()
        tokens = ["Visible ", "<thi", "nk>", "secret reasoning", "</th", "ink>", "answer"]
        result = "".join(filter_.feed(token) for token in tokens) + filter_.flush()
        assert result == "Visible answer"


class TestSentenceChunker:
    """Tests for incremental sentence chunking."""

    def test_chunks_on_period(self):
        """Split on period boundaries."""
        chunker = SentenceChunker()
        result = list(chunker.feed("Hello. World."))
        remainder = chunker.flush()
        if remainder:
            result.append(remainder)
        assert result == ["Hello.", "World."]

    def test_chunks_on_exclamation(self):
        """Split on exclamation mark."""
        chunker = SentenceChunker()
        result = list(chunker.feed("What! How!"))
        assert result == ["What!", "How!"]

    def test_chunks_on_question(self):
        """Split on question mark."""
        chunker = SentenceChunker()
        result = list(chunker.feed("Why? Yes?"))
        assert result == ["Why?", "Yes?"]

    def test_chunks_on_newline(self):
        """Split on newline boundaries."""
        chunker = SentenceChunker()
        result = list(chunker.feed("Line one\nLine two"))
        assert result == ["Line one"]
        flush = chunker.flush()
        assert flush == "Line two"

    def test_incremental_feeding(self):
        """Feed text in multiple chunks; accumulate sentences."""
        chunker = SentenceChunker()
        chunks_in = ["Hello ", "world. ", "How ", "are ", "you?"]
        all_sentences = []
        for chunk in chunks_in:
            all_sentences.extend(chunker.feed(chunk))
        remainder = chunker.flush()
        if remainder:
            all_sentences.append(remainder)
        assert "Hello world." in all_sentences
        assert "How are you?" in all_sentences

    def test_guards_against_abbreviations(self):
        """Don't split on abbreviations like U.S. (min length guard)."""
        chunker = SentenceChunker()
        result = list(chunker.feed("U.S. is great."))
        remainder = chunker.flush()
        if remainder:
            result.append(remainder)
        # Should not emit "U." because it's < 3 chars
        assert "U." not in result
        assert "U.S. is great." in result

    def test_flush_returns_remainder(self):
        """Flush returns trailing text."""
        chunker = SentenceChunker()
        _ = list(chunker.feed("Hello. Incomplete"))
        remainder = chunker.flush()
        assert remainder == "Incomplete"

    def test_flush_empty_if_too_short(self):
        """Flush returns empty if remainder is too short."""
        chunker = SentenceChunker()
        chunker.feed("Hi")
        remainder = chunker.flush()
        assert remainder == ""

    def test_multiple_punctuation(self):
        """Handle multiple punctuation marks (???, !!!.)"""
        chunker = SentenceChunker()
        result = list(chunker.feed("Really??? Yes!!!"))
        assert any("Really" in s for s in result)
        assert any("Yes" in s for s in result)

    def test_whitespace_stripping(self):
        """Strip leading/trailing whitespace from sentences."""
        chunker = SentenceChunker()
        result = list(chunker.feed("  Hello  .  World  ."))
        remainder = chunker.flush()
        if remainder:
            result.append(remainder)
        assert result == ["Hello.", "World."]

    def test_custom_min_length(self):
        """Respect custom minimum sentence length."""
        chunker = SentenceChunker(min_length=5)
        result = list(chunker.feed("Hi. Hello world."))
        remainder = chunker.flush()
        if remainder:
            result.append(remainder)
        # "Hi" is < 5, should not be emitted
        assert "Hi." not in result
        assert "Hello world." in result
