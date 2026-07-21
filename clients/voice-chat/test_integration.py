"""
Integration test for voice chat with mocked LLM (tests --text --no-audio mode).
"""

import sys
from unittest.mock import Mock, patch, MagicMock
from pipeline import parse_sse_stream, strip_reasoning, SentenceChunker


def test_text_mode_with_reasoning_stripping():
    """Test --text mode with LLM response containing reasoning tags."""
    # Mock an LLM response with reasoning
    mock_response = [
        b'data: {"choices":[{"delta":{"content":"<think>"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"I should"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" think about this"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"</think>"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"The answer is: vectors"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" are arrays."}}]}\n',
        b'data: [DONE]\n',
    ]

    # Parse and strip
    tokens = list(parse_sse_stream(iter(mock_response)))
    full_text = "".join(tokens)

    # Strip reasoning
    stripped = strip_reasoning(full_text)

    # Verify reasoning is removed
    assert "<think>" not in stripped
    assert "</think>" not in stripped
    assert "I should" not in stripped
    assert "The answer is: vectors are arrays." in stripped


def test_text_mode_sentence_parsing():
    """Test sentence parsing in streamed response."""
    mock_response = [
        b'data: {"choices":[{"delta":{"content":"First sentence."}}]}\n',
        b'data: {"choices":[{"delta":{"content":" Second sentence."}}]}\n',
        b'data: {"choices":[{"delta":{"content":" Final one!"}}]}\n',
        b'data: [DONE]\n',
    ]

    # Parse LLM response
    tokens = list(parse_sse_stream(iter(mock_response)))
    full_text = "".join(tokens)

    # Chunk into sentences
    chunker = SentenceChunker()
    sentences = []
    for token in tokens:
        sentences.extend(chunker.feed(token))
    remainder = chunker.flush()
    if remainder:
        sentences.append(remainder)

    # Verify sentences
    assert "First sentence." in sentences
    assert "Second sentence." in sentences
    assert "Final one!" in sentences


def test_text_mode_with_abbreviation():
    """Test that abbreviations like U.S. are not split incorrectly."""
    mock_response = [
        b'data: {"choices":[{"delta":{"content":"The U.S."}}]}\n',
        b'data: {"choices":[{"delta":{"content":" is great."}}]}\n',
        b'data: [DONE]\n',
    ]

    tokens = list(parse_sse_stream(iter(mock_response)))

    # Chunk into sentences
    chunker = SentenceChunker()
    sentences = []
    for token in tokens:
        sentences.extend(chunker.feed(token))
    remainder = chunker.flush()
    if remainder:
        sentences.append(remainder)

    # Verify full sentence with abbreviation
    assert "The U.S. is great." in sentences


def test_full_pipeline_integration():
    """Test full pipeline: reasoning strip + sentence chunking."""
    mock_response = [
        b'data: {"choices":[{"delta":{"content":"<think>hmm</think>"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"A vector database"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" stores embeddings."}}]}\n',
        b'data: {"choices":[{"delta":{"content":" U.S. companies use them."}}]}\n',
        b'data: [DONE]\n',
    ]

    # Full pipeline
    all_text = ""
    sentences = []
    chunker = SentenceChunker()

    for token in parse_sse_stream(iter(mock_response)):
        clean = strip_reasoning(token)
        all_text += clean
        for sentence in chunker.feed(clean):
            sentences.append(sentence)

    remainder = chunker.flush()
    if remainder:
        sentences.append(remainder)

    # Verify
    assert "<think>" not in all_text
    assert "hmm" not in all_text
    assert "A vector database stores embeddings." in sentences
    assert "U.S. companies use them." in sentences


if __name__ == "__main__":
    test_text_mode_with_reasoning_stripping()
    test_text_mode_sentence_parsing()
    test_text_mode_with_abbreviation()
    test_full_pipeline_integration()
    print("All integration tests passed!")
