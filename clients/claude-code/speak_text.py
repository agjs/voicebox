"""Clean assistant text for spoken Claude Code Stop-hook output."""

from __future__ import annotations

import argparse
import re
import sys


def clean_assistant_message(message: str, max_chars: int = 600) -> str:
    """Strip code fences, URLs, and markdown noise; optionally truncate."""
    text = re.sub(r"```[^`]*```", "", message, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("`", "")
    text = re.sub(r"[*#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


def first_sentence(text: str) -> tuple[str, str]:
    """Split into first sentence and remainder (OpenAI-style first-chunk speak)."""
    match = re.search(r"[.!?](?:\s|$)", text)
    if not match:
        return text, ""
    end = match.end()
    # Include the terminator but not trailing whitespace after it.
    first = text[: match.start() + 1]
    remaining = text[end:].strip()
    return first, remaining


def auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument(
        "--split-first",
        action="store_true",
        help="Print first sentence, then a NUL, then the remainder",
    )
    args = parser.parse_args(argv)
    message = sys.stdin.read()
    cleaned = clean_assistant_message(message, max_chars=args.max_chars)
    if not cleaned:
        return 0
    if args.split_first:
        first, remaining = first_sentence(cleaned)
        sys.stdout.write(first)
        sys.stdout.write("\0")
        sys.stdout.write(remaining)
    else:
        sys.stdout.write(cleaned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
