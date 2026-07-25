#!/usr/bin/env python3
"""Bump the voicebox semver across versioned source and docs files."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r'^version = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)


def read_version(root: Path = ROOT) -> tuple[int, int, int]:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise ValueError(f"could not find version in {root / 'pyproject.toml'}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump(part: str, current: tuple[int, int, int]) -> tuple[int, int, int]:
    major, minor, patch = current
    if part == "major":
        return major + 1, 0, 0
    if part == "minor":
        return major, minor + 1, 0
    if part == "patch":
        return major, minor, patch + 1
    raise ValueError(f"unknown bump part: {part}")


def format_version(parts: tuple[int, int, int]) -> str:
    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def parse_version(value: str) -> tuple[int, int, int]:
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError(f"invalid version: {value!r}")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def _replace_pyproject(text: str, new_version: str) -> str:
    updated, count = re.subn(
        r'(?m)^version = "\d+\.\d+\.\d+"',
        f'version = "{new_version}"',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("pyproject version line not found")
    return updated


def _replace_app(text: str, new_version: str) -> str:
    updated, count = re.subn(
        r'(FastAPI\(title="voicebox", version=")(\d+\.\d+\.\d+)("\))',
        rf"\g<1>{new_version}\3",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("app.py FastAPI version not found")
    return updated


def _replace_readme(text: str, new_version: str) -> str:
    text, n1 = re.subn(
        r"(badge/version-)(\d+\.\d+\.\d+)(-blue)",
        rf"\g<1>{new_version}\3",
        text,
        count=1,
    )
    text, n2 = re.subn(
        r'(alt="version )(\d+\.\d+\.\d+)(")',
        rf"\g<1>{new_version}\3",
        text,
        count=1,
    )
    text, n3 = re.subn(
        r"(ghcr\.io/agjs/voicebox:)(\d+\.\d+\.\d+)",
        rf"\g<1>{new_version}",
        text,
    )
    if n1 != 1 or n2 != 1 or n3 < 1:
        raise ValueError("README version markers not found")
    return text


def _replace_svg(text: str, new_version: str) -> str:
    updated, count = re.subn(
        r"(voicebox )(\d+\.\d+\.\d+)( - localhost)",
        rf"\g<1>{new_version}\3",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("svg version marker not found")
    return updated


def apply_version(new_version: str, root: Path = ROOT) -> list[Path]:
    updates: list[tuple[Path, Callable[[str, str], str]]] = [
        (root / "pyproject.toml", _replace_pyproject),
        (root / "clients/voice-chat/pyproject.toml", _replace_pyproject),
        (root / "src/voicebox/app.py", _replace_app),
        (root / "README.md", _replace_readme),
        (root / "assets/banner.svg", _replace_svg),
        (root / "assets/social.svg", _replace_svg),
    ]
    touched: list[Path] = []
    for path, replacer in updates:
        updated = replacer(path.read_text(encoding="utf-8"), new_version)
        path.write_text(updated, encoding="utf-8")
        touched.append(path)
    return touched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=("major", "minor", "patch"),
        default="patch",
        help="Semver component to bump (default: patch)",
    )
    parser.add_argument("--set", metavar="X.Y.Z", help="Set an exact version")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the next version without writing files",
    )
    args = parser.parse_args(argv)

    current = read_version()
    if args.set:
        new_version = format_version(parse_version(args.set))
    else:
        new_version = format_version(bump(args.part, current))

    if args.dry_run:
        print(new_version)
        return 0

    touched = apply_version(new_version)
    print(new_version)
    for path in touched:
        print(f"updated {path.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
