"""Tests for scripts/bump_version.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_bump():
    path = ROOT / "scripts" / "bump_version.py"
    spec = importlib.util.spec_from_file_location("bump_version", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bump_version = _load_bump()


def test_bump_parts():
    assert bump_version.bump("patch", (0, 2, 9)) == (0, 2, 10)
    assert bump_version.bump("minor", (0, 2, 9)) == (0, 3, 0)
    assert bump_version.bump("major", (0, 2, 9)) == (1, 0, 0)


def test_read_version_from_repo():
    assert bump_version.read_version(ROOT) == (0, 2, 9)


def test_apply_version_updates_all_markers(tmp_path: Path):
    # Minimal mirror of the versioned files.
    (tmp_path / "clients/voice-chat").mkdir(parents=True)
    (tmp_path / "src/voicebox").mkdir(parents=True)
    (tmp_path / "assets").mkdir(parents=True)

    (tmp_path / "pyproject.toml").write_text('version = "0.2.9"\n', encoding="utf-8")
    (tmp_path / "clients/voice-chat/pyproject.toml").write_text(
        'version = "0.2.9"\n', encoding="utf-8"
    )
    (tmp_path / "src/voicebox/app.py").write_text(
        'app = FastAPI(title="voicebox", version="0.2.9")\n', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "badge/version-0.2.9-blue\n"
        'alt="version 0.2.9"\n'
        "ghcr.io/agjs/voicebox:0.2.9\n"
        "ghcr.io/agjs/voicebox:0.2.9-cuda\n",
        encoding="utf-8",
    )
    (tmp_path / "assets/banner.svg").write_text("voicebox 0.2.9 - localhost:8790", encoding="utf-8")
    (tmp_path / "assets/social.svg").write_text("voicebox 0.2.9 - localhost:8790", encoding="utf-8")

    touched = bump_version.apply_version("0.2.10", root=tmp_path)
    assert len(touched) == 6
    assert 'version = "0.2.10"' in (tmp_path / "pyproject.toml").read_text()
    assert 'version="0.2.10"' in (tmp_path / "src/voicebox/app.py").read_text()
    readme = (tmp_path / "README.md").read_text()
    assert "0.2.9" not in readme
    assert "ghcr.io/agjs/voicebox:0.2.10-cuda" in readme
    assert "voicebox 0.2.10 - localhost" in (tmp_path / "assets/banner.svg").read_text()


def test_cli_dry_run(capsys: pytest.CaptureFixture[str]):
    assert bump_version.main(["--dry-run", "--part", "patch"]) == 0
    assert capsys.readouterr().out.strip() == "0.2.10"
