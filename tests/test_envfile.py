"""Tests for the stdlib .env loader."""

from __future__ import annotations

import os
from pathlib import Path

from pipeline.envfile import load_dotenv, parse_env_text


def test_parse_ignores_comments_blanks_and_junk() -> None:
    text = "# comment\n\nTMDB_API_KEY=abc123\nBROKEN LINE\nQUOTED='xyz'\nEMPTY=\n"
    assert parse_env_text(text) == {"TMDB_API_KEY": "abc123", "QUOTED": "xyz", "EMPTY": ""}


def test_load_does_not_override_existing_env(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CINESCOPE_TEST_A=from_file\nCINESCOPE_TEST_B=set_me\n")
    monkeypatch.setenv("CINESCOPE_TEST_A", "from_process")
    monkeypatch.delenv("CINESCOPE_TEST_B", raising=False)

    loaded = load_dotenv(env_file)

    assert loaded == ["CINESCOPE_TEST_B"]
    assert os.environ["CINESCOPE_TEST_A"] == "from_process"
    assert os.environ["CINESCOPE_TEST_B"] == "set_me"
    monkeypatch.delenv("CINESCOPE_TEST_B", raising=False)


def test_load_missing_file_is_noop(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / ".env") == []
