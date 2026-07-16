"""Tests for checksum parsing/verification and TMDB export date walk-back."""

from __future__ import annotations

import gzip
import hashlib
from datetime import date
from pathlib import Path

import pytest

from pipeline.download import (
    _is_valid_gzip,
    download_file,
    md5_of_file,
    parse_md5_text,
    tmdb_export_candidate_dates,
)

DIGEST = "6b51fb2759a8657d3bfcbfc42b592ada"


class TestParseMd5Text:
    def test_bsd_style(self) -> None:
        text = f"MD5 (ml-25m.zip) = {DIGEST}\n"
        assert parse_md5_text(text, "ml-25m.zip") == DIGEST

    def test_gnu_style(self) -> None:
        text = f"{DIGEST}  ml-25m.zip\n"
        assert parse_md5_text(text, "ml-25m.zip") == DIGEST

    def test_gnu_style_with_path(self) -> None:
        text = f"{DIGEST}  ./downloads/ml-25m.zip\n"
        assert parse_md5_text(text, "ml-25m.zip") == DIGEST

    def test_bare_digest(self) -> None:
        assert parse_md5_text(f"{DIGEST}\n", "ml-25m.zip") == DIGEST

    def test_uppercase_normalized(self) -> None:
        text = f"MD5 (ml-25m.zip) = {DIGEST.upper()}"
        assert parse_md5_text(text, "ml-25m.zip") == DIGEST

    def test_wrong_filename_rejected(self) -> None:
        text = f"MD5 (ml-latest.zip) = {DIGEST}"
        with pytest.raises(ValueError, match="No MD5 digest"):
            parse_md5_text(text, "ml-25m.zip")

    def test_garbage_rejected(self) -> None:
        with pytest.raises(ValueError, match="No MD5 digest"):
            parse_md5_text("not a checksum file", "ml-25m.zip")


class TestMd5OfFile:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        payload = b"cinescope" * 10_000
        f = tmp_path / "blob.bin"
        f.write_bytes(payload)
        assert md5_of_file(f) == hashlib.md5(payload).hexdigest()


class TestDownloadFileCheckpointing:
    def test_skips_when_existing_file_verifies(self, tmp_path: Path) -> None:
        payload = b"already downloaded"
        dest = tmp_path / "ml-25m.zip"
        dest.write_bytes(payload)
        expected = hashlib.md5(payload).hexdigest()
        # url is bogus on purpose: if the skip logic fails, this raises.
        result = download_file("http://invalid.invalid/x.zip", dest, expected_md5=expected)
        assert result == dest
        assert dest.read_bytes() == payload


class TestTmdbExportDates:
    def test_walks_back_newest_first(self) -> None:
        dates = tmdb_export_candidate_dates(date(2026, 7, 16), max_days_back=3)
        assert dates == [date(2026, 7, 16), date(2026, 7, 15), date(2026, 7, 14)]

    def test_url_date_format(self) -> None:
        assert date(2026, 7, 15).strftime("%m_%d_%Y") == "07_15_2026"


class TestGzipValidation:
    def test_valid_gzip(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.json.gz"
        f.write_bytes(gzip.compress(b'{"id": 1}\n'))
        assert _is_valid_gzip(f)

    def test_truncated_html_error_page(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json.gz"
        f.write_bytes(b"<html>Access Denied</html>")
        assert not _is_valid_gzip(f)
