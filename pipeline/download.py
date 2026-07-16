"""Download helpers: checksum verification and resumable fetches.

No API keys are needed for anything in this module — the MovieLens archive and
the TMDB daily-export ID file are both public downloads.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import re
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 22  # 4 MiB
_PROGRESS_EVERY_BYTES = 50 * (1 << 20)  # log every ~50 MiB


def md5_of_file(path: Path) -> str:
    """Return the hex MD5 digest of a file, streamed in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def parse_md5_text(text: str, filename: str) -> str:
    """Extract the MD5 hex digest for ``filename`` from a checksum file's text.

    Handles the formats seen in the wild:
    - BSD style:   ``MD5 (ml-25m.zip) = 6b51fb2759a8657d3bfcbfc42b592ada``
    - GNU style:   ``6b51fb2759a8657d3bfcbfc42b592ada  ml-25m.zip``
    - Bare digest: ``6b51fb2759a8657d3bfcbfc42b592ada``
    """
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        bsd = re.match(r"MD5\s*\((?P<name>[^)]+)\)\s*=\s*(?P<hash>[0-9a-fA-F]{32})", line)
        if bsd and bsd.group("name").strip() == filename:
            return bsd.group("hash").lower()
        gnu = re.match(r"(?P<hash>[0-9a-fA-F]{32})\s+\*?(?P<name>\S+)", line)
        if gnu and Path(gnu.group("name")).name == filename:
            return gnu.group("hash").lower()
        bare = re.fullmatch(r"[0-9a-fA-F]{32}", line)
        if bare:
            return line.lower()
    raise ValueError(f"No MD5 digest for {filename!r} found in checksum text")


def fetch_text(url: str, timeout: float = 60.0) -> str:
    """Fetch a small text resource (e.g. a published checksum file)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def download_file(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    """Download ``url`` to ``dest``. Checkpointed: if ``dest`` already exists and
    matches ``expected_md5`` (or no digest is given), the download is skipped.

    The transfer streams to ``dest.part`` and renames on success, so an
    interrupted download never leaves a truncated file behind as ``dest``.
    """
    if dest.exists():
        if expected_md5 is None:
            logger.info("SKIP download (exists): %s", dest)
            return dest
        actual = md5_of_file(dest)
        if actual == expected_md5:
            logger.info("SKIP download (MD5 verified %s): %s", actual, dest)
            return dest
        logger.warning(
            "MD5 mismatch on existing %s (got %s, want %s) — re-downloading",
            dest,
            actual,
            expected_md5,
        )
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=120.0) as resp, part.open("wb") as out:
        done = 0
        next_mark = _PROGRESS_EVERY_BYTES
        while chunk := resp.read(_CHUNK_SIZE):
            out.write(chunk)
            done += len(chunk)
            if done >= next_mark:
                logger.info("  ... %.0f MiB", done / (1 << 20))
                next_mark += _PROGRESS_EVERY_BYTES
    if expected_md5 is not None:
        actual = md5_of_file(part)
        if actual != expected_md5:
            part.unlink()
            raise RuntimeError(f"MD5 mismatch for {url}: got {actual}, expected {expected_md5}")
        logger.info("MD5 verified: %s", actual)
    part.replace(dest)
    return dest


def tmdb_export_candidate_dates(start: date, max_days_back: int) -> list[date]:
    """Dates to try for the TMDB daily export, newest first.

    The export for a given day appears around 08:00 UTC, so the current day may
    not exist yet — we walk back up to ``max_days_back`` days.
    """
    return [start - timedelta(days=n) for n in range(max_days_back)]


def download_tmdb_export(
    dest_dir: Path, start: date, url_template: str, date_format: str, max_days_back: int
) -> Path:
    """Download the newest available TMDB movie-IDs daily export (no key needed).

    Checkpointed: if a previously downloaded export in ``dest_dir`` is a valid
    gzip file, it is reused.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("movie_ids_*.json.gz"), reverse=True)
    for path in existing:
        if _is_valid_gzip(path):
            logger.info("SKIP TMDB export download (exists, valid gzip): %s", path)
            return path
        logger.warning("Removing corrupt TMDB export: %s", path)
        path.unlink()

    errors: list[str] = []
    for day in tmdb_export_candidate_dates(start, max_days_back):
        stamp = day.strftime(date_format)
        url = url_template.format(date=stamp)
        dest = dest_dir / f"movie_ids_{stamp}.json.gz"
        try:
            download_file(url, dest)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")
            logger.info("TMDB export not available for %s (%s) — trying previous day", stamp, exc)
            continue
        if not _is_valid_gzip(dest):
            dest.unlink()
            errors.append(f"{url}: invalid gzip")
            continue
        return dest
    raise RuntimeError(
        "Could not download any TMDB daily export within "
        f"{max_days_back} days of {start}:\n" + "\n".join(errors)
    )


def _is_valid_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            fh.read(1024)
        return True
    except (OSError, EOFError):
        return False
