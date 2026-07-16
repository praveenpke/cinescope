"""TMDB API client for detail hydration (real HTTP path).

Needs ``TMDB_API_KEY`` (v3 auth, free tier — https://www.themoviedb.org/settings/api).
The client is fully functional the moment the key appears in ``.env``; without
it, ``pipeline hydrate`` falls back to MovieLens-derived records (see
``pipeline/jobs/hydrate.py``).

Design notes:
* One request per movie via ``append_to_response=keywords`` (details +
  keywords in a single call).
* Token-interval rate limiting (``config.TMDB_MAX_REQUESTS_PER_SECOND``).
* Backoff: 429 honors ``Retry-After``; 5xx retries with exponential backoff;
  404 returns ``None`` (dead ID — recorded so resume skips it); other 4xx
  raise.
* The API key is only ever placed in the query string, never logged.
* ``urlopen``/``sleep``/``monotonic`` are injectable for unit tests.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from pipeline import config

logger = logging.getLogger(__name__)


class TMDBError(RuntimeError):
    """Non-retryable TMDB API failure (bad key, malformed request, ...)."""


class RateLimiter:
    """Enforces a minimum interval between calls (single-threaded)."""

    def __init__(
        self,
        max_per_second: float,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval = 1.0 / max_per_second
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_allowed = 0.0

    def wait(self) -> None:
        now = self._monotonic()
        if now < self._next_allowed:
            self._sleep(self._next_allowed - now)
            now = self._next_allowed
        self._next_allowed = now + self._interval


class TMDBClient:
    """Minimal TMDB v3 client: movie details + keywords, rate-limited."""

    def __init__(
        self,
        api_key: str,
        base_url: str = config.TMDB_API_BASE_URL,
        max_per_second: float = config.TMDB_MAX_REQUESTS_PER_SECOND,
        max_retries: int = config.TMDB_MAX_RETRIES,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise TMDBError("TMDB_API_KEY is empty — cannot construct TMDBClient")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._urlopen = urlopen
        self._sleep = sleep
        self._limiter = RateLimiter(max_per_second, sleep=sleep)

    def fetch_movie(self, tmdb_id: int) -> dict[str, Any] | None:
        """Fetch details+keywords for one movie. ``None`` for a 404 (dead ID)."""
        query = urllib.parse.urlencode(
            {"api_key": self._api_key, "append_to_response": "keywords", "language": "en-US"}
        )
        url = f"{self._base_url}/movie/{tmdb_id}?{query}"
        attempt = 0
        while True:
            self._limiter.wait()
            try:
                with self._urlopen(url, timeout=30.0) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code == 401:
                    raise TMDBError(
                        "TMDB rejected the API key (HTTP 401). Check TMDB_API_KEY in .env."
                    ) from exc
                attempt += 1
                if attempt > self._max_retries:
                    raise TMDBError(
                        f"TMDB request for movie {tmdb_id} failed after "
                        f"{self._max_retries} retries (last status {exc.code})"
                    ) from exc
                if exc.code == 429:
                    delay = _retry_after_seconds(exc, default=2.0)
                    logger.warning(
                        "TMDB 429 for movie %d — sleeping %.1fs (Retry-After)", tmdb_id, delay
                    )
                elif 500 <= exc.code < 600:
                    delay = min(2.0**attempt, 60.0)
                    logger.warning(
                        "TMDB %d for movie %d — retry %d/%d in %.1fs",
                        exc.code,
                        tmdb_id,
                        attempt,
                        self._max_retries,
                        delay,
                    )
                else:
                    raise TMDBError(
                        f"TMDB request for movie {tmdb_id} failed with HTTP {exc.code}"
                    ) from exc
                self._sleep(delay)
            except urllib.error.URLError as exc:
                attempt += 1
                if attempt > self._max_retries:
                    raise TMDBError(
                        f"TMDB request for movie {tmdb_id}: network failure after "
                        f"{self._max_retries} retries ({exc.reason})"
                    ) from exc
                delay = min(2.0**attempt, 60.0)
                logger.warning(
                    "Network error for movie %d (%s) — retry %d/%d in %.1fs",
                    tmdb_id,
                    exc.reason,
                    attempt,
                    self._max_retries,
                    delay,
                )
                self._sleep(delay)


def parse_movie_payload(payload: dict[str, Any], movie_id: int | None) -> dict[str, Any]:
    """Normalize a TMDB details(+keywords) payload into a hydrated record.

    ``movie_id`` is the MovieLens id (None for export-only popular titles).
    The output shape matches the hydrated parquet schema in
    ``pipeline/jobs/hydrate.py`` for both the real and fallback paths.
    """
    release_date = payload.get("release_date") or ""
    year_text = release_date[:4]
    release_year = int(year_text) if year_text.isdigit() else None
    keywords_obj = payload.get("keywords") or {}
    keywords = [k["name"] for k in keywords_obj.get("keywords", []) if k.get("name")]
    genres = [g["name"] for g in payload.get("genres", []) if g.get("name")]
    runtime = payload.get("runtime")
    return {
        "movie_id": movie_id,
        "tmdb_id": int(payload["id"]),
        "title": payload.get("title") or payload.get("original_title"),
        "release_year": release_year,
        "overview": payload.get("overview") or None,
        "genres": genres,
        "keywords": keywords,
        "poster_path": payload.get("poster_path"),
        "vote_average": _as_float(payload.get("vote_average")),
        "vote_count": _as_int(payload.get("vote_count")),
        "popularity": _as_float(payload.get("popularity")),
        "runtime": int(runtime) if runtime else None,
        "source": "tmdb",
    }


def _retry_after_seconds(exc: urllib.error.HTTPError, default: float) -> float:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return max(float(value), 0.1) if value is not None else default
    except ValueError:
        return default


def _as_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _as_int(value: Any) -> int | None:
    return int(value) if value is not None else None
