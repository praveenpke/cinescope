"""TMDB client: payload parsing, 404/429/5xx handling, rate limiting.

All HTTP is mocked — these tests never touch the network.
"""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from typing import Any

import pytest

from pipeline.tmdb_client import RateLimiter, TMDBClient, TMDBError, parse_movie_payload

INCEPTION_PAYLOAD: dict[str, Any] = {
    "id": 27205,
    "title": "Inception",
    "original_title": "Inception",
    "overview": "Cobb, a skilled thief who commits corporate espionage...",
    "genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}],
    "keywords": {"keywords": [{"id": 1, "name": "dream"}, {"id": 2, "name": "subconscious"}]},
    "poster_path": "/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
    "release_date": "2010-07-15",
    "vote_average": 8.4,
    "vote_count": 34000,
    "popularity": 83.5,
    "runtime": 148,
}


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _http_error(code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    msg = Message()
    for key, value in (headers or {}).items():
        msg[key] = value
    return urllib.error.HTTPError("https://api.example", code, "err", msg, io.BytesIO(b""))


def _client(responses: list[Any], **kwargs: Any) -> tuple[TMDBClient, list[float]]:
    """Client whose urlopen pops scripted responses; sleeps are recorded, not slept."""
    sleeps: list[float] = []
    calls = iter(responses)

    def fake_urlopen(url: str, timeout: float = 0) -> _FakeResponse:
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)

    client = TMDBClient("test-key", urlopen=fake_urlopen, sleep=sleeps.append, **kwargs)
    return client, sleeps


class TestParsing:
    def test_parse_full_payload(self) -> None:
        rec = parse_movie_payload(INCEPTION_PAYLOAD, movie_id=79132)
        assert rec["movie_id"] == 79132
        assert rec["tmdb_id"] == 27205
        assert rec["title"] == "Inception"
        assert rec["release_year"] == 2010
        assert rec["genres"] == ["Action", "Science Fiction"]
        assert rec["keywords"] == ["dream", "subconscious"]
        assert rec["poster_path"] == "/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg"
        assert rec["vote_average"] == pytest.approx(8.4)
        assert rec["runtime"] == 148
        assert rec["source"] == "tmdb"

    def test_parse_sparse_payload(self) -> None:
        rec = parse_movie_payload({"id": 99, "original_title": "Obscure"}, movie_id=None)
        assert rec["movie_id"] is None
        assert rec["title"] == "Obscure"
        assert rec["release_year"] is None
        assert rec["overview"] is None
        assert rec["genres"] == []
        assert rec["keywords"] == []
        assert rec["runtime"] is None

    def test_parse_empty_release_date_and_zero_runtime(self) -> None:
        rec = parse_movie_payload({"id": 7, "title": "X", "release_date": "", "runtime": 0}, 1)
        assert rec["release_year"] is None
        assert rec["runtime"] is None


class TestFetch:
    def test_success(self) -> None:
        client, _ = _client([INCEPTION_PAYLOAD])
        payload = client.fetch_movie(27205)
        assert payload is not None and payload["id"] == 27205

    def test_404_returns_none(self) -> None:
        client, _ = _client([_http_error(404)])
        assert client.fetch_movie(1) is None

    def test_401_raises_actionable_error(self) -> None:
        client, _ = _client([_http_error(401)])
        with pytest.raises(TMDBError, match="TMDB_API_KEY"):
            client.fetch_movie(1)

    def test_429_honors_retry_after_then_succeeds(self) -> None:
        client, sleeps = _client([_http_error(429, {"Retry-After": "7"}), INCEPTION_PAYLOAD])
        payload = client.fetch_movie(27205)
        assert payload is not None
        assert 7.0 in sleeps

    def test_5xx_retries_with_backoff_then_succeeds(self) -> None:
        client, sleeps = _client([_http_error(500), _http_error(503), INCEPTION_PAYLOAD])
        assert client.fetch_movie(27205) is not None
        backoffs = [s for s in sleeps if s >= 1.0]
        assert backoffs == [2.0, 4.0]  # exponential: 2^1, 2^2

    def test_gives_up_after_max_retries(self) -> None:
        client, _ = _client([_http_error(500)] * 3, max_retries=2)
        with pytest.raises(TMDBError, match="after 2 retries"):
            client.fetch_movie(42)

    def test_other_4xx_raises_immediately(self) -> None:
        client, _ = _client([_http_error(400)])
        with pytest.raises(TMDBError, match="HTTP 400"):
            client.fetch_movie(42)

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(TMDBError, match="empty"):
            TMDBClient("")


class TestRateLimiter:
    def test_spaces_out_calls(self) -> None:
        sleeps: list[float] = []
        clock = iter([0.0, 0.0, 0.0])  # three wait() calls, clock frozen at 0
        limiter = RateLimiter(4.0, sleep=sleeps.append, monotonic=lambda: next(clock))
        limiter.wait()  # first call: free
        limiter.wait()  # must wait 0.25s
        limiter.wait()  # must wait 0.50s
        assert sleeps == pytest.approx([0.25, 0.5])

    def test_no_sleep_when_slow_enough(self) -> None:
        sleeps: list[float] = []
        clock = iter([0.0, 10.0])
        limiter = RateLimiter(4.0, sleep=sleeps.append, monotonic=lambda: next(clock))
        limiter.wait()
        limiter.wait()
        assert sleeps == []
