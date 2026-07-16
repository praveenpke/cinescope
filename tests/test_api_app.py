"""Endpoint tests with the DB, embedder, and retrieval seams stubbed out."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api.main import app
from api.schemas import DiscoverResult, MovieDetail, MovieSummary, SimilarList, Why

if TYPE_CHECKING:
    from collections.abc import Iterator


def _result(movie_id: int = 1) -> DiscoverResult:
    return DiscoverResult(
        movie_id=movie_id,
        title=f"Movie {movie_id}",
        source="movielens_fallback",
        score=0.9,
        why=Why(semantic_similarity=0.8, matched_filters=["genre: Comedy"]),
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with fake DB/embedder/retrieval (no Postgres, no torch)."""

    @contextmanager
    def fake_connect() -> Iterator[object]:
        yield object()

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(api_main.db, "connect", fake_connect)
    monkeypatch.setattr(api_main.db, "resolve_table", lambda conn: "movies_sample")
    monkeypatch.setattr(api_main.embedder, "embed_query", lambda spec: np.zeros(4, dtype="float32"))
    monkeypatch.setattr(
        api_main.retrieval,
        "discover",
        lambda conn, table, spec, vec, limit: [_result(1), _result(2)][:limit],
    )
    return TestClient(app)


def test_discover_parses_query_with_heuristic_fallback(client: TestClient) -> None:
    response = client.post("/api/discover", json={"query": "like Jaws but funnier"})
    assert response.status_code == 200
    body = response.json()
    assert body["parser"] == "heuristic_fallback"
    assert body["spec"]["reference_titles"] == ["Jaws"]
    assert body["spec"]["genres_include"] == ["Comedy"]
    assert body["table"] == "movies_sample"
    assert [r["movie_id"] for r in body["results"]] == [1, 2]
    assert body["results"][0]["why"]["matched_filters"] == ["genre: Comedy"]


def test_discover_with_provided_spec_skips_parsing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(query: str) -> Any:
        raise AssertionError("parse_query must not be called when a spec is provided")

    monkeypatch.setattr(api_main.parsers, "parse_query", boom)
    response = client.post(
        "/api/discover",
        json={"query": "", "spec": {"genres_include": ["Comedy"]}, "limit": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["parser"] == "provided_spec"
    assert body["spec"]["genres_include"] == ["Comedy"]
    assert len(body["results"]) == 1


def test_discover_rejects_empty_body(client: TestClient) -> None:
    assert client.post("/api/discover", json={"query": "   "}).status_code == 422
    assert client.post("/api/discover", json={}).status_code == 422


def test_movie_detail_and_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    detail = MovieDetail(movie_id=7, title="Jaws", source="movielens_fallback")
    similar = [
        SimilarList(
            label="Similar story & vibe (plot/genre embedding)",
            basis="embedding",
            results=[MovieSummary(movie_id=8, title="Piranha", source="movielens_fallback")],
        ),
        SimilarList(
            label="Fans also loved (ALS collaborative filtering)",
            basis="als_factors",
            results=[],
        ),
    ]
    monkeypatch.setattr(
        api_main.retrieval,
        "fetch_movie",
        lambda conn, table, movie_id: detail if movie_id == 7 else None,
    )
    monkeypatch.setattr(api_main.retrieval, "more_like_this", lambda conn, table, movie_id: similar)

    response = client.get("/api/movies/7")
    assert response.status_code == 200
    body = response.json()
    assert body["movie"]["title"] == "Jaws"
    assert [s["basis"] for s in body["more_like_this"]] == ["embedding", "als_factors"]

    assert client.get("/api/movies/999").status_code == 404


def test_cors_allows_vite_dev_server(client: TestClient) -> None:
    response = client.options(
        "/api/discover",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
