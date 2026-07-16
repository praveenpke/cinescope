"""Retrieval unit tests: filter-SQL building, behavioral math, hybrid rank.

The SQL builders must be fully parameterized — user-controlled values may
never appear inside the SQL text itself, only in the bind-parameter list.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from api import retrieval
from api.retrieval import (
    ResolvedReference,
    _like_pattern,
    behavioral_scores,
    build_filter_clauses,
    describe_filters,
    rank,
)
from api.schemas import QuerySpec, YearRange

# --- filter SQL building ------------------------------------------------------


def test_empty_spec_builds_no_filters() -> None:
    clauses, params = build_filter_clauses(QuerySpec())
    assert clauses == []
    assert params == []


def _full_spec() -> QuerySpec:
    return QuerySpec(
        genres_include=["Comedy", "Sci-Fi"],
        genres_exclude=["Horror"],
        year_range=YearRange(start=1990, end=1999),
        min_rating=7.5,
    )


def test_full_spec_clauses_are_parameterized() -> None:
    clauses, params = build_filter_clauses(_full_spec())
    assert len(clauses) == 6  # 2 includes (conjunctive), exclude, year start/end, rating
    # one placeholder per clause, values only in params — never in the SQL text
    for clause in clauses:
        assert clause.count("%s") == 1
        assert "Comedy" not in clause
        assert "Horror" not in clause
        assert "1990" not in clause
        assert "7.5" not in clause
    assert "comedy" in params
    assert "sci-fi" in params
    assert ["horror"] in params
    assert 1990 in params
    assert 1999 in params
    assert 7.5 in params


def test_user_strings_never_interpolated_into_sql() -> None:
    hostile = "'; DROP TABLE movies_sample; --"
    clauses, params = build_filter_clauses(QuerySpec(genres_include=[hostile]))
    assert all(hostile not in clause for clause in clauses)
    assert hostile.lower() in params


def test_min_rating_uses_vote_average_with_movielens_fallback() -> None:
    clauses, _ = build_filter_clauses(QuerySpec(min_rating=6.0))
    assert clauses == ["COALESCE(vote_average, rating_mean * 2) >= %s"]


def test_describe_filters_human_readable() -> None:
    assert describe_filters(_full_spec()) == [
        "genre: Comedy",
        "genre: Sci-Fi",
        "not genre: Horror",
        "year: 1990-1999",
        "rating: >= 7.5/10",
    ]
    assert describe_filters(QuerySpec(year_range=YearRange(start=2010))) == ["year: 2010 or later"]


def test_like_pattern_escapes_wildcards() -> None:
    assert _like_pattern("100% Wolf_Movie") == r"%100\% Wolf\_Movie%"


def test_check_table_rejects_unknown_tables() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        retrieval._check_table("movies; DROP TABLE movies")


# --- behavioral scores ---------------------------------------------------------


def _candidate(movie_id: int, factor: list[float] | None, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "movie_id": movie_id,
        "tmdb_id": movie_id * 10,
        "title": f"Movie {movie_id}",
        "release_year": 2000,
        "overview": None,
        "genres": ["Comedy"],
        "poster_path": None,
        "vote_average": None,
        "rating_count": 5,
        "source": "movielens_fallback",
        "bayes_score": 3.0,
        "semantic": 0.5,
        "factor": factor,
    }
    base.update(extra)
    return base


def test_behavioral_none_without_reference_factors() -> None:
    candidates = [_candidate(1, [1.0, 0.0])]
    assert behavioral_scores(candidates, []) is None
    refs = [ResolvedReference(9, "Ref", factor=None)]
    assert behavioral_scores(candidates, refs) is None


def test_behavioral_cosine_and_nan_for_factorless() -> None:
    refs = [ResolvedReference(9, "Ref", factor=np.array([1.0, 0.0]))]
    candidates = [
        _candidate(1, [1.0, 0.0]),  # identical -> cosine 1
        _candidate(2, [0.0, 1.0]),  # orthogonal -> cosine 0
        _candidate(3, None),  # no ALS factor -> NaN (unscorable)
    ]
    scores = behavioral_scores(candidates, refs)
    assert scores is not None
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert np.isnan(scores[2])


class _FakeVector:
    """Mimics pgvector.Vector (0.5+), which is not a plain sequence."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._values, dtype=np.float32)


def test_behavioral_handles_pgvector_vector_objects() -> None:
    refs = [ResolvedReference(9, "Ref", factor=np.array([1.0, 0.0]))]
    candidates = [_candidate(1, None)]
    candidates[0]["factor"] = _FakeVector([1.0, 0.0])
    scores = behavioral_scores(candidates, refs)
    assert scores is not None
    assert scores[0] == pytest.approx(1.0)


def test_behavioral_averages_multiple_references() -> None:
    refs = [
        ResolvedReference(8, "A", factor=np.array([1.0, 0.0])),
        ResolvedReference(9, "B", factor=np.array([0.0, 1.0])),
    ]
    candidates = [_candidate(1, [1.0, 0.0])]
    scores = behavioral_scores(candidates, refs)
    assert scores is not None
    assert scores[0] == pytest.approx(0.5)  # mean of cosine 1.0 and 0.0


# --- hybrid rank ---------------------------------------------------------------


def test_rank_orders_by_hybrid_score_and_explains() -> None:
    refs = [ResolvedReference(9, "Jaws", factor=np.array([1.0, 0.0]))]
    candidates = [
        _candidate(1, [1.0, 0.0], semantic=0.2, bayes_score=3.0),  # behavioral winner
        _candidate(2, [0.0, 1.0], semantic=0.9, bayes_score=4.5),  # semantic+quality winner
        _candidate(3, None, semantic=0.1, bayes_score=2.0),  # weakest everywhere
    ]
    spec = QuerySpec(genres_include=["Comedy"])
    results = rank(candidates, refs, spec, limit=3)

    assert [r.movie_id for r in results] == [2, 1, 3]
    top = results[0]
    assert top.why.semantic_similarity == pytest.approx(0.9)
    assert top.why.behavioral_boost == pytest.approx(0.0)
    assert top.why.quality_score == pytest.approx(4.5)
    assert top.why.matched_filters == ["genre: Comedy"]
    assert top.why.liked_by_fans_of == ["Jaws"]
    # candidate 3 has no ALS factor: behavioral is null, not a fake number
    factorless = results[2]
    assert factorless.why.behavioral_boost is None
    assert factorless.why.liked_by_fans_of == []
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_rank_without_references_redistributes_weight() -> None:
    candidates = [
        _candidate(1, None, semantic=0.9, bayes_score=1.0),
        _candidate(2, None, semantic=0.1, bayes_score=5.0),
    ]
    results = rank(candidates, [], QuerySpec(), limit=2)
    # semantic (0.45) outweighs quality (0.15) after redistribution
    assert [r.movie_id for r in results] == [1, 2]
    assert all(r.why.behavioral_boost is None for r in results)


def test_rank_respects_limit_and_empty_pool() -> None:
    assert rank([], [], QuerySpec(), limit=5) == []
    candidates = [_candidate(i, None, semantic=i / 10) for i in range(1, 6)]
    assert len(rank(candidates, [], QuerySpec(), limit=2)) == 2
