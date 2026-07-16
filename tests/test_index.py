"""Index-build row transforms and SQL generation (no Spark, no Postgres)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pipeline import config
from pipeline.jobs.index import (
    INDEX_COLUMNS,
    build_ddl,
    chunked,
    hnsw_index_statements,
    row_to_record,
    table_name,
)


def _row(**overrides: Any) -> dict[str, Any]:
    """A joined pipeline row as a mapping (Spark Rows support the same [] access)."""
    row: dict[str, Any] = {
        "movie_id": 1,
        "tmdb_id": 862,
        "title": "Toy Story",
        "release_year": 1995,
        "overview": None,
        "genres": ["Adventure", "Animation"],
        "keywords": ["pixar animation"],
        "poster_path": None,
        "vote_average": None,
        "vote_count": None,
        "popularity": None,
        "runtime": None,
        "source": "movielens_fallback",
        "rating_count": 100,
        "rating_mean": 3.9,
        "bayes_score": 3.93,
        "embedding": [0.1] * config.EMBED_DIM,
        "factor": [0.5] * config.ALS_RANK,
    }
    row.update(overrides)
    return row


class TestRowToRecord:
    def test_column_order_matches_index_columns(self) -> None:
        record = row_to_record(_row())
        assert len(record) == len(INDEX_COLUMNS)
        assert record[INDEX_COLUMNS.index("movie_id")] == 1
        assert record[INDEX_COLUMNS.index("title")] == "Toy Story"
        assert record[INDEX_COLUMNS.index("source")] == "movielens_fallback"

    def test_vectors_become_float32_arrays(self) -> None:
        record = row_to_record(_row())
        embedding = record[INDEX_COLUMNS.index("embedding")]
        factor = record[INDEX_COLUMNS.index("factor")]
        assert isinstance(embedding, np.ndarray) and embedding.dtype == np.float32
        assert embedding.shape == (config.EMBED_DIM,)
        assert isinstance(factor, np.ndarray) and factor.shape == (config.ALS_RANK,)

    def test_missing_als_factor_stays_null(self) -> None:
        """Titles absent from staged ratings have no CF factor -> SQL NULL."""
        record = row_to_record(_row(factor=None))
        assert record[INDEX_COLUMNS.index("factor")] is None

    def test_null_array_columns_become_empty_lists(self) -> None:
        record = row_to_record(_row(genres=None, keywords=None))
        assert record[INDEX_COLUMNS.index("genres")] == []
        assert record[INDEX_COLUMNS.index("keywords")] == []


class TestSql:
    def test_ddl_declares_both_vector_columns(self) -> None:
        ddl = build_ddl("movies", 384, 64)
        assert "CREATE TABLE movies" in ddl
        assert "embedding     vector(384) NOT NULL" in ddl
        assert "factor        vector(64)" in ddl

    def test_hnsw_statements_cover_both_vectors_with_cosine_ops(self) -> None:
        statements = hnsw_index_statements("movies_sample")
        assert len(statements) == 2
        assert all("USING hnsw" in s and "vector_cosine_ops" in s for s in statements)
        assert any("(embedding " in s for s in statements)
        assert any("(factor " in s for s in statements)

    def test_table_name_per_mode(self) -> None:
        assert table_name(sample=False) == config.INDEX_TABLE
        assert table_name(sample=True) == config.INDEX_TABLE_SAMPLE


class TestChunked:
    def test_batches_of_size(self) -> None:
        records = [(i,) for i in range(5)]
        assert [len(b) for b in chunked(records, 2)] == [2, 2, 1]

    def test_preserves_order_and_content(self) -> None:
        records = [(i,) for i in range(4)]
        assert [row for batch in chunked(records, 3) for row in batch] == records

    def test_bad_size(self) -> None:
        with pytest.raises(ValueError):
            list(chunked([(1,)], 0))
