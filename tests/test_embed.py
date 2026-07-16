"""Embedding-text composition + shard checkpoint logic (no model, no Spark)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.jobs.embed import (
    compose_embedding_text,
    shard_is_complete,
    shard_path,
    shard_ranges,
)


class TestComposeEmbeddingText:
    def test_full_tmdb_record(self) -> None:
        text = compose_embedding_text(
            title="Inception",
            overview="A thief steals corporate secrets through dream-sharing.",
            genres=["Action", "Sci-Fi"],
            keywords=["dream", "heist"],
            release_year=2010,
        )
        assert text == (
            "Inception (2010). A thief steals corporate secrets through dream-sharing. "
            "Genres: Action, Sci-Fi. Keywords: dream, heist."
        )

    def test_movielens_fallback_record_has_no_overview(self) -> None:
        """Fallback hydration leaves overview NULL — text must still be rich."""
        text = compose_embedding_text(
            title="Toy Story",
            overview=None,
            genres=["Adventure", "Animation"],
            keywords=["pixar animation", "toys"],
            release_year=1995,
        )
        assert text == (
            "Toy Story (1995). Genres: Adventure, Animation. Keywords: pixar animation, toys."
        )

    def test_whitespace_overview_is_skipped(self) -> None:
        text = compose_embedding_text("X", "   ", ["Drama"], None, None)
        assert text == "X. Genres: Drama."

    def test_bare_title_only(self) -> None:
        """'(no genres listed)' titles with no genome tags still embed the title."""
        assert compose_embedding_text("Obscure Film", None, [], [], None) == "Obscure Film."

    def test_overview_precedes_genres_and_keywords(self) -> None:
        text = compose_embedding_text("T", "Plot.", ["G"], ["k"], 2000)
        assert text.index("Plot.") < text.index("Genres:") < text.index("Keywords:")


class TestShardRanges:
    def test_exact_multiple(self) -> None:
        assert shard_ranges(10, 5) == [(0, 5), (5, 10)]

    def test_remainder_shard(self) -> None:
        assert shard_ranges(11, 5) == [(0, 5), (5, 10), (10, 11)]

    def test_single_partial_shard(self) -> None:
        assert shard_ranges(3, 512) == [(0, 3)]

    def test_zero_rows(self) -> None:
        assert shard_ranges(0, 5) == []

    def test_bad_shard_size(self) -> None:
        with pytest.raises(ValueError):
            shard_ranges(10, 0)


class TestShardCheckpoints:
    def _write(self, path: Path, n: int) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.table(
                {
                    "movie_id": pa.array(range(n), type=pa.int32()),
                    "embedding": pa.array([[0.0]] * n, type=pa.list_(pa.float32())),
                }
            ),
            path,
        )

    def test_missing_file_is_incomplete(self, tmp_path: Path) -> None:
        assert not shard_is_complete(tmp_path / "shard_00000.parquet", 5)

    def test_matching_row_count_is_complete(self, tmp_path: Path) -> None:
        path = shard_path(tmp_path, 0)
        self._write(path, 5)
        assert shard_is_complete(path, 5)

    def test_stale_row_count_forces_rewrite(self, tmp_path: Path) -> None:
        path = shard_path(tmp_path, 0)
        self._write(path, 4)
        assert not shard_is_complete(path, 5)

    def test_corrupt_file_forces_rewrite(self, tmp_path: Path) -> None:
        path = shard_path(tmp_path, 1)
        path.write_bytes(b"not a parquet file")
        assert not shard_is_complete(path, 5)

    def test_shard_path_naming(self, tmp_path: Path) -> None:
        assert shard_path(tmp_path, 7).name == "shard_00007.parquet"
