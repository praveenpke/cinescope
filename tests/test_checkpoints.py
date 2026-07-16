"""Checkpoint primitives: done markers and JSONL resume logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import checkpoints


class TestMarkers:
    def test_missing_marker_returns_none(self, tmp_path: Path) -> None:
        assert checkpoints.read_marker(tmp_path, "cf_stats") is None

    def test_roundtrip_with_extra_metadata(self, tmp_path: Path) -> None:
        checkpoints.write_marker(tmp_path, "cf_movie_factors", 1234, rank=64, seed=42)
        marker = checkpoints.read_marker(tmp_path, "cf_movie_factors")
        assert marker is not None
        assert marker["rows"] == 1234
        assert marker["rank"] == 64
        assert "completed_at" in marker

    def test_markers_are_namespaced_per_step(self, tmp_path: Path) -> None:
        checkpoints.write_marker(tmp_path, "a", 1)
        assert checkpoints.read_marker(tmp_path, "b") is None


class TestJsonlResume:
    def test_read_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert checkpoints.read_jsonl(tmp_path / "records.jsonl") == []
        assert checkpoints.completed_ids(tmp_path / "records.jsonl", "tmdb_id") == set()

    def test_append_then_completed_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "records.jsonl"
        checkpoints.append_jsonl(path, {"tmdb_id": 603, "title": "The Matrix"})
        checkpoints.append_jsonl(path, {"tmdb_id": 27205, "title": "Inception"})
        assert checkpoints.completed_ids(path, "tmdb_id") == {603, 27205}

    def test_partial_trailing_line_is_dropped_for_resume(self, tmp_path: Path) -> None:
        """A crash mid-write must not poison the checkpoint — resume drops it."""
        path = tmp_path / "records.jsonl"
        checkpoints.append_jsonl(path, {"tmdb_id": 603})
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"tmdb_id": 27205, "title": "Ince')  # simulated kill mid-write
        assert checkpoints.completed_ids(path, "tmdb_id") == {603}

    def test_corrupt_middle_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "records.jsonl"
        path.write_text('not json\n{"tmdb_id": 603}\n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            checkpoints.read_jsonl(path)

    def test_records_missing_id_field_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "records.jsonl"
        checkpoints.append_jsonl(path, {"status": 404})
        checkpoints.append_jsonl(path, {"tmdb_id": 5})
        assert checkpoints.completed_ids(path, "tmdb_id") == {5}


class TestCrawlResume:
    """crawl_tmdb skips checkpointed IDs and checkpoints 404s (pipeline/jobs/hydrate.py)."""

    def test_crawl_skips_done_and_records_misses(self, tmp_path: Path) -> None:
        from pipeline.jobs.hydrate import crawl_tmdb

        records = tmp_path / "records.jsonl"
        misses = tmp_path / "misses.jsonl"
        checkpoints.append_jsonl(records, {"tmdb_id": 1, "title": "already done"})

        fetched_ids: list[int] = []

        class FakeClient:
            def fetch_movie(self, tmdb_id: int) -> dict | None:
                fetched_ids.append(tmdb_id)
                if tmdb_id == 3:
                    return None  # dead ID -> 404
                return {"id": tmdb_id, "title": f"Movie {tmdb_id}", "genres": []}

        targets = [(101, 1), (102, 2), (103, 3)]
        fetched, skipped = crawl_tmdb(FakeClient(), targets, records, misses)  # type: ignore[arg-type]

        assert fetched_ids == [2, 3]  # id 1 resumed from checkpoint
        assert (fetched, skipped) == (2, 1)
        assert checkpoints.completed_ids(misses, "tmdb_id") == {3}
        assert checkpoints.completed_ids(records, "tmdb_id") == {1, 2}

        # Second run: everything checkpointed, nothing fetched.
        fetched_ids.clear()
        fetched, skipped = crawl_tmdb(FakeClient(), targets, records, misses)  # type: ignore[arg-type]
        assert fetched_ids == []
        assert (fetched, skipped) == (0, 3)
