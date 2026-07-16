"""Query-time embedding text must match the index-time composition exactly."""

from __future__ import annotations

from api.embedder import query_text
from api.schemas import QuerySpec
from pipeline.jobs.embed import compose_embedding_text


def test_query_text_reuses_index_time_composer() -> None:
    spec = QuerySpec(
        similarity_text="A shark terrorizes a beach town",
        genres_include=["Horror"],
        mood_adjustments=["funnier"],
    )
    assert query_text(spec) == compose_embedding_text(
        title="A shark terrorizes a beach town",
        overview=None,
        genres=["Horror"],
        keywords=["funnier"],
    )
    assert query_text(spec) == (
        "A shark terrorizes a beach town. Genres: Horror. Keywords: funnier."
    )


def test_query_text_falls_back_to_references_then_placeholder() -> None:
    spec = QuerySpec(reference_titles=["Jaws"], mood_adjustments=["funnier"])
    assert query_text(spec).startswith("Jaws funnier.")
    assert query_text(QuerySpec()) == "a movie."
