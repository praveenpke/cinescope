"""Query parsers: heuristic regex cases + ClaudeParser against a mock client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from api import parsers
from api.parsers import ClaudeParser, HeuristicParser
from api.schemas import QuerySpec
from pipeline import config

# --- HeuristicParser ---------------------------------------------------------


@pytest.fixture()
def heuristic() -> HeuristicParser:
    return HeuristicParser()


def test_like_title_with_mood_and_genre(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("like Inception but funnier")
    assert spec.reference_titles == ["Inception"]
    assert spec.genres_include == ["Comedy"]
    assert spec.mood_adjustments == ["funnier"]
    assert spec.similarity_text == "like Inception but funnier"


def test_similar_to_extraction_stops_at_punctuation(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("something similar to The Godfather, set in the 70s")
    assert spec.reference_titles == ["The Godfather"]
    assert spec.year_range is not None
    assert (spec.year_range.start, spec.year_range.end) == (1970, 1979)


def test_decade_and_multi_genre(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("romantic sci-fi from the 2000s")
    assert set(spec.genres_include) == {"Romance", "Sci-Fi"}
    assert spec.year_range is not None
    assert (spec.year_range.start, spec.year_range.end) == (2000, 2009)


def test_negated_genre_goes_to_exclude(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("a crime thriller but no horror please")
    assert set(spec.genres_include) == {"Crime", "Thriller"}
    assert spec.genres_exclude == ["Horror"]


def test_between_years(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("war movies between 1994 and 1999")
    assert spec.year_range is not None
    assert (spec.year_range.start, spec.year_range.end) == (1994, 1999)
    assert spec.genres_include == ["War"]


def test_after_year_open_ended(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("animated movies after 2010")
    assert spec.genres_include == ["Animation"]
    assert spec.year_range is not None
    assert (spec.year_range.start, spec.year_range.end) == (2010, None)


def test_before_year_open_start(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("westerns before 1970")
    assert spec.year_range is not None
    assert (spec.year_range.start, spec.year_range.end) == (None, 1970)


def test_highly_rated_sets_min_rating(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("highly rated documentaries")
    assert spec.min_rating == 7.5
    assert spec.genres_include == ["Documentary"]


def test_no_signals_still_returns_similarity_text(heuristic: HeuristicParser) -> None:
    spec = heuristic.parse("movies about chess prodigies")
    assert spec.reference_titles == []
    assert spec.genres_include == []
    assert spec.year_range is None
    assert spec.similarity_text == "movies about chess prodigies"


# --- ClaudeParser (mocked SDK client) ----------------------------------------


class _FakeMessages:
    def __init__(self, parsed_output: QuerySpec | None) -> None:
        self.parsed_output = parsed_output
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self.parsed_output, stop_reason="end_turn")


def _fake_client(parsed_output: QuerySpec | None) -> SimpleNamespace:
    return SimpleNamespace(messages=_FakeMessages(parsed_output))


def test_claude_parser_uses_haiku_and_structured_output() -> None:
    expected = QuerySpec(
        reference_titles=["Inception"],
        genres_include=["Comedy"],
        similarity_text="A mind-bending heist that stays light and witty.",
    )
    client = _fake_client(expected)
    spec = ClaudeParser(client=client).parse("like Inception but funnier")  # type: ignore[arg-type]

    assert spec == expected
    (call,) = client.messages.calls
    assert call["model"] == config.CLAUDE_PARSE_MODEL == "claude-haiku-4-5"
    assert call["output_format"] is QuerySpec
    assert call["messages"] == [{"role": "user", "content": "like Inception but funnier"}]
    assert "genres_exclude" in call["system"]  # prompt documents the schema rules


def test_claude_parser_backfills_empty_similarity_text() -> None:
    client = _fake_client(QuerySpec(similarity_text="  "))
    spec = ClaudeParser(client=client).parse("dark heist movies")  # type: ignore[arg-type]
    assert spec.similarity_text == "dark heist movies"


def test_claude_parser_raises_on_unparseable_response() -> None:
    client = _fake_client(None)
    with pytest.raises(ValueError, match="no parseable spec"):
        ClaudeParser(client=client).parse("anything")  # type: ignore[arg-type]


# --- parse_query dispatch -----------------------------------------------------


def test_parse_query_without_key_uses_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec, name = parsers.parse_query("like Jaws but funnier")
    assert name == parsers.PARSER_HEURISTIC == "heuristic_fallback"
    assert spec.reference_titles == ["Jaws"]


def test_parse_query_falls_back_when_claude_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        ClaudeParser, "parse", lambda self, query: (_ for _ in ()).throw(RuntimeError("api down"))
    )
    spec, name = parsers.parse_query("like Jaws but funnier")
    assert name == parsers.PARSER_HEURISTIC
    assert spec.reference_titles == ["Jaws"]
