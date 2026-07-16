"""QuerySpec / request schema validation (the parse contract)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import DiscoverRequest, QuerySpec, YearRange, poster_url


def test_year_range_swaps_reversed_bounds() -> None:
    yr = YearRange(start=1999, end=1990)
    assert (yr.start, yr.end) == (1990, 1999)


def test_year_range_drops_implausible_years() -> None:
    yr = YearRange(start=90, end=1999)  # "90" is parse noise, not a year
    assert yr.start is None
    assert yr.end == 1999


def test_empty_year_range_normalizes_to_none() -> None:
    spec = QuerySpec(year_range=YearRange(start=None, end=None))
    assert spec.year_range is None


def test_genre_lists_are_cleaned_and_deduped() -> None:
    spec = QuerySpec(genres_include=[" Comedy ", "comedy", "", "Sci-Fi"])
    assert spec.genres_include == ["Comedy", "Sci-Fi"]


def test_min_rating_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        QuerySpec(min_rating=11.0)
    with pytest.raises(ValidationError):
        QuerySpec(min_rating=-1.0)


def test_discover_request_needs_query_or_spec() -> None:
    with pytest.raises(ValidationError):
        DiscoverRequest(query="   ")
    assert DiscoverRequest(query="like Jaws").query == "like Jaws"
    assert DiscoverRequest(spec=QuerySpec()).spec is not None


def test_discover_request_limit_bounds() -> None:
    with pytest.raises(ValidationError):
        DiscoverRequest(query="x", limit=0)
    with pytest.raises(ValidationError):
        DiscoverRequest(query="x", limit=101)


def test_poster_url() -> None:
    assert poster_url(None) is None
    assert poster_url("/abc.jpg") == "https://image.tmdb.org/t/p/w342/abc.jpg"
