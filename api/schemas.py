"""Pydantic schemas for the CineScope API.

:class:`QuerySpec` is the contract between query parsing and retrieval. Both
parsers (Claude and the heuristic fallback) emit it, the client may send a
pre-parsed one (editable filter chips re-query without re-parsing), and the
retrieval layer only ever consumes the validated model — so filter values are
normalized in exactly one place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline import config

# Earliest film in MovieLens territory / a generous future ceiling. Years
# outside this window are almost always parse noise ("2 hours", "90 minutes").
MIN_YEAR = 1870
MAX_YEAR = 2100


def _clean_str_list(values: list[str]) -> list[str]:
    """Strip, drop empties, and dedupe (case-insensitive, order-preserving)."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


class YearRange(BaseModel):
    """Inclusive release-year window; either bound may be open."""

    start: int | None = None
    end: int | None = None

    @field_validator("start", "end")
    @classmethod
    def _plausible_year(cls, v: int | None) -> int | None:
        return v if v is not None and MIN_YEAR <= v <= MAX_YEAR else None

    @model_validator(mode="after")
    def _ordered(self) -> YearRange:
        if self.start is not None and self.end is not None and self.start > self.end:
            self.start, self.end = self.end, self.start
        return self

    def is_empty(self) -> bool:
        return self.start is None and self.end is None


class QuerySpec(BaseModel):
    """Structured interpretation of a natural-language discovery query."""

    reference_titles: list[str] = Field(
        default_factory=list,
        description="Movie titles the user wants results similar to (e.g. 'Inception').",
    )
    mood_adjustments: list[str] = Field(
        default_factory=list,
        description="Tone shifts relative to the references (e.g. 'funnier', 'darker').",
    )
    genres_include: list[str] = Field(
        default_factory=list,
        description="Genres the results must have (MovieLens genre names).",
    )
    genres_exclude: list[str] = Field(
        default_factory=list,
        description="Genres the results must NOT have.",
    )
    year_range: YearRange | None = Field(
        default=None, description="Inclusive release-year window, if the user constrained it."
    )
    min_rating: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Minimum rating on a 0-10 scale (TMDB vote_average scale).",
    )
    similarity_text: str = Field(
        default="",
        description="Short free-text description of plot/tone/themes to embed for search.",
    )

    @field_validator("reference_titles", "mood_adjustments", "genres_include", "genres_exclude")
    @classmethod
    def _clean_lists(cls, v: list[str]) -> list[str]:
        return _clean_str_list(v)

    @model_validator(mode="after")
    def _drop_empty_year_range(self) -> QuerySpec:
        if self.year_range is not None and self.year_range.is_empty():
            self.year_range = None
        return self


class DiscoverRequest(BaseModel):
    """POST /api/discover body: free text, or a pre-parsed spec (chips)."""

    query: str = ""
    spec: QuerySpec | None = None
    limit: int = Field(default=config.DISCOVER_DEFAULT_LIMIT, ge=1, le=config.DISCOVER_MAX_LIMIT)

    @model_validator(mode="after")
    def _query_or_spec(self) -> DiscoverRequest:
        if self.spec is None and not self.query.strip():
            raise ValueError("Provide a non-empty 'query' or a pre-parsed 'spec'.")
        return self


class Why(BaseModel):
    """Per-result explanation: which retrieval signals matched and how hard."""

    semantic_similarity: float | None = Field(
        default=None, description="Cosine similarity between query text and plot/genre embedding."
    )
    behavioral_boost: float | None = Field(
        default=None,
        description=(
            "Mean ALS-factor cosine to the resolved reference titles "
            "(people-who-liked-X signal); null when unavailable."
        ),
    )
    quality_score: float | None = Field(
        default=None, description="Bayesian-weighted MovieLens rating score (catalog-wide prior)."
    )
    matched_filters: list[str] = Field(
        default_factory=list, description="Human-readable spec filters this result satisfies."
    )
    liked_by_fans_of: list[str] = Field(
        default_factory=list, description="Resolved reference titles driving the behavioral boost."
    )


class MovieSummary(BaseModel):
    """Compact movie card used in result lists."""

    model_config = ConfigDict(from_attributes=True)

    movie_id: int
    tmdb_id: int | None = None
    title: str
    release_year: int | None = None
    overview: str | None = None
    genres: list[str] = Field(default_factory=list)
    poster_url: str | None = None
    vote_average: float | None = None
    rating_count: int | None = None
    source: str


class DiscoverResult(MovieSummary):
    score: float = Field(description="Hybrid rank score in [0, 1] (weighted, normalized).")
    why: Why


class DiscoverResponse(BaseModel):
    query: str
    parser: str = Field(
        description="'claude', 'heuristic_fallback', or 'provided_spec' (chips re-query)."
    )
    spec: QuerySpec
    table: str = Field(description="Postgres table served ('movies' or 'movies_sample').")
    results: list[DiscoverResult]


class MovieDetail(MovieSummary):
    keywords: list[str] = Field(default_factory=list)
    runtime: int | None = None
    popularity: float | None = None
    vote_count: int | None = None
    rating_mean: float | None = None
    bayes_score: float | None = None


class SimilarList(BaseModel):
    """One labeled more-like-this row (semantic vs behavioral)."""

    label: str
    basis: str = Field(description="'embedding' (semantic) or 'als_factors' (behavioral).")
    results: list[MovieSummary]


class MovieDetailResponse(BaseModel):
    movie: MovieDetail
    more_like_this: list[SimilarList]


def poster_url(poster_path: str | None) -> str | None:
    """TMDB CDN URL for a poster path (fallback records have none)."""
    return f"{config.TMDB_POSTER_BASE_URL}{poster_path}" if poster_path else None
