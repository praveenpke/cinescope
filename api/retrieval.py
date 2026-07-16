"""Hybrid retrieval: pgvector ANN + ALS-neighbor boost + SQL spec filters.

Flow for ``POST /api/discover``:

1. ``build_filter_clauses`` turns the parsed :class:`~api.schemas.QuerySpec`
   into **parameterized** WHERE fragments (values only ever travel as bind
   parameters — the table name is allowlisted, never user input).
2. ``fetch_candidates`` runs one HNSW scan ordered by embedding cosine
   distance, restricted by those filters, returning a candidate pool larger
   than the final page.
3. ``behavioral_scores`` computes the people-who-liked-X signal: mean ALS
   factor cosine between each candidate and the resolved reference titles.
4. ``rank`` re-ranks the pool with :func:`pipeline.scoring.combine_hybrid`
   under :data:`pipeline.config.HYBRID_WEIGHTS` — the exact function and
   weights the offline eval harness scores, so precision@k measures the
   ranking code that serves traffic.

Every result carries a ``why`` payload with the raw (pre-normalization)
signal values so the UI can explain the match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from psycopg.rows import dict_row

from api import db
from api.schemas import DiscoverResult, MovieDetail, MovieSummary, SimilarList, Why, poster_url
from pipeline import config
from pipeline.scoring import combine_hybrid, cosine_scores, top_k_indices

if TYPE_CHECKING:
    import psycopg

    from api.schemas import QuerySpec

# Columns fetched for every candidate/summary row (subset of index.INDEX_COLUMNS).
_MOVIE_COLUMNS = (
    "movie_id, tmdb_id, title, release_year, overview, genres, keywords, poster_path, "
    "vote_average, vote_count, popularity, runtime, source, rating_count, rating_mean, "
    "bayes_score"
)


def _check_table(table: str) -> str:
    """Guard every SQL composition site: table names are allowlisted constants."""
    if table not in db.ALLOWED_TABLES:
        raise ValueError(f"Table {table!r} not in allowlist {db.ALLOWED_TABLES}")
    return table


def build_filter_clauses(spec: QuerySpec) -> tuple[list[str], list[Any]]:
    """Parameterized WHERE fragments + bind values for a parsed spec.

    Genre matching is case-insensitive equality against the unnested
    ``genres`` array, so parser output ("Sci-Fi") and user chips ("sci-fi")
    both hit. ``min_rating`` is on the 0-10 TMDB scale; fallback-hydrated
    rows have no ``vote_average``, so the 0-5 MovieLens mean is doubled.
    """
    clauses: list[str] = []
    params: list[Any] = []
    # Included genres are conjunctive (each must be present) so every entry in
    # why.matched_filters is literally true for every returned movie.
    for genre in spec.genres_include:
        clauses.append("EXISTS (SELECT 1 FROM unnest(genres) g WHERE lower(g) = %s)")
        params.append(genre.lower())
    if spec.genres_exclude:
        clauses.append("NOT EXISTS (SELECT 1 FROM unnest(genres) g WHERE lower(g) = ANY(%s))")
        params.append([g.lower() for g in spec.genres_exclude])
    if spec.year_range is not None:
        if spec.year_range.start is not None:
            clauses.append("release_year >= %s")
            params.append(spec.year_range.start)
        if spec.year_range.end is not None:
            clauses.append("release_year <= %s")
            params.append(spec.year_range.end)
    if spec.min_rating is not None:
        clauses.append("COALESCE(vote_average, rating_mean * 2) >= %s")
        params.append(spec.min_rating)
    return clauses, params


def describe_filters(spec: QuerySpec) -> list[str]:
    """Human-readable filter chips for the ``why.matched_filters`` payload."""
    described: list[str] = []
    described.extend(f"genre: {g}" for g in spec.genres_include)
    described.extend(f"not genre: {g}" for g in spec.genres_exclude)
    if spec.year_range is not None:
        start, end = spec.year_range.start, spec.year_range.end
        if start is not None and end is not None:
            described.append(f"year: {start}-{end}" if start != end else f"year: {start}")
        elif start is not None:
            described.append(f"year: {start} or later")
        else:
            described.append(f"year: up to {end}")
    if spec.min_rating is not None:
        described.append(f"rating: >= {spec.min_rating:g}/10")
    return described


@dataclass(frozen=True)
class ResolvedReference:
    """A reference title matched to an indexed movie (factor may be NULL)."""

    movie_id: int
    title: str
    factor: np.ndarray | None


def _as_float_array(value: Any) -> np.ndarray:
    """pgvector ``Vector`` (0.5+) or any sequence -> float64 numpy array."""
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float64)


def _like_pattern(title: str) -> str:
    """Substring ILIKE pattern with user %/_/\\ neutralized."""
    escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def resolve_references(
    conn: psycopg.Connection, table: str, titles: list[str]
) -> list[ResolvedReference]:
    """Best indexed match per reference title (exact beats popular substring)."""
    _check_table(table)
    resolved: list[ResolvedReference] = []
    for title in titles:
        row = conn.execute(
            f"SELECT movie_id, title, factor FROM {table} "
            "WHERE title ILIKE %s "
            "ORDER BY (lower(title) = lower(%s)) DESC, rating_count DESC NULLS LAST "
            "LIMIT 1",
            (_like_pattern(title), title),
        ).fetchone()
        if row is not None:
            factor = None if row[2] is None else _as_float_array(row[2])
            resolved.append(ResolvedReference(movie_id=row[0], title=row[1], factor=factor))
    return resolved


def fetch_candidates(
    conn: psycopg.Connection,
    table: str,
    query_vector: np.ndarray,
    spec: QuerySpec,
    exclude_ids: list[int],
    pool: int = config.DISCOVER_CANDIDATE_POOL,
) -> list[dict[str, Any]]:
    """One filtered HNSW scan: nearest ``pool`` rows by embedding cosine."""
    _check_table(table)
    clauses, params = build_filter_clauses(spec)
    if exclude_ids:
        clauses.append("NOT (movie_id = ANY(%s))")
        params.append(exclude_ids)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        f"SELECT {_MOVIE_COLUMNS}, factor, "
        "1 - (embedding <=> %s) AS semantic "
        f"FROM {table} {where} "
        "ORDER BY embedding <=> %s "
        "LIMIT %s"
    )
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, [query_vector, *params, query_vector, pool]).fetchall()


def behavioral_scores(
    candidates: list[dict[str, Any]], references: list[ResolvedReference]
) -> np.ndarray | None:
    """Mean ALS-factor cosine of each candidate to the reference titles.

    ``None`` when no reference has a factor (signal absent — its hybrid
    weight is redistributed); ``NaN`` per candidate without a factor
    (unscorable — normalizes to the bottom of the signal, never a boost).
    """
    ref_factors = [r.factor for r in references if r.factor is not None]
    if not ref_factors or not candidates:
        return None
    dim = ref_factors[0].shape[0]
    matrix = np.zeros((len(candidates), dim), dtype=np.float64)
    has_factor = np.zeros(len(candidates), dtype=bool)
    for i, candidate in enumerate(candidates):
        factor = candidate.get("factor")
        if factor is not None:
            matrix[i] = _as_float_array(factor)
            has_factor[i] = True
    sims = np.mean([cosine_scores(ref, matrix) for ref in ref_factors], axis=0)
    sims[~has_factor] = np.nan
    return sims


def _round(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 4)


def _summary_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "movie_id": row["movie_id"],
        "tmdb_id": row["tmdb_id"],
        "title": row["title"],
        "release_year": row["release_year"],
        "overview": row["overview"],
        "genres": list(row["genres"] or []),
        "poster_url": poster_url(row["poster_path"]),
        "vote_average": row["vote_average"],
        "rating_count": row["rating_count"],
        "source": row["source"],
    }


def rank(
    candidates: list[dict[str, Any]],
    references: list[ResolvedReference],
    spec: QuerySpec,
    limit: int,
) -> list[DiscoverResult]:
    """Hybrid re-rank of the candidate pool -> final scored, explained page."""
    if not candidates:
        return []
    semantic = np.array([c["semantic"] for c in candidates], dtype=np.float64)
    behavioral = behavioral_scores(candidates, references)
    quality = np.array(
        [c["bayes_score"] if c["bayes_score"] is not None else np.nan for c in candidates],
        dtype=np.float64,
    )
    scores = combine_hybrid(
        {"semantic": semantic, "behavioral": behavioral, "quality": quality},
        config.HYBRID_WEIGHTS,
    )
    matched = describe_filters(spec)
    fan_titles = [r.title for r in references if r.factor is not None]

    results: list[DiscoverResult] = []
    for i in top_k_indices(scores, min(limit, len(candidates))):
        row = candidates[i]
        boost = _round(behavioral[i]) if behavioral is not None else None
        why = Why(
            semantic_similarity=_round(semantic[i]),
            behavioral_boost=boost,
            quality_score=_round(quality[i]),
            matched_filters=matched,
            liked_by_fans_of=fan_titles if boost is not None else [],
        )
        results.append(
            DiscoverResult(**_summary_fields(row), score=_round(scores[i]) or 0.0, why=why)
        )
    return results


def discover(
    conn: psycopg.Connection,
    table: str,
    spec: QuerySpec,
    query_vector: np.ndarray,
    limit: int,
) -> list[DiscoverResult]:
    """Full discover flow: resolve references -> candidates -> hybrid rank."""
    references = resolve_references(conn, table, spec.reference_titles)
    candidates = fetch_candidates(
        conn,
        table,
        query_vector,
        spec,
        exclude_ids=[r.movie_id for r in references],
    )
    return rank(candidates, references, spec, limit)


# --- Movie detail + more-like-this ------------------------------------------


def fetch_movie(conn: psycopg.Connection, table: str, movie_id: int) -> MovieDetail | None:
    _check_table(table)
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            f"SELECT {_MOVIE_COLUMNS} FROM {table} WHERE movie_id = %s",
            (movie_id,),
        ).fetchone()
    if row is None:
        return None
    return MovieDetail(
        **_summary_fields(row),
        keywords=list(row["keywords"] or []),
        runtime=row["runtime"],
        popularity=row["popularity"],
        vote_count=row["vote_count"],
        rating_mean=row["rating_mean"],
        bayes_score=row["bayes_score"],
    )


def _neighbors(
    conn: psycopg.Connection, table: str, movie_id: int, column: str, limit: int
) -> list[MovieSummary]:
    """Nearest neighbors of one movie by cosine distance on a vector column."""
    _check_table(table)
    if column not in ("embedding", "factor"):
        raise ValueError(f"Unexpected vector column {column!r}")
    sql = (
        f"SELECT {_MOVIE_COLUMNS} FROM {table} "
        f"WHERE movie_id <> %s AND {column} IS NOT NULL "
        f"AND (SELECT {column} FROM {table} WHERE movie_id = %s) IS NOT NULL "
        f"ORDER BY {column} <=> (SELECT {column} FROM {table} WHERE movie_id = %s) "
        "LIMIT %s"
    )
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(sql, (movie_id, movie_id, movie_id, limit)).fetchall()
    return [MovieSummary(**_summary_fields(row)) for row in rows]


def more_like_this(
    conn: psycopg.Connection,
    table: str,
    movie_id: int,
    limit: int = config.MORE_LIKE_THIS_LIMIT,
) -> list[SimilarList]:
    """Two labeled neighbor lists: semantic (embedding) and behavioral (ALS)."""
    return [
        SimilarList(
            label="Similar story & vibe (plot/genre embedding)",
            basis="embedding",
            results=_neighbors(conn, table, movie_id, "embedding", limit),
        ),
        SimilarList(
            label="Fans also loved (ALS collaborative filtering)",
            basis="als_factors",
            results=_neighbors(conn, table, movie_id, "factor", limit),
        ),
    ]
