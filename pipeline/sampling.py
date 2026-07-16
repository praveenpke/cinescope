"""Sample-mode staging logic shared by all pipeline jobs.

``--sample`` runs write 1% samples to ``data/staging_sample/`` so end-to-end
wiring can be verified in minutes, while full runs stay in ``data/staging/``.

Fact tables are sampled; small dimension tables are kept whole so joins in
later jobs (movies <-> links <-> genome-tags) stay referentially intact.
"""

from __future__ import annotations

from pathlib import Path

from pipeline import config

# Large fact tables that get row-sampled in --sample mode.
FACT_TABLES: frozenset[str] = frozenset({"ratings", "tags", "genome_scores", "tmdb_export"})
# Small dimension tables always ingested in full (sampling them would break joins).
DIMENSION_TABLES: frozenset[str] = frozenset({"movies", "links", "genome_tags"})

ALL_TABLES: tuple[str, ...] = (
    "ratings",
    "movies",
    "links",
    "tags",
    "genome_scores",
    "genome_tags",
    "tmdb_export",
)


def staging_dir(sample: bool) -> Path:
    """Root staging directory for the given mode."""
    return config.STAGING_SAMPLE_DIR if sample else config.STAGING_DIR


def sample_fraction(table: str, sample: bool) -> float | None:
    """Fraction to sample ``table`` at, or None for 'keep every row'."""
    if table not in FACT_TABLES and table not in DIMENSION_TABLES:
        raise ValueError(f"Unknown table: {table!r}")
    if sample and table in FACT_TABLES:
        return config.SAMPLE_FRACTION
    return None
