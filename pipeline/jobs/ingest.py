"""Ingest job: MovieLens 25M + TMDB daily export -> partitioned Parquet.

Steps (each checkpointed, so re-runs skip completed work):
  1. Download ml-25m.zip and verify it against the published MD5 checksum.
  2. Download the newest TMDB movie-IDs daily export (public, no API key).
  3. Extract the MovieLens archive.
  4. Convert each CSV (+ the TMDB export) to partitioned Parquet with genuine
     Spark DataFrame code, writing done-markers with row counts.
  5. Print a row-count summary and, in full mode, assert the dataset-scale
     claims: >=25,000,000 ratings and >=1,000,000 TMDB titles.

``--sample`` writes 1% samples of the fact tables to data/staging_sample/.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline import config, download, sampling
from pipeline.spark_utils import get_spark

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

_MOVIELENS_CSVS: dict[str, str] = {
    "ratings": "ratings.csv",
    "movies": "movies.csv",
    "links": "links.csv",
    "tags": "tags.csv",
    "genome_scores": "genome-scores.csv",
    "genome_tags": "genome-tags.csv",
}

_CSV_SCHEMAS: dict[str, str] = {
    "ratings": "userId INT, movieId INT, rating DOUBLE, timestamp LONG",
    "movies": "movieId INT, title STRING, genres STRING",
    "links": "movieId INT, imdbId STRING, tmdbId INT",
    "tags": "userId INT, movieId INT, tag STRING, timestamp LONG",
    "genome_scores": "movieId INT, tagId INT, relevance DOUBLE",
    "genome_tags": "tagId INT, tag STRING",
}


@dataclass(frozen=True)
class TableResult:
    table: str
    rows: int
    skipped: bool


def _marker_path(staging: Path, table: str) -> Path:
    return staging / "_done" / f"{table}.json"


def _read_marker(staging: Path, table: str) -> int | None:
    marker = _marker_path(staging, table)
    if not marker.exists():
        return None
    return int(json.loads(marker.read_text())["rows"])


def _write_marker(staging: Path, table: str, rows: int) -> None:
    marker = _marker_path(staging, table)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"rows": rows, "completed_at": datetime.now(UTC).isoformat()}, indent=2)
    )


def download_movielens() -> Path:
    """Download + MD5-verify the MovieLens 25M archive (skips if verified copy exists)."""
    md5_text = download.fetch_text(config.MOVIELENS_MD5_URL)
    expected = download.parse_md5_text(md5_text, config.MOVIELENS_ARCHIVE_NAME)
    logger.info("Published MD5 for %s: %s", config.MOVIELENS_ARCHIVE_NAME, expected)
    return download.download_file(
        config.MOVIELENS_URL,
        config.RAW_DIR / config.MOVIELENS_ARCHIVE_NAME,
        expected_md5=expected,
    )


def extract_movielens(archive: Path) -> Path:
    """Extract the MovieLens zip; skips if all expected CSVs are present."""
    extract_root = config.RAW_DIR
    target = extract_root / config.MOVIELENS_EXTRACT_DIRNAME
    expected = [target / name for name in _MOVIELENS_CSVS.values()]
    if all(p.exists() for p in expected):
        logger.info("SKIP extraction (all CSVs present): %s", target)
        return target
    logger.info("Extracting %s -> %s", archive, extract_root)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_root)
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"Extraction incomplete, missing: {missing}")
    return target


def _read_table(spark: SparkSession, table: str, ml_dir: Path, tmdb_export: Path) -> DataFrame:
    if table == "tmdb_export":
        # One JSON object per line: {"adult", "id", "original_title", "popularity", "video"}
        return spark.read.json(str(tmdb_export))
    return spark.read.csv(
        str(ml_dir / _MOVIELENS_CSVS[table]),
        header=True,
        schema=_CSV_SCHEMAS[table],
        escape='"',
    )


def _transform(df: DataFrame, table: str) -> tuple[DataFrame, list[str]]:
    """Table-specific column derivations; returns (df, partition_columns)."""
    from pyspark.sql import functions as F

    if table == "ratings":
        df = df.withColumn("rating_year", F.year(F.from_unixtime(F.col("timestamp"))))
        return df, ["rating_year"]
    if table == "tmdb_export":
        df = df.select(
            F.col("id").cast("long").alias("tmdb_id"),
            F.col("original_title"),
            F.col("popularity").cast("double"),
            F.col("adult").cast("boolean"),
            F.col("video").cast("boolean"),
        ).withColumn("bucket", F.pmod(F.col("tmdb_id"), F.lit(16)).cast("int"))
        return df, ["bucket"]
    return df, []


def convert_table(
    spark: SparkSession,
    table: str,
    ml_dir: Path,
    tmdb_export: Path,
    sample: bool,
) -> TableResult:
    """CSV/JSON -> partitioned Parquet for one table, with a done-marker checkpoint."""
    staging = sampling.staging_dir(sample)
    cached_rows = _read_marker(staging, table)
    if cached_rows is not None:
        logger.info("SKIP %-13s (done marker, %s rows)", table, f"{cached_rows:,}")
        return TableResult(table, cached_rows, skipped=True)

    df = _read_table(spark, table, ml_dir, tmdb_export)
    fraction = sampling.sample_fraction(table, sample)
    if fraction is not None:
        df = df.sample(fraction=fraction, seed=config.SAMPLE_SEED)
    df, partition_cols = _transform(df, table)

    out_path = staging / table
    out_df = df if partition_cols else df.coalesce(4)
    writer = out_df.write.mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(str(out_path))

    rows = spark.read.parquet(str(out_path)).count()
    _write_marker(staging, table, rows)
    logger.info("WROTE %-12s %s rows -> %s", table, f"{rows:,}", out_path)
    return TableResult(table, rows, skipped=False)


def run(sample: bool = False, tables: list[str] | None = None) -> dict[str, int]:
    """Run the ingest job. Returns {table: row_count}."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    mode = "SAMPLE (1%)" if sample else "FULL"
    logger.info("=== CineScope ingest — %s mode ===", mode)
    logger.info(
        "Note: neither download needs an API key (MovieLens is public; the TMDB "
        "daily-export ID file is public — TMDB_API_KEY is only needed later for "
        "detail hydration)."
    )

    archive = download_movielens()
    tmdb_export = download.download_tmdb_export(
        config.RAW_DIR / "tmdb",
        start=datetime.now(UTC).date(),
        url_template=config.TMDB_EXPORT_URL_TEMPLATE,
        date_format=config.TMDB_EXPORT_DATE_FORMAT,
        max_days_back=config.TMDB_EXPORT_MAX_DAYS_BACK,
    )
    ml_dir = extract_movielens(archive)

    selected = tables or list(sampling.ALL_TABLES)
    unknown = set(selected) - set(sampling.ALL_TABLES)
    if unknown:
        raise SystemExit(f"Unknown table(s): {sorted(unknown)}; valid: {sampling.ALL_TABLES}")

    spark = get_spark("cinescope-ingest")
    results = [convert_table(spark, t, ml_dir, tmdb_export, sample) for t in selected]
    spark.stop()

    counts = {r.table: r.rows for r in results}
    print(f"\n=== Ingest row counts ({mode} mode -> {sampling.staging_dir(sample)}) ===")
    for r in results:
        flag = " (cached)" if r.skipped else ""
        print(f"  {r.table:<14} {r.rows:>12,}{flag}")

    if not sample and set(selected) == set(sampling.ALL_TABLES):
        assert counts["ratings"] >= config.MIN_RATINGS_FULL, (
            f"Expected >= {config.MIN_RATINGS_FULL:,} ratings, got {counts['ratings']:,}"
        )
        assert counts["tmdb_export"] >= config.MIN_TMDB_TITLES_FULL, (
            f"Expected >= {config.MIN_TMDB_TITLES_FULL:,} TMDB titles, "
            f"got {counts['tmdb_export']:,}"
        )
        print(
            f"\nASSERTIONS PASSED: ratings {counts['ratings']:,} >= "
            f"{config.MIN_RATINGS_FULL:,} and TMDB titles {counts['tmdb_export']:,} >= "
            f"{config.MIN_TMDB_TITLES_FULL:,}"
        )
    return counts
