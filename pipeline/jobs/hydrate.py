"""Hydrate job: per-title metadata (plot, genres, keywords, poster).

Two modes behind one interface, both writing the same parquet schema to
``<staging>/hydrated/``:

* **Real TMDB fetcher** (``TMDB_API_KEY`` set): rate-limited, resumable crawl
  of the MovieLens-linked titles (via links.csv tmdbId) plus the most popular
  export-only titles. Each fetched record is appended to a JSONL checkpoint,
  so an interrupted crawl resumes exactly where it stopped; dead IDs (404)
  are checkpointed too so they are never refetched. JSONL is converted to
  parquet at the end.
* **MovieLens fallback** (no key — clearly logged): hydrated records are
  built from real MovieLens data only — title/year from movies.csv, genres
  from movies.csv, and the top-``config.FALLBACK_TOP_TAGS`` genome tags by
  relevance as ``keywords``. Records are marked ``source='movielens_fallback'``.

Delete ``<staging>/_done/hydrated.json`` to force a rebuild (e.g. after
adding a TMDB key).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline import checkpoints, config, sampling
from pipeline.envfile import load_dotenv
from pipeline.spark_utils import get_spark
from pipeline.tmdb_client import TMDBClient, parse_movie_payload

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

HYDRATED_STEP = "hydrated"

# One schema for both modes — keeps M3 (embeddings) source-agnostic.
HYDRATED_SCHEMA = (
    "movie_id INT, tmdb_id LONG, title STRING, release_year INT, overview STRING, "
    "genres ARRAY<STRING>, keywords ARRAY<STRING>, poster_path STRING, "
    "vote_average DOUBLE, vote_count LONG, popularity DOUBLE, runtime INT, source STRING"
)

_FALLBACK_BANNER = """
================================================================================
TMDB_API_KEY is NOT set -> running hydration in OFFLINE FALLBACK mode.

Records are built from real MovieLens data (title/genres from movies.csv,
top-{top_tags} genome tags by relevance as keywords) and are marked
source='movielens_fallback'. No plot overviews or poster paths are available
in this mode.

To enable the real TMDB fetcher:
  1. Get a free v3 API key: https://www.themoviedb.org/settings/api
  2. Put it in .env at the repo root:  TMDB_API_KEY=<your key>
  3. Delete the done marker so hydration re-runs:
       rm {marker}
  4. Re-run:  uv run pipeline hydrate{sample_flag}
================================================================================
"""


def linked_targets(spark: SparkSession, staging: Path, sample: bool) -> DataFrame:
    """MovieLens-linked titles: movie_id, tmdb_id, ml_title, ml_genres.

    In --sample mode the (full) dimension tables are down-sampled here at
    ``config.SAMPLE_FRACTION`` so the hydration set stays proportionally
    small, deterministic (seeded), and fast to crawl.
    """
    from pyspark.sql import functions as F

    links = spark.read.parquet(str(staging / "links")).where(F.col("tmdbId").isNotNull())
    movies = spark.read.parquet(str(staging / "movies"))
    targets = links.join(movies, "movieId").select(
        F.col("movieId").alias("movie_id"),
        F.col("tmdbId").cast("long").alias("tmdb_id"),
        F.col("title").alias("ml_title"),
        F.col("genres").alias("ml_genres"),
    )
    if sample:
        targets = targets.sample(fraction=config.SAMPLE_FRACTION, seed=config.SAMPLE_SEED)
    return targets


def popular_export_targets(
    spark: SparkSession, staging: Path, linked: DataFrame, sample: bool
) -> DataFrame:
    """Top export-only titles by popularity (no MovieLens id): tmdb_id only."""
    from pyspark.sql import functions as F

    limit = (
        config.HYDRATE_POPULAR_EXPORT_LIMIT_SAMPLE
        if sample
        else config.HYDRATE_POPULAR_EXPORT_LIMIT
    )
    export = spark.read.parquet(str(staging / "tmdb_export"))
    return (
        export.where(~F.col("adult") & ~F.col("video"))
        .join(linked.select("tmdb_id"), "tmdb_id", "left_anti")
        .orderBy(F.col("popularity").desc())
        .limit(limit)
        .select("tmdb_id")
    )


def build_fallback(spark: SparkSession, staging: Path, targets: DataFrame) -> DataFrame:
    """Hydrated records from MovieLens data only (source='movielens_fallback').

    * title/release_year parsed from the MovieLens "Title (YYYY)" convention
    * genres from the pipe-separated movies.csv field
    * keywords = top-N genome tags by relevance per movie (real tag data)
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    scores = spark.read.parquet(str(staging / "genome_scores"))
    tags = spark.read.parquet(str(staging / "genome_tags"))
    rank_window = Window.partitionBy("movieId").orderBy(F.col("relevance").desc())
    top_tags = (
        scores.withColumn("rank", F.row_number().over(rank_window))
        .where(F.col("rank") <= config.FALLBACK_TOP_TAGS)
        .join(tags, "tagId")
        .groupBy("movieId")
        .agg(
            F.transform(
                F.sort_array(F.collect_list(F.struct("relevance", "tag")), asc=False),
                lambda pair: pair.tag,
            ).alias("keywords")
        )
        .select(F.col("movieId").alias("movie_id"), "keywords")
    )

    year_re = r"\((\d{4})\)\s*$"
    return targets.join(top_tags, "movie_id", "left").select(
        F.col("movie_id"),
        F.col("tmdb_id"),
        F.trim(F.regexp_replace("ml_title", year_re, "")).alias("title"),
        F.nullif(F.regexp_extract("ml_title", year_re, 1), F.lit(""))
        .cast("int")
        .alias("release_year"),
        F.lit(None).cast("string").alias("overview"),
        F.when(F.col("ml_genres") == "(no genres listed)", F.array().cast("array<string>"))
        .otherwise(F.split("ml_genres", r"\|"))
        .alias("genres"),
        F.coalesce("keywords", F.array().cast("array<string>")).alias("keywords"),
        F.lit(None).cast("string").alias("poster_path"),
        F.lit(None).cast("double").alias("vote_average"),
        F.lit(None).cast("long").alias("vote_count"),
        F.lit(None).cast("double").alias("popularity"),
        F.lit(None).cast("int").alias("runtime"),
        F.lit("movielens_fallback").alias("source"),
    )


def crawl_tmdb(
    client: TMDBClient,
    targets: list[tuple[int | None, int]],
    records_path: Path,
    misses_path: Path,
    progress_every: int = 100,
) -> tuple[int, int]:
    """Fetch every (movie_id, tmdb_id) target not already checkpointed.

    Returns (fetched_now, skipped_as_done). Fully resumable: both successful
    records and 404 misses live in JSONL checkpoints keyed by tmdb_id.
    """
    done = checkpoints.completed_ids(records_path, "tmdb_id") | checkpoints.completed_ids(
        misses_path, "tmdb_id"
    )
    pending = [(mid, tid) for mid, tid in targets if tid not in done]
    logger.info(
        "TMDB crawl: %d targets (%d already checkpointed, %d to fetch)",
        len(targets),
        len(targets) - len(pending),
        len(pending),
    )
    fetched = 0
    for movie_id, tmdb_id in pending:
        payload = client.fetch_movie(tmdb_id)
        if payload is None:
            checkpoints.append_jsonl(misses_path, {"tmdb_id": tmdb_id, "status": 404})
        else:
            checkpoints.append_jsonl(records_path, parse_movie_payload(payload, movie_id))
        fetched += 1
        if fetched % progress_every == 0:
            logger.info("  ... %d/%d fetched", fetched, len(pending))
    return fetched, len(targets) - len(pending)


def _jsonl_to_parquet(spark: SparkSession, records_path: Path, out_path: Path) -> None:
    df = spark.read.schema(HYDRATED_SCHEMA).json(str(records_path))
    df.coalesce(8).write.mode("overwrite").parquet(str(out_path))


def run(sample: bool = False) -> dict[str, Any]:
    """Run the hydrate job. Returns marker payload (rows + per-source counts)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    loaded = load_dotenv()
    if loaded:
        logger.info("Loaded %d key(s) from .env: %s", len(loaded), ", ".join(sorted(loaded)))
    staging = sampling.staging_dir(sample)
    mode = "SAMPLE (1%)" if sample else "FULL"
    logger.info("=== CineScope hydrate — %s mode (staging: %s) ===", mode, staging)

    for dep in ("links", "movies", "genome_scores", "genome_tags", "tmdb_export"):
        if checkpoints.read_marker(staging, dep) is None:
            raise SystemExit(
                f"Missing staged table '{dep}'. Run `uv run pipeline ingest"
                f"{' --sample' if sample else ''}` first."
            )

    marker = checkpoints.read_marker(staging, HYDRATED_STEP)
    if marker is not None:
        logger.info(
            "SKIP hydrate (done marker, %s rows, sources=%s). Delete %s to rebuild.",
            f"{int(marker['rows']):,}",
            marker.get("sources"),
            checkpoints.marker_path(staging, HYDRATED_STEP),
        )
        print(f"Hydrated (cached): {int(marker['rows']):,} rows, sources={marker.get('sources')}")
        return marker

    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    out_path = staging / HYDRATED_STEP
    spark = get_spark("cinescope-hydrate")
    linked = linked_targets(spark, staging, sample)

    if api_key:
        logger.info("TMDB_API_KEY detected — running the real TMDB fetcher.")
        popular = popular_export_targets(spark, staging, linked, sample)
        targets: list[tuple[int | None, int]] = [
            (int(r["movie_id"]), int(r["tmdb_id"]))
            for r in linked.select("movie_id", "tmdb_id").collect()
        ] + [(None, int(r["tmdb_id"])) for r in popular.collect()]
        targets.sort(key=lambda pair: pair[1])
        ckpt_dir = staging / "hydrate_checkpoints"
        records_path, misses_path = ckpt_dir / "records.jsonl", ckpt_dir / "misses.jsonl"
        client = TMDBClient(api_key)
        fetched, skipped = crawl_tmdb(client, targets, records_path, misses_path)
        logger.info("Crawl complete: %d fetched now, %d resumed from checkpoint", fetched, skipped)
        _jsonl_to_parquet(spark, records_path, out_path)
    else:
        print(
            _FALLBACK_BANNER.format(
                top_tags=config.FALLBACK_TOP_TAGS,
                marker=checkpoints.marker_path(staging, HYDRATED_STEP),
                sample_flag=" --sample" if sample else "",
            )
        )
        n_popular = (
            config.HYDRATE_POPULAR_EXPORT_LIMIT_SAMPLE
            if sample
            else config.HYDRATE_POPULAR_EXPORT_LIMIT
        )
        logger.info(
            "Fallback covers MovieLens-linked titles only; the %d popular export-only "
            "titles are skipped (no MovieLens metadata exists for them).",
            n_popular,
        )
        build_fallback(spark, staging, linked).write.mode("overwrite").parquet(str(out_path))

    hydrated = spark.read.parquet(str(out_path))
    rows = hydrated.count()
    sources = {r["source"]: r["count"] for r in hydrated.groupBy("source").count().collect()}
    example = hydrated.where(hydrated.keywords.isNotNull()).first()
    spark.stop()

    checkpoints.write_marker(staging, HYDRATED_STEP, rows, sources=sources)
    print(f"\n=== Hydrated titles ({mode} mode -> {out_path}) ===")
    print(f"  rows: {rows:,}  sources: {sources}")
    if example is not None:
        print(
            f"  example: movie_id={example['movie_id']} tmdb_id={example['tmdb_id']} "
            f"title={example['title']!r} year={example['release_year']} "
            f"genres={list(example['genres'] or [])} "
            f"keywords={list(example['keywords'] or [])[:5]}... source={example['source']}"
        )
    return {"rows": rows, "sources": sources}
