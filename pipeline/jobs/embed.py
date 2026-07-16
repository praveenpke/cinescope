"""Embedding job: sentence-transformers over composed text per hydrated title.

Reads ``<staging>/hydrated/`` (both TMDB and movielens_fallback records share
one schema), composes one text per title from plot + genres + keywords, and
encodes it with ``config.EMBED_MODEL_NAME`` (all-MiniLM-L6-v2, 384-dim, CPU
is fine).

Checkpointing: titles are sorted by ``movie_id`` and split into fixed-size
shards; each shard is written atomically (tmp file + rename) to
``<staging>/embeddings/shard_NNNNN.parquet``. A re-run skips every shard whose
file exists with the expected row count, so an interrupted run resumes at the
first incomplete shard. Delete ``<staging>/_done/embeddings.json`` *and* the
``embeddings/`` directory to force a rebuild (e.g. after re-hydrating).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pipeline import checkpoints, config, sampling
from pipeline.spark_utils import get_spark

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

EMBED_STEP = "embeddings"


def compose_embedding_text(
    title: str,
    overview: str | None,
    genres: Sequence[str] | None,
    keywords: Sequence[str] | None,
    release_year: int | None = None,
) -> str:
    """One embedding input per title: plot + genres + keywords.

    Works for both hydration modes:

    * TMDB records: overview is the plot, keywords are TMDB keywords.
    * movielens_fallback records: overview is None (skipped), keywords are
      the top genome tags — real MovieLens data, so the text stays honest.
    """
    parts = [f"{title} ({release_year})." if release_year else f"{title}."]
    if overview and overview.strip():
        parts.append(overview.strip())
    if genres:
        parts.append("Genres: " + ", ".join(genres) + ".")
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords) + ".")
    return " ".join(parts)


def shard_ranges(n_rows: int, shard_size: int) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` row ranges covering ``n_rows``."""
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    return [(start, min(start + shard_size, n_rows)) for start in range(0, n_rows, shard_size)]


def shard_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"shard_{index:05d}.parquet"


def shard_is_complete(path: Path, expected_rows: int) -> bool:
    """True if a shard file exists with exactly the expected row count.

    A row-count mismatch means the shard came from a different (stale)
    hydrated set — it gets rewritten rather than trusted.
    """
    import pyarrow.parquet as pq

    if not path.exists():
        return False
    try:
        return pq.read_metadata(path).num_rows == expected_rows
    except Exception:  # corrupt/partial file -> rewrite
        return False


def _write_shard(path: Path, movie_ids: list[int], vectors: Any) -> None:
    """Atomic parquet write: movie_id int32 + embedding list<float32>."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "movie_id": pa.array(movie_ids, type=pa.int32()),
            "embedding": pa.array([v.tolist() for v in vectors], type=pa.list_(pa.float32())),
        }
    )
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _load_titles(sample: bool) -> list[tuple[int, str]]:
    """(movie_id, composed_text) for every hydrated title, sorted by movie_id."""
    staging = sampling.staging_dir(sample)
    spark = get_spark("cinescope-embed")
    rows = (
        spark.read.parquet(str(staging / "hydrated"))
        .select("movie_id", "title", "release_year", "overview", "genres", "keywords")
        .orderBy("movie_id")
        .collect()
    )
    spark.stop()  # free the JVM before torch loads
    return [
        (
            int(r["movie_id"]),
            compose_embedding_text(
                r["title"], r["overview"], r["genres"], r["keywords"], r["release_year"]
            ),
        )
        for r in rows
    ]


def run(sample: bool = False) -> dict[str, Any]:
    """Run the embed job. Returns the marker payload."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    staging = sampling.staging_dir(sample)
    mode = "SAMPLE (1%)" if sample else "FULL"
    logger.info("=== CineScope embed — %s mode (staging: %s) ===", mode, staging)

    if checkpoints.read_marker(staging, "hydrated") is None:
        raise SystemExit(
            f"Missing staged 'hydrated'. Run `uv run pipeline hydrate"
            f"{' --sample' if sample else ''}` first."
        )
    marker = checkpoints.read_marker(staging, EMBED_STEP)
    if marker is not None:
        logger.info(
            "SKIP embed (done marker, %s rows). Delete %s and %s to rebuild.",
            f"{int(marker['rows']):,}",
            checkpoints.marker_path(staging, EMBED_STEP),
            staging / EMBED_STEP,
        )
        print(f"Embeddings (cached): {int(marker['rows']):,} rows")
        return marker

    titles = _load_titles(sample)
    logger.info("Composed %d embedding texts (example: %r)", len(titles), titles[0][1][:120])

    out_dir = staging / EMBED_STEP
    out_dir.mkdir(parents=True, exist_ok=True)
    ranges = shard_ranges(len(titles), config.EMBED_SHARD_SIZE)
    pending = [
        (i, lo, hi)
        for i, (lo, hi) in enumerate(ranges)
        if not shard_is_complete(shard_path(out_dir, i), hi - lo)
    ]
    logger.info(
        "%d shards of <=%d rows (%d complete from a previous run, %d to encode)",
        len(ranges),
        config.EMBED_SHARD_SIZE,
        len(ranges) - len(pending),
        len(pending),
    )

    if pending:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading model %s (CPU ok, first run downloads it)", config.EMBED_MODEL_NAME)
        model = SentenceTransformer(config.EMBED_MODEL_NAME)
        for i, lo, hi in pending:
            chunk = titles[lo:hi]
            vectors = model.encode(
                [text for _, text in chunk],
                batch_size=config.EMBED_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if vectors.shape[1] != config.EMBED_DIM:
                raise RuntimeError(
                    f"Model produced dim {vectors.shape[1]}, expected {config.EMBED_DIM}"
                )
            _write_shard(shard_path(out_dir, i), [mid for mid, _ in chunk], vectors)
            logger.info("  shard %05d written (rows %d..%d of %d)", i, lo, hi, len(titles))

    # Stale shards from a previously larger hydrated set would corrupt the join.
    for extra in sorted(out_dir.glob("shard_*.parquet")):
        if int(extra.stem.split("_")[1]) >= len(ranges):
            logger.warning("Removing stale shard %s", extra)
            extra.unlink()

    rows = len(titles)
    checkpoints.write_marker(
        staging,
        EMBED_STEP,
        rows,
        model=config.EMBED_MODEL_NAME,
        dim=config.EMBED_DIM,
        shards=len(ranges),
    )
    print(f"\n=== Embeddings ({mode} mode -> {out_dir}) ===")
    print(f"  rows: {rows:,}  dim: {config.EMBED_DIM}  shards: {len(ranges)}")
    return {"rows": rows, "model": config.EMBED_MODEL_NAME, "dim": config.EMBED_DIM}
