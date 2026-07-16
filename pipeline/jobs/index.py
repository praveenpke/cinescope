"""Index build: join CF factors + embeddings + metadata -> Postgres/pgvector.

Spark joins the staged artifacts (``hydrated`` metadata, ``embeddings``
shards, ``cf_movie_factors``, ``cf_stats``) into one row per title, then the
driver bulk-loads Postgres with two vector columns:

* ``embedding vector(384)``  — sentence-transformers text embedding
* ``factor    vector(<ALS_RANK>)`` — ALS latent factors (NULL for titles that
  never appear in the staged ratings; pgvector HNSW simply skips NULLs)

plus HNSW (``vector_cosine_ops``) indexes on both.

Idempotency choice: **drop-and-recreate** (not upsert). The index is a bulk
offline artifact published in one shot; rebuilding from parquet is cheap,
survives schema changes (e.g. a different ALS rank), and leaves no stale
rows behind. Sample runs load ``movies_sample`` so a full ``movies`` load is
never clobbered by a smoke test.

Connection comes from ``DATABASE_URL`` (.env or environment), defaulting to
the docker-compose Postgres on host port 5433.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from pipeline import checkpoints, config, sampling
from pipeline.envfile import load_dotenv
from pipeline.spark_utils import get_spark

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

INDEX_STEP = "pg_index"

# Column order shared by the Spark select, the INSERT statement, and
# row_to_record — keep these three in sync.
INDEX_COLUMNS: tuple[str, ...] = (
    "movie_id",
    "tmdb_id",
    "title",
    "release_year",
    "overview",
    "genres",
    "keywords",
    "poster_path",
    "vote_average",
    "vote_count",
    "popularity",
    "runtime",
    "source",
    "rating_count",
    "rating_mean",
    "bayes_score",
    "embedding",
    "factor",
)


def table_name(sample: bool) -> str:
    return config.INDEX_TABLE_SAMPLE if sample else config.INDEX_TABLE


def build_ddl(table: str, embed_dim: int, factor_dim: int) -> str:
    """CREATE TABLE statement with both pgvector columns."""
    return f"""
    CREATE TABLE {table} (
        movie_id      INTEGER PRIMARY KEY,
        tmdb_id       BIGINT,
        title         TEXT NOT NULL,
        release_year  INTEGER,
        overview      TEXT,
        genres        TEXT[] NOT NULL DEFAULT '{{}}',
        keywords      TEXT[] NOT NULL DEFAULT '{{}}',
        poster_path   TEXT,
        vote_average  DOUBLE PRECISION,
        vote_count    BIGINT,
        popularity    DOUBLE PRECISION,
        runtime       INTEGER,
        source        TEXT NOT NULL,
        rating_count  BIGINT,
        rating_mean   DOUBLE PRECISION,
        bayes_score   DOUBLE PRECISION,
        embedding     vector({embed_dim}) NOT NULL,
        factor        vector({factor_dim})
    )
    """


def hnsw_index_statements(table: str) -> list[str]:
    return [
        f"CREATE INDEX {table}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX {table}_factor_hnsw ON {table} USING hnsw (factor vector_cosine_ops)",
    ]


def row_to_record(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """One joined Spark row -> psycopg insert tuple (INDEX_COLUMNS order).

    Vector columns become float32 numpy arrays (pgvector's psycopg adapter);
    a missing ALS factor stays None -> SQL NULL. Genres/keywords become
    plain lists (NULL from the parquet side -> empty array).
    """
    import numpy as np

    def vec(values: Sequence[float] | None) -> Any:
        return None if values is None else np.asarray(values, dtype=np.float32)

    record: list[Any] = []
    for col in INDEX_COLUMNS:
        value = row[col]
        if col in ("embedding", "factor"):
            record.append(vec(value))
        elif col in ("genres", "keywords"):
            record.append(list(value) if value is not None else [])
        else:
            record.append(value)
    return tuple(record)


def chunked(records: Sequence[tuple[Any, ...]], size: int) -> Iterator[Sequence[tuple[Any, ...]]]:
    """Yield successive slices of ``records`` with at most ``size`` items."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _load_joined_rows(sample: bool) -> list[Any]:
    """Spark join of hydrated + embeddings + CF factors + behavioral stats."""
    from pyspark.sql import functions as F

    staging = sampling.staging_dir(sample)
    spark = get_spark("cinescope-index")
    hydrated = spark.read.parquet(str(staging / "hydrated"))
    embeddings = spark.read.parquet(str(staging / "embeddings"))
    factors = spark.read.parquet(str(staging / "cf_movie_factors")).select(
        "movie_id", F.col("features").alias("factor")
    )
    stats = spark.read.parquet(str(staging / "cf_stats"))

    joined = (
        hydrated.join(embeddings, "movie_id")  # inner: every indexed row has an embedding
        .join(factors, "movie_id", "left")  # NULL factor for titles absent from ratings
        .join(stats, "movie_id", "left")
        .select(*INDEX_COLUMNS)
        .orderBy("movie_id")
    )
    rows = joined.collect()
    spark.stop()
    return rows


def _publish(records: list[tuple[Any, ...]], table: str, factor_dim: int, url: str) -> None:
    """Drop-and-recreate ``table``, bulk-insert, then build both HNSW indexes."""
    import psycopg
    from pgvector.psycopg import register_vector

    placeholders = ", ".join(["%s"] * len(INDEX_COLUMNS))
    insert_sql = f"INSERT INTO {table} ({', '.join(INDEX_COLUMNS)}) VALUES ({placeholders})"

    with psycopg.connect(url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(build_ddl(table, config.EMBED_DIM, factor_dim))
        with conn.cursor() as cur:
            for i, batch in enumerate(chunked(records, config.PG_INSERT_BATCH_SIZE)):
                cur.executemany(insert_sql, batch)
                logger.info(
                    "  inserted batch %d (%d/%d rows)",
                    i + 1,
                    min((i + 1) * config.PG_INSERT_BATCH_SIZE, len(records)),
                    len(records),
                )
        for stmt in hnsw_index_statements(table):
            logger.info("Creating HNSW index: %s", stmt.split(" ON ")[0])
            conn.execute(stmt)
        conn.execute(f"ANALYZE {table}")
        conn.commit()


def _smoke_query(table: str, url: str) -> list[tuple[Any, ...]]:
    """Nearest-neighbor sanity check against the freshly built index."""
    import psycopg

    with psycopg.connect(url) as conn:
        anchor = conn.execute(
            f"SELECT movie_id, title FROM {table} ORDER BY rating_count DESC NULLS LAST LIMIT 1"
        ).fetchone()
        if anchor is None:
            return []
        neighbors = conn.execute(
            f"SELECT movie_id, title, "
            f"round((embedding <=> (SELECT embedding FROM {table} WHERE movie_id = %s))::numeric,"
            " 4) AS cos_dist "
            f"FROM {table} WHERE movie_id <> %s ORDER BY cos_dist LIMIT 5",
            (anchor[0], anchor[0]),
        ).fetchall()
        print(f"\n  smoke query — nearest neighbors of {anchor[1]!r} (movie_id={anchor[0]}):")
        for movie_id, title, dist in neighbors:
            print(f"    {dist}  {title} (movie_id={movie_id})")
        return neighbors


def run(sample: bool = False) -> dict[str, Any]:
    """Run the index-build job. Always reloads (drop-and-recreate is cheap)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_dotenv()
    staging = sampling.staging_dir(sample)
    mode = "SAMPLE (1%)" if sample else "FULL"
    table = table_name(sample)
    url = os.environ.get("DATABASE_URL", config.DEFAULT_DATABASE_URL)
    logger.info("=== CineScope index — %s mode -> table %s ===", mode, table)

    for dep in ("hydrated", "embeddings", "cf_movie_factors", "cf_stats"):
        if checkpoints.read_marker(staging, dep) is None:
            raise SystemExit(
                f"Missing staged '{dep}'. Run the earlier pipeline jobs"
                f"{' with --sample' if sample else ''} first (ingest -> cf -> hydrate -> embed)."
            )

    rows = _load_joined_rows(sample)
    if not rows:
        raise SystemExit("Join produced 0 rows — check the staged inputs.")
    records = [row_to_record(r) for r in rows]
    factor_dim = next(
        (
            len(rec[INDEX_COLUMNS.index("factor")])
            for rec in records
            if rec[INDEX_COLUMNS.index("factor")] is not None
        ),
        config.ALS_RANK,
    )
    if factor_dim != config.ALS_RANK:
        logger.warning("Staged factor dim %d != config.ALS_RANK %d", factor_dim, config.ALS_RANK)
    with_factor = sum(1 for rec in records if rec[INDEX_COLUMNS.index("factor")] is not None)

    logger.info("Publishing %d rows to %s (%d with ALS factors)", len(records), table, with_factor)
    _publish(records, table, factor_dim, url)

    checkpoints.write_marker(
        staging,
        INDEX_STEP,
        len(records),
        table=table,
        factor_dim=factor_dim,
        rows_with_factor=with_factor,
    )
    print(f"\n=== Index built ({mode} mode -> Postgres table {table}) ===")
    print(f"  rows: {len(records):,}  with ALS factor: {with_factor:,}")
    print(f"  vector columns: embedding vector({config.EMBED_DIM}), factor vector({factor_dim})")
    _smoke_query(table, url)
    return {"rows": len(records), "table": table, "rows_with_factor": with_factor}
