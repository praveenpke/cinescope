"""CLI entry point: `uv run pipeline <job> [--sample]`.

Jobs are added milestone by milestone; M1 ships `ingest`.
Spark jobs need the JVM env first: `source scripts/env.sh`.
"""

from __future__ import annotations

import argparse

from pipeline.sampling import ALL_TABLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="CineScope offline pipeline (PySpark). Source scripts/env.sh first.",
    )
    sub = parser.add_subparsers(dest="job", required=True)

    ingest = sub.add_parser(
        "ingest",
        help="Download MovieLens 25M + TMDB daily export and convert to Parquet.",
    )
    ingest.add_argument(
        "--sample",
        action="store_true",
        help="Write 1%% samples of fact tables to data/staging_sample/ instead of "
        "full data to data/staging/.",
    )
    ingest.add_argument(
        "--tables",
        nargs="+",
        choices=ALL_TABLES,
        default=None,
        metavar="TABLE",
        help=f"Only convert these tables (resumable chunking). Choices: {', '.join(ALL_TABLES)}",
    )

    cf = sub.add_parser(
        "cf",
        help="Train Spark MLlib ALS on ratings (per-movie latent factors) + behavioral stats.",
    )
    cf.add_argument(
        "--sample",
        action="store_true",
        help="Use the 1%% sample staging area (data/staging_sample/).",
    )

    hydrate = sub.add_parser(
        "hydrate",
        help="Fetch TMDB details for MovieLens-linked + popular titles "
        "(falls back to MovieLens-derived records without TMDB_API_KEY).",
    )
    hydrate.add_argument(
        "--sample",
        action="store_true",
        help="Use the 1%% sample staging area (data/staging_sample/).",
    )

    embed = sub.add_parser(
        "embed",
        help="Encode hydrated titles (plot+genres+keywords) with sentence-transformers "
        "into checkpointed parquet shards.",
    )
    embed.add_argument(
        "--sample",
        action="store_true",
        help="Use the 1%% sample staging area (data/staging_sample/).",
    )

    index = sub.add_parser(
        "index",
        help="Join CF factors + embeddings + metadata and publish to Postgres/pgvector "
        "(movies table, HNSW indexes on both vector columns).",
    )
    index.add_argument(
        "--sample",
        action="store_true",
        help="Read the 1%% sample staging area and load the movies_sample table.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.job == "ingest":
        from pipeline.jobs import ingest as ingest_job

        ingest_job.run(sample=args.sample, tables=args.tables)
    elif args.job == "cf":
        from pipeline.jobs import cf as cf_job

        cf_job.run(sample=args.sample)
    elif args.job == "hydrate":
        from pipeline.jobs import hydrate as hydrate_job

        hydrate_job.run(sample=args.sample)
    elif args.job == "embed":
        from pipeline.jobs import embed as embed_job

        embed_job.run(sample=args.sample)
    elif args.job == "index":
        from pipeline.jobs import index as index_job

        index_job.run(sample=args.sample)


if __name__ == "__main__":
    main()
