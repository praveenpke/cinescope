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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.job == "ingest":
        from pipeline.jobs import ingest as ingest_job

        ingest_job.run(sample=args.sample, tables=args.tables)


if __name__ == "__main__":
    main()
