"""Postgres access for the serving API.

Connections are short-lived (one per request) against the local pgvector
instance — simple, safe with ``--reload``, and plenty for a demo-scale app.
``DATABASE_URL`` comes from the environment / ``.env`` with the same
docker-compose default the pipeline uses (host port 5433).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import psycopg
from pgvector.psycopg import register_vector

from pipeline import config

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Table names are never user input; they are additionally allowlisted here
# before any SQL string composition.
ALLOWED_TABLES: tuple[str, ...] = (config.INDEX_TABLE, config.INDEX_TABLE_SAMPLE)


def database_url() -> str:
    return os.environ.get("DATABASE_URL", config.DEFAULT_DATABASE_URL)


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """One pgvector-enabled connection (read-only usage)."""
    with psycopg.connect(database_url()) as conn:
        register_vector(conn)
        yield conn


def table_exists(conn: psycopg.Connection, table: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()
    return row is not None and row[0] is not None


def resolve_table(conn: psycopg.Connection) -> str:
    """Pick the serving table: env override, else full index, else sample.

    ``CINESCOPE_TABLE`` forces a choice (must be an allowlisted name);
    otherwise prefer the full ``movies`` table and fall back to
    ``movies_sample`` so a sample-only checkout serves out of the box.
    """
    override = os.environ.get("CINESCOPE_TABLE")
    if override:
        if override not in ALLOWED_TABLES:
            raise ValueError(f"CINESCOPE_TABLE must be one of {ALLOWED_TABLES}, got {override!r}")
        if not table_exists(conn, override):
            raise RuntimeError(
                f"CINESCOPE_TABLE={override} does not exist — run `uv run pipeline index"
                f"{' --sample' if override == config.INDEX_TABLE_SAMPLE else ''}` first."
            )
        return override
    for table in ALLOWED_TABLES:
        if table_exists(conn, table):
            return table
    raise RuntimeError(
        "No serving table found. Build one with `uv run pipeline index --sample` "
        "(after ingest -> cf -> hydrate -> embed)."
    )
