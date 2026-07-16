"""CineScope FastAPI app.

Run::

    uv run uvicorn api.main:app --reload

Endpoints:

* ``POST /api/discover``  — natural-language discovery. Body may carry a
  free-text ``query`` (parsed by Claude Haiku, or the deterministic heuristic
  fallback when ``ANTHROPIC_API_KEY`` is missing) **or** a pre-parsed
  ``spec`` (skips parsing entirely — powers editable filter chips).
* ``GET /api/movies/{id}`` — detail + two labeled more-like-this lists
  (semantic embedding neighbors vs ALS behavioral neighbors).
* ``GET /api/health``     — DB/table/parser status.

CORS is open to the Vite dev server (http://localhost:5173).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import db, embedder, parsers, retrieval
from api.schemas import DiscoverRequest, DiscoverResponse, MovieDetailResponse
from pipeline import config
from pipeline.envfile import load_dotenv

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger("api")

PARSER_PROVIDED_SPEC = "provided_spec"
CORS_ORIGINS = ["http://localhost:5173"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_dotenv()  # non-overriding; picks up ANTHROPIC_API_KEY / DATABASE_URL
    if parsers.claude_available():
        logger.info(
            "Query parsing: Claude (%s) with heuristic fallback on errors",
            config.CLAUDE_PARSE_MODEL,
        )
    else:
        logger.warning(
            "=== OFFLINE MODE: ANTHROPIC_API_KEY is not set ===\n"
            "Natural-language queries will be parsed by the deterministic heuristic parser\n"
            "(responses are marked parser='heuristic_fallback'). To enable Claude parsing:\n"
            "  1. Get a key at https://console.anthropic.com/settings/keys\n"
            "  2. Put ANTHROPIC_API_KEY=... in .env\n"
            "  3. Restart the server (model: %s)",
            config.CLAUDE_PARSE_MODEL,
        )
    try:
        with db.connect() as conn:
            table = db.resolve_table(conn)
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        logger.info(
            "Serving table %s (%s titles) at %s",
            table,
            f"{count[0]:,}" if count else "?",
            db.database_url(),
        )
    except Exception:
        logger.exception(
            "Postgres not reachable at startup — /api/* will return 503 until it is. "
            "Start it with `docker compose up -d` and build the index with "
            "`uv run pipeline index --sample`."
        )
    yield


app = FastAPI(title="CineScope API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(psycopg.OperationalError)
def _db_unavailable(request: Request, exc: psycopg.OperationalError) -> JSONResponse:
    logger.error("Database unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database unavailable — run `docker compose up -d` and "
            "`uv run pipeline index --sample`."
        },
    )


@app.get("/api/health")
def health() -> dict[str, object]:
    with db.connect() as conn:
        table = db.resolve_table(conn)
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
    return {
        "status": "ok",
        "table": table,
        "titles": count[0] if count else 0,
        "parser": parsers.PARSER_CLAUDE if parsers.claude_available() else parsers.PARSER_HEURISTIC,
    }


@app.post("/api/discover", response_model=DiscoverResponse)
def discover(request: DiscoverRequest) -> DiscoverResponse:
    if request.spec is not None:
        spec, parser_name = request.spec, PARSER_PROVIDED_SPEC  # chips: no re-parse
    else:
        spec, parser_name = parsers.parse_query(request.query)

    query_vector = embedder.embed_query(spec)
    with db.connect() as conn:
        table = db.resolve_table(conn)
        results = retrieval.discover(conn, table, spec, query_vector, request.limit)
    return DiscoverResponse(
        query=request.query,
        parser=parser_name,
        spec=spec,
        table=table,
        results=results,
    )


@app.get("/api/movies/{movie_id}", response_model=MovieDetailResponse)
def movie_detail(movie_id: int) -> MovieDetailResponse:
    with db.connect() as conn:
        table = db.resolve_table(conn)
        movie = retrieval.fetch_movie(conn, table, movie_id)
        if movie is None:
            raise HTTPException(status_code=404, detail=f"No movie with id {movie_id} in {table}")
        similar = retrieval.more_like_this(conn, table, movie_id)
    return MovieDetailResponse(movie=movie, more_like_this=similar)
