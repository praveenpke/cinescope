# CineScope — Semantic Movie Discovery Engine

Hybrid semantic + behavioral movie discovery over the MovieLens 25M dataset
(25M+ ratings) and the TMDB catalog (1M+ titles).

**Stack:** Python 3.11 / PySpark (offline pipeline) · PostgreSQL + pgvector ·
FastAPI · sentence-transformers · React (later milestones).

## Quickstart

Prereqs: [uv](https://docs.astral.sh/uv/), Docker, JDK 17
(and on Windows: Hadoop `winutils.exe` — paths are wired in `scripts/env.sh`).

```bash
# 1. Install Python deps
uv sync

# 2. Start Postgres (pgvector) on host port 5433
docker compose up -d

# 3. Set up env (keys optional — see .env.example; ingest needs NO keys)
cp .env.example .env

# 4. JVM env for Spark (required before any pipeline job)
source scripts/env.sh

# 5. Fast end-to-end check: 1% sample ingest (~2 min)
uv run pipeline ingest --sample

# 6. Full ingest: downloads ml-25m.zip (~262 MB, MD5-verified) + the TMDB
#    daily-export ID file, converts everything to partitioned Parquet, and
#    asserts >=25M ratings / >=1M TMDB titles.
uv run pipeline ingest
```

Both downloads are public — no API keys needed for ingest. `TMDB_API_KEY` and
`ANTHROPIC_API_KEY` are only used by later stages (metadata hydration, query
parsing) and those stages fall back to a labeled offline mode when unset.

## Pipeline jobs

Run as `uv run pipeline <job> [--sample]`. Every job is checkpointed and
resumable; `--sample` writes 1% data to `data/staging_sample/` so wiring can be
verified in minutes before full runs (which write to `data/staging/`).

| Job | Status | What it does |
| --- | --- | --- |
| `ingest` | ✅ M1 | MovieLens 25M CSVs + TMDB daily export → partitioned Parquet, row-count assertions |
| `train-als` | M2 | Spark MLlib ALS → per-movie latent factors + behavioral stats |
| `hydrate` | M2 | TMDB detail fetcher (plots/posters), rate-limited + resumable |
| `embed` | M3 | sentence-transformers over plot+genres+keywords |
| `build-index` | M3 | Join factors + embeddings + metadata → Postgres/pgvector (HNSW) |
| `eval` | M4 | precision@k / recall@k on held-out ratings, CI eval gate |

Ingest chunking/resume: partial converts are skipped via done-markers under
`data/staging*/_done/`; you can also restrict work with
`uv run pipeline ingest --tables ratings genome_scores`.

## Layout

```
pipeline/        PySpark jobs + CLI (uv run pipeline <job>)
api/             FastAPI app (M5)
web/             React frontend (M6)
scripts/env.sh   JVM/Hadoop/venv env for Spark on Windows
docker-compose.yml  pgvector/pgvector:pg16 on host port 5433
data/            raw downloads + parquet staging (gitignored)
```

## Development

```bash
uv run pytest          # unit tests
uv run ruff check .    # lint
uv run ruff format .   # format
```
