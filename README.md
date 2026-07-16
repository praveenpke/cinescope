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
| `cf` | ✅ M2 | Spark MLlib ALS → per-movie latent factors + behavioral stats (Bayesian-weighted score) |
| `hydrate` | ✅ M2 | TMDB detail fetcher (plots/posters), rate-limited + resumable; offline MovieLens fallback without a key |
| `embed` | ✅ M3 | sentence-transformers (all-MiniLM-L6-v2) over plot+genres+keywords → checkpointed parquet shards |
| `index` | ✅ M3 | Join factors + embeddings + metadata → Postgres table with `vector(384)` + `vector(64)` columns, HNSW cosine indexes on both |
| `eval` | ✅ M4 | Offline eval: per-user timestamp split (most recent 20% held out), precision/recall@{10,25} for embeddings-only / CF-only / hybrid rankers → `eval/results/<git-sha>.json` |
| `eval-gate` | ✅ M4 | Fails (non-zero exit) if hybrid precision@10 in the newest results regresses vs `eval/baseline.json` — wired into CI |

Jobs run in order: `ingest → cf → hydrate → embed → index → eval` (each checks its
upstream done-markers and tells you what to run first). `index --sample` loads
the `movies_sample` Postgres table so a full `movies` load is never clobbered
by a smoke test; the load is drop-and-recreate, so re-running is always safe.

Ingest chunking/resume: partial converts are skipped via done-markers under
`data/staging*/_done/`; you can also restrict work with
`uv run pipeline ingest --tables ratings genome_scores`.

## Offline eval + the ranking gate

No ranking change ships without offline scoring. The eval holds out each
user's most recent 20% of ratings (timestamp split, no leakage — ALS factors
and rating stats are retrained on the train split only) and scores three
rankers over the embedded catalog. The hybrid ranker uses the *same*
`pipeline/scoring.py` weighted combination the API serves.

```bash
source scripts/env.sh
uv run pipeline eval --sample        # writes eval/results/<git-sha>.json
uv run pipeline eval-gate            # non-zero exit on precision@10 regression
uv run pipeline eval-gate --update-baseline   # promote good results, commit both
```

`make eval-gate` wraps the gate. CI (GitHub Actions) runs ruff, pytest, and
the gate in **committed-results mode**: it compares the committed results JSON
against the committed baseline — no Spark or data in CI, so a ranking PR that
skipped the offline eval (or regressed) fails the build. The committed
baseline is generated in `--sample` mode (1% ratings, small catalog) and is
labeled as such inside the JSON; regenerate it after full runs.

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
