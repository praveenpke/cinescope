# Contributing to CineScope

Thanks for your interest! CineScope is a portfolio-grade reference implementation
of a hybrid semantic + behavioral recommender, so contributions that keep it
clean, runnable, and well-tested are very welcome.

## Getting set up

```bash
uv sync                       # Python deps (Python 3.11)
docker compose up -d          # Postgres + pgvector on host port 5433
cp .env.example .env          # keys optional — everything has an offline fallback
source scripts/env.sh         # JVM/Hadoop/venv env, required before any Spark job
```

See the [Quickstart](README.md#quickstart) in the README for the full
ingest → cf → hydrate → embed → index → eval → serve loop.

## Before you open a PR

Run the same checks CI runs:

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
uv run pytest                 # unit tests
```

If you touch **anything that affects ranking** (query parsing, retrieval SQL,
`pipeline/scoring.py`, embedding composition, or the hybrid weights), you must
re-score offline and commit the new results, or the CI eval gate fails:

```bash
source scripts/env.sh
uv run pipeline eval --sample     # writes eval/results/<git-sha>.json
uv run pipeline eval-gate         # non-zero exit = hybrid precision@10 regression
```

This is the "recommendation quality scored offline before any ranking change
ships" guarantee — it is a hard gate, not a suggestion.

## Conventions

- **Python:** type hints throughout, `ruff` clean. Spark jobs use genuine
  DataFrame/MLlib APIs (not pandas-with-a-Spark-import).
- **Frontend:** TypeScript; `web/src/lib/types.ts` mirrors `api/schemas.py` —
  update both together if the API contract changes.
- **Secrets:** never commit `.env` or `data/` (both gitignored). Every
  key-requiring boundary (TMDB, Anthropic) must keep its labeled offline
  fallback working when the key is absent.
- **Tests** on the risky seams: query-parse validation, hybrid score math,
  eval-split correctness (no timestamp leakage).

## Commit messages

Small, focused commits with a clear subject line. Reference the milestone or
module you touched.
