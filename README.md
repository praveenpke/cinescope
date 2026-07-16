# CineScope — Semantic Movie Discovery Engine

[![CI](https://github.com/praveenpke/cinescope/actions/workflows/ci.yml/badge.svg)](https://github.com/praveenpke/cinescope/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Made with PySpark](https://img.shields.io/badge/PySpark-3.5-e25a1c.svg)](https://spark.apache.org/)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-336791.svg)](https://github.com/pgvector/pgvector)

**Ask for movies the way you'd ask a friend — _"like Inception but funnier"_ —**
and get back a ranked grid that blends what the film is *about* (plot/genre
embeddings) with what people who liked your reference films *actually watched*
(collaborative filtering). CineScope is a full, runnable stack: a distributed
PySpark pipeline over **25M+ MovieLens ratings and 1M+ TMDB titles**, a
sentence-transformer + ALS vector index in **Postgres/pgvector**, a **FastAPI**
hybrid-retrieval service with an LLM query parser, and a **React** frontend —
with recommendation quality **scored offline (precision@k) before any ranking
change ships**.

![CineScope discovery results](docs/screenshots/02-results.png)

**Stack:** Python 3.11 · PySpark (MLlib ALS) · sentence-transformers
(all-MiniLM-L6-v2) · PostgreSQL + pgvector (HNSW) · FastAPI · Anthropic Claude
(query parsing) · React + Vite + TanStack Query.

> **Runs fully offline out of the box.** No API keys? CineScope hydrates
> metadata from MovieLens genome tags and parses queries with a deterministic
> heuristic — every result is clearly labeled. Drop a `TMDB_API_KEY` /
> `ANTHROPIC_API_KEY` into `.env` and it upgrades automatically on restart
> (see [Enabling real TMDB + Anthropic keys](#enabling-real-tmdb--anthropic-keys)).

---

## Architecture

```mermaid
flowchart LR
    subgraph sources[Real data]
        ML[(MovieLens 25M<br/>ratings + genome tags)]
        TMDBX[(TMDB daily export<br/>1M+ title IDs)]
        TMDBAPI{{TMDB API<br/>plots / posters}}
    end

    subgraph pipeline[PySpark pipeline · uv run pipeline JOB]
        ING[ingest<br/>CSV to Parquet]
        CF[cf<br/>MLlib ALS + Bayesian stats]
        HYD[hydrate<br/>TMDB client / MovieLens fallback]
        EMB[embed<br/>sentence-transformers]
        IDX[index<br/>join + load]
        EVAL[eval<br/>timestamp split precision@k]
    end

    subgraph store[Serving store]
        PG[(PostgreSQL + pgvector<br/>movies table<br/>vector 384 embedding<br/>vector 64 ALS factor<br/>HNSW cosine indexes)]
    end

    subgraph serve[Serving]
        API[FastAPI<br/>POST /api/discover<br/>GET /api/movies/:id]
        CLAUDE{{Claude Haiku<br/>query to spec}}
        WEB[React + Vite<br/>NL search + filter chips]
    end

    ML --> ING --> CF --> IDX
    TMDBX --> ING
    ING --> HYD
    TMDBAPI -.optional.-> HYD --> EMB --> IDX --> PG
    PG --> API --> WEB
    CLAUDE -.optional.-> API
    PG --> EVAL
    EVAL -->|precision@10 gate| CI[[CI eval gate]]
```

The pipeline is offline and checkpointed; the serving layer reads only from
Postgres. The **eval gate** (precision@10 on held-out ratings) sits in CI so a
ranking regression can never merge.

---

## Quickstart

A fresh clone to a working discovery UI in a handful of copy-paste commands.
Every command below was run to produce this README.

**Prereqs:** [uv](https://docs.astral.sh/uv/), Docker, JDK 17, Node 18+
(and on Windows: Hadoop `winutils.exe` — paths are wired in `scripts/env.sh`).

```bash
# 0. Clone
git clone https://github.com/praveenpke/cinescope.git && cd cinescope

# 1. Python deps
uv sync

# 2. Start Postgres + pgvector on host port 5433
docker compose up -d

# 3. Env file (keys optional — everything has a labeled offline fallback)
cp .env.example .env

# 4. JVM/Hadoop/venv env for Spark — required before ANY pipeline job
source scripts/env.sh

# 5. Build the sample index end-to-end (~5 min, 1% data, no keys needed).
#    Each job is checkpointed: re-running skips completed steps.
uv run pipeline ingest   --sample     # MovieLens + TMDB export -> Parquet
uv run pipeline cf       --sample     # ALS latent factors + Bayesian stats
uv run pipeline hydrate  --sample     # plots/tags (MovieLens fallback, no key)
uv run pipeline embed    --sample     # sentence-transformer embeddings
uv run pipeline index    --sample     # load Postgres movies_sample + HNSW
uv run pipeline eval     --sample     # offline precision@k + write results JSON

# 6. Serve the API (Spark env NOT needed here)
uv run uvicorn api.main:app --reload  # http://127.0.0.1:8000

# 7. Frontend (second terminal; Vite proxies /api -> :8000)
cd web && npm install && npm run dev  # http://localhost:5173
```

Try it:

```bash
curl -s -X POST http://127.0.0.1:8000/api/discover \
  -H "Content-Type: application/json" \
  -d '{"query": "like Jaws but funnier", "limit": 5}'
```

### Going full-scale later

Drop the `--sample` flag to run over the entire dataset. The full ingest
downloads `ml-25m.zip` (~262 MB, MD5-verified) plus the TMDB daily-export ID
file and asserts **≥25M ratings / ≥1M TMDB titles**. Full and sample runs use
separate staging areas (`data/staging/` vs `data/staging_sample/`) and separate
Postgres tables (`movies` vs `movies_sample`), so a smoke test never clobbers a
full build. The compute-heavy jobs (ALS, embeddings, eval) can take a while
full-scale, so they are checkpointed and resumable via done-markers under
`data/staging*/_done/` — delete a marker to force that step to rebuild.

---

## Pipeline jobs

Run as `uv run pipeline <job> [--sample]`. Every job is checkpointed and
resumable; `--sample` writes 1% data to `data/staging_sample/` so wiring can be
verified in minutes before full runs (which write to `data/staging/`).

| Job | Milestone | What it does |
| --- | --- | --- |
| `ingest` | M1 | MovieLens 25M CSVs + TMDB daily export → partitioned Parquet, row-count assertions (≥25M ratings / ≥1M titles) |
| `cf` | M2 | Spark MLlib ALS → per-movie latent factors + behavioral stats (Bayesian-weighted score) |
| `hydrate` | M2 | TMDB detail fetcher (plots/posters), rate-limited + resumable; offline MovieLens genome-tag fallback without a key |
| `embed` | M3 | sentence-transformers (all-MiniLM-L6-v2) over plot+genres+keywords → checkpointed parquet shards |
| `index` | M3 | Join factors + embeddings + metadata → Postgres table with `vector(384)` + `vector(64)` columns, HNSW cosine indexes on both |
| `eval` | M4 | Offline eval: per-user timestamp split (most recent 20% held out), precision/recall@{10,25} for embeddings-only / CF-only / hybrid → `eval/results/<git-sha>.json` |
| `eval-gate` | M4 | Fails (non-zero exit) if hybrid precision@10 in the newest results regresses vs `eval/baseline.json` — wired into CI |

Jobs run in order: `ingest → cf → hydrate → embed → index → eval` (each checks its
upstream done-markers and tells you what to run first). Restrict ingest work with
`uv run pipeline ingest --tables ratings genome_scores`.

---

## Offline eval + the ranking gate

**No ranking change ships without offline scoring.** The eval holds out each
user's most recent 20% of ratings (timestamp split, no leakage — ALS factors
and rating stats are retrained on the train split only) and scores three
rankers over the embedded catalog. The hybrid ranker uses the *same*
`pipeline/scoring.py` weighted combination the API serves, so the number in CI
is the number in production.

### Sample-mode results (committed baseline)

Generated with `uv run pipeline eval --sample` — **1% of ratings, 675-title
catalog, 23 evaluated users**. Absolute values are small at this scale; the
signal is the *ordering*: hybrid beats both single-signal rankers on every
metric. Regenerate after a full run for headline numbers.

| Ranker | precision@10 | precision@25 | recall@10 | recall@25 |
| --- | --- | --- | --- | --- |
| Embeddings only | 0.0043 | 0.0070 | 0.0435 | 0.1739 |
| CF only (ALS) | 0.0130 | 0.0226 | 0.1304 | 0.5652 |
| **Hybrid** | **0.0348** | **0.0278** | **0.3478** | **0.6957** |

The hybrid ranker's precision@10 (**0.0348**) is ~8× embeddings-only and ~2.7×
CF-only — the behavioral and semantic signals are complementary.
`eval/baseline.json` is the committed gate baseline (labeled `mode: sample`).

```bash
source scripts/env.sh
uv run pipeline eval --sample                 # writes eval/results/<git-sha>.json
uv run pipeline eval-gate                      # non-zero exit on precision@10 regression
uv run pipeline eval-gate --update-baseline    # promote good results, then commit both
```

`make eval-gate` wraps the gate. CI (GitHub Actions, `.github/workflows/ci.yml`)
runs ruff, pytest, and the gate in **committed-results mode**: it compares the
committed results JSON against the committed baseline — no Spark or data in CI,
so a ranking PR that skipped the offline eval (or regressed) fails the build.

---

## How the hybrid ranker works

Given a natural-language query, discovery is three stages — **parse → retrieve →
rank** — and the ranking is a tunable weighted blend of three signals:

| Signal | Weight | Source | "Why" it means |
| --- | --- | --- | --- |
| **Semantic** | 0.45 | Cosine similarity of the query embedding to each movie's plot+genre+keyword embedding (pgvector HNSW) | *"About the same thing"* |
| **Behavioral** | 0.40 | Mean ALS-factor cosine from the query's resolved `reference_titles` to the candidate | *"People who liked X also liked…"* |
| **Quality** | 0.15 | Bayesian-weighted rating score (shrinks sparse-vote movies toward the global mean) | *"…and it's actually good"* |

The weights live in one place — `config.HYBRID_WEIGHTS` — and are consumed by a
single function, `pipeline/scoring.combine_hybrid`, used **identically** by the
serving API (`api/retrieval.py`) and the offline eval (`pipeline/jobs/evaluate.py`).
Component handling is explicit: a `None` component (signal absent — e.g. no
reference titles, so no behavioral signal) is dropped and its weight
redistributed; a `NaN` component (present but unscorable) normalizes to 0.
Because eval and serving import the exact same code and weights, the offline
precision@10 gate genuinely guards the served ranking.

Every `/api/discover` result carries a `why` object exposing each signal's
contribution plus which spec filters matched — so the UI can explain *why this
movie*.

---

## Serving API (FastAPI)

After `index --sample` (or a full `index`) has populated Postgres:

```bash
uv run uvicorn api.main:app --reload
```

The app auto-selects the serving table (`movies` if a full index exists, else
`movies_sample`; override with `CINESCOPE_TABLE` in `.env`).

### `POST /api/discover` — natural-language discovery

```bash
curl -s -X POST http://127.0.0.1:8000/api/discover \
  -H "Content-Type: application/json" \
  -d '{"query": "like Jaws but funnier", "limit": 5}'
```

1. **Parse** — Claude Haiku (`claude-haiku-4-5`, official `anthropic` SDK with
   structured outputs) turns the query into a validated spec:
   `{reference_titles, mood_adjustments, genres_include/exclude, year_range,
   min_rating, similarity_text}`. Without `ANTHROPIC_API_KEY` a deterministic
   heuristic parser (regex years/decades, genre keyword map with negation,
   "like &lt;title&gt;" extraction) serves the same interface — responses are
   labeled `"parser": "heuristic_fallback"`. Sending a pre-parsed `"spec"` in
   the body skips parsing entirely (this powers editable filter chips: chips
   re-query without re-parsing).
2. **Retrieve** — the spec's `similarity_text` is embedded with the *same*
   composer + model the index used, then one filtered pgvector HNSW scan
   (cosine) pulls a candidate pool; spec filters are fully parameterized SQL.
3. **Rank** — candidates are re-ranked by `pipeline/scoring.combine_hybrid`
   under `config.HYBRID_WEIGHTS` — the exact function and weights the offline
   eval gate scores (see [How the hybrid ranker works](#how-the-hybrid-ranker-works)).

### `GET /api/movies/{id}` — detail + more-like-this

Returns the movie plus **two labeled** neighbor lists: semantic (embedding
HNSW) and behavioral (ALS-factor HNSW). `GET /api/health` reports the table,
title count, and active parser. CORS is open to the Vite dev server
(`http://localhost:5173`).

---

## Frontend (React + Vite + TanStack Query)

A polished dark UI for natural-language discovery, in `web/`.

```bash
uv run uvicorn api.main:app --reload   # terminal 1: API (Postgres up + indexed)

cd web
npm install
npm run dev            # terminal 2: http://localhost:5173 (Vite proxies /api -> :8000)
npm run build          # typecheck (tsc -b) + production bundle
```

- **Single natural-language search box** ("like Inception but funnier") →
  `POST /api/discover`.
- **Editable interpretation chips** — the parsed spec renders as chips. Removing
  a chip re-queries by POSTing the *modified spec* back in the `spec` field, so
  the backend **skips re-parsing** (`parser: "provided_spec"`). A subtle badge
  flags when the parse came from the heuristic fallback (no `ANTHROPIC_API_KEY`).
- **Results grid** — poster from the TMDB image CDN when `poster_path` exists,
  otherwise a styled deterministic gradient tile (fallback data has no posters —
  it still looks intentional). Each card shows the compact "why this matched"
  signals (semantic %, fans-of, quality ★, filters).
- **Detail drawer** — `GET /api/movies/{id}` with two labeled more-like-this
  rows: *Similar story & vibe* (embedding neighbors) and *Fans also loved* (ALS
  collaborative-filtering neighbors).
- Sensible loading (skeletons), empty, and error states throughout.

Screenshots live in [`docs/screenshots/`](docs/screenshots) and are
reproducible with `npm run screenshots` (Playwright, requires both servers).

| Landing | Detail drawer |
| --- | --- |
| ![Landing](docs/screenshots/01-landing.png) | ![Detail drawer](docs/screenshots/03-detail.png) |

---

## Enabling real TMDB + Anthropic keys

CineScope is fully functional offline; two keys unlock richer data. Add them to
`.env` (never committed — see `.env.example`) and restart the affected process.

| Key | Unlocks | Without it (offline fallback) |
| --- | --- | --- |
| `TMDB_API_KEY` | Real plots, posters, genres, ratings during `hydrate`. Rows are marked `source='tmdb'`. Get one (free) at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api). | `hydrate` builds metadata from MovieLens genome tags (real data), rows marked `source='movielens_fallback'`; `overview`/`poster_path`/`vote_average` are null. |
| `ANTHROPIC_API_KEY` | Claude Haiku query parsing — nuanced NL → structured spec. Get one at [console.anthropic.com](https://console.anthropic.com/settings/keys). | Deterministic heuristic parser (regex + genre keyword map); responses labeled `parser: "heuristic_fallback"`. |

To re-hydrate with a real TMDB key after already building the offline index:

```bash
# remove the stale hydrate + embeddings markers, then re-run downstream jobs
rm data/staging_sample/_done/hydrated.json data/staging_sample/_done/embeddings.json
rm -rf data/staging_sample/embeddings
source scripts/env.sh
uv run pipeline hydrate --sample && uv run pipeline embed --sample && uv run pipeline index --sample
```

The Anthropic key needs no rebuild — just add it to `.env` and restart uvicorn;
`/api/health` will report `parser: "claude"`.

---

## Resume claims → where in the code

Each claim on the résumé, mapped to the code that backs it:

| Claim | Where |
| --- | --- |
| **Distributed pipeline (PySpark) over 25M+ MovieLens ratings and 1M+ TMDB titles** | `pipeline/jobs/ingest.py` (Spark DataFrame CSV→Parquet, row-count assertions), `pipeline/spark_utils.py` (`local[*]` session), `pipeline/sampling.py` (fact/dimension split) |
| **Collaborative-filtering signals** | `pipeline/jobs/cf.py` — Spark **MLlib ALS** → per-movie latent factors + Bayesian-weighted stats (`config.BAYES_PRIOR_WEIGHT`) |
| **…joined with plot/genre embeddings** | `pipeline/jobs/embed.py` (`compose_embedding_text`, sentence-transformers all-MiniLM-L6-v2, checkpointed shards) |
| **…into a vector index** | `pipeline/jobs/index.py` — Postgres `movies` table, `vector(384)` embedding + `vector(64)` ALS factor, **HNSW cosine** indexes on both |
| **hybrid semantic + behavioral retrieval** | `api/retrieval.py` (pgvector HNSW scan + ALS-neighbor boost) + `pipeline/scoring.py` `combine_hybrid` under `config.HYBRID_WEIGHTS` |
| **Natural-language discovery in React ("like Inception but funnier")** | `web/src/App.tsx`, `web/src/components/SearchBar.tsx`, `web/src/lib/spec.ts` (editable filter chips) |
| **an LLM parses queries into structured filters plus embedding search** | `api/parsers.py` (`ClaudeParser` via `anthropic` structured outputs + `HeuristicParser` fallback), `api/schemas.py` (`QuerySpec`), `api/embedder.py` (query embedding, index parity) |
| **recommendation quality scored offline (precision@k on held-out ratings) before any ranking change ships** | `pipeline/jobs/evaluate.py` (timestamp split, precision/recall@k), `pipeline/jobs/eval_gate.py` + `.github/workflows/ci.yml` (CI gate), `eval/baseline.json` |

---

## Layout

```
pipeline/        PySpark jobs + CLI (uv run pipeline <job>)
api/             FastAPI serving layer (discover + movie detail)
web/             React frontend (Vite + TanStack Query)
eval/            committed eval results + gate baseline
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution workflow,
including the mandatory eval-gate re-score for any ranking change.

## License

[MIT](LICENSE) © 2026 Praveen Emani.
