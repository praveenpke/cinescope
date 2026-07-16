"""Offline eval harness: precision@k / recall@k on a per-user timestamp split.

Protocol
--------
1. **Split** — for every user, hold out their most recent
   ``EVAL_HOLDOUT_FRACTION`` (20%) of ratings by timestamp as the test set
   (``ceil``, capped so at least one training rating remains). Ordering is
   ``(timestamp, movieId)`` so the boundary is deterministic and every test
   timestamp is >= the user's max train timestamp — no temporal leakage.
2. **Train-side artifacts** — ALS factors and Bayesian rating stats are
   retrained *on the train split only* (reusing :mod:`pipeline.jobs.cf`).
   The M2 ``cf_*`` artifacts saw the held-out ratings and would leak.
   Embeddings are text-derived (never touch ratings), so the staged
   embedding shards are reused as-is.
3. **Cohort** — evaluated users need >= 1 train rating on an embedded
   ("candidate") title (to build a semantic profile) and >= 1 held-out
   rating >= ``EVAL_POSITIVE_THRESHOLD`` on a candidate title (something to
   find). Capped at ``EVAL_MAX_USERS`` (ascending userId — deterministic).
4. **Rankers** — each ranks the full candidate catalog per user, excluding
   the user's train items:

   * ``embeddings_only`` — cosine(mean embedding of the user's liked train
     titles, candidate embedding).
   * ``cf_only``         — ALS user·movie factor dot product.
   * ``hybrid``          — ``scoring.combine_hybrid`` over semantic +
     behavioral + quality, weights from ``config.HYBRID_WEIGHTS``. This is
     the exact function the serving API imports.

5. **Metrics** — mean precision@k and recall@k (k = 10, 25) over the cohort,
   written to ``eval/results/<git-sha>.json`` for the eval gate.

Checkpointing: the split (``eval_train``/``eval_test``) and the retrained
artifacts (``eval_movie_factors``/``eval_user_factors``/``eval_stats``) live
in the staging area with done markers — delete the markers (and re-run) to
rebuild, e.g. after re-ingesting ratings. Metric computation itself is fast
and always re-runs.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline import checkpoints, config, sampling, scoring
from pipeline.jobs import cf
from pipeline.spark_utils import get_spark

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

TRAIN_STEP = "eval_train"
TEST_STEP = "eval_test"
MOVIE_FACTORS_STEP = "eval_movie_factors"
USER_FACTORS_STEP = "eval_user_factors"
STATS_STEP = "eval_stats"

RANKER_NAMES: tuple[str, ...] = ("embeddings_only", "cf_only", "hybrid")


# --------------------------------------------------------------------------
# Split + metrics (unit-tested seams)
# --------------------------------------------------------------------------


def split_ratings(ratings: DataFrame, holdout_fraction: float) -> DataFrame:
    """Add ``is_test``: True for each user's most recent ``holdout_fraction``.

    Rows are ordered per user by ``(timestamp, movieId)``; the top
    ``ceil(n * holdout_fraction)`` rows (capped at ``n - 1`` so every user
    keeps at least one training rating) become the test set. Because the
    primary sort key is the timestamp, ``min(test.timestamp) >=
    max(train.timestamp)`` holds for every user.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    order = Window.partitionBy("userId").orderBy(F.col("timestamp").asc(), F.col("movieId").asc())
    per_user = Window.partitionBy("userId")
    n = F.count(F.lit(1)).over(per_user)
    n_test = F.least(F.ceil(n * F.lit(holdout_fraction)), n - 1)
    return (
        ratings.withColumn("_rn", F.row_number().over(order))
        .withColumn("is_test", F.col("_rn") > (n - n_test))
        .drop("_rn")
    )


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant (divisor k)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of the relevant items found in the top-k (0.0 if none exist)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


# --------------------------------------------------------------------------
# Checkpointed Spark stages
# --------------------------------------------------------------------------


def _ensure_split(spark: SparkSession, staging: Path) -> None:
    """Materialize eval_train/ and eval_test/ parquet (skip if markers exist)."""
    if (
        checkpoints.read_marker(staging, TRAIN_STEP) is not None
        and checkpoints.read_marker(staging, TEST_STEP) is not None
    ):
        logger.info("SKIP split (done markers exist — delete _done/eval_*.json to redo)")
        return
    ratings = spark.read.parquet(str(staging / "ratings"))
    tagged = split_ratings(ratings, config.EVAL_HOLDOUT_FRACTION).cache()
    for step, keep_test in ((TRAIN_STEP, False), (TEST_STEP, True)):
        from pyspark.sql import functions as F

        subset = tagged.filter(F.col("is_test") == keep_test).drop("is_test")
        cf._write_step(spark, subset, staging, step, holdout_fraction=config.EVAL_HOLDOUT_FRACTION)
    tagged.unpersist()


def _ensure_train_artifacts(spark: SparkSession, staging: Path) -> None:
    """Retrain ALS + rating stats on the train split (skip if markers exist)."""
    if all(
        checkpoints.read_marker(staging, step) is not None
        for step in (MOVIE_FACTORS_STEP, USER_FACTORS_STEP, STATS_STEP)
    ):
        logger.info("SKIP train-split ALS/stats (done markers exist)")
        return
    train = spark.read.parquet(str(staging / TRAIN_STEP))
    cf._write_step(spark, cf.compute_stats(train), staging, STATS_STEP)
    movie_factors, user_factors = cf.train_als(train)
    cf._write_step(spark, movie_factors, staging, MOVIE_FACTORS_STEP, rank=config.ALS_RANK)
    cf._write_step(spark, user_factors, staging, USER_FACTORS_STEP, rank=config.ALS_RANK)


# --------------------------------------------------------------------------
# Driver-side ranking + metrics
# --------------------------------------------------------------------------


@dataclass
class _EvalInputs:
    cand_ids: list[int]  # sorted candidate movie ids (the embedded catalog)
    embeddings: np.ndarray  # (n_cand, EMBED_DIM) float32
    factors: np.ndarray  # (n_cand, ALS_RANK); NaN rows = no factor
    bayes: np.ndarray  # (n_cand,); NaN = never rated in train
    train_by_user: dict[int, list[tuple[int, float]]]  # userId -> [(movieId, rating)]
    relevant_by_user: dict[int, set[int]]  # userId -> relevant held-out movieIds
    user_factors: dict[int, np.ndarray]  # userId -> (ALS_RANK,)


def _collect_inputs(spark: SparkSession, staging: Path) -> _EvalInputs:
    """Spark joins -> compact numpy/dict structures for driver-side ranking."""
    from pyspark.sql import functions as F

    emb_rows = spark.read.parquet(str(staging / "embeddings")).orderBy("movie_id").collect()
    cand_ids = [int(r["movie_id"]) for r in emb_rows]
    idx = {mid: i for i, mid in enumerate(cand_ids)}
    embeddings = np.asarray([r["embedding"] for r in emb_rows], dtype=np.float32)

    cand_df = spark.createDataFrame([(mid,) for mid in cand_ids], schema="movie_id INT").cache()

    factors = np.full((len(cand_ids), config.ALS_RANK), np.nan, dtype=np.float32)
    for row in (
        spark.read.parquet(str(staging / MOVIE_FACTORS_STEP)).join(cand_df, "movie_id").collect()
    ):
        factors[idx[int(row["movie_id"])]] = row["features"]

    bayes = np.full(len(cand_ids), np.nan, dtype=np.float64)
    for row in spark.read.parquet(str(staging / STATS_STEP)).join(cand_df, "movie_id").collect():
        bayes[idx[int(row["movie_id"])]] = row["bayes_score"]

    train = spark.read.parquet(str(staging / TRAIN_STEP))
    test = spark.read.parquet(str(staging / TEST_STEP))
    cand_movie = cand_df.withColumnRenamed("movie_id", "movieId")
    train_c = train.join(cand_movie, "movieId").select("userId", "movieId", "rating").cache()
    relevant = (
        test.filter(F.col("rating") >= config.EVAL_POSITIVE_THRESHOLD)
        .join(cand_movie, "movieId")
        .select("userId", "movieId")
        .cache()
    )

    # Cohort: profile-able AND has something to find; deterministic cap.
    cohort_df = (
        train_c.select("userId")
        .distinct()
        .join(relevant.select("userId").distinct(), "userId")
        .orderBy("userId")
        .limit(config.EVAL_MAX_USERS)
        .cache()
    )

    train_by_user: dict[int, list[tuple[int, float]]] = {}
    for row in train_c.join(cohort_df, "userId").collect():
        train_by_user.setdefault(int(row["userId"]), []).append(
            (int(row["movieId"]), float(row["rating"]))
        )
    relevant_by_user: dict[int, set[int]] = {}
    for row in relevant.join(cohort_df, "userId").collect():
        relevant_by_user.setdefault(int(row["userId"]), set()).add(int(row["movieId"]))

    user_factors: dict[int, np.ndarray] = {}
    uf = spark.read.parquet(str(staging / USER_FACTORS_STEP)).withColumnRenamed("user_id", "userId")
    for row in uf.join(cohort_df, "userId").collect():
        user_factors[int(row["userId"])] = np.asarray(row["features"], dtype=np.float32)

    return _EvalInputs(
        cand_ids=cand_ids,
        embeddings=embeddings,
        factors=factors,
        bayes=bayes,
        train_by_user=train_by_user,
        relevant_by_user=relevant_by_user,
        user_factors=user_factors,
    )


def _semantic_profile(
    embeddings: np.ndarray, train_idx: list[int], train_ratings: list[float]
) -> np.ndarray:
    """Mean embedding of the user's liked train titles (all titles if none liked)."""
    liked = [
        i
        for i, rating in zip(train_idx, train_ratings, strict=True)
        if rating >= config.EVAL_POSITIVE_THRESHOLD
    ]
    rows = liked if liked else train_idx
    return embeddings[rows].mean(axis=0)


def _score_user(inputs: _EvalInputs, user_id: int) -> dict[str, np.ndarray] | None:
    """Per-ranker score arrays over the candidate catalog, or None to skip."""
    train = inputs.train_by_user[user_id]
    idx = {mid: i for i, mid in enumerate(inputs.cand_ids)}
    train_idx = [idx[mid] for mid, _ in train]
    train_ratings = [rating for _, rating in train]

    semantic = scoring.cosine_scores(
        _semantic_profile(inputs.embeddings, train_idx, train_ratings).astype(np.float64),
        inputs.embeddings.astype(np.float64),
    )

    user_factor = inputs.user_factors.get(user_id)
    if user_factor is None:  # shouldn't happen: cohort users have train ratings
        return None
    behavioral = inputs.factors.astype(np.float64) @ user_factor.astype(np.float64)

    return {
        "embeddings_only": semantic,
        "cf_only": behavioral,
        "hybrid": scoring.combine_hybrid(
            {"semantic": semantic, "behavioral": behavioral, "quality": inputs.bayes},
            config.HYBRID_WEIGHTS,
        ),
        "_train_idx": np.asarray(train_idx, dtype=np.intp),
    }


def compute_metrics(inputs: _EvalInputs, k_values: Sequence[int]) -> dict[str, Any]:
    """Mean precision/recall@k per ranker over the cohort."""
    sums: dict[str, dict[str, float]] = {
        name: {f"{metric}@{k}": 0.0 for metric in ("precision", "recall") for k in k_values}
        for name in RANKER_NAMES
    }
    n_users = 0
    max_k = max(k_values)
    for user_id in sorted(inputs.train_by_user):
        scored = _score_user(inputs, user_id)
        if scored is None:
            continue
        relevant = inputs.relevant_by_user[user_id]
        train_idx = scored["_train_idx"]
        n_users += 1
        for name in RANKER_NAMES:
            top = scoring.top_k_indices(scored[name], max_k, exclude=train_idx)
            ranked_ids = [inputs.cand_ids[i] for i in top]
            for k in k_values:
                sums[name][f"precision@{k}"] += precision_at_k(ranked_ids, relevant, k)
                sums[name][f"recall@{k}"] += recall_at_k(ranked_ids, relevant, k)
    if n_users == 0:
        raise SystemExit(
            "Eval cohort is empty: no user has both a train rating and a relevant "
            "held-out rating on embedded titles. Hydrate/embed more titles first."
        )
    rankers = {
        name: {metric: value / n_users for metric, value in metrics.items()}
        for name, metrics in sums.items()
    }
    return {"n_users_evaluated": n_users, "rankers": rankers}


# --------------------------------------------------------------------------
# Job entry point
# --------------------------------------------------------------------------


def _git_sha() -> tuple[str, str]:
    """(full sha, short sha) of HEAD; ('unknown', 'unknown') outside git."""
    try:
        sha = subprocess.run(  # fixed argv, no user input
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.REPO_ROOT,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"
    return sha, sha[:12]


def run(sample: bool = False) -> dict[str, Any]:
    """Run the eval job; write eval/results/<git-sha>.json and return it."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    staging = sampling.staging_dir(sample)
    mode = "sample" if sample else "full"
    logger.info("=== CineScope eval — %s mode (staging: %s) ===", mode.upper(), staging)

    for dep in ("ratings", "embeddings"):
        if checkpoints.read_marker(staging, dep) is None:
            raise SystemExit(
                f"Missing staged '{dep}'. Run the earlier jobs"
                f"{' with --sample' if sample else ''} first (ingest -> cf -> hydrate -> embed)."
            )

    spark = get_spark("cinescope-eval")
    _ensure_split(spark, staging)
    _ensure_train_artifacts(spark, staging)
    inputs = _collect_inputs(spark, staging)
    spark.stop()

    logger.info(
        "Cohort: %d users over %d candidate titles (cap %d)",
        len(inputs.train_by_user),
        len(inputs.cand_ids),
        config.EVAL_MAX_USERS,
    )
    metrics = compute_metrics(inputs, config.EVAL_K_VALUES)

    sha, short = _git_sha()
    results: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": sha,
        "mode": mode,
        "sample_fraction": config.SAMPLE_FRACTION if sample else None,
        "holdout_fraction": config.EVAL_HOLDOUT_FRACTION,
        "positive_threshold": config.EVAL_POSITIVE_THRESHOLD,
        "k_values": list(config.EVAL_K_VALUES),
        "hybrid_weights": config.HYBRID_WEIGHTS,
        "n_candidates": len(inputs.cand_ids),
        **metrics,
    }
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_RESULTS_DIR / f"{short}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n=== Eval results ({mode} mode) -> {out_path} ===")
    print(f"  users: {results['n_users_evaluated']:,}  candidates: {results['n_candidates']:,}")
    for name in RANKER_NAMES:
        row = results["rankers"][name]
        cells = "  ".join(f"{m}={row[m]:.4f}" for m in sorted(row))
        print(f"  {name:<16} {cells}")
    return results
