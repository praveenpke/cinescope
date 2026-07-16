"""Collaborative-filtering job: Spark MLlib ALS + behavioral stats.

Outputs (per staging area, each with its own done marker):

* ``cf_movie_factors/`` — per-movie latent factor vectors (``movie_id``,
  ``features: array<float>`` of length ``config.ALS_RANK``). These power the
  "people who liked X also liked" signal in the hybrid ranker.
* ``cf_user_factors/`` — per-user latent factors (kept for offline eval).
* ``cf_stats/`` — behavioral stats per movie: ``rating_count``,
  ``rating_mean``, and ``bayes_score``.

Bayesian-weighted score
-----------------------
A raw mean is unreliable for movies with few ratings (one 5-star vote should
not outrank a 4.3 average over 10,000 votes). We shrink each movie's mean
toward the global mean ``m`` using a conjugate-prior weight ``C``
(``config.BAYES_PRIOR_WEIGHT``), which acts like ``C`` virtual ratings at
``m``::

    bayes_score = (C * m + n * mean) / (C + n)

* ``n = 0``       -> score == global mean (pure prior)
* ``n -> inf``    -> score -> the movie's own mean (data overwhelms prior)
* ``n == C``      -> score is the midpoint of the two

This is the standard "IMDb weighted rating" / Dirichlet-smoothing formula.

Checkpointing: re-runs skip any step whose ``_done`` marker exists; delete
``data/staging*/_done/cf_*.json`` to force retraining.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pipeline import checkpoints, config, sampling
from pipeline.spark_utils import get_spark

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

STATS_STEP = "cf_stats"
FACTORS_STEP = "cf_movie_factors"
USER_FACTORS_STEP = "cf_user_factors"

# Works for plain floats (unit tests) and pyspark Columns (the job) alike.
Num = TypeVar("Num")


def bayesian_score(mean: Num, count: Num, global_mean: float, prior_weight: float) -> Num:
    """Shrink a per-movie mean toward the global mean (see module docstring).

    ``(C*m + n*mean) / (C + n)`` — written to work on both python floats
    (unit-testable) and Spark Columns (used in the job), which keeps the
    tested math and the executed math literally the same expression.
    """
    return (prior_weight * global_mean + count * mean) / (prior_weight + count)  # type: ignore[operator]


def compute_stats(ratings: DataFrame, prior_weight: float = config.BAYES_PRIOR_WEIGHT) -> DataFrame:
    """Per-movie rating_count / rating_mean / bayes_score."""
    from pyspark.sql import functions as F

    global_mean = ratings.select(F.avg("rating")).first()[0]
    logger.info("Global mean rating: %.4f (prior weight C=%s)", global_mean, prior_weight)
    per_movie = ratings.groupBy("movieId").agg(
        F.count("rating").alias("rating_count"),
        F.avg("rating").alias("rating_mean"),
    )
    return per_movie.select(
        F.col("movieId").alias("movie_id"),
        "rating_count",
        "rating_mean",
        bayesian_score(
            F.col("rating_mean"), F.col("rating_count"), float(global_mean), prior_weight
        ).alias("bayes_score"),
    )


def train_als(ratings: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Train MLlib ALS; return (movie_factors, user_factors) DataFrames."""
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import functions as F

    als = ALS(
        rank=config.ALS_RANK,
        maxIter=config.ALS_MAX_ITER,
        regParam=config.ALS_REG_PARAM,
        seed=config.ALS_SEED,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=False,
    )
    logger.info(
        "Training ALS: rank=%d maxIter=%d regParam=%s seed=%d",
        config.ALS_RANK,
        config.ALS_MAX_ITER,
        config.ALS_REG_PARAM,
        config.ALS_SEED,
    )
    model = als.fit(ratings.select("userId", "movieId", "rating"))
    movie_factors = model.itemFactors.select(F.col("id").alias("movie_id"), F.col("features"))
    user_factors = model.userFactors.select(F.col("id").alias("user_id"), F.col("features"))
    return movie_factors, user_factors


def _write_step(
    spark: SparkSession, df: DataFrame, staging: Path, step: str, **marker_extra: object
) -> int:
    out = staging / step
    df.coalesce(8).write.mode("overwrite").parquet(str(out))
    rows = spark.read.parquet(str(out)).count()
    checkpoints.write_marker(staging, step, rows, **marker_extra)
    logger.info("WROTE %-17s %s rows -> %s", step, f"{rows:,}", out)
    return rows


def run(sample: bool = False) -> dict[str, int]:
    """Run the CF job. Returns {step: row_count}."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    staging = sampling.staging_dir(sample)
    mode = "SAMPLE (1%)" if sample else "FULL"
    logger.info("=== CineScope cf — %s mode (staging: %s) ===", mode, staging)

    ratings_path = staging / "ratings"
    if (
        not (ratings_path / "_SUCCESS").exists()
        and checkpoints.read_marker(staging, "ratings") is None
    ):
        raise SystemExit(
            f"No staged ratings at {ratings_path}. Run `uv run pipeline ingest"
            f"{' --sample' if sample else ''}` first."
        )

    counts: dict[str, int] = {}
    spark = get_spark("cinescope-cf")
    ratings = spark.read.parquet(str(ratings_path))

    stats_marker = checkpoints.read_marker(staging, STATS_STEP)
    if stats_marker is not None:
        counts[STATS_STEP] = int(stats_marker["rows"])
        logger.info("SKIP %s (done marker, %s rows)", STATS_STEP, f"{counts[STATS_STEP]:,}")
    else:
        counts[STATS_STEP] = _write_step(
            spark,
            compute_stats(ratings),
            staging,
            STATS_STEP,
            prior_weight=config.BAYES_PRIOR_WEIGHT,
        )

    factors_marker = checkpoints.read_marker(staging, FACTORS_STEP)
    user_marker = checkpoints.read_marker(staging, USER_FACTORS_STEP)
    if factors_marker is not None and user_marker is not None:
        counts[FACTORS_STEP] = int(factors_marker["rows"])
        counts[USER_FACTORS_STEP] = int(user_marker["rows"])
        logger.info(
            "SKIP ALS training (done markers: %s movie factors, %s user factors)",
            f"{counts[FACTORS_STEP]:,}",
            f"{counts[USER_FACTORS_STEP]:,}",
        )
    else:
        movie_factors, user_factors = train_als(ratings)
        hyperparams = {
            "rank": config.ALS_RANK,
            "max_iter": config.ALS_MAX_ITER,
            "reg_param": config.ALS_REG_PARAM,
            "seed": config.ALS_SEED,
        }
        counts[FACTORS_STEP] = _write_step(
            spark, movie_factors, staging, FACTORS_STEP, **hyperparams
        )
        counts[USER_FACTORS_STEP] = _write_step(
            spark, user_factors, staging, USER_FACTORS_STEP, **hyperparams
        )

    example = spark.read.parquet(str(staging / FACTORS_STEP)).first()
    spark.stop()

    print(f"\n=== CF outputs ({mode} mode -> {staging}) ===")
    for step, rows in counts.items():
        print(f"  {step:<18} {rows:>12,}")
    if example is not None:
        print(
            f"  example factor: movie_id={example['movie_id']} "
            f"dim={len(example['features'])} "
            f"head={[round(float(x), 4) for x in example['features'][:4]]}"
        )
    return counts
