"""Timestamp-split correctness: the no-leakage guarantee behind the eval.

Runs a real local SparkSession (skipped without JAVA_HOME) because the split
is a genuine Spark window transformation — testing a python re-implementation
would prove nothing about the code that runs.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import pytest

from pipeline.jobs.evaluate import split_ratings
from tests.conftest import requires_spark

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

pytestmark = requires_spark

SCHEMA = "userId INT, movieId INT, rating DOUBLE, timestamp LONG"
HOLDOUT = 0.2


def _split_lists(
    spark: SparkSession, rows: list[tuple[int, int, float, int]], fraction: float = HOLDOUT
) -> dict[int, dict[str, list[tuple[int, int]]]]:
    """{userId: {'train': [(ts, movieId)], 'test': [...]}} from the Spark split."""
    df: DataFrame = spark.createDataFrame(rows, schema=SCHEMA)
    out: dict[int, dict[str, list[tuple[int, int]]]] = {}
    for row in split_ratings(df, fraction).collect():
        side = "test" if row["is_test"] else "train"
        out.setdefault(row["userId"], {"train": [], "test": []})[side].append(
            (row["timestamp"], row["movieId"])
        )
    return out


class TestSplitBoundary:
    def test_holdout_is_the_most_recent_20_percent(self, spark: SparkSession) -> None:
        rows = [(1, movie, 3.0, 1000 + movie) for movie in range(1, 11)]  # 10 ratings
        split = _split_lists(spark, rows)[1]
        assert len(split["test"]) == 2  # ceil(10 * 0.2)
        assert {ts for ts, _ in split["test"]} == {1009, 1010}  # the two newest

    def test_no_timestamp_leakage(self, spark: SparkSession) -> None:
        """Every held-out timestamp >= the user's max train timestamp."""
        rng = random.Random(7)
        rows = [
            (user, movie, float(rng.randint(1, 10)) / 2, rng.randint(0, 500))
            for user in range(1, 41)
            for movie in range(1, rng.randint(3, 30))  # >= 2 ratings per user
        ]
        for user, split in _split_lists(spark, rows).items():
            assert split["train"], f"user {user} lost all train rows"
            assert split["test"], f"user {user} has no held-out rows"
            max_train = max(ts for ts, _ in split["train"])
            min_test = min(ts for ts, _ in split["test"])
            assert min_test >= max_train, (
                f"user {user}: held-out ts {min_test} < train max {max_train}"
            )

    def test_holdout_count_is_ceil_capped_at_n_minus_1(self, spark: SparkSession) -> None:
        rows = [
            (user, movie, 4.0, movie)
            for user, n in ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 21))
            for movie in range(1, n + 1)
        ]
        split = _split_lists(spark, rows)
        for user, n in ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 21)):
            expected = min(math.ceil(n * HOLDOUT), n - 1)
            assert len(split[user]["test"]) == expected, f"user {user} (n={n})"
            assert len(split[user]["train"]) == n - expected

    def test_single_rating_user_keeps_it_for_training(self, spark: SparkSession) -> None:
        split = _split_lists(spark, [(9, 1, 5.0, 100)])
        assert split[9]["train"] == [(100, 1)]
        assert split[9]["test"] == []

    def test_tied_timestamps_at_the_boundary_cannot_leak(self, spark: SparkSession) -> None:
        """Ties order by movieId, so min(test ts) >= max(train ts) still holds."""
        rows = [(1, movie, 3.0, 50) for movie in (4, 2, 9, 7)] + [(1, 1, 3.0, 10)]
        split = _split_lists(spark, rows)[1]
        assert len(split["test"]) == 1
        ts, movie = split["test"][0]
        assert ts == 50
        assert movie == 9  # highest movieId among the tied newest timestamps
        assert max(t for t, _ in split["train"]) <= ts

    def test_split_is_a_partition_of_the_input(self, spark: SparkSession) -> None:
        rows = [(1, movie, 2.5, movie * 3) for movie in range(1, 8)]
        split = _split_lists(spark, rows)[1]
        assert sorted(split["train"] + split["test"]) == sorted(
            (ts, movie) for _, movie, _, ts in rows
        )

    def test_bad_fraction_rejected(self, spark: SparkSession) -> None:
        df = spark.createDataFrame([(1, 1, 1.0, 1)], schema=SCHEMA)
        for fraction in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                split_ratings(df, fraction)
