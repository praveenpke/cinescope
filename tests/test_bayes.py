"""Bayesian-weighted score math (pipeline/jobs/cf.py).

The formula is ``(C*m + n*mean) / (C + n)`` — written to run identically on
python floats (here) and Spark Columns (in the job).
"""

from __future__ import annotations

import pytest

from pipeline.jobs.cf import bayesian_score

GLOBAL_MEAN = 3.5
C = 50.0


def test_zero_ratings_returns_pure_prior() -> None:
    assert bayesian_score(0.0, 0, GLOBAL_MEAN, C) == pytest.approx(GLOBAL_MEAN)


def test_exact_formula() -> None:
    # 10 ratings averaging 5.0: (50*3.5 + 10*5.0) / 60 = 225/60 = 3.75
    assert bayesian_score(5.0, 10, GLOBAL_MEAN, C) == pytest.approx(3.75)


def test_count_equal_to_prior_weight_is_midpoint() -> None:
    score = bayesian_score(5.0, 50, GLOBAL_MEAN, C)
    assert score == pytest.approx((5.0 + GLOBAL_MEAN) / 2)


def test_large_count_converges_to_own_mean() -> None:
    score = bayesian_score(4.3, 1_000_000, GLOBAL_MEAN, C)
    assert score == pytest.approx(4.3, abs=1e-3)


def test_one_five_star_vote_does_not_beat_established_movie() -> None:
    lone_five_star = bayesian_score(5.0, 1, GLOBAL_MEAN, C)
    established = bayesian_score(4.3, 10_000, GLOBAL_MEAN, C)
    assert lone_five_star < established


def test_shrinkage_is_monotone_in_count() -> None:
    scores = [bayesian_score(5.0, n, GLOBAL_MEAN, C) for n in (0, 1, 10, 100, 1000)]
    assert scores == sorted(scores)
    assert all(GLOBAL_MEAN <= s <= 5.0 for s in scores)


def test_below_average_movie_is_pulled_up_toward_prior() -> None:
    assert bayesian_score(1.0, 5, GLOBAL_MEAN, C) > 1.0
    assert bayesian_score(1.0, 5, GLOBAL_MEAN, C) < GLOBAL_MEAN
