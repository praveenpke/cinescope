"""Tests for --sample staging separation and per-table fraction logic."""

from __future__ import annotations

import pytest

from pipeline import config
from pipeline.sampling import (
    ALL_TABLES,
    DIMENSION_TABLES,
    FACT_TABLES,
    sample_fraction,
    staging_dir,
)


def test_staging_dirs_are_separate() -> None:
    full, sampled = staging_dir(sample=False), staging_dir(sample=True)
    assert full != sampled
    assert full == config.STAGING_DIR
    assert sampled == config.STAGING_SAMPLE_DIR


def test_every_table_classified_exactly_once() -> None:
    assert set(ALL_TABLES) == FACT_TABLES | DIMENSION_TABLES
    assert not FACT_TABLES & DIMENSION_TABLES


def test_full_mode_never_samples() -> None:
    for table in ALL_TABLES:
        assert sample_fraction(table, sample=False) is None


def test_sample_mode_samples_fact_tables_at_one_percent() -> None:
    for table in FACT_TABLES:
        assert sample_fraction(table, sample=True) == pytest.approx(0.01)


def test_sample_mode_keeps_dimension_tables_whole() -> None:
    for table in DIMENSION_TABLES:
        assert sample_fraction(table, sample=True) is None


def test_unknown_table_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown table"):
        sample_fraction("nonexistent", sample=True)
