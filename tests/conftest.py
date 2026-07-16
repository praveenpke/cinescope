"""Shared fixtures. Spark-backed tests skip when the JVM env is missing.

Locally: ``source scripts/env.sh`` first so JAVA_HOME (and PYSPARK_PYTHON)
are set and the split tests actually run. On GitHub's ubuntu runners Java is
preinstalled with JAVA_HOME exported, so CI runs them too.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyspark.sql import SparkSession

requires_spark = pytest.mark.skipif(
    not os.environ.get("JAVA_HOME"),
    reason="JAVA_HOME not set — `source scripts/env.sh` to run Spark-backed tests",
)


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    from pipeline.spark_utils import get_spark

    session = get_spark("cinescope-tests")
    yield session
    session.stop()
