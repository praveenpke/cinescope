"""Env-guard tests for pipeline.spark_utils.get_spark.

These exercise the fail-fast validation without ever starting a JVM: the guards
raise before Spark is touched. This covers the cross-platform env.sh hardening
(scripts/env.sh ships this author's Windows paths; on a fresh mac/linux clone a
bogus JAVA_HOME must be rejected loudly rather than launching a broken JVM).
"""

from __future__ import annotations

import os

import pytest

from pipeline.spark_utils import get_spark


def test_missing_java_home_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAVA_HOME", raising=False)
    with pytest.raises(RuntimeError, match="JAVA_HOME is not set"):
        get_spark("test-app")


def test_invalid_java_home_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    # A JAVA_HOME that exists as a dir but has no bin/java must be rejected —
    # this is exactly the fresh-clone failure mode the review flagged.
    bogus = os.fspath(tmp_path)  # type: ignore[arg-type]
    monkeypatch.setenv("JAVA_HOME", bogus)
    monkeypatch.setenv("HADOOP_HOME", bogus)  # so the Windows HADOOP guard passes
    with pytest.raises(RuntimeError, match="does not contain bin/java"):
        get_spark("test-app")
