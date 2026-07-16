"""SparkSession factory for local-mode pipeline jobs (Windows-aware)."""

from __future__ import annotations

import logging
import os

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def get_spark(app_name: str) -> SparkSession:
    """Create (or reuse) a local[*] SparkSession.

    Fails fast with actionable instructions if the JVM environment is missing —
    on this machine Java is not on PATH, so ``source scripts/env.sh`` first.
    """
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        raise RuntimeError(
            "JAVA_HOME is not set. Run `source scripts/env.sh` before any Spark job "
            "(it exports JAVA_HOME, HADOOP_HOME, PATH, and PYSPARK_PYTHON). On "
            "mac/linux set your own JAVA_HOME (e.g. JAVA_HOME=$(/usr/libexec/"
            "java_home -v 17) on macOS)."
        )
    java_bin = "java.exe" if os.name == "nt" else "java"
    if not os.path.isfile(os.path.join(java_home, "bin", java_bin)):
        raise RuntimeError(
            f"JAVA_HOME={java_home!r} does not contain bin/{java_bin}. It points at a "
            "nonexistent or wrong JDK. scripts/env.sh ships this author's Windows "
            "paths — on mac/linux export your own JAVA_HOME for a real JDK 17."
        )
    if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
        raise RuntimeError(
            "HADOOP_HOME is not set (winutils.exe is required for Spark writes on "
            "Windows). Run `source scripts/env.sh` first."
        )
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession up: %s (Spark %s)", app_name, spark.version)
    return spark
