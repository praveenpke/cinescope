# Source this before running any Spark/PySpark command:
#   source scripts/env.sh
#
# Windows + git-bash environment for PySpark local mode.
# - JAVA_HOME / HADOOP_HOME use Windows-style paths (consumed by the JVM launcher
#   and Hadoop native libs — winutils.exe/hadoop.dll live in $HADOOP_HOME/bin and
#   are required for Spark writes on Windows).
# - PATH additions use POSIX-style paths for git-bash.

export JAVA_HOME='D:\tools\jdk-17.0.19+10'
export HADOOP_HOME='D:\tools\hadoop'
export PATH="/d/tools/jdk-17.0.19+10/bin:/d/tools/hadoop/bin:$PATH"

# Point PySpark workers at the project venv python.
_CINESCOPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYSPARK_PYTHON="$_CINESCOPE_ROOT/.venv/Scripts/python.exe"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
unset _CINESCOPE_ROOT
