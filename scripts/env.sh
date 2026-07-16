# Source this before running any Spark/PySpark command:
#   source scripts/env.sh
#
# Cross-platform env for PySpark local mode. Detects the OS and exports only
# what that OS needs:
#   - Windows (git-bash): JAVA_HOME + HADOOP_HOME point at this author's local
#     JDK 17 / Hadoop (winutils.exe + hadoop.dll live in $HADOOP_HOME/bin and are
#     required for Spark writes on Windows). Override CINESCOPE_JAVA_HOME /
#     CINESCOPE_HADOOP_HOME if your paths differ.
#   - macOS: derives JAVA_HOME from `/usr/libexec/java_home -v 17` unless one is
#     already set. No winutils/HADOOP_HOME needed.
#   - Linux: respects an existing JAVA_HOME; otherwise leaves it to the caller
#     (most distros put `java` on PATH already). No winutils/HADOOP_HOME needed.
#
# This script never clobbers a JAVA_HOME you already exported.

_cinescope_uname="$(uname -s 2>/dev/null || echo unknown)"

case "$_cinescope_uname" in
  MINGW* | MSYS* | CYGWIN*)
    # --- Windows (git-bash / MSYS) ---
    # JAVA_HOME / HADOOP_HOME use Windows-style paths (consumed by the JVM
    # launcher and Hadoop native libs); PATH additions use POSIX-style paths.
    export JAVA_HOME="${CINESCOPE_JAVA_HOME:-D:\\tools\\jdk-17.0.19+10}"
    export HADOOP_HOME="${CINESCOPE_HADOOP_HOME:-D:\\tools\\hadoop}"
    export PATH="/d/tools/jdk-17.0.19+10/bin:/d/tools/hadoop/bin:$PATH"
    ;;
  Darwin)
    # --- macOS ---
    if [ -z "$JAVA_HOME" ]; then
      if [ -x /usr/libexec/java_home ]; then
        export JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null)"
      fi
    fi
    if [ -z "$JAVA_HOME" ]; then
      echo "scripts/env.sh: JAVA_HOME is unset and no JDK 17 was found via" \
           "/usr/libexec/java_home -v 17. Install JDK 17 or export JAVA_HOME." >&2
    fi
    # No HADOOP_HOME/winutils on macOS.
    ;;
  *)
    # --- Linux / other POSIX ---
    if [ -z "$JAVA_HOME" ]; then
      echo "scripts/env.sh: JAVA_HOME is unset. On Linux, install JDK 17 and" \
           "export JAVA_HOME (e.g. export JAVA_HOME=/usr/lib/jvm/java-17-openjdk)." \
           "winutils/HADOOP_HOME are Windows-only and not needed here." >&2
    fi
    # No HADOOP_HOME/winutils on Linux.
    ;;
esac

unset _cinescope_uname

# Point PySpark workers at the project venv python (path differs per OS).
_CINESCOPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -x "$_CINESCOPE_ROOT/.venv/Scripts/python.exe" ]; then
  export PYSPARK_PYTHON="$_CINESCOPE_ROOT/.venv/Scripts/python.exe"   # Windows venv
elif [ -x "$_CINESCOPE_ROOT/.venv/bin/python" ]; then
  export PYSPARK_PYTHON="$_CINESCOPE_ROOT/.venv/bin/python"          # mac/linux venv
fi
export PYSPARK_DRIVER_PYTHON="${PYSPARK_PYTHON:-python}"
unset _CINESCOPE_ROOT
