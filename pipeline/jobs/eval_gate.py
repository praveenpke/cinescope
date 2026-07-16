"""Eval gate: block ranking changes that regress hybrid precision@10.

``uv run pipeline eval-gate`` (or ``make eval-gate``) compares the *newest*
results file in ``eval/results/`` (by its ``generated_at`` field) against the
committed baseline ``eval/baseline.json`` and exits non-zero if the hybrid
ranker's precision@10 dropped by more than ``config.EVAL_GATE_TOLERANCE``.

This is the "recommendation quality scored offline before any ranking change
ships" check. Two modes of use:

* **Locally** — run ``uv run pipeline eval --sample`` (or a full eval) after
  touching ranking code, then ``uv run pipeline eval-gate``. If the new
  numbers are good, promote them: ``uv run pipeline eval-gate
  --update-baseline`` and commit both files.
* **CI (committed-results mode)** — GitHub Actions runs the gate against the
  results committed in the PR. No Spark and no data in CI: the gate only
  reads JSON, so a ranking change that skipped the offline eval (or shipped
  a regression) fails the build.

Sample-mode results are only comparable to a sample-mode baseline (different
catalog + cohort), so the gate refuses to compare mismatched modes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline import config

logger = logging.getLogger(__name__)

GATE_RANKER = "hybrid"
GATE_METRIC = "precision@10"


class GateError(Exception):
    """Structural problem (missing files/fields) — the gate fails closed."""


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise GateError(f"Missing {path} — run `uv run pipeline eval` and commit the results.")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise GateError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError(f"{path} must contain a JSON object.")
    return payload


def newest_results(results_dir: Path) -> Path:
    """The results file with the latest ``generated_at`` (filename tiebreak)."""
    candidates = sorted(results_dir.glob("*.json"))
    if not candidates:
        raise GateError(
            f"No results in {results_dir} — run `uv run pipeline eval` (use --sample "
            "for the 1% staging area) and commit the JSON."
        )
    return max(candidates, key=lambda p: (str(_load(p).get("generated_at", "")), p.name))


def gate_metric(payload: dict[str, Any], source: Path) -> float:
    try:
        value = payload["rankers"][GATE_RANKER][GATE_METRIC]
    except (KeyError, TypeError) as exc:
        raise GateError(f"{source} lacks rankers.{GATE_RANKER}.{GATE_METRIC}") from exc
    return float(value)


def compare(baseline: dict[str, Any], results: dict[str, Any], tolerance: float) -> list[str]:
    """Return failure messages (empty list == gate passes)."""
    failures: list[str] = []
    if baseline.get("mode") != results.get("mode"):
        failures.append(
            f"mode mismatch: baseline is {baseline.get('mode')!r}, results are "
            f"{results.get('mode')!r} — regenerate the baseline for this mode."
        )
        return failures
    base = gate_metric(baseline, config.EVAL_BASELINE_PATH)
    new = gate_metric(results, Path("results"))
    if new < base - tolerance:
        failures.append(
            f"hybrid {GATE_METRIC} regressed: {new:.6f} < baseline {base:.6f} "
            f"(tolerance {tolerance})"
        )
    return failures


def update_baseline(results_path: Path, baseline_path: Path) -> None:
    """Promote a results file to the committed baseline (adds a label)."""
    payload = _load(results_path)
    mode = payload.get("mode", "unknown")
    payload["baseline_note"] = (
        f"Committed eval-gate baseline ({mode} mode"
        + (
            f", {payload['sample_fraction']:.0%} ratings sample"
            if payload.get("sample_fraction")
            else ""
        )
        + f"). Gate metric: {GATE_RANKER} {GATE_METRIC}. Promoted from {results_path.name}."
    )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Baseline updated: %s (from %s)", baseline_path, results_path)


def run(
    results_dir: Path | None = None,
    baseline_path: Path | None = None,
    promote: bool = False,
) -> int:
    """Run the gate; returns a process exit code (0 = pass)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results_dir = results_dir or config.EVAL_RESULTS_DIR
    baseline_path = baseline_path or config.EVAL_BASELINE_PATH
    try:
        results_path = newest_results(results_dir)
        results = _load(results_path)
        if promote:
            update_baseline(results_path, baseline_path)
        baseline = _load(baseline_path)
        failures = compare(baseline, results, config.EVAL_GATE_TOLERANCE)
        base = gate_metric(baseline, baseline_path)
        new = gate_metric(results, results_path)
    except GateError as exc:
        print(f"EVAL GATE ERROR: {exc}")
        return 2

    print(f"eval-gate: {GATE_RANKER} {GATE_METRIC}")
    print(f"  baseline {baseline_path.name}: {base:.6f} (mode={baseline.get('mode')})")
    print(
        f"  results  {results_path.name}: {new:.6f} "
        f"(mode={results.get('mode')}, generated {results.get('generated_at')})"
    )
    if failures:
        for failure in failures:
            print(f"EVAL GATE FAILED: {failure}")
        return 1
    print("EVAL GATE PASSED")
    return 0
