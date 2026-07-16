"""Eval-gate regression logic (pure JSON — this is what CI runs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.jobs import eval_gate


def _results(
    precision10: float,
    mode: str = "sample",
    generated_at: str = "2026-07-16T00:00:00+00:00",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "mode": mode,
        "rankers": {"hybrid": {"precision@10": precision10, "recall@10": 0.1}},
        **extra,
    }


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def eval_dirs(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "results", tmp_path / "baseline.json"


class TestCompare:
    def test_equal_metric_passes(self) -> None:
        assert eval_gate.compare(_results(0.25), _results(0.25), tolerance=1e-9) == []

    def test_improvement_passes(self) -> None:
        assert eval_gate.compare(_results(0.25), _results(0.30), tolerance=1e-9) == []

    def test_regression_fails(self) -> None:
        failures = eval_gate.compare(_results(0.25), _results(0.20), tolerance=1e-9)
        assert len(failures) == 1
        assert "regressed" in failures[0]

    def test_tolerance_absorbs_float_noise_only(self) -> None:
        base = _results(0.25)
        assert eval_gate.compare(base, _results(0.25 - 1e-12), tolerance=1e-9) == []
        assert eval_gate.compare(base, _results(0.25 - 1e-3), tolerance=1e-9) != []

    def test_mode_mismatch_fails(self) -> None:
        failures = eval_gate.compare(
            _results(0.25, mode="sample"), _results(0.99, mode="full"), tolerance=1e-9
        )
        assert len(failures) == 1
        assert "mode mismatch" in failures[0]

    def test_missing_metric_fails_closed(self) -> None:
        broken = {"mode": "sample", "rankers": {"hybrid": {}}}
        with pytest.raises(eval_gate.GateError):
            eval_gate.compare(_results(0.25), broken, tolerance=1e-9)


class TestNewestResults:
    def test_picks_latest_generated_at_not_filename(self, tmp_path: Path) -> None:
        _write(tmp_path / "zzz.json", _results(0.1, generated_at="2026-01-01T00:00:00+00:00"))
        newest = _write(
            tmp_path / "aaa.json", _results(0.2, generated_at="2026-07-01T00:00:00+00:00")
        )
        assert eval_gate.newest_results(tmp_path) == newest

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(eval_gate.GateError):
            eval_gate.newest_results(tmp_path)


class TestRunExitCodes:
    def test_pass_returns_zero(self, eval_dirs: tuple[Path, Path]) -> None:
        results_dir, baseline = eval_dirs
        _write(baseline, _results(0.25))
        _write(results_dir / "abc.json", _results(0.25))
        assert eval_gate.run(results_dir=results_dir, baseline_path=baseline) == 0

    def test_regression_returns_nonzero(self, eval_dirs: tuple[Path, Path]) -> None:
        results_dir, baseline = eval_dirs
        _write(baseline, _results(0.25))
        _write(results_dir / "abc.json", _results(0.10))
        assert eval_gate.run(results_dir=results_dir, baseline_path=baseline) == 1

    def test_gate_compares_against_newest_results(self, eval_dirs: tuple[Path, Path]) -> None:
        """An old good run must not mask a newer regression."""
        results_dir, baseline = eval_dirs
        _write(baseline, _results(0.25))
        _write(results_dir / "old.json", _results(0.30, generated_at="2026-01-01T00:00:00+00:00"))
        _write(results_dir / "new.json", _results(0.10, generated_at="2026-07-01T00:00:00+00:00"))
        assert eval_gate.run(results_dir=results_dir, baseline_path=baseline) == 1

    def test_missing_baseline_returns_two(self, eval_dirs: tuple[Path, Path]) -> None:
        results_dir, baseline = eval_dirs
        _write(results_dir / "abc.json", _results(0.25))
        assert eval_gate.run(results_dir=results_dir, baseline_path=baseline) == 2

    def test_missing_results_returns_two(self, eval_dirs: tuple[Path, Path]) -> None:
        results_dir, baseline = eval_dirs
        results_dir.mkdir()
        _write(baseline, _results(0.25))
        assert eval_gate.run(results_dir=results_dir, baseline_path=baseline) == 2

    def test_update_baseline_promotes_newest_and_passes(self, eval_dirs: tuple[Path, Path]) -> None:
        results_dir, baseline = eval_dirs
        _write(baseline, _results(0.50))  # would fail without promotion
        _write(results_dir / "abc.json", _results(0.10, sample_fraction=0.01))
        code = eval_gate.run(results_dir=results_dir, baseline_path=baseline, promote=True)
        assert code == 0
        promoted = json.loads(baseline.read_text())
        assert promoted["rankers"]["hybrid"]["precision@10"] == 0.10
        assert "baseline_note" in promoted
        assert "sample" in promoted["baseline_note"]
