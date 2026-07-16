# CineScope dev targets. On Windows git-bash without make, run the underlying
# `uv run ...` commands directly (each target is a single command).

.PHONY: lint test eval eval-sample eval-gate update-baseline

lint:
	uv run ruff check . && uv run ruff format --check .

test:
	uv run pytest

# Full offline eval (needs staged data + `source scripts/env.sh` for Spark).
eval:
	uv run pipeline eval

# 1% sample eval — minutes, used to verify wiring and refresh sample results.
eval-sample:
	uv run pipeline eval --sample

# Fails (non-zero) if hybrid precision@10 in the newest eval/results/*.json
# regresses vs eval/baseline.json. JSON-only: no Spark, no data — CI runs this.
eval-gate:
	uv run pipeline eval-gate

# Promote the newest results file to the committed baseline (then commit both).
update-baseline:
	uv run pipeline eval-gate --update-baseline
