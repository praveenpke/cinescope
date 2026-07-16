"""Checkpoint primitives shared by pipeline jobs.

Two flavors:

* **Done markers** — ``<staging>/_done/<step>.json`` files recording a row
  count (plus optional extra metadata). A job step whose marker exists is
  skipped on re-run; delete the marker to force a rebuild.
* **JSONL checkpoints** — append-only ``.jsonl`` files used by the TMDB
  fetcher so a long-running, rate-limited crawl can be interrupted and
  resumed without refetching completed IDs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def marker_path(staging: Path, step: str) -> Path:
    return staging / "_done" / f"{step}.json"


def read_marker(staging: Path, step: str) -> dict[str, Any] | None:
    """Return the marker payload for ``step``, or None if the step is not done."""
    marker = marker_path(staging, step)
    if not marker.exists():
        return None
    return json.loads(marker.read_text())


def write_marker(staging: Path, step: str, rows: int, **extra: Any) -> None:
    marker = marker_path(staging, step)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "rows": rows,
        "completed_at": datetime.now(UTC).isoformat(),
        **extra,
    }
    marker.write_text(json.dumps(payload, indent=2))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL checkpoint file (creates parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL checkpoint file (empty if missing).

    Tolerates a trailing partial line (e.g. the process was killed mid-write):
    the corrupt final line is dropped so the crawl resumes from the last
    complete record.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # partial trailing write — resume from here
            raise
    return records


def completed_ids(path: Path, id_field: str) -> set[int]:
    """IDs already present in a JSONL checkpoint (for resume-skip logic)."""
    return {int(rec[id_field]) for rec in read_jsonl(path) if rec.get(id_field) is not None}
