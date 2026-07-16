"""Tiny .env loader (stdlib-only) so pipeline jobs pick up keys from .env.

Values already present in the process environment always win — .env only
fills gaps. Secrets are never logged.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline import config


def parse_env_text(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignores blanks, comments, and malformed lines."""
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            result[key] = value
    return result


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load ``.env`` into ``os.environ`` (non-overriding). Returns loaded key names."""
    env_path = path or (config.REPO_ROOT / ".env")
    if not env_path.exists():
        return []
    loaded: list[str] = []
    for key, value in parse_env_text(env_path.read_text(encoding="utf-8")).items():
        if key not in os.environ and value:
            os.environ[key] = value
            loaded.append(key)
    return loaded
