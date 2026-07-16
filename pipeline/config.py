"""Paths and constants shared across pipeline jobs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
STAGING_DIR: Path = DATA_DIR / "staging"
STAGING_SAMPLE_DIR: Path = DATA_DIR / "staging_sample"

MOVIELENS_URL: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MOVIELENS_MD5_URL: str = MOVIELENS_URL + ".md5"
MOVIELENS_ARCHIVE_NAME: str = "ml-25m.zip"
MOVIELENS_EXTRACT_DIRNAME: str = "ml-25m"

# TMDB publishes a daily export of all movie IDs — no API key required.
# http://files.tmdb.org/p/exports/movie_ids_MM_DD_YYYY.json.gz
TMDB_EXPORT_URL_TEMPLATE: str = "http://files.tmdb.org/p/exports/movie_ids_{date}.json.gz"
TMDB_EXPORT_DATE_FORMAT: str = "%m_%d_%Y"
TMDB_EXPORT_MAX_DAYS_BACK: int = 8

SAMPLE_FRACTION: float = 0.01
SAMPLE_SEED: int = 42

MIN_RATINGS_FULL: int = 25_000_000
MIN_TMDB_TITLES_FULL: int = 1_000_000
