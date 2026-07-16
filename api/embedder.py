"""Query-time embeddings with index-time parity.

The query text is composed with the *same* :func:`compose_embedding_text`
the embed job used for every indexed title, and encoded with the same
sentence-transformers model — so query and document vectors live in one
space by construction, not by convention.

The model is loaded lazily on the first request (keeps app import cheap and
tests light) and cached for the process lifetime.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from pipeline import config
from pipeline.jobs.embed import compose_embedding_text

if TYPE_CHECKING:
    import numpy as np

    from api.schemas import QuerySpec

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def query_text(spec: QuerySpec) -> str:
    """Compose the embedding input for a parsed query (index-time format).

    ``similarity_text`` plays the title+overview role; included genres and
    mood adjustments land in the same ``Genres:`` / ``Keywords:`` slots the
    indexed documents use.
    """
    base = spec.similarity_text.strip()
    if not base:
        base = " ".join(spec.reference_titles + spec.mood_adjustments).strip() or "a movie"
    return compose_embedding_text(
        title=base,
        overview=None,
        genres=spec.genres_include or None,
        keywords=spec.mood_adjustments or None,
    )


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "Loading embedding model %s (first request only)", config.EMBED_MODEL_NAME
                )
                _model = SentenceTransformer(config.EMBED_MODEL_NAME)
    return _model


def embed_query(spec: QuerySpec) -> np.ndarray:
    """float32 vector for a parsed query, same model/dim as the index."""
    vector = _get_model().encode(query_text(spec), convert_to_numpy=True)
    if vector.shape[-1] != config.EMBED_DIM:
        raise RuntimeError(f"Model produced dim {vector.shape[-1]}, expected {config.EMBED_DIM}")
    return vector.astype("float32")
