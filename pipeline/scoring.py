"""Ranking math shared by the offline eval harness and the serving API.

The hybrid ranker is a weighted combination of per-candidate signal arrays:

* ``semantic``   — cosine similarity between a query/profile embedding and
  each candidate's sentence-transformers embedding.
* ``behavioral`` — collaborative-filtering affinity (ALS user·movie factor
  dot product, or ALS-neighbor similarity at serving time).
* ``quality``    — the Bayesian-weighted rating score (a per-catalog prior,
  identical for every query).

Each component is min-max normalized to ``[0, 1]`` before weighting so the
weights in :data:`pipeline.config.HYBRID_WEIGHTS` are comparable across
signals with different natural scales. ``NaN`` entries mean "this candidate
cannot be scored by this signal" (e.g. no ALS factor for a title nobody
rated) and normalize to 0 — the bottom of the signal's range, never an
artificial boost.

The eval harness (``pipeline eval``) and the FastAPI ranking path must import
*this* module so offline metrics always measure the exact scoring code that
serves traffic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

_EPS: float = 1e-12


def l2_normalize(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    """Scale vectors to unit L2 norm (zero vectors stay zero)."""
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    return matrix / np.maximum(norms, _EPS)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against each row of ``matrix``."""
    if query.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {query.shape}")
    if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError(f"matrix shape {matrix.shape} incompatible with query {query.shape}")
    return l2_normalize(matrix) @ l2_normalize(query)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Map finite scores to ``[0, 1]``; NaN (unscorable) maps to 0.

    A constant finite vector maps to 0.5 everywhere (no information — the
    signal neither boosts nor buries any candidate).
    """
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.zeros_like(scores)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi - lo < _EPS:
        out = np.where(np.isfinite(scores), 0.5, np.nan)
    else:
        out = (scores - lo) / (hi - lo)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def combine_hybrid(
    components: Mapping[str, np.ndarray | None],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Weighted combination of normalized signal arrays -> one score array.

    ``None`` components are absent (e.g. no reference titles -> no behavioral
    signal at serving time); their weight is redistributed proportionally
    across the present components, so a two-signal query still produces
    scores on the same ``[0, 1]`` scale as a three-signal one.
    """
    present = {name: np.asarray(comp) for name, comp in components.items() if comp is not None}
    if not present:
        raise ValueError("combine_hybrid needs at least one non-None component")
    unknown = set(present) - set(weights)
    if unknown:
        raise ValueError(f"No weight configured for component(s): {sorted(unknown)}")
    lengths = {name: comp.shape for name, comp in present.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Component arrays must share one shape, got {lengths}")
    total = sum(weights[name] for name in present)
    if total <= 0:
        raise ValueError(f"Present-component weights must sum > 0, got {total}")
    combined = np.zeros(next(iter(present.values())).shape, dtype=np.float64)
    for name, comp in present.items():
        combined += (weights[name] / total) * minmax_normalize(comp)
    return combined


def top_k_indices(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the ``k`` highest scores, descending, deterministic.

    ``exclude`` is an optional index array (e.g. items the user already
    rated in training) forced to the very bottom before ranking. Ties break
    by ascending index (stable sort) so results are reproducible.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    ranked = np.asarray(scores, dtype=np.float64).copy()
    ranked = np.nan_to_num(ranked, nan=-np.inf)
    if exclude is not None and len(exclude) > 0:
        ranked[np.asarray(exclude, dtype=np.intp)] = -np.inf
    order = np.argsort(-ranked, kind="stable")
    return order[:k]
