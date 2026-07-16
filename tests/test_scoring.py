"""Hybrid scoring math (pipeline/scoring.py — shared with the serving API)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.scoring import (
    combine_hybrid,
    cosine_scores,
    l2_normalize,
    minmax_normalize,
    top_k_indices,
)


class TestCosineScores:
    def test_identical_orthogonal_opposite(self) -> None:
        query = np.array([1.0, 0.0])
        matrix = np.array([[2.0, 0.0], [0.0, 5.0], [-3.0, 0.0]])
        assert cosine_scores(query, matrix) == pytest.approx([1.0, 0.0, -1.0])

    def test_zero_vector_scores_zero(self) -> None:
        scores = cosine_scores(np.array([1.0, 0.0]), np.zeros((1, 2)))
        assert scores == pytest.approx([0.0])

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError):
            cosine_scores(np.zeros((2, 2)), np.zeros((2, 2)))
        with pytest.raises(ValueError):
            cosine_scores(np.zeros(3), np.zeros((2, 2)))

    def test_l2_normalize_rows(self) -> None:
        out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        assert out[0] == pytest.approx([0.6, 0.8])
        assert out[1] == pytest.approx([0.0, 0.0])


class TestMinmaxNormalize:
    def test_hand_computed(self) -> None:
        assert minmax_normalize(np.array([2.0, 4.0, 6.0])) == pytest.approx([0.0, 0.5, 1.0])

    def test_nan_means_unscorable_and_maps_to_zero(self) -> None:
        out = minmax_normalize(np.array([1.0, np.nan, 3.0]))
        assert out == pytest.approx([0.0, 0.0, 1.0])

    def test_constant_vector_is_uninformative(self) -> None:
        assert minmax_normalize(np.array([7.0, 7.0])) == pytest.approx([0.5, 0.5])

    def test_all_nan(self) -> None:
        assert minmax_normalize(np.array([np.nan, np.nan])) == pytest.approx([0.0, 0.0])


class TestCombineHybrid:
    def test_hand_computed_weighted_sum(self) -> None:
        # semantic normalizes to [0, .5, 1]; behavioral to [1, 0, .5]
        combined = combine_hybrid(
            {
                "semantic": np.array([0.0, 0.5, 1.0]),
                "behavioral": np.array([10.0, 8.0, 9.0]),
            },
            {"semantic": 0.75, "behavioral": 0.25},
        )
        assert combined == pytest.approx([0.25, 0.375, 0.875])

    def test_missing_component_weight_is_redistributed(self) -> None:
        semantic = np.array([0.0, 1.0])
        combined = combine_hybrid(
            {"semantic": semantic, "behavioral": None},
            {"semantic": 0.5, "behavioral": 0.5},
        )
        # only semantic present -> its weight renormalizes to 1.0
        assert combined == pytest.approx([0.0, 1.0])

    def test_all_missing_raises(self) -> None:
        with pytest.raises(ValueError):
            combine_hybrid({"semantic": None}, {"semantic": 1.0})

    def test_unknown_component_raises(self) -> None:
        with pytest.raises(ValueError):
            combine_hybrid({"mystery": np.array([1.0])}, {"semantic": 1.0})

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            combine_hybrid(
                {"semantic": np.array([1.0]), "behavioral": np.array([1.0, 2.0])},
                {"semantic": 0.5, "behavioral": 0.5},
            )

    def test_zero_total_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            combine_hybrid({"semantic": np.array([1.0])}, {"semantic": 0.0})


class TestTopKIndices:
    def test_descending_order(self) -> None:
        assert top_k_indices(np.array([0.1, 0.9, 0.5]), 3).tolist() == [1, 2, 0]

    def test_exclusion_forces_items_out(self) -> None:
        top = top_k_indices(np.array([0.9, 0.8, 0.1]), 2, exclude=np.array([0]))
        assert top.tolist() == [1, 2]

    def test_nan_ranks_last(self) -> None:
        assert top_k_indices(np.array([np.nan, 0.2, 0.3]), 3).tolist() == [2, 1, 0]

    def test_ties_break_by_ascending_index(self) -> None:
        assert top_k_indices(np.array([0.5, 0.5, 0.5]), 2).tolist() == [0, 1]

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            top_k_indices(np.array([1.0]), 0)
