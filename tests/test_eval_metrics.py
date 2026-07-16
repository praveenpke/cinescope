"""precision@k / recall@k on tiny hand-computed fixtures (no Spark)."""

from __future__ import annotations

import pytest

from pipeline.jobs.evaluate import precision_at_k, recall_at_k

RANKED = [10, 20, 30, 40, 50]  # recommendation order
RELEVANT = {20, 50, 99}  # 99 was never recommended


class TestPrecisionAtK:
    def test_hand_computed_at_5(self) -> None:
        # hits in top-5: 20 (rank 2) and 50 (rank 5) -> 2/5
        assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.4)

    def test_hand_computed_at_2(self) -> None:
        # top-2 = [10, 20]; only 20 is relevant -> 1/2
        assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)

    def test_divisor_stays_k_when_fewer_recommendations(self) -> None:
        # standard definition: 1 hit in a 2-item list at k=10 -> 1/10, not 1/2
        assert precision_at_k([20, 30], RELEVANT, 10) == pytest.approx(0.1)

    def test_no_hits(self) -> None:
        assert precision_at_k([1, 2, 3], RELEVANT, 3) == 0.0

    def test_perfect_top_k(self) -> None:
        assert precision_at_k([20, 50], {20, 50}, 2) == 1.0

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k(RANKED, RELEVANT, 0)


class TestRecallAtK:
    def test_hand_computed_at_5(self) -> None:
        # 2 of the 3 relevant items appear in the top-5 -> 2/3
        assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)

    def test_hand_computed_at_2(self) -> None:
        # top-2 finds only item 20 -> 1/3
        assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)

    def test_empty_relevant_set_scores_zero(self) -> None:
        assert recall_at_k(RANKED, set(), 5) == 0.0

    def test_all_relevant_found(self) -> None:
        assert recall_at_k([99, 20, 50], RELEVANT, 3) == 1.0

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k(RANKED, RELEVANT, -1)
