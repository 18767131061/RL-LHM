import numpy as np
import pytest

from matching.geometry import hungarian_assignment


def _pairs(assignments):
    return {
        (item.gt_index, item.prediction_index)
        for item in assignments
        if item.gt_index is not None and item.prediction_index is not None
    }


def test_hungarian_finds_global_not_greedy_optimum() -> None:
    assignments = hungarian_assignment(np.asarray([[0.9, 0.8], [0.7, 0.1]]))
    assert _pairs(assignments) == {(0, 1), (1, 0)}
    assert sum(item.similarity for item in assignments) == pytest.approx(1.5)


def test_hungarian_emits_unmatched_records_for_rectangular_matrix() -> None:
    assignments = hungarian_assignment(np.asarray([[0.9, 0.2, 0.1]]))
    assert _pairs(assignments) == {(0, 0)}
    assert {(item.gt_index, item.prediction_index) for item in assignments} == {
        (0, 0),
        (None, 1),
        (None, 2),
    }


def test_hungarian_handles_empty_and_rejects_nonfinite() -> None:
    assignments = hungarian_assignment(np.empty((2, 0)))
    assert [(item.gt_index, item.prediction_index) for item in assignments] == [(0, None), (1, None)]
    with pytest.raises(ValueError, match="finite"):
        hungarian_assignment(np.asarray([[np.nan]]))
