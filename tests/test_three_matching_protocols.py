import numpy as np
import pytest

from evaluation.detection import evaluate_similarity_matrix
from matching.protocols import (
    CLASS_AGNOSTIC,
    GEOMETRY_FIRST_CLASS_AWARE,
    SAME_CLASS_PRIORITY,
    match_protocol,
)


def _pairs(matches):
    return {
        (item.gt_index, item.prediction_index)
        for item in matches
        if item.gt_index is not None and item.prediction_index is not None
    }


def test_three_protocols_are_semantically_distinct() -> None:
    similarity = np.asarray([[0.9, 0.8], [0.8, 0.9]])
    gt_labels = ["A", "B"]
    prediction_labels = ["B", "A"]

    agnostic = evaluate_similarity_matrix(
        similarity, gt_labels, prediction_labels, CLASS_AGNOSTIC, 0.5
    )
    same_class = evaluate_similarity_matrix(
        similarity, gt_labels, prediction_labels, SAME_CLASS_PRIORITY, 0.5
    )
    aware = evaluate_similarity_matrix(
        similarity, gt_labels, prediction_labels, GEOMETRY_FIRST_CLASS_AWARE, 0.5
    )

    assert _pairs(agnostic.matches) == {(0, 0), (1, 1)}
    assert _pairs(aware.matches) == _pairs(agnostic.matches)
    assert _pairs(same_class.matches) == {(0, 1), (1, 0)}
    assert (agnostic.tp, same_class.tp, aware.tp) == (2, 2, 0)


def test_same_class_priority_is_lexicographic_before_iou() -> None:
    matches = match_protocol(
        np.asarray([[0.99, 0.20], [0.20, 0.99]]),
        ["A", "B"],
        ["B", "A"],
        SAME_CLASS_PRIORITY,
    )
    assert _pairs(matches) == {(0, 1), (1, 0)}
    assert all(item.label_match for item in matches)


def test_unknown_protocol_and_shape_mismatch_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        match_protocol(np.ones((1, 1)), ["A"], ["A"], "other")
    with pytest.raises(ValueError, match="shape"):
        match_protocol(np.ones((2, 1)), ["A"], ["A"], CLASS_AGNOSTIC)
