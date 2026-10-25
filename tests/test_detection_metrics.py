import json

import numpy as np
import pytest

from evaluation.detection import evaluate_2d_boxes, evaluate_similarity_matrix
from evaluation.schema import SchemaError, parse_completion
from matching.protocols import CLASS_AGNOSTIC, GEOMETRY_FIRST_CLASS_AWARE


def test_detection_counts_and_missed_gt_iou_zero() -> None:
    result = evaluate_2d_boxes(
        [[0, 0, 2, 2], [5, 5, 7, 7]],
        [[0, 0, 2, 2]],
        ["A", "A"],
        ["A"],
        CLASS_AGNOSTIC,
        0.5,
    )
    assert (result.tp, result.fp, result.fn) == (1, 0, 1)
    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.f1 == pytest.approx(2.0 / 3.0)
    assert result.mean_iou_with_misses_zero == 0.5


def test_wrong_class_geometry_match_is_fp_and_fn() -> None:
    result = evaluate_2d_boxes(
        [[0, 0, 1, 1]],
        [[0, 0, 1, 1]],
        ["A"],
        ["B"],
        GEOMETRY_FIRST_CLASS_AWARE,
        0.5,
    )
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert result.mean_iou_with_misses_zero == 0.0


def test_threshold_zero_is_strict_and_positive_threshold_is_inclusive() -> None:
    zero_overlap = evaluate_similarity_matrix(
        np.asarray([[0.0]]), ["A"], ["A"], CLASS_AGNOSTIC, 0.0
    )
    boundary = evaluate_similarity_matrix(
        np.asarray([[0.5]]), ["A"], ["A"], CLASS_AGNOSTIC, 0.5
    )
    assert zero_overlap.tp == 0
    assert boundary.tp == 1


def test_schema_parser_accepts_background_and_rejects_mixed_phase() -> None:
    valid = json.dumps(
        {
            "detections": [
                {"phase": "a", "label": "HCC", "bbox_2d": [0, 0, 10, 10]},
                {"phase": "p", "label": "no-lesion", "bbox_2d": [-1, -1, -1, -1]},
            ]
        }
    )
    parsed = parse_completion(valid, ["HCC"])
    assert len(parsed.detections) == 1
    assert parsed.background_phases == ("p",)

    mixed = json.dumps(
        {
            "detections": [
                {"phase": "a", "label": "HCC", "bbox_2d": [0, 0, 10, 10]},
                {"phase": "a", "label": "background", "bbox_2d": [-1, -1, -1, -1]},
            ]
        }
    )
    with pytest.raises(SchemaError, match="both"):
        parse_completion(mixed, ["HCC"])
