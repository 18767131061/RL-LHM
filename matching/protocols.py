"""Class-agnostic, geometry-first class-aware, and same-class-first matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import Assignment, hungarian_assignment


CLASS_AGNOSTIC = "class_agnostic"
# Paper terminology: "same-class-first" matching; exported key: same_class_priority.
SAME_CLASS_PRIORITY = "same_class_priority"
GEOMETRY_FIRST_CLASS_AWARE = "geometry_first_class_aware"
PROTOCOLS = (CLASS_AGNOSTIC, SAME_CLASS_PRIORITY, GEOMETRY_FIRST_CLASS_AWARE)


@dataclass(frozen=True)
class ProtocolMatch:
    gt_index: int | None
    prediction_index: int | None
    similarity: float
    label_match: bool
    protocol: str


def _validate_inputs(
    similarity: np.ndarray,
    gt_labels: Sequence[str],
    prediction_labels: Sequence[str],
) -> np.ndarray:
    if any(not isinstance(label, str) or not label for label in (*gt_labels, *prediction_labels)):
        raise ValueError("Each instance needs a non-empty class label")
    matrix = np.asarray(similarity, dtype=float)
    if matrix.shape != (len(gt_labels), len(prediction_labels)):
        raise ValueError("Similarity shape does not match the label sequences")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("Similarity values must be finite and in [0, 1]")
    return matrix


def _same_class_priority_score(
    similarity: np.ndarray,
    gt_labels: Sequence[str],
    prediction_labels: Sequence[str],
) -> np.ndarray:
    # One additional same-class assignment dominates every possible IoU-sum
    # difference; IoU then breaks ties within the maximum same-class count.
    maximum_pairs = min(len(gt_labels), len(prediction_labels))
    priority_bonus = float(maximum_pairs + 1)
    same_class = np.asarray(
        [
            [float(gt_label == prediction_label) for prediction_label in prediction_labels]
            for gt_label in gt_labels
        ],
        dtype=float,
    ).reshape(similarity.shape)
    return similarity + priority_bonus * same_class


def _to_protocol_match(
    assignment: Assignment,
    similarity: np.ndarray,
    gt_labels: Sequence[str],
    prediction_labels: Sequence[str],
    protocol: str,
) -> ProtocolMatch:
    if assignment.gt_index is None or assignment.prediction_index is None:
        return ProtocolMatch(
            assignment.gt_index,
            assignment.prediction_index,
            0.0,
            False,
            protocol,
        )
    gt_index = assignment.gt_index
    prediction_index = assignment.prediction_index
    return ProtocolMatch(
        gt_index,
        prediction_index,
        float(similarity[gt_index, prediction_index]),
        gt_labels[gt_index] == prediction_labels[prediction_index],
        protocol,
    )


def match_protocol(
    similarity: np.ndarray,
    gt_labels: Sequence[str],
    prediction_labels: Sequence[str],
    protocol: str,
) -> list[ProtocolMatch]:
    """Match one evaluation unit under one of the three named protocols.

    Geometry-first class-aware uses the class-agnostic geometry
    assignment and applies the label condition only when counting a TP.
    Same-class-priority uses a lexicographic objective: maximize the number of
    same-class pairs first, then maximize total geometry similarity.
    """

    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown matching protocol: {protocol}")
    matrix = _validate_inputs(similarity, gt_labels, prediction_labels)
    objective = (
        _same_class_priority_score(matrix, gt_labels, prediction_labels)
        if protocol == SAME_CLASS_PRIORITY
        else matrix
    )
    return [
        _to_protocol_match(
            assignment,
            matrix,
            gt_labels,
            prediction_labels,
            protocol,
        )
        for assignment in hungarian_assignment(objective)
    ]

# Threshold-aware protocols optimize eligible matches at the selected IoU gate.
THRESHOLD_CLASS_AGNOSTIC = "threshold_aware_class_agnostic"
THRESHOLD_CLASS_AWARE = "threshold_aware_class_aware"
THRESHOLD_PROTOCOLS = (THRESHOLD_CLASS_AGNOSTIC, THRESHOLD_CLASS_AWARE)
ALL_PROTOCOLS = PROTOCOLS + THRESHOLD_PROTOCOLS


def match_threshold_aware(similarity, gt_labels, prediction_labels, protocol, threshold):
    """Maximize eligible TP count, then IoU sum of eligible pairs only.

    Rejected pairings carry zero objective and are returned as unmatched entries.
    Class-aware eligibility requires both the IoU threshold and identical labels.
    """
    if protocol not in THRESHOLD_PROTOCOLS:
        raise ValueError("unknown threshold-aware protocol")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must lie in [0,1]")
    matrix = _validate_inputs(similarity, gt_labels, prediction_labels)
    eligible = matrix > 0 if threshold == 0 else matrix >= threshold
    if protocol == THRESHOLD_CLASS_AWARE:
        eligible &= np.asarray([[gt == pred for pred in prediction_labels] for gt in gt_labels],
                               dtype=bool).reshape(matrix.shape)
    bonus = min(matrix.shape) + 1
    objective = np.where(eligible, bonus + matrix, 0.0)
    matches = []
    for assignment in hungarian_assignment(objective):
        i, j = assignment.gt_index, assignment.prediction_index
        if i is not None and j is not None and not eligible[i, j]:
            matches.extend((ProtocolMatch(i, None, 0.0, False, protocol),
                            ProtocolMatch(None, j, 0.0, False, protocol)))
        else:
            matches.append(_to_protocol_match(assignment, matrix, gt_labels, prediction_labels, protocol))
    return matches
