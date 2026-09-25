"""Explicit implementations of the three frozen matching protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import Assignment, hungarian_assignment


CLASS_AGNOSTIC = "class_agnostic"
# Paper terminology: "same-class-first" matching (the code key keeps the
# earlier internal name for backward compatibility with exported CSVs).
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

    Geometry-first class-aware deliberately reuses the class-agnostic geometry
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
