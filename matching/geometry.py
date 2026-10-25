"""IoU and deterministic one-to-one Hungarian assignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Assignment:
    gt_index: int | None
    prediction_index: int | None
    similarity: float


def _validated_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("Boxes must contain exactly four xyxy coordinates")
    values = tuple(float(value) for value in box)
    if not all(np.isfinite(values)):
        raise ValueError("Box coordinates must be finite")
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise ValueError("Boxes must use ordered xyxy coordinates")
    return x1, y1, x2, y2


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return continuous-coordinate 2D IoU for two xyxy boxes."""

    ax1, ay1, ax2, ay2 = _validated_box(first)
    bx1, by1, bx2, by2 = _validated_box(second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def box_iou_matrix(
    ground_truth: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
) -> np.ndarray:
    return np.asarray(
        [[box_iou(gt, prediction) for prediction in predictions] for gt in ground_truth],
        dtype=float,
    ).reshape((len(ground_truth), len(predictions)))


def hungarian_assignment(similarity: np.ndarray) -> list[Assignment]:
    """Maximize total similarity and return matched plus unmatched records."""

    matrix = np.asarray(similarity, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Similarity must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Similarity values must be finite")
    gt_count, prediction_count = matrix.shape
    if gt_count and prediction_count:
        rows, columns = linear_sum_assignment(-matrix)
    else:
        rows = np.asarray([], dtype=int)
        columns = np.asarray([], dtype=int)
    assigned_gt = {int(row) for row in rows}
    assigned_prediction = {int(column) for column in columns}
    assignments = [
        Assignment(int(row), int(column), float(matrix[row, column]))
        for row, column in zip(rows, columns)
    ]
    assignments.extend(
        Assignment(index, None, 0.0)
        for index in range(gt_count)
        if index not in assigned_gt
    )
    assignments.extend(
        Assignment(None, index, 0.0)
        for index in range(prediction_count)
        if index not in assigned_prediction
    )
    return sorted(
        assignments,
        key=lambda item: (
            item.gt_index is None,
            -1 if item.gt_index is None else item.gt_index,
            -1 if item.prediction_index is None else item.prediction_index,
        ),
    )
