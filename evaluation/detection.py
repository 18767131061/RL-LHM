"""Two-dimensional lesion detection metrics under explicit matching protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from matching.geometry import box_iou_matrix
from matching.protocols import CLASS_AGNOSTIC, ProtocolMatch, match_protocol


@dataclass(frozen=True)
class DetectionEvaluation:
    protocol: str
    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    gt_instances: int
    predicted_instances: int
    mean_iou_with_misses_zero: float
    matches: tuple[ProtocolMatch, ...]


def passes_iou_threshold(iou: float, threshold: float) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be in [0, 1]")
    return iou > 0.0 if threshold == 0.0 else iou >= threshold


def binary_detection_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    if min(tp, fp, fn) < 0:
        raise ValueError("TP, FP, and FN must be non-negative")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_similarity_matrix(
    similarity: np.ndarray,
    gt_labels: Sequence[str],
    prediction_labels: Sequence[str],
    protocol: str,
    threshold: float,
) -> DetectionEvaluation:
    matrix = np.asarray(similarity, dtype=float)
    matches = match_protocol(matrix, gt_labels, prediction_labels, protocol)
    true_positive = 0
    iou_sum = 0.0
    for item in matches:
        if item.gt_index is None or item.prediction_index is None:
            continue
        label_eligible = protocol == CLASS_AGNOSTIC or item.label_match
        if label_eligible:
            iou_sum += item.similarity
        if label_eligible and passes_iou_threshold(item.similarity, threshold):
            true_positive += 1
    gt_count = len(gt_labels)
    prediction_count = len(prediction_labels)
    false_positive = prediction_count - true_positive
    false_negative = gt_count - true_positive
    metrics = binary_detection_metrics(true_positive, false_positive, false_negative)
    return DetectionEvaluation(
        protocol=protocol,
        threshold=threshold,
        tp=true_positive,
        fp=false_positive,
        fn=false_negative,
        gt_instances=gt_count,
        predicted_instances=prediction_count,
        mean_iou_with_misses_zero=iou_sum / gt_count if gt_count else 0.0,
        matches=tuple(matches),
        **metrics,
    )


def evaluate_2d_boxes(
    gt_boxes: Sequence[Sequence[float]],
    prediction_boxes: Sequence[Sequence[float]],
    gt_labels: Sequence[str],
    prediction_labels: Sequence[str],
    protocol: str,
    threshold: float,
) -> DetectionEvaluation:
    if len(gt_boxes) != len(gt_labels) or len(prediction_boxes) != len(prediction_labels):
        raise ValueError("Each 2D instance requires exactly one label")
    return evaluate_similarity_matrix(
        box_iou_matrix(gt_boxes, prediction_boxes),
        gt_labels,
        prediction_labels,
        protocol,
        threshold,
    )
