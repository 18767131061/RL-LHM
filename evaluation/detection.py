"""Lesion-level detection metrics under three one-to-one matching protocols."""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from matching.geometry import box_iou_matrix
from matching.protocols import (CLASS_AGNOSTIC, THRESHOLD_CLASS_AGNOSTIC, THRESHOLD_PROTOCOLS,
                                ProtocolMatch, match_protocol, match_threshold_aware)


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
    assigned_iou_sum: float
    positive_overlap_matches: int
    matches: tuple[ProtocolMatch, ...]


def passes_iou_threshold(iou: float, threshold: float) -> bool:
    """Zero denotes the strictly positive-overlap endpoint; others use >=."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be finite and in [0, 1]")
    return iou > 0 if threshold == 0 else iou >= threshold


def binary_detection_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (tp, fp, fn)):
        raise ValueError("TP, FP, and FN must be non-negative integers")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_similarity_matrix(
    similarity: np.ndarray,
    gt_labels: Sequence[str],
    prediction_labels: Sequence[str],
    protocol: str,
    threshold: float,
) -> DetectionEvaluation:
    """Evaluate the explicitly named assignment protocol at one threshold.

    assigned_iou_sum contains every geometric pair under this protocol before
    thresholding. The cohort's Mean IoU uses only the class-agnostic value.
    """
    passes_iou_threshold(0.0, threshold)
    matrix = np.asarray(similarity, dtype=float)
    matches = (match_threshold_aware(matrix, gt_labels, prediction_labels, protocol, threshold)
               if protocol in THRESHOLD_PROTOCOLS else match_protocol(matrix, gt_labels, prediction_labels, protocol))
    true_positive, positive_overlap = 0, 0
    assigned_iou_sum = 0.0
    for item in matches:
        if item.gt_index is None or item.prediction_index is None:
            continue
        assigned_iou_sum += item.similarity
        label_eligible = protocol in (CLASS_AGNOSTIC, THRESHOLD_CLASS_AGNOSTIC) or item.label_match
        if label_eligible:
            positive_overlap += int(item.similarity > 0)
            true_positive += int(passes_iou_threshold(item.similarity, threshold))
    gt_count, prediction_count = len(gt_labels), len(prediction_labels)
    false_positive, false_negative = prediction_count - true_positive, gt_count - true_positive
    return DetectionEvaluation(
        protocol=protocol, threshold=threshold,
        tp=true_positive, fp=false_positive, fn=false_negative,
        gt_instances=gt_count, predicted_instances=prediction_count,
        assigned_iou_sum=assigned_iou_sum, positive_overlap_matches=positive_overlap,
        matches=tuple(matches),
        **binary_detection_metrics(true_positive, false_positive, false_negative),
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
        raise ValueError("each 2D instance requires one label")
    return evaluate_similarity_matrix(
        box_iou_matrix(gt_boxes, prediction_boxes), gt_labels, prediction_labels,
        protocol, threshold,
    )
