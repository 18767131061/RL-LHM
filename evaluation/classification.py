"""Fixed-label multiclass metrics with explicit zero-division behavior."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _binary_metrics(tp: int, fp: int, fn: int, support: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def classification_metrics(
    truth: Sequence[str],
    predictions: Sequence[str | None],
    classes: Sequence[str],
) -> dict:
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction lengths differ")
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Classes must be a non-empty unique sequence")
    unknown_truth = sorted(set(truth) - set(classes))
    if unknown_truth:
        raise ValueError(f"Truth contains labels outside the evaluation space: {unknown_truth}")
    per_class = {}
    for label in classes:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(truth, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(truth, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(truth, predictions))
        per_class[label] = _binary_metrics(tp, fp, fn, sum(actual == label for actual in truth))
    return {
        "n": len(truth),
        "accuracy": float(np.mean([actual == predicted for actual, predicted in zip(truth, predictions)]))
        if truth
        else 0.0,
        "macro_precision": float(np.mean([per_class[label]["precision"] for label in classes])),
        "macro_recall": float(np.mean([per_class[label]["recall"] for label in classes])),
        "macro_f1": float(np.mean([per_class[label]["f1"] for label in classes])),
        "per_class": per_class,
    }
