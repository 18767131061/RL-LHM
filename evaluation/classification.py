"""Patient classification metrics over a caller-supplied fixed class space."""
from __future__ import annotations

from collections.abc import Sequence


def classification_metrics(
    truth: Sequence[str],
    predictions: Sequence[str | None],
    classes: Sequence[str],
) -> dict:
    """Compute accuracy and macro metrics, retaining undefined predictions.

    A None prediction contributes a false negative to its true class and no
    false positive. Every supplied patient remains in the accuracy denominator.
    Classes without support retain zero precision, recall, and F1 in the macro.
    """
    labels = tuple(classes)
    if (not labels or len(labels) != len(set(labels))
            or any(not isinstance(label, str) or not label for label in labels)):
        raise ValueError("classes must be a non-empty sequence of unique strings")
    if len(truth) != len(predictions):
        raise ValueError("truth and prediction lengths must match")
    if any(label not in labels for label in truth):
        raise ValueError("patient truth must belong to the active class space")
    if any(label is not None and label not in labels for label in predictions):
        raise ValueError("patient predictions must be active labels or None")
    per_class = {}
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(truth, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(truth, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(truth, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[label] = {
            "tp": tp, "fp": fp, "fn": fn, "support": sum(actual == label for actual in truth),
            "precision": precision, "recall": recall, "f1": f1,
        }
    return {
        "n": len(truth),
        "accuracy": sum(actual == predicted for actual, predicted in zip(truth, predictions)) / len(truth) if len(truth) else 0.0,
        "macro_precision": sum(per_class[label]["precision"] for label in labels) / len(labels),
        "macro_recall": sum(per_class[label]["recall"] for label in labels) / len(labels),
        "macro_f1": sum(per_class[label]["f1"] for label in labels) / len(labels),
        "per_class": per_class,
    }
