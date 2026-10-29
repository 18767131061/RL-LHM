"""Patient classification metrics over a caller-supplied fixed class space."""
from __future__ import annotations

from collections.abc import Sequence


def class_spaces(output_classes, metric_classes=None):
    output = tuple(output_classes)
    metric = output if metric_classes is None else tuple(metric_classes)
    for name, labels in (("output_classes", output), ("metric_classes", metric)):
        if (not labels or len(labels) != len(set(labels))
                or any(not isinstance(label, str) or not label for label in labels)):
            raise ValueError(f"{name} must contain unique nonempty strings")
    if not set(metric) <= set(output):
        raise ValueError("metric_classes must be a fixed subset of output_classes")
    return output, metric


def classification_metrics(
    truth: Sequence[str],
    predictions: Sequence[str | None],
    output_classes: Sequence[str],
    *,
    metric_classes: Sequence[str] | None = None,
) -> dict:
    """Compute accuracy and macro metrics, retaining undefined predictions.

    A None prediction contributes a false negative to its true class and no
    false positive. Every supplied patient remains in the accuracy denominator.
    Classes without support retain zero precision, recall, and F1 in the macro.
    """
    labels, macro_labels = class_spaces(output_classes, metric_classes)
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
        "output_classes": list(labels), "metric_classes": list(macro_labels),
        "accuracy": sum(actual == predicted for actual, predicted in zip(truth, predictions)) / len(truth) if len(truth) else 0.0,
        "macro_precision": sum(per_class[label]["precision"] for label in macro_labels) / len(macro_labels),
        "macro_recall": sum(per_class[label]["recall"] for label in macro_labels) / len(macro_labels),
        "macro_f1": sum(per_class[label]["f1"] for label in macro_labels) / len(macro_labels),
        "per_class": per_class,
    }
