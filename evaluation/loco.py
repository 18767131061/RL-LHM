"""Four-class target-space LOCO evaluation."""

from __future__ import annotations

from typing import Sequence

from .classification import classification_metrics


FOUR_CLASS_LOCO_CLASSES = ("HCC", "ICC", "HA_Hemangioma", "nodule_CYST")
TARGET_ABSENT_CLASS = "FNH"


def evaluate_four_class_loco(
    truth: Sequence[str],
    predictions: Sequence[str | None],
    classes: Sequence[str] = FOUR_CLASS_LOCO_CLASSES,
    target_absent_class: str = TARGET_ABSENT_CLASS,
) -> dict:
    if target_absent_class in truth:
        raise ValueError(f"Target cohort unexpectedly contains {target_absent_class}")
    point = classification_metrics(truth, predictions, classes)
    return {
        **point,
        "target_supported_classes": list(classes),
        "target_absent_class": target_absent_class,
        "predictions_of_absent_class": sum(value == target_absent_class for value in predictions),
        "macro_average_denominator": len(classes),
    }
