"""Evaluation APIs for two-dimensional detection, classification, LOCO, and statistics."""

__all__ = ["classification_metrics", "evaluate_2d_boxes"]


def __getattr__(name):
    # Schema-only conversion needs no NumPy/SciPy/model runtime.
    if name == "classification_metrics":
        from .classification import classification_metrics
        return classification_metrics
    if name in {"evaluate_2d_boxes"}:
        from . import detection
        return getattr(detection, name)
    raise AttributeError(name)
