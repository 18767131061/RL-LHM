"""Geometry primitives for the core RARS assignments."""

from .geometry import (
    Assignment,
    box_iou,
    box_iou_matrix,
    generalized_box_iou,
    hungarian_assignment,
)

__all__ = [
    "Assignment", "box_iou", "box_iou_matrix", "generalized_box_iou",
    "hungarian_assignment",
]
