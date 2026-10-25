"""Geometry and lesion-matching protocols used by RL-LHM evaluation."""

from .geometry import Assignment, box_iou, box_iou_matrix, hungarian_assignment
from .protocols import (
    CLASS_AGNOSTIC,
    GEOMETRY_FIRST_CLASS_AWARE,
    SAME_CLASS_PRIORITY,
    ProtocolMatch,
    match_protocol,
)

__all__ = [
    "Assignment",
    "CLASS_AGNOSTIC",
    "GEOMETRY_FIRST_CLASS_AWARE",
    "ProtocolMatch",
    "SAME_CLASS_PRIORITY",
    "box_iou",
    "box_iou_matrix",
    "hungarian_assignment",
    "match_protocol",
]
