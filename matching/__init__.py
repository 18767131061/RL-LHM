"""Box geometry and training/evaluation matching primitives."""

from .geometry import (
    Assignment,
    box_iou,
    box_iou_matrix,
    generalized_box_iou,
    hungarian_assignment,
)

from .protocols import (
    CLASS_AGNOSTIC, GEOMETRY_FIRST_CLASS_AWARE, SAME_CLASS_PRIORITY,
    PROTOCOLS, ProtocolMatch, match_protocol,
)

__all__ = [
    "Assignment", "box_iou", "box_iou_matrix", "generalized_box_iou",
    "hungarian_assignment", "CLASS_AGNOSTIC", "GEOMETRY_FIRST_CLASS_AWARE",
    "SAME_CLASS_PRIORITY", "PROTOCOLS", "ProtocolMatch", "match_protocol",
]
