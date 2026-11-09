"""Fixed-patient classification and lesion-level detection evaluation."""
from .classification import classification_metrics
from .detection import DetectionEvaluation, evaluate_2d_boxes, evaluate_similarity_matrix

__all__ = ["classification_metrics", "DetectionEvaluation", "evaluate_2d_boxes", "evaluate_similarity_matrix"]
