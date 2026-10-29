"""Core structured RARS reward and explicit input schema."""

from .full_reward import RewardConfig, compute_score
from .rars import (
    RARSParameters,
    build_rars_confusion_cost,
    completion_likelihood,
    normalize_class_counts,
)
from .schema import Detection, ParsedCompletion, SchemaError, parse_completion

__all__ = [
    "RewardConfig", "RARSParameters", "build_rars_confusion_cost",
    "completion_likelihood", "compute_score",
    "normalize_class_counts", "Detection",
    "ParsedCompletion", "SchemaError", "parse_completion",
]
