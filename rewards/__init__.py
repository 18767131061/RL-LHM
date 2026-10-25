"""Public RL-LHM reward and RARS utilities."""

from .full_reward import RewardConfig, compute_score, detection_reward

from .rars import (
    RARSParameters,
    build_rars_confusion_cost,
    freeze_risk_selection,
    generate_risk_grid,
    validate_risk_freeze,
)

__all__ = [
    "RewardConfig",
    "RARSParameters",
    "build_rars_confusion_cost",
    "compute_score",
    "detection_reward",
    "freeze_risk_selection",
    "generate_risk_grid",
    "validate_risk_freeze",
]
