"""Structured RARS reward: independent IoU/GIoU matching, then clipping."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from matching.geometry import box_iou, generalized_box_iou, hungarian_assignment
from .schema import Detection, PHASES, ParsedCompletion, SchemaError, parse_completion
from .rars import (
    RARSParameters,
    build_rars_confusion_cost,
    completion_likelihood,
    normalize_class_counts,
    validate_risk_matrix,
)


@dataclass(frozen=True)
class RewardConfig:
    classes: tuple[str, ...]
    severity_matrix: Mapping[str, Mapping[str, float]]
    rars_parameters: RARSParameters
    giou_for_classification: float
    detection_weight: float
    classification_weight: float
    missing_match_penalty: float
    clip_min: float
    clip_max: float
    rars_enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.rars_parameters, RARSParameters):
            raise TypeError("rars_parameters must be explicit RARSParameters")
        classes = tuple(self.classes)
        if (not classes or len(classes) != len(set(classes))
                or any(not isinstance(label, str) or not label for label in classes)):
            raise ValueError("classes must be non-empty, unique diagnostic labels")
        validate_risk_matrix(self.severity_matrix, classes)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "severity_matrix", MappingProxyType({
            label: MappingProxyType(dict(self.severity_matrix[label])) for label in classes
        }))
        if (not math.isfinite(self.giou_for_classification)
                or not -1 <= self.giou_for_classification <= 1):
            raise ValueError("giou_for_classification must be finite and in [-1,1]")
        for name in ("detection_weight", "classification_weight", "missing_match_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (not math.isfinite(self.clip_min) or not math.isfinite(self.clip_max)
                or not self.clip_min <= 0 <= self.clip_max or self.clip_min >= self.clip_max):
            raise ValueError("Reward bounds must be finite, have positive width, and include invalid-output zero")
        if not isinstance(self.rars_enabled, bool):
            raise ValueError("rars_enabled must be a boolean")


def _phase_assignment(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    *,
    use_giou: bool,
) -> list[tuple[Detection, Detection, float]]:
    similarity = generalized_box_iou if use_giou else box_iou
    matrix = np.asarray([
        [similarity(gt.box, prediction.box) for prediction in predictions]
        for gt in ground_truth
    ], dtype=float).reshape(len(ground_truth), len(predictions))
    return [
        (ground_truth[pair.gt_index], predictions[pair.prediction_index], pair.similarity)
        for pair in hungarian_assignment(matrix)
        if pair.gt_index is not None and pair.prediction_index is not None
    ]


def detection_reward(
    ground_truth: ParsedCompletion, prediction: ParsedCompletion
) -> tuple[float, dict[str, float], int]:
    """Phase-mean IoU; unmatched ground truths and absent phases get zero."""
    phase_scores = {}
    background_correct = 0
    for phase in PHASES:
        truth = [item for item in ground_truth.detections if item.phase == phase]
        predicted = [item for item in prediction.detections if item.phase == phase]
        if truth:
            pairs = _phase_assignment(truth, predicted, use_giou=False)
            score = sum(similarity for _, _, similarity in pairs) / len(truth)
        else:
            score = float(phase in ground_truth.background_phases
                          and phase in prediction.background_phases and not predicted)
            background_correct += int(score)
        phase_scores[phase] = float(score)
    return sum(phase_scores.values()) / len(PHASES), phase_scores, background_correct


def classification_reward(
    ground_truth: ParsedCompletion,
    prediction: ParsedCompletion,
    *,
    sequence_confidence: float,
    class_counts: Mapping[str, int],
    config: RewardConfig,
) -> tuple[float, int, float]:
    """Mean agreement/cost over GIoU-assigned pairs passing the threshold."""
    counts = normalize_class_counts(class_counts, config.classes)
    if not math.isfinite(sequence_confidence) or not 0 <= sequence_confidence <= 1:
        raise ValueError("sequence_confidence must be a finite likelihood in [0,1]")
    contributions = []
    maximum_severity = 1.0
    for phase in PHASES:
        pairs = _phase_assignment(
            [item for item in ground_truth.detections if item.phase == phase],
            [item for item in prediction.detections if item.phase == phase],
            use_giou=True,
        )
        for truth, predicted, giou in pairs:
            if giou < config.giou_for_classification:
                continue
            if predicted.label == truth.label:
                contributions.append(1.0)
            elif config.rars_enabled:
                penalty, severity = build_rars_confusion_cost(
                    predicted.label, truth.label, sequence_confidence, counts,
                    config.severity_matrix, config.rars_parameters,
                )
                contributions.append(-penalty)
                maximum_severity = max(maximum_severity, severity)
            else:
                contributions.append(-1.0)
    eligible = len(contributions)
    # Divide before accumulation: the mean can be finite even when its sum
    # would overflow for large caller-supplied severity weights.
    mean = math.fsum(value / eligible for value in contributions) if eligible else 0.0
    return mean, eligible, maximum_severity


def compute_score(
    solution: str,
    ground_truth: str,
    *,
    completion_nll: float,
    class_counts: Mapping[str, int],
    config: RewardConfig,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Score JSON with the same explicit schema for prediction and ground truth.

    Only malformed model output receives a zero reward. Invalid ground truth
    or caller metadata raises an error. NLL is the mean over generated tokens,
    excluding prompts and padding. ``missing_match_penalty`` is a magnitude.
    """
    confidence = completion_likelihood(completion_nll)
    counts = normalize_class_counts(class_counts, config.classes)
    truth = parse_completion(ground_truth, config.classes, aliases)
    result = {
        "score": 0.0, "reward_raw": 0.0, "parse_ok": False,
        "detection_reward": 0.0, "classification_reward": 0.0,
        "classification_matches": 0, "sequence_confidence": confidence,
        "phase_scores": {phase: 0.0 for phase in PHASES},
        "missing_match_reward": 0.0, "severity_factor_max": 1.0,
        "background_correct_phases": 0,
    }
    try:
        predicted = parse_completion(solution, config.classes, aliases)
    except SchemaError:
        return result
    detection, phase_scores, background_correct = detection_reward(truth, predicted)
    classification, matches, severity = classification_reward(
        truth, predicted, sequence_confidence=confidence, class_counts=counts, config=config
    )
    missing = -config.missing_match_penalty if not matches and not background_correct else 0.0
    raw = config.detection_weight * detection + config.classification_weight * classification + missing
    result.update({
        "score": max(config.clip_min, min(config.clip_max, raw)),
        "reward_raw": raw, "parse_ok": True,
        "detection_reward": detection, "classification_reward": classification,
        "classification_matches": matches, "phase_scores": phase_scores,
        "missing_match_reward": missing, "severity_factor_max": severity,
        "background_correct_phases": background_correct,
    })
    return result
