"""Bounded structured reward with explicit missed-lesion and false-positive costs."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from matching.geometry import box_iou, generalized_box_iou, hungarian_assignment
from .schema import PHASES, SchemaError, parse_completion
from .rars import (RARSParameters, build_rars_confusion_cost, completion_likelihood,
                   normalize_class_counts, validate_risk_matrix)


@dataclass(frozen=True)
class RewardConfig:
    classes: tuple[str, ...]
    severity_matrix: Mapping[str, Mapping[str, float]]
    rars_parameters: RARSParameters
    giou_for_classification: float
    detection_weight: float
    classification_weight: float
    clip_min: float
    clip_max: float
    rars_enabled: bool

    def __post_init__(self):
        classes = tuple(self.classes)
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("classes must be nonempty and unique")
        validate_risk_matrix(self.severity_matrix, classes)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "severity_matrix", MappingProxyType({
            label: MappingProxyType(dict(self.severity_matrix[label])) for label in classes
        }))
        if not isinstance(self.rars_parameters, RARSParameters):
            raise TypeError("rars_parameters must be RARSParameters")
        if not math.isfinite(self.giou_for_classification) or not -1 <= self.giou_for_classification <= 1:
            raise ValueError("GIoU threshold must lie in [-1,1]")
        weights = (self.detection_weight, self.classification_weight)
        if not all(math.isfinite(w) and w > 0 for w in weights):
            raise ValueError("reward weights must be finite and positive")
        width = sum(weights)
        if (not math.isfinite(width) or not math.isfinite(self.clip_min)
                or not math.isfinite(self.clip_max)
                or self.clip_min > -width or self.clip_max < width):
            raise ValueError("clipping bounds must enclose the full structured reward range")
        if type(self.rars_enabled) is not bool:
            raise ValueError("rars_enabled must be boolean")


def compute_score(solution: str, ground_truth: str, *, completion_nll: float,
                  class_counts: Mapping[str, int], config: RewardConfig,
                  aliases: Mapping[str, str] | None = None) -> dict[str, object]:
    """Average phase utilities, each in [-1,1], then apply component weights.

    One GIoU assignment defines eligible pairs for both components. Each missed
    GT and each unmatched prediction costs one unit. The denominator is the GT
    count plus unmatched predictions (one background slot for empty GT phases).
    Wrong diagnoses cost RARS / maximum possible RARS over the fixed class space
    at likelihood one. Thus omission cannot improve an eligible wrong diagnosis,
    and adding unmatched records lowers utility unless already at its minimum.
    Invalid JSON receives the minimum structured utility. GT is fully annotated
    across a/p/v; a phase without lesions is a background phase.
    """
    confidence = completion_likelihood(completion_nll)
    counts = normalize_class_counts(class_counts, config.classes)
    truth = parse_completion(ground_truth, config.classes, aliases)
    lower = -(config.detection_weight + config.classification_weight)
    result = dict(score=lower, reward_raw=lower, parse_ok=False,
                  detection_reward=-1.0, classification_reward=-1.0,
                  classification_matches=0, sequence_confidence=confidence,
                  phase_scores={phase: -1.0 for phase in PHASES},
                  false_positives=0, missed_lesions=len(truth.detections),
                  background_correct_phases=0)
    # A fixed maximum preserves relative class scarcity/severity, and prevents
    # abstention from bypassing a larger matched-error penalty.
    risk_scale = max([1.0] + [
        build_rars_confusion_cost(pred, gt, 1.0, counts, config.severity_matrix,
                                 config.rars_parameters)[0]
        for gt in config.classes for pred in config.classes if gt != pred
    ]) if config.rars_enabled else 1.0
    try:
        predicted = parse_completion(solution, config.classes, aliases)
    except SchemaError:
        return result
    phase_scores, class_scores = {}, []
    fp_total = fn_total = matches = background_correct = 0
    for phase in PHASES:
        targets = [item for item in truth.detections if item.phase == phase]
        outputs = [item for item in predicted.detections if item.phase == phase]
        matrix = np.asarray([[generalized_box_iou(gt.box, pr.box) for pr in outputs]
                             for gt in targets], dtype=float).reshape(len(targets), len(outputs))
        pairs = [pair for pair in hungarian_assignment(matrix)
                 if pair.gt_index is not None and pair.prediction_index is not None
                 and pair.similarity >= config.giou_for_classification]
        fp, fn = len(outputs) - len(pairs), len(targets) - len(pairs)
        denominator = max(1, len(targets)) + fp
        loc = diag = -float(fn + fp)
        if not targets:
            correct = phase in predicted.background_phases and not outputs
            loc += 1.0 if correct else -1.0
            diag += 1.0 if correct else -1.0
            background_correct += int(correct)
        for pair in pairs:
            gt, pr = targets[pair.gt_index], outputs[pair.prediction_index]
            loc += box_iou(gt.box, pr.box)
            if gt.label == pr.label:
                diag += 1.0
            elif config.rars_enabled:
                cost, _ = build_rars_confusion_cost(pr.label, gt.label, confidence,
                                                   counts, config.severity_matrix,
                                                   config.rars_parameters)
                diag -= cost / risk_scale
            else:
                diag -= 1.0
        phase_scores[phase] = loc / denominator
        class_scores.append(diag / denominator)
        fp_total += fp
        fn_total += fn
        matches += len(pairs)
    detection = math.fsum(phase_scores.values()) / len(PHASES)
    classification = math.fsum(class_scores) / len(PHASES)
    raw = config.detection_weight * detection + config.classification_weight * classification
    result.update(score=max(config.clip_min, min(config.clip_max, raw)), reward_raw=raw,
                  parse_ok=True, detection_reward=detection, classification_reward=classification,
                  classification_matches=matches, phase_scores=phase_scores,
                  false_positives=fp_total, missed_lesions=fn_total,
                  background_correct_phases=background_correct)
    return result
