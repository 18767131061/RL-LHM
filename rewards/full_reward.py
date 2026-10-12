"""Public RL-LHM detection/classification reward orchestration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.schema import Detection, ParsedCompletion, SchemaError, parse_completion
from matching.geometry import box_iou
from rewards.rars import (
    PRIVATE_CLASSES,
    PUBLIC_CLASSES,
    RARSParameters,
    build_rars_confusion_cost,
    default_matrix,
    matrix_from_pair_weights,
    profile_class_space,
)

PHASES = ("a", "p", "v")


@dataclass(frozen=True)
class RewardConfig:
    data_mode: str = "public_5class"
    reward_mode: str = "rars"
    matrix_mode: str = "ordinal"
    matrix_scale: float = 1.0
    giou_for_classification: float = 0.30
    classification_weight: float = 1.0
    clip_min: float = -6.0
    clip_max: float = 3.0
    rars_parameters: RARSParameters = RARSParameters()
    risk_profile: Mapping[str, object] | None = None


def class_space(data_mode: str) -> tuple[str, ...]:
    if data_mode == "private_5class":
        return PRIVATE_CLASSES
    if data_mode == "public_3class":
        return ("HCC", "ICC", "HA_Hemangioma")
    if data_mode == "public_5class":
        return PUBLIC_CLASSES
    raise ValueError(f"Unsupported data mode: {data_mode}")


def _profile_settings(config: RewardConfig):
    classes = class_space(config.data_mode)
    cohort = "private" if config.data_mode == "private_5class" else "public"
    if config.risk_profile is None:
        return classes, default_matrix(cohort)
    try:
        classes = profile_class_space(config.risk_profile, cohort, classes)
        weights = {pair.replace("--", "->"): value
                   for pair, value in config.risk_profile["pair_weights"].items()}
        return classes, matrix_from_pair_weights(classes, weights, config.risk_profile["other_weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid reward risk profile: {error}") from error


def _empty_reward(sequence_confidence: float, config: RewardConfig) -> dict[str, object]:
    return {
        "score": 0.0,
        "reward_raw": 0.0,
        "parse_ok": 0,
        "detection_reward": 0.0,
        "classification_reward": 0.0,
        "classification_matches": 0,
        "sequence_confidence": float(sequence_confidence),
        "reward_mode": config.reward_mode,
        "risk_matrix_mode": config.matrix_mode,
        "risk_matrix_scale": config.matrix_scale,
        "severity_factor_max": 1.0,
        "background_correct_phases": 0,
        "background_total_phases": 0,
    }


def generalized_box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Continuous-coordinate generalized IoU for two ordered xyxy boxes."""
    iou = box_iou(first, second)
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    intersection = iou * (first_area + second_area) / (1.0 + iou) if iou < 1.0 else first_area
    union = first_area + second_area - intersection
    cover = (max(ax2, bx2) - min(ax1, bx1)) * (max(ay2, by2) - min(ay1, by1))
    return float(iou - (cover - union) / cover) if cover > 0.0 else float(iou)


def _phase_assignment(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    *,
    use_giou: bool,
) -> list[tuple[Detection, Detection, float]]:
    if not ground_truth or not predictions:
        return []
    similarity = generalized_box_iou if use_giou else box_iou
    matrix = np.asarray(
        [[similarity(gt.box, prediction.box) for prediction in predictions] for gt in ground_truth],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(-matrix)
    return [
        (ground_truth[int(row)], predictions[int(column)], float(matrix[row, column]))
        for row, column in zip(rows, columns)
    ]


def _all_phase_assignments(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    *,
    use_giou: bool,
) -> list[tuple[Detection, Detection, float]]:
    matches: list[tuple[Detection, Detection, float]] = []
    for phase in PHASES:
        matches.extend(
            _phase_assignment(
                [item for item in ground_truth if item.phase == phase],
                [item for item in predictions if item.phase == phase],
                use_giou=use_giou,
            )
        )
    return matches


def detection_reward(
    ground_truth: ParsedCompletion,
    prediction: ParsedCompletion,
) -> tuple[float, dict[str, float], int]:
    """Return the reward-side phase mean IoU with missed GT lesions set to zero."""
    predicted_background = set(prediction.background_phases)
    ground_truth_background = set(ground_truth.background_phases)
    phase_scores: dict[str, float] = {}
    background_correct = 0
    for phase in PHASES:
        phase_gt = [item for item in ground_truth.detections if item.phase == phase]
        phase_prediction = [item for item in prediction.detections if item.phase == phase]
        if phase_gt:
            matches = _phase_assignment(phase_gt, phase_prediction, use_giou=False)
            phase_score = sum(score for _, _, score in matches) / len(phase_gt)
        elif phase in ground_truth_background:
            phase_score = float(phase in predicted_background and not phase_prediction)
            background_correct += int(phase_score)
        else:
            phase_score = 0.0
        phase_scores[phase] = float(phase_score)
    return float(sum(phase_scores.values()) / len(PHASES)), phase_scores, background_correct


def classification_reward(
    ground_truth: ParsedCompletion,
    prediction: ParsedCompletion,
    *,
    sequence_confidence: float,
    class_counts: Mapping[str, int],
    config: RewardConfig,
) -> tuple[float, int, float]:
    classes, matrix = _profile_settings(config)
    normalized_counts = dict(class_counts)
    normalized_counts.update({label: max(1, int(class_counts.get(label, 1))) for label in classes})
    reward = 0.0
    eligible = 0
    maximum_severity_factor = 1.0
    for truth, predicted, giou in _all_phase_assignments(
        ground_truth.detections,
        prediction.detections,
        use_giou=True,
    ):
        if giou < config.giou_for_classification:
            continue
        eligible += 1
        if predicted.label == truth.label:
            reward += 1.0
            continue
        penalty, severity_factor = build_rars_confusion_cost(
            predicted.label,
            truth.label,
            sequence_confidence,
            normalized_counts,
            matrix,
            parameters=config.rars_parameters,
            reward_mode=config.reward_mode,
            matrix_mode=config.matrix_mode,
            matrix_scale=config.matrix_scale,
        )
        reward -= penalty
        maximum_severity_factor = max(maximum_severity_factor, severity_factor)
    return (float(reward / eligible) if eligible else 0.0, eligible, maximum_severity_factor)


def _parse_ground_truth(payload: str | Mapping[str, object], classes: Sequence[str]) -> ParsedCompletion:
    try:
        source = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise SchemaError("Ground truth is not a valid object") from error
    if "gt_dets" in source:
        detections = []
        for item in source.get("gt_dets", []) or []:
            converted = dict(item)
            if "bbox_2d" not in converted and "bbox" in converted:
                converted["bbox_2d"] = converted.pop("bbox")
            detections.append(converted)
        source = {
            "detections": detections + [
                {"phase": phase, "label": "background", "bbox_2d": [-1, -1, -1, -1]}
                for phase in source.get("background_phases", []) or []
            ]
        }
    return parse_completion(json.dumps(source), classes)


def compute_score(
    solution: str,
    ground_truth: str | Mapping[str, object],
    *,
    extra_info: Mapping[str, object] | str | None = None,
    config: RewardConfig = RewardConfig(),
) -> dict[str, object]:
    """Compute the deployed reward contract without environment-specific state."""
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = {}
    metadata = dict(extra_info or {})
    sequence_confidence = float(metadata.get("sequence_confidence", metadata.get("seq_confidence", 0.0)))
    if not math.isfinite(sequence_confidence):
        sequence_confidence = 0.0
    empty = _empty_reward(sequence_confidence, config)
    try:
        if config.reward_mode not in {"vanilla", "rars"}:
            raise ValueError(f"Unsupported reward mode: {config.reward_mode}")
        if config.matrix_mode not in {"ordinal", "uniform", "zero"}:
            raise ValueError(f"Unsupported risk matrix mode: {config.matrix_mode}")
        if not math.isfinite(config.matrix_scale) or config.matrix_scale < 0.0:
            raise ValueError("risk matrix scale must be finite and non-negative")
        classes, _ = _profile_settings(config)
        truth = _parse_ground_truth(ground_truth, classes)
        predicted = parse_completion(solution, classes)
        detection, phase_scores, background_correct = detection_reward(truth, predicted)
        classification, classification_matches, severity_factor = classification_reward(
            truth,
            predicted,
            sequence_confidence=sequence_confidence,
            class_counts=metadata.get("class_counts", {}),
            config=config,
        )
        raw = detection
        if classification_matches:
            raw += config.classification_weight * classification
        elif not background_correct:
            raw -= 0.1
        score = max(config.clip_min, min(config.clip_max, raw))
        return {
            "score": float(score),
            "reward_raw": float(raw),
            "parse_ok": 1,
            "detection_reward": float(detection),
            "phase_a_iou": phase_scores["a"],
            "phase_p_iou": phase_scores["p"],
            "phase_v_iou": phase_scores["v"],
            "classification_reward": float(classification),
            "classification_matches": int(classification_matches),
            "sequence_confidence": sequence_confidence,
            "reward_mode": config.reward_mode,
            "risk_matrix_mode": config.matrix_mode,
            "risk_matrix_scale": config.matrix_scale,
            "severity_factor_max": float(severity_factor),
            "background_correct_phases": int(background_correct),
            "background_total_phases": len(truth.background_phases),
        }
    except (SchemaError, KeyError, TypeError, IndexError, ValueError):
        return empty
