import json
import math
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# VeRL also loads this callback by absolute file path outside the repository
# working directory. Make the sibling schema package available in that mode.
_repository_root = str(Path(__file__).resolve().parents[1])
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)

from evaluation.schema import DEFAULT_LABEL_ALIASES, SchemaError, canonical_label, parse_completion

CLASSES_PRIVATE = ["HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"]
CLASSES_5CLS = ["HCC", "ICC", "HA_Hemangioma", "BCLM", "CRLM"]
CLASSES_3CLS = ["HCC", "ICC", "HA_Hemangioma"]

GIOU_FOR_CLS = 0.30
RARS_SCARCITY_ALPHA = 0.35
RARS_CONFIDENCE_ETA = 0.75
RARS_COUNT_EPS = 1e-6
RARS_SEVERITY_KAPPA = 0.5

CLS_WEIGHT = 1.0

REWARD_CLIP_MIN = -6.0
REWARD_CLIP_MAX = 3.0

# Frozen tuning-selected severity matrices (risk_tuning_v1/primary_selection_v7
# freeze, status "frozen_selected_risk_matrix_for_primary_full_rl"). These are
# the operative weights for the primary Full RL-LHM configuration and match
# rewards/rars.py (default_matrix) and the
# paper's Table of frozen RARS error-pair weights. The joint grid
# operates on unordered error pairs, so every matrix is symmetric; pairs not
# listed individually take the shared "other" weight 0.40. The mode name
# "ordinal" is retained only for scheduler/freeze contract compatibility.
RISK_SEVERITY_MATRIX_PRIVATE = {
    "HCC": {"ICC": 1.10, "FNH": 0.90, "HA_Hemangioma": 0.50, "nodule_CYST": 0.40},
    "ICC": {"HCC": 1.10, "FNH": 0.40, "HA_Hemangioma": 0.70, "nodule_CYST": 0.40},
    "FNH": {"HCC": 0.90, "ICC": 0.40, "HA_Hemangioma": 0.40, "nodule_CYST": 0.40},
    "HA_Hemangioma": {"HCC": 0.50, "ICC": 0.70, "FNH": 0.40, "nodule_CYST": 0.40},
    "nodule_CYST": {"HCC": 0.40, "ICC": 0.40, "FNH": 0.40, "HA_Hemangioma": 0.40},
}

RISK_SEVERITY_MATRIX_MCT = {
    "HCC": {"ICC": 1.10, "HA_Hemangioma": 0.60, "BCLM": 0.40, "CRLM": 0.40},
    "ICC": {"HCC": 1.10, "HA_Hemangioma": 0.80, "BCLM": 1.00, "CRLM": 1.10},
    "HA_Hemangioma": {"HCC": 0.60, "ICC": 0.80, "BCLM": 0.40, "CRLM": 0.40},
    "BCLM": {"HCC": 0.40, "ICC": 1.00, "HA_Hemangioma": 0.40, "CRLM": 0.90},
    "CRLM": {"HCC": 0.40, "ICC": 1.10, "HA_Hemangioma": 0.40, "BCLM": 0.90},
}

# Author-nominal pre-selection matrices, retained only as the documented
# sensitivity control (freeze field "author_defined_nominal_retained_as_control").
# They are NOT used by the primary Full RL-LHM configuration
# ("author_nominal_used_in_primary_candidates": False).
AUTHOR_NOMINAL_RISK_SEVERITY_MATRIX_PRIVATE = {
    "HCC": {"ICC": 0.55, "FNH": 0.20, "HA_Hemangioma": 0.15, "nodule_CYST": 0.10},
    "ICC": {"HCC": 1.35, "FNH": 0.45, "HA_Hemangioma": 0.35, "nodule_CYST": 0.30},
    "FNH": {"HCC": 0.25, "ICC": 0.45, "HA_Hemangioma": 0.20, "nodule_CYST": 0.15},
    "HA_Hemangioma": {"HCC": 0.25, "ICC": 0.45, "FNH": 0.20, "nodule_CYST": 0.10},
    "nodule_CYST": {"HCC": 0.20, "ICC": 0.35, "FNH": 0.15, "HA_Hemangioma": 0.10},
}

AUTHOR_NOMINAL_RISK_SEVERITY_MATRIX_MCT = {
    "HCC": {"ICC": 1.35, "HA_Hemangioma": 0.20, "BCLM": 0.60, "CRLM": 0.60},
    "ICC": {"HCC": 1.45, "HA_Hemangioma": 0.35, "BCLM": 0.55, "CRLM": 0.55},
    "HA_Hemangioma": {"HCC": 0.25, "ICC": 0.45, "BCLM": 0.30, "CRLM": 0.30},
    "BCLM": {"HCC": 0.85, "ICC": 0.75, "HA_Hemangioma": 0.35, "CRLM": 0.50},
    "CRLM": {"HCC": 0.85, "ICC": 0.75, "HA_Hemangioma": 0.35, "BCLM": 0.50},
}


def _empty_reward(seq_confidence: float = 0.0) -> Dict[str, Any]:
    return {
        "score": 0.0,
        "reward_raw": 0.0,
        "parse_ok": 0,
        "det_reward": 0.0,
        "phase_a_iou": 0.0,
        "phase_p_iou": 0.0,
        "phase_v_iou": 0.0,
        "cls_score": 0.0,
        "cls_matched_cnt": 0,
        "major_match": 0,
        "pred_icc_frac": 0.0,
        "spurious_icc_aligned": 0,
        "consistency_bonus": 0.0,
        "seq_confidence": float(seq_confidence),
        "reward_mode": os.environ.get("RL_LHM_REWARD_MODE", "rars").lower(),
        "risk_matrix_mode": os.environ.get("RL_LHM_RISK_MATRIX_MODE", "ordinal").lower(),
        "risk_matrix_scale": float(os.environ.get("RL_LHM_RISK_MATRIX_SCALE", "1.0")),
        "severity_factor_max": 1.0,
        "background_correct_phases": 0,
        "background_total_phases": 0,
    }


def canon_label(x: Optional[str]) -> Optional[str]:
    """Resolve only the exact aliases accepted by the evaluation schema.

    A label embedded in free text (for example ``not HCC`` or ``HCC or ICC``)
    is not a diagnostic label. Keep the optional-return adapter for callers
    while sharing the whitelist with the strict completion parser.
    """
    try:
        return canonical_label(x)
    except SchemaError:
        return None


def is_background_label(x: Optional[str]) -> bool:
    if x is None:
        return False
    value = str(x).strip().lower().replace("_", "-").replace(" ", "")
    return value in {"no-lesion", "nolesion", "background", "bg"}


def bbox_cxcywh_to_xyxy(bbox: List[float]) -> List[float]:
    cx, cy, w, h = [float(v) for v in bbox]
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def normalize_bbox_to_xyxy(
    bbox: Any,
    assume_cxcywh: Optional[bool] = None,
) -> Optional[List[float]]:
    if bbox is None:
        return None
    inferred_cxcywh = assume_cxcywh
    if isinstance(bbox, dict):
        kk = {k.lower(): k for k in bbox.keys()}
        if all(k in kk for k in ["x1", "y1", "x2", "y2"]):
            bbox = [bbox[kk[k]] for k in ["x1", "y1", "x2", "y2"]]
            inferred_cxcywh = False
        elif all(k in kk for k in ["xmin", "ymin", "xmax", "ymax"]):
            bbox = [bbox[kk[k]] for k in ["xmin", "ymin", "xmax", "ymax"]]
            inferred_cxcywh = False
        elif all(k in kk for k in ["cx", "cy", "w", "h"]):
            bbox = [bbox[kk[k]] for k in ["cx", "cy", "w", "h"]]
            inferred_cxcywh = True
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    vals = [float(v) for v in bbox]
    if inferred_cxcywh is None:
        inferred_cxcywh = False
    return bbox_cxcywh_to_xyxy(vals) if inferred_cxcywh else vals


def bbox_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(min(ax1, ax2), min(bx1, bx2))
    inter_y1 = max(min(ay1, ay2), min(by1, by2))
    inter_x2 = min(max(ax1, ax2), max(bx1, bx2))
    inter_y2 = min(max(ay1, ay2), max(by1, by2))
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, max(ax1, ax2) - min(ax1, ax2)) * max(0.0, max(ay1, ay2) - min(ay1, ay2))
    area_b = max(0.0, max(bx1, bx2) - min(bx1, bx2)) * max(0.0, max(by1, by2) - min(by1, by2))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_giou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx1, by1, bx2, by2 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return -1.0
    iou = intersection / union
    cover_x1, cover_y1 = min(ax1, bx1), min(ay1, by1)
    cover_x2, cover_y2 = max(ax2, bx2), max(ay2, by2)
    cover_area = max(0.0, cover_x2 - cover_x1) * max(0.0, cover_y2 - cover_y1)
    return float(iou - ((cover_area - union) / cover_area)) if cover_area > 0 else float(iou)


def parse_completion_from_text(
    txt: str,
    classes: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Adapt the shared strict parser to the legacy reward record structure.

    Without a cohort argument, accept the union of documented class labels;
    callers with an active label space must supply it or apply their own
    cohort filter. Any invalid record invalidates the entire completion.
    """
    try:
        parsed = parse_completion(
            txt,
            DEFAULT_LABEL_ALIASES.values() if classes is None else classes,
        )
    except SchemaError:
        return None
    return {
        "lesions": [
            {"label": item.label, "phase": item.phase, "bbox": list(item.box)}
            for item in parsed.detections
        ],
        "background_phases": list(parsed.background_phases),
    }


def parse_detections_from_text(txt: str) -> List[Dict[str, Any]]:
    parsed = parse_completion_from_text(txt)
    return [] if parsed is None else parsed["lesions"]


def best_iou_match_any_label(gt_det: Dict[str, Any], pred_dets: List[Dict[str, Any]]) -> Tuple[float, Optional[Dict[str, Any]]]:
    best_iou = 0.0
    best_pred = None
    if gt_det.get("bbox") is None:
        return best_iou, best_pred
    for p in pred_dets:
        if p.get("bbox") is None:
            continue
        iou = bbox_iou_xyxy(gt_det["bbox"], p["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_pred = p
    return best_iou, best_pred


def best_iou_of_pred_to_any_gt(pred_det: Dict[str, Any], gt_dets: List[Dict[str, Any]]) -> float:
    if pred_det.get("bbox") is None:
        return 0.0
    best = 0.0
    for g in gt_dets:
        if g.get("bbox") is None:
            continue
        best = max(best, bbox_iou_xyxy(pred_det["bbox"], g["bbox"]))
    return best


def match_phase_detections(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    gt = [item for item in ground_truth if item.get("bbox") is not None]
    pred = [item for item in predictions if item.get("bbox") is not None]
    if not gt or not pred:
        return []
    matrix = np.asarray(
        [[bbox_iou_xyxy(gt_item["bbox"], pred_item["bbox"]) for pred_item in pred] for gt_item in gt],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(-matrix)
    return [
        (gt[int(row)], pred[int(column)], float(matrix[row, column]))
        for row, column in zip(rows, columns)
    ]


def match_detections(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    matches = []
    for phase in ("a", "p", "v"):
        matches.extend(
            match_phase_detections(
                [item for item in ground_truth if item.get("phase") == phase],
                [item for item in predictions if item.get("phase") == phase],
            )
        )
    return matches


def match_detections_for_classification(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    matches = []
    for phase in ("a", "p", "v"):
        phase_gt = [item for item in ground_truth if item.get("phase") == phase and item.get("bbox") is not None]
        phase_pred = [item for item in predictions if item.get("phase") == phase and item.get("bbox") is not None]
        if not phase_gt or not phase_pred:
            continue
        matrix = np.asarray(
            [[bbox_giou_xyxy(gt_item["bbox"], pred_item["bbox"]) for pred_item in phase_pred] for gt_item in phase_gt],
            dtype=np.float64,
        )
        rows, columns = linear_sum_assignment(-matrix)
        matches.extend(
            (phase_gt[int(row)], phase_pred[int(column)], float(matrix[row, column]))
            for row, column in zip(rows, columns)
        )
    return matches


def paper_detection_reward(
    preds: List[Dict[str, Any]],
    gt_dets: List[Dict[str, Any]],
    predicted_background_phases: Optional[List[str]] = None,
    ground_truth_background_phases: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float], int]:
    predicted_background = set(predicted_background_phases or [])
    ground_truth_background = set(ground_truth_background_phases or [])
    phase_scores: Dict[str, float] = {}
    scores = []
    background_correct = 0
    for ph in ("a", "p", "v"):
        phase_gt = [item for item in gt_dets if item.get("phase") == ph and item.get("bbox") is not None]
        phase_pred = [item for item in preds if item.get("phase") == ph and item.get("bbox") is not None]
        if phase_gt:
            matches = match_phase_detections(phase_gt, phase_pred)
            val = sum(iou for _, _, iou in matches) / len(phase_gt)
        elif ph in ground_truth_background:
            val = float(ph in predicted_background and not phase_pred)
            background_correct += int(val)
        else:
            val = 0.0
        phase_scores[ph] = val
        scores.append(val)
    return float(sum(scores) / 3.0), phase_scores, background_correct


def get_setting(data_mode: str):
    if data_mode == "private":
        return CLASSES_PRIVATE
    if data_mode == "3cls":
        return CLASSES_3CLS
    return CLASSES_5CLS


@lru_cache(maxsize=4)
def _explicit_risk_profile(path: str):
    from training.rars_joint_grid import validate_profile
    return validate_profile(json.loads(Path(path).read_text(encoding="utf-8")))


def get_risk_severity(pred_label: str, true_label: str, data_mode: str, matrix_mode: str) -> float:
    if pred_label == true_label or matrix_mode == "zero":
        return 0.0
    if matrix_mode == "uniform":
        return 1.0
    profile_path = os.environ.get("RL_LHM_RISK_PROFILE")
    if profile_path:
        profile = _explicit_risk_profile(profile_path)
        expected = "private" if data_mode == "private" else "public"
        if profile["cohort"] != expected:
            raise RuntimeError("Risk profile cohort differs from reward data_mode")
        for pair, weight in profile["pair_weights"].items():
            if set(pair.split("--")) == {pred_label, true_label}:
                return float(weight)
        return float(profile["other_weight"])
    matrix = RISK_SEVERITY_MATRIX_PRIVATE if data_mode == "private" else RISK_SEVERITY_MATRIX_MCT
    return float(matrix.get(true_label, {}).get(pred_label, 0.0))


def build_rars_confusion_cost(
    pred_label: str,
    true_label: str,
    confidence: float,
    class_counts: Dict[str, int],
    data_mode: str,
    reward_mode: str,
    matrix_mode: str,
    matrix_scale: float,
) -> tuple[float, float]:
    if pred_label == true_label:
        return 0.0, 1.0
    if reward_mode == "vanilla":
        return 1.0, 1.0
    total = max(1, sum(int(v) for v in class_counts.values()))
    n_true = max(1, int(class_counts.get(true_label, 1)))
    scarcity = (total / (n_true + RARS_COUNT_EPS)) ** RARS_SCARCITY_ALPHA
    calib = 1.0 + RARS_CONFIDENCE_ETA * max(0.0, min(1.0, float(confidence)))
    severity = get_risk_severity(pred_label, true_label, data_mode, matrix_mode)
    severity_factor = math.exp(RARS_SEVERITY_KAPPA * matrix_scale * severity)
    return float(scarcity * severity_factor * calib), float(severity_factor)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    seq_confidence = 0.0
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            extra_info = {}
    extra_info = extra_info or {}
    seq_confidence = float(extra_info.get("seq_confidence", 0.0))

    try:
        gt_obj = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    except Exception:
        return _empty_reward(seq_confidence=seq_confidence)

    try:
        data_mode = extra_info.get("data_mode", "5cls")
        class_counts = extra_info.get("class_counts", {})
        reward_mode = os.environ.get("RL_LHM_REWARD_MODE", "rars").lower()
        matrix_mode = os.environ.get("RL_LHM_RISK_MATRIX_MODE", "ordinal").lower()
        matrix_scale = float(os.environ.get("RL_LHM_RISK_MATRIX_SCALE", "1.0"))
        if reward_mode not in {"vanilla", "rars"}:
            raise ValueError(f"Unsupported reward mode: {reward_mode}")
        if matrix_mode not in {"ordinal", "uniform", "zero"}:
            raise ValueError(f"Unsupported risk matrix mode: {matrix_mode}")
        if not math.isfinite(matrix_scale) or matrix_scale < 0:
            raise ValueError(f"Invalid risk matrix scale: {matrix_scale}")

        classes = get_setting(data_mode)
        for c in classes:
            class_counts[c] = max(1, int(class_counts.get(c, 1)))

        gt_dets = gt_obj.get("gt_dets", []) or []
        gt_background_phases = gt_obj.get("background_phases", []) or []
        parsed = parse_completion_from_text(solution_str, classes)
        if parsed is None:
            return _empty_reward(seq_confidence=seq_confidence)
        preds = parsed["lesions"]
        pred_background_phases = parsed["background_phases"]
        if any(prediction["label"] not in classes for prediction in preds):
            return _empty_reward(seq_confidence=seq_confidence)

        reward, phase_scores, background_correct = paper_detection_reward(
            preds,
            gt_dets,
            pred_background_phases,
            gt_background_phases,
        )

        pred_labels = [p["label"] for p in preds if p.get("label") in classes]
        pred_agg = Counter(pred_labels)
        pred_icc_frac = pred_agg.get("ICC", 0) / max(len(pred_labels), 1)

        cls_score = 0.0
        cls_cnt = 0
        severity_factor_max = 1.0
        for g, p, giou in match_detections_for_classification(gt_dets, preds):
            if giou < GIOU_FOR_CLS:
                continue
            cls_cnt += 1
            gt_lab = g["label"]
            pr_lab = p["label"]
            if pr_lab == gt_lab:
                cls_score += 1.0
            else:
                penalty, severity_factor = build_rars_confusion_cost(
                    pr_lab,
                    gt_lab,
                    seq_confidence,
                    class_counts,
                    data_mode,
                    reward_mode,
                    matrix_mode,
                    matrix_scale,
                )
                severity_factor_max = max(severity_factor_max, severity_factor)
                cls_score -= penalty

        cls_score_avg = float(cls_score / cls_cnt) if cls_cnt > 0 else 0.0
        if cls_cnt > 0:
            reward += CLS_WEIGHT * cls_score_avg
        elif background_correct == 0:
            reward -= 0.1

        reward_raw = float(reward)
        reward_clipped = max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, reward_raw))
        return {
            "score": float(reward_clipped),
            "reward_raw": reward_raw,
            "parse_ok": 1,
            "det_reward": float(sum(phase_scores.values()) / 3.0),
            "phase_a_iou": float(phase_scores.get("a", 0.0)),
            "phase_p_iou": float(phase_scores.get("p", 0.0)),
            "phase_v_iou": float(phase_scores.get("v", 0.0)),
            "cls_score": cls_score_avg,
            "cls_matched_cnt": int(cls_cnt),
            "major_match": 0,
            "pred_icc_frac": float(pred_icc_frac),
            "spurious_icc_aligned": 0,
            "consistency_bonus": 0.0,
            "seq_confidence": float(seq_confidence),
            "reward_mode": reward_mode,
            "risk_matrix_mode": matrix_mode,
            "risk_matrix_scale": matrix_scale,
            "severity_factor_max": float(severity_factor_max),
            "background_correct_phases": int(background_correct),
            "background_total_phases": int(len(set(gt_background_phases))),
        }
    except (json.JSONDecodeError, KeyError, TypeError, IndexError, ValueError):
        return _empty_reward(seq_confidence=seq_confidence)
