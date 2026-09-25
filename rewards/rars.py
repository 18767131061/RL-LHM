"""RARS confusion weighting, grid construction, and immutable selection freeze."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


PRIVATE_CLASSES = ("HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST")
PUBLIC_CLASSES = ("HCC", "ICC", "HA_Hemangioma", "BCLM", "CRLM")

RISK_SEVERITY_MATRIX_PRIVATE = {
    "HCC": {"ICC": 1.10, "FNH": 0.90, "HA_Hemangioma": 0.50, "nodule_CYST": 0.40},
    "ICC": {"HCC": 1.10, "FNH": 0.40, "HA_Hemangioma": 0.70, "nodule_CYST": 0.40},
    "FNH": {"HCC": 0.90, "ICC": 0.40, "HA_Hemangioma": 0.40, "nodule_CYST": 0.40},
    "HA_Hemangioma": {"HCC": 0.50, "ICC": 0.70, "FNH": 0.40, "nodule_CYST": 0.40},
    "nodule_CYST": {"HCC": 0.40, "ICC": 0.40, "FNH": 0.40, "HA_Hemangioma": 0.40},
}

RISK_SEVERITY_MATRIX_PUBLIC = {
    "HCC": {"ICC": 1.10, "HA_Hemangioma": 0.60, "BCLM": 0.40, "CRLM": 0.40},
    "ICC": {"HCC": 1.10, "HA_Hemangioma": 0.80, "BCLM": 1.00, "CRLM": 1.10},
    "HA_Hemangioma": {"HCC": 0.60, "ICC": 0.80, "BCLM": 0.40, "CRLM": 0.40},
    "BCLM": {"HCC": 0.40, "ICC": 1.00, "HA_Hemangioma": 0.40, "CRLM": 0.90},
    "CRLM": {"HCC": 0.40, "ICC": 1.10, "HA_Hemangioma": 0.40, "BCLM": 0.90},
}

UNIFORM_RISK_WEIGHT = {"private": 0.5600, "public": 0.7100}


@dataclass(frozen=True)
class RARSParameters:
    scarcity_alpha: float = 0.35
    confidence_eta: float = 0.75
    severity_kappa: float = 0.5
    count_epsilon: float = 1e-6
    reward_clip_min: float = -6.0
    reward_clip_max: float = 3.0


def _validate_matrix(matrix: Mapping[str, Mapping[str, float]], labels: Sequence[str]) -> None:
    if set(matrix) != set(labels):
        raise ValueError("Risk matrix rows must equal the declared labels")
    for truth in labels:
        expected = set(labels) - {truth}
        if set(matrix[truth]) != expected:
            raise ValueError(f"Risk matrix row {truth} must contain every off-diagonal label")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in matrix[truth].values()):
            raise ValueError("Risk weights must be finite and non-negative")


def risk_severity(
    predicted_label: str,
    true_label: str,
    matrix: Mapping[str, Mapping[str, float]],
    matrix_mode: str = "ordinal",
) -> float:
    if predicted_label == true_label or matrix_mode == "zero":
        return 0.0
    if matrix_mode == "uniform":
        return 1.0
    if matrix_mode != "ordinal":
        raise ValueError(f"Unsupported matrix mode: {matrix_mode}")
    return float(matrix.get(true_label, {}).get(predicted_label, 0.0))


def build_rars_confusion_cost(
    predicted_label: str,
    true_label: str,
    confidence: float,
    class_counts: Mapping[str, int],
    matrix: Mapping[str, Mapping[str, float]],
    parameters: RARSParameters = RARSParameters(),
    reward_mode: str = "rars",
    matrix_mode: str = "ordinal",
    matrix_scale: float = 1.0,
) -> tuple[float, float]:
    if predicted_label == true_label:
        return 0.0, 1.0
    if reward_mode == "vanilla":
        return 1.0, 1.0
    if reward_mode != "rars":
        raise ValueError(f"Unsupported reward mode: {reward_mode}")
    if not math.isfinite(matrix_scale) or matrix_scale < 0.0:
        raise ValueError("Matrix scale must be finite and non-negative")
    total = max(1, sum(max(0, int(value)) for value in class_counts.values()))
    true_count = max(1, int(class_counts.get(true_label, 1)))
    scarcity = (total / (true_count + parameters.count_epsilon)) ** parameters.scarcity_alpha
    calibrated = 1.0 + parameters.confidence_eta * min(1.0, max(0.0, float(confidence)))
    severity = risk_severity(predicted_label, true_label, matrix, matrix_mode)
    severity_factor = math.exp(parameters.severity_kappa * matrix_scale * severity)
    return float(scarcity * severity_factor * calibrated), float(severity_factor)


def matrix_from_pair_weights(
    labels: Sequence[str],
    pair_weights: Mapping[str, float],
    shared_other: float,
) -> dict[str, dict[str, float]]:
    if not math.isfinite(shared_other) or shared_other < 0.0:
        raise ValueError("shared_other must be finite and non-negative")
    for left, right in itertools.combinations(labels, 2):
        forward = f"{left}->{right}"
        reverse = f"{right}->{left}"
        if forward in pair_weights and reverse in pair_weights:
            if float(pair_weights[forward]) != float(pair_weights[reverse]):
                raise ValueError(
                    f"Conflicting weights for unordered class pair {left}<->{right}"
                )

    matrix = {}
    for truth in labels:
        matrix[truth] = {}
        for prediction in labels:
            if prediction == truth:
                continue
            forward = f"{truth}->{prediction}"
            reverse = f"{prediction}->{truth}"
            value = float(pair_weights.get(forward, pair_weights.get(reverse, shared_other)))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid risk weight for {truth}<->{prediction}")
            matrix[truth][prediction] = value
    _validate_matrix(matrix, labels)
    return matrix


def _profile_hash(profile: Mapping) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_risk_grid(
    labels: Sequence[str],
    parameter_values: Mapping[str, Sequence[float]],
    shared_other_values: Sequence[float],
    fixed_pair_weights: Mapping[str, float] | None = None,
) -> list[dict]:
    """Generate a two-axis grid while holding other selected weights fixed."""

    names = sorted(parameter_values)
    axes = [tuple(float(value) for value in parameter_values[name]) for name in names]
    if any(not values for values in axes) or not shared_other_values:
        raise ValueError("Every risk-grid axis must contain at least one value")
    profiles = []
    for values in itertools.product(*axes, tuple(float(value) for value in shared_other_values)):
        parameters = dict(zip(names, values[:-1]))
        pair_weights = dict(fixed_pair_weights or {})
        pair_weights.update(parameters)
        matrix = matrix_from_pair_weights(labels, pair_weights, values[-1])
        body = {
            "parameter_values": parameters,
            "fixed_pair_weights": dict(fixed_pair_weights or {}),
            "shared_other": values[-1],
            "matrix": matrix,
        }
        profiles.append({"profile_id": _profile_hash(body)[:16], **body})
    return profiles


def freeze_risk_selection(
    selected_profile: Mapping,
    development_manifest_sha256: str,
    output_path: str | Path | None = None,
) -> dict:
    if len(development_manifest_sha256) != 64:
        raise ValueError("development_manifest_sha256 must be a SHA-256 hex digest")
    profile = dict(selected_profile)
    profile_sha256 = _profile_hash(profile)
    freeze = {
        "schema_version": 1,
        "status": "frozen_selected_risk_matrix_for_primary_full_rl",
        "selection_role": "development_only_pre_test_selection",
        "formal_test_used_for_selection": False,
        "selected_profile": profile,
        "selected_profile_sha256": profile_sha256,
        "development_manifest_sha256": development_manifest_sha256,
    }
    freeze["freeze_sha256"] = _profile_hash(freeze)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    return freeze


def validate_risk_freeze(freeze: Mapping) -> None:
    required = {
        "schema_version": 1,
        "status": "frozen_selected_risk_matrix_for_primary_full_rl",
        "selection_role": "development_only_pre_test_selection",
        "formal_test_used_for_selection": False,
    }
    failed = [key for key, expected in required.items() if freeze.get(key) != expected]
    if failed:
        raise ValueError(f"Invalid freeze fields: {failed}")
    profile = freeze.get("selected_profile")
    if not isinstance(profile, Mapping) or freeze.get("selected_profile_sha256") != _profile_hash(profile):
        raise ValueError("Selected profile hash mismatch")
    body = dict(freeze)
    recorded = body.pop("freeze_sha256", None)
    if recorded != _profile_hash(body):
        raise ValueError("Freeze hash mismatch")


def default_matrix(cohort: str) -> dict[str, dict[str, float]]:
    if cohort == "private":
        return {truth: dict(row) for truth, row in RISK_SEVERITY_MATRIX_PRIVATE.items()}
    if cohort == "public":
        return {truth: dict(row) for truth, row in RISK_SEVERITY_MATRIX_PUBLIC.items()}
    raise ValueError(f"Unsupported cohort: {cohort}")


def uniform_matrix(cohort: str) -> dict[str, dict[str, float]]:
    if cohort == "private":
        labels = PRIVATE_CLASSES
    elif cohort == "public":
        labels = PUBLIC_CLASSES
    else:
        raise ValueError(f"Unsupported cohort: {cohort}")
    return matrix_from_pair_weights(labels, {}, UNIFORM_RISK_WEIGHT[cohort])


def parameters_as_dict(parameters: RARSParameters = RARSParameters()) -> dict:
    return asdict(parameters)
