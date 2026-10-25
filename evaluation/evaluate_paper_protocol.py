#!/usr/bin/env python3
"""Evaluate the explicit RL-LHM 2D paper protocol from phase-wise annotations.

Inputs are complete group predictions and an evaluation manifest containing
patient identities, corresponding phase images and phase-specific lesion labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.patient import (
    BEST_LIKELIHOOD_INPUT, LIKELIHOOD_WEIGHTED_VOTE, MAJORITY_NLL_TIEBREAK,
    VoteRecord, aggregate_patient,
)
from evaluation.classification import classification_metrics
from evaluation.detection import binary_detection_metrics, evaluate_2d_boxes
from evaluation.schema import SchemaError, canonical_label, parse_completion
from matching.protocols import PROTOCOLS

PHASES = ("a", "p", "v")
PROTOCOL_VERSION = "rl_lhm_paper_2d_v1"
AGGREGATION_RULES = {
    MAJORITY_NLL_TIEBREAK: "group_plurality_patient_plurality_mean_finite_nll_fixed_class_order_v1",
    LIKELIHOOD_WEIGHTED_VOTE: "group_plurality_patient_exp_clip_nll_upper_median_relative_tie_v1",
    BEST_LIKELIHOOD_INPUT: "group_plurality_patient_min_finite_nll_input_order_class_rank_v1",
}


def _identifier(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SchemaError(f"{description} must be a non-empty string without surrounding whitespace")
    return value


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object)
    except json.JSONDecodeError as error:
        raise SchemaError(f"Invalid JSON: {error.msg}") from error


def _macro_classes(payload: dict, classes: Sequence[str]) -> list[str]:
    raw = payload.get("macro_classes", list(classes))
    if not isinstance(raw, list) or not raw:
        raise SchemaError("Manifest macro_classes must be a non-empty ordered list")
    labels = [canonical_label(label) for label in raw]
    if len(labels) != len(set(labels)):
        raise SchemaError("Manifest macro_classes must remain unique after alias normalization")
    if not set(labels).issubset(classes):
        raise SchemaError("Manifest macro_classes must be a subset of classes")
    return labels


def validate_manifest(payload: Any) -> tuple[list[str], dict[str, str], list[dict]]:
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise SchemaError("Manifest must be an object with schema_version=1")
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise SchemaError("Manifest classes must be a non-empty ordered list")
    classes = [canonical_label(label) for label in raw_classes]
    if len(classes) != len(set(classes)):
        raise SchemaError("Manifest classes must remain unique after alias normalization")
    macro_classes = _macro_classes(payload, classes)

    rows = payload.get("patients")
    if not isinstance(rows, list) or not rows:
        raise SchemaError("Manifest patients must contain the entire non-empty evaluation cohort")
    patients = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SchemaError("Each patient must be an object")
        patient_id = _identifier(row.get("patient_id"), "patient_id")
        truth = canonical_label(row.get("truth"))
        if patient_id in patients or truth not in classes:
            raise SchemaError(f"Duplicate patient or out-of-vocabulary patient truth: {patient_id}")
        if truth not in macro_classes:
            raise SchemaError(f"Manifest macro_classes must contain every patient truth: {patient_id}")
        patients[patient_id] = truth

    groups = payload.get("input_groups")
    if not isinstance(groups, list):
        raise SchemaError("Manifest input_groups must be a list of complete three-phase groups")
    validated = []
    seen = set()
    for group in groups:
        if not isinstance(group, dict):
            raise SchemaError("Each input group must be an object")
        group_id = _identifier(group.get("group_id"), "group_id")
        patient_id = _identifier(group.get("patient_id"), "group patient_id")
        if group_id in seen or patient_id not in patients:
            raise SchemaError(f"Duplicate group or unknown patient: {group_id}")
        seen.add(group_id)
        phases = group.get("phases")
        if not isinstance(phases, dict) or set(phases) != set(PHASES):
            raise SchemaError(f"Group {group_id} requires explicit a/p/v phase annotations")
        ground_truth = {}
        for phase in PHASES:
            annotation = phases[phase]
            if not isinstance(annotation, dict):
                raise SchemaError(f"Group {group_id}, phase {phase}: annotation must be an object")
            _identifier(annotation.get("image_ref"), f"{group_id}/{phase} image_ref")
            lesions = annotation.get("ground_truth")
            if not isinstance(lesions, list):
                raise SchemaError(f"Group {group_id}, phase {phase}: ground_truth must be an explicit list")
            records = []
            for lesion in lesions:
                if not isinstance(lesion, dict) or "label" not in lesion or "bbox_2d" not in lesion:
                    raise SchemaError(f"Group {group_id}, phase {phase}: every GT lesion needs its own label and box")
                # An empty list represents a truly lesion-negative phase. Never
                # derive lesion labels or boxes from the patient's diagnosis.
                label = canonical_label(lesion["label"])
                records.append({"phase": phase, "label": label, "bbox_2d": lesion["bbox_2d"]})
            parsed = parse_completion(json.dumps({"detections": records}), classes)
            ground_truth[phase] = parsed.detections
        validated.append({"group_id": group_id, "patient_id": patient_id, "ground_truth": ground_truth})
    return classes, patients, validated


def _completion_nll(value: Any) -> tuple[float | None, str]:
    if value is None:
        return None, "missing"
    if isinstance(value, str):
        # JSON has no numeric NaN/Infinity literals. These explicit strings let
        # exporters retain that distinction without invalid JSON output.
        if value.lower() not in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            raise SchemaError("completion_nll must be numeric, null, or an explicit non-finite string")
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SchemaError("completion_nll must be numeric, null, or an explicit non-finite string")
    value = float(value)
    if not math.isfinite(value):
        return None, "nonfinite"
    return value, "finite"


def read_predictions(path: Path, expected_group_ids: set[str]) -> dict[str, dict]:
    predictions = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = _read_json(line)
            if not isinstance(row, dict):
                raise SchemaError(f"Prediction line {line_number} must be an object")
            group_id = _identifier(row.get("group_id"), "prediction group_id")
            if group_id not in expected_group_ids or group_id in predictions:
                raise SchemaError(f"Unknown or duplicate prediction group_id: {group_id}")
            if not isinstance(row.get("raw"), str):
                raise SchemaError(f"Prediction {group_id} requires the actual raw completion string")
            nll, nll_status = _completion_nll(row.get("completion_nll"))
            predictions[group_id] = {"raw": row["raw"], "nll": nll, "nll_status": nll_status}
    missing = expected_group_ids - predictions.keys()
    if missing:
        raise SchemaError(f"Missing prediction records for {len(missing)} manifest groups; first IDs: {sorted(missing)[:5]}")
    return predictions


def evaluate(manifest: Any, prediction_path: Path, thresholds: Sequence[float],
             aggregation_strategy: str = MAJORITY_NLL_TIEBREAK) -> dict:
    if aggregation_strategy not in AGGREGATION_RULES:
        raise SchemaError(f"Unknown paper aggregation strategy: {aggregation_strategy}")
    classes, patients, groups = validate_manifest(manifest)
    macro_classes = _macro_classes(manifest, classes)
    thresholds = list(thresholds)
    if not thresholds or len(thresholds) != len(set(thresholds)) or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise SchemaError("IoU thresholds must be unique finite values in [0,1]")
    predictions = read_predictions(prediction_path, {group["group_id"] for group in groups})
    rank = {label: index for index, label in enumerate(classes)}
    votes = defaultdict(list)
    group_counts = Counter(group["patient_id"] for group in groups)
    group_details = []
    positive_phases = 0
    excluded_phases = 0
    excluded_predictions = 0
    totals = {
        (protocol, threshold): {"tp": 0, "fp": 0, "fn": 0, "gt_instances": 0, "predicted_instances": 0, "iou_sum": 0.0}
        for protocol in PROTOCOLS for threshold in thresholds
    }
    patient_totals = {patient_id: {key: dict.fromkeys(counts, 0) for key, counts in totals.items()} for patient_id in patients}
    for order, group in enumerate(groups):
        prediction = predictions[group["group_id"]]
        try:
            parsed = parse_completion(prediction["raw"], classes)
            detections = parsed.detections
            valid = True
        except SchemaError:
            detections = ()
            valid = False
        label_counts = Counter(item.label for item in detections)
        group_prediction = min(label_counts, key=lambda label: (-label_counts[label], rank[label])) if label_counts else None
        if group_prediction is not None:
            votes[group["patient_id"]].append(VoteRecord(group_prediction, prediction["nll"], order))
        group_details.append({
            "group_id": group["group_id"], "patient_id": group["patient_id"],
            "schema_valid": valid, "diagnostic_eligible": group_prediction is not None,
            "prediction": group_prediction, "completion_nll": prediction["nll"],
            "nll_status": prediction["nll_status"], "predicted_lesions": len(detections),
        })
        for phase in PHASES:
            truth = group["ground_truth"][phase]
            phase_predictions = [item for item in detections if item.phase == phase]
            if not truth:
                excluded_phases += 1
                excluded_predictions += len(phase_predictions)
                continue
            positive_phases += 1
            for protocol in PROTOCOLS:
                for threshold in thresholds:
                    result = evaluate_2d_boxes(
                        [item.box for item in truth], [item.box for item in phase_predictions],
                        [item.label for item in truth], [item.label for item in phase_predictions],
                        protocol, threshold,
                    )
                    for current in (totals[(protocol, threshold)], patient_totals[group["patient_id"]][(protocol, threshold)]):
                        for name in ("tp", "fp", "fn", "gt_instances", "predicted_instances"):
                            current[name] += getattr(result, name)
                        current["iou_sum"] += result.mean_iou_with_misses_zero * result.gt_instances

    patient_details = []
    for patient_id in sorted(patients):
        result = aggregate_patient(votes[patient_id], classes, aggregation_strategy)
        patient_details.append({
            "patient_id": patient_id, "truth": patients[patient_id], "prediction": result.prediction,
            "input_groups": group_counts[patient_id], "valid_votes": result.valid_votes,
            "tied": result.tied, "abstained": result.abstained,
        })
    metrics = classification_metrics(
        [row["truth"] for row in patient_details], [row["prediction"] for row in patient_details], macro_classes,
    )
    detection = {protocol: [] for protocol in PROTOCOLS}
    for (protocol, threshold), counts in totals.items():
        iou_sum = counts.pop("iou_sum")
        detection[protocol].append({
            "iou_threshold": threshold,
            "threshold_rule": "IoU > 0" if threshold == 0 else "IoU >= threshold",
            **counts, **binary_detection_metrics(counts["tp"], counts["fp"], counts["fn"]),
            "mean_iou_with_misses_zero": iou_sum / counts["gt_instances"] if counts["gt_instances"] else 0.0,
        })
    return {
        "protocol_version": PROTOCOL_VERSION,
        "aggregation_strategy": aggregation_strategy,
        "evaluation_identity": {
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "aggregation_rule": AGGREGATION_RULES[aggregation_strategy],
        },
        "sources": {"predictions": {
            "path": str(prediction_path.resolve()), "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        }},
        "patient_detection": [
            {"patient_id": patient_id, "detection": {
                protocol: [{"iou_threshold": threshold, **patient_totals[patient_id][(protocol, threshold)]}
                           for threshold in thresholds]
                for protocol in PROTOCOLS
            }} for patient_id in sorted(patients)
        ],
        "classes": classes,
        "macro_classes": macro_classes,
        "classification": metrics,
        "coverage": {
            "patients": len(patients), "manifest_groups": len(groups), "prediction_records": len(predictions),
            "missing_prediction_records": 0,
            "schema_valid_groups": sum(row["schema_valid"] for row in group_details),
            "diagnostic_eligible_groups": sum(row["diagnostic_eligible"] for row in group_details),
            "patients_without_valid_vote": sum(row["abstained"] for row in patient_details),
            "patients_without_complete_input_groups": sum(row["input_groups"] == 0 for row in patient_details),
            "patient_diagnostic_coverage": sum(not row["abstained"] for row in patient_details) / len(patients),
        },
        "detection_scope": {
            "scope": "lesion-positive phase slices only",
            "positive_phase_slices": positive_phases,
            "excluded_no_lesion_phase_slices": excluded_phases,
            "eligible_predictions_on_excluded_phases": excluded_predictions,
        },
        "detection": detection,
        "patients": patient_details,
        "input_groups": group_details,
        "input_contract": {
            "coverage_source": "complete patient and input-group manifest",
            "annotations": "independent phase-wise lesion annotations",
        },
    }


def _write_csv(path: Path, rows: list[dict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(result: dict, output_dir: Path) -> None:
    """Write the canonical JSON and keyed CSV exports to a fresh destination."""
    paths = [output_dir / name for name in ("metrics.json", "patient_predictions.csv", "input_group_audit.csv")]
    if any(path.exists() for path in paths):
        raise SchemaError("Output files already exist; choose a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(paths[1], result["patients"], ("patient_id", "truth", "prediction", "input_groups", "valid_votes", "tied", "abstained"))
    _write_csv(paths[2], result["input_groups"], ("group_id", "patient_id", "schema_valid", "diagnostic_eligible", "prediction", "completion_nll", "nll_status", "predicted_lesions"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="One JSONL prediction record per manifest group")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--aggregation-strategy", choices=tuple(AGGREGATION_RULES), default=MAJORITY_NLL_TIEBREAK)
    parser.add_argument("--model-name", help="Run identity for joint five-seed statistics")
    parser.add_argument("--model-seed", type=int, help="Training or inference seed that produced these predictions")
    args = parser.parse_args()
    if (args.model_name is None) != (args.model_seed is None):
        parser.error("--model-name and --model-seed must be supplied together")
    try:
        result = evaluate(_read_json(args.manifest.read_text(encoding="utf-8")), args.predictions, args.iou_thresholds, args.aggregation_strategy)
        if args.model_name is not None:
            result["run_identity"] = {"model_name": _identifier(args.model_name, "model_name"), "seed": args.model_seed}
        result["sources"] = {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("manifest", args.manifest), ("predictions", args.predictions))
        }
        write_evaluation(result, args.output_dir)
    except (OSError, SchemaError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "protocol_version": PROTOCOL_VERSION, "coverage": result["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
