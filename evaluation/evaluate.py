"""Evaluate keyed input-group predictions over a fixed patient manifest."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from aggregation.patient import VoteRecord, aggregate_patient, group_prediction
from matching.geometry import box_iou_matrix
from matching.protocols import CLASS_AGNOSTIC, PROTOCOLS
from rewards.schema import PHASES, SchemaError, parse_completion
from .classification import classification_metrics
from .detection import binary_detection_metrics, evaluate_similarity_matrix

if TYPE_CHECKING:
    from data.manifest import Manifest


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"JSON numbers must be finite: {value}")


def read_predictions(path: str | Path) -> list[dict]:
    """Read JSONL rows containing uid, completion, and nll (number or null)."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
            except (ValueError, TypeError, RecursionError) as error:
                raise ValueError(f"prediction line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"prediction line {line_number} must be an object")
            rows.append(row)
    return rows


def _prediction_map(
    rows: Sequence[Mapping[str, object]], expected: Mapping[str, str]
) -> dict[str, dict]:
    predictions = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each prediction must be an object")
        uid = row.get("uid")
        if not isinstance(uid, str) or uid not in expected:
            raise ValueError(f"prediction uid is absent from the manifest: {uid!r}")
        if uid in predictions:
            raise ValueError(f"duplicate prediction uid: {uid}")
        if "patient_id" in row and row["patient_id"] != expected[uid]:
            raise ValueError(f"prediction patient_id disagrees with the manifest: {uid}")
        if not isinstance(row.get("completion"), str):
            raise ValueError(f"prediction {uid} needs its raw completion string")
        if "nll" not in row:
            raise ValueError(f"prediction {uid} needs nll (number or null)")
        nll = row["nll"]
        if nll is not None:
            if isinstance(nll, bool) or not isinstance(nll, (int, float)):
                raise ValueError(f"prediction {uid} nll must be numeric or null")
            try:
                nll = float(nll)
            except OverflowError as error:
                raise ValueError(f"prediction {uid} nll must be finite or null") from error
            if not math.isfinite(nll):
                raise ValueError(f"prediction {uid} nll must be finite or null")
        predictions[uid] = {"completion": row["completion"], "nll": nll}
    missing = set(expected) - predictions.keys()
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} manifest groups: {sorted(missing)[:5]}")
    return predictions


def evaluate(
    manifest: Manifest,
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    iou_thresholds: Sequence[float],
) -> dict:
    """Score all manifest patients and all input phase slices.

    Prediction UIDs must cover the manifest records exactly. A schema-invalid
    completion contributes no lesions or diagnostic vote. Patients with no vote,
    including patients with zero records, remain in the classification metrics.
    Mean IoU uses unthresholded class-agnostic assignment with all GT lesions in
    its denominator; class-aware TP counting follows each named protocol.
    """
    classes = tuple(manifest.classes)
    patients = dict(manifest.patients)
    if not patients:
        raise ValueError("the patient manifest must be nonempty")
    # Check the fixed class space and every patient truth before processing rows.
    classification_metrics(list(patients.values()), [None] * len(patients), classes)
    thresholds = tuple(iou_thresholds)
    if (not thresholds or len(thresholds) != len(set(thresholds))
            or any(isinstance(t, bool) or not isinstance(t, (int, float))
                   or not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds)):
        raise ValueError("iou_thresholds must be unique finite values in [0,1]")
    expected = {}
    truth_by_uid = {}
    for row in manifest.records:
        uid, patient_id = row["uid"], row["patient_id"]
        if not isinstance(uid, str) or not uid.strip() or uid in expected:
            raise ValueError("manifest record UIDs must be unique nonempty strings")
        if patient_id not in patients:
            raise ValueError(f"record {uid} refers to an unlisted patient")
        if "patient_label" in row and row["patient_label"] != patients[patient_id]:
            raise ValueError(f"record {uid} patient_label disagrees with patient manifest")
        expected[uid] = patient_id
        truth_by_uid[uid] = parse_completion(row["ground_truth"], classes)
    predictions = _prediction_map(prediction_rows, expected)
    counts = {
        (protocol, threshold): {"tp": 0, "fp": 0, "fn": 0, "gt_instances": 0, "predicted_instances": 0}
        for protocol in PROTOCOLS for threshold in thresholds
    }
    positive_matches = dict.fromkeys(PROTOCOLS, 0)
    positive_phases, excluded_phases, excluded_predictions = 0, 0, 0
    ca_iou_sum, gt_instances = 0.0, 0
    votes = defaultdict(list)
    group_counts = Counter(expected.values())
    groups = []
    for uid, patient_id in expected.items():
        row = predictions[uid]
        try:
            parsed = parse_completion(row["completion"], classes)
            detections = parsed.detections
            label = group_prediction(parsed, classes)
            schema_valid = True
        except SchemaError:
            detections, label, schema_valid = (), None, False
        votes[patient_id].append(VoteRecord(label, row["nll"]))
        groups.append({
            "uid": uid, "patient_id": patient_id, "schema_valid": schema_valid,
            "prediction": label, "nll": row["nll"], "predicted_lesions": len(detections),
        })
        for phase in PHASES:
            truth = [item for item in truth_by_uid[uid].detections if item.phase == phase]
            predicted = [item for item in detections if item.phase == phase]
            if not truth:
                excluded_phases += 1
                excluded_predictions += len(predicted)
            else:
                positive_phases += 1
            gt_instances += len(truth)
            matrix = box_iou_matrix([item.box for item in truth], [item.box for item in predicted])
            for protocol in PROTOCOLS:
                for index, threshold in enumerate(thresholds):
                    result = evaluate_similarity_matrix(
                        matrix, [item.label for item in truth], [item.label for item in predicted],
                        protocol, threshold,
                    )
                    for name in counts[(protocol, threshold)]:
                        counts[(protocol, threshold)][name] += getattr(result, name)
                    if index == 0:
                        positive_matches[protocol] += result.positive_overlap_matches
                        if protocol == CLASS_AGNOSTIC:
                            ca_iou_sum += result.assigned_iou_sum
    patient_rows = []
    for patient_id, truth in patients.items():
        result = aggregate_patient(votes[patient_id], classes)
        patient_rows.append({
            "patient_id": patient_id, "truth": truth, "prediction": result.prediction,
            "input_groups": group_counts[patient_id], "valid_votes": result.valid_votes,
            "tied": result.tied,
        })
    classification = classification_metrics(
        [row["truth"] for row in patient_rows], [row["prediction"] for row in patient_rows], classes,
    )
    detection = {protocol: [] for protocol in PROTOCOLS}
    for (protocol, threshold), total in counts.items():
        detection[protocol].append({
            "iou_threshold": threshold,
            "threshold_rule": "IoU > 0" if threshold == 0 else "IoU >= threshold",
            **total, **binary_detection_metrics(total["tp"], total["fp"], total["fn"]),
        })
    conditional = {protocol: [] for protocol in PROTOCOLS}
    for protocol, rows in detection.items():
        for row in rows:
            total = {name: row[name] for name in ("tp", "fp", "fn", "gt_instances", "predicted_instances")}
            total["fp"] -= excluded_predictions
            total["predicted_instances"] -= excluded_predictions
            conditional[protocol].append({
                "iou_threshold": row["iou_threshold"], "threshold_rule": row["threshold_rule"],
                **total, **binary_detection_metrics(total["tp"], total["fp"], total["fn"]),
            })
    return {
        "detection_scope": "all_input_phase_slices",
        "detection_positive_only": conditional,
        "background_false_positives": {
            "count": excluded_predictions,
            "phase_slices": excluded_phases,
            "per_background_phase_slice": excluded_predictions / excluded_phases if excluded_phases else 0.0,
            "per_patient": excluded_predictions / len(patients),
        },
        "classes": list(classes), "classification": classification, "detection": detection,
        "localization": {
            "mean_iou_class_agnostic": ca_iou_sum / gt_instances if gt_instances else 0.0,
            "gt_instances": gt_instances,
            "acc_at_iou_positive": {
                protocol: positive_matches[protocol] / gt_instances if gt_instances else 0.0
                for protocol in PROTOCOLS
            },
            "positive_phase_slices": positive_phases,
            "background_phase_slices": excluded_phases,
        },
        "coverage": {
            "patients": len(patients), "input_groups": len(expected),
            "prediction_records": len(predictions),
            "schema_valid_groups": sum(row["schema_valid"] for row in groups),
            "diagnostic_eligible_groups": sum(row["prediction"] is not None for row in groups),
            "patients_without_vote": sum(row["prediction"] is None for row in patient_rows),
            "patients_without_input_groups": sum(row["input_groups"] == 0 for row in patient_rows),
        },
        "patients": patient_rows, "input_groups": groups,
    }


def write_evaluation(result: dict, output_dir: str | Path) -> None:
    """Write metrics and keyed patient/input-group predictions as JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    for filename, key in (("patient_predictions.jsonl", "patients"), ("group_predictions.jsonl", "input_groups")):
        with (destination / filename).open("w", encoding="utf-8") as stream:
            for row in result[key]:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--iou-thresholds", required=True, nargs="+", type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from data.manifest import load_manifest
    try:
        result = evaluate(load_manifest(args.manifest), read_predictions(args.predictions),
                          iou_thresholds=args.iou_thresholds)
        write_evaluation(result, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir), "coverage": result["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
