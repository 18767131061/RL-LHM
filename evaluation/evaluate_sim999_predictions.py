#!/usr/bin/env python3
"""Legacy evaluation: GT-positive voting and label-dependent matching.

Use evaluate_paper_protocol.py for the phase-specific paper protocol.
This entry point requires --allow-legacy-protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

try:
    from rewards.verl_liver_reward import parse_completion_from_text
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rewards.verl_liver_reward import parse_completion_from_text


CLASSES = ["HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-legacy-protocol", action="store_true",
                        help="Enable GT-positive voting and legacy label-dependent matching")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--expected-patients", type=int, default=198)
    parser.add_argument("--dataset-label", default="sim999")
    parser.add_argument(
        "--classes",
        default=",".join(CLASSES),
        help="Comma-separated ordered class labels used for aggregation and metrics.",
    )
    parser.add_argument(
        "--cluster-field",
        choices=("logical_key", "statistical_patient_id"),
        default="logical_key",
    )
    args = parser.parse_args()
    if not args.allow_legacy_protocol:
        parser.error("Legacy protocol disabled. Use evaluation/evaluate_paper_protocol.py, or explicitly pass --allow-legacy-protocol to use this GT-positive voting protocol.")
    print("WARNING: LEGACY PROTOCOL: GT-positive-only diagnostic votes, shared metadata boxes, and legacy label-dependent matching. These results are not interchangeable with evaluate_paper_protocol.py.", file=sys.stderr)
    return args


def canon_label(value: object) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "")
    if "hcc" in text or "肝细胞" in text:
        return "HCC"
    if "icc" in text or "胆管" in text:
        return "ICC"
    if "fnh" in text or "局灶性结节" in text:
        return "FNH"
    if "hemangioma" in text or "血管瘤" in text or text in {"hh", "ha"}:
        return "HA_Hemangioma"
    if "nodule" in text or "cyst" in text or "囊肿" in text:
        return "nodule_CYST"
    if "bclm" in text or "breast" in text or "乳腺" in text:
        return "BCLM"
    if "crlm" in text or "colorectal" in text or "结直肠" in text:
        return "CRLM"
    return None


def parse_detections(raw: str) -> list[dict]:
    parsed = parse_completion_from_text(str(raw or ""))
    if parsed is None or any(item["label"] not in CLASSES for item in parsed["lesions"]):
        return []
    return [
        {"phase": item["phase"], "label": item["label"], "box": item["bbox"]}
        for item in parsed["lesions"]
    ]


def iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    ground_truth: list[list[float]],
    predictions: list[dict],
    true_label: str,
    threshold: float,
    class_aware: bool,
) -> dict[str, object]:
    # NOTE: this utility's ``class_aware`` mode zeroes the IoU of label-mismatched
    # pairs against the patient-level truth label before assignment. The canonical
    # paper protocols (class-agnostic / same-class-priority / geometry-first
    # class-aware) are implemented in matching/protocols.py; use that module for
    # any number reported in the manuscript.
    if not ground_truth:
        return {"tp": 0, "fp": len(predictions), "fn": 0, "gt_ious": []}
    if not predictions:
        return {"tp": 0, "fp": 0, "fn": len(ground_truth), "gt_ious": [0.0] * len(ground_truth)}
    matrix = np.asarray(
        [
            [
                iou(gt, pred["box"])
                if not class_aware or pred["label"] == true_label
                else 0.0
                for pred in predictions
            ]
            for gt in ground_truth
        ],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(-matrix)
    assigned = {int(row): float(matrix[row, column]) for row, column in zip(rows, columns)}
    matched = sum(value >= threshold for value in assigned.values())
    return {
        "tp": matched,
        "fp": len(predictions) - matched,
        "fn": len(ground_truth) - matched,
        "gt_ious": [assigned.get(index, 0.0) for index in range(len(ground_truth))],
    }


def vote(labels: list[str | None], nlls: list[float | None]) -> tuple[str | None, bool]:
    valid = [(label, nll) for label, nll in zip(labels, nlls) if label in CLASSES]
    if not valid:
        return None, False
    counts = Counter(label for label, _ in valid)
    best = max(counts.values())
    tied = [label for label in CLASSES if counts[label] == best]
    if len(tied) == 1:
        return tied[0], False
    mean_nll = {}
    for label in tied:
        scores = [score for current, score in valid if current == label and score is not None and math.isfinite(score)]
        mean_nll[label] = float(np.mean(scores)) if scores else float("inf")
    return min(tied, key=lambda label: (mean_nll[label], CLASSES.index(label))), True


def aggregate_inputs(items: list[dict], mode: str) -> tuple[str | None, bool]:
    valid = [item for item in items if item["prediction"] in CLASSES]
    if not valid:
        return None, False
    labels = [item["prediction"] for item in valid]
    nlls = [item["nll"] for item in valid]
    if mode == "majority_likelihood_tiebreak":
        return vote(labels, nlls)
    if mode == "majority_fixed_tiebreak":
        counts = Counter(labels)
        best = max(counts.values())
        tied = [label for label in CLASSES if counts[label] == best]
        return tied[0], len(tied) > 1
    if mode == "likelihood_weighted_vote":
        finite = [score for score in nlls if score is not None and math.isfinite(score)]
        fallback = float(np.median(finite)) if finite else 0.0
        totals = {label: 0.0 for label in CLASSES}
        for label, score in zip(labels, nlls):
            current = score if score is not None and math.isfinite(score) else fallback
            totals[label] += math.exp(-max(-50.0, min(50.0, current)))
        best = max(totals.values())
        tied = [label for label in CLASSES if abs(totals[label] - best) <= 1e-12]
        return tied[0], len(tied) > 1
    if mode == "best_likelihood_input":
        ranked = [
            (
                item["nll"]
                if item["nll"] is not None and math.isfinite(item["nll"])
                else float("inf"),
                index,
                item["prediction"],
            )
            for index, item in enumerate(valid)
        ]
        return min(ranked)[2], False
    raise ValueError(f"Unknown aggregation mode: {mode}")


def classification_metrics(patients: list[dict]) -> dict:
    confusion = {truth: {prediction: 0 for prediction in CLASSES + ["UNDEFINED"]} for truth in CLASSES}
    for item in patients:
        confusion[item["truth"]][item["prediction"] or "UNDEFINED"] += 1
    correct = sum(item["truth"] == item["prediction"] for item in patients)
    per_class = {}
    for current in CLASSES:
        tp = sum(item["truth"] == current and item["prediction"] == current for item in patients)
        fp = sum(item["truth"] != current and item["prediction"] == current for item in patients)
        fn = sum(item["truth"] == current and item["prediction"] != current for item in patients)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[current] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return {
        "n": len(patients),
        "accuracy": correct / len(patients) if patients else 0.0,
        "macro_precision": float(np.mean([per_class[label]["precision"] for label in CLASSES])),
        "macro_recall": float(np.mean([per_class[label]["recall"] for label in CLASSES])),
        "macro_f1": float(np.mean([per_class[label]["f1"] for label in CLASSES])),
        "undefined": sum(item["prediction"] is None for item in patients),
        "per_class": per_class,
        "confusion": confusion,
    }


def bootstrap_classification(patients: list[dict], repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    class_values = {
        label: {name: [] for name in ("precision", "recall", "f1")}
        for label in CLASSES
    }
    for _ in range(repeats):
        sample = [patients[index] for index in rng.integers(0, len(patients), len(patients))]
        metrics = classification_metrics(sample)
        for name in values:
            values[name].append(metrics[name])
        for label in CLASSES:
            for name in class_values[label]:
                class_values[label][name].append(metrics["per_class"][label][name])

    def intervals(series: dict[str, list[float]]) -> dict:
        return {
            name: {
                "lower": float(np.percentile(items, 2.5)),
                "upper": float(np.percentile(items, 97.5)),
            }
            for name, items in series.items()
        }

    return {
        "overall": intervals(values),
        "per_class": {label: intervals(class_values[label]) for label in CLASSES},
    }


def binomial_interval(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def detection_summary(counts: dict[str, object]) -> dict:
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ious = list(counts["gt_ious"])
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc_iou_gt_0": sum(value > 0 for value in ious) / len(ious) if ious else 0.0,
        "acc_iou_ge_0_5": sum(value >= 0.5 for value in ious) / len(ious) if ious else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "ground_truth_lesions": len(ious),
    }


def bootstrap_detection(
    per_patient: dict[str, dict[str, dict[str, object]]],
    repeats: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    patient_ids = sorted(per_patient)
    if not patient_ids:
        raise RuntimeError("Cannot bootstrap detection without patients")
    metric_names = ("precision", "recall", "f1", "acc_iou_gt_0", "acc_iou_ge_0_5", "mean_iou")
    values = {
        mode: {name: [] for name in metric_names}
        for mode in ("class_agnostic", "class_aware")
    }
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        sampled = rng.integers(0, len(patient_ids), len(patient_ids))
        for mode in values:
            combined = {"tp": 0, "fp": 0, "fn": 0, "gt_ious": []}
            for index in sampled:
                current = per_patient[patient_ids[int(index)]][mode]
                for name in ("tp", "fp", "fn"):
                    combined[name] += int(current[name])
                combined["gt_ious"].extend(current["gt_ious"])
            summary = detection_summary(combined)
            for name in metric_names:
                values[mode][name].append(summary[name])
    return {
        mode: {
            name: {
                "lower": float(np.percentile(items, 2.5)),
                "upper": float(np.percentile(items, 97.5)),
            }
            for name, items in current.items()
        }
        for mode, current in values.items()
    }


def main() -> None:
    global CLASSES
    args = parse_args()
    requested_classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    if len(requested_classes) < 2 or len(requested_classes) != len(set(requested_classes)):
        raise RuntimeError(f"Invalid --classes value: {args.classes}")
    CLASSES = requested_classes
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.metadata).open(newline="", encoding="utf-8") as stream:
        metadata_rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    metadata_by_key = {}
    metadata_by_image = {}
    for row in metadata_rows:
        key = (row["logical_key"], int(row["slice_index"]))
        if key in metadata_by_key:
            raise RuntimeError(f"Duplicate test metadata key: {key}")
        metadata_by_key[key] = row
        first_image = json.loads(row["images"])[0]
        image_name = os.path.basename(first_image)
        if image_name in metadata_by_image:
            raise RuntimeError(f"Duplicate first-image basename in metadata: {image_name}")
        metadata_by_image[image_name] = row

    with Path(args.predictions).open(newline="", encoding="utf-8-sig") as stream:
        prediction_rows = list(csv.DictReader(stream))
    if len(prediction_rows) != len(metadata_rows):
        raise RuntimeError(f"Prediction/metadata rows differ: {len(prediction_rows)} vs {len(metadata_rows)}")

    patient_inputs = defaultdict(list)
    aggregate_detection = {
        "class_agnostic": {"tp": 0, "fp": 0, "fn": 0, "gt_ious": []},
        "class_aware": {"tp": 0, "fp": 0, "fn": 0, "gt_ious": []},
    }
    per_patient_detection = defaultdict(
        lambda: {
            "class_agnostic": {"tp": 0, "fp": 0, "fn": 0, "gt_ious": []},
            "class_aware": {"tp": 0, "fp": 0, "fn": 0, "gt_ious": []},
        }
    )
    joined_rows = []
    matched_metadata_keys = set()
    excluded_duplicate_prediction_rows = 0
    for prediction in prediction_rows:
        prediction_key = None
        if prediction.get("logical_key") and prediction.get("slice_index") not in (None, ""):
            prediction_key = (prediction["logical_key"], int(prediction["slice_index"]))
        if prediction_key is not None:
            metadata = metadata_by_key.get(prediction_key)
            lookup_description = str(prediction_key)
        else:
            image_name = os.path.basename(prediction.get("a", ""))
            metadata = metadata_by_image.get(image_name)
            lookup_description = image_name
        if metadata is None:
            raise RuntimeError(f"Prediction not found in metadata: {lookup_description}")
        metadata_key = (metadata["logical_key"], int(metadata["slice_index"]))
        if metadata_key in matched_metadata_keys:
            raise RuntimeError(f"Duplicate prediction for metadata sample: {metadata_key}")
        matched_metadata_keys.add(metadata_key)
        if args.cluster_field == "statistical_patient_id" and metadata.get("is_duplicate") == "1":
            excluded_duplicate_prediction_rows += 1
            continue
        evaluation_unit_id = metadata.get(args.cluster_field) or metadata["logical_key"]
        detections = parse_detections(prediction.get("raw", ""))
        boxes = json.loads(metadata["bbox_2d"]) if metadata["bbox_2d"] else []
        labels = [item["label"] for item in detections if item["label"] in CLASSES]
        input_prediction, input_tie = vote(labels, [None] * len(labels))
        try:
            nll = float(prediction["avg_nll"]) if prediction.get("avg_nll") else None
        except Exception:
            nll = None
        item = {
            "logical_key": metadata["logical_key"],
            "evaluation_unit_id": evaluation_unit_id,
            "truth": CLASS_MAP.get(metadata["class"], metadata["model_label"]),
            "prediction": input_prediction,
            "nll": nll,
            "center": metadata["analysis_center"],
            "input_tie": input_tie,
        }
        if int(metadata["positive"]) == 1:
            patient_inputs[evaluation_unit_id].append(item)
        for phase in ("a", "p", "v"):
            phase_predictions = [current for current in detections if current["phase"] == phase]
            for mode, class_aware in (("class_agnostic", False), ("class_aware", True)):
                matched = match_boxes(boxes, phase_predictions, item["truth"], args.iou_threshold, class_aware)
                for key in ("tp", "fp", "fn"):
                    aggregate_detection[mode][key] += matched[key]
                    per_patient_detection[evaluation_unit_id][mode][key] += matched[key]
                aggregate_detection[mode]["gt_ious"].extend(matched["gt_ious"])
                per_patient_detection[evaluation_unit_id][mode]["gt_ious"].extend(matched["gt_ious"])
        joined_rows.append({**metadata, "input_prediction": input_prediction or "", "input_tie": int(input_tie), "avg_nll": "" if nll is None else nll, "raw": prediction.get("raw", "")})

    missing_metadata = set(metadata_by_key) - matched_metadata_keys
    if missing_metadata:
        raise RuntimeError(f"Missing predictions for metadata samples: {sorted(missing_metadata)[:5]}")

    aggregation_modes = (
        "majority_likelihood_tiebreak",
        "majority_fixed_tiebreak",
        "likelihood_weighted_vote",
        "best_likelihood_input",
    )
    aggregation_patients = {mode: [] for mode in aggregation_modes}
    for evaluation_unit_id, items in patient_inputs.items():
        for mode in aggregation_modes:
            prediction, tied = aggregate_inputs(items, mode)
            aggregation_patients[mode].append(
                {
                    "logical_key": evaluation_unit_id,
                    "truth": items[0]["truth"],
                    "prediction": prediction,
                    "center": items[0]["center"],
                    "valid_votes": sum(item["prediction"] is not None for item in items),
                    "input_groups": len(items),
                    "patient_tie": tied,
                }
            )
    patients = aggregation_patients["majority_likelihood_tiebreak"]
    patients.sort(key=lambda item: item["logical_key"])
    if len(patients) != args.expected_patients:
        raise RuntimeError(
            f"Expected {args.expected_patients} logical test patients, found {len(patients)}"
        )

    overall = classification_metrics(patients)
    classification_bootstrap = bootstrap_classification(patients, args.bootstrap, args.seed)
    overall["bootstrap_95_ci"] = classification_bootstrap["overall"]
    for label in CLASSES:
        current = overall["per_class"][label]
        current["bootstrap_95_ci"] = classification_bootstrap["per_class"][label]
        current["precision_exact_95_ci"] = binomial_interval(current["tp"], current["tp"] + current["fp"])
        current["recall_exact_95_ci"] = binomial_interval(current["tp"], current["tp"] + current["fn"])
    by_center = {}
    centers = sorted({item["center"] for item in patients})
    for center_index, center in enumerate(centers):
        center_patients = [item for item in patients if item["center"] == center]
        current = classification_metrics(center_patients)
        center_bootstrap = bootstrap_classification(
            center_patients,
            args.bootstrap,
            args.seed + 100 + center_index,
        )
        current["bootstrap_95_ci"] = center_bootstrap["overall"]
        for label in CLASSES:
            current["per_class"][label]["bootstrap_95_ci"] = center_bootstrap["per_class"][label]
        by_center[center] = current
    aggregation_sensitivity = {}
    for mode, mode_patients in aggregation_patients.items():
        current = classification_metrics(mode_patients)
        aggregation_sensitivity[mode] = {
            "accuracy": current["accuracy"],
            "macro_precision": current["macro_precision"],
            "macro_recall": current["macro_recall"],
            "macro_f1": current["macro_f1"],
            "undefined": current["undefined"],
            "ties": sum(item["patient_tie"] for item in mode_patients),
        }
    detection = {mode: detection_summary(counts) for mode, counts in aggregate_detection.items()}
    detection_bootstrap = bootstrap_detection(per_patient_detection, args.bootstrap, args.seed + 1)
    for mode in detection:
        detection[mode]["clustered_patient_bootstrap_95_ci"] = detection_bootstrap[mode]
    active_metadata_rows = [
        row
        for row in metadata_rows
        if not (args.cluster_field == "statistical_patient_id" and row.get("is_duplicate") == "1")
    ]
    multi_slice_histogram = Counter(
        int(row["lesion_count"])
        for row in active_metadata_rows
        if int(row["positive"]) == 1
    )

    result = {
        "predictions": str(Path(args.predictions).resolve()),
        "metadata": str(Path(args.metadata).resolve()),
        "bootstrap_repeats": args.bootstrap,
        "bootstrap_seed": args.seed,
        "cluster_field": args.cluster_field,
        "test_metadata_rows": len(metadata_rows),
        "active_test_metadata_rows": len(active_metadata_rows),
        "excluded_duplicate_prediction_rows": excluded_duplicate_prediction_rows,
        "iou_threshold": args.iou_threshold,
        "classification_overall": overall,
        "classification_by_center": by_center,
        "aggregation_sensitivity": aggregation_sensitivity,
        "lesion_detection": detection,
        "positive_slice_lesion_count_histogram": dict(sorted(multi_slice_histogram.items())),
        "multi_lesion_positive_slices": sum(count for lesions, count in multi_slice_histogram.items() if lesions > 1),
        "patient_ties": sum(item["patient_tie"] for item in patients),
        "patients_without_valid_vote": sum(item["prediction"] is None for item in patients),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    patient_path = output_dir / "patient_predictions.csv"
    with patient_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(patients[0]))
        writer.writeheader()
        writer.writerows(patients)
    joined_path = output_dir / "input_predictions_joined.csv"
    with joined_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined_rows[0]))
        writer.writeheader()
        writer.writerows(joined_rows)

    lines = [
        f"# Qwen3-VL-4B {args.dataset_label} evaluation",
        "",
        f"- Patients: {overall['n']}",
        f"- Accuracy: {overall['accuracy']:.4f} ({overall['bootstrap_95_ci']['accuracy']['lower']:.4f}-{overall['bootstrap_95_ci']['accuracy']['upper']:.4f})",
        f"- Macro-F1: {overall['macro_f1']:.4f} ({overall['bootstrap_95_ci']['macro_f1']['lower']:.4f}-{overall['bootstrap_95_ci']['macro_f1']['upper']:.4f})",
        f"- Class-aware lesion precision/recall/F1: {detection['class_aware']['precision']:.4f}/{detection['class_aware']['recall']:.4f}/{detection['class_aware']['f1']:.4f}",
        f"- Class-agnostic lesion precision/recall/F1: {detection['class_agnostic']['precision']:.4f}/{detection['class_agnostic']['recall']:.4f}/{detection['class_agnostic']['f1']:.4f}",
        f"- Multi-lesion positive slices: {result['multi_lesion_positive_slices']}",
        f"- Patient-level ties: {result['patient_ties']}",
        "",
        "| Class | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for label in CLASSES:
        current = overall["per_class"][label]
        lines.append(f"| {label} | {current['precision']:.4f} | {current['recall']:.4f} | {current['f1']:.4f} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


CLASS_MAP = {
    "HCC": "HCC",
    "ICC": "ICC",
    "FNH": "FNH",
    "HH": "HA_Hemangioma",
    "NC": "nodule_CYST",
    "HA_Hemangioma": "HA_Hemangioma",
    "BCLM": "BCLM",
    "CRLM": "CRLM",
}


if __name__ == "__main__":
    main()
