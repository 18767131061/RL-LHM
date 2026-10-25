#!/usr/bin/env python3
"""Legacy single-seed patient-paired tests for two prediction tables.

Use joint_seed_statistics.py for the fixed-five-seed manuscript protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


CLASSES = ["HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = row.get("logical_key") or row.get("patient_id")
        if not key or (row.get("logical_key") and row.get("patient_id") and row["logical_key"] != row["patient_id"]):
            raise RuntimeError(f"Missing or conflicting patient identifier in {path}")
        if key in result:
            raise RuntimeError(f"Duplicate patient key in {path}: {key}")
        result[key] = row
    return result


def metrics(truth: list[str], prediction: list[str]) -> dict[str, float]:
    accuracy = float(np.mean([a == b for a, b in zip(truth, prediction)]))
    recall_values = []
    f1_values = []
    for label in CLASSES:
        tp = sum(a == label and b == label for a, b in zip(truth, prediction))
        fp = sum(a != label and b == label for a, b in zip(truth, prediction))
        fn = sum(a == label and b != label for a, b in zip(truth, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recall_values.append(recall)
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "accuracy": accuracy,
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
    }


def interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper)}


def main() -> None:
    global CLASSES
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--classes", default=",".join(CLASSES))
    parser.add_argument("--cohort", help="Frozen test cohort identifier")
    parser.add_argument("--model-seed", type=int, help="Training/evaluation Seed shared by the paired runs")
    parser.add_argument("--aggregation-rule", help="Frozen patient aggregation rule")
    parser.add_argument("--evaluation-version", help="Evaluation definition/version identifier")
    parser.add_argument("--test-manifest", help="Frozen test-patient manifest used by both models")
    parser.add_argument("--paired-detail-output", help="CSV destination for the auditable paired patient rows")
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Require cohort, model Seed, aggregation/evaluation versions, and a test manifest",
    )
    args = parser.parse_args()
    requested_classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    if len(requested_classes) < 2 or len(requested_classes) != len(set(requested_classes)):
        raise RuntimeError(f"Invalid --classes value: {args.classes}")
    if args.strict_provenance:
        missing = [
            name
            for name, value in {
                "--cohort": args.cohort,
                "--model-seed": args.model_seed,
                "--aggregation-rule": args.aggregation_rule,
                "--evaluation-version": args.evaluation_version,
                "--test-manifest": args.test_manifest,
                "--paired-detail-output": args.paired_detail_output,
            }.items()
            if value is None
        ]
        if missing:
            raise RuntimeError(f"Strict provenance requires: {', '.join(missing)}")
    CLASSES = requested_classes

    reference_path = Path(args.reference).resolve()
    candidate_path = Path(args.candidate).resolve()
    manifest_path = Path(args.test_manifest).resolve() if args.test_manifest else None
    if manifest_path is not None and not manifest_path.is_file():
        raise RuntimeError(f"Test manifest does not exist: {manifest_path}")
    reference = read_predictions(reference_path)
    candidate = read_predictions(candidate_path)
    if set(reference) != set(candidate):
        missing_reference = sorted(set(candidate) - set(reference))
        missing_candidate = sorted(set(reference) - set(candidate))
        raise RuntimeError(
            f"Patient sets differ; missing_reference={missing_reference[:10]} "
            f"missing_candidate={missing_candidate[:10]}"
        )

    keys = sorted(reference)
    truth = [reference[key]["truth"] for key in keys]
    candidate_truth = [candidate[key]["truth"] for key in keys]
    if truth != candidate_truth:
        raise RuntimeError("Ground-truth labels differ between paired prediction files")
    reference_prediction = [reference[key]["prediction"] for key in keys]
    candidate_prediction = [candidate[key]["prediction"] for key in keys]
    reference_correct = np.asarray(
        [a == b for a, b in zip(truth, reference_prediction)], dtype=np.int8
    )
    candidate_correct = np.asarray(
        [a == b for a, b in zip(truth, candidate_prediction)], dtype=np.int8
    )
    reference_only = int(np.sum((reference_correct == 1) & (candidate_correct == 0)))
    candidate_only = int(np.sum((reference_correct == 0) & (candidate_correct == 1)))
    discordant = reference_only + candidate_only
    mcnemar_p = (
        float(binomtest(candidate_only, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )

    reference_metrics = metrics(truth, reference_prediction)
    candidate_metrics = metrics(truth, candidate_prediction)
    observed_difference = {
        name: candidate_metrics[name] - reference_metrics[name]
        for name in ("accuracy", "macro_recall", "macro_f1")
    }

    rng = np.random.default_rng(args.seed)
    n = len(keys)
    bootstrap_values = {"accuracy": [], "macro_recall": [], "macro_f1": []}
    for _ in range(args.bootstrap):
        indices = rng.integers(0, n, n)
        current_truth = [truth[index] for index in indices]
        current_reference = [reference_prediction[index] for index in indices]
        current_candidate = [candidate_prediction[index] for index in indices]
        reference_sample = metrics(current_truth, current_reference)
        candidate_sample = metrics(current_truth, current_candidate)
        for name in bootstrap_values:
            bootstrap_values[name].append(candidate_sample[name] - reference_sample[name])

    permutation_extreme = {"accuracy": 0, "macro_recall": 0, "macro_f1": 0}
    for _ in range(args.permutations):
        swap = rng.integers(0, 2, n, dtype=np.int8).astype(bool)
        perm_reference = [
            candidate_prediction[index] if swap[index] else reference_prediction[index]
            for index in range(n)
        ]
        perm_candidate = [
            reference_prediction[index] if swap[index] else candidate_prediction[index]
            for index in range(n)
        ]
        reference_sample = metrics(truth, perm_reference)
        candidate_sample = metrics(truth, perm_candidate)
        for name in permutation_extreme:
            difference = candidate_sample[name] - reference_sample[name]
            if abs(difference) >= abs(observed_difference[name]) - 1e-12:
                permutation_extreme[name] += 1

    provenance = {
        "cohort": args.cohort,
        "model_seed": args.model_seed,
        "class_space": CLASSES,
        "aggregation_rule": args.aggregation_rule,
        "evaluation_version": args.evaluation_version,
        "reference_prediction": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        },
        "candidate_prediction": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "test_manifest": (
            {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
            if manifest_path is not None
            else None
        ),
    }
    result = {
        "analysis": "legacy_single_seed_patient_paired",
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "patients": n,
        "classes": CLASSES,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_reference": observed_difference,
        "paired_bootstrap_repeats": args.bootstrap,
        "paired_bootstrap_95_ci": {
            name: interval(np.asarray(values, dtype=np.float64))
            for name, values in bootstrap_values.items()
        },
        "mcnemar": {
            "reference_correct_candidate_wrong": reference_only,
            "reference_wrong_candidate_correct": candidate_only,
            "discordant_pairs": discordant,
            "exact_two_sided_p": mcnemar_p,
        },
        "paired_permutation_repeats": args.permutations,
        "paired_permutation_two_sided_p": {
            name: (count + 1) / (args.permutations + 1)
            for name, count in permutation_extreme.items()
        },
        "seed": args.seed,
        "provenance": provenance,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    if args.paired_detail_output:
        detail = Path(args.paired_detail_output).resolve()
        detail.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "logical_key",
            "truth",
            "reference_model",
            "reference_prediction",
            "reference_correct",
            "candidate_model",
            "candidate_prediction",
            "candidate_correct",
            "paired_accuracy_state",
            "cohort",
            "model_seed",
            "class_space",
            "aggregation_rule",
            "evaluation_version",
            "reference_prediction_sha256",
            "candidate_prediction_sha256",
            "test_manifest_sha256",
        ]
        with detail.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key in keys:
                reference_ok = int(reference[key]["truth"] == reference[key]["prediction"])
                candidate_ok = int(candidate[key]["truth"] == candidate[key]["prediction"])
                state = (
                    "both_correct"
                    if reference_ok and candidate_ok
                    else "both_wrong"
                    if not reference_ok and not candidate_ok
                    else "b_reference_correct_candidate_wrong"
                    if reference_ok
                    else "c_reference_wrong_candidate_correct"
                )
                writer.writerow(
                    {
                        "logical_key": key,
                        "truth": reference[key]["truth"],
                        "reference_model": args.reference_name,
                        "reference_prediction": reference[key]["prediction"],
                        "reference_correct": reference_ok,
                        "candidate_model": args.candidate_name,
                        "candidate_prediction": candidate[key]["prediction"],
                        "candidate_correct": candidate_ok,
                        "paired_accuracy_state": state,
                        "cohort": args.cohort,
                        "model_seed": args.model_seed,
                        "class_space": "|".join(CLASSES),
                        "aggregation_rule": args.aggregation_rule,
                        "evaluation_version": args.evaluation_version,
                        "reference_prediction_sha256": provenance["reference_prediction"]["sha256"],
                        "candidate_prediction_sha256": provenance["candidate_prediction"]["sha256"],
                        "test_manifest_sha256": provenance["test_manifest"]["sha256"]
                        if provenance["test_manifest"]
                        else None,
                    }
                )
        result["paired_detail"] = {"path": str(detail), "sha256": sha256_file(detail)}
        with output.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
