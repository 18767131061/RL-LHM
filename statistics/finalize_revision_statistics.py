#!/usr/bin/env python3
"""Legacy statistics finalizer for the older experiment layout and metric schema.

Requires classification_overall/lesion_detection with nested bootstrap intervals
and helper scripts under the experiment root's scripts/ directory. It cannot
directly consume evaluate_paper_protocol.py outputs. The paired-patient tool
accepts patient_id or logical_key while preserving all patients; it remains a
single-seed statistic. Use evaluation/joint_seed_statistics.py for the current
canonical five-seed exports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SEEDS = (42, 2026, 3407, 1234, 5678)
METHODS = ("sft", "vanilla_grpo", "full_rl_lhm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        print("[command] " + " ".join(command), file=log)
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"[exit] {completed.returncode}", file=log)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def patient_predictions(root: Path, method: str, seed: int) -> Path:
    if method == "sft":
        return root / "eval" / f"seed_{seed}" / "patient_predictions.csv"
    return root / "rl_runs" / f"seed_{seed}" / method / "evaluation" / "patient_predictions.csv"


def metrics_path(root: Path, method: str, seed: int) -> Path:
    if method == "sft":
        return root / "eval" / f"seed_{seed}" / "metrics.json"
    return root / "rl_runs" / f"seed_{seed}" / method / "evaluation" / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy-protocol", action="store_true",
                        help="Explicitly use the older experiment layout and statistical/evaluation definitions")
    parser.add_argument("--root", required=True)
    parser.add_argument("--cohort", choices=("private", "public"), required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--test-manifest", help="Frozen test-patient manifest shared by the paired models")
    parser.add_argument("--aggregation-rule", help="Frozen patient aggregation rule")
    parser.add_argument("--evaluation-version", help="Patient-level evaluation definition/version")
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail unless all R1-1 per-Seed inputs carry a test manifest and evaluation metadata",
    )
    args = parser.parse_args()
    if not args.allow_legacy_protocol:
        parser.error("Legacy interface: pass --allow-legacy-protocol only for the older experiment layout; "
                     "use evaluate_paper_protocol.py and joint_seed_statistics.py for current outputs")
    root = Path(args.root).resolve()
    # Resolving a venv executable's symlink would select the base environment.
    python = str(Path(args.python).expanduser().absolute())
    if args.strict_provenance:
        missing = [
            name
            for name, value in {
                "--test-manifest": args.test_manifest,
                "--aggregation-rule": args.aggregation_rule,
                "--evaluation-version": args.evaluation_version,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Strict provenance requires: {', '.join(missing)}")
    classes = (
        "HCC,ICC,FNH,HA_Hemangioma,nodule_CYST"
        if args.cohort == "private"
        else "HCC,ICC,HA_Hemangioma,BCLM,CRLM"
    )
    output_root = root / "analysis"
    ensemble_root = output_root / "ensembles"
    paired_root = output_root / "paired_tests"
    tables_root = output_root / "tables"
    for directory in (ensemble_root, paired_root, tables_root):
        directory.mkdir(parents=True, exist_ok=True)

    required = [
        path
        for method in METHODS
        for seed in SEEDS
        for path in (patient_predictions(root, method, seed), metrics_path(root, method, seed))
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Formal statistics inputs are incomplete: {missing}")

    ensembles = {}
    for method in METHODS:
        output = ensemble_root / f"{method}_five_seed_ensemble.csv"
        audit = output.with_suffix(output.suffix + ".audit.json")
        if not output.is_file() or not audit.is_file():
            run(
                [
                    python,
                    str(root / "scripts" / "ensemble_patient_predictions.py"),
                    "--inputs",
                    *[str(patient_predictions(root, method, seed)) for seed in SEEDS],
                    "--seeds",
                    *[str(seed) for seed in SEEDS],
                    "--classes",
                    classes,
                    "--method",
                    method,
                    "--output",
                    str(output),
                ],
                output_root / "finalize.log",
            )
        ensembles[method] = output

    ensemble_pairs = (
        ("sft", "vanilla_grpo"),
        ("sft", "full_rl_lhm"),
        ("vanilla_grpo", "full_rl_lhm"),
    )
    paired_outputs: list[Path] = []
    for reference, candidate in ensemble_pairs:
        output = paired_root / f"ensemble_{reference}_vs_{candidate}.json"
        if not output.is_file():
            run(
                [
                    python,
                    str(root / "scripts" / "paired_patient_tests.py"),
                    "--reference",
                    str(ensembles[reference]),
                    "--candidate",
                    str(ensembles[candidate]),
                    "--reference-name",
                    reference,
                    "--candidate-name",
                    candidate,
                    "--classes",
                    classes,
                    "--output",
                    str(output),
                    "--bootstrap",
                    str(args.bootstrap),
                    "--permutations",
                    str(args.permutations),
                    "--seed",
                    "2026",
                ],
                output_root / "finalize.log",
            )
        paired_outputs.append(output)

    for seed in SEEDS:
        for reference, candidate in ensemble_pairs:
            output = paired_root / f"seed_{seed}_{reference}_vs_{candidate}.json"
            if not output.is_file():
                detail = paired_root / f"seed_{seed}_{reference}_vs_{candidate}_paired_detail.csv"
                command = [
                    python,
                    str(root / "scripts" / "paired_patient_tests.py"),
                    "--reference",
                    str(patient_predictions(root, reference, seed)),
                    "--candidate",
                    str(patient_predictions(root, candidate, seed)),
                    "--reference-name",
                    f"{reference}_seed{seed}",
                    "--candidate-name",
                    f"{candidate}_seed{seed}",
                    "--classes",
                    classes,
                    "--output",
                    str(output),
                    "--paired-detail-output",
                    str(detail),
                    "--cohort",
                    args.cohort,
                    "--model-seed",
                    str(seed),
                    "--aggregation-rule",
                    args.aggregation_rule or "UNSPECIFIED",
                    "--evaluation-version",
                    args.evaluation_version or "UNSPECIFIED",
                    "--bootstrap",
                    str(args.bootstrap),
                    "--permutations",
                    str(args.permutations),
                    "--seed",
                    str(seed),
                ]
                if args.test_manifest:
                    command.extend(["--test-manifest", str(Path(args.test_manifest).resolve())])
                if args.strict_provenance:
                    command.append("--strict-provenance")
                run(
                    command,
                    output_root / "finalize.log",
                )
            paired_outputs.append(output)

    core_metrics_table = tables_root / "core_method_seed_metrics_with_ci.csv"
    with core_metrics_table.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["method", "seed", "level", "label", "metric", "estimate", "ci_lower", "ci_upper"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            for seed in SEEDS:
                metrics = json.loads(metrics_path(root, method, seed).read_text(encoding="utf-8"))
                classification = metrics["classification_overall"]
                overall_intervals = classification["bootstrap_95_ci"]
                for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                    writer.writerow(
                        {
                            "method": method,
                            "seed": seed,
                            "level": "overall",
                            "label": "all",
                            "metric": metric,
                            "estimate": classification[metric],
                            "ci_lower": overall_intervals[metric]["lower"],
                            "ci_upper": overall_intervals[metric]["upper"],
                        }
                    )
                for label in classes.split(","):
                    current = classification["per_class"][label]
                    intervals = current["bootstrap_95_ci"]
                    for metric in ("precision", "recall", "f1"):
                        writer.writerow(
                            {
                                "method": method,
                                "seed": seed,
                                "level": "class",
                                "label": label,
                                "metric": metric,
                                "estimate": current[metric],
                                "ci_lower": intervals[metric]["lower"],
                                "ci_upper": intervals[metric]["upper"],
                            }
                        )
                for mode in ("class_aware", "class_agnostic"):
                    current = metrics["lesion_detection"][mode]
                    intervals = current["clustered_patient_bootstrap_95_ci"]
                    for metric in ("precision", "recall", "f1", "acc_iou_gt_0", "acc_iou_ge_0_5", "mean_iou"):
                        writer.writerow(
                            {
                                "method": method,
                                "seed": seed,
                                "level": "lesion",
                                "label": mode,
                                "metric": metric,
                                "estimate": current[metric],
                                "ci_lower": intervals[metric]["lower"],
                                "ci_upper": intervals[metric]["upper"],
                            }
                        )

    paired_table = tables_root / "core_paired_tests.csv"
    with paired_table.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "comparison",
            "patients",
            "reference",
            "candidate",
            "reference_accuracy",
            "candidate_accuracy",
            "accuracy_difference",
            "accuracy_difference_ci_lower",
            "accuracy_difference_ci_upper",
            "mcnemar_exact_p",
            "accuracy_permutation_p",
            "reference_macro_recall",
            "candidate_macro_recall",
            "macro_recall_difference",
            "macro_recall_difference_ci_lower",
            "macro_recall_difference_ci_upper",
            "macro_recall_permutation_p",
            "reference_macro_f1",
            "candidate_macro_f1",
            "macro_f1_difference",
            "macro_f1_difference_ci_lower",
            "macro_f1_difference_ci_upper",
            "macro_f1_permutation_p",
            "bh_family_id",
            "q_accuracy",
            "q_macro_recall",
            "q_macro_f1",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for path in paired_outputs:
            paired = json.loads(path.read_text(encoding="utf-8"))
            writer.writerow(
                {
                    "comparison": path.stem,
                    "patients": paired["patients"],
                    "reference": paired["reference"],
                    "candidate": paired["candidate"],
                    "reference_accuracy": paired["reference_metrics"]["accuracy"],
                    "candidate_accuracy": paired["candidate_metrics"]["accuracy"],
                    "accuracy_difference": paired["candidate_minus_reference"]["accuracy"],
                    "accuracy_difference_ci_lower": paired["paired_bootstrap_95_ci"]["accuracy"]["lower"],
                    "accuracy_difference_ci_upper": paired["paired_bootstrap_95_ci"]["accuracy"]["upper"],
                    "mcnemar_exact_p": paired["mcnemar"]["exact_two_sided_p"],
                    "accuracy_permutation_p": paired["paired_permutation_two_sided_p"]["accuracy"],
                    "reference_macro_recall": paired["reference_metrics"]["macro_recall"],
                    "candidate_macro_recall": paired["candidate_metrics"]["macro_recall"],
                    "macro_recall_difference": paired["candidate_minus_reference"]["macro_recall"],
                    "macro_recall_difference_ci_lower": paired["paired_bootstrap_95_ci"]["macro_recall"]["lower"],
                    "macro_recall_difference_ci_upper": paired["paired_bootstrap_95_ci"]["macro_recall"]["upper"],
                    "macro_recall_permutation_p": paired["paired_permutation_two_sided_p"]["macro_recall"],
                    "reference_macro_f1": paired["reference_metrics"]["macro_f1"],
                    "candidate_macro_f1": paired["candidate_metrics"]["macro_f1"],
                    "macro_f1_difference": paired["candidate_minus_reference"]["macro_f1"],
                    "macro_f1_difference_ci_lower": paired["paired_bootstrap_95_ci"]["macro_f1"]["lower"],
                    "macro_f1_difference_ci_upper": paired["paired_bootstrap_95_ci"]["macro_f1"]["upper"],
                    "macro_f1_permutation_p": paired["paired_permutation_two_sided_p"]["macro_f1"],
                    "bh_family_id": paired.get("bh", {}).get("family_id", ""),
                    "q_accuracy": paired.get("bh", {}).get("q_values", {}).get("accuracy", ""),
                    "q_macro_recall": paired.get("bh", {}).get("q_values", {}).get("macro_recall", ""),
                    "q_macro_f1": paired.get("bh", {}).get("q_values", {}).get("macro_f1", ""),
                }
            )

    center_table = tables_root / "center_stratified_five_seed.csv"
    with center_table.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "method",
            "seed",
            "center",
            "patients",
            "accuracy",
            "accuracy_ci_lower",
            "accuracy_ci_upper",
            "macro_f1",
            "macro_f1_ci_lower",
            "macro_f1_ci_upper",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            for seed in SEEDS:
                metrics = json.loads(metrics_path(root, method, seed).read_text(encoding="utf-8"))
                for center, current in metrics["classification_by_center"].items():
                    intervals = current["bootstrap_95_ci"]
                    writer.writerow(
                        {
                            "method": method,
                            "seed": seed,
                            "center": center,
                            "patients": current["n"],
                            "accuracy": current["accuracy"],
                            "accuracy_ci_lower": intervals["accuracy"]["lower"],
                            "accuracy_ci_upper": intervals["accuracy"]["upper"],
                            "macro_f1": current["macro_f1"],
                            "macro_f1_ci_lower": intervals["macro_f1"]["lower"],
                            "macro_f1_ci_upper": intervals["macro_f1"]["upper"],
                        }
                    )

    artifacts = [
        *ensembles.values(),
        *[path.with_suffix(path.suffix + ".audit.json") for path in ensembles.values()],
        *paired_outputs,
        core_metrics_table,
        paired_table,
        center_table,
    ]
    result = {
        "status": "formal_statistics_complete",
        "cohort": args.cohort,
        "scope": "core_sft_vanilla_grpo_full_rl_lhm_only",
        "classes": classes.split(","),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifacts
        ],
    }
    marker = output_root / "formal_statistics_complete.json"
    atomic_write_json(marker, result)
    print(json.dumps({"status": result["status"], "marker": str(marker)}, indent=2))


if __name__ == "__main__":
    main()
