#!/usr/bin/env python3
"""Compare three patient readouts of the same saved group predictions per run.

Each cohort specification names one manifest, five fixed seeds, the reference
model, and one prediction JSONL for each model/seed. Every inferential comparison
uses the reference model's default readout; changed-patient counts use each
model's own default readout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.patient import MAJORITY_NLL_TIEBREAK
from evaluation.evaluate_paper_protocol import (
    AGGREGATION_RULES, _identifier, _read_json, evaluate, write_evaluation,
)
from evaluation.joint_seed_statistics import (
    analyze, extract_run, load_study, require, validate_aggregation_pair,
)


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _summary(values: list[float | None]) -> dict:
    # Undefined selective accuracy is retained, never silently replaced by zero
    # or summarized over a different number of training seeds.
    return {"values": values, "defined_seed_count": sum(value is not None for value in values),
            "mean": float(np.mean(values)) if None not in values else None,
            "sample_sd": float(np.std(values, ddof=1)) if None not in values else None}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def compare(spec_path: Path, output_dir: Path, *, permutations: int = 100000,
            bootstrap: int = 10000, random_seed: int = 2026,
            exact: bool = False) -> dict:
    """Generate canonical exports and the full default-reference comparison."""
    spec_path, output_dir = spec_path.resolve(), output_dir.resolve()
    require(not output_dir.exists(), "Output directory already exists; choose a fresh destination")
    require(type(permutations) is int and permutations > 0 and type(bootstrap) is int and bootstrap > 0,
            "Positive permutation/bootstrap repeats required")
    spec = _read_json(spec_path.read_text(encoding="utf-8"))
    require(isinstance(spec, dict) and spec.get("schema_version") == 1, "Spec requires schema_version=1")
    cohort = _identifier(spec.get("cohort"), "cohort")
    seeds = spec.get("seeds")
    require(isinstance(seeds, list) and len(seeds) == 5 and all(type(seed) is int for seed in seeds)
            and len(set(seeds)) == 5, "Exactly five distinct integer seeds are required")
    models = spec.get("models")
    require(isinstance(models, dict) and bool(models), "At least one named model is required")
    for name in models:
        _identifier(name, "model_name")
    reference = spec.get("reference_model")
    require(reference in models, "reference_model must name an included model")
    require(isinstance(spec.get("manifest"), str) and spec["manifest"], "Missing manifest path")
    manifest_path = (spec_path.parent / spec["manifest"]).resolve()
    manifest = _read_json(manifest_path.read_text(encoding="utf-8"))
    manifest_source = _source(manifest_path)
    thresholds = spec.get("iou_thresholds", [0.5])
    require(isinstance(thresholds, list), "iou_thresholds must be a list")

    # All inputs and identity invariants are checked before creating outputs.
    pending, arms, model_specs = [], [], {}
    used_inputs = set()
    for model_index, (model, runs) in enumerate(models.items()):
        require(isinstance(runs, list) and len(runs) == 5, f"Model {model} needs exactly five prediction runs")
        by_seed = {}
        for run in runs:
            require(isinstance(run, dict) and type(run.get("seed")) is int, f"Invalid seed for {model}")
            seed = run["seed"]
            require(seed in seeds and seed not in by_seed, f"Duplicate or mismatched seed for {model}: {seed}")
            require(isinstance(run.get("predictions"), str) and run["predictions"], "Missing predictions path")
            path = (spec_path.parent / run["predictions"]).resolve()
            require(path not in used_inputs, "A prediction file cannot represent more than one model/seed run")
            used_inputs.add(path)
            by_seed[seed] = path
        results = {}
        for seed in seeds:
            source = _source(by_seed[seed])
            for strategy in AGGREGATION_RULES:
                result = evaluate(manifest, by_seed[seed], thresholds, strategy)
                require(result["sources"]["predictions"] == source,
                        "Prediction source changed while evaluating strategies")
                result["run_identity"] = {"model_name": model, "seed": seed}
                result["sources"]["manifest"] = manifest_source
                results[strategy, seed] = result
            anchor = results[MAJORITY_NLL_TIEBREAK, seed]
            for strategy in AGGREGATION_RULES:
                validate_aggregation_pair(anchor, results[strategy, seed])
        for strategy in AGGREGATION_RULES:
            arm_name = f"{model}::{strategy}"
            require(arm_name not in model_specs, "Duplicate aggregation arm name")
            arm = {"arm": arm_name, "model_name": model, "aggregation_strategy": strategy, "runs": []}
            summaries = {"changed_patients": [], "coverage": [], "selective_accuracy": []}
            model_specs[arm_name] = []
            for seed in seeds:
                result, anchor = results[strategy, seed], results[MAJORITY_NLL_TIEBREAK, seed]
                patients = result["patients"]
                defaults = {row["patient_id"]: row["prediction"] for row in anchor["patients"]}
                changed = sum(row["prediction"] != defaults[row["patient_id"]] for row in patients)
                covered = [row for row in patients if row["prediction"] is not None]
                coverage = len(covered) / len(patients)
                selective = sum(row["truth"] == row["prediction"] for row in covered) / len(covered) if covered else None
                relative = Path("runs") / f"model-{model_index}" / f"seed-{seed}" / strategy / "metrics.json"
                arm["runs"].append({"seed": seed, "metrics": relative.as_posix(),
                                    "changed_patients": changed, "coverage": coverage, "selective_accuracy": selective})
                model_specs[arm_name].append({"seed": seed, "metrics": relative.as_posix()})
                for key, value in (("changed_patients", changed), ("coverage", coverage), ("selective_accuracy", selective)):
                    summaries[key].append(value)
                layout, _, units, _ = extract_run(result)
                for metric, value in zip(layout.names, layout.metrics(units.sum(axis=0))):
                    summaries.setdefault(metric, []).append(float(value))
                pending.append((relative.parent, result))
            arm["summary"] = {key: _summary(values) for key, values in summaries.items()}
            arms.append(arm)

    require(not exact or len(manifest["patients"]) <= 16, "Exact enumeration is limited to 16 patients")
    reference_arm = f"{reference}::{MAJORITY_NLL_TIEBREAK}"
    joint_spec = {"schema_version": 1, "seeds": seeds, "reference": reference_arm,
                  "comparison_mode": "aggregation_table" if len(models) > 1 else "aggregation",
                  "models": model_specs,
                  "arms": {arm["arm"]: {key: arm[key] for key in ("model_name", "aggregation_strategy")} for arm in arms}}
    output_dir.mkdir(parents=True)
    for relative, result in pending:
        write_evaluation(result, output_dir / relative)
    joint_spec_path = output_dir / "joint_seed_spec.json"
    _write_json(joint_spec_path, joint_spec)
    study = load_study(joint_spec_path)
    statistics = analyze(study, permutations=permutations, bootstrap=bootstrap,
                         random_seed=random_seed, exact=exact, adjustment="none")
    statistics["spec"] = _source(joint_spec_path)
    _write_json(output_dir / "joint_seed_statistics.json", statistics)
    fields = ["reference", "comparator", "metric", *next(iter(statistics["comparisons"][0]["metrics"].values()))]
    with (output_dir / "joint_seed_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in statistics["comparisons"]:
            for metric, values in pair["metrics"].items():
                writer.writerow({"reference": pair["reference"], "comparator": pair["comparator"], "metric": metric,
                                 **{key: json.dumps(value) if isinstance(value, list) else value for key, value in values.items()}})
    report = {"schema_version": 1, "analysis": "aggregation_strategy_comparison", "cohort": cohort,
              "seeds": seeds, "reference_model": reference, "reference_arm": reference_arm,
              "changed_patients_reference": "same model and seed, majority_nll_tiebreak",
              "classification_summary": "unrounded per-seed metrics; sample SD with ddof=1",
              "selective_accuracy": "null when no patient receives a prediction; five-seed summary null if any seed is undefined",
              "arms": arms, "joint_statistics": "joint_seed_statistics.json", "joint_spec": "joint_seed_spec.json",
              "sources": {"spec": _source(spec_path), "manifest": manifest_source}}
    _write_json(output_dir / "aggregation_comparison.json", report)
    fields = ["cohort", "model_name", "aggregation_strategy", "metric", "mean", "sample_sd", "defined_seed_count", "values"]
    with (output_dir / "aggregation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm in arms:
            for metric, values in arm["summary"].items():
                writer.writerow({"cohort": cohort, "model_name": arm["model_name"], "aggregation_strategy": arm["aggregation_strategy"],
                                 "metric": metric, **values, "values": json.dumps(values["values"])})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--exact-permutations", action="store_true", help="Enumerate swaps for at most 16 patients")
    args = parser.parse_args()
    try:
        result = compare(args.spec, args.output_dir, permutations=args.permutations, bootstrap=args.bootstrap,
                         random_seed=args.random_seed, exact=args.exact_permutations)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "cohort": result["cohort"],
                      "arms": len(result["arms"]), "reference_arm": result["reference_arm"]}))


if __name__ == "__main__":
    main()
