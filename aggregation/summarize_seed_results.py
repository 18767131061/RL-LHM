#!/usr/bin/env python3
"""Summarize independent runs without averaging predictions or p-values.

Canonical CA, GFCA and SCF retain their distinct names. The explicit legacy
schema remains readable, but its class_aware key is never relabelled GFCA/SCF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_paper_protocol import _read_json
from evaluation.joint_seed_statistics import extract_run, require, validate_run_identity


def summarize(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and len(array) > 0 and bool(np.all(np.isfinite(array))), "Metrics must be nonempty finite numbers")
    return {"values": array.tolist(), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "min": float(array.min()), "max": float(array.max())}


def summarize_runs(
    metrics: list[dict], seeds: list[int], classes: list[str] | None = None,
    *, classification_seed_decimals: int | None = None,
) -> dict:
    require(classification_seed_decimals is None or
            (type(classification_seed_decimals) is int and classification_seed_decimals == 4),
            "classification_seed_decimals must be None or 4")
    require(bool(metrics) and len(metrics) == len(seeds), "--metrics and --seeds must have the same nonzero length")
    require(all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds), "Seeds must be distinct integers")
    schemas = ["canonical" if "protocol_version" in item else "legacy" for item in metrics]
    require(len(set(schemas)) == 1, "Canonical and legacy metric schemas cannot be mixed")
    series = {}
    identities = [item.get("run_identity") for item in metrics]
    require(all(identity is None for identity in identities) or all(isinstance(identity, dict) for identity in identities), "Run identity is missing from only some seeds")
    if schemas[0] == "canonical" or identities[0] is not None:
        identities = [validate_run_identity(item) for item in metrics]
        require(len({identity["model_name"] for identity in identities}) == 1, "Cannot summarize different model identities as seeds")
        require(all(identity["seed"] == seed for seed, identity in zip(seeds, identities)), "Declared seeds disagree with evaluator run identities")
    if schemas[0] == "canonical":
        runs = [extract_run(item) for item in metrics]
        layout, population, _, gt = runs[0]
        require(all(run[1] == population and np.array_equal(run[3], gt) for run in runs), "Canonical runs have different patients/GT/classes/protocols/manifests")
        require(classes is None or tuple(classes) == layout.macro_classes, "--classes must preserve canonical macro class order")
        classes = list(layout.macro_classes)
        values = np.stack([layout.metrics(run[2].sum(axis=0)) for run in runs])
        series = {name: values[:, i].tolist() for i, name in enumerate(layout.names)}
    else:
        require(all("classification_overall" in item and "lesion_detection" in item for item in metrics), "Unknown metric schema")
        orders = [list(item["classification_overall"]["per_class"]) for item in metrics]
        require(all(order == orders[0] for order in orders), "Legacy per-class label order differs across seeds")
        classes = orders[0] if classes is None else classes
        require(bool(classes) and len(classes) == len(set(classes)) and set(classes) == set(orders[0]), "Requested classes differ from legacy metrics")
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            series[f"classification:{name}"] = [item["classification_overall"][name] for item in metrics]
        for label in classes:
            for name in ("precision", "recall", "f1"):
                series[f"class:{label}:{name}"] = [item["classification_overall"]["per_class"][label][name] for item in metrics]
        for mode in ("class_aware", "class_agnostic"):
            for name in ("precision", "recall", "f1", "acc_iou_gt_0", "acc_iou_ge_0_5", "mean_iou"):
                series[f"lesion:{mode}:{name}"] = [item["lesion_detection"][mode][name] for item in metrics]
    unrounded = {}
    if classification_seed_decimals is not None:
        for name, values in series.items():
            if name.startswith(("classification:", "class:")):
                summarize(values)
                unrounded[name] = list(values)
                series[name] = [float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN))
                                for value in values]
    return {"schema": schemas[0], "seeds": seeds, "runs": len(metrics), "classes": classes,
            "run_identity_verified": identities[0] is not None,
            "classification_seed_decimals": classification_seed_decimals,
            "classification_seed_rounding": "decimal_half_even" if classification_seed_decimals is not None else "none",
            "unrounded_classification_seed_values": unrounded,
            "summary": {name: summarize(values) for name, values in sorted(series.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--classes")
    parser.add_argument("--classification-seed-decimals", type=int, choices=[4],
                        help="Round each seed's classification metrics to four decimals before mean/sample-SD aggregation")
    args = parser.parse_args()
    try:
        require(args.output_json.resolve() != args.output_csv.resolve(), "JSON and CSV destinations must differ")
        require(not args.output_json.exists() and not args.output_csv.exists(), "Output files already exist")
        paths = [Path(path).resolve() for path in args.metrics]
        require(len(paths) == len(set(paths)), "A metrics file cannot represent more than one run")
        inputs = [(path, path.read_bytes()) for path in paths]
        metrics = [_read_json(raw.decode("utf-8")) for _, raw in inputs]
        classes = [label.strip() for label in args.classes.split(",") if label.strip()] if args.classes else None
        result = summarize_runs(metrics, args.seeds, classes,
                                classification_seed_decimals=args.classification_seed_decimals)
        result["sources"] = [{"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()} for path, raw in inputs]
        for path in (args.output_json, args.output_csv):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["metric", "mean", "sd", "min", "max", "values"])
            writer.writeheader()
            for name, values in result["summary"].items():
                writer.writerow({"metric": name, **values, "values": json.dumps(values["values"])})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"runs": result["runs"], "schema": result["schema"], "output_json": str(args.output_json.resolve())}))


if __name__ == "__main__":
    main()
