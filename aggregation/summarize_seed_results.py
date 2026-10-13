#!/usr/bin/env python3
"""Legacy repeated-seed summarizer for the pre-paper-evaluator metric schema.

Requires classification_overall and lesion_detection.{class_aware,class_agnostic},
not evaluate_paper_protocol.py's classification and per-threshold detection rows.
Legacy class_aware is not automatically equivalent to GFCA or SCF. No schema
adapter is supplied here; see README.md's Evaluation interface notes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--classes")
    args = parser.parse_args()
    if len(args.metrics) != len(args.seeds):
        raise RuntimeError("--metrics and --seeds must have the same length")

    metrics = []
    for seed, path in zip(args.seeds, args.metrics):
        with Path(path).resolve().open(encoding="utf-8") as stream:
            metrics.append((seed, json.load(stream)))

    observed_class_orders = [
        list(item["classification_overall"]["per_class"])
        for _, item in metrics
    ]
    observed_class_sets = [set(labels) for labels in observed_class_orders]
    if any(labels != observed_class_sets[0] for labels in observed_class_sets[1:]):
        raise RuntimeError(f"Per-class metric labels differ across seeds: {observed_class_orders}")
    classes = (
        [label.strip() for label in args.classes.split(",") if label.strip()]
        if args.classes
        else observed_class_orders[0]
    )
    if not classes or len(classes) != len(set(classes)) or set(classes) != observed_class_sets[0]:
        raise RuntimeError(
            f"Requested classes do not match metrics: requested={classes}, observed={observed_class_orders[0]}"
        )

    series = {}
    for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        series[f"classification:{name}"] = [
            item["classification_overall"][name] for _, item in metrics
        ]
    for label in classes:
        for name in ("precision", "recall", "f1"):
            series[f"class:{label}:{name}"] = [
                item["classification_overall"]["per_class"][label][name]
                for _, item in metrics
            ]
    for mode in ("class_aware", "class_agnostic"):
        for name in ("precision", "recall", "f1", "acc_iou_gt_0", "acc_iou_ge_0_5", "mean_iou"):
            series[f"lesion:{mode}:{name}"] = [
                item["lesion_detection"][mode][name] for _, item in metrics
            ]

    result = {
        "seeds": args.seeds,
        "runs": len(metrics),
        "classes": classes,
        "summary": {name: summarize(values) for name, values in sorted(series.items())},
    }
    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "mean", "sd", "min", "max", "values"])
        writer.writeheader()
        for name, values in result["summary"].items():
            writer.writerow({"metric": name, **values, "values": json.dumps(values["values"])})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
