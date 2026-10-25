#!/usr/bin/env python3
"""Create one patient prediction per method by majority voting across seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


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
        key = row["logical_key"]
        if key in result:
            raise RuntimeError(f"Duplicate patient key in {path}: {key}")
        result[key] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.seeds) or len(args.inputs) < 2:
        raise RuntimeError("--inputs and --seeds must have the same length of at least two")
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise RuntimeError(f"Invalid --classes value: {args.classes}")

    paths = [Path(value).resolve() for value in args.inputs]
    tables = [read_predictions(path) for path in paths]
    keys = set(tables[0])
    if any(set(table) != keys for table in tables[1:]):
        raise RuntimeError("Seed prediction files do not contain the same patients")
    class_rank = {label: index for index, label in enumerate(classes)}
    rows = []
    disagreement = 0
    three_way_ties = 0
    for key in sorted(keys):
        seed_rows = [table[key] for table in tables]
        truths = {row["truth"] for row in seed_rows}
        centers = {row.get("center", "") for row in seed_rows}
        if len(truths) != 1 or len(centers) != 1:
            raise RuntimeError(f"Truth/center differs across seeds for patient {key}")
        predictions = [row["prediction"] for row in seed_rows]
        if any(label not in class_rank for label in predictions):
            raise RuntimeError(f"Out-of-vocabulary seed prediction for patient {key}: {predictions}")
        counts = Counter(predictions)
        best_count = max(counts.values())
        tied = [label for label, count in counts.items() if count == best_count]
        prediction = min(tied, key=lambda label: class_rank[label])
        disagreement += int(len(counts) > 1)
        three_way_ties += int(len(tied) > 1)
        rows.append(
            {
                "logical_key": key,
                "truth": next(iter(truths)),
                "prediction": prediction,
                "center": next(iter(centers)),
                "valid_votes": len(predictions),
                "input_groups": "",
                "patient_tie": int(len(tied) > 1),
                "seed_predictions": json.dumps(dict(zip(args.seeds, predictions)), separators=(",", ":")),
            }
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "passed",
        "method": args.method,
        "classes": classes,
        "patients": len(rows),
        "seeds": args.seeds,
        "seed_disagreement_patients": disagreement,
        "fixed_order_three_way_ties": three_way_ties,
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
