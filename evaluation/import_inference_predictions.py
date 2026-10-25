#!/usr/bin/env python3
"""Convert inference CSV to canonical JSONL using an explicit frozen group map.

The map resolves identities only; raw invalid or empty completions are retained.
Ground truth is supplied independently by the evaluation manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_paper_protocol import validate_manifest, read_predictions


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or any(not name or name != name.strip() for name in headers):
            raise ValueError(f"{path}: duplicate, blank or padded CSV headers")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {sorted(required-set(reader.fieldnames or []))}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"{path}: CSV row width does not match its header")
        return rows


def key(row):
    logical = row.get("logical_key")
    if not isinstance(logical, str) or not logical.strip() or logical != logical.strip():
        raise ValueError("logical_key must be a nonempty unpadded string")
    value = row.get("slice_index")
    if value is None or not str(value).isdigit():
        raise ValueError("slice_index must be a nonnegative integer")
    return logical, int(value)


def convert(csv_path: Path, manifest_path: Path, group_map_path: Path, output: Path):
    if output.exists() or output.with_suffix(output.suffix + ".provenance.json").exists():
        raise ValueError("Use a new output path; existing predictions/provenance are not overwritten")
    source_hashes = {name: sha256(path) for name, path in
                     [("inference_csv", csv_path), ("manifest", manifest_path), ("group_map", group_map_path)]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, _, groups = validate_manifest(manifest)
    expected = {group["group_id"]: group["patient_id"] for group in groups}
    mapping = {}
    mapped_groups = set()
    for row in read_csv(group_map_path, {"logical_key", "slice_index", "group_id", "patient_id"}):
        k = key(row)
        group = row["group_id"]
        if k in mapping or group in mapped_groups:
            raise ValueError("Duplicate source key or group_id in group map")
        if group not in expected or row["patient_id"] != expected[group]:
            raise ValueError("Group map contains unknown groups or inconsistent patient identities")
        mapping[k] = row
        mapped_groups.add(group)
    if mapped_groups != set(expected):
        raise ValueError("Group map must cover every manifest group exactly once")
    records = {}
    for row in read_csv(csv_path, {"logical_key", "slice_index", "patient_id", "avg_nll", "raw"}):
        k = key(row)
        if k not in mapping or k in records:
            raise ValueError("Inference CSV has unknown or duplicate source keys")
        mapped = mapping[k]
        source_patient = mapped.get("source_patient_id") or mapped["patient_id"]
        if row["patient_id"] != source_patient:
            raise ValueError("Inference patient_id does not match the explicit group map")
        if row["raw"] is None:
            raise ValueError("CSV row is missing the raw completion field")
        raw_nll = row["avg_nll"]
        nll = float(raw_nll) if raw_nll else None
        if nll is not None and not math.isfinite(nll):
            nll = str(nll)
        elif nll is not None and nll < 0:
            raise ValueError("Finite completion NLL cannot be negative")
        records[k] = {"group_id": mapped["group_id"], "raw": row["raw"], "completion_nll": nll}
    if set(records) != set(mapping):
        raise ValueError("Inference CSV must include every manifest group; no intersection or filtering")
    for name, path in [("inference_csv", csv_path), ("manifest", manifest_path), ("group_map", group_map_path)]:
        if sha256(path) != source_hashes[name]:
            raise ValueError("Source changed during conversion")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in records.values():
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    read_predictions(output, set(expected))
    provenance = {"schema_version": 1, "sources_sha256": source_hashes,
                  "output_sha256": sha256(output), "groups": len(records),
                  "completion_filtering": "none", "gt_source": "independent evaluation manifest"}
    output.with_suffix(output.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--group-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.csv, args.manifest, args.group_map, args.output), indent=2))


if __name__ == "__main__":
    main()
