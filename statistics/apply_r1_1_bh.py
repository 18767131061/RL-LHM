#!/usr/bin/env python3
"""Apply the specified 60-test BH family across private and public cohorts.

This finalization step deliberately runs only after both cohorts have emitted
real per-Seed paired-test JSON files. It prevents accidental mixing of q
values calculated over different comparison families.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from evaluation.statistics import benjamini_hochberg  # noqa: E402


SEEDS = (42, 2026, 3407, 1234, 5678)
COMPARISONS = (
    ("vanilla_grpo", "full_rl_lhm"),
    ("sft", "full_rl_lhm"),
)
ENDPOINTS = (
    ("accuracy", ("mcnemar", "exact_two_sided_p")),
    ("macro_recall", ("paired_permutation_two_sided_p", "macro_recall")),
    ("macro_f1", ("paired_permutation_two_sided_p", "macro_f1")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def nested(payload: dict[str, Any], location: tuple[str, str]) -> float:
    value = payload[location[0]][location[1]]
    if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"Invalid p value at {location}: {value!r}")
    return float(value)


def load_one(path: Path, cohort: str, seed: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing required R1-1 paired result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    if provenance.get("cohort") != cohort:
        raise RuntimeError(f"Cohort provenance mismatch in {path}")
    if provenance.get("model_seed") != seed:
        raise RuntimeError(f"Model Seed provenance mismatch in {path}")
    if not provenance.get("aggregation_rule") or not provenance.get("evaluation_version"):
        raise RuntimeError(f"Missing aggregation/evaluation provenance in {path}")
    if not provenance.get("test_manifest"):
        raise RuntimeError(f"Missing frozen test manifest provenance in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-paired-root", required=True)
    parser.add_argument("--public-paired-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = {
        "private": Path(args.private_paired_root).resolve(),
        "public": Path(args.public_paired_root).resolve(),
    }
    entries: list[tuple[Path, str, int, str, str, dict[str, Any]]] = []
    p_values: dict[str, float] = {}
    for cohort, root in roots.items():
        for seed in SEEDS:
            for reference, candidate in COMPARISONS:
                path = root / f"seed_{seed}_{reference}_vs_{candidate}.json"
                payload = load_one(path, cohort, seed)
                entries.append((path, cohort, seed, reference, candidate, payload))
                prefix = f"{cohort}|seed={seed}|{reference}_vs_{candidate}"
                for endpoint, location in ENDPOINTS:
                    p_values[f"{prefix}|{endpoint}"] = nested(payload, location)

    expected = 2 * len(SEEDS) * len(COMPARISONS) * len(ENDPOINTS)
    if len(p_values) != expected:
        raise RuntimeError(f"Expected {expected} R1-1 hypotheses, found {len(p_values)}")
    q_values = benjamini_hochberg(p_values)
    family_id = f"R1-1-private-public-2comparison-5seed-3endpoint-n{expected}"
    rows = []
    for path, cohort, seed, reference, candidate, payload in entries:
        prefix = f"{cohort}|seed={seed}|{reference}_vs_{candidate}"
        q = {endpoint: q_values[f"{prefix}|{endpoint}"] for endpoint, _ in ENDPOINTS}
        payload["bh"] = {
            "family_id": family_id,
            "hypothesis_count": expected,
            "q_values": q,
        }
        atomic_write_json(path, payload)
        provenance = payload["provenance"]
        rows.append(
            {
                "cohort": cohort,
                "seed": seed,
                "reference_model": payload["reference"],
                "candidate_model": payload["candidate"],
                "accuracy_mcnemar_p": nested(payload, ENDPOINTS[0][1]),
                "macro_recall_permutation_p": nested(payload, ENDPOINTS[1][1]),
                "macro_f1_permutation_p": nested(payload, ENDPOINTS[2][1]),
                "q_accuracy": q["accuracy"],
                "q_macro_recall": q["macro_recall"],
                "q_macro_f1": q["macro_f1"],
                "bh_family_id": family_id,
                "test_manifest_sha256": provenance["test_manifest"]["sha256"],
                "reference_prediction_sha256": provenance["reference_prediction"]["sha256"],
                "candidate_prediction_sha256": provenance["candidate_prediction"]["sha256"],
                "aggregation_rule": provenance["aggregation_rule"],
                "evaluation_version": provenance["evaluation_version"],
            }
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "complete",
        "family_id": family_id,
        "hypothesis_count": expected,
        "rows": len(rows),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
