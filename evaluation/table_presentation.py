"""Presentation views of existing results; no metric or test recomputation.

Inputs contain aggregate results only. Every view retains a deep copy of its
source records so omitted display columns cannot remove evaluation fields.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

SEEDS = (42, 2026, 3407, 1234, 5678)
COHORTS = ("private", "public")
COMPARATORS = ("sft", "vanilla_grpo")
ENDPOINTS = ("accuracy", "macro_recall", "macro_f1")


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite numeric data") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def paired_summary(rows):
    """Summarize the existing 60 comparisons, retaining each actual seed CI.

    Delta is supplied as Full minus comparator. q must already be the BH value
    from the complete family; this function neither fits nor averages it.
    Numeric strings are retained, including their source precision.
    """
    source = copy.deepcopy(list(rows))
    indexed = {}
    expected = set(itertools.product(COHORTS, SEEDS, COMPARATORS, ENDPOINTS))
    for row in source:
        key = tuple(row[field] for field in ("cohort", "seed", "comparator", "endpoint"))
        if key not in expected or key in indexed:
            raise ValueError("Expected one row per cohort/seed/comparator/endpoint in the 60-item family")
        if row.get("candidate") != "full_rl_lhm":
            raise ValueError("Paired direction must be full_rl_lhm minus comparator")
        delta, lower, upper, p, q = (_number(row[field], field) for field in
                                    ("delta", "ci_lower", "ci_upper", "p", "q"))
        if lower > upper or not 0 <= p <= 1 or not 0 <= q <= 1:
            raise ValueError("Invalid supplied CI or p/q")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("All 60 existing comparisons are required; no partial family is inferred")
    summaries = []
    for cohort, comparator, endpoint in itertools.product(COHORTS, COMPARATORS, ENDPOINTS):
        details = [indexed[(cohort, seed, comparator, endpoint)] for seed in SEEDS]
        summary = {"cohort": cohort, "candidate": "full_rl_lhm",
                   "comparator": comparator, "endpoint": endpoint,
                   "seed_details": copy.deepcopy(details)}
        for field in ("delta", "q"):
            values = [row[field] for row in details]
            summary[field + "_range"] = [min(values, key=float), max(values, key=float)]
        summaries.append(summary)
    return {"family_size": 60, "direction": "full_rl_lhm minus comparator",
            "source_rows": source, "rows": summaries}


def factorial_sections(rows):
    """Separate all eight component settings from the Uniform severity control."""
    source = copy.deepcopy(list(rows))
    factorial, controls, identities = [], [], set()
    for row in source:
        if row.get("section") == "additional_severity_control":
            if row.get("control") != "uniform":
                raise ValueError("The additional severity control must explicitly identify Uniform")
            controls.append(row)
            continue
        if row.get("section") != "factorial" or row.get("control") is not None:
            raise ValueError("Every row must explicitly identify its experimental section")
        key = tuple(row[field] for field in ("rars", "shae", "vgs"))
        if any(type(flag) is not bool for flag in key) or key in identities:
            raise ValueError("Factorial settings must be unique Boolean component combinations")
        identities.add(key)
        factorial.append(row)
    if identities != set(itertools.product((False, True), repeat=3)) or len(controls) != 1:
        raise ValueError("Require the eight factorial settings and one separate Uniform control")
    return {"source_rows": source, "factorial_rows": copy.deepcopy(factorial),
            "additional_severity_control": copy.deepcopy(controls)}


def aggregation_view(rows):
    """Hide redundant coverage/accuracy columns only after full-coverage checks.

    Supply one record per cohort/model/strategy with five seed-level coverage
    values and independently supplied summary cells (for example mean and SD).
    This function does not derive SD, precision, or patient results.
    """
    source = copy.deepcopy(list(rows))
    if not source:
        raise ValueError("Aggregation rows must be nonempty")
    displayed, identities = [], set()
    fields = ("cohort", "model", "strategy", "accuracy", "macro_f1", "changed_patients")
    for row in source:
        key = tuple(row[field] for field in fields[:3])
        if key in identities:
            raise ValueError("Duplicate cohort/model/strategy identity")
        identities.add(key)
        coverage = row["coverage"]
        if not isinstance(coverage, dict) or set(coverage) != {str(seed) for seed in SEEDS}:
            raise ValueError("Supply coverage for each of the five seeds")
        if any(_number(value, "coverage") != 1 for value in coverage.values()):
            raise ValueError("Coverage may be hidden only when every seed has full coverage")
        if row["accuracy"] != row["selective_accuracy"]:
            raise ValueError("Overall and selective accuracy cells differ; do not hide the distinction")
        displayed.append({field: copy.deepcopy(row[field]) for field in fields})
    return {"source_rows": source, "rows": displayed,
            "coverage_note": "All displayed readouts have 100% patient coverage; Accuracy uses all test patients."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("paired", "factorial", "aggregation"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Choose a fresh output file, distinct from the source")
    functions = {"paired": paired_summary, "factorial": factorial_sections, "aggregation": aggregation_view}
    result = functions[args.kind](json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
