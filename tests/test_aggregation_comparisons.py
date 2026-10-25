"""Independent synthetic readout expectations and complete comparison contracts."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from aggregation.patient import (
    BEST_LIKELIHOOD_INPUT as BEST, LIKELIHOOD_WEIGHTED_VOTE as WEIGHTED,
    MAJORITY_NLL_TIEBREAK as DEFAULT, VoteRecord, aggregate_patient,
)
from evaluation.compare_aggregation_strategies import _summary, compare
from evaluation.evaluate_paper_protocol import AGGREGATION_RULES, evaluate
from evaluation.joint_seed_statistics import PairingError, load_study

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 2026, 3407, 1234, 5678]


def inputs(tmp_path, *, models=("Full",), empty=False):
    classes = ["HCC", "ICC"]
    truths = ["HCC", "ICC", "HCC", "ICC", "HCC"]
    manifest = {"schema_version": 1, "classes": classes,
                "patients": [{"patient_id": f"p{i}", "truth": truth} for i, truth in enumerate(truths)], "input_groups": []}
    definitions = [(0, "HCC", 5), (0, "HCC", 5), (0, "ICC", 0),
                   (1, "HCC", "NaN"), (1, "ICC", 1),
                   (2, "invalid", 0), (2, None, None),
                   (4, "HCC", .4), (4, "HCC", .4), (4, "ICC", 0)] if not empty else []
    predictions = []
    for i, (patient, label, nll) in enumerate(definitions):
        group = f"g{i}"
        manifest["input_groups"].append({"group_id": group, "patient_id": f"p{patient}", "phases": {
            phase: {"image_ref": f"{group}-{phase}.png", "ground_truth": [{"label": truths[patient], "bbox_2d": [0, 0, 10, 10]}] if phase == "a" else []}
            for phase in ("a", "p", "v")}})
        raw = json.dumps({"detections": [{"phase": "a", "label": label, "bbox_2d": [0, 0, 10, 10]}] if label else []})
        predictions.append({"group_id": group, "raw": raw, "completion_nll": nll})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    specs = {}
    for model in models:
        specs[model] = []
        for seed in SEEDS:
            path = tmp_path / f"{model}-{seed}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
            specs[model].append({"seed": seed, "predictions": path.name})
    spec = {"schema_version": 1, "cohort": "synthetic", "manifest": "manifest.json", "seeds": SEEDS,
            "reference_model": "Full", "models": specs, "iou_thresholds": [0, .5]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, manifest, spec


@pytest.mark.parametrize("strategy,expected,accuracy,f1", [
    (DEFAULT, ["HCC", "ICC", None, None, "HCC"], .6, 11/15),
    (WEIGHTED, ["ICC", "HCC", None, None, "HCC"], .2, .2),
    (BEST, ["ICC", "ICC", None, None, "ICC"], .2, .2),
])
def test_evaluator_expected_predictions_fixed_denominator_and_unchanged_detection(tmp_path, strategy, expected, accuracy, f1):
    _, manifest, _ = inputs(tmp_path)
    prediction = tmp_path / "Full-42.jsonl"
    baseline = evaluate(manifest, prediction, [0, .5])
    result = evaluate(manifest, prediction, [0, .5], strategy)
    assert [row["prediction"] for row in result["patients"]] == expected
    assert result["classification"]["n"] == 5
    assert result["classification"]["accuracy"] == pytest.approx(accuracy)
    assert result["classification"]["macro_f1"] == pytest.approx(f1)
    for key in ("coverage", "input_groups", "detection", "patient_detection", "detection_scope", "sources"):
        assert result[key] == baseline[key]
    assert result["coverage"]["patient_diagnostic_coverage"] == .6
    assert result["coverage"]["patients_without_complete_input_groups"] == 1
    ca = result["detection"]["class_agnostic"][1]
    assert (ca["tp"], ca["fp"], ca["fn"]) == (8, 0, 2)
    assert result["evaluation_identity"]["aggregation_rule"] == AGGREGATION_RULES[strategy]


@pytest.mark.parametrize("records,strategy,expected", [
    # Upper median [0, 2] is 2: using midpoint 1 would flip A/B because
    # exp(-2)+3*exp(-1)>1, while 4*exp(-2)<1.
    ([VoteRecord("B", 0), VoteRecord("A", 2), VoteRecord("A", None), VoteRecord("A", math.inf), VoteRecord("A", math.nan)], WEIGHTED, "B"),
    ([VoteRecord("B", math.inf, 0), VoteRecord("A", math.nan, 1)], WEIGHTED, "A"),
    ([VoteRecord("B", math.inf, 0), VoteRecord("A", math.nan, 1)], BEST, "B"),
    ([VoteRecord("B", 1, 0), VoteRecord("A", 1, 1)], BEST, "B"),
    ([VoteRecord("B", 1, 0), VoteRecord("A", 1, 0)], BEST, "A"),
    ([VoteRecord("A", 0), VoteRecord("B", -5e-13)], WEIGHTED, "A"),
    ([VoteRecord("A", 0), VoteRecord("B", -2e-12)], WEIGHTED, "B"),
    ([VoteRecord("B", 1000), VoteRecord("A", 50)], WEIGHTED, "A"),
    ([VoteRecord("A", -1000), VoteRecord("B", -50)], WEIGHTED, "A"),
    ([VoteRecord("B", 50)], WEIGHTED, "B"),
    ([VoteRecord(None, 0), VoteRecord("outside", -1)], BEST, None),
])
def test_readout_expected_values_for_missing_nll_ties_clipping_and_ineligible_votes(records, strategy, expected):
    assert aggregate_patient(records, ["A", "B"], strategy).prediction == expected


def test_comparison_cli_exports_every_readout_and_default_reference_inference(tmp_path):
    path, _, _ = inputs(tmp_path, models=("GRPO", "Full"))
    output = tmp_path / "results"
    command = [sys.executable, str(ROOT / "evaluation/compare_aggregation_strategies.py"),
               "--spec", str(path), "--output-dir", str(output), "--permutations", "9", "--bootstrap", "13", "--exact-permutations"]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads((output / "aggregation_comparison.json").read_text())
    statistics = json.loads((output / report["joint_statistics"]).read_text())
    assert len(report["arms"]) == 6
    assert statistics["comparison_mode"] == "aggregation_table"
    assert len(statistics["sources"]) == 30
    assert len(statistics["comparisons"]) == 5
    assert {pair["reference"] for pair in statistics["comparisons"]} == {f"Full::{DEFAULT}"}
    for arm in report["arms"]:
        strategy = arm["aggregation_strategy"]
        assert arm["summary"]["changed_patients"]["values"] == [0 if strategy == DEFAULT else 2] * 5
        assert arm["summary"]["coverage"]["values"] == [.6] * 5
        assert arm["summary"]["selective_accuracy"]["values"] == pytest.approx([1 if strategy == DEFAULT else 1/3] * 5)
        for run in arm["runs"]:
            canonical = output / run["metrics"]
            assert canonical.exists()
            assert (canonical.parent / "patient_predictions.csv").exists()
            assert (canonical.parent / "input_group_audit.csv").exists()
    pair = next(pair for pair in statistics["comparisons"] if pair["comparator"] == f"Full::{WEIGHTED}")
    # Only p0 and p1 differ in correctness: of 32 complete-patient masks,
    # half preserve |difference|=.4 and half give zero. Seeds share that mask.
    assert pair["metrics"]["classification:accuracy"]["p_raw"] == .5
    assert pair["metrics"]["classification:accuracy"]["reference_minus_comparator"] == pytest.approx(.4)
    assert pair["metrics"]["detection:class_agnostic:iou=0.5:f1"]["reference_minus_comparator"] == 0
    assert (output / "aggregation_summary.csv").read_text().startswith("cohort,model_name,aggregation_strategy")
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and "already exists" in second.stderr


def test_no_votes_have_null_selective_accuracy_and_retain_all_patients(tmp_path):
    path, _, _ = inputs(tmp_path, empty=True)
    result = compare(path, tmp_path / "result", permutations=3, bootstrap=3)
    for arm in result["arms"]:
        assert arm["summary"]["selective_accuracy"] == {"values": [None]*5, "defined_seed_count": 0, "mean": None, "sample_sd": None}
        assert arm["summary"]["coverage"]["mean"] == 0
        assert arm["summary"]["classification:accuracy"]["mean"] == 0
        assert all(run["changed_patients"] == 0 for run in arm["runs"])


def test_partial_undefined_selective_metric_keeps_five_seed_summary_undefined():
    assert _summary([1, .5, None, 0, .5]) == {
        "values": [1, .5, None, 0, .5], "defined_seed_count": 4, "mean": None, "sample_sd": None}


def test_nonconstant_five_seed_summary_uses_sample_standard_deviation(tmp_path):
    path, _, _ = inputs(tmp_path)
    predictions = tmp_path / "Full-42.jsonl"
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[0]["completion_nll"] = rows[1]["completion_nll"] = 0
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = compare(path, tmp_path / "result", permutations=3, bootstrap=3)
    weighted = next(arm for arm in report["arms"] if arm["aggregation_strategy"] == WEIGHTED)
    metric = weighted["summary"]["classification:accuracy"]
    assert metric["values"] == pytest.approx([.4, .2, .2, .2, .2])
    assert metric["mean"] == pytest.approx(.24)
    assert metric["sample_sd"] == pytest.approx(math.sqrt(.008))
    assert weighted["summary"]["changed_patients"]["values"] == [1, 2, 2, 2, 2]
    assert weighted["summary"]["changed_patients"]["sample_sd"] == pytest.approx(math.sqrt(.2))


@pytest.mark.parametrize("fault,expected", [
    ("source", "source identities differ"), ("missing_source", "source SHA256"),
    ("nll", "input groups"), ("coverage", "alter coverage"),
    ("seed", "Run identity"), ("model", "Run identity"),
    ("manifest", "identity mismatch"), ("rule", "rule identity"),
    ("undeclared", "explicit aggregation"), ("ordinary", "Run identity|identity mismatch"),
    ("missing_anchor", "default aggregation anchor"),
])
def test_cross_strategy_pairing_is_explicit_and_source_checked(tmp_path, fault, expected):
    path, _, _ = inputs(tmp_path)
    output = tmp_path / "result"
    compare(path, output, permutations=3, bootstrap=3)
    joint_path = output / "joint_seed_spec.json"
    spec = json.loads(joint_path.read_text())
    alt = f"Full::{WEIGHTED}"
    metrics_path = output / spec["models"][alt][0]["metrics"]
    payload = json.loads(metrics_path.read_text())
    if fault == "source": payload["sources"]["predictions"]["sha256"] = "f" * 64
    elif fault == "missing_source": del payload["sources"]["predictions"]
    elif fault == "nll": payload["input_groups"][0]["completion_nll"] = 99
    elif fault == "coverage": payload["coverage"]["patient_diagnostic_coverage"] = 1
    elif fault == "seed": payload["run_identity"]["seed"] = 99
    elif fault == "model": payload["run_identity"]["model_name"] = "GRPO"
    elif fault == "manifest": payload["evaluation_identity"]["manifest_sha256"] = "f" * 64
    elif fault == "rule": payload["evaluation_identity"]["aggregation_rule"] = AGGREGATION_RULES[DEFAULT]
    elif fault == "undeclared": spec.pop("comparison_mode")
    elif fault == "ordinary":
        spec.pop("comparison_mode"); spec.pop("arms")
    elif fault == "missing_anchor":
        old = f"Full::{DEFAULT}"
        spec["arms"].pop(old); spec["models"].pop(old); spec["reference"] = alt
    metrics_path.write_text(json.dumps(payload))
    joint_path.write_text(json.dumps(spec))
    with pytest.raises(PairingError, match=expected):
        load_study(joint_path)


def test_regular_model_comparison_still_rejects_different_aggregation_rules(tmp_path):
    path, _, _ = inputs(tmp_path, models=("Full", "GRPO"))
    output = tmp_path / "result"
    compare(path, output, permutations=3, bootstrap=3)
    mixed = json.loads((output / "joint_seed_spec.json").read_text())
    ordinary = {"schema_version": 1, "seeds": SEEDS, "reference": "Full", "models": {
        "Full": mixed["models"][f"Full::{DEFAULT}"], "GRPO": mixed["models"][f"GRPO::{WEIGHTED}"]}}
    spec_path = output / "ordinary.json"
    spec_path.write_text(json.dumps(ordinary))
    with pytest.raises(PairingError, match="identity mismatch"):
        load_study(spec_path)


@pytest.mark.parametrize("fault,expected", [
    ("seed", "Duplicate or mismatched seed"), ("source", "more than one"),
    ("manifest_record", "Missing prediction records"), ("strategy", "Unknown paper aggregation"),
])
def test_input_contract_fails_before_creating_output(tmp_path, fault, expected):
    path, manifest, spec = inputs(tmp_path)
    if fault == "strategy":
        with pytest.raises(ValueError, match=expected):
            evaluate(manifest, tmp_path / "Full-42.jsonl", [.5], "unknown")
        return
    if fault == "seed": spec["models"]["Full"][1]["seed"] = 42
    elif fault == "source": spec["models"]["Full"][1]["predictions"] = "Full-42.jsonl"
    else: (tmp_path / "Full-42.jsonl").write_text("")
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=expected):
        compare(path, tmp_path / "result", permutations=3, bootstrap=3)
    assert not (tmp_path / "result").exists()
