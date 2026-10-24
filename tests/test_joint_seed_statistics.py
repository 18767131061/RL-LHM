"""Small exact checks of patient-paired inference; fixtures are synthetic only."""
from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aggregation.summarize_seed_results import summarize_runs
from evaluation.classification import classification_metrics
from evaluation.evaluate_paper_protocol import evaluate
from evaluation.joint_seed_statistics import (
    PairingError, analyze, benjamini_hochberg, extract_run, load_study,
    paired_statistics, permutation_differences, weighted_totals,
)
from evaluation.paired_patient_tests import read_predictions

SEEDS = [42, 2026, 3407, 1234, 5678]
CLASSES = ["HCC", "ICC"]
ROOT = Path(__file__).resolve().parents[1]


def make_run(tmp_path, predictions, *, name="Full", seed=42, truth=None, no_group=(), extra_predictions=()):
    truth = truth or ["HCC", "ICC"]
    manifest = {"schema_version": 1, "classes": CLASSES,
                "patients": [{"patient_id": f"p{i}", "truth": t} for i, t in enumerate(truth)], "input_groups": []}
    rows = []
    for i, (actual, predicted) in enumerate(zip(truth, predictions)):
        if i in no_group:
            continue
        manifest["input_groups"].append({"group_id": f"g{i}", "patient_id": f"p{i}", "phases": {
            phase: {"image_ref": f"g{i}-{phase}.png", "ground_truth": [{"label": actual, "bbox_2d": [0, 0, 10, 10]}] if phase == "a" else []}
            for phase in ("a", "p", "v")}})
        detections = [] if predicted is None else [{"phase": "a", "label": predicted, "bbox_2d": [0, 0, 10, 10]}]
        if i in extra_predictions:
            detections.append({"phase": "a", "label": predicted, "bbox_2d": [20, 20, 30, 30]})
        rows.append({"group_id": f"g{i}", "raw": json.dumps({"detections": detections}), "completion_nll": 0.5})
    path = tmp_path/f"{name}-{seed}-prediction.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    result = evaluate(manifest, path, [0, .5, .75])
    result["run_identity"] = {"model_name": name, "seed": seed}
    return result


def make_study(tmp_path, *, reference_predictions=None, comparator_predictions=None, truth=None, no_group=()):
    truth = truth or CLASSES
    reference_predictions = reference_predictions or [truth]*5
    comparator_predictions = comparator_predictions or [list(reversed(truth))]*5
    models = {}
    payloads = {}
    for name, predictions in [("Full", reference_predictions), ("SFT", comparator_predictions)]:
        models[name] = []
        for seed, predicted in zip(SEEDS, predictions):
            payload = make_run(tmp_path, predicted, name=name, seed=seed, truth=truth, no_group=no_group)
            path = tmp_path/f"{name}-{seed}.json"
            path.write_text(json.dumps(payload))
            models[name].append({"seed": seed, "metrics": path.name})
            payloads[name, seed] = payload
    spec = {"schema_version": 1, "seeds": SEEDS, "reference": "Full", "models": models}
    path = tmp_path/"study.json"; path.write_text(json.dumps(spec))
    return path, spec, payloads


def test_export_per_patient_counts_sum_to_whole_cohort_and_retain_empty_patients(tmp_path):
    run = make_run(tmp_path, ["HCC", None, None], truth=["HCC", "ICC", "HCC"], no_group=[2], extra_predictions=[0])
    layout, _, units, gt = extract_run(run)
    assert gt.tolist() == [1, 1, 0]
    assert len(run["patient_detection"]) == 3
    metrics = dict(zip(layout.names, layout.metrics(units.sum(axis=0))))
    assert metrics["classification:accuracy"] == pytest.approx(1/3)
    assert metrics["class:ICC:recall"] == 0
    assert metrics["detection:class_agnostic:iou=0.5:precision"] == .5
    assert metrics["detection:class_agnostic:iou=0.5:recall"] == .5
    assert metrics["detection:class_agnostic:mean_iou"] == .5
    assert sum(r["detection"]["class_agnostic"][1]["gt_instances"] for r in run["patient_detection"]) == 2


def test_exact_patient_swaps_are_synchronized_across_all_five_seeds(tmp_path):
    path, _, _ = make_study(tmp_path)
    study = load_study(path)
    result = analyze(study, permutations=7, bootstrap=50, random_seed=2026, exact=True, adjustment="bh")
    pair = result["comparisons"][0]
    accuracy = pair["metrics"]["classification:accuracy"]
    # Four patient masks: differences +1, 0, 0, -1, hence exact p=2/4.
    # Treating the 10 seed-patient observations as independent would give 2/1024.
    assert accuracy["reference_minus_comparator"] == 1
    assert accuracy["p_raw"] == .5
    assert accuracy["difference_patient_bootstrap_95_ci"] == [1, 1]
    assert accuracy["reference_sample_sd"] == 0
    assert pair["permutation_repeats"] == 4
    assert pair["permutation_method"] == "exact_enumeration"
    assert result["family_size"] == len(study.layout.names)
    assert all(row["q_bh"] >= row["p_raw"] for row in pair["metrics"].values())


def test_five_seed_non_linear_metrics_are_not_ensembled_or_pooled(tmp_path):
    truth = ["HCC", "HCC", "ICC", "ICC"]
    refs = [truth, ["HCC"]*4, ["ICC"]*4, ["HCC", "ICC", "ICC", None], [None]*4]
    comps = [["HCC", "ICC", "HCC", "ICC"]]*5
    path, _, _ = make_study(tmp_path, reference_predictions=refs, comparator_predictions=comps, truth=truth)
    study = load_study(path)
    result = analyze(study, permutations=8, bootstrap=30, exact=True)
    row = result["comparisons"][0]["metrics"]["classification:macro_f1"]
    expected = [classification_metrics(truth, p, CLASSES)["macro_f1"] for p in refs]
    pooled = classification_metrics(truth*5, sum(refs, []), CLASSES)["macro_f1"]
    assert row["reference_values"] == pytest.approx(expected)
    assert row["reference_mean"] == pytest.approx(np.mean(expected))
    assert row["reference_sample_sd"] == pytest.approx(np.std(expected, ddof=1))
    assert row["reference_mean"] != pytest.approx(pooled)
    masks = np.array(list(itertools.product([False, True], repeat=4)))
    actual = permutation_differences(study.layout, study.units["Full"], study.units["SFT"], masks)
    metric_index = study.layout.names.index("classification:macro_f1")
    expected_differences = []
    for mask in masks:
        differences = []
        for a, b in zip(refs, comps):
            left = [b[i] if mask[i] else a[i] for i in range(4)]
            right = [a[i] if mask[i] else b[i] for i in range(4)]
            differences.append(classification_metrics(truth, left, CLASSES)["macro_f1"]-classification_metrics(truth, right, CLASSES)["macro_f1"])
        expected_differences.append(np.mean(differences))
    assert actual[:, metric_index] == pytest.approx(expected_differences)
    observed = expected_differences[0]
    expected_p = sum(abs(x) >= abs(observed)-1e-12 for x in expected_differences)/16
    assert row["p_raw"] == pytest.approx(expected_p)
    # A bootstrap sample duplicates p0, drops p1, and retains p2/p3. Every seed
    # and both models must use those same patient multiplicities.
    weights = np.array([[2, 0, 1, 1]])
    boot = study.layout.metrics(weighted_totals(study.units["Full"], weights))[0]
    ids = [0, 0, 2, 3]
    for i in range(5):
        expected_boot = classification_metrics([truth[j] for j in ids], [refs[i][j] for j in ids], CLASSES)
        assert boot[i, metric_index] == pytest.approx(expected_boot["macro_f1"])


def test_mc_plus_one_has_nonzero_p_and_is_reproducible(tmp_path):
    path, _, _ = make_study(tmp_path)
    study = load_study(path)
    kwargs = {"permutations": 23, "bootstrap": 31, "random_seed": 7}
    first = analyze(study, **kwargs)
    second = analyze(study, **kwargs)
    assert first == second
    for row in first["comparisons"][0]["metrics"].values():
        assert row["p_raw"] == (row["extreme_permutations"]+1)/24
        assert row["q_bh"] is None
        assert all(np.isfinite(row["difference_patient_bootstrap_95_ci"]))


@pytest.mark.parametrize("fault, expected", [
    ("seed", "Run identity"), ("duplicate_seed", "Duplicate or mismatched seed"),
    ("patient", "populations differ"), ("truth", "Aggregate classification"),
    ("class_order", "identity mismatch"), ("manifest", "identity mismatch"),
    ("gt", "count identities"), ("summary", "Aggregate classification"),
    ("missing_units", "patient_detection"), ("protocol", "canonical keys"),
])
def test_misaligned_or_tampered_exports_are_rejected(tmp_path, fault, expected):
    path, spec, payloads = make_study(tmp_path)
    payload = copy.deepcopy(payloads["SFT", 42])
    if fault == "seed": payload["run_identity"]["seed"] = 999
    elif fault == "duplicate_seed":
        spec["models"]["SFT"][1]["seed"] = 42
        path.write_text(json.dumps(spec))
    elif fault == "patient": payload["patients"].pop()
    elif fault == "truth": payload["patients"][0]["truth"] = "ICC"
    elif fault == "class_order": payload["classes"].reverse()
    elif fault == "manifest": payload["evaluation_identity"]["manifest_sha256"] = "f"*64
    elif fault == "gt": payload["patient_detection"][0]["detection"]["class_agnostic"][0]["gt_instances"] += 1
    elif fault == "summary": payload["classification"]["accuracy"] = .5
    elif fault == "missing_units": del payload["patient_detection"]
    elif fault == "protocol": payload["detection"]["class_aware"] = payload["detection"].pop("geometry_first_class_aware")
    (tmp_path/"SFT-42.json").write_text(json.dumps(payload))
    with pytest.raises(PairingError, match=expected): load_study(path)


def test_summary_preserves_distinct_matching_protocols_and_no_vote_denominator(tmp_path):
    reports = [make_run(tmp_path, ["ICC", None], seed=seed) for seed in SEEDS]
    summary = summarize_runs(reports, SEEDS)
    metrics = summary["summary"]
    assert summary["schema"] == "canonical" and summary["run_identity_verified"]
    assert metrics["classification:accuracy"]["mean"] == 0
    assert metrics["detection:class_agnostic:iou=0.5:recall"]["mean"] == .5
    assert metrics["detection:geometry_first_class_aware:iou=0.5:recall"]["mean"] == 0
    assert metrics["detection:same_class_priority:iou=0.5:recall"]["mean"] == 0
    assert not any(":class_aware:" in name for name in metrics)
    with pytest.raises(PairingError, match="Declared seeds"):
        summarize_runs(reports, list(reversed(SEEDS)))
    with pytest.raises(PairingError, match="cannot be mixed"):
        summarize_runs([reports[0], {"classification_overall": {}}], [42, 2026])


@pytest.mark.parametrize("fault", ["missing", "missing_model", "blank_model", "padded_model", "noninteger_seed"])
def test_canonical_summary_requires_complete_valid_run_identities(tmp_path, fault):
    reports = [make_run(tmp_path, ["HCC", "ICC"], seed=seed) for seed in SEEDS]
    for report in reports:
        if fault == "missing":
            report.pop("run_identity")
        elif fault == "missing_model":
            report["run_identity"].pop("model_name")
        elif fault == "blank_model":
            report["run_identity"]["model_name"] = ""
        elif fault == "padded_model":
            report["run_identity"]["model_name"] = " Full "
        else:
            report["run_identity"]["seed"] = float(report["run_identity"]["seed"])
    with pytest.raises(PairingError, match="run_identity|Run identity"):
        summarize_runs(reports, SEEDS)


def test_joint_loader_rejects_numeric_equal_but_noninteger_run_seed(tmp_path):
    path, spec, payloads = make_study(tmp_path)
    payload = payloads["Full", 42]
    payload["run_identity"]["seed"] = 42.0
    (tmp_path / spec["models"]["Full"][0]["metrics"]).write_text(json.dumps(payload))
    with pytest.raises(PairingError, match="seed must be an integer"):
        load_study(path)


def test_summary_cli_rejects_reusing_one_metrics_path_as_distinct_runs(tmp_path):
    payload = make_run(tmp_path, ["HCC", "ICC"])
    payload.pop("run_identity")
    source = tmp_path / "one-run.json"
    source.write_text(json.dumps(payload))
    destination = tmp_path / "summary.json"
    command = [sys.executable, str(ROOT / "aggregation/summarize_seed_results.py"),
               "--metrics", *([str(source)] * 5), "--seeds", *map(str, SEEDS),
               "--output-json", str(destination), "--output-csv", str(tmp_path / "summary.csv")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "cannot represent more than one run" in result.stderr
    assert not destination.exists()


def test_legacy_summary_without_identities_is_explicitly_unverified(tmp_path):
    canonical = make_run(tmp_path, ["HCC", "ICC"])
    old_detection = {key: 1.0 for key in ("precision", "recall", "f1", "acc_iou_gt_0", "acc_iou_ge_0_5", "mean_iou")}
    legacy = {"classification_overall": canonical["classification"],
              "lesion_detection": {"class_aware": old_detection, "class_agnostic": old_detection}}
    result = summarize_runs([legacy], [42])
    assert result["schema"] == "legacy" and result["run_identity_verified"] is False
    assert "lesion:class_aware:recall" in result["summary"]


def test_close_iou_thresholds_retain_distinct_metrics_and_bh_hypotheses(tmp_path):
    # IoU=0.50000015 lies strictly between these valid thresholds. Formatting
    # both thresholds as :g previously overwrote one endpoint with the other.
    manifest = {"schema_version": 1, "classes": ["HCC"],
                "patients": [{"patient_id": "p", "truth": "HCC"}], "input_groups": [
                    {"group_id": "g", "patient_id": "p", "phases": {
                        phase: {"image_ref": phase + ".png", "ground_truth": [
                            {"label": "HCC", "bbox_2d": [0, 0, 10, 10]}] if phase == "a" else []}
                        for phase in ("a", "p", "v")}}]}
    prediction = tmp_path / "close-iou.jsonl"
    prediction.write_text(json.dumps({"group_id": "g", "raw": json.dumps({"detections": [
        {"phase": "a", "label": "HCC", "bbox_2d": [0, 0, 5.0000015, 10]}]})}) + "\n")
    template = evaluate(manifest, prediction, [0.5000001, 0.5000002])
    models = {}
    for name in ("Full", "SFT"):
        models[name] = []
        for seed in SEEDS:
            payload = copy.deepcopy(template)
            payload["run_identity"] = {"model_name": name, "seed": seed}
            source = tmp_path / f"{name}-{seed}.json"
            source.write_text(json.dumps(payload))
            models[name].append({"seed": seed, "metrics": source.name})
    spec = tmp_path / "close-iou-spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "seeds": SEEDS, "reference": "Full", "models": models}))
    result = analyze(load_study(spec), permutations=3, bootstrap=3, adjustment="bh")
    metrics = result["comparisons"][0]["metrics"]
    assert metrics["detection:class_agnostic:iou=0.5000001:recall"]["reference_mean"] == 1
    assert metrics["detection:class_agnostic:iou=0.5000002:recall"]["reference_mean"] == 0
    # 4 overall + 3 class + (3 protocols * 2 thresholds * 3 ratios) + 3 MeanIoU.
    assert len(metrics) == result["family_size"] == 28


def test_bh_uses_whole_family_and_retains_original_order():
    assert benjamini_hochberg([.04, .01, .03, .2]) == pytest.approx([.0533333333333, .04, .0533333333333, .2])


def test_cli_writes_auditable_json_csv_and_refuses_overwrite(tmp_path):
    path, _, _ = make_study(tmp_path)
    output = tmp_path/"stats"
    command = [sys.executable, str(ROOT/"evaluation/joint_seed_statistics.py"), "--spec", str(path), "--output-dir", str(output), "--permutations", "9", "--bootstrap", "11", "--adjustment", "bh"]
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads((output/"joint_seed_statistics.json").read_text())
    assert len(result["sources"]) == 10
    assert result["seeds"] == SEEDS
    assert (output/"joint_seed_statistics.csv").read_text().startswith("reference,comparator,metric")
    again = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode != 0 and "already exist" in again.stderr


def test_legacy_single_seed_reader_accepts_canonical_patient_id(tmp_path):
    path = tmp_path/"patients.csv"
    path.write_text("patient_id,truth,prediction\np1,HCC,\np2,ICC,ICC\n")
    records = read_predictions(path)
    assert set(records) == {"p1", "p2"}
    assert records["p1"]["prediction"] == ""
