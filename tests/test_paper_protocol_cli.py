"""End-to-end synthetic contracts for the paper-facing evaluation command."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "evaluation" / "evaluate_paper_protocol.py"
CLASSES = ["HCC", "ICC", "FNH"]


def lesion(label="HCC", box=(0, 0, 10, 10)):
    return {"label": label, "bbox_2d": list(box)}


def group(group_id, patient_id, annotations):
    return {
        "group_id": group_id,
        "patient_id": patient_id,
        "phases": {
            phase: {"image_ref": f"synthetic/{group_id}-{phase}.png", "ground_truth": annotations.get(phase, [])}
            for phase in ("a", "p", "v")
        },
    }


def completion(detections):
    return json.dumps({"detections": [{"phase": phase, **item} for phase, item in detections]})


def prediction(group_id, detections, nll=None):
    return {"group_id": group_id, "raw": completion(detections), "completion_nll": nll}


def run_cli(tmp_path, patients, groups, predictions, thresholds=(0.5,)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "classes": CLASSES, "patients": patients, "input_groups": groups}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    output = tmp_path / "results"
    command = [
        sys.executable, str(CLI), "--manifest", str(tmp_path / "manifest.json"),
        "--predictions", str(tmp_path / "predictions.jsonl"), "--output-dir", str(output),
        "--iou-thresholds", *[str(value) for value in thresholds],
    ]
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    result = json.loads((output / "metrics.json").read_text()) if (output / "metrics.json").exists() else None
    return process, result, output


def test_all_eligible_votes_positive_phase_detection_and_full_patient_denominator(tmp_path):
    patients = [{"patient_id": "p1", "truth": "ICC"}, {"patient_id": "p2", "truth": "HCC"}, {"patient_id": "p3", "truth": "HCC"}]
    groups = [
        group("g1", "p1", {phase: [lesion()] for phase in ("a", "p", "v")}),
        group("g2", "p1", {}),
        group("g3", "p2", {"a": [lesion()]}),
    ]
    predictions = [
        prediction("g1", [(phase, lesion()) for phase in ("a", "p", "v")], 2),
        prediction("g2", [("a", lesion("ICC"))], 0.1),
        # One invalid label must invalidate the entire completion, including
        # its otherwise correct lesion. It remains an explicit output record.
        prediction("g3", [("a", lesion()), ("p", lesion("not HCC"))]),
    ]
    process, result, output = run_cli(tmp_path, patients, groups, predictions)
    assert process.returncode == 0, process.stderr
    assert result["classification"]["n"] == 3
    assert result["classification"]["accuracy"] == pytest.approx(1 / 3)
    rows = {row["patient_id"]: row for row in result["patients"]}
    assert rows["p1"]["prediction"] == "ICC"
    assert rows["p1"]["valid_votes"] == 2
    assert rows["p2"]["prediction"] is None
    assert rows["p3"]["prediction"] is None
    assert result["coverage"]["patients_without_valid_vote"] == 2
    assert result["coverage"]["patients_without_complete_input_groups"] == 1
    assert result["coverage"]["schema_valid_groups"] == 2
    assert result["detection_scope"]["positive_phase_slices"] == 4
    assert result["detection_scope"]["excluded_no_lesion_phase_slices"] == 5
    assert result["detection_scope"]["eligible_predictions_on_excluded_phases"] == 1
    for protocol in result["detection"].values():
        metrics = protocol[0]
        assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (3, 0, 1)
        assert metrics["precision"] == 1
        assert metrics["recall"] == pytest.approx(0.75)
        assert metrics["tp"] + metrics["fp"] == metrics["predicted_instances"]
        assert metrics["tp"] + metrics["fn"] == metrics["gt_instances"]
    with (output / "patient_predictions.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert set(result["sources"]) == {"manifest", "predictions"}


def test_three_protocols_use_independent_phase_and_lesion_ground_truth(tmp_path):
    # In phase a the geometrically perfect pairs have wrong classes. Same-class
    # reassignment gives IoU=2/3 for both lesions. Phase p has a different GT
    # box and no overlap; copying phase a's GT would incorrectly award credit.
    groups = [group("g1", "p1", {
        "a": [lesion("HCC"), lesion("ICC", (2, 0, 12, 10))],
        "p": [lesion("HCC", (100, 0, 110, 10))],
    })]
    predictions = [prediction("g1", [
        ("a", lesion("ICC")), ("a", lesion("HCC", (2, 0, 12, 10))), ("p", lesion("HCC")),
    ])]
    process, result, _ = run_cli(tmp_path, [{"patient_id": "p1", "truth": "HCC"}], groups, predictions, (0, 0.5, 0.9))
    assert process.returncode == 0, process.stderr
    assert set(result["detection"]) == {"class_agnostic", "same_class_priority", "geometry_first_class_aware"}
    ca = result["detection"]["class_agnostic"]
    scf = result["detection"]["same_class_priority"]
    gfca = result["detection"]["geometry_first_class_aware"]
    assert [row["tp"] for row in ca] == [2, 2, 2]
    assert [row["tp"] for row in scf] == [2, 2, 0]
    assert [row["tp"] for row in gfca] == [0, 0, 0]
    assert ca[0]["threshold_rule"] == "IoU > 0"
    assert ca[0]["mean_iou_with_misses_zero"] == pytest.approx(2 / 3)
    assert scf[0]["mean_iou_with_misses_zero"] == pytest.approx(4 / 9)
    assert all(row["gt_instances"] == 3 for rows in result["detection"].values() for row in rows)


def test_nonfinite_nll_retains_votes_and_fixed_order_only_breaks_minimum_nll_ties(tmp_path):
    patients = [{"patient_id": f"p{i}", "truth": "HCC"} for i in range(1, 5)]
    definitions = [
        ("g1", "p1", [("a", lesion("HCC"))], "NaN"),
        ("g2", "p1", [("a", lesion("ICC"))], 0.2),
        ("g3", "p2", [("a", lesion("HCC"))], None),
        ("g4", "p2", [("a", lesion("ICC"))], "Infinity"),
        ("g5", "p3", [("a", lesion("ICC")), ("p", lesion("HCC"))], 0.01),
        ("g6", "p4", [("a", lesion("HCC"))], 5),
        ("g7", "p4", [("a", lesion("ICC"))], 1),
        ("g8", "p4", [("a", lesion("FNH"))], 1),
    ]
    groups = [group(group_id, patient_id, {}) for group_id, patient_id, _, _ in definitions]
    predictions = [prediction(group_id, detections, nll) for group_id, _, detections, nll in definitions]
    process, result, _ = run_cli(tmp_path, patients, groups, predictions)
    assert process.returncode == 0, process.stderr
    rows = {row["patient_id"]: row for row in result["patients"]}
    assert [rows[f"p{i}"]["prediction"] for i in range(1, 5)] == ["ICC", "HCC", "HCC", "ICC"]
    assert [rows[f"p{i}"]["valid_votes"] for i in range(1, 5)] == [2, 2, 1, 3]
    assert result["coverage"]["diagnostic_eligible_groups"] == 8
    assert result["detection_scope"]["positive_phase_slices"] == 0
    assert all(row["gt_instances"] == 0 for rows in result["detection"].values() for row in rows)


@pytest.mark.parametrize("fault, expected", [
    ("missing_prediction", "Missing prediction records"),
    ("duplicate_prediction", "Unknown or duplicate prediction"),
    ("unknown_prediction", "Unknown or duplicate prediction"),
    ("missing_phase", "explicit a/p/v"),
    ("legacy_box", "explicit a/p/v"),
    ("missing_lesion_label", "every GT lesion needs its own label"),
    ("invalid_gt_box", "out of contract"),
    ("unknown_patient", "Duplicate group or unknown patient"),
])
def test_manifest_and_prediction_integrity_fail_closed(tmp_path, fault, expected):
    patients = [{"patient_id": "p1", "truth": "HCC"}]
    groups = [group("g1", "p1", {"a": [lesion()]})]
    predictions = [prediction("g1", [("a", lesion())])]
    if fault == "missing_prediction":
        predictions = []
    elif fault == "duplicate_prediction":
        predictions.append(copy.deepcopy(predictions[0]))
    elif fault == "unknown_prediction":
        predictions[0]["group_id"] = "other"
    elif fault == "missing_phase":
        del groups[0]["phases"]["v"]
    elif fault == "legacy_box":
        del groups[0]["phases"]
        groups[0]["bbox_2d"] = [[0, 0, 10, 10]]
    elif fault == "missing_lesion_label":
        del groups[0]["phases"]["a"]["ground_truth"][0]["label"]
    elif fault == "invalid_gt_box":
        groups[0]["phases"]["a"]["ground_truth"][0]["bbox_2d"] = [5, 5, 1, 1]
    elif fault == "unknown_patient":
        groups[0]["patient_id"] = "unknown"
    process, result, output = run_cli(tmp_path, patients, groups, predictions)
    assert process.returncode != 0
    assert expected in process.stderr
    assert result is None
    assert not output.exists()


def test_cohort_with_no_complete_input_groups_keeps_all_patients(tmp_path):
    process, result, _ = run_cli(tmp_path, [{"patient_id": "p1", "truth": "HCC"}, {"patient_id": "p2", "truth": "ICC"}], [], [])
    assert process.returncode == 0, process.stderr
    assert result["classification"]["n"] == 2
    assert result["classification"]["accuracy"] == 0
    assert result["coverage"]["patients_without_complete_input_groups"] == 2
    assert result["coverage"]["patients_without_valid_vote"] == 2


def test_existing_evaluation_outputs_are_not_overwritten(tmp_path):
    patients = [{"patient_id": "p1", "truth": "HCC"}]
    first, _, output = run_cli(tmp_path, patients, [], [])
    assert first.returncode == 0, first.stderr
    original = (output / "metrics.json").read_bytes()
    second, _, _ = run_cli(tmp_path, patients, [], [])
    assert second.returncode != 0
    assert "Output files already exist" in second.stderr
    assert (output / "metrics.json").read_bytes() == original
