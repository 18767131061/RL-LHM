"""Classification seed summaries with synthetic patient records."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from aggregation.summarize_seed_results import summarize_runs
from evaluation.evaluate_paper_protocol import evaluate

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 2026, 3407, 1234, 5678]


def reports(tmp_path):
    truth = ["HCC", "ICC", "HCC", "ICC", "HCC", "ICC", "HCC"]
    manifest = {"schema_version": 1, "classes": ["HCC", "ICC"],
                "patients": [{"patient_id": f"p{i}", "truth": label} for i, label in enumerate(truth)],
                "input_groups": [{"group_id": f"g{i}", "patient_id": f"p{i}", "phases": {
                    phase: {"image_ref": f"g{i}-{phase}.png", "ground_truth":
                            [{"label": label, "bbox_2d": [0, 0, 10, 10]}] if phase == "a" else []}
                    for phase in ("a", "p", "v")}} for i, label in enumerate(truth)]}
    result = []
    for correct, seed in zip(range(2, 7), SEEDS):
        rows = []
        for i, label in enumerate(truth):
            predicted = label if i < correct else ("ICC" if label == "HCC" else "HCC")
            rows.append({"group_id": f"g{i}", "completion_nll": 0.5,
                         "raw": json.dumps({"detections": [{"phase": "a", "label": predicted,
                                  "bbox_2d": [0, 0, 10, 10] if i == 0 else [20, 20, 30, 30]}]})})
        path = tmp_path / f"predictions-{seed}.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        payload = evaluate(manifest, path, [0.5])
        payload["run_identity"] = {"model_name": "Auxiliary", "seed": seed}
        result.append(payload)
    return result


def test_classification_rounds_before_mean_sd_but_detection_keeps_counts(tmp_path):
    data = reports(tmp_path)
    raw = summarize_runs(data, SEEDS)
    rounded = summarize_runs(data, SEEDS, classification_seed_decimals=4)
    expected = [0.2857, 0.4286, 0.5714, 0.7143, 0.8571]
    mean = math.fsum(expected)/5
    sd = math.sqrt(math.fsum((x-mean)**2 for x in expected)/4)
    row = rounded["summary"]["classification:accuracy"]
    assert row["values"] == expected
    assert row["mean"] == pytest.approx(mean, abs=1e-15)
    assert row["sd"] == pytest.approx(sd, abs=1e-15)
    assert rounded["unrounded_classification_seed_values"]["classification:accuracy"] == pytest.approx([i/7 for i in range(2, 7)])
    assert raw["summary"]["classification:accuracy"]["mean"] != row["mean"]
    assert rounded["classification_seed_rounding"] == "decimal_half_even"
    for name in raw["summary"]:
        if name.startswith("detection:"):
            assert rounded["summary"][name] == raw["summary"][name]
        else:
            assert all(value == round(value, 4) for value in rounded["summary"][name]["values"])
    detection = rounded["summary"]["detection:class_agnostic:iou=0.5:recall"]["values"]
    assert detection == pytest.approx([1/7]*5)
    assert detection != [0.1429]*5


def test_cli_exports_four_decimal_seed_values_and_full_precision_detection(tmp_path):
    paths = []
    for seed, data in zip(SEEDS, reports(tmp_path)):
        path = tmp_path/f"metrics-{seed}.json"
        path.write_text(json.dumps(data)); paths.append(str(path))
    output = tmp_path/"summary.json"
    subprocess.run([sys.executable, str(ROOT/"aggregation/summarize_seed_results.py"),
                    "--metrics", *paths, "--seeds", *map(str, SEEDS),
                    "--classification-seed-decimals", "4", "--output-json", str(output),
                    "--output-csv", str(tmp_path/"summary.csv")], check=True, capture_output=True, text=True)
    payload = json.loads(output.read_text())
    assert payload["summary"]["classification:accuracy"]["values"] == [0.2857, 0.4286, 0.5714, 0.7143, 0.8571]
    assert payload["summary"]["detection:class_agnostic:iou=0.5:recall"]["values"] == [1/7]*5


@pytest.mark.parametrize("decimals", [True, 0, 3, 4.0, "4", -1])
def test_unsupported_classification_rounding_is_rejected(tmp_path, decimals):
    with pytest.raises(ValueError, match="classification_seed_decimals"):
        summarize_runs(reports(tmp_path), SEEDS, classification_seed_decimals=decimals)
