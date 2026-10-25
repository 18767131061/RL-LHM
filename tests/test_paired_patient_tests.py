import csv
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "evaluation"))

import paired_patient_tests  # noqa: E402


def write_predictions(path, predictions):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["logical_key", "truth", "prediction"])
        writer.writeheader()
        writer.writerows(predictions)


def test_paired_outputs_include_macro_recall_and_provenance(tmp_path, monkeypatch):
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    manifest = tmp_path / "test_manifest.csv"
    output = tmp_path / "paired.json"
    detail = tmp_path / "paired_detail.csv"
    rows = [
        {"logical_key": "P01", "truth": "A", "prediction": "A"},
        {"logical_key": "P02", "truth": "A", "prediction": "B"},
        {"logical_key": "P03", "truth": "B", "prediction": "A"},
        {"logical_key": "P04", "truth": "B", "prediction": "B"},
    ]
    write_predictions(reference, rows)
    write_predictions(
        candidate,
        [
            {"logical_key": "P01", "truth": "A", "prediction": "A"},
            {"logical_key": "P02", "truth": "A", "prediction": "A"},
            {"logical_key": "P03", "truth": "B", "prediction": "B"},
            {"logical_key": "P04", "truth": "B", "prediction": "B"},
        ],
    )
    manifest.write_text("logical_key\nP01\nP02\nP03\nP04\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paired_patient_tests.py",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--reference-name",
            "vanilla",
            "--candidate-name",
            "full",
            "--classes",
            "A,B",
            "--output",
            str(output),
            "--paired-detail-output",
            str(detail),
            "--cohort",
            "private",
            "--model-seed",
            "42",
            "--aggregation-rule",
            "majority_nll_tiebreak",
            "--evaluation-version",
            "v1",
            "--test-manifest",
            str(manifest),
            "--strict-provenance",
            "--bootstrap",
            "50",
            "--permutations",
            "50",
        ],
    )
    paired_patient_tests.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["candidate_minus_reference"]) == {
        "accuracy",
        "macro_recall",
        "macro_f1",
    }
    assert set(payload["paired_bootstrap_95_ci"]) == {
        "accuracy",
        "macro_recall",
        "macro_f1",
    }
    assert set(payload["paired_permutation_two_sided_p"]) == {
        "accuracy",
        "macro_recall",
        "macro_f1",
    }
    assert payload["provenance"]["model_seed"] == 42
    assert payload["provenance"]["test_manifest"]["sha256"]
    assert payload["paired_detail"]["sha256"]
    assert len(list(csv.DictReader(detail.open(encoding="utf-8")))) == 4
