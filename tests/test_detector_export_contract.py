"""Prepared-data contracts for detector export; all examples are artificial."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from preprocess.build_real961_detector_dataset import sha256_file


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "preprocess" / "build_real961_detector_dataset.py"


def lesion(label="HCC", phase="a", box=None):
    return {"phase": phase, "label": label, "bbox_2d": [100, 200, 400, 600] if box is None else box}


def background(phase):
    return {"phase": phase, "label": "no-lesion", "bbox_2d": [-1, -1, -1, -1]}


def source_row(data_root, split, label="HCC"):
    images = []
    for phase in ("a", "p", "v"):
        image = data_root / f"{split}_{phase}.png"
        # The exporter reads bytes/hashes only; clinical images and models are unnecessary.
        image.write_bytes(f"artificial {split} {phase}".encode())
        images.append({"type": "image", "image": image.name})
    return {
        "metadata": {"logical_key": f"group_{split}", "statistical_patient_id": f"patient_{split}",
                     "patient_id": f"source_{split}", "slice_index": 7, "positive": 1, "split": split},
        "messages": [
            {"role": "user", "content": images + [{"type": "text", "text": "Artificial test input"}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps({
                "detections": [lesion(label), background("p"), background("v")]
            })}]},
        ],
    }


def set_targets(row, detections):
    row["messages"][1]["content"][0]["text"] = json.dumps({"detections": detections})


def dataset(tmp_path, label="HCC"):
    data_root = tmp_path / "data"
    data_root.mkdir()
    return {split: [source_row(data_root, split, label)] for split in ("train", "validation", "test")}


def run_export(tmp_path, sources, *options):
    for split, rows in sources.items():
        (tmp_path / "data" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return subprocess.run([sys.executable, str(EXPORTER), "--root", str(tmp_path), *options],
                          text=True, capture_output=True, cwd=ROOT)


def export_dir(tmp_path):
    return tmp_path / "baselines_revision" / "detectors" / "data"


def assert_rejected(result, tmp_path, message):
    assert result.returncode != 0
    assert message in result.stderr
    assert not export_dir(tmp_path).exists(), "Invalid source must fail before writing exports"


def test_default_keeps_original_splits_and_mixed_phase_background(tmp_path):
    result = run_export(tmp_path, dataset(tmp_path))
    assert result.returncode == 0, result.stderr
    target = export_dir(tmp_path)
    yaml = (target / "dataset.yaml").read_text()
    assert "train: images/train\n" in yaml
    assert "val: images/validation\n" in yaml
    assert "test: images/test\n" in yaml
    audit = json.loads((target / "dataset_audit.json").read_text())
    assert audit["source_patient_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert audit["refit_trainval"] is False
    assert "test_not_used_for_training_or_selection" not in audit
    for split in ("train", "validation", "test"):
        assert audit["counts"][split] == {"patients": 1, "input_groups": 1, "phase_images": 3,
                                           "positive_phase_images": 1, "background_phase_images": 2, "lesions": 1}
    with (target / "phase_mapping.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    for row in rows:
        assert row["split"] == row["source_split"]
        assert Path(row["linked_image"]).is_symlink()
        assert row["source_image_sha256"] == sha256_file(Path(row["source_image"]))
        assert row["label_sha256"] == sha256_file(Path(row["label_file"]))
        assert Path(row["label_file"]).read_text() == ("0 0.25000000 0.40000000 0.30000000 0.40000000\n" if row["phase"] == "a" else "")


def test_public_small_cohort_and_string_assistant_without_private_counts(tmp_path):
    sources = dataset(tmp_path, "BCLM")
    sources["train"][0]["messages"][1]["content"] = sources["train"][0]["messages"][1]["content"][0]["text"]
    result = run_export(tmp_path, sources, "--data-mode", "5cls", "--expected-test-patients", "1")
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    assert audit["classes"] == ["HCC", "ICC", "HA_Hemangioma", "BCLM", "CRLM"]
    assert next((export_dir(tmp_path) / "labels" / "train").glob("*train_a.txt")).read_text().startswith("3 ")


def test_explicit_refit_has_no_validation_and_preserves_source_identity(tmp_path):
    result = run_export(tmp_path, dataset(tmp_path), "--refit-trainval")
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    yaml = (export_dir(tmp_path) / "dataset.yaml").read_text()
    assert "train: images/trainval\n" in yaml
    assert "val: null\n" in yaml
    assert audit["validation_split"] is None
    assert audit["counts"]["trainval"]["patients"] == 2
    assert not (export_dir(tmp_path) / "images" / "validation").exists()
    with (export_dir(tmp_path) / "phase_mapping.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["source_split"] for row in rows if row["split"] == "trainval"} == {"train", "validation"}


@pytest.mark.parametrize("first,second", [("train", "validation"), ("train", "test"), ("validation", "test")])
@pytest.mark.parametrize("field", ["patient_id", "statistical_patient_id"])
def test_cross_split_patient_overlap_rejected_even_for_refit(tmp_path, first, second, field):
    sources = dataset(tmp_path)
    sources[second][0]["metadata"][field] = sources[first][0]["metadata"][field]
    assert_rejected(run_export(tmp_path, sources, "--refit-trainval"), tmp_path, "Patient overlap")


@pytest.mark.parametrize("box", [[-1, 0, 10, 10], [0, 0, 1001, 10], [50, 0, 10, 10],
                                  [0, 10, 10, 0], [0, 0, 0, 10], [0, 0, float("nan"), 10],
                                  [0, 0, float("inf"), 10]])
def test_bad_coordinates_are_rejected_without_clipping_or_sorting(tmp_path, box):
    sources = dataset(tmp_path)
    set_targets(sources["test"][0], [lesion(box=box), background("p"), background("v")])
    assert_rejected(run_export(tmp_path, sources), tmp_path, "SchemaError")


@pytest.mark.parametrize("targets,error", [
    ([lesion(), background("a"), background("p"), background("v")], "both background and lesion"),
    ([lesion(), background("p"), background("p"), background("v")], "one [-1,-1,-1,-1]"),
    ([lesion(), background("p")], "Each detector target phase"),
    ([lesion("not HCC"), background("p"), background("v")], "Unknown label"),
])
def test_conflicting_missing_or_unknown_phase_targets_rejected(tmp_path, targets, error):
    sources = dataset(tmp_path)
    set_targets(sources["train"][0], targets)
    assert_rejected(run_export(tmp_path, sources), tmp_path, error)


def test_background_group_exports_empty_labels(tmp_path):
    sources = dataset(tmp_path)
    row = sources["train"][0]
    row["metadata"]["positive"] = "0"
    set_targets(row, [background(phase) for phase in ("a", "p", "v")])
    result = run_export(tmp_path, sources)
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    assert audit["counts"]["train"]["positive_phase_images"] == 0
    assert all(path.read_text() == "" for path in (export_dir(tmp_path) / "labels" / "train").iterdir())


def test_positive_flag_must_describe_group_not_every_phase(tmp_path):
    sources = dataset(tmp_path)
    sources["train"][0]["metadata"]["positive"] = 0
    assert_rejected(run_export(tmp_path, sources), tmp_path, "positive flag disagrees")


def test_source_image_reused_between_different_patient_ids_is_rejected(tmp_path):
    sources = dataset(tmp_path)
    sources["test"][0]["messages"][0]["content"][0]["image"] = "train_a.png"
    assert_rejected(run_export(tmp_path, sources), tmp_path, "Source image reused across")


def test_explicit_expected_count_is_checked(tmp_path):
    assert_rejected(run_export(tmp_path, dataset(tmp_path), "--expected-train-patients", "654"),
                    tmp_path, "Unexpected train patient count")


def test_duplicate_group_is_rejected(tmp_path):
    sources = dataset(tmp_path)
    sources["train"].append(sources["train"][0])
    assert_rejected(run_export(tmp_path, sources), tmp_path, "Duplicate input group")


@pytest.mark.parametrize("entry", ["evaluate_real961_predictions.py", "evaluate_sim999_predictions.py"])
def test_legacy_cli_requires_opt_in_before_reading_inputs(tmp_path, entry):
    result = subprocess.run([sys.executable, str(ROOT / "evaluation" / entry),
                             "--predictions", "missing.csv", "--metadata", "missing.csv",
                             "--output-dir", str(tmp_path / "out")], text=True, capture_output=True)
    assert result.returncode == 2
    assert "Legacy protocol disabled" in result.stderr
    assert not (tmp_path / "out").exists()


def test_legacy_opt_in_emits_warning(monkeypatch, capsys):
    from evaluation.evaluate_sim999_predictions import parse_args
    monkeypatch.setattr(sys, "argv", ["legacy", "--predictions", "old.csv", "--metadata", "old_meta.csv",
                                     "--output-dir", "old_output", "--allow-legacy-protocol"])
    args = parse_args()
    assert args.allow_legacy_protocol
    assert "GT-positive-only" in capsys.readouterr().err
