import csv

import pytest

from data.manifest import load_dataset_manifest, manifest_summary


FIELDS = ["sample_id", "patient_id", "center_id", "split", "label", "asset_ref"]


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_summary_and_patient_split_gate(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(
        path,
        [
            {"sample_id": "s1", "patient_id": "p1", "center_id": "A", "split": "train", "label": "HCC", "asset_ref": "images/a.png"},
            {"sample_id": "s2", "patient_id": "p2", "center_id": "B", "split": "test", "label": "ICC", "asset_ref": "images/b.png"},
        ],
    )
    summary = manifest_summary(load_dataset_manifest(path))
    assert summary["samples"] == 2
    assert summary["patients_crossing_splits"] == []
    assert summary["by_center"] == {"A": 1, "B": 1}


def test_manifest_rejects_absolute_assets_and_duplicate_samples(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(
        path,
        [{"sample_id": "s1", "patient_id": "p1", "center_id": "A", "split": "train", "label": "HCC", "asset_ref": "/private/a.png"}],
    )
    with pytest.raises(ValueError, match="release-relative"):
        load_dataset_manifest(path)

    write_manifest(
        path,
        [
            {"sample_id": "s1", "patient_id": "p1", "center_id": "A", "split": "train", "label": "HCC", "asset_ref": "a.png"},
            {"sample_id": "s1", "patient_id": "p2", "center_id": "A", "split": "train", "label": "ICC", "asset_ref": "b.png"},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset_manifest(path)
