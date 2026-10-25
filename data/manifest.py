"""De-identified dataset-manifest schema and validation."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    patient_id: str
    center_id: str
    split: str
    label: str
    asset_ref: str


REQUIRED_FIELDS = tuple(DatasetRecord.__dataclass_fields__)


def _relative_asset_reference(value: str) -> str:
    reference = PurePosixPath(value)
    if reference.is_absolute() or ".." in reference.parts or value.strip() == "":
        raise ValueError("asset_ref must be a non-empty release-relative POSIX path")
    return reference.as_posix()


def load_dataset_manifest(path: str | Path) -> list[DatasetRecord]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset manifest is missing fields: {sorted(missing)}")
        records = [
            DatasetRecord(
                sample_id=str(row["sample_id"]).strip(),
                patient_id=str(row["patient_id"]).strip(),
                center_id=str(row["center_id"]).strip(),
                split=str(row["split"]).strip(),
                label=str(row["label"]).strip(),
                asset_ref=_relative_asset_reference(str(row["asset_ref"]).strip()),
            )
            for row in reader
        ]
    if not records:
        raise ValueError("Dataset manifest has no records")
    if any(not all((row.sample_id, row.patient_id, row.center_id, row.split, row.label)) for row in records):
        raise ValueError("Dataset manifest identifiers and labels must be non-empty")
    duplicates = [key for key, count in Counter(row.sample_id for row in records).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample_id values: {duplicates[:5]}")
    return records


def manifest_summary(records: list[DatasetRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("Cannot summarize an empty manifest")
    by_split = Counter(row.split for row in records)
    by_center = Counter(row.center_id for row in records)
    by_class = Counter(row.label for row in records)
    patient_splits: dict[str, set[str]] = {}
    for row in records:
        patient_splits.setdefault(row.patient_id, set()).add(row.split)
    return {
        "samples": len(records),
        "patients": len(patient_splits),
        "by_split": dict(sorted(by_split.items())),
        "by_center": dict(sorted(by_center.items())),
        "by_class": dict(sorted(by_class.items())),
        "patients_crossing_splits": sorted(
            patient for patient, splits in patient_splits.items() if len(splits) > 1
        ),
    }
