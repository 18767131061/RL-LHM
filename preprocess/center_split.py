from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping


def _patient_record(
    row: Mapping[str, object],
    *,
    patient_key: str,
    class_key: str,
    center_key: str,
) -> dict[str, str]:
    try:
        return {
            "patient_id": str(row[patient_key]),
            "class": str(row[class_key]),
            "center_id": str(row[center_key]),
        }
    except KeyError as error:
        raise ValueError(f"Missing manifest field: {error.args[0]}") from error


def stratified_tuning_split(
    rows: Iterable[Mapping[str, object]],
    *,
    source_center_id: str,
    tuning_fraction: float,
    seed: int,
    patient_key: str = "patient_id",
    class_key: str = "class",
    center_key: str = "center_id",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a deterministic patient-level fit/tuning split from one center."""
    if not 0 < tuning_fraction < 0.5:
        raise ValueError("tuning_fraction must be between 0 and 0.5")

    patients: dict[str, dict[str, str]] = {}
    for row in rows:
        record = _patient_record(
            row,
            patient_key=patient_key,
            class_key=class_key,
            center_key=center_key,
        )
        patient_id = record["patient_id"]
        if patient_id in patients and patients[patient_id] != record:
            raise ValueError(f"Inconsistent metadata for patient {patient_id}")
        patients[patient_id] = record

    if not patients:
        raise ValueError("No patients were supplied")
    centers = {record["center_id"] for record in patients.values()}
    if centers != {source_center_id}:
        raise ValueError(f"Expected source center {source_center_id}, found {sorted(centers)}")

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in patients.values():
        by_class[record["class"]].append(record)

    fit: list[dict[str, str]] = []
    tuning: list[dict[str, str]] = []
    for label in sorted(by_class):
        records = by_class[label]
        if len(records) < 2:
            raise ValueError(f"Class {label} needs at least two patients for fit/tuning")
        ordered = sorted(
            records,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item['patient_id']}".encode("utf-8")
            ).hexdigest(),
        )
        tuning_count = min(
            len(ordered) - 1,
            max(1, round(len(ordered) * tuning_fraction)),
        )
        tuning.extend(ordered[:tuning_count])
        fit.extend(ordered[tuning_count:])

    return (
        sorted(fit, key=lambda item: item["patient_id"]),
        sorted(tuning, key=lambda item: item["patient_id"]),
    )


def audit_center_isolation(
    rows: Iterable[Mapping[str, object]],
    *,
    source_center_id: str,
    target_center_id: str,
    patient_key: str = "patient_id",
    center_key: str = "center_id",
    split_key: str = "split",
    source_split: str = "train",
    target_split: str = "test",
) -> dict[str, object]:
    """Fail if a patient or center crosses the frozen source/target split."""
    by_split: dict[str, set[str]] = defaultdict(set)
    centers: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        try:
            split = str(row[split_key])
            patient_id = str(row[patient_key])
            center_id = str(row[center_key])
        except KeyError as error:
            raise ValueError(f"Missing manifest field: {error.args[0]}") from error
        by_split[split].add(patient_id)
        centers[split][center_id] += 1

    overlap = by_split[source_split] & by_split[target_split]
    if overlap:
        raise ValueError(f"Patient overlap across center split: {sorted(overlap)[:5]}")
    if set(centers[source_split]) != {source_center_id}:
        raise ValueError(f"Source split contains unexpected centers: {dict(centers[source_split])}")
    if set(centers[target_split]) != {target_center_id}:
        raise ValueError(f"Target split contains unexpected centers: {dict(centers[target_split])}")

    return {
        "source_center_id": source_center_id,
        "target_center_id": target_center_id,
        "source_patients": len(by_split[source_split]),
        "target_patients": len(by_split[target_split]),
        "patient_overlap": 0,
    }
