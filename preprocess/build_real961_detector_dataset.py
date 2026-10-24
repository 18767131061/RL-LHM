#!/usr/bin/env python3
"""Export phase-image detector inputs while preserving supplied patient splits.

The historical --root layout is supported. Fitting and validation stay separate
unless --refit-trainval is explicitly requested; a refit has no validation split.
See docs/detector_export.md for the source, annotation and provenance contracts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.schema import canonical_label, parse_completion


CLASS_SPACES = {
    "private": ("HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"),
    "5cls": ("HCC", "ICC", "HA_Hemangioma", "BCLM", "CRLM"),
    "3cls": ("HCC", "ICC", "HA_Hemangioma"),
}
CLASSES = list(CLASS_SPACES["private"])
CLASS_ID = {label: index for index, label in enumerate(CLASSES)}
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def answer_json(row: dict) -> dict:
    assistant = next(message for message in row["messages"] if message["role"] == "assistant")
    content = assistant["content"]
    text = content if isinstance(content, str) else next(
        segment["text"] for segment in content if segment["type"] == "text"
    )
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("detections"), list):
        raise RuntimeError("Invalid assistant detection JSON")
    return parsed


def nonempty_id(metadata: dict, field: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RuntimeError(f"Metadata {field} must be a nonempty, unpadded string")
    return value


def group_key(row: dict) -> tuple[str, int]:
    metadata = row["metadata"]
    key = nonempty_id(metadata, "logical_key")
    raw_index = metadata.get("slice_index")
    if isinstance(raw_index, bool) or not isinstance(raw_index, (str, int)):
        raise RuntimeError(f"Invalid slice_index for {key}")
    try:
        index = int(raw_index)
    except ValueError as error:
        raise RuntimeError(f"Invalid slice_index for {key}") from error
    if index < 0:
        raise RuntimeError(f"Invalid slice_index for {key}")
    return key, index


def validate_boundaries(sources: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Check original partitions, including before an explicitly requested refit."""
    owners = {field: {} for field in ("statistical_patient_id", "patient_id")}
    keys = set()
    patients = {}
    identity_map = {}
    reverse_map = {}
    for split, rows in sources.items():
        if not rows:
            raise RuntimeError(f"Empty source split: {split}")
        current = set()
        for row in rows:
            metadata = row["metadata"]
            patient = nonempty_id(metadata, "statistical_patient_id")
            original = nonempty_id(metadata, "patient_id") if "patient_id" in metadata else patient
            for field, identity in (("statistical_patient_id", patient), ("patient_id", original)):
                owner = owners[field].setdefault(identity, split)
                if owner != split:
                    raise RuntimeError(f"Patient overlap between {owner} and {split} ({field}={identity})")
            if identity_map.setdefault(original, patient) != patient or reverse_map.setdefault(patient, original) != original:
                raise RuntimeError("Patient identity mapping is not one-to-one")
            if metadata.get("split", split) != split:
                raise RuntimeError(f"Metadata split differs from supplied {split} source")
            key = group_key(row)
            if key in keys:
                raise RuntimeError(f"Duplicate input group: {key}")
            keys.add(key)
            current.add(patient)
        patients[split] = sorted(current)
    return patients


def plan_phase_rows(rows: list[dict], data_root: Path, output_root: Path, split: str,
                    classes=CLASSES, source_split: str | None = None) -> list[dict]:
    """Validate a complete group before deriving YOLO labels; never repair boxes."""
    output = []
    class_ids = {label: index for index, label in enumerate(classes)}
    seen_names = set()
    for row in rows:
        metadata = row["metadata"]
        logical_key, slice_index = group_key(row)
        patient = nonempty_id(metadata, "statistical_patient_id")
        user = next(message for message in row["messages"] if message["role"] == "user")
        segments = [segment for segment in user["content"] if segment["type"] == "image"]
        if len(segments) != 3:
            raise RuntimeError(f"Expected three images for {logical_key}")
        parsed = parse_completion(json.dumps(answer_json(row)), classes)
        annotated_phases = {item.phase for item in parsed.detections} | set(parsed.background_phases)
        if annotated_phases != {"a", "p", "v"}:
            raise RuntimeError(f"Each detector target phase needs a lesion or explicit background: {logical_key}")
        positive = metadata.get("positive")
        if positive not in (0, 1, "0", "1"):
            raise RuntimeError(f"Invalid group positive flag: {logical_key}")
        if bool(int(positive)) != bool(parsed.detections):
            raise RuntimeError(f"Group positive flag disagrees with phase targets: {logical_key}")
        for phase, segment in zip(("a", "p", "v"), segments):
            source = (data_root / segment["image"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            token = hashlib.sha256(f"{logical_key}:{slice_index}:{phase}".encode()).hexdigest()[:20]
            name = f"{token}_{source.name}"
            if name in seen_names:
                raise RuntimeError(f"Detector filename collision: {name}")
            seen_names.add(name)
            image_destination = output_root / "images" / split / name
            label_destination = output_root / "labels" / split / f"{Path(name).stem}.txt"
            label_lines = []
            labels = []
            for detection in parsed.detections:
                if detection.phase != phase:
                    continue
                x1, y1, x2, y2 = detection.box
                label_lines.append(
                    f"{class_ids[detection.label]} {(x1 + x2) / 2000:.8f} {(y1 + y2) / 2000:.8f} "
                    f"{(x2 - x1) / 1000:.8f} {(y2 - y1) / 1000:.8f}"
                )
                labels.append(detection.label)
            output.append({
                "split": split, "source_split": source_split or split,
                "logical_key": logical_key, "statistical_patient_id": patient,
                "slice_index": slice_index, "phase": phase,
                "positive": int(bool(label_lines)), "lesion_count": len(label_lines),
                "labels": json.dumps(labels, separators=(",", ":")),
                "source_image": str(source), "linked_image": str(image_destination),
                "label_file": str(label_destination),
                "_label_content": "\n".join(label_lines) + ("\n" if label_lines else ""),
            })
    return output


def write_phase_rows(planned: list[dict]) -> list[dict]:
    output = []
    for item in planned:
        row = dict(item)
        content = row.pop("_label_content")
        destination = Path(row["linked_image"])
        label = Path(row["label_file"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink() or label.exists():
            raise RuntimeError(f"Refusing to overwrite detector export: {destination}")
        os.symlink(row["source_image"], destination)
        label.write_text(content, encoding="utf-8")
        row["label_sha256"] = sha256_file(label)
        output.append(row)
    return output


def phase_rows(rows: list[dict], data_root: Path, output_root: Path, split: str, classes=CLASSES) -> list[dict]:
    """Compatibility helper; the CLI also validates all source split boundaries."""
    return write_phase_rows(plan_phase_rows(rows, data_root, output_root, split, classes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Legacy experiment root; defaults to ROOT/data inputs")
    parser.add_argument("--data-root", help="Directory containing train/validation/test.jsonl and source images")
    parser.add_argument("--output-dir", help="New or empty export directory")
    parser.add_argument("--data-mode", choices=tuple(CLASS_SPACES), default="private")
    parser.add_argument("--classes", help="Explicit comma-separated ordered labels; overrides --data-mode")
    parser.add_argument("--refit-trainval", action="store_true", help="Explicitly combine fitting and validation; export has no validation set")
    for split in SPLITS:
        parser.add_argument(f"--expected-{split}-patients", type=int, help="Optional count assertion on the original source split")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else root / "data"
    output_root = Path(args.output_dir).resolve() if args.output_dir else root / "baselines_revision" / "detectors" / "data"
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise RuntimeError(f"Export directory must be new or empty: {output_root}")
    classes = tuple(canonical_label(label.strip()) for label in args.classes.split(",")) if args.classes else CLASS_SPACES[args.data_mode]
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise RuntimeError("Expected at least two distinct canonical class labels")
    paths = {split: data_root / f"{split}.jsonl" for split in SPLITS}
    hashes = {split: sha256_file(path) for split, path in paths.items()}
    sources = {split: load_jsonl(path) for split, path in paths.items()}
    source_patients = validate_boundaries(sources)
    for split in SPLITS:
        expected = getattr(args, f"expected_{split}_patients")
        if expected is not None and expected != len(source_patients[split]):
            raise RuntimeError(f"Unexpected {split} patient count: {len(source_patients[split])}; expected {expected}")
    planned = []
    for source_split, source_rows in sources.items():
        split = "trainval" if args.refit_trainval and source_split != "test" else source_split
        planned.extend(plan_phase_rows(source_rows, data_root, output_root, split, classes, source_split))
    image_owners = {}
    image_hashes = {}
    for row in planned:
        source = row["source_image"]
        owner = image_owners.setdefault(source, row["source_split"])
        if owner != row["source_split"]:
            raise RuntimeError(f"Source image reused across {owner} and {row['source_split']}: {source}")
        if source not in image_hashes:
            image_hashes[source] = sha256_file(Path(source))
        row["source_image_sha256"] = image_hashes[source]
    if any(sha256_file(paths[split]) != hashes[split] for split in SPLITS):
        raise RuntimeError("Source JSONL changed during export preparation")
    rows = write_phase_rows(planned)
    mapping = output_root / "phase_mapping.csv"
    with mapping.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text("\n".join([
        f"path: {json.dumps(str(output_root))}",
        f"train: images/{'trainval' if args.refit_trainval else 'train'}",
        "val: null" if args.refit_trainval else "val: images/validation",
        "test: images/test", "names:",
        *[f"  {index}: {json.dumps(label)}" for index, label in enumerate(classes)], "",
    ]), encoding="utf-8")
    output_splits = sorted({row["split"] for row in rows})
    counts = {}
    for split in output_splits:
        selected = [row for row in rows if row["split"] == split]
        counts[split] = {
            "patients": len({row["statistical_patient_id"] for row in selected}),
            "input_groups": len({(row["logical_key"], row["slice_index"]) for row in selected}),
            "phase_images": len(selected),
            "positive_phase_images": sum(row["positive"] for row in selected),
            "background_phase_images": sum(not row["positive"] for row in selected),
            "lesions": sum(row["lesion_count"] for row in selected),
        }
    audit = {
        "schema_version": "detector-export-v2", "status": "export_complete",
        "classes": list(classes), "refit_trainval": args.refit_trainval,
        "validation_split": None if args.refit_trainval else "validation",
        "source_patient_ids": source_patients,
        "source_patient_counts": {split: len(ids) for split, ids in source_patients.items()},
        "source_patient_overlap": 0, "counts": counts,
        "source_files": {split: {"path": str(paths[split]), "sha256": hashes[split]} for split in SPLITS},
        "exporter_sha256": sha256_file(Path(__file__)),
        "annotation_contract": "canonical parse_completion; all three phases explicitly annotated; no coordinate repair",
        "image_storage": "symlinks; verify source_image_sha256 before training",
        "scope": "Dataset export only; downstream fitting, selection and test usage are not verified.",
        "outputs": {
            "dataset_yaml": str(dataset_yaml), "dataset_yaml_sha256": sha256_file(dataset_yaml),
            "mapping": str(mapping), "mapping_sha256": sha256_file(mapping),
        },
    }
    (output_root / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
