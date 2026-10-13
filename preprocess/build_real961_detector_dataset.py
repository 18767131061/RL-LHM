#!/usr/bin/env python3
"""Export private development phase images from prepared real961 JSONL.

The 654+117 patient union is assigned to both train/val YAML entries. This
preparation export does not specify any detector's actual fitting population
or independently validate training/checkpoint selection. Test patients remain
separate. See docs/paper_comparison_settings.md for downstream interfaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


CLASSES = ["HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"]
CLASS_ID = {label: index for index, label in enumerate(CLASSES)}


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
    if not isinstance(parsed, dict) or not isinstance(parsed.get("detections", []), list):
        raise RuntimeError("Invalid assistant detection JSON")
    return parsed


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite detector image: {destination}")
    os.symlink(source, destination)


def phase_rows(rows: list[dict], data_root: Path, output_root: Path, split: str) -> list[dict]:
    output = []
    seen_names = set()
    for row in rows:
        metadata = row["metadata"]
        user = next(message for message in row["messages"] if message["role"] == "user")
        segments = [segment for segment in user["content"] if segment["type"] == "image"]
        if len(segments) != 3:
            raise RuntimeError(f"Expected three images for {metadata['logical_key']}")
        detections = answer_json(row)["detections"]
        for phase, segment in zip(("a", "p", "v"), segments):
            source = (data_root / segment["image"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            token = hashlib.sha256(
                f"{metadata['logical_key']}:{int(metadata['slice_index'])}:{phase}".encode()
            ).hexdigest()[:20]
            name = f"{token}_{source.name}"
            if name in seen_names:
                raise RuntimeError(f"Detector filename collision: {name}")
            seen_names.add(name)
            image_destination = output_root / "images" / split / name
            label_destination = output_root / "labels" / split / f"{Path(name).stem}.txt"
            link_image(source, image_destination)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            label_lines = []
            labels = []
            for detection in detections:
                if str(detection.get("phase", "")).lower() != phase:
                    continue
                label = detection.get("label")
                box = detection.get("bbox_2d")
                if label == "no-lesion":
                    if bool(metadata["positive"]):
                        raise RuntimeError(
                            f"Positive sample contains no-lesion target: {metadata['logical_key']}"
                        )
                    continue
                if label not in CLASS_ID or not isinstance(box, list) or len(box) != 4:
                    raise RuntimeError(f"Invalid detector target for {metadata['logical_key']}")
                x1, y1, x2, y2 = [max(0.0, min(1000.0, float(value))) for value in box]
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                if x2 <= x1 or y2 <= y1:
                    raise RuntimeError(f"Degenerate detector target for {metadata['logical_key']}")
                label_lines.append(
                    f"{CLASS_ID[label]} {(x1 + x2) / 2000:.8f} {(y1 + y2) / 2000:.8f} "
                    f"{(x2 - x1) / 1000:.8f} {(y2 - y1) / 1000:.8f}"
                )
                labels.append(label)
            label_destination.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            output.append(
                {
                    "split": split,
                    "logical_key": metadata["logical_key"],
                    "statistical_patient_id": metadata["statistical_patient_id"],
                    "slice_index": int(metadata["slice_index"]),
                    "phase": phase,
                    "positive": int(bool(label_lines)),
                    "labels": json.dumps(labels, separators=(",", ":")),
                    "source_image": str(source),
                    "linked_image": str(image_destination),
                    "label_file": str(label_destination),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    data_root = root / "data"
    output_root = root / "baselines_revision" / "detectors" / "data"
    train_path = data_root / "train.jsonl"
    validation_path = data_root / "validation.jsonl"
    test_path = data_root / "test.jsonl"
    development = load_jsonl(train_path) + load_jsonl(validation_path)
    test = load_jsonl(test_path)
    development_patients = {row["metadata"]["statistical_patient_id"] for row in development}
    test_patients = {row["metadata"]["statistical_patient_id"] for row in test}
    if development_patients & test_patients:
        raise RuntimeError("Patient overlap between detector development and test splits")
    if (len(development_patients), len(test_patients)) != (771, 190):
        raise RuntimeError("Unexpected detector patient counts")

    rows = [
        *phase_rows(development, data_root, output_root, "development"),
        *phase_rows(test, data_root, output_root, "test"),
    ]
    mapping = output_root / "phase_mapping.csv"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    with mapping.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root}",
                "train: images/development",
                "val: images/development",
                "names:",
                *[f"  {index}: {label}" for index, label in enumerate(CLASSES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    by_split = Counter(row["split"] for row in rows)
    positives = Counter(row["split"] for row in rows if row["positive"])
    audit = {
        "status": "passed",
        "classes": CLASSES,
        "development_patients": len(development_patients),
        "test_patients": len(test_patients),
        "patient_overlap": 0,
        "phase_images": dict(by_split),
        "positive_phase_images": dict(positives),
        "test_not_used_for_training_or_selection": True,
        "training_validation_disabled": True,
        "source_hashes": {
            path.name: sha256_file(path) for path in (train_path, validation_path, test_path)
        },
        "outputs": {
            "dataset_yaml": str(dataset_yaml),
            "dataset_yaml_sha256": sha256_file(dataset_yaml),
            "mapping": str(mapping),
            "mapping_sha256": sha256_file(mapping),
        },
    }
    audit_path = output_root / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
