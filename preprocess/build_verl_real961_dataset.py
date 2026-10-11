#!/usr/bin/env python3
"""Convert real961 JSONL while preserving supplied train/validation boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


CLASSES = {"HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unpack(row: dict, data_root: Path) -> dict:
    metadata = row["metadata"]
    is_positive = bool(int(metadata["positive"]))
    messages = row["messages"]
    user = next(message for message in messages if message["role"] == "user")
    assistant = next(message for message in messages if message["role"] == "assistant")
    prompt = next(segment["text"] for segment in user["content"] if segment["type"] == "text")
    image_paths = [
        str((data_root / segment["image"]).resolve())
        for segment in user["content"]
        if segment["type"] == "image"
    ]
    if len(image_paths) != 3 or not all(Path(path).is_file() for path in image_paths):
        raise RuntimeError(f"Expected three existing images for {metadata['logical_key']}")
    answer = next(segment["text"] for segment in assistant["content"] if segment["type"] == "text")
    parsed = json.loads(answer)
    gt_dets = []
    background_phases = []
    for detection in parsed.get("detections", []):
        label = detection.get("label")
        box = detection.get("bbox_2d")
        if label in CLASSES and isinstance(box, list) and len(box) == 4:
            gt_dets.append({"phase": detection["phase"], "label": label, "bbox": box})
        elif label == "no-lesion" and box == [-1, -1, -1, -1]:
            background_phases.append(str(detection.get("phase", "")).lower())
    if is_positive and not gt_dets:
        raise RuntimeError(f"Positive sample has no lesion detections: {metadata['logical_key']}")
    if not is_positive and (gt_dets or sorted(background_phases) != ["a", "p", "v"]):
        raise RuntimeError(f"Invalid background target: {metadata['logical_key']}")
    labels = Counter(item["label"] for item in gt_dets)
    gt_major = max(CLASSES, key=lambda label: (labels[label], label)) if gt_dets else None
    uid = f"{metadata['logical_key']}:{int(metadata['slice_index']):04d}"
    return {
        "uid": uid,
        "prompt": [{"role": "user", "content": "<image>\n<image>\n<image>\n" + prompt}],
        "images": [{"image": path, "max_pixels": 262144} for path in image_paths],
        "gt_dets": gt_dets,
        "background_phases": background_phases,
        "gt_major": gt_major,
        "metadata": metadata,
    }


def write_verl(rows: list[dict], output: Path, source: str, class_counts: dict[str, int]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            ground_truth = json.dumps(
                {
                    "gt_dets": row["gt_dets"],
                    "background_phases": row["background_phases"],
                    "gt_major": row["gt_major"],
                },
                ensure_ascii=False,
            )
            extra_info = {
                "index": index,
                "sample_idx": index,
                "sample_uid": row["uid"],
                "patient_id": row["metadata"]["statistical_patient_id"],
                "data_mode": "private",
                "class_counts": class_counts,
                "visual_z": None,
            }
            payload = {
                "data_source": source,
                "uid": row["uid"],
                "prompt": row["prompt"],
                "images": row["images"],
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "reward_model": {"ground_truth": ground_truth, "extra_info": extra_info},
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def class_counts_for(rows: list[dict]) -> dict[str, int]:
    """Count positive input groups by the current converter's majority label."""
    return dict(sorted(Counter(row["gt_major"] for row in rows if row["gt_major"] is not None).items()))


def row_summary(rows: list[dict]) -> dict:
    return {
        "input_groups": len(rows),
        "positive_input_groups": sum(bool(row["gt_dets"]) for row in rows),
        "background_input_groups": sum(not row["gt_dets"] for row in rows),
        "statistical_patients": len({row["metadata"]["statistical_patient_id"] for row in rows}),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_boundaries(train: list[dict], validation: list[dict]) -> None:
    # Check the identity exported to VeRL as well as the source patient ID.
    for field in ("patient_id", "statistical_patient_id"):
        if {row["metadata"][field] for row in train} & {
            row["metadata"][field] for row in validation
        }:
            raise RuntimeError(f"Patient overlap between train and validation sources ({field})")
    uids = [row["uid"] for row in train + validation]
    if len(uids) != len(set(uids)):
        raise RuntimeError("Duplicate input-group UID in supplied train/validation sources")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--refit-train-validation",
        action="store_true",
        help=(
            "Additionally write a named combined refit file after model selection; "
            "never replace the train-only file"
        ),
    )
    args = parser.parse_args()
    if args.refit_train_validation and not args.validation_jsonl:
        parser.error("--refit-train-validation requires --validation-jsonl")

    data_root = Path(args.data_root).resolve()
    train_path = Path(args.train_jsonl).resolve()
    validation_path = Path(args.validation_jsonl).resolve() if args.validation_jsonl else None
    train = [unpack(row, data_root) for row in load_jsonl(train_path)]
    validation = (
        [unpack(row, data_root) for row in load_jsonl(validation_path)]
        if validation_path is not None else []
    )
    if not train:
        raise RuntimeError("The supplied train source is empty")
    if validation_path is not None and not validation:
        raise RuntimeError("The supplied validation source is empty")
    validate_source_boundaries(train, validation)
    class_counts = class_counts_for(train)
    output_dir = Path(args.output_dir).resolve()
    outputs = {}

    def emit(rows: list[dict], stem: str, counts: dict[str, int], role: str, population: str) -> None:
        path = output_dir / f"{stem}.jsonl"
        write_verl(rows, path, stem, counts)
        outputs[stem] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "role": role,
            **row_summary(rows),
            "class_counts": counts,
            "class_counts_population": population,
        }

    emit(train, "real961_rl_train", class_counts, "training_only", "supplied_train_source")
    if validation:
        emit(validation, "real961_rl_validation", class_counts, "held_out_validation", "supplied_train_source")
    monitor = validation if validation else train[: min(len(train), 64)]
    emit(
        monitor,
        "real961_rl_monitor",
        class_counts,
        "held_out_validation_monitor" if validation else "training_subset_diagnostic_not_validation",
        "supplied_train_source",
    )
    if args.refit_train_validation:
        refit = train + validation
        emit(
            refit,
            "real961_rl_refit_train_validation",
            class_counts_for(refit),
            "explicit_combined_refit_after_selection",
            "supplied_train_and_validation_sources",
        )
    audit = {
        "contract_version": 2,
        "split_policy": "preserve_supplied_train_validation_boundaries",
        "sources": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path), **row_summary(train)},
            "validation": (
                None if validation_path is None else
                {"path": str(validation_path), "sha256": sha256_file(validation_path), **row_summary(validation)}
            ),
        },
        "outputs": outputs,
        "class_counts_definition": {
            "unit": "positive_three_phase_input_group",
            "label": "gt_major (majority lesion-record label; lexical maximum breaks ties)",
            "background_groups_included": False,
            "zero_count_classes_included": False,
        },
        "refit_requested": args.refit_train_validation,
        "validation_held_out_from_refit": False if args.refit_train_validation else None,
        "source_partition_provenance": "caller_supplied",
        "visual_z_status": "not_extracted",
    }
    with (output_dir / "audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
