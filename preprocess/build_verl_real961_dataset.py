#!/usr/bin/env python3
"""Convert real961 multimodal JSONL into VERL development datasets."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    train = [unpack(row, data_root) for row in load_jsonl(Path(args.train_jsonl).resolve())]
    validation = (
        [unpack(row, data_root) for row in load_jsonl(Path(args.validation_jsonl).resolve())]
        if args.validation_jsonl
        else []
    )
    development = train + validation
    if validation and {row["metadata"]["patient_id"] for row in train} & {
        row["metadata"]["patient_id"] for row in validation
    }:
        raise RuntimeError("Patient overlap between original train and validation sources")
    class_counts = dict(
        sorted(Counter(row["gt_major"] for row in development if row["gt_major"] is not None).items())
    )
    output_dir = Path(args.output_dir).resolve()
    write_verl(development, output_dir / "real961_rl_train.jsonl", "real961_rl_train", class_counts)
    monitor = validation if validation else train[: min(len(train), 64)]
    write_verl(monitor, output_dir / "real961_rl_monitor.jsonl", "real961_rl_monitor", class_counts)
    audit = {
        "development_input_groups": len(development),
        "development_positive_input_groups": sum(bool(row["gt_dets"]) for row in development),
        "development_background_input_groups": sum(not row["gt_dets"] for row in development),
        "monitor_input_groups": len(monitor),
        "development_patients": len({row["metadata"]["patient_id"] for row in development}),
        "monitor_patients": len({row["metadata"]["patient_id"] for row in monitor}),
        "class_positive_slice_counts": class_counts,
        "test_data_used": False,
        "visual_z_status": "not_extracted",
    }
    with (output_dir / "audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
