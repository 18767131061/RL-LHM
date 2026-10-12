#!/usr/bin/env python3
"""Convert real961 JSONL while preserving supplied train/validation boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.schema import canonical_label, parse_completion
from preprocess.center_split import stratified_tuning_split

CLASS_SPACES = {
    "private": ("HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"),
    "5cls": ("HCC", "ICC", "HA_Hemangioma", "BCLM", "CRLM"),
    "3cls": ("HCC", "ICC", "HA_Hemangioma"),
}
CLASSES = CLASS_SPACES["private"]


def center_id(metadata: dict) -> str | None:
    values = [metadata[key] for key in ("center_id", "analysis_center") if metadata.get(key) is not None]
    if not values:
        return None
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in values) or len(set(values)) != 1:
        raise RuntimeError("Center metadata must contain consistent nonempty center_id/analysis_center")
    return values[0]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unpack(row: dict, data_root: Path, classes=CLASSES) -> dict:
    metadata = row["metadata"]
    center_id(metadata)
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
    parsed = parse_completion(answer, classes)
    gt_dets = [{"phase": item.phase, "label": item.label, "bbox": list(item.box)}
               for item in parsed.detections]
    background_phases = list(parsed.background_phases)
    if is_positive and not gt_dets:
        raise RuntimeError(f"Positive sample has no lesion detections: {metadata['logical_key']}")
    if not is_positive and (gt_dets or sorted(background_phases) != ["a", "p", "v"]):
        raise RuntimeError(f"Invalid background target: {metadata['logical_key']}")
    labels = Counter(item["label"] for item in gt_dets)
    gt_major = max(classes, key=lambda label: (labels[label], label)) if gt_dets else None
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


def write_verl(rows: list[dict], output: Path, source: str, class_counts: dict[str, int], data_mode="private") -> None:
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
                "data_mode": data_mode,
                "class_counts": class_counts,
                "visual_z": None,
            }
            center = center_id(row["metadata"])
            if center is not None:
                extra_info["center_id"] = center
            if row["metadata"].get("class") is not None:
                extra_info["patient_class"] = canonical_label(row["metadata"]["class"])
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


def loco_partition(rows, *, source_center, held_out_center, fraction, seed, classes, reserved_test):
    """Split explicit patient diagnoses; group-level majority labels never stratify patients."""
    if not source_center or not held_out_center or source_center == held_out_center:
        raise RuntimeError("LOCO requires different source and held-out centers")
    test = json.loads(Path(reserved_test).read_text())
    patients = test.get("patients", [])
    if test.get("split") != "test" or not patients:
        raise RuntimeError("LOCO requires a nonempty identity-only reserved-test manifest")
    test_ids = [item.get("patient_id") for item in patients]
    if any(not isinstance(value, str) or not value.strip() for value in test_ids) or len(test_ids) != len(set(test_ids)) or any(item.get("center_id") != held_out_center for item in patients):
        raise RuntimeError("Reserved-test identities must be unique and from held_out_center")
    records = []
    source_to_stat = {}
    for row in rows:
        meta = row["metadata"]
        if center_id(meta) != source_center:
            raise RuntimeError("LOCO source rows must all belong to source_center")
        if "class" not in meta:
            raise RuntimeError("LOCO stratification needs explicit patient reference metadata.class")
        label = canonical_label(meta["class"])
        if label not in classes:
            raise RuntimeError("Patient reference diagnosis is outside the active source classes")
        patient = meta["statistical_patient_id"]
        original = meta["patient_id"]
        if original in source_to_stat and source_to_stat[original] != patient:
            raise RuntimeError("One source patient maps to multiple statistical patient IDs")
        source_to_stat[original] = patient
        if patient in test_ids or original in test_ids:
            raise RuntimeError("LOCO source and reserved-test patients overlap")
        records.append({"patient_id": patient, "class": label, "center_id": source_center})
    fit, tuning = stratified_tuning_split(records, source_center_id=source_center,
                                        tuning_fraction=fraction, seed=seed)
    fit_ids = {item["patient_id"] for item in fit}
    train = [row for row in rows if row["metadata"]["statistical_patient_id"] in fit_ids]
    validation = [row for row in rows if row["metadata"]["statistical_patient_id"] not in fit_ids]
    return train, validation, {"source_center": source_center, "held_out_center": held_out_center,
        "split_seed": seed, "tuning_fraction": fraction,
        "patient_stratification_label": "metadata.class (reference diagnosis, not gt_major)",
        "train_patients": len(fit), "validation_patients": len(tuning),
        "train_patients_by_class": dict(sorted(Counter(item["class"] for item in fit).items())),
        "validation_patients_by_class": dict(sorted(Counter(item["class"] for item in tuning).items())),
        "reserved_test_manifest": str(Path(reserved_test).resolve()),
        "reserved_test_sha256": sha256_file(Path(reserved_test))}


def evaluation_manifest(rows, classes, split_id):
    patients = {}
    groups = []
    for row in rows:
        meta = row["metadata"]
        patient = meta["statistical_patient_id"]
        patients[patient] = {"patient_id": patient, "truth": canonical_label(meta["class"]),
                             "center_id": center_id(meta)}
        groups.append({"group_id": row["uid"], "patient_id": patient, "phases": {
            phase: {"image_ref": row["images"][index]["image"], "ground_truth": [
                {"label": item["label"], "bbox_2d": item["bbox"]} for item in row["gt_dets"] if item["phase"] == phase]}
            for index, phase in enumerate(("a", "p", "v"))}})
    return {"schema_version": 1, "classes": list(classes), "split": "validation", "split_id": split_id,
            "patients": [patients[key] for key in sorted(patients)], "input_groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--train-jsonl")
    inputs.add_argument("--loco-source-jsonl", help="Already prepared groups from the source center only")
    parser.add_argument("--validation-jsonl")
    parser.add_argument("--data-mode", choices=sorted(CLASS_SPACES), default="private")
    parser.add_argument("--source-center")
    parser.add_argument("--held-out-center")
    parser.add_argument("--tuning-fraction", type=float)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--active-classes", nargs="+")
    parser.add_argument("--split-id")
    parser.add_argument("--test-identity-manifest", type=Path)
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
    if args.refit_train_validation and not (args.validation_jsonl or args.loco_source_jsonl):
        parser.error("--refit-train-validation requires --validation-jsonl")
    loco_options = (args.source_center, args.held_out_center, args.tuning_fraction, args.split_seed,
                    args.active_classes, args.split_id, args.test_identity_manifest)
    if args.loco_source_jsonl and (args.validation_jsonl or any(value is None for value in loco_options)):
        parser.error("LOCO source mode requires all explicit center/split/class/test-identity options and no --validation-jsonl")
    if not args.loco_source_jsonl and any(value is not None for value in loco_options):
        parser.error("LOCO split options require --loco-source-jsonl")
    classes = tuple(canonical_label(value) for value in args.active_classes) if args.active_classes else CLASS_SPACES[args.data_mode]
    if len(classes) != len(set(classes)) or not set(classes) <= set(CLASS_SPACES[args.data_mode]):
        parser.error("Active classes must be unique and belong to data_mode")

    data_root = Path(args.data_root).resolve()
    train_path = Path(args.loco_source_jsonl or args.train_jsonl).resolve()
    validation_path = Path(args.validation_jsonl).resolve() if args.validation_jsonl else None
    source_rows = load_jsonl(train_path)
    train = [unpack(row, data_root, classes) for row in source_rows]
    validation = (
        [unpack(row, data_root, classes) for row in load_jsonl(validation_path)]
        if validation_path is not None else []
    )
    if not train:
        raise RuntimeError("The supplied train source is empty")
    if validation_path is not None and not validation:
        raise RuntimeError("The supplied validation source is empty")
    source_summary = row_summary(train)
    loco = None
    if args.loco_source_jsonl:
        train, validation, loco = loco_partition(train, source_center=args.source_center,
            held_out_center=args.held_out_center, fraction=args.tuning_fraction, seed=args.split_seed,
            classes=classes, reserved_test=args.test_identity_manifest)
    validate_source_boundaries(train, validation)
    class_counts = class_counts_for(train)
    output_dir = Path(args.output_dir).resolve()
    outputs = {}

    def emit(rows: list[dict], stem: str, counts: dict[str, int], role: str, population: str) -> None:
        path = output_dir / f"{stem}.jsonl"
        write_verl(rows, path, stem, counts, args.data_mode)
        outputs[stem] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "role": role,
            **row_summary(rows),
            "class_counts": counts,
            "class_counts_population": population,
        }

    count_population = "source_center_fit_subset" if loco else "supplied_train_source"
    emit(train, "real961_rl_train", class_counts, "training_only", count_population)
    if validation:
        emit(validation, "real961_rl_validation", class_counts, "held_out_validation", count_population)
    monitor = validation if validation else train[: min(len(train), 64)]
    emit(
        monitor,
        "real961_rl_monitor",
        class_counts,
        "held_out_validation_monitor" if validation else "training_subset_diagnostic_not_validation",
        count_population,
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
    if loco is not None:
        manifest_path = output_dir / "validation_manifest.json"
        manifest_path.write_text(json.dumps(evaluation_manifest(validation, classes, args.split_id), indent=2))
        validation_ids = {row["uid"] for row in validation}
        source_path = output_dir / "validation_source.jsonl"
        with source_path.open("w") as stream:
            for row in source_rows:
                uid = f"{row['metadata']['logical_key']}:{int(row['metadata']['slice_index']):04d}"
                if uid in validation_ids:
                    stream.write(json.dumps(row) + "\n")
        loco.update(split_id=args.split_id, active_classes=list(classes),
            validation_manifest=str(manifest_path), validation_manifest_sha256=sha256_file(manifest_path),
            validation_source=str(source_path), validation_source_sha256=sha256_file(source_path),
            final_fit_scope="train_plus_validation" if args.refit_train_validation else "train_only",
            final_training_file=outputs["real961_rl_refit_train_validation" if args.refit_train_validation else "real961_rl_train"]["path"])
    audit = {
        "contract_version": 2,
        "split_policy": "explicit_source_center_patient_stratification" if loco else "preserve_supplied_train_validation_boundaries",
        "data_mode": args.data_mode,
        "active_classes": list(classes),
        "loco": loco,
        "sources": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path), **source_summary,
                      "role": "source_center_before_partition" if loco else "supplied_training_subset"},
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
        "source_partition_provenance": "explicit_source_center_patient_stratification" if loco else "caller_supplied",
        "visual_z_status": "not_extracted",
    }
    with (output_dir / "audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
