#!/usr/bin/env python3
"""Prepare explicit five-seed SFT recipes from already prepared three-phase data.

Writes preparation commands and input provenance. Execute the generated launch
commands to train models and retain their runtime logs and checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as packages
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.build_verl_real961_dataset import CLASS_SPACES, unpack, validate_source_boundaries
from training.framework_sft_common import load_json_array, prepare_sft_framework_artifacts

PAPER_SEEDS = (42, 2026, 3407, 1234, 5678)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sources(train_path, validation_path, data_root, data_mode):
    loaded = []
    for source in (train_path, validation_path):
        rows = load_json_array(str(source))
        if not isinstance(rows, list) or not rows:
            raise ValueError("Both fitting and validation sources must be nonempty")
        prepared = [unpack(row, data_root, CLASS_SPACES[data_mode]) for row in rows]
        for row, item in zip(rows, prepared):
            roles = [m.get("role") for m in row["messages"]]
            if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
                raise ValueError("SFT expects one user/assistant exchange, with optional leading system")
            answer_content = row["messages"][-1]["content"]
            if (not isinstance(answer_content, list) or len(answer_content) != 1
                    or answer_content[0].get("type") != "text"):
                raise ValueError("SFT assistant target must contain exactly one JSON text segment")
            phases = {d["phase"] for d in item["gt_dets"]} | set(item["background_phases"])
            if phases != {"a", "p", "v"}:
                raise ValueError("Each SFT target phase needs explicit lesions or no-lesion annotation")
            for field in ("patient_id", "statistical_patient_id"):
                value = item["metadata"].get(field)
                if not isinstance(value, str) or not value.strip() or value != value.strip():
                    raise ValueError(f"Missing or invalid patient identity: {field}")
        loaded.append(prepared)
    validate_source_boundaries(*loaded)
    train_images = {image["image"] for row in loaded[0] for image in row["images"]}
    validation_images = {image["image"] for row in loaded[1] for image in row["images"]}
    if train_images & validation_images:
        raise ValueError("Fitting and validation sources reuse image paths")
    return loaded


def prepare(args):
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        raise ValueError("Provide five distinct training seeds")
    if args.artifacts.exists():
        raise ValueError("Use a new artifacts directory to preserve prior configurations")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise ValueError("--model must be an existing local model directory containing config.json")
    if (not math.isfinite(args.epochs) or not math.isfinite(args.learning_rate)
            or args.epochs <= 0 or args.learning_rate <= 0 or args.cutoff_len <= 0):
        raise ValueError("Training schedule and cutoff length must be positive")
    if not math.isfinite(args.warmup_ratio) or not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be finite and in [0, 1]")
    optional_eval = [args.eval_json, args.eval_manifest, args.eval_group_map]
    if any(optional_eval) and not all(optional_eval):
        raise ValueError("Supply all three evaluation paths, or none")
    for path in [args.train_json, args.validation_json, *[p for p in optional_eval if p]]:
        if not path.is_file():
            raise ValueError(f"Missing source file: {path}")
    train, validation = validate_sources(args.train_json, args.validation_json, args.data_root, args.data_mode)
    source_hashes = {str(path.resolve()): file_hash(path) for path in
                     [args.train_json, args.validation_json, *[p for p in optional_eval if p]]}
    artifacts, checkpoints = args.artifacts.resolve(), args.checkpoints.resolve()
    for seed in args.seeds:
        target = checkpoints / f"seed_{seed}"
        if target.exists():
            raise ValueError(f"Checkpoint output already exists: {target}")
    runs = []
    for seed in args.seeds:
        result = prepare_sft_framework_artifacts(
            family_tag=args.model_family, model_family=args.model_family,
            data_mode=args.data_mode, exp_name=f"seed_{seed}",
            model_name_or_path=str(args.model.resolve()), llamafactory_template=args.template,
            sft_output_dir=str(checkpoints / f"seed_{seed}"), dataset_root=str(args.data_root.resolve()),
            train_json=str(args.train_json.resolve()), val_json=str(args.validation_json.resolve()),
            framework_root=str(artifacts), framework_gpus=args.data_parallel_size,
            per_device_batch=args.per_device_batch,
            global_batch=args.per_device_batch * args.accumulation * args.data_parallel_size,
            cutoff_len=args.cutoff_len, learning_rate=args.learning_rate, num_train_epochs=args.epochs,
            preprocessing_num_workers=args.preprocessing_workers, overwrite_cache=False, min_images=3,
            lr_scheduler_type=args.scheduler, warmup_ratio=args.warmup_ratio,
            seed=seed, data_seed=seed,
            precision=args.precision,
            eval_json=str(args.eval_json.resolve()) if args.eval_json else "",
            eval_manifest=str(args.eval_manifest.resolve()) if args.eval_manifest else "",
            eval_group_map=str(args.eval_group_map.resolve()) if args.eval_group_map else "",
        )
        runs.append({"seed": seed, "data_seed": seed, "checkpoint_output": str(checkpoints / f"seed_{seed}"),
                     "config": str(result["sft_cfg"]), "config_sha256": file_hash(result["sft_cfg"]),
                     "train_script": str(result["sft_cmd_file"]),
                     "evaluation_script": str(result["eval_cmd_file"]) if result["eval_cmd_file"] else None})
    versions = {}
    for name in ("llamafactory", "transformers", "torch"):
        try:
            versions[name] = packages.version(name)
        except packages.PackageNotFoundError:
            versions[name] = None
    result = {"schema_version": 1, "status": "prepared_not_trained", "runs": runs,
              "sources_sha256": source_hashes, "preparation_environment": versions,
              "fitting_groups": len(train), "validation_groups": len(validation),
              "data_parallel_size": args.data_parallel_size, "gradient_accumulation_steps": args.accumulation,
              "precision": args.precision,
              "effective_group_batch": args.per_device_batch * args.accumulation * args.data_parallel_size,
              "checkpoint_selection": "final_epoch", "validation_used_for_training": False}
    for path, digest in source_hashes.items():
        if file_hash(path) != digest:
            raise ValueError("Source changed during SFT preparation")
    (artifacts / "sft_runs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-family", default="qwen")
    parser.add_argument("--template", default="qwen3_vl")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--data-mode", choices=tuple(CLASS_SPACES), required=True)
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--validation-json", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs=5, default=list(PAPER_SEEDS))
    parser.add_argument("--data-parallel-size", type=int, default=4)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--scheduler", default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--preprocessing-workers", type=int, default=4)
    parser.add_argument("--eval-json", type=Path)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--eval-group-map", type=Path)
    args = parser.parse_args()
    for name in ("data_parallel_size", "per_device_batch", "accumulation", "preprocessing_workers"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
