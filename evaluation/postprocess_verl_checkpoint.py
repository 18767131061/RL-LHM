#!/usr/bin/env python3
"""Legacy checkpoint postprocessor using the older evaluator CLI/output contract.

Passes --metadata, --bootstrap and --seed, and requires legacy outputs including
input_predictions_joined.csv and report.md. evaluate_paper_protocol.py instead
requires --manifest and keyed JSONL predictions and is not a drop-in replacement.
See README.md's Evaluation interface notes; no automatic schema adapter is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_PATTERN = re.compile(r"^global_step_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def discover_latest_checkpoint(run_dir: Path) -> tuple[int, Path]:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("global_step_*"):
        match = STEP_PATTERN.match(path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"No global_step_* checkpoint found in {run_dir}")

    marker = run_dir / "latest_checkpointed_iteration.txt"
    if marker.is_file():
        try:
            marked_step = int(marker.read_text(encoding="utf-8").strip())
        except ValueError as error:
            raise RuntimeError(f"Invalid checkpoint marker: {marker}") from error
        marked_path = run_dir / f"global_step_{marked_step}"
        if not marked_path.is_dir():
            raise RuntimeError(f"Checkpoint marker points to a missing directory: {marked_path}")
        selected = (marked_step, marked_path)
    else:
        selected = max(candidates, key=lambda item: item[0])

    actor = selected[1] / "actor"
    config_path = actor / "fsdp_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Incomplete actor checkpoint; missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    world_size = int(config.get("world_size", 0))
    if world_size <= 0:
        raise RuntimeError(f"Invalid FSDP world size in {config_path}: {world_size}")
    missing = [
        actor / f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        if not (actor / f"model_world_size_{world_size}_rank_{rank}.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"Incomplete actor checkpoint; missing model shards: {missing}")
    return selected


def validate_merged_model(model_dir: Path) -> None:
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"Merged model is missing config.json: {model_dir}")
    weights = [
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in model_dir.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not weights:
        raise RuntimeError(f"Merged model has no non-empty weight files: {model_dir}")


def selected_file_manifest(root: Path, include_model_shards: bool = False) -> list[dict[str, Any]]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if include_model_shards:
            if not (
                path.name == "fsdp_config.json"
                or path.name.startswith("model_world_size_")
                or "huggingface" in path.parts
            ):
                continue
        files.append(path)
    return [
        {
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]


def run_logged(command: list[str], log_path: Path, env: dict[str, str], cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        print("[command] " + " ".join(command), file=log)
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        print(f"[exit] {completed.returncode}", file=log)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def gpu_compute_pids(gpu: int) -> tuple[int, ...]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return tuple(
        sorted(
            {
                int(line.strip())
                for line in output.splitlines()
                if line.strip() and line.strip().isdigit()
            }
        )
    )


def wait_for_capacity_gpu(
    log_path: Path,
    requested_gpu: int | None,
    minimum_free_mib: int,
    poll_seconds: int,
    stable_polls: int = 3,
) -> int:
    stable: dict[int, int] = {}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        while True:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            states = [
                tuple(int(value.strip()) for value in row[:3])
                for row in csv.reader(io.StringIO(output))
            ]
            compute_pids = {gpu: gpu_compute_pids(gpu) for gpu, _, _ in states}
            for gpu, free_mib, utilization in states:
                selected = requested_gpu is None or gpu == requested_gpu
                ready = selected and free_mib >= minimum_free_mib
                stable[gpu] = stable.get(gpu, 0) + 1 if ready else 0
            print(
                f"[gpu] policy=capacity_coexistence requested_gpu={requested_gpu} "
                f"min_free_mib={minimum_free_mib} "
                + ";".join(
                    f"{gpu}:free{free_mib}:util{utilization}:pids{list(compute_pids[gpu])}:"
                    f"stable{stable.get(gpu, 0)}"
                    for gpu, free_mib, utilization in states
                ),
                file=log,
            )
            ready_gpus = [gpu for gpu, _, _ in states if stable.get(gpu, 0) >= stable_polls]
            if ready_gpus:
                return sorted(ready_gpus)[0]
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-legacy-protocol", action="store_true",
                        help="Explicitly use the older experiment layout and statistical/evaluation definitions")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cohort", choices=("private", "public"), required=True)
    parser.add_argument("--merge-python", required=True)
    parser.add_argument("--eval-python", required=True)
    parser.add_argument("--verl-repo", required=True)
    parser.add_argument("--inference-script", required=True)
    parser.add_argument("--evaluation-script", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--minimum-free-mib", type=int, default=72000)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--expected-patients", type=int)
    parser.add_argument("--dataset-label")
    args = parser.parse_args()
    if not args.allow_legacy_protocol:
        parser.error("Legacy interface: pass --allow-legacy-protocol only for the older experiment layout; "
                     "use evaluate_paper_protocol.py and joint_seed_statistics.py for current outputs")
    if args.minimum_free_mib <= 0 or args.poll_seconds <= 0:
        parser.error("--minimum-free-mib and --poll-seconds must be positive")

    run_dir = Path(args.run_dir).resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise RuntimeError(f"Missing formal run manifest: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("smoke"):
        raise RuntimeError("Refusing to postprocess a smoke-test run as formal output")
    if run_manifest.get("variant") != args.variant or int(run_manifest.get("seed", -1)) != args.seed:
        raise RuntimeError("Run manifest variant/seed does not match the requested postprocessing job")

    step, checkpoint = discover_latest_checkpoint(run_dir)
    if args.discover_only:
        print(json.dumps({"step": step, "checkpoint": str(checkpoint)}, indent=2))
        return

    merged_dir = run_dir / "merged_hf"
    merge_log = run_dir / "postprocess" / "merge.log"
    if merged_dir.exists():
        validate_merged_model(merged_dir)
    else:
        temporary = run_dir / f".merged_hf.tmp.{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        merge_env = os.environ.copy()
        merge_env["CUDA_VISIBLE_DEVICES"] = ""
        command = [
            str(Path(args.merge_python).resolve()),
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--use_cpu_initialization",
            "--local_dir",
            str(checkpoint / "actor"),
            "--target_dir",
            str(temporary),
        ]
        try:
            run_logged(command, merge_log, merge_env, Path(args.verl_repo).resolve())
            validate_merged_model(temporary)
            os.replace(temporary, merged_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    validate_merged_model(merged_dir)

    eval_dir = run_dir / "evaluation"
    predictions = eval_dir / "predictions.csv"
    inference_log = eval_dir / "inference.log"
    metrics_log = eval_dir / "metrics.log"
    eval_env = os.environ.copy()
    gpu_index = wait_for_capacity_gpu(
        eval_dir / "gpu_wait.log",
        args.gpu_index,
        args.minimum_free_mib,
        args.poll_seconds,
    )
    eval_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    eval_env["TOKENIZERS_PARALLELISM"] = "false"
    eval_env["PYTHONHASHSEED"] = str(args.seed)
    run_logged(
        [
            str(Path(args.eval_python).resolve()),
            str(Path(args.inference_script).resolve()),
            "--model-dir",
            str(merged_dir),
            "--json-path",
            str(Path(args.test_json).resolve()),
            "--data-root",
            str(Path(args.data_root).resolve()),
            "--output",
            str(predictions),
            "--device",
            args.device,
            "--max-new-tokens",
            "256",
            "--image-max-pixels",
            str(512 * 512),
            "--image-min-pixels",
            str(32 * 32),
            "--seed",
            str(args.seed),
        ],
        inference_log,
        eval_env,
    )
    metrics_command = [
        str(Path(args.eval_python).resolve()),
        str(Path(args.evaluation_script).resolve()),
        "--predictions",
        str(predictions),
        "--metadata",
        str(Path(args.metadata).resolve()),
        "--output-dir",
        str(eval_dir),
        "--bootstrap",
        "10000",
        "--seed",
        "2026",
    ]
    if args.expected_patients is not None:
        metrics_command.extend(["--expected-patients", str(args.expected_patients)])
    if args.dataset_label:
        metrics_command.extend(["--dataset-label", args.dataset_label])
    run_logged(
        metrics_command,
        metrics_log,
        eval_env,
    )

    required_outputs = [
        predictions,
        eval_dir / "metrics.json",
        eval_dir / "patient_predictions.csv",
        eval_dir / "input_predictions_joined.csv",
        eval_dir / "report.md",
    ]
    missing = [str(path) for path in required_outputs if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Formal evaluation outputs are incomplete: {missing}")

    actor_manifest = selected_file_manifest(checkpoint / "actor", include_model_shards=True)
    merged_manifest = selected_file_manifest(merged_dir)
    trace_inputs = [
        run_manifest_path,
        Path(args.dataset_manifest).resolve(),
        Path(args.dataset_audit).resolve(),
        Path(args.test_json).resolve(),
        Path(args.metadata).resolve(),
    ]
    trace_outputs = required_outputs
    artifact = {
        "status": "formal_evaluation_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": args.cohort,
        "variant": args.variant,
        "experiment_label": run_manifest.get("experiment_label", args.variant),
        "seed": args.seed,
        "global_step": step,
        "checkpoint": str(checkpoint),
        "merged_model": str(merged_dir),
        "actor_files": actor_manifest,
        "merged_files": merged_manifest,
        "trace_inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in trace_inputs
        ],
        "trace_outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in trace_outputs
        ],
    }
    marker = run_dir / "formal_evaluation_complete.json"
    atomic_write_json(marker, artifact)
    print(json.dumps({"status": artifact["status"], "marker": str(marker)}, indent=2))


if __name__ == "__main__":
    main()
