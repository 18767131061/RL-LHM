#!/usr/bin/env python3
"""Launch one audited VERL ablation variant with the manuscript hyperparameters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = {
    "vanilla_grpo": {"reward": "vanilla", "shae": False, "vgs": False},
    "shae_grpo": {"reward": "vanilla", "shae": True, "vgs": False},
    "dacvq_vgs_grpo": {"reward": "vanilla", "shae": False, "vgs": True},
    "shae_dacvq_vgs": {"reward": "vanilla", "shae": True, "vgs": True},
    "rars_grpo": {"reward": "rars", "shae": False, "vgs": False},
    "rars_shae": {"reward": "rars", "shae": True, "vgs": False},
    "rars_dacvq_vgs": {"reward": "rars", "shae": False, "vgs": True},
    "full_rl_lhm": {"reward": "rars", "shae": True, "vgs": True},
}

REQUIRED_RISK_SCRIPTS = (
    "verl_lhm_launcher.py",
    "verl_liver_reward.py",
    "verl_completion_likelihood_reward_manager.py",
    "verl_lhm_extensions.py",
    "patch_verl_sample_uid.py",
)
POST_PRIMARY_RISK_CONTROLS = {
    "author_nominal_control",
    "uniform",
    "risk_off",
    "selected_minus_0p1",
    "selected_plus_0p1",
}

WATCHDOG_POLL_SECONDS = 15
WATCHDOG_WARMUP_SECONDS = 180
WATCHDOG_CONSECUTIVE_POLLS = 4
WATCHDOG_GPU_UTILIZATION_MAX = 10
SINGLE_GPU_SMOKE_ROLLOUT_MEMORY_UTILIZATION = 0.35
FORMAL_ROLLOUT_MEMORY_UTILIZATION = 0.50


def validate_gpu_request(
    n_gpus: int, *, smoke: bool = False, formal_single_gpu: bool = False
) -> None:
    if n_gpus < 1:
        raise ValueError("RL-LHM/GRPO requires at least one GPU")
    if smoke and formal_single_gpu:
        raise ValueError("--formal-single-gpu cannot be combined with --smoke")
    if formal_single_gpu and n_gpus != 1:
        raise ValueError("--formal-single-gpu requires exactly one GPU")
    if n_gpus < 2 and not (smoke or formal_single_gpu):
        raise ValueError(
            "Single-GPU formal RL-LHM/GRPO requires explicit --formal-single-gpu "
            "authorization; otherwise at least two GPUs are required."
        )


def rollout_gpu_memory_utilization(
    *, smoke: bool, n_gpus: int, formal_single_gpu: bool = False
) -> float:
    if n_gpus == 1 and (smoke or formal_single_gpu):
        return SINGLE_GPU_SMOKE_ROLLOUT_MEMORY_UTILIZATION
    return FORMAL_ROLLOUT_MEMORY_UTILIZATION


def use_single_gpu_cpu_offload(
    *, smoke: bool, n_gpus: int, formal_single_gpu: bool = False
) -> bool:
    return n_gpus == 1 and (smoke or formal_single_gpu)


def visible_gpu_indices(n_gpus: int, env: dict[str, str]) -> list[int]:
    raw = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return list(range(n_gpus))
    try:
        indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"CUDA_VISIBLE_DEVICES must contain physical GPU indices: {raw}") from error
    if len(indices) < n_gpus:
        raise ValueError(
            f"Requested {n_gpus} GPUs but CUDA_VISIBLE_DEVICES exposes only {indices}"
        )
    return indices[:n_gpus]


def query_gpu_utilization(indices: list[int]) -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    states = {
        int(row[0].strip()): int(row[1].strip())
        for row in csv.reader(io.StringIO(output))
    }
    return [states[index] for index in indices]


def query_gpu_states(indices: list[int]) -> list[dict[str, object]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    basic = {
        int(row[0].strip()): {
            "index": int(row[0].strip()),
            "free_mib": int(row[1].strip()),
            "utilization_percent": int(row[2].strip()),
        }
        for row in csv.reader(io.StringIO(output))
    }
    missing = [index for index in indices if index not in basic]
    if missing:
        raise RuntimeError(f"nvidia-smi did not report requested GPUs: {missing}")
    states = []
    for index in indices:
        state = dict(basic[index])
        state["existing_compute_pids"] = query_gpu_compute_pids(index)
        states.append(state)
    return states


def query_gpu_compute_pids(index: int) -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]


def insufficient_gpu_capacity(
    states: list[dict[str, object]], minimum_free_mib: int
) -> list[dict[str, object]]:
    return [
        state for state in states if int(state["free_mib"]) < minimum_free_mib
    ]


def terminate_process_group(process: subprocess.Popen, log, reason: str) -> None:
    print(f"[watchdog] aborting process_group={process.pid}: {reason}", file=log)
    log.flush()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait(timeout=30)


def cleanup_process_group(process_group: int, log) -> None:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return
    print(f"[cleanup] terminating remaining process_group={process_group}", file=log)
    log.flush()
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_with_watchdog(
    command: list[str],
    env: dict[str, str],
    log,
    output_dir: Path,
    n_gpus: int,
    minimum_free_mib: int,
    manifest_path: Path,
) -> int:
    indices = visible_gpu_indices(n_gpus, env)
    states = query_gpu_states(indices)
    insufficient = insufficient_gpu_capacity(states, minimum_free_mib)
    if insufficient:
        atomic_write_json(
            output_dir / "capacity_preflight_wait.json",
            {
                "status": "capacity_preflight_rejected_before_launch",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "minimum_free_mib": minimum_free_mib,
                "states": states,
                "process_started": False,
                "threshold_escalated": False,
                "unrelated_processes_signalled": False,
            },
        )
        raise RuntimeError(
            f"GPU capacity preflight failed: minimum_free_mib={minimum_free_mib}, "
            f"states={states}"
        )
    preflight_wait = output_dir / "capacity_preflight_wait.json"
    if preflight_wait.exists():
        preflight_wait.unlink()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gpu_capacity_policy"]["final_prelaunch_states"] = states
    manifest["gpu_capacity_policy"]["final_prelaunch_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    atomic_write_json(manifest_path, manifest)
    print(
        f"[preflight] policy=capacity_coexistence minimum_free_mib={minimum_free_mib} "
        f"states={states}",
        file=log,
    )
    log.flush()
    cpu_count = max(1, os.cpu_count() or 1)
    load_limit = max(32.0, 1.25 * cpu_count)
    started = time.monotonic()
    consecutive = 0
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    while True:
        try:
            returncode = process.wait(timeout=WATCHDOG_POLL_SECONDS)
            cleanup_process_group(process.pid, log)
            return returncode
        except subprocess.TimeoutExpired:
            pass
        elapsed = time.monotonic() - started
        load_1m = os.getloadavg()[0]
        try:
            utilization = query_gpu_utilization(indices)
            mean_utilization = sum(utilization) / len(utilization)
        except Exception as error:
            print(f"[watchdog] telemetry_error={type(error).__name__}: {error}", file=log)
            log.flush()
            continue
        unsafe = (
            elapsed >= WATCHDOG_WARMUP_SECONDS
            and load_1m >= load_limit
            and mean_utilization <= WATCHDOG_GPU_UTILIZATION_MAX
        )
        consecutive = consecutive + 1 if unsafe else 0
        print(
            f"[watchdog] elapsed={elapsed:.0f}s load1={load_1m:.1f}/{load_limit:.1f} "
            f"gpu_util={utilization} unsafe={consecutive}/{WATCHDOG_CONSECUTIVE_POLLS}",
            file=log,
        )
        log.flush()
        if consecutive < WATCHDOG_CONSECUTIVE_POLLS:
            continue
        reason = (
            f"load1={load_1m:.1f} >= {load_limit:.1f} while GPUs {indices} "
            f"remained at mean utilization {mean_utilization:.1f}%"
        )
        atomic_write_json(
            output_dir / "watchdog_abort.json",
            {
                "status": "aborted_by_resource_watchdog",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "process_group": process.pid,
            },
        )
        terminate_process_group(process, log, reason)
        return process.returncode if process.returncode is not None else 1


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_risk_selection_freeze(freeze_path: Path, scripts_dir: Path) -> dict:
    """Validate the selected-first contract used by primary Full RL-LHM."""
    try:
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unreadable risk-selection freeze: {freeze_path}") from error
    expected = {
        "status": "frozen_selected_risk_matrix_for_primary_full_rl",
        "selection_role": "pre_primary_full_rl_hyperparameter_selection",
        "selected_replaces_author_nominal": True,
        "author_defined_nominal_retained_as_control": True,
        "reviewed_schema_version": 7,
        "risk_parameterization": "direct_absolute_pair_weights_plus_shared_other",
        "author_nominal_used_in_primary_candidates": False,
        "selected_matrix_derived_from_author_nominal": False,
        "formal_test_used_for_selection": False,
    }
    failed = [key for key, value in expected.items() if payload.get(key) != value]
    if failed:
        raise RuntimeError(
            f"Invalid selected-first risk freeze {freeze_path}; failed fields: {failed}"
        )
    selected_scripts = Path(str(payload.get("selected_scripts", ""))).resolve()
    if selected_scripts != scripts_dir.resolve():
        raise RuntimeError(
            f"Selected scripts mismatch: freeze={selected_scripts}, requested={scripts_dir.resolve()}"
        )
    expected_hashes = payload.get("selected_scripts_hashes")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("Risk freeze has no selected_scripts_hashes mapping")
    for filename in REQUIRED_RISK_SCRIPTS:
        script = scripts_dir / filename
        expected_hash = expected_hashes.get(filename)
        if not expected_hash or not script.is_file() or sha256_file(script) != expected_hash:
            raise RuntimeError(f"Selected risk script hash mismatch: {script}")
    values = payload.get("selected_parameter_values")
    if not isinstance(values, dict) or not values:
        raise RuntimeError("Risk freeze has no selected_parameter_values")
    matrix_path = Path(str(payload.get("selected_matrix", ""))).resolve()
    matrix_hash = payload.get("selected_matrix_sha256")
    if (
        not matrix_path.is_file()
        or not isinstance(matrix_hash, str)
        or sha256_file(matrix_path) != matrix_hash
    ):
        raise RuntimeError(f"Selected risk matrix hash mismatch: {matrix_path}")
    author_hash = payload.get("author_nominal_matrix_sha256")
    author_matrix = payload.get("author_nominal_matrix")
    calculated_author_hash = hashlib.sha256(
        json.dumps(author_matrix, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not isinstance(author_matrix, dict) or author_hash != calculated_author_hash:
        raise RuntimeError("Risk freeze has an invalid author-control matrix hash")
    return payload


def validate_post_primary_risk_control_manifest(
    manifest_path: Path,
    scripts_dir: Path,
) -> dict:
    """Validate a frozen sensitivity control without making it a primary run."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unreadable post-primary risk-control manifest: {manifest_path}") from error
    expected = {
        "status": "frozen_post_primary_risk_sensitivity_control",
        "contract_version": 1,
        "post_primary_sensitivity": True,
        "primary_selected_completed_before_control_freeze": True,
        "formal_test_used_for_selection": False,
        "frozen_before_formal_control_evaluation": True,
        "control_cannot_satisfy_primary_completion": True,
        "seeds": [42, 2026, 3407, 1234, 5678],
    }
    failed = [key for key, value in expected.items() if payload.get(key) != value]
    control = payload.get("control")
    if control not in POST_PRIMARY_RISK_CONTROLS:
        failed.append("control")
    cohort = payload.get("cohort")
    if cohort not in {"private", "public"}:
        failed.append("cohort")
    if failed:
        raise RuntimeError(
            f"Invalid post-primary risk-control manifest {manifest_path}; failed fields: {failed}"
        )

    selected_freeze_path = Path(str(payload.get("selected_freeze", ""))).resolve()
    selected_freeze_hash = payload.get("selected_freeze_sha256")
    if (
        not selected_freeze_path.is_file()
        or not isinstance(selected_freeze_hash, str)
        or sha256_file(selected_freeze_path) != selected_freeze_hash
    ):
        raise RuntimeError(f"Control selected-freeze hash mismatch: {selected_freeze_path}")
    try:
        selected_payload = json.loads(selected_freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unreadable selected freeze: {selected_freeze_path}") from error
    if selected_payload.get("cohort") != cohort:
        raise RuntimeError("Control manifest cohort differs from its selected freeze")
    selected_scripts = Path(str(selected_payload.get("selected_scripts", ""))).resolve()
    validate_risk_selection_freeze(selected_freeze_path, selected_scripts)

    completion_markers = payload.get("primary_completion_markers")
    if not isinstance(completion_markers, dict) or set(completion_markers) != {"private", "public"}:
        raise RuntimeError("Control manifest must bind both selected-primary completion markers")
    for expected_cohort, trace in completion_markers.items():
        if not isinstance(trace, dict):
            raise RuntimeError(f"Invalid completion trace for {expected_cohort}")
        marker_path = Path(str(trace.get("path", ""))).resolve()
        marker_hash = trace.get("sha256")
        if (
            not marker_path.is_file()
            or not isinstance(marker_hash, str)
            or sha256_file(marker_path) != marker_hash
        ):
            raise RuntimeError(f"Primary completion hash mismatch: {marker_path}")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Unreadable primary completion marker: {marker_path}") from error
        marker_checks = {
            "status": marker.get("status") == "formal_pipeline_complete",
            "scope": marker.get("scope") == "core_sft_vanilla_grpo_full_rl_lhm_only",
            "cohort": marker.get("cohort") == expected_cohort,
            "seeds": marker.get("seeds") == [42, 2026, 3407, 1234, 5678],
            "methods": marker.get("main_experiments") == ["vanilla_grpo", "full_rl_lhm"],
            "primary": marker.get("full_rl_lhm_risk_matrix_role")
            == "development_selected_primary",
        }
        if not all(marker_checks.values()):
            bad = [key for key, value in marker_checks.items() if not value]
            raise RuntimeError(
                f"Invalid selected-primary completion marker {marker_path}; failed fields: {bad}"
            )

    control_scripts = Path(str(payload.get("control_scripts", ""))).resolve()
    if control_scripts != scripts_dir.resolve():
        raise RuntimeError(
            f"Control scripts mismatch: manifest={control_scripts}, requested={scripts_dir.resolve()}"
        )
    script_hashes = payload.get("control_scripts_hashes")
    if not isinstance(script_hashes, dict):
        raise RuntimeError("Control manifest has no script-hash mapping")
    for filename in REQUIRED_RISK_SCRIPTS:
        script = scripts_dir / filename
        if (
            not script.is_file()
            or not script_hashes.get(filename)
            or sha256_file(script) != script_hashes[filename]
        ):
            raise RuntimeError(f"Post-primary control script hash mismatch: {script}")

    matrix_path = Path(str(payload.get("control_matrix", ""))).resolve()
    matrix_hash = payload.get("control_matrix_sha256")
    if not matrix_path.is_file() or not isinstance(matrix_hash, str) or sha256_file(matrix_path) != matrix_hash:
        raise RuntimeError(f"Post-primary control matrix hash mismatch: {matrix_path}")
    try:
        matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unreadable post-primary control matrix: {matrix_path}") from error
    matrix = matrix_payload.get("matrix")
    matrix_value_hash = hashlib.sha256(
        json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not isinstance(matrix, dict) or payload.get("control_matrix_value_sha256") != matrix_value_hash:
        raise RuntimeError("Post-primary control matrix value hash mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument(
        "--validation-file",
        help="Optional separate validation JSONL; periodic validation remains disabled by default",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scripts-dir", required=True)
    parser.add_argument(
        "--risk-selection-freeze",
        help="Selected-first freeze required by the formal Full RL-LHM scheduler",
    )
    parser.add_argument(
        "--risk-control-manifest",
        help="Frozen manifest for a post-primary Full RL-LHM risk sensitivity control",
    )
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--minimum-free-mib", type=int, default=50000)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal-single-gpu", action="store_true")
    parser.add_argument(
        "--risk-matrix-mode",
        choices=("ordinal", "uniform", "zero"),
        default="ordinal",
    )
    parser.add_argument("--risk-matrix-scale", type=float, default=1.0)
    parser.add_argument("--experiment-label")
    args = parser.parse_args()
    try:
        validate_gpu_request(
            args.n_gpus,
            smoke=args.smoke,
            formal_single_gpu=args.formal_single_gpu,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.risk_matrix_scale < 0:
        parser.error("--risk-matrix-scale must be non-negative")
    if args.minimum_free_mib <= 0:
        parser.error("--minimum-free-mib must be positive")
    experiment_label = args.experiment_label or args.variant
    if not experiment_label.replace("_", "").replace("-", "").isalnum():
        parser.error("--experiment-label may contain only letters, digits, underscores, and hyphens")

    train_file = Path(args.train_file).resolve()
    validation_file = Path(args.validation_file).resolve() if args.validation_file else None
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    scripts_dir = Path(args.scripts_dir).resolve()
    if not train_file.is_file() or not model_path.is_dir():
        raise RuntimeError(f"Missing train file or model path: {train_file}, {model_path}")
    if validation_file is not None and not validation_file.is_file():
        raise RuntimeError(f"Missing validation file: {validation_file}")
    if validation_file == train_file:
        raise RuntimeError("--validation-file must differ from --train-file")
    required = {
        "launcher": scripts_dir / "verl_lhm_launcher.py",
        "reward": scripts_dir / "verl_liver_reward.py",
        "manager": scripts_dir / "verl_completion_likelihood_reward_manager.py",
        "extensions": scripts_dir / "verl_lhm_extensions.py",
        "sample_uid_patch": scripts_dir / "patch_verl_sample_uid.py",
    }
    # Accept the repository layout while preserving the original flat bundle
    # layout used by frozen server manifests.
    if not all(path.is_file() for path in required.values()):
        package_root = Path(__file__).resolve().parents[1]
        required = {
            "launcher": package_root / "training" / "verl_lhm_launcher.py",
            "reward": package_root / "rewards" / "verl_liver_reward.py",
            "manager": package_root / "rewards" / "verl_completion_likelihood_reward_manager.py",
            "extensions": package_root / "rl" / "verl_lhm_extensions.py",
            "sample_uid_patch": package_root / "rl" / "patch_verl_sample_uid.py",
        }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing RL-LHM modules: {missing}")
    risk_freeze_path = (
        Path(args.risk_selection_freeze).resolve() if args.risk_selection_freeze else None
    )
    risk_control_path = (
        Path(args.risk_control_manifest).resolve() if args.risk_control_manifest else None
    )
    if risk_freeze_path is not None and risk_control_path is not None:
        raise RuntimeError("A run cannot be both selected-primary and a post-primary control")
    risk_freeze = None
    if risk_freeze_path is not None:
        if args.variant != "full_rl_lhm":
            raise RuntimeError("A risk-selection freeze is only valid for full_rl_lhm")
        risk_freeze = validate_risk_selection_freeze(risk_freeze_path, scripts_dir)
    risk_control = None
    if risk_control_path is not None:
        if args.variant != "full_rl_lhm":
            raise RuntimeError("A risk-control manifest is only valid for full_rl_lhm")
        risk_control = validate_post_primary_risk_control_manifest(
            risk_control_path, scripts_dir
        )
    attention_runtime = None
    if not args.config_only:
        attention_spec = importlib.util.find_spec("verl.utils.attention_utils")
        if attention_spec is None or attention_spec.origin is None:
            raise RuntimeError("Unable to locate VERL attention_utils runtime module")
        attention_runtime = Path(attention_spec.origin).resolve()
        if not attention_runtime.is_file():
            raise RuntimeError(f"Missing VERL attention runtime module: {attention_runtime}")
    config = VARIANTS[args.variant]
    if config["vgs"] and not args.config_only:
        first = next(
            json.loads(line)
            for line in train_file.open(encoding="utf-8")
            if line.strip()
        )
        visual_z = first.get("extra_info", {}).get("visual_z")
        if not isinstance(visual_z, list) or not visual_z:
            raise RuntimeError(f"{args.variant} requires extracted visual_z in {train_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = count_rows(train_file)
    train_batch = 4
    max_response_length = 64 if args.smoke else 256
    steps_per_epoch = train_rows // train_batch
    if steps_per_epoch < 1:
        raise RuntimeError(f"Training dataset is smaller than one batch: {train_rows} < {train_batch}")
    # A VGS smoke run needs a second epoch: epoch 1 populates the difficulty
    # memory and epoch 2 is the first time the weighted sampler can consume it.
    smoke_steps = 2 if config["vgs"] else 1
    epochs = smoke_steps if args.smoke else 6
    save_freq = -1 if args.smoke else steps_per_epoch
    ray_cpus = max(8, 4 * args.n_gpus)
    rollout_memory_utilization = rollout_gpu_memory_utilization(
        smoke=args.smoke,
        n_gpus=args.n_gpus,
        formal_single_gpu=args.formal_single_gpu,
    )
    single_gpu_cpu_offload = use_single_gpu_cpu_offload(
        smoke=args.smoke,
        n_gpus=args.n_gpus,
        formal_single_gpu=args.formal_single_gpu,
    )
    overrides = [
        f"algorithm.adv_estimator={'shae' if config['shae'] else 'grpo'}",
        f"data.train_files={train_file}",
        f"data.val_files={validation_file or train_file}",
        "data.prompt_key=prompt",
        "data.image_key=images",
        "data.max_prompt_length=1024",
        f"data.max_response_length={max_response_length}",
        f"data.train_batch_size={train_batch}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.shuffle=True",
        "data.dataloader_num_workers=2",
        f"data.seed={args.seed}",
        "+data.apply_chat_template_kwargs.enable_thinking=False",
        f"reward.custom_reward_function.path={required['reward']}",
        "reward.custom_reward_function.name=compute_score",
        "reward.reward_manager.source=importlib",
        "reward.reward_manager.name=CompletionLikelihoodRewardManager",
        f"reward.reward_manager.module.path={required['manager']}",
        "reward.num_workers=2",
        f"actor_rollout_ref.model.path={model_path}",
        # The audited VERL environment has no flash_attn package. Keeping the
        # standard padded path preserves training semantics and avoids an
        # optional-kernel import failure during old-log-prob computation.
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.clip_ratio=0.20",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={single_gpu_cpu_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={single_gpu_cpu_offload}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_memory_utilization:.2f}",
        "actor_rollout_ref.rollout.max_num_batched_tokens=2048",
        "actor_rollout_ref.rollout.max_num_seqs=1",
        "actor_rollout_ref.rollout.max_model_len=2048",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=0.95",
        "actor_rollout_ref.rollout.n=2",
        # Two rollout workers are sufficient for the eight smoke/formal
        # completions per training batch and avoid eight independent image
        # preprocessing pools competing for host CPU.
        "actor_rollout_ref.rollout.agent.num_workers=2",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={single_gpu_cpu_offload}",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.val_before_train=False",
        "trainer.logger=[console]",
        f"trainer.project_name=rl_lhm_revision",
        f"trainer.experiment_name={experiment_label}_seed{args.seed}",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        "trainer.test_freq=-1",
        "trainer.resume_mode=auto",
        f"trainer.total_epochs={epochs}",
        "trainer.max_actor_ckpt_to_keep=2",
        f"trainer.default_local_dir={output_dir}",
        f"+trainer.seed={args.seed}",
        f"ray_kwargs.ray_init.num_cpus={ray_cpus}",
    ]
    if args.smoke:
        overrides.extend(
            [
                f"trainer.total_training_steps={smoke_steps}",
                "data.train_max_samples=4",
                "data.val_max_samples=4",
            ]
        )
    if single_gpu_cpu_offload:
        # PyTorch's foreach AdamW path materializes a full optimizer-state
        # temporary. The scalar implementation is mathematically equivalent
        # and avoids that single-GPU peak allocation.
        overrides.append(
            "actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}"
        )
    if config["shae"]:
        overrides.extend(
            [
                "+algorithm.shae_gate_mid=0.05",
                "+algorithm.shae_gate_temp=0.015",
                "+algorithm.shae_baseline_beta=0.95",
                "+algorithm.shae_noise_std=0.001",
                f"+algorithm.shae_save_every={1 if args.smoke else 200}",
                f"+algorithm.shae_baseline_path={output_dir / 'baseline_reward.json'}",
            ]
        )
    if config["vgs"]:
        overrides.extend(
            [
                "data.dataloader_num_workers=0",
                f"data.sampler.class_path=file://{required['extensions']}",
                "data.sampler.class_name=VLGSHybridSampler",
                "+data.sampler.use_vgs=True",
                f"+data.sampler.state_path={output_dir / 'curriculum_state.json'}",
                f"+data.sampler.save_every={1 if args.smoke else 200}",
                "+data.sampler.vq_code_capacity=64",
                "+data.sampler.vq_feat_decay=0.95",
                "+data.sampler.vq_reward_decay=0.90",
                "+data.sampler.vq_difficulty_beta=3.0",
                "+data.sampler.vq_dead_code_thresh=0.25",
                "+data.sampler.vq_dead_code_replace_prob=0.30",
                "+data.sampler.vq_revival_noise_std=0.001",
                "+data.sampler.vq_reward_init=0.0",
            ]
        )
    command = [sys.executable, str(required["launcher"]), *overrides]
    extension_pythonpath = str(required["extensions"].parent)
    if os.environ.get("PYTHONPATH"):
        extension_pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    # Explicit Ray job environment also covers workers attached to an existing
    # local Ray service, which need not inherit this launcher's process env.
    command.append(
        "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=" + json.dumps(extension_pythonpath)
    )
    runtime_environment = {
        "RL_LHM_REWARD_MODE": config["reward"],
        "RL_LHM_UID_AUDIT_PATH": str(output_dir / "sample_uid_runtime_audit.json"),
        "PYTHONPATH": extension_pythonpath,
        "RL_LHM_RISK_MATRIX_MODE": args.risk_matrix_mode,
        "RL_LHM_RISK_MATRIX_SCALE": str(args.risk_matrix_scale),
        "PYTHONHASHSEED": str(args.seed),
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        # Ray limits schedulable CPUs, but native libraries can otherwise
        # create a full thread pool in every worker and starve the host.
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }
    if single_gpu_cpu_offload:
        runtime_environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    manifest = {
        "variant": args.variant,
        "experiment_label": experiment_label,
        "components": config,
        "seed": args.seed,
        "train_file": str(train_file),
        "train_file_sha256": sha256_file(train_file),
        "train_rows": train_rows,
        "validation_file": None if validation_file is None else str(validation_file),
        "validation_file_sha256": (
            None if validation_file is None else sha256_file(validation_file)
        ),
        "validation_usage": {
            "val_before_train": False,
            "test_freq": -1,
            "data_val_files_role": (
                "explicit_validation_file_partition_not_checked" if validation_file is not None
                else "training_file_placeholder_not_held_out_validation"
            ),
        },
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "scripts_dir": str(scripts_dir),
        "n_gpus": args.n_gpus,
        "formal_single_gpu": args.formal_single_gpu,
        "ray_cpus": ray_cpus,
        "cpu_offload": single_gpu_cpu_offload,
        "cpu_offload_components": {
            "actor_parameters": single_gpu_cpu_offload,
            "actor_optimizer": single_gpu_cpu_offload,
            "reference_parameters": single_gpu_cpu_offload,
        },
        "optimizer_foreach": not single_gpu_cpu_offload,
        "rollout_gpu_memory_utilization": rollout_memory_utilization,
        "gpu_capacity_policy": {
            "mode": "capacity_coexistence",
            "minimum_free_mib_per_selected_gpu": args.minimum_free_mib,
            "existing_cuda_compute_pids_allowed": True,
            "utilization_is_audit_only": True,
            "low_positive_free_memory_never_triggers_proactive_termination": True,
            "explicit_cuda_oom_required_for_capacity_escalation": True,
            "final_prelaunch_states": None,
            "final_prelaunch_utc": None,
        },
        "resource_watchdog": {
            "poll_seconds": WATCHDOG_POLL_SECONDS,
            "warmup_seconds": WATCHDOG_WARMUP_SECONDS,
            "consecutive_polls": WATCHDOG_CONSECUTIVE_POLLS,
            "gpu_utilization_max": WATCHDOG_GPU_UTILIZATION_MAX,
            "load_limit_factor": 1.25,
        },
        "epochs": epochs,
        "expected_training_steps": smoke_steps if args.smoke else None,
        "steps_per_epoch": steps_per_epoch,
        "max_response_length": max_response_length,
        "risk_matrix_mode": args.risk_matrix_mode,
        "risk_matrix_scale": args.risk_matrix_scale,
        "risk_selection_freeze": None if risk_freeze_path is None else str(risk_freeze_path),
        "risk_selection_freeze_sha256": (
            None if risk_freeze_path is None else sha256_file(risk_freeze_path)
        ),
        "risk_selection_profile": (
            None if risk_freeze is None else risk_freeze["selected_profile"]
        ),
        "risk_selection_parameters": (
            None if risk_freeze is None else risk_freeze["selected_parameter_values"]
        ),
        "risk_control_manifest": (
            None if risk_control_path is None else str(risk_control_path)
        ),
        "risk_control_manifest_sha256": (
            None if risk_control_path is None else sha256_file(risk_control_path)
        ),
        "risk_control_name": None if risk_control is None else risk_control["control"],
        "risk_control_parameter_values": (
            None if risk_control is None else risk_control.get("control_parameter_values")
        ),
        "policy_objective": "VERL token-level clipped PPO/GRPO (vanilla policy loss, clip ratio 0.20)",
        "sample_identity_contract": {
            "rollout_group_key": "uid (per visit; unchanged from VeRL)",
            "persistent_sample_key": "extra_info.sample_uid",
            "repeated_sample_updates": "one EMA update per visit in first-occurrence batch order",
            "shae_bridge_required": bool(config["shae"]),
            "runtime_audit": str(output_dir / "sample_uid_runtime_audit.json") if config["shae"] else None,
        },
        "policy_objective_verification": {
            "status": "unverified_external_runtime",
            "description": (
                "The policy_objective field records the existing carrier declaration. "
                "This launcher does not pin or inspect the external VeRL policy-loss "
                "implementation; compatibility with the manuscript policy-loss conventions "
                "has not been established."
            ),
        },
        "configuration_only": args.config_only,
        "code_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in [*required.values(), *([] if attention_runtime is None else [attention_runtime])]
        ],
        "smoke": args.smoke,
        "command": command,
        "runtime_environment": runtime_environment,
    }
    manifest_path = output_dir / "run_manifest.json"
    atomic_write_json(manifest_path, manifest)
    if args.config_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    env = os.environ.copy()
    env.update(runtime_environment)
    with (output_dir / "train.log").open("a", encoding="utf-8", buffering=1) as log:
        returncode = run_with_watchdog(
            command,
            env,
            log,
            output_dir,
            args.n_gpus,
            args.minimum_free_mib,
            manifest_path,
        )
    if returncode == 0:
        smoke_evidence = {}
        if args.smoke and config["shae"]:
            baseline_path = output_dir / "baseline_reward.json"
            if not baseline_path.is_file():
                raise RuntimeError(f"SHAE smoke run produced no baseline state: {baseline_path}")
            baseline_state = json.loads(baseline_path.read_text(encoding="utf-8"))
            if not baseline_state:
                raise RuntimeError("SHAE smoke baseline state is empty")
            smoke_evidence["shae_baseline_entries"] = len(baseline_state)
        if args.smoke and config["vgs"]:
            curriculum_path = output_dir / "curriculum_state.json"
            if not curriculum_path.is_file():
                raise RuntimeError(f"DA-CVQ/VGS smoke run produced no state: {curriculum_path}")
            curriculum_state = json.loads(curriculum_path.read_text(encoding="utf-8"))
            reward_seen = curriculum_state.get("vq", {}).get("reward_seen", [])
            epoch_count = int(curriculum_state.get("epoch_count", 0))
            update_count = int(curriculum_state.get("update_count", 0))
            if epoch_count < 2 or update_count < 2 or sum(float(value) for value in reward_seen) <= 0:
                raise RuntimeError(
                    "DA-CVQ/VGS smoke evidence is incomplete: "
                    f"epoch_count={epoch_count}, update_count={update_count}, "
                    f"reward_seen_sum={sum(float(value) for value in reward_seen)}"
                )
            smoke_evidence.update(
                {
                    "vgs_epoch_count": epoch_count,
                    "dacvq_update_count": update_count,
                    "dacvq_reward_seen_sum": sum(float(value) for value in reward_seen),
                }
            )
        atomic_write_json(
            output_dir / "training_complete.json",
            {
                "status": "training_complete",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "variant": args.variant,
                "experiment_label": experiment_label,
                "seed": args.seed,
                "smoke": args.smoke,
                "formal_single_gpu": args.formal_single_gpu,
                "expected_training_steps": smoke_steps if args.smoke else None,
                "smoke_evidence": smoke_evidence,
                "risk_matrix_mode": args.risk_matrix_mode,
                "risk_matrix_scale": args.risk_matrix_scale,
                "risk_selection_freeze": (
                    None if risk_freeze_path is None else str(risk_freeze_path)
                ),
                "risk_selection_freeze_sha256": (
                    None if risk_freeze_path is None else sha256_file(risk_freeze_path)
                ),
                "risk_selection_profile": (
                    None if risk_freeze is None else risk_freeze["selected_profile"]
                ),
                "risk_selection_parameters": (
                    None if risk_freeze is None else risk_freeze["selected_parameter_values"]
                ),
                "risk_control_manifest": (
                    None if risk_control_path is None else str(risk_control_path)
                ),
                "risk_control_manifest_sha256": (
                    None if risk_control_path is None else sha256_file(risk_control_path)
                ),
                "risk_control_name": (
                    None if risk_control is None else risk_control["control"]
                ),
                "risk_control_parameter_values": (
                    None if risk_control is None else risk_control.get("control_parameter_values")
                ),
            },
        )
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
