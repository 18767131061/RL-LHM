#!/usr/bin/env python3
"""Plan and collect a new five-seed, one-axis-at-a-time RARS validation search.

No training is executed and no historical scores are supplied by this tool.
"""
from __future__ import annotations

import argparse
import json
import math
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rars_joint_grid import (
    PAIRS, SEEDS, VALUES, digest, identities, load_config, sha,
    validate_profile, write_json,
)

VERSION = "rars_oat_v1"
PLACEHOLDERS = {"seed", "profile_path", "train_file", "model_path", "output_dir"}


def decode_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def read_json(path):
    return decode_json(Path(path).read_text(encoding="utf-8"))


def checkpoint_digest(path):
    """Hash file names and contents, including model weights; an empty dir is not a checkpoint."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Checkpoint directory is missing: {path}")
    files = {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError(f"Checkpoint directory has no files: {path}")
    return digest(files)


def oat_design(cohort):
    """Return all axis points, retaining the shared all-one point on each axis."""
    if cohort not in PAIRS:
        raise ValueError("cohort must be private or public")
    axes = ["--".join(pair) for pair in PAIRS[cohort]] + ["other"]
    profiles, points = {}, []
    for axis in axes:
        for value in VALUES:
            profile = {"protocol": "oat", "cohort": cohort, "other_weight": 1.0,
                       "pair_weights": {key: 1.0 for key in axes if key != "other"}}
            if axis == "other":
                profile["other_weight"] = value
            else:
                profile["pair_weights"][axis] = value
            validate_profile(profile)
            profile_id = digest(profile)
            profiles[profile_id] = profile
            points.append({"axis": axis, "value": value, "profile_id": profile_id,
                           "task_ids": [f"{profile_id}:{seed}" for seed in SEEDS]})
    return axes, profiles, points


def command_template(config):
    command = config.get("training_command_argv")
    if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
        raise ValueError("Supply an explicit nonempty training_command_argv list; there is no default training command")
    fields = set()
    for arg in command:
        for _, field, spec, conversion in string.Formatter().parse(arg):
            if field is not None:
                if field not in PLACEHOLDERS or spec or conversion:
                    raise ValueError(f"Unsupported command placeholder: {field}")
                fields.add(field)
    if fields != PLACEHOLDERS:
        raise ValueError(f"training_command_argv must bind every task input: {sorted(PLACEHOLDERS)}")
    return command


def build_plan(config_path):
    # The common config describes a main-cohort P1/U1 validation split. The
    # profile's protocol='oat' separately selects its variable background weight.
    config_path = Path(config_path).resolve()
    raw = read_json(config_path)
    config = load_config(config_path)
    if config["protocol"] != "main" or config["values"] != VALUES:
        raise ValueError("OAT requires protocol=main and all 15 values, including for synthetic tests")
    command = command_template(raw)
    identity = identities(config)
    identity["seed_model_sha256"] = {key: checkpoint_digest(path) for key, path in config["seed_model_paths"].items()}
    identity["source_sha256"] = {name: sha(ROOT / name) for name in (
        "training/rars_oat.py", "training/rars_joint_grid.py", "training/run_verl_lhm_variant.py",
        "rewards/verl_liver_reward.py", "evaluation/evaluate_paper_protocol.py",
        "evaluation/classification.py", "evaluation/schema.py", "aggregation/patient.py")}
    axes, profiles, points = oat_design(config["cohort"])
    search_hash = digest({"version": VERSION, "config": config, "identity": identity})
    folder = Path(config["output_root"]) / ("oat_" + search_hash[:16])
    tasks = []
    for profile_id, profile in profiles.items():
        profile_path = folder / "profiles" / f"{profile_id}.json"
        for seed in SEEDS:
            task_dir = folder / "tasks" / profile_id / f"seed_{seed}"
            inputs = {"identity": identity, "profile_id": profile_id, "seed": seed}
            values = {"seed": str(seed), "profile_path": str(profile_path),
                      "train_file": config["seed_train_files"][str(seed)],
                      "model_path": config["seed_model_paths"][str(seed)],
                      "output_dir": str(task_dir / "training")}
            tasks.append({"task_id": f"{profile_id}:{seed}", "profile_id": profile_id,
                          "input_hash": digest(inputs), "task_dir": str(task_dir), **values,
                          "seed": seed, "training_command_argv": [v.format_map(values) for v in command]})
    plan = {"schema_version": 1, "workflow": VERSION, "config_path": str(config_path),
            "config_sha256": sha(config_path), "search_hash": search_hash, "identity": identity,
            "cohort": config["cohort"], "synthetic_test": config.get("synthetic_test", False),
            "selection_split": "validation", "split_id": config["split_id"],
            "historical_results_verified": False, "axes": axes, "values": VALUES, "seeds": SEEDS,
            "profiles": profiles, "axis_points": points, "tasks": tasks,
            "counts": {"axes": len(axes), "axis_points": len(points),
                       "axis_seed_records": len(points) * len(SEEDS),
                       "unique_profiles": len(profiles), "unique_tasks": len(tasks)},
            "plan_path": str(folder / "plan.json")}
    plan["plan_digest"] = digest(plan)
    return plan


def result_template(task, plan):
    return {"schema_version": 1, "task_id": task["task_id"], "profile_id": task["profile_id"],
            "seed": task["seed"], "input_hash": task["input_hash"], "partition": "validation",
            "split_id": plan["split_id"], "validation_sha256": plan["identity"]["validation_sha256"],
            "synthetic_test": plan["synthetic_test"], "source": None, "source_sha256": None,
            "checkpoint": None, "checkpoint_sha256": None, "inference_receipt": None,
            "inference_receipt_sha256": None, "metrics": {"macro_f1": None}}


def inference_receipt_template(task, plan):
    """Schema for an inference producer's receipt, not a completed inference record."""
    receipt = result_template(task, plan)
    for key in ("metrics", "inference_receipt", "inference_receipt_sha256"):
        receipt.pop(key)
    return {**receipt, "kind": "oat_validation_inference", "status": "pending"}


def plan_search(config_path):
    plan = build_plan(config_path)
    plan_path = Path(plan["plan_path"])
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Output plan conflicts with the requested inputs")
    # Check every existing input before writing anything. Never overwrite a
    # conflicting task or an experiment's partially completed result template.
    outputs = [(Path(t["profile_path"]), plan["profiles"][t["profile_id"]]) for t in plan["tasks"]]
    outputs += [(Path(t["task_dir"]) / "task.json", t) for t in plan["tasks"]]
    for path, payload in outputs:
        if path.exists() and read_json(path) != payload:
            raise ValueError(f"Conflicting prepared input: {path}")
    for path, payload in outputs:
        if not path.exists():
            write_json(path, payload)
    for task in plan["tasks"]:
        template = Path(task["task_dir"]) / "result_template.json"
        if not template.exists():
            write_json(template, result_template(task, plan))
        receipt = Path(task["task_dir"]) / "inference_receipt_template.json"
        if not receipt.exists():
            write_json(receipt, inference_receipt_template(task, plan))
    write_json(plan_path, plan)
    return plan


def validate_result(record, task, plan, manifest):
    from evaluation.evaluate_paper_protocol import evaluate
    if not isinstance(record, dict):
        raise ValueError("Each result must be an object")
    expected = result_template(task, plan)
    identity_keys = ("schema_version", "task_id", "profile_id", "seed", "input_hash",
                     "partition", "split_id", "validation_sha256", "synthetic_test")
    if any(type(record.get(key)) is not type(expected[key]) or record.get(key) != expected[key] for key in identity_keys):
        raise ValueError("Result task/profile/seed/input/validation identity mismatch")
    folder = Path(task["task_dir"])
    profile_path = Path(task["profile_path"])
    if read_json(folder / "task.json") != task or read_json(profile_path) != plan["profiles"][task["profile_id"]]:
        raise ValueError("Prepared task or risk profile was changed")
    train = Path(task["output_dir"])
    run_path, done_path = train / "run_manifest.json", train / "training_complete.json"
    run, done = read_json(run_path), read_json(done_path)
    checks = (run.get("seed") == task["seed"], done.get("seed") == task["seed"],
              run.get("variant") == "full_rl_lhm", done.get("variant") == "full_rl_lhm",
              run.get("configuration_only") is False, run.get("smoke") is False,
              done.get("status") == "training_complete", done.get("smoke") is False,
              done.get("run_manifest_sha256") == sha(run_path),
              run.get("train_file_sha256") == plan["identity"]["seed_train_sha256"][str(task["seed"])],
              run.get("model_path") == task["model_path"], run.get("risk_profile_sha256") == sha(profile_path))
    if not all(checks):
        raise ValueError("Training provenance is incomplete or conflicts with this task")
    environment = run.get("runtime_environment")
    if not isinstance(environment, dict) or environment.get("RL_LHM_RISK_PROFILE") != str(profile_path):
        raise ValueError("Training runtime risk-profile environment differs from this task")
    command = run.get("command")
    profile_key = "ray_kwargs.ray_init.runtime_env.env_vars.RL_LHM_RISK_PROFILE"
    if not isinstance(command, list) or any(not isinstance(arg, str) for arg in command):
        raise ValueError("Training command provenance is missing")
    profile_args = [arg.split("=", 1)[1] for arg in command
                    if "=" in arg and arg.split("=", 1)[0].lstrip("+") == profile_key]
    if len(profile_args) != 1 or decode_json(profile_args[0]) != str(profile_path):
        raise ValueError("Training command must bind this task's risk profile exactly once")
    for key in ("source", "checkpoint"):
        if not isinstance(record.get(key), str) or not Path(record[key]).is_absolute():
            raise ValueError(f"Result requires an absolute {key} path")
    checkpoint = Path(record["checkpoint"]).resolve()
    final_name = done.get("final_checkpoint")
    if not isinstance(final_name, str) or Path(final_name).name != final_name or final_name in ("", ".", ".."):
        raise ValueError("Invalid final checkpoint identifier")
    if checkpoint != (train / final_name).resolve() or not checkpoint.is_relative_to(train.resolve()):
        raise ValueError("Checkpoint must be this task's completed final checkpoint")
    if record.get("checkpoint_sha256") != checkpoint_digest(checkpoint):
        raise ValueError("Checkpoint hash differs from result evidence")
    source = Path(record["source"])
    if not source.is_file() or record.get("source_sha256") != sha(source):
        raise ValueError("Prediction source is missing or its hash differs")
    receipt_path = record.get("inference_receipt")
    if not isinstance(receipt_path, str) or not Path(receipt_path).is_absolute():
        raise ValueError("Result requires an absolute inference_receipt path")
    receipt_path = Path(receipt_path)
    if record.get("inference_receipt_sha256") != sha(receipt_path):
        raise ValueError("Inference receipt hash differs")
    receipt = read_json(receipt_path)
    required_receipt = {key: record[key] for key in identity_keys}
    required_receipt.update(kind="oat_validation_inference", status="inference_complete",
                            source=str(source), source_sha256=record["source_sha256"],
                            checkpoint=record["checkpoint"], checkpoint_sha256=record["checkpoint_sha256"])
    if not isinstance(receipt, dict) or any(type(receipt.get(key)) is not type(value) or receipt.get(key) != value
                                          for key, value in required_receipt.items()):
        raise ValueError("Inference receipt does not bind predictions to this task, seed, validation input and checkpoint")
    metrics = record.get("metrics")
    score = metrics.get("macro_f1") if isinstance(metrics, dict) else None
    if not isinstance(metrics, dict) or set(metrics) != {"macro_f1"} or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Validation macro_f1 must be finite and in [0,1]")
    group_patients = {group["group_id"]: group["patient_id"] for group in manifest["input_groups"]}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = decode_json(line)
        if not isinstance(row, dict):
            raise ValueError("Prediction rows must be objects")
        if any(key in row and row[key] != "validation" for key in ("split", "partition")):
            raise ValueError("Prediction source is not identified as validation")
        if "patient_id" in row and row["patient_id"] != group_patients.get(row.get("group_id")):
            raise ValueError("Prediction source patient identity differs from validation")
    actual = evaluate(manifest, source, [0.5])
    if not math.isclose(score, actual["classification"]["macro_f1"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Reported metric differs from the supplied validation predictions")
    return {"task_id": task["task_id"], "profile_id": task["profile_id"], "seed": task["seed"],
            "macro_f1": actual["classification"]["macro_f1"], "result_digest": digest(record),
            "evidence_sha256": {str(path.resolve()): sha(path) for path in
                                (run_path, done_path, source, receipt_path, profile_path)},
            "checkpoint_sha256": record["checkpoint_sha256"]}


def select_axes(plan, results):
    """Summarize only complete five-seed profiles; exact mean ties choose the smaller value."""
    expected = {task["task_id"] for task in plan["tasks"]}
    if set(results) != expected:
        raise ValueError("Selection requires every unique profile x seed result")
    means = {}
    for profile_id in plan["profiles"]:
        scores = [results[f"{profile_id}:{seed}"]["macro_f1"] for seed in SEEDS]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
            raise ValueError("Selection requires finite scores in [0,1]")
        means[profile_id] = math.fsum(scores) / len(SEEDS)
    curves, selected = {}, {}
    for axis in plan["axes"]:
        curve = [{**point, "mean_validation_macro_f1": means[point["profile_id"]]}
                 for point in plan["axis_points"] if point["axis"] == axis]
        curves[axis] = curve
        selected[axis] = min(curve, key=lambda p: (-p["mean_validation_macro_f1"], p["value"]))
    profile = {"protocol": "oat", "cohort": plan["cohort"], "other_weight": selected["other"]["value"],
               "pair_weights": {axis: selected[axis]["value"] for axis in plan["axes"] if axis != "other"}}
    validate_profile(profile)
    return {"curves": curves, "selected_axis_points": selected, "combined_profile": profile,
            "joint_optimum_established": False, "combined_profile_requires_training": True,
            "combined_profile_validation_metric": None,
            "tie_rule": "exact five-seed mean tie: choose the smaller weight",
            "interpretation": "Axis peaks were measured with all other weights fixed at 1.0. The combined profile requires its own five-seed training and validation; OAT does not establish a joint optimum."}


def collect(plan_path, results_path, output):
    plan = read_json(plan_path)
    if build_plan(plan["config_path"]) != plan:
        raise ValueError("Plan or its bound config/data/model/source inputs changed; create a new plan")
    records = {}
    # Parse each JSONL line with the same duplicate-key/nonfinite checks as files.
    for line in Path(results_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = decode_json(line)
        if not isinstance(record, dict) or not isinstance(record.get("task_id"), str):
            raise ValueError("Each result requires a task_id")
        if record["task_id"] in records:
            raise ValueError("Duplicate result task_id; each task must appear exactly once")
        records[record["task_id"]] = record
    expected = {task["task_id"] for task in plan["tasks"]}
    if set(records) != expected:
        raise ValueError(f"Incomplete/unexpected results: missing={len(expected - records.keys())}, unexpected={len(records.keys() - expected)}")
    config = load_config(plan["config_path"])
    manifest = read_json(config["validation_manifest"])
    verified = {task["task_id"]: validate_result(records[task["task_id"]], task, plan, manifest)
                for task in plan["tasks"]}
    result = {"schema_version": 1, "workflow": VERSION, "plan_digest": plan["plan_digest"],
              "status": "synthetic_complete" if plan["synthetic_test"] else "validation_complete",
              "synthetic_test": plan["synthetic_test"], "historical_results_verified": False,
              "results_sha256": sha(results_path), "counts": plan["counts"],
              "verified_results": list(verified.values()), **select_axes(plan, verified)}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Collection output directory must be absent or empty")
    write_json(output / "selection.json", result)
    write_json(output / "combined_profile.json", result["combined_profile"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Write profiles and task templates without executing training")
    planning.add_argument("--config", required=True, type=Path)
    collection = commands.add_parser("collect", help="Verify a complete new validation search and combine axis peaks")
    collection.add_argument("--plan", required=True, type=Path)
    collection.add_argument("--results", required=True, type=Path)
    collection.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = plan_search(args.config) if args.command == "plan" else collect(args.plan, args.results, args.output)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("counts", "historical_results_verified")}
                     | ({"plan_path": result["plan_path"]} if args.command == "plan" else
                        {"status": result["status"], "combined_profile_requires_training": True}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
