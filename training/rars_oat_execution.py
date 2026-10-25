"""Sequential, resumable OAT training and validation execution."""
from __future__ import annotations

import fcntl
import json
import math
import shutil
import string
import subprocess
from pathlib import Path

from training.rars_joint_grid import SEEDS, digest, load_config, sha, write_json

INFERENCE_FIELDS = {"checkpoint", "validation_manifest", "seed", "output_path", "task_path",
                    "profile_path", "inference_input", "image_root", "group_map", "sft_model"}
REQUIRED_INFERENCE_FIELDS = {"checkpoint", "validation_manifest", "seed", "output_path"}


def referenced_image_hashes(input_path, image_root, hashes):
    """Bind local image operands in prepared training and inference records."""
    from training.rars_oat import decode_json
    for line in Path(input_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = decode_json(line)
        entries = list(row.get("images", []))
        for message in row.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, list):
                entries.extend(segment for segment in content
                               if isinstance(segment, dict) and segment.get("type") == "image")
        for entry in entries:
            value = entry.get("image") if isinstance(entry, dict) else entry
            if not isinstance(value, str) or not value:
                raise ValueError("Prepared image operands must be explicit local file paths")
            path = (Path(image_root) / value).resolve()
            if not path.is_file():
                raise ValueError(f"Referenced image file is missing: {path}")
            if str(path) not in hashes:
                hashes[str(path)] = sha(path)


def execution_config(raw, base, required=True, training_files=None):
    """Validate explicit commands and bind their local input files."""
    command = raw.get("inference_command_argv")
    if command is None and not required:
        return None
    if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
        raise ValueError("run requires explicit inference_command_argv")
    fields = set()
    for arg in command:
        for _, field, spec, conversion in string.Formatter().parse(arg):
            if field is not None:
                if field not in INFERENCE_FIELDS or spec or conversion:
                    raise ValueError(f"Unsupported inference placeholder: {field}")
                fields.add(field)
    if not REQUIRED_INFERENCE_FIELDS <= fields:
        raise ValueError(f"Inference command must bind {sorted(REQUIRED_INFERENCE_FIELDS)}")
    fmt = raw.get("inference_format")
    if fmt not in ("canonical_jsonl", "inference_csv"):
        raise ValueError("inference_format must be canonical_jsonl or inference_csv")
    config = {"command_argv": command, "format": fmt, "input_sha256": {}, "image_sha256": {}}
    for key in ("inference_input", "group_map", "image_root"):
        value = raw.get(key)
        if value is None:
            if key in fields or key == "group_map" and fmt == "inference_csv":
                raise ValueError(f"Supply explicit {key}")
            continue
        if not isinstance(value, str) or not value or "${" in value:
            raise ValueError(f"Supply explicit {key}")
        path = (base / value).resolve()
        if key == "image_root":
            if not path.is_dir():
                raise ValueError(f"Missing image_root: {path}")
        else:
            config["input_sha256"][str(path)] = sha(path)
        config[key] = str(path)
    if "inference_input" in config:
        source = Path(config["inference_input"])
        referenced_image_hashes(source, config.get("image_root", source.parent), config["image_sha256"])
    for source in sorted(set((training_files or {}).values())):
        source = Path(source)
        sidecar = source.with_suffix(source.suffix + ".audit.json")
        root = config.get("image_root", source.parent)
        if sidecar.is_file():
            from training.rars_oat import read_json
            identity = read_json(sidecar).get("identity", {})
            root = identity.get("image_root", root)
        referenced_image_hashes(source, root, config["image_sha256"])
    sources = raw.get("execution_source_files", [])
    if not isinstance(sources, list) or any(not isinstance(v, str) or not v for v in sources):
        raise ValueError("execution_source_files must list command-wrapper source files")
    # Hash local script operands automatically; external helpers can additionally
    # declare imported files through execution_source_files.
    for value in [*raw.get("training_command_argv", []), *command, *sources]:
        if "{" not in value:
            candidate = (base / value).resolve()
            if not candidate.is_file():
                from training.rars_oat import ROOT
                candidate = (ROOT / value).resolve()
            if candidate.is_file() and (candidate.suffix in (".py", ".sh") or value in sources):
                config["input_sha256"][str(candidate)] = sha(candidate)
            elif value in sources:
                raise ValueError(f"Missing execution source: {value}")
    return config


def file_inventory(roots):
    """Snapshot file names and metadata without rereading file contents."""
    entries = []
    for root in roots:
        paths = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in paths:
            if path.is_file():
                stat = path.stat()
                entries.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
    return entries


class CheckpointHashes:
    """Reuse stable digests and revalidate content when file metadata changes."""
    def __init__(self):
        self.entries = {}

    def __call__(self, path):
        from training.rars_oat import checkpoint_digest
        path = Path(path).resolve()
        inventory = file_inventory([path])
        previous = self.entries.get(path)
        if previous is not None and inventory == previous[0]:
            return previous[1]
        try:
            value = checkpoint_digest(path)
        except (OSError, ValueError) as error:
            if previous is not None:
                raise ValueError(f"Checkpoint changed during execution: {path}") from error
            raise
        if (file_inventory([path]) != inventory
                or previous is not None and value != previous[1]):
            raise ValueError(f"Checkpoint changed during execution: {path}")
        self.entries[path] = (inventory, value)
        return value


class InputInventory:
    """Keep a bound plan valid while allowing content-preserving metadata changes."""
    def __init__(self, config, execution=None, *, plan):
        self.config_path = plan["config_path"]
        self.plan_digest = plan["plan_digest"]
        self.paths = [Path(config["validation_manifest"]), Path(config["test_identity_manifest"]),
                      *map(Path, config["seed_train_files"].values()),
                      *map(Path, config["seed_model_paths"].values())]
        if execution:
            self.paths.extend(map(Path, execution["input_sha256"]))
            self.paths.extend(map(Path, execution["image_sha256"]))
        self.initial = self.snapshot()

    def snapshot(self):
        return file_inventory(self.paths)

    def check(self):
        from training.rars_oat import build_plan
        current = self.snapshot()
        if current == self.initial:
            return
        if [entry[0] for entry in current] != [entry[0] for entry in self.initial]:
            raise ValueError("Bound training data, model or validation input changed during execution")
        try:
            refreshed = build_plan(self.config_path)
        except (OSError, ValueError) as error:
            raise ValueError("Bound training data, model or validation input changed during execution") from error
        if self.snapshot() != current or refreshed["plan_digest"] != self.plan_digest:
            raise ValueError("Bound training data, model or validation input changed during execution")
        self.initial = current


def validate_execution_evidence(receipt, task, plan):
    """Check executor-produced command/import evidence when present."""
    from training.rars_oat import read_json
    folder = Path(task["task_dir"])
    if not (folder / "execution.json").exists():
        if "execution_sha256" in receipt:
            raise ValueError("Inference execution evidence is missing")
        return
    paths = {"execution_sha256": folder / "execution.json",
             "inference_command_complete_sha256": folder / "inference_command_complete.json",
             "canonical_import_sha256": folder / "canonical_import.json",
             "evaluation_sha256": folder / "evaluation.json"}
    for key, path in paths.items():
        if receipt.get(key) != sha(path):
            raise ValueError(f"Inference execution evidence changed: {path}")
    execution = read_json(paths["execution_sha256"])
    expected = {"task_id": task["task_id"], "input_hash": task["input_hash"], "plan_digest": plan["plan_digest"],
                "training_command_argv": task["training_command_argv"], "inference": plan["identity"]["execution"],
                "synthetic_test": plan["synthetic_test"]}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise ValueError("Execution context differs from its planned task")
    command = read_json(paths["inference_command_complete_sha256"])
    if (command.get("command_argv") != receipt.get("command_argv") or command.get("task_id") != task["task_id"]
            or command.get("input_hash") != task["input_hash"] or command.get("checkpoint_sha256") != receipt["checkpoint_sha256"]
            or command.get("validation_sha256") != plan["identity"]["validation_sha256"]):
        raise ValueError("Inference command identity differs from its result")
    extension = ".csv" if execution["inference"]["format"] == "inference_csv" else ".jsonl"
    raw = folder / ("inference_predictions" + extension)
    imported = read_json(paths["canonical_import_sha256"])
    if command.get("output_sha256") != sha(raw) or imported.get("raw_sha256") != sha(raw) or imported.get("output_sha256") != receipt["source_sha256"]:
        raise ValueError("Inference raw output or canonical import changed")


def run_logged(command, log_path):
    from training.rars_oat import ROOT
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"argv": command}) + "\n")
        log.flush()
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"Command exited {process.returncode}; see {log_path}")


def immutable_json(path, payload):
    from training.rars_oat import read_json
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"Existing execution artifact conflicts with its inputs: {path}")
    else:
        write_json(path, payload)


def check_small_inputs(plan):
    """Check source/config/inference inputs without rereading every SFT checkpoint."""
    from training.rars_oat import ROOT
    if sha(plan["config_path"]) != plan["config_sha256"]:
        raise ValueError("Config changed during execution")
    for name, expected in plan["identity"]["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Source changed during execution: {name}")
    for path, expected in plan["identity"]["execution"]["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Execution input changed: {path}")


def execute_task(task, plan, manifest, config, hashes):
    from evaluation.evaluate_paper_protocol import evaluate
    from evaluation.import_inference_predictions import convert
    from training.rars_oat import (METRICS, inference_receipt_template, read_json,
                                   result_template, validate_result, validate_training)
    folder = Path(task["task_dir"])
    result_path = folder / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        validate_result(result, task, plan, manifest, checkpoint_hasher=hashes)
        return result, False
    execution = plan["identity"]["execution"]
    context = {"schema_version": 1, "task_id": task["task_id"], "input_hash": task["input_hash"],
               "plan_digest": plan["plan_digest"], "training_command_argv": task["training_command_argv"],
               "inference": execution, "synthetic_test": plan["synthetic_test"]}
    immutable_json(folder / "execution.json", context)
    if not (Path(task["output_dir"]) / "training_complete.json").exists():
        run_logged(task["training_command_argv"], folder / "training_command.log")
    checkpoint = validate_training(task, plan)
    checkpoint_hash = hashes(checkpoint)
    suffix = ".csv" if execution["format"] == "inference_csv" else ".jsonl"
    raw_path = folder / ("inference_predictions" + suffix)
    source = folder / "predictions.jsonl"
    values = {"checkpoint": str(checkpoint), "validation_manifest": config["validation_manifest"],
              "seed": str(task["seed"]), "output_path": str(raw_path),
              "task_path": str(folder / "task.json"), "profile_path": task["profile_path"], "sft_model": task["model_path"],
              **{key: execution[key] for key in ("inference_input", "image_root", "group_map") if key in execution}}
    command = [arg.format_map(values) for arg in execution["command_argv"]]
    completed_path = folder / "inference_command_complete.json"
    command_identity = {"task_id": task["task_id"], "input_hash": task["input_hash"],
                        "checkpoint_sha256": checkpoint_hash, "command_argv": command,
                        "validation_sha256": plan["identity"]["validation_sha256"]}
    if completed_path.exists():
        completed = read_json(completed_path)
        if any(completed.get(k) != v for k, v in command_identity.items()) or completed.get("output_sha256") != sha(raw_path):
            raise ValueError("Completed inference command or raw output changed")
    else:
        run_logged(command, folder / "inference_command.log")
        # The command must succeed and produce a complete source. A partial output
        # may be resumed only by the same explicitly configured inference engine.
        if not raw_path.is_file():
            raise ValueError("Inference command produced no predictions")
        if hashes(checkpoint) != checkpoint_hash:
            raise ValueError("Checkpoint changed during inference")
        write_json(completed_path, {**command_identity, "output_sha256": sha(raw_path)})
    import_path = folder / "canonical_import.json"
    import_identity = {"format": execution["format"], "raw_sha256": sha(raw_path),
                       "validation_sha256": plan["identity"]["validation_sha256"],
                       "group_map_sha256": sha(execution["group_map"]) if execution["format"] == "inference_csv" else None}
    if import_path.exists():
        imported = read_json(import_path)
        if any(imported.get(k) != v for k, v in import_identity.items()) or imported.get("output_sha256") != sha(source):
            raise ValueError("Canonical predictions or import identity changed")
    else:
        if execution["format"] == "inference_csv":
            if source.exists():
                # Interrupted converters may leave a complete output before its
                # provenance sidecar. Recreate only this task-owned derived file.
                source.unlink()
                source.with_suffix(source.suffix + ".provenance.json").unlink(missing_ok=True)
            convert(raw_path, Path(config["validation_manifest"]), Path(execution["group_map"]), source)
        else:
            shutil.copyfile(raw_path, source)
        write_json(import_path, {**import_identity, "output_sha256": sha(source)})
    actual = evaluate(manifest, source, [0.5])
    write_json(folder / "evaluation.json", actual)
    record = result_template(task, plan)
    record.update(source=str(source), source_sha256=sha(source), checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_hash, metrics={key: actual["classification"][key] for key in METRICS})
    receipt = inference_receipt_template(task, plan)
    receipt.update({key: record[key] for key in ("source", "source_sha256", "checkpoint", "checkpoint_sha256")})
    receipt.update(status="inference_complete", command_argv=command,
                   inference_command_complete_sha256=sha(completed_path), canonical_import_sha256=sha(import_path),
                   evaluation_sha256=sha(folder / "evaluation.json"), execution_sha256=sha(folder / "execution.json"))
    receipt_path = folder / "inference_receipt.json"
    immutable_json(receipt_path, receipt)
    record.update(inference_receipt=str(receipt_path), inference_receipt_sha256=sha(receipt_path))
    validate_result(record, task, plan, manifest, checkpoint_hasher=hashes)
    immutable_json(result_path, record)
    return record, True


def build_combined_plan(plan, selection, config):
    from training.rars_oat import command_template, read_json, result_template, inference_receipt_template
    profile = selection["combined_profile"]
    pid = digest(profile)
    base = Path(plan["plan_path"]).parent / "combined"
    profile_path = base / "profiles" / f"{pid}.json"
    tasks = []
    command = command_template(read_json(plan["config_path"]))
    for seed in SEEDS:
        folder = base / "tasks" / f"seed_{seed}"
        values = {"seed": str(seed), "profile_path": str(profile_path),
                  "train_file": config["seed_train_files"][str(seed)],
                  "model_path": config["seed_model_paths"][str(seed)], "output_dir": str(folder / "training")}
        tasks.append({"task_id": f"combined:{pid}:{seed}", "profile_id": pid, "seed": seed,
                      "input_hash": digest({"parent": plan["plan_digest"], "selection": digest(selection),
                                            "stage": "combined", "profile_id": pid, "seed": seed}),
                      "task_dir": str(folder), **values,
                      "training_command_argv": [arg.format_map(values) for arg in command]})
        tasks[-1]["seed"] = seed
    combined = {**plan, "stage": "combined", "parent_plan_digest": plan["plan_digest"],
                "selection_digest": digest(selection), "profiles": {pid: profile}, "tasks": tasks,
                "axis_points": [], "axes": [], "plan_path": str(base / "plan.json"),
                "counts": {"unique_profiles": 1, "unique_tasks": 5}}
    combined.pop("plan_digest")
    combined["plan_digest"] = digest(combined)
    immutable_json(Path(combined["plan_path"]), combined)
    immutable_json(profile_path, profile)
    for task in tasks:
        folder = Path(task["task_dir"])
        immutable_json(folder / "task.json", task)
        immutable_json(folder / "result_template.json", result_template(task, combined))
        immutable_json(folder / "inference_receipt_template.json", inference_receipt_template(task, combined))
    return combined


def results_file(plan):
    from training.rars_oat import read_json
    records = []
    for task in plan["tasks"]:
        path = Path(task["task_dir"]) / "result.json"
        if not path.exists():
            raise ValueError("A complete result for every task and all five seeds is required")
        records.append(read_json(path))
    path = Path(plan["plan_path"]).parent / "results.jsonl"
    content = "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records)
    if path.exists() and path.read_text() != content:
        raise ValueError("Existing aggregate results changed")
    if not path.exists():
        temp = path.with_suffix(".jsonl.tmp")
        temp.write_text(content)
        temp.replace(path)
    return path, records


def run_workflow(config_path, stage="all", max_tasks=None):
    from training.rars_oat import (METRICS, collect, exact_seed_bootstrap, plan_search,
                                   read_json, validate_result)
    if stage not in ("scan", "combined", "all") or max_tasks is not None and (type(max_tasks) is not int or max_tasks <= 0):
        raise ValueError("stage must be scan/combined/all and max_tasks must be positive")
    config_path = Path(config_path).resolve()
    execution_config(read_json(config_path), config_path.parent)
    plan = plan_search(config_path)  # Recompute all bound data/model/source hashes once on entry.
    config = load_config(config_path)
    manifest = read_json(config["validation_manifest"])
    folder = Path(plan["plan_path"]).parent
    hashes = CheckpointHashes()
    input_inventory = InputInventory(config, plan["identity"]["execution"], plan=plan)
    ran = 0
    with (folder / ".execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another executor holds this OAT plan") from error
        for active_stage in (("scan",) if stage == "scan" else ("scan", "combined")):
            if active_stage == "combined":
                active = build_combined_plan(plan, selection, config)
            else:
                active = plan
            for task in active["tasks"]:
                check_small_inputs(plan)
                input_inventory.check()
                existing = (Path(task["task_dir"]) / "result.json").exists()
                if not existing and (stage == "combined" and active_stage == "scan"):
                    raise ValueError("Combined execution requires all completed scan results")
                if not existing and max_tasks is not None and ran >= max_tasks:
                    return {"status": "incomplete", "stage": active_stage, "executed_tasks": ran,
                            "plan_path": plan["plan_path"], "synthetic_test": plan["synthetic_test"]}
                _, executed = execute_task(task, active, manifest, config, hashes)
                ran += int(executed)
                input_inventory.check()
                check_small_inputs(plan)
            path, records = results_file(active)
            if active_stage == "scan":
                selection = collect(plan["plan_path"], path, folder / "selection", validated_plan=plan,
                                    checkpoint_hasher=hashes)
            else:
                verified = [validate_result(row, task, active, manifest, checkpoint_hasher=hashes)
                            for row, task in zip(records, active["tasks"])]
                means = {key: math.fsum(row[key] for row in verified) / 5 for key in METRICS}
                summary = {"schema_version": 1, "status": "synthetic_complete" if plan["synthetic_test"] else "validation_complete",
                           "synthetic_test": plan["synthetic_test"], "stage": "combined", "seeds": SEEDS,
                           "plan_digest": active["plan_digest"], "selection_digest": digest(selection),
                           "profile": selection["combined_profile"], "verified_results": verified,
                           "mean": means, "sample_sd": {key: math.sqrt(math.fsum((row[key] - means[key]) ** 2 for row in verified) / 4) for key in METRICS},
                           "macro_f1_seed_bootstrap_95": exact_seed_bootstrap([row["macro_f1"] for row in verified]),
                           "joint_optimum_established": False}
                immutable_json(Path(active["plan_path"]).parent / "validation_summary.json", summary)
        return {"status": "synthetic_complete" if plan["synthetic_test"] else "validation_complete",
                "stage": stage, "executed_tasks": ran, "plan_path": plan["plan_path"],
                "selection_path": str(folder / "selection" / "selection.json"),
                "combined_validation_path": str(folder / "combined" / "validation_summary.json") if stage != "scan" else None,
                "synthetic_test": plan["synthetic_test"]}
