"""Synthetic CPU execution, exact seed bootstrap and OAT plotting checks."""
import itertools
import csv
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from test_rars_oat import synthetic_config
from training.rars_joint_grid import SEEDS, load_config, sha, write_json
from training.rars_oat import build_plan, exact_seed_bootstrap, read_json
from training.rars_oat_execution import CheckpointHashes, InputInventory, run_workflow
from visualization.rars_oat import load_oat_selection, plot_oat


def executable_config(tmp_path):
    path, config, manifest = synthetic_config(tmp_path)
    backend = Path(__file__).parent / "helpers/rars_oat_synthetic_backend.py"
    config["training_command_argv"] = [sys.executable, str(backend), "train", "--synthetic-test", "--seed", "{seed}",
        "--profile", "{profile_path}", "--train", "{train_file}", "--model", "{model_path}", "--output", "{output_dir}"]
    config["inference_command_argv"] = [sys.executable, str(backend), "infer", "--synthetic-test", "--seed", "{seed}",
        "--checkpoint", "{checkpoint}", "--manifest", "{validation_manifest}", "--output", "{output_path}", "--task", "{task_path}"]
    config["inference_format"] = "canonical_jsonl"
    write_json(path, config)
    return path


def change_file_metadata(path, change):
    before = path.stat()
    if change == "chmod":
        os.chmod(path, before.st_mode ^ stat.S_IXUSR)
    else:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    after = path.stat()
    assert after.st_size == before.st_size and after.st_ino == before.st_ino
    if change == "chmod":
        assert after.st_mtime_ns == before.st_mtime_ns
        assert after.st_ctime_ns != before.st_ctime_ns
    else:
        assert after.st_mtime_ns != before.st_mtime_ns


def bound_input_inventory(tmp_path):
    path = executable_config(tmp_path)
    raw = read_json(path)
    image = tmp_path / "image.png"
    image.write_bytes(b"SYNTHETIC image A")
    inference = tmp_path / "inference.jsonl"
    inference.write_text(json.dumps({"images": [str(image)]}) + "\n")
    raw["inference_input"] = str(inference)
    write_json(path, raw)
    config = load_config(path)
    plan = build_plan(path)
    inventory = InputInventory(config, plan["identity"]["execution"], plan=plan)
    return inventory, {"model": Path(config["seed_model_paths"]["42"]) / "synthetic_weights.txt",
                       "training": Path(config["seed_train_files"]["42"]),
                       "validation": Path(config["validation_manifest"]),
                       "reserved": Path(config["test_identity_manifest"]),
                       "image": image, "inference": inference}


@pytest.mark.parametrize("change", ["chmod", "utime"])
def test_checkpoint_hashes_refresh_metadata_without_rehashing_stable_files(tmp_path, monkeypatch, change):
    from training import rars_oat
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"SYNTHETIC weights")
    original_digest = rars_oat.checkpoint_digest
    calls = []
    def counted_digest(path):
        calls.append(path)
        return original_digest(path)
    monkeypatch.setattr(rars_oat, "checkpoint_digest", counted_digest)
    hashes = CheckpointHashes()
    expected = hashes(checkpoint)
    assert hashes(checkpoint) == expected and len(calls) == 1
    before = sha(weights)
    change_file_metadata(weights, change)
    assert sha(weights) == before
    assert hashes(checkpoint) == expected and len(calls) == 2
    assert hashes(checkpoint) == expected and len(calls) == 2


@pytest.mark.parametrize("change", ["content", "delete", "add", "rename"])
def test_checkpoint_hashes_reject_changed_bytes_and_file_names(tmp_path, change):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"SYNTHETIC weights A")
    (checkpoint / "config.json").write_text("{}")
    hashes = CheckpointHashes()
    hashes(checkpoint)
    if change == "content":
        before = weights.stat()
        weights.write_bytes(b"SYNTHETIC weights B")
        os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "delete":
        weights.unlink()
    elif change == "add":
        (checkpoint / "new.bin").write_bytes(weights.read_bytes())
    else:
        weights.rename(checkpoint / "renamed.bin")
    for _ in range(2):
        with pytest.raises(ValueError, match="Checkpoint changed during execution"):
            hashes(checkpoint)


@pytest.mark.parametrize("cached", [False, True])
def test_checkpoint_hashes_reject_changes_during_hashing(tmp_path, monkeypatch, cached):
    from training import rars_oat
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"SYNTHETIC weights")
    hashes = CheckpointHashes()
    if cached:
        hashes(checkpoint)
        change_file_metadata(weights, "utime")
    original_digest = rars_oat.checkpoint_digest
    def changing_digest(path):
        value = original_digest(path)
        weights.write_bytes(b"CHANGED weights")
        return value
    monkeypatch.setattr(rars_oat, "checkpoint_digest", changing_digest)
    with pytest.raises(ValueError, match="Checkpoint changed during execution"):
        hashes(checkpoint)


@pytest.mark.parametrize("change", ["chmod", "utime"])
@pytest.mark.parametrize("source_kind", ["model", "training", "validation", "reserved", "image", "inference"])
def test_input_inventory_revalidates_plan_only_after_metadata_changes(tmp_path, monkeypatch, source_kind, change):
    from training import rars_oat
    inventory, paths = bound_input_inventory(tmp_path)
    original_build = rars_oat.build_plan
    calls = []
    def counted_build(path):
        calls.append(path)
        return original_build(path)
    monkeypatch.setattr(rars_oat, "build_plan", counted_build)
    inventory.check()
    assert calls == []
    before = sha(paths[source_kind])
    change_file_metadata(paths[source_kind], change)
    assert sha(paths[source_kind]) == before
    inventory.check()
    assert len(calls) == 1
    inventory.check()
    assert len(calls) == 1


@pytest.mark.parametrize("source_kind", ["model", "training", "validation", "reserved", "image", "inference"])
def test_input_inventory_rejects_changed_bound_contents(tmp_path, source_kind):
    inventory, paths = bound_input_inventory(tmp_path)
    path = paths[source_kind]
    path.write_bytes(path.read_bytes() + b"\n")
    for _ in range(2):
        with pytest.raises(ValueError, match="changed during execution"):
            inventory.check()


@pytest.mark.parametrize("change", ["delete", "add", "rename"])
def test_input_inventory_rejects_changed_model_file_names(tmp_path, change):
    inventory, paths = bound_input_inventory(tmp_path)
    weights = paths["model"]
    if change == "delete":
        weights.unlink()
    elif change == "add":
        weights.with_name("new.bin").write_bytes(weights.read_bytes())
    else:
        weights.rename(weights.with_name("renamed.bin"))
    with pytest.raises(ValueError, match="changed during execution"):
        inventory.check()


def test_input_inventory_rejects_changes_during_plan_revalidation(tmp_path, monkeypatch):
    from training import rars_oat
    inventory, paths = bound_input_inventory(tmp_path)
    weights = paths["model"]
    change_file_metadata(weights, "utime")
    original_build = rars_oat.build_plan
    def changing_build(path):
        plan = original_build(path)
        weights.write_bytes(b"CHANGED weights")
        return plan
    monkeypatch.setattr(rars_oat, "build_plan", changing_build)
    with pytest.raises(ValueError, match="changed during execution"):
        inventory.check()


def test_exact_bootstrap_matches_independent_multinomial_enumeration():
    values = [0.02, 0.19, 0.44, 0.68, 0.97]
    expected = []
    # An independent enumeration of 126 count vectors, weighted by their
    # multinomial multiplicities, is equivalent to 3125 ordered draws.
    for counts in itertools.product(range(6), repeat=5):
        if sum(counts) != 5:
            continue
        multiplicity = math.factorial(5) // math.prod(math.factorial(n) for n in counts)
        mean = math.fsum(value * count for value, count in zip(values, counts)) / 5
        expected.extend([mean] * multiplicity)
    expected.sort()
    assert len(expected) == 3125
    def linear(p):
        position = (len(expected) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return expected[low] + (position-low) * (expected[high]-expected[low])
    result = exact_seed_bootstrap(values)
    assert result["lower"] == pytest.approx(linear(.025), abs=1e-15)
    assert result["upper"] == pytest.approx(linear(.975), abs=1e-15)
    assert result["resamples"] == 3125 and result["unit"] == "training_seed" and result["quantile"] == "linear"
    for invalid in ([1]*4, [1]*6, [0, 0, 0, 0, float("nan")]):
        with pytest.raises(ValueError): exact_seed_bootstrap(invalid)


def test_cpu_commands_resume_collect_plot_and_separate_combined_five_seed_run(tmp_path):
    path = executable_config(tmp_path)
    first = run_workflow(path, max_tasks=2)
    assert first["status"] == "incomplete" and first["executed_tasks"] == 2
    plan = read_json(first["plan_path"])
    first_result = Path(plan["tasks"][0]["task_dir"]) / "result.json"
    first_mtime = first_result.stat().st_mtime_ns
    with pytest.raises(ValueError, match="all completed scan"):
        run_workflow(path, stage="combined")
    completed = run_workflow(path)
    assert completed["status"] == "synthetic_complete" and completed["executed_tasks"] == 358
    assert first_result.stat().st_mtime_ns == first_mtime
    selection = load_oat_selection(completed["selection_path"])
    assert len(selection["verified_results"]) == 355
    assert all(point["value"] == .3 for point in selection["selected_axis_points"].values())
    for curve in selection["curves"].values():
        assert len(curve) == 15
        assert all(len(point["seed_metrics"]) == 5 for point in curve)
        assert all(set(row) == {"seed", "accuracy", "macro_recall", "macro_f1"} for point in curve for row in point["seed_metrics"])
    combined = read_json(completed["combined_validation_path"])
    assert len(combined["verified_results"]) == 5 and combined["seeds"] == SEEDS
    assert all(row["task_id"].startswith("combined:") for row in combined["verified_results"])
    assert combined["mean"] == {"accuracy": .8, "macro_recall": .8, "macro_f1": .8}
    assert combined["profile"] == selection["combined_profile"]
    assert run_workflow(path)["executed_tasks"] == 0
    pytest.importorskip("matplotlib")
    output = plot_oat(completed["selection_path"], tmp_path / "SYNTHETIC_oat.pdf")
    assert output.read_bytes().startswith(b"%PDF")
    point = next(iter(selection["curves"].values()))[0]
    point["macro_f1_seed_bootstrap_95"]["unit"] = "patient"
    altered = tmp_path / "altered.json"
    write_json(altered, selection)
    with pytest.raises(ValueError, match="3125 ordered seed"):
        load_oat_selection(altered)
    # Completed task identities and checkpoint bytes are rechecked on resume.
    result = read_json(first_result)
    weights = Path(result["checkpoint"]) / "SYNTHETIC_NOT_MODEL_WEIGHTS.txt"
    weights.write_text("CHANGED SYNTHETIC CHECKPOINT")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        run_workflow(path)


def test_failed_training_command_never_generates_receipt_or_result(tmp_path):
    path = executable_config(tmp_path)
    config = read_json(path)
    config["training_command_argv"][0] = "/usr/bin/false"
    write_json(path, config)
    with pytest.raises(RuntimeError, match="exited"):
        run_workflow(path, max_tasks=1)
    assert not list((tmp_path / "runs").rglob("result.json"))
    assert not list((tmp_path / "runs").rglob("inference_receipt.json"))


@pytest.mark.parametrize("source_kind", ["inference", "training"])
def test_referenced_image_bytes_bind_plan_and_live_inventory(tmp_path, source_kind):
    from training.rars_joint_grid import load_config
    from training.rars_oat import build_plan, collect
    from training.rars_oat_execution import InputInventory
    path = executable_config(tmp_path)
    config = read_json(path)
    images = tmp_path / "images"
    images.mkdir()
    image = images / "validation.png"
    image.write_bytes(b"SYNTHETIC image A")
    config["image_root"] = str(images)
    if source_kind == "inference":
        source = tmp_path / "inference.jsonl"
        source.write_text(json.dumps({"messages": [{"role": "user", "content": [
            {"type": "image", "image": image.name}]}]}) + "\n")
        config["inference_input"] = str(source)
    else:
        for filename in config["seed_train_files"].values():
            source = Path(filename)
            row = json.loads(source.read_text())
            row["images"] = [{"image": image.name}]*3
            source.write_text(json.dumps(row) + "\n")
    write_json(path, config)
    first = run_workflow(path, max_tasks=1)
    plan = read_json(first["plan_path"])
    inventory = InputInventory(load_config(path), plan["identity"]["execution"], plan=plan)
    image.write_bytes(b"SYNTHETIC image B")
    with pytest.raises(ValueError, match="changed during execution"):
        inventory.check()
    assert build_plan(path)["plan_digest"] != plan["plan_digest"]
    with pytest.raises(ValueError, match="changed"):
        collect(first["plan_path"], tmp_path / "unused.jsonl", tmp_path / "selection")
    second = run_workflow(path, max_tasks=1)
    assert second["plan_path"] != first["plan_path"] and second["executed_tasks"] == 1
    image.unlink()
    with pytest.raises(ValueError, match="image file is missing"):
        build_plan(path)
    with pytest.raises(ValueError, match="changed during execution"):
        inventory.check()


def test_csv_inference_import_and_raw_source_tamper_rejected_on_resume(tmp_path):
    path = executable_config(tmp_path)
    config = read_json(path)
    manifest = read_json(config["validation_manifest"])
    mapping = tmp_path / "group_map.csv"
    with mapping.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["logical_key", "slice_index", "group_id", "patient_id"])
        writer.writeheader()
        for group in manifest["input_groups"]:
            writer.writerow({"logical_key": group["group_id"], "slice_index": 0,
                             "group_id": group["group_id"], "patient_id": group["patient_id"]})
    config.update(inference_format="inference_csv", group_map=str(mapping))
    config["inference_command_argv"].append("--csv")
    write_json(path, config)
    script = Path(__file__).resolve().parents[1] / "training/rars_oat.py"
    process = subprocess.run([sys.executable, str(script), "run", "--config", str(path), "--max-tasks", "1"],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    status = json.loads(process.stdout)
    assert status["status"] == "incomplete" and status["executed_tasks"] == 1
    task = read_json(status["plan_path"])["tasks"][0]
    folder = Path(task["task_dir"])
    assert (folder / "predictions.jsonl.provenance.json").is_file()
    record = read_json(folder / "result.json")
    assert record["metrics"] == {"accuracy": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
    raw = folder / "inference_predictions.csv"
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="raw output"):
        run_workflow(path, max_tasks=1)
