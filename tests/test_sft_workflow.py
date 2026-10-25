"""SFT preparation and evaluation contracts, using synthetic data and real PNGs.

Preparation subprocesses run with -S and need no training packages. The shell
launcher test captures a stand-in executable's environment, not a GPU run.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
from types import SimpleNamespace
import zlib

import pytest


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 2026, 3407, 1234, 5678]
PHASES = ("a", "p", "v")


def write_png(path, color):
    """Write a valid 8x8 RGB PNG, without an optional imaging dependency."""
    def chunk(kind, payload):
        checksum = zlib.crc32(kind + payload) & 0xffffffff
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    pixels = b"".join(b"\0" + bytes(color) * 8 for _ in range(8))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def source_row(root, uid, label):
    image_names = []
    for index, phase in enumerate(PHASES):
        name = f"images/{uid} {phase}.png"
        write_png(root / name, (index * 60, 30, 90))
        image_names.append(name)
    return {
        "metadata": {"logical_key": uid, "slice_index": 0, "patient_id": uid,
                     "statistical_patient_id": "stat-" + uid, "positive": int(label is not None)},
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Synthetic test context."}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe the a/p/v images as structured detections."},
                *[{"type": "image", "image": name} for name in image_names]]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps({"detections": [
                {"phase": phase, "label": label or "no-lesion",
                 "bbox_2d": [100, 100, 800, 800] if label else [-1, -1, -1, -1]}
                for phase in PHASES]})}]},
        ],
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def load_config(path):
    # The emitted YAML uses only JSON scalars/collections. Reading those values
    # here keeps the preparation contract tests free of a PyYAML dependency.
    return {key: json.loads(value.strip()) for key, value in
            (line.split(":", 1) for line in Path(path).read_text().splitlines() if line.strip())}


@pytest.fixture
def fixture(tmp_path):
    work = tmp_path / "working directory with spaces"
    work.mkdir()
    model = work / "local model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    train = [source_row(work, "train-1", "HCC"), source_row(work, "train-2", "ICC"),
             source_row(work, "train-background", None)]
    validation = [source_row(work, "val-1", "HCC"), source_row(work, "val-2", "ICC")]
    test = [source_row(work, "test-1", "HCC"), source_row(work, "test-2", "ICC")]
    paths = {name: work / (name + ".jsonl") for name in ("train", "validation", "test")}
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_jsonl(paths[name], rows)
    manifest = {"schema_version": 1, "classes": ["HCC", "ICC"], "patients": [], "input_groups": []}
    group_map = work / "test group map.csv"
    with group_map.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["logical_key", "slice_index", "group_id", "patient_id", "source_patient_id"])
        writer.writeheader()
        for row in test:
            meta = row["metadata"]
            uid, patient = meta["logical_key"], meta["statistical_patient_id"]
            target = json.loads(row["messages"][-1]["content"][0]["text"])["detections"]
            manifest["patients"].append({"patient_id": patient, "truth": target[0]["label"]})
            manifest["input_groups"].append({"group_id": uid + ":0000", "patient_id": patient, "phases": {
                phase: {"image_ref": str(work / f"images/{uid} {phase}.png"), "ground_truth": [
                    {"label": item["label"], "bbox_2d": item["bbox_2d"]} for item in target if item["phase"] == phase]}
                for phase in PHASES}})
            writer.writerow({"logical_key": uid, "slice_index": 0, "group_id": uid + ":0000",
                             "patient_id": patient, "source_patient_id": meta["patient_id"]})
    manifest_path = work / "test manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(work=work, model=model, train=train, validation=validation, test=test,
                           paths=paths, manifest=manifest_path, group_map=group_map,
                           artifacts=work / "artifacts", checkpoints=work / "checkpoints")


def prepare(fixture, *, extra=(), evaluate=False, succeeds=True):
    args = [sys.executable, "-S", str(ROOT / "training/prepare_sft_runs.py"),
            "--model", fixture.model, "--data-mode", "private", "--train-json", fixture.paths["train"],
            "--validation-json", fixture.paths["validation"], "--data-root", fixture.work,
            "--artifacts", fixture.artifacts, "--checkpoints", fixture.checkpoints]
    if evaluate:
        args += ["--eval-json", fixture.paths["test"], "--eval-manifest", fixture.manifest,
                 "--eval-group-map", fixture.group_map]
    result = subprocess.run([*map(str, args), *map(str, extra)], cwd=fixture.work,
                            capture_output=True, text=True, timeout=30)
    if succeeds:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode != 0, result.stdout
    return result


def script_commands(path):
    return [shlex.split(line) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith(("#", "set "))]


def option(command, flag):
    return command[command.index(flag) + 1]


def run_generated_python(command, cwd):
    assert command[0] == "python3"
    return subprocess.run([sys.executable, *command[1:]], cwd=cwd,
                          capture_output=True, text=True, timeout=30)


def test_cli_exports_every_record_and_preserves_five_seed_schedule(fixture):
    prepared = prepare(fixture)
    assert prepared == json.loads((fixture.artifacts / "sft_runs.json").read_text())
    assert prepared["status"] == "prepared_not_trained"
    assert prepared["fitting_groups"] == 3 and prepared["validation_groups"] == 2
    assert prepared["effective_group_batch"] == 128
    assert prepared["checkpoint_selection"] == "final_epoch"
    assert prepared["validation_used_for_training"] is False
    assert [run["seed"] for run in prepared["runs"]] == SEEDS
    assert not fixture.checkpoints.exists()
    for path, expected_hash in prepared["sources_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash
    for run in prepared["runs"]:
        config = load_config(run["config"])
        assert config["seed"] == config["data_seed"] == run["seed"] == run["data_seed"]
        assert config["bf16"] is True and config["fp16"] is False
        assert config["per_device_train_batch_size"] == 2
        assert config["gradient_accumulation_steps"] == 16
        assert config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] * prepared["data_parallel_size"] == 128
        assert config["num_train_epochs"] == 6 and config["learning_rate"] == 1e-5
        assert config["lr_scheduler_type"] == "constant" and config["warmup_ratio"] == 0
        assert config["overwrite_output_dir"] is False and config["overwrite_cache"] is False
        assert config["output_dir"] == run["checkpoint_output"]
        assert config["dataset"] == "private_sft_train"
        assert "eval_dataset" not in config and "val_size" not in config
        assert run["evaluation_script"] is None
        data_dir = Path(config["dataset_dir"])
        info = json.loads((data_dir / "dataset_info.json").read_text())
        for name, original in (("private_sft_train", fixture.train), ("private_sft_val", fixture.validation)):
            exported = json.loads((data_dir / info[name]["file_name"]).read_text())
            assert len(exported) == len(original)
            for row, source in zip(exported, original):
                assert len(row["images"]) == 3
                assert all(Path(image).is_absolute() and Path(image).is_file() for image in row["images"])
                assert row["messages"][0] == {"role": "system", "content": "Synthetic test context."}
                assert row["messages"][1]["content"].count("<image>") == 3
                assert row["messages"][-1]["content"] == source["messages"][-1]["content"][0]["text"]
        assert hashlib.sha256(Path(run["config"]).read_bytes()).hexdigest() == run["config_sha256"]


def test_custom_seed_precision_and_parallel_environment_reach_shell_command(fixture):
    custom = [7, 11, 13, 17, 19]
    prepared = prepare(fixture, extra=["--seeds", *custom, "--precision", "fp32",
                                      "--data-parallel-size", 2, "--per-device-batch", 3, "--accumulation", 5])
    assert [run["seed"] for run in prepared["runs"]] == custom
    assert prepared["effective_group_batch"] == 30
    for run in prepared["runs"]:
        config = load_config(run["config"])
        assert config["seed"] == config["data_seed"] == run["seed"]
        assert config["bf16"] is False and config["fp16"] is False
        assert config["gradient_accumulation_steps"] == 5
    mock_bin = fixture.work / "stub bin"
    mock_bin.mkdir()
    capture = fixture.work / "captured launcher.json"
    executable = mock_bin / "llamafactory-cli"
    executable.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['SFT_CAPTURE_PATH']).write_text(json.dumps({'argv': sys.argv[1:], "
        "'environment': {k: os.environ.get(k) for k in ['FORCE_TORCHRUN', 'NNODES', 'NPROC_PER_NODE']}}))\n"
    )
    executable.chmod(0o755)
    env = {**os.environ, "PATH": str(mock_bin) + os.pathsep + os.environ.get("PATH", ""),
           "SFT_CAPTURE_PATH": str(capture), "FORCE_TORCHRUN": "0", "NNODES": "9", "NPROC_PER_NODE": "99"}
    run = prepared["runs"][0]
    result = subprocess.run(["bash", run["train_script"]], cwd=fixture.work, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    captured = json.loads(capture.read_text())
    assert captured["argv"] == ["train", run["config"]]
    assert captured["environment"] == {"FORCE_TORCHRUN": "1", "NNODES": "1", "NPROC_PER_NODE": "2"}


@pytest.mark.parametrize("family,entry", [("qwen", "infer_qwen_real961.py"), ("other_hf", "infer_hf_vlm_revision.py")])
def test_generated_eval_entries_exist_and_cli_help_accepts_their_flags(fixture, family, entry):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    prepared = prepare(fixture, evaluate=True, extra=["--model-family", family])
    for run in prepared["runs"]:
        commands = script_commands(run["evaluation_script"])
        assert len(commands) == 3
        assert Path(commands[0][1]).name == entry
        assert option(commands[0], "--model-dir") == run["checkpoint_output"]
        assert option(commands[0], "--seed") == str(run["seed"])
        assert option(commands[0], "--decoding") == "greedy"
        assert option(commands[0], "--max-new-tokens") == "256"
        assert option(commands[1], "--csv") == option(commands[0], "--output")
        assert option(commands[2], "--predictions") == option(commands[1], "--output")
        assert option(commands[2], "--model-seed") == str(run["seed"])
        assert option(commands[2], "--model-name") == family + " SFT"
        assert option(commands[1], "--manifest") == option(commands[2], "--manifest") == str(fixture.manifest)
        assert option(commands[1], "--group-map") == str(fixture.group_map)
    # Real entry points and argument parsers run; no model is loaded for --help.
    for command in script_commands(prepared["runs"][0]["evaluation_script"]):
        assert Path(command[1]).is_file()
        result = run_generated_python([*command, "--help"], fixture.work)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_generated_conversion_and_evaluation_keep_invalid_completions_and_seed_identity(fixture):
    prepared = prepare(fixture, evaluate=True)
    run = prepared["runs"][0]
    infer, convert, evaluate = script_commands(run["evaluation_script"])
    output = Path(option(infer, "--output"))
    # Synthetic inference output exercises the real downstream commands. This
    # does not invoke a model or represent an experimental prediction artifact.
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["logical_key", "slice_index", "patient_id", "avg_nll", "raw"])
        writer.writeheader()
        for index, row in enumerate(fixture.test):
            meta = row["metadata"]
            writer.writerow({"logical_key": meta["logical_key"], "slice_index": 0,
                             "patient_id": meta["patient_id"], "avg_nll": "0.2",
                             "raw": row["messages"][-1]["content"][0]["text"] if index == 0 else "invalid completion"})
    for command in (convert, evaluate):
        result = run_generated_python(command, fixture.work)
        assert result.returncode == 0, result.stderr
    predictions = [json.loads(line) for line in Path(option(convert, "--output")).read_text().splitlines()]
    assert len(predictions) == 2 and predictions[1]["raw"] == "invalid completion"
    metrics = json.loads((Path(option(evaluate, "--output-dir")) / "metrics.json").read_text())
    assert metrics["run_identity"] == {"model_name": "qwen SFT", "seed": 42}
    assert metrics["classification"]["accuracy"] == .5
    assert len(metrics["patients"]) == len(metrics["patient_detection"]) == 2
    assert metrics["patients"][1]["prediction"] is None


def test_cli_refuses_to_overwrite_prepared_artifacts(fixture):
    prepared = prepare(fixture)
    paths = [Path(run["config"]) for run in prepared["runs"]]
    original = {path: path.read_bytes() for path in paths}
    failed = prepare(fixture, extra=["--learning-rate", "0.01"], succeeds=False)
    assert "new artifacts directory" in failed.stderr
    assert all(path.read_bytes() == content for path, content in original.items())


def test_cli_checks_all_checkpoint_targets_before_creating_any_artifact(fixture):
    existing = fixture.checkpoints / "seed_5678"
    existing.mkdir(parents=True)
    sentinel = existing / "existing-training-record.txt"
    sentinel.write_text("preserve\n")
    failed = prepare(fixture, succeeds=False)
    assert "Checkpoint output already exists" in failed.stderr
    assert sentinel.read_text() == "preserve\n"
    assert not fixture.artifacts.exists()


@pytest.mark.parametrize("kind", ["patient", "statistical_patient", "image", "duplicate_group"])
def test_cli_rejects_split_leakage_and_duplicate_groups_before_writing(fixture, kind):
    if kind == "patient":
        fixture.validation[0]["metadata"]["patient_id"] = fixture.train[0]["metadata"]["patient_id"]
    elif kind == "statistical_patient":
        fixture.validation[0]["metadata"]["statistical_patient_id"] = fixture.train[0]["metadata"]["statistical_patient_id"]
    elif kind == "image":
        fixture.validation[0]["messages"][1]["content"][1]["image"] = fixture.train[0]["messages"][1]["content"][1]["image"]
    else:
        fixture.train.append(copy.deepcopy(fixture.train[0]))
    write_jsonl(fixture.paths["train"], fixture.train)
    write_jsonl(fixture.paths["validation"], fixture.validation)
    failed = prepare(fixture, succeeds=False)
    assert any(word in failed.stderr for word in ("overlap", "reuse image", "Duplicate"))
    assert not fixture.artifacts.exists()


@pytest.mark.parametrize("kind", ["two_images", "four_images", "missing_image", "invalid_json", "bad_bbox",
                                  "unknown_class", "missing_phase", "extra_assistant_text", "extra_exchange"])
def test_cli_rejects_invalid_record_instead_of_silently_filtering_it(fixture, kind):
    row = fixture.train[1]
    content = row["messages"][1]["content"]
    assistant = row["messages"][-1]["content"]
    if kind == "two_images":
        content.pop()
    elif kind == "four_images":
        content.append(copy.deepcopy(content[-1]))
    elif kind == "missing_image":
        content[-1]["image"] = "does-not-exist.png"
    elif kind == "invalid_json":
        assistant[0]["text"] = "not-json"
    elif kind == "extra_assistant_text":
        assistant.append({"type": "text", "text": "This unvalidated suffix is not part of a JSON target."})
    elif kind == "extra_exchange":
        row["messages"].extend(copy.deepcopy(row["messages"][1:]))
    else:
        answer = json.loads(assistant[0]["text"])
        if kind == "bad_bbox":
            answer["detections"][0]["bbox_2d"] = [100, 100, 50, 50]
        elif kind == "unknown_class":
            answer["detections"][0]["label"] = "not-a-class"
        else:
            answer["detections"].pop()
        assistant[0]["text"] = json.dumps(answer)
    write_jsonl(fixture.paths["train"], fixture.train)
    prepare(fixture, succeeds=False)
    assert not fixture.artifacts.exists()


@pytest.mark.parametrize("flags", [
    ["--seeds", "42", "42", "3407", "1234", "5678"],
    ["--per-device-batch", "0"], ["--data-parallel-size", "0"], ["--accumulation", "0"],
    ["--epochs", "nan"], ["--epochs", "inf"], ["--learning-rate", "nan"], ["--learning-rate", "inf"],
    ["--warmup-ratio", "nan"], ["--warmup-ratio", "1.5"], ["--warmup-ratio", "-0.1"],
])
def test_cli_rejects_invalid_training_parameters(fixture, flags):
    prepare(fixture, extra=flags, succeeds=False)
    assert not fixture.artifacts.exists()


def test_cli_requires_the_entire_optional_evaluation_bundle(fixture):
    failed = prepare(fixture, extra=["--eval-json", fixture.paths["test"]], succeeds=False)
    assert "all three evaluation paths" in failed.stderr
    assert not fixture.artifacts.exists()
