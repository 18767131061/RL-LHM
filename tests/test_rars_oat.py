"""All training/checkpoint/score fixtures in this module are explicitly synthetic.

The collector runs the real protocol evaluator; no fixture is experimental evidence.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.rars_joint_grid import LABELS, SEEDS, VALUES, candidate, sha, validate_profile, write_json
from training.rars_oat import (build_plan, checkpoint_digest, collect, decode_json, oat_design,
                               plan_search, result_template, inference_receipt_template, select_axes, validate_result)


def synthetic_config(tmp_path, cohort="private"):
    files, models = {}, {}
    for seed in SEEDS:
        path = tmp_path / f"cache-{seed}.jsonl"
        path.write_text(json.dumps({"extra_info": {"sample_uid": "fit-group", "patient_id": "fit-person",
                                                   "visual_z": [1., 0.]}}) + "\n")
        files[str(seed)] = str(path)
        model = tmp_path / f"sft-{seed}"
        model.mkdir()
        (model / "synthetic_weights.txt").write_text(f"NOT MODEL WEIGHTS: test seed {seed}")
        models[str(seed)] = str(model)
    labels = LABELS[cohort]
    manifest = {"schema_version": 1, "split": "validation", "split_id": "P1" if cohort == "private" else "U1",
                "classes": labels, "patients": [{"patient_id": f"val-{i}", "truth": label} for i, label in enumerate(labels)],
                "input_groups": [{"group_id": f"val-group-{i}", "patient_id": f"val-{i}", "phases": {
                    ph: {"image_ref": f"synthetic/{i}/{ph}.png", "ground_truth": [{"label": label, "bbox_2d": [0, 0, 10, 10]}]}
                    for ph in ("a", "p", "v")}} for i, label in enumerate(labels)]}
    write_json(tmp_path / "validation.json", manifest)
    write_json(tmp_path / "reserved.json", {"split": "test", "patients": [{"patient_id": "reserved-person"}]})
    config = {"protocol": "main", "cohort": cohort, "split_id": manifest["split_id"],
              "selection_split": "validation", "formal_test_used_for_selection": False, "synthetic_test": True,
              "seeds": SEEDS, "values": VALUES, "train_file": files["42"], "seed_train_files": files,
              "seed_model_paths": models, "validation_manifest": str(tmp_path / "validation.json"),
              "test_identity_manifest": str(tmp_path / "reserved.json"), "output_root": str(tmp_path / "runs"),
              "training_command_argv": ["explicit-test-command-never-run", "--seed", "{seed}", "--profile", "{profile_path}",
                                        "--train", "{train_file}", "--model", "{model_path}", "--output", "{output_dir}"]}
    path = tmp_path / "config.json"
    write_json(path, config)
    return path, config, manifest


def synthetic_result(task, plan, manifest, correct=True):
    """Synthetic test receipts and small files, marked synthetic_test=True."""
    assert plan["synthetic_test"] is True
    folder = Path(task["task_dir"])
    train = Path(task["output_dir"])
    checkpoint = train / "global_step_6"
    checkpoint.mkdir(parents=True)
    (checkpoint / "synthetic_weights.txt").write_text("NOT A TRAINED MODEL: " + task["task_id"])
    run = {"seed": task["seed"], "variant": "full_rl_lhm", "configuration_only": False, "smoke": False,
           "train_file_sha256": plan["identity"]["seed_train_sha256"][str(task["seed"])] ,
           "model_path": task["model_path"], "risk_profile_sha256": sha(task["profile_path"]),
           "runtime_environment": {"RL_LHM_RISK_PROFILE": task["profile_path"]},
           "command": ["synthetic-training-not-executed",
                       "++ray_kwargs.ray_init.runtime_env.env_vars.RL_LHM_RISK_PROFILE=" + json.dumps(task["profile_path"])]}
    write_json(train / "run_manifest.json", run)
    done = {"seed": task["seed"], "variant": "full_rl_lhm", "smoke": False, "status": "training_complete",
            "run_manifest_sha256": sha(train / "run_manifest.json"), "final_checkpoint": checkpoint.name}
    write_json(train / "training_complete.json", done)
    predictions = folder / "synthetic_predictions.jsonl"
    rows = []
    for index, group in enumerate(manifest["input_groups"]):
        label = manifest["classes"][index if correct else (index + 1) % len(manifest["classes"])]
        rows.append({"group_id": group["group_id"], "patient_id": group["patient_id"], "partition": "validation",
                     "raw": json.dumps({"detections": [{"phase": ph, "label": label, "bbox_2d": [0, 0, 10, 10]}
                                                       for ph in ("a", "p", "v")]}), "completion_nll": .2})
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    record = result_template(task, plan)
    record.update(source=str(predictions), source_sha256=sha(predictions), checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_digest(checkpoint), metrics={key: 1.0 if correct else 0.0 for key in ("accuracy", "macro_recall", "macro_f1")})
    receipt = inference_receipt_template(task, plan)
    receipt.update({key: record[key] for key in ("source", "source_sha256", "checkpoint", "checkpoint_sha256")})
    receipt["status"] = "inference_complete"
    receipt_path = folder / "synthetic_inference_receipt.json"
    write_json(receipt_path, receipt)
    record.update(inference_receipt=str(receipt_path), inference_receipt_sha256=sha(receipt_path))
    return record


@pytest.mark.parametrize("cohort,axes,points,profiles,tasks,records", [
    ("private", 5, 75, 71, 355, 375), ("public", 7, 105, 99, 495, 525)])
def test_exact_design_and_shared_baseline(cohort, axes, points, profiles, tasks, records):
    actual_axes, actual_profiles, actual_points = oat_design(cohort)
    assert (len(actual_axes), len(actual_points), len(actual_profiles)) == (axes, points, profiles)
    assert len(actual_profiles) * len(SEEDS) == tasks
    assert sum(len(point["task_ids"]) for point in actual_points) == records
    baselines = [point for point in actual_points if point["value"] == 1.0]
    assert len({point["profile_id"] for point in baselines}) == 1
    assert len({task for point in baselines for task in point["task_ids"]}) == 5
    for point in actual_points:
        profile = actual_profiles[point["profile_id"]]
        weights = dict(profile["pair_weights"], other=profile["other_weight"])
        assert weights[point["axis"]] == point["value"]
        assert all(value == 1.0 for axis, value in weights.items() if axis != point["axis"])


@pytest.mark.parametrize("other", [False, True, float("nan"), float("inf"), 0.0, .11, 1.6, "0.4"])
def test_oat_profile_rejects_invalid_background(other):
    profile = next(iter(oat_design("private")[1].values()))
    profile["other_weight"] = other
    with pytest.raises(ValueError, match="other_weight"):
        validate_profile(profile)


def test_oat_profile_remains_compatible_with_actual_reward_callback(tmp_path, monkeypatch):
    from rewards.rars import profile_class_space
    from rewards.verl_liver_reward import get_risk_severity
    for cohort, data_mode in (("private", "private"), ("public", "5cls")):
        profile = next(profile for profile in oat_design(cohort)[1].values() if profile["other_weight"] == 1.5)
        path = tmp_path / f"{cohort}.json"
        write_json(path, profile)
        monkeypatch.setenv("RL_LHM_RISK_PROFILE", str(path))
        assert set(profile_class_space(profile, cohort, LABELS[cohort])) == set(LABELS[cohort])
        background_pair = ("ICC", "FNH") if cohort == "private" else ("HCC", "BCLM")
        assert get_risk_severity(*background_pair, data_mode, "ordinal") == 1.5
        assert get_risk_severity("HCC", "ICC", data_mode, "ordinal") == 1.0
        assert get_risk_severity("ICC", "HCC", data_mode, "ordinal") == 1.0
    old = candidate({"cohort": "private", "values": VALUES}, 0)
    assert validate_profile(old)["other_weight"] == .4
    old["other_weight"] = 1.0
    with pytest.raises(ValueError, match="fixed other_weight"):
        validate_profile(old)


def test_actual_launcher_accepts_oat_profile_in_config_only_mode():
    from test_training_data_cli import TrainingDataCliTests
    fixture = TrainingDataCliTests()
    fixture.setUp()
    try:
        prepared, _ = fixture.convert()
        profile = next(profile for profile in oat_design("private")[1].values() if profile["other_weight"] == 1.5)
        profile_path = fixture.work / "synthetic_oat_profile.json"
        write_json(profile_path, profile)
        manifest = fixture.launch_config(prepared / "real961_rl_train.jsonl", "full_rl_lhm",
                                         extra=("--risk-profile", profile_path))
        assert manifest["risk_profile_sha256"] == sha(profile_path)
        assert manifest["runtime_environment"]["RL_LHM_RISK_PROFILE"] == str(profile_path)
        assert manifest["configuration_only"] is True
    finally:
        fixture.doCleanups()


def test_plan_cli_emits_inputs_without_running_training_and_is_idempotent(tmp_path):
    path, config, _ = synthetic_config(tmp_path)
    script = Path(__file__).resolve().parents[1] / "training/rars_oat.py"
    process = subprocess.run([sys.executable, str(script), "plan", "--config", str(path)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    plan = json.loads(Path(json.loads(process.stdout)["plan_path"]).read_text())
    assert plan["counts"] == {"axes": 5, "axis_points": 75, "axis_seed_records": 375, "unique_profiles": 71, "unique_tasks": 355}
    assert plan_search(path) == plan
    assert all(not Path(task["output_dir"]).exists() for task in plan["tasks"])
    task = plan["tasks"][0]
    template = json.loads((Path(task["task_dir"]) / "result_template.json").read_text())
    assert template["metrics"]["macro_f1"] is None
    assert "{" not in " ".join(task["training_command_argv"])
    assert task["training_command_argv"][0] == config["training_command_argv"][0]


@pytest.mark.parametrize("fault", ["missing_command", "missing_seed", "unknown_placeholder", "subset", "test_split", "test_overlap", "empty_model"])
def test_plan_rejects_unbound_or_invalid_inputs(tmp_path, fault):
    path, config, _ = synthetic_config(tmp_path)
    if fault == "missing_command": config.pop("training_command_argv")
    elif fault == "missing_seed": config["training_command_argv"].remove("{seed}")
    elif fault == "unknown_placeholder": config["training_command_argv"].append("{unknown}")
    elif fault == "subset": config["values"] = [.1, 1.]
    elif fault == "test_split": config["selection_split"] = "test"
    elif fault == "test_overlap": write_json(tmp_path / "reserved.json", {"split": "test", "patients": [{"patient_id": "val-0"}]})
    elif fault == "empty_model": (Path(config["seed_model_paths"]["42"]) / "synthetic_weights.txt").unlink()
    write_json(path, config)
    with pytest.raises(ValueError): plan_search(path)


@pytest.mark.parametrize("fault", ["test", "seed", "profile", "input_hash", "nonfinite", "missing_metric", "wrong_metric", "source_hash",
                                   "checkpoint_hash", "outside_checkpoint", "unfinished", "smoke", "test_source", "test_patient", "unknown_group",
                                   "missing_receipt", "receipt_seed", "receipt_task", "receipt_input", "receipt_checkpoint", "receipt_validation",
                                   "runtime_profile", "command_profile", "duplicate_command_profile"])
def test_result_validation_rejects_bad_evidence(tmp_path, fault):
    path, _, manifest = synthetic_config(tmp_path)
    plan = plan_search(path)
    task = plan["tasks"][0]
    record = synthetic_result(task, plan, manifest)
    if fault == "test": record["partition"] = "test"
    elif fault == "seed": record["seed"] = 7
    elif fault == "profile": record["profile_id"] = "wrong"
    elif fault == "input_hash": record["input_hash"] = "wrong"
    elif fault == "nonfinite": record["metrics"]["macro_f1"] = float("nan")
    elif fault == "missing_metric": record["metrics"] = {}
    elif fault == "wrong_metric": record["metrics"]["macro_f1"] = .3
    elif fault == "source_hash": record["source_sha256"] = "wrong"
    elif fault == "checkpoint_hash": record["checkpoint_sha256"] = "wrong"
    elif fault == "outside_checkpoint": record["checkpoint"] = task["model_path"]
    elif fault == "missing_receipt": record["inference_receipt"] = None
    elif fault in ("receipt_seed", "receipt_task", "receipt_input", "receipt_checkpoint", "receipt_validation"):
        receipt_path = Path(record["inference_receipt"])
        receipt = json.loads(receipt_path.read_text())
        key = {"receipt_seed": "seed", "receipt_task": "task_id", "receipt_input": "input_hash",
               "receipt_checkpoint": "checkpoint_sha256", "receipt_validation": "validation_sha256"}[fault]
        receipt[key] = 7 if key == "seed" else "wrong"
        write_json(receipt_path, receipt)
        record["inference_receipt_sha256"] = sha(receipt_path)
    elif fault in ("runtime_profile", "command_profile", "duplicate_command_profile"):
        run_path = Path(task["output_dir"]) / "run_manifest.json"
        run = json.loads(run_path.read_text())
        if fault == "runtime_profile": run["runtime_environment"]["RL_LHM_RISK_PROFILE"] = "wrong-profile.json"
        elif fault == "command_profile": run["command"][-1] = run["command"][-1].split("=", 1)[0] + '="wrong-profile.json"'
        else: run["command"].append(run["command"][-1])
        write_json(run_path, run)
        done_path = Path(task["output_dir"]) / "training_complete.json"
        done = json.loads(done_path.read_text())
        done["run_manifest_sha256"] = sha(run_path)
        write_json(done_path, done)
    elif fault in ("unfinished", "smoke"):
        done_path = Path(task["output_dir"]) / "training_complete.json"
        done = json.loads(done_path.read_text())
        done.update({"status": "running"} if fault == "unfinished" else {"smoke": True})
        write_json(done_path, done)
    elif fault in ("test_source", "test_patient", "unknown_group"):
        source = Path(record["source"])
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if fault == "test_source": rows[0]["partition"] = "test"
        elif fault == "test_patient": rows[0]["patient_id"] = "reserved-person"
        else: rows[0]["group_id"] = "reserved-test-group"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        record["source_sha256"] = sha(source)
    with pytest.raises(ValueError): validate_result(record, task, plan, manifest)


def test_predictions_cannot_be_borrowed_from_another_task_even_when_values_match(tmp_path):
    path, _, manifest = synthetic_config(tmp_path)
    plan = plan_search(path)
    first_task, second_task = plan["tasks"][:2]
    first = synthetic_result(first_task, plan, manifest)
    second = synthetic_result(second_task, plan, manifest)
    # Identical greedy predictions are legitimate when each run has its own
    # matching receipt; duplicate content hashes alone must not reject them.
    assert first["source_sha256"] == second["source_sha256"]
    validate_result(first, first_task, plan, manifest)
    validate_result(second, second_task, plan, manifest)
    # The old bug accepted this swap because the checkpoint and source hashes
    # were independently valid. A receipt must join them to the same task.
    second.update({key: first[key] for key in ("source", "source_sha256", "metrics")})
    with pytest.raises(ValueError, match="Inference receipt"):
        validate_result(second, second_task, plan, manifest)
    second.update({key: first[key] for key in ("inference_receipt", "inference_receipt_sha256")})
    with pytest.raises(ValueError, match="Inference receipt"):
        validate_result(second, second_task, plan, manifest)


def test_five_seed_average_ties_and_combination_are_explicitly_synthetic():
    axes, profiles, points = oat_design("private")
    tasks = [{"task_id": f"{pid}:{seed}"} for pid in profiles for seed in SEEDS]
    plan = {"cohort": "private", "axes": axes, "profiles": profiles, "axis_points": points, "tasks": tasks}
    # Synthetic arithmetic fixture: a single excellent seed at .2 must lose to
    # consistently better mean performance at .3; .3 and .4 have an exact tie.
    results = {task["task_id"]: {key: 0.1 for key in ("accuracy", "macro_recall", "macro_f1")} for task in tasks}
    for point in points:
        for seed in SEEDS:
            value = .5 if point["value"] in (.3, .4) else (.9 if point["value"] == .2 and seed == 42 else .1)
            results[f'{point["profile_id"]}:{seed}'].update({key: value for key in ("accuracy", "macro_recall", "macro_f1")})
    summary = select_axes(plan, results)
    assert all(point["value"] == .3 for point in summary["selected_axis_points"].values())
    assert summary["combined_profile"]["other_weight"] == .3
    assert set(summary["combined_profile"]["pair_weights"].values()) == {.3}
    assert summary["joint_optimum_established"] is False
    assert summary["combined_profile_requires_training"] is True
    assert summary["combined_profile_validation_metric"] is None


def test_complete_synthetic_collection_recomputes_predictions_and_retains_all_axis_mappings(tmp_path):
    path, _, manifest = synthetic_config(tmp_path)
    plan = plan_search(path)
    records = [synthetic_result(task, plan, manifest, correct=(task["seed"] != 42)) for task in plan["tasks"]]
    results_path = tmp_path / "synthetic_results.jsonl"
    results_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    output = tmp_path / "selection"
    summary = collect(plan["plan_path"], results_path, output)
    assert summary["status"] == "synthetic_complete"
    assert len(summary["verified_results"]) == 355
    assert sum(len(curve) for curve in summary["curves"].values()) == 75
    assert all(point["value"] == .1 and point["mean_validation_macro_f1"] == pytest.approx(.8)
               for point in summary["selected_axis_points"].values())
    assert json.loads((output / "combined_profile.json").read_text()) == summary["combined_profile"]
    # Repeated or incomplete input is rejected before collecting any partial selection.
    for bad in (records[:-1], records + [records[0]]):
        results_path.write_text("".join(json.dumps(row) + "\n" for row in bad))
        with pytest.raises(ValueError, match="Incomplete|Duplicate"):
            collect(plan["plan_path"], results_path, tmp_path / "should_not_exist")
    assert not (tmp_path / "should_not_exist").exists()


def test_changed_plan_inputs_are_rejected_before_collection(tmp_path):
    path, config, _ = synthetic_config(tmp_path)
    plan = plan_search(path)
    (Path(config["seed_model_paths"]["42"]) / "synthetic_weights.txt").write_text("MODIFIED TEST FIXTURE")
    results = tmp_path / "empty.jsonl"
    results.write_text("")
    with pytest.raises(ValueError, match="inputs changed"):
        collect(plan["plan_path"], results, tmp_path / "selection")


def test_strict_json_rejects_duplicate_and_nonfinite_input():
    for raw in ('{"seed": 42, "seed": 7}', '{"metrics": {"macro_f1": NaN}}'):
        with pytest.raises(ValueError): decode_json(raw)
