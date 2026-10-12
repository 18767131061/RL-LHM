"""Synthetic tasks substitute training/inference; evaluation and search are real."""
import copy
import json
from pathlib import Path
import pytest
from training.rars_joint_grid import Search, SEEDS, VALUES, PAIRS, candidate, load_config, digest, sha, write_json


def configuration(tmp_path, values=(0.1, 0.2)):
    rows = [{"extra_info": {"sample_uid": "train-a", "patient_id": "train-p", "visual_z": [1., 0.]}}]
    files = {}
    models = {}
    for seed in SEEDS:
        path = tmp_path / f"train-{seed}.jsonl"
        path.write_text(json.dumps(rows[0]) + "\n")
        files[str(seed)] = str(path)
        models[str(seed)] = str(tmp_path / f"sft-{seed}")
        Path(models[str(seed)]).mkdir()
    manifest = {"schema_version": 1, "classes": ["HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"],
                "split": "validation", "split_id": "P1", "patients": [{"patient_id": "val-p", "truth": "HCC"}],
                "input_groups": [{"group_id": "val-a", "patient_id": "val-p", "phases": {
                    ph: {"image_ref": f"synthetic/{ph}.png", "ground_truth": [{"label": "HCC", "bbox_2d": [0,0,10,10]}]}
                    for ph in ["a", "p", "v"]}}]}
    write_json(tmp_path / "val.json", manifest)
    write_json(tmp_path / "test.json", {"split": "test", "patients": [{"patient_id": "test-p"}]})
    cfg = {"cohort": "private", "protocol": "main", "split_id": "P1", "selection_split": "validation",
           "formal_test_used_for_selection": False, "values": list(values), "seeds": SEEDS,
           "synthetic_test": True, "train_file": files["42"], "seed_train_files": files,
           "validation_manifest": str(tmp_path / "val.json"), "test_identity_manifest": str(tmp_path / "test.json"),
           "seed_model_paths": models, "output_root": str(tmp_path / "runs")}
    write_json(tmp_path / "config.json", cfg)
    return load_config(tmp_path / "config.json")


def simulated_training_and_prediction(search, task_id, correct=True):
    task, command = search.prepare(task_id)
    folder = Path(task["task_dir"])
    train = folder / "training"
    (train / "global_step_6" / "actor").mkdir(parents=True)
    run = {"seed": task["seed"], "variant": "full_rl_lhm", "configuration_only": False, "smoke": False,
           "train_file_sha256": search.identity["seed_train_sha256"][str(task["seed"])] , "model_path": task["model_path"],
           "risk_profile_sha256": sha(folder / "risk_profile.json")}
    write_json(train / "run_manifest.json", run)
    write_json(train / "training_complete.json", {"seed": task["seed"], "variant": "full_rl_lhm",
               "status": "training_complete", "smoke": False, "run_manifest_sha256": sha(train / "run_manifest.json"),
               "final_checkpoint": "global_step_6"})
    predictions = folder / "predictions.jsonl"
    label = "HCC" if correct else "ICC"
    predictions.write_text(json.dumps({"group_id": "val-a", "raw": json.dumps({"detections": [
        {"phase": ph, "label": label, "bbox_2d": [0,0,10,10]} for ph in ["a","p","v"]]}), "completion_nll": .2}) + "\n")
    receipt = folder / "receipt.json"
    write_json(receipt, {"task_digest": digest(task), "validation_sha256": search.identity["validation_sha256"],
                         "predictions_sha256": sha(predictions), "partition": "validation",
                         "checkpoint": str(train / "global_step_6")})
    return predictions, receipt, run


def test_full_dimensions_arithmetic_and_streaming():
    for cohort, total in [("private", 50625), ("public", 11390625)]:
        config = {"cohort": cohort, "values": VALUES}
        assert len(VALUES) ** len(PAIRS[cohort]) == total
        assert set(candidate(config, 0)["pair_weights"].values()) == {.1}
        assert set(candidate(config, total-1)["pair_weights"].values()) == {1.5}
        with pytest.raises(ValueError): candidate(config, total)
    assert (50625+11390625)*5 == 57206250


def test_sharded_resume_five_seed_selection_and_partial_not_global(tmp_path):
    cfg = configuration(tmp_path)
    search = Search(cfg, tmp_path / "search.db")
    shards = [[t["task_id"] for t in search.pending(shard, 3)] for shard in range(3)]
    assert sorted(sum(shards, [])) == list(range(80))
    assert search.summary()["global_best_in_grid"] is None
    for task_id in range(80):
        predictions, receipt, _ = simulated_training_and_prediction(search, task_id, correct=(task_id // 5 == 9))
        search.score(task_id, predictions, receipt)
        if task_id == 5:
            assert search.summary()["status"] == "partial"
            assert search.summary()["global_best_in_grid"] is None
            # Exact replay is idempotent and never doubles the seed count.
            search.score(task_id, predictions, receipt)
    resumed = Search(cfg, tmp_path / "search.db")
    assert list(resumed.pending()) == []
    result = resumed.summary()
    assert result["status"] == "synthetic_complete"
    assert result["exhaustive"] is False
    assert result["completed_tasks"] == 80
    assert result["global_best_in_grid"]["candidate_index"] == 9
    assert result["global_best_in_grid"]["mean_validation_macro_f1"] == pytest.approx(.2)


@pytest.mark.parametrize("fault", ["seed", "unfinished", "dryrun", "profile", "train", "test_score", "checkpoint", "duplicate_prediction"])
def test_reject_mislabeled_or_unfinished_results(tmp_path, fault):
    search = Search(configuration(tmp_path), tmp_path / "search.db")
    predictions, receipt, run = simulated_training_and_prediction(search, 0)
    folder = Path(search.task(0)["task_dir"])
    run_path = folder / "training/run_manifest.json"
    done_path = folder / "training/training_complete.json"
    done = json.loads(done_path.read_text())
    evidence = json.loads(receipt.read_text())
    if fault == "seed": run["seed"] = 2026
    elif fault == "unfinished": done["status"] = "running"
    elif fault == "dryrun": run["configuration_only"] = True
    elif fault == "profile": run["risk_profile_sha256"] = "wrong"
    elif fault == "train": run["train_file_sha256"] = "wrong"
    elif fault == "test_score": evidence["partition"] = "test"
    elif fault == "checkpoint": evidence["checkpoint"] = str(tmp_path)
    elif fault == "duplicate_prediction":
        predictions.write_text(predictions.read_text()*2)
        evidence["predictions_sha256"] = sha(predictions)
    write_json(run_path, run)
    done["run_manifest_sha256"] = sha(run_path)
    write_json(done_path, done)
    write_json(receipt, evidence)
    with pytest.raises(ValueError): search.score(0, predictions, receipt)
    assert search.summary()["completed_tasks"] == 0


def test_leakage_and_nonfinite_scores_fail_closed(tmp_path):
    cfg = configuration(tmp_path)
    search = Search(cfg, tmp_path / "search.db")
    pred, receipt, _ = simulated_training_and_prediction(search, 0)
    search.score(0, pred, receipt)
    search.db.execute("UPDATE completed SET score=? WHERE task=0", (float("inf"),))
    search.db.commit()
    with pytest.raises(ValueError, match="finite score"): search.summary()
    test = json.loads(Path(cfg["test_identity_manifest"]).read_text())
    test["patients"][0]["patient_id"] = "val-p"
    write_json(cfg["test_identity_manifest"], test)
    with pytest.raises(ValueError, match="overlap"): Search(cfg, tmp_path / "different.db")


def test_reward_callback_uses_candidate_symmetrically(tmp_path, monkeypatch):
    from rewards.verl_liver_reward import get_risk_severity
    profile = candidate({"cohort": "private", "values": VALUES}, 0)
    path = tmp_path / "profile.json"
    write_json(path, profile)
    monkeypatch.setenv("RL_LHM_RISK_PROFILE", str(path))
    assert get_risk_severity("HCC", "ICC", "private", "ordinal") == .1
    assert get_risk_severity("ICC", "HCC", "private", "ordinal") == .1
    assert get_risk_severity("FNH", "ICC", "private", "ordinal") == .4
    with pytest.raises(RuntimeError, match="cohort"): get_risk_severity("HCC", "ICC", "5cls", "ordinal")


def test_formal_subset_rejected_and_loco_axes_must_be_explicit(tmp_path):
    cfg = configuration(tmp_path)
    path = tmp_path / "config.json"
    payload = json.loads(path.read_text())
    payload["synthetic_test"] = False
    write_json(path, payload)
    with pytest.raises(ValueError, match="all 15"): load_config(path)
    payload["synthetic_test"] = True
    payload.update(protocol="loco", split_id="A_to_B", training_center="A", held_out_center="B",
                   classes=["HCC","ICC","HA_Hemangioma"], loco_pairs=[["HCC","ICC"],["ICC","HA_Hemangioma"]])
    write_json(path, payload)
    loco = load_config(path)
    assert len(candidate(loco, 0)["pair_weights"]) == 2
    assert len(loco["values"]) ** len(loco["loco_pairs"]) == 4
    payload["loco_pairs"] = [["HCC","FNH"]]
    write_json(path,payload)
    with pytest.raises(ValueError, match="active classes"):load_config(path)
