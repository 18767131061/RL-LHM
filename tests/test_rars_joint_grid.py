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


@pytest.mark.parametrize('fault',['patient_mapping','image','ground_truth','dimension'])
def test_seed_payload_conflict(tmp_path,fault):
    cfg=configuration(tmp_path)
    rows=[{'extra_info':{'sample_uid':'train-a','patient_id':'patient-a','visual_z':[1.,0.]},'images':['a.png'],'reward_model':{'ground_truth':'HCC'}},
          {'extra_info':{'sample_uid':'train-b','patient_id':'patient-b','visual_z':[0.,1.]},'images':['b.png'],'reward_model':{'ground_truth':'ICC'}}]
    for cache in cfg['seed_train_files'].values():Path(cache).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    changed=copy.deepcopy(rows)
    if fault=='patient_mapping':
        changed[0]['extra_info']['patient_id'],changed[1]['extra_info']['patient_id']='patient-b','patient-a'
    elif fault=='image':changed[0]['images']=['different.png']
    elif fault=='ground_truth':changed[0]['reward_model']['ground_truth']='ICC'
    elif fault=='dimension':
        for row in changed:row['extra_info']['visual_z'].append(0.)
    Path(cfg['seed_train_files']['2026']).write_text(''.join(json.dumps(row)+'\n' for row in changed))
    with pytest.raises(ValueError):Search(cfg,tmp_path/'s.db')


def test_seed_features_may_change_without_changing_training_payload(tmp_path):
    cfg = configuration(tmp_path)
    cache = Path(cfg["seed_train_files"]["2026"])
    row = json.loads(cache.read_text())
    row["extra_info"]["visual_z"] = [0., 1.]
    row["extra_info"]["visual_z_cache_id"] = "different-seed-feature-cache"
    cache.write_text(json.dumps(row) + "\n")
    search = Search(cfg, tmp_path / "search.db")
    assert search.identity["feature_dimension"] == 2
    assert len(search.identity["training_payload_sha256"]) == 64


def test_source_validation_cannot_copy_target_macro_subset(tmp_path):
    cfg = configuration(tmp_path)
    manifest = json.loads(Path(cfg['validation_manifest']).read_text())
    manifest['macro_classes'] = ['HCC', 'ICC']
    write_json(cfg['validation_manifest'], manifest)
    with pytest.raises(ValueError, match='Source validation macro'):
        Search(cfg, tmp_path / 's.db')


def prepared_loco_configuration(tmp_path):
    import subprocess, sys
    from test_training_data_cli import TrainingDataCliTests
    source = tmp_path / 'source.jsonl'
    for phase in ('a', 'p', 'v'):
        (tmp_path / f'{phase}.png').write_bytes(b'synthetic-image-not-loaded')
    rows = []
    for label in ('HCC', 'ICC'):
        for i in range(4):
            row = TrainingDataCliTests.source_row(f'{label}{i}', 'HCC')
            row['metadata'].update(analysis_center='A', **{'class': label})
            rows.append(row)
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write_json(tmp_path / 'test.json', {'split': 'test', 'patients': [{'patient_id': 'test-p', 'center_id': 'B'}]})
    prepared = tmp_path / 'prepared'
    process = subprocess.run([sys.executable, '-S', str(Path(__file__).resolve().parents[1] / 'preprocess/build_verl_real961_dataset.py'),
        '--loco-source-jsonl', str(source), '--data-root', str(tmp_path), '--output-dir', str(prepared),
        '--source-center', 'A', '--held-out-center', 'B', '--tuning-fraction', '.25', '--split-seed', '97',
        '--active-classes', 'HCC', 'ICC', '--split-id', 'A_to_B', '--test-identity-manifest', str(tmp_path / 'test.json'),
        '--refit-train-validation'], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    files, models = {}, {}
    for seed in SEEDS:
        path = tmp_path / f'train-{seed}.jsonl'
        cached = [json.loads(line) for line in (prepared / 'real961_rl_train.jsonl').read_text().splitlines()]
        for row in cached:
            row['extra_info']['visual_z'] = row['reward_model']['extra_info']['visual_z'] = [1., 0.]
        path.write_text(''.join(json.dumps(row) + '\n' for row in cached))
        files[str(seed)] = str(path)
        models[str(seed)] = str(tmp_path / f'sft-{seed}')
        Path(models[str(seed)]).mkdir()
    cfg = {'cohort': 'private', 'protocol': 'loco', 'split_id': 'A_to_B', 'selection_split': 'validation',
        'formal_test_used_for_selection': False, 'training_center': 'A', 'held_out_center': 'B',
        'classes': ['HCC', 'ICC'], 'loco_pairs': [['HCC', 'ICC']], 'values': [.1, .2], 'seeds': SEEDS,
        'synthetic_test': True, 'train_file': files['42'], 'seed_train_files': files, 'seed_model_paths': models,
        'validation_manifest': str(prepared / 'validation_manifest.json'), 'test_identity_manifest': str(tmp_path / 'test.json'),
        'preparation_audit': str(prepared / 'audit.json'), 'output_root': str(tmp_path / 'runs')}
    write_json(tmp_path / 'config.json', cfg)
    return load_config(tmp_path / 'config.json'), prepared


def complete_prepared_loco(search):
    manifest = json.loads(Path(search.config['validation_manifest']).read_text())
    truths = {row['patient_id']: row['truth'] for row in manifest['patients']}
    for task_id in range(search.n_tasks):
        pred, receipt, _ = simulated_training_and_prediction(search, task_id)
        rows = [{'group_id': group['group_id'], 'raw': json.dumps({'detections': [
            {'phase': phase, 'label': truths[group['patient_id']], 'bbox_2d': [0, 0, 1, 1]} for phase in ('a', 'p', 'v')]}),
            'completion_nll': .2} for group in manifest['input_groups']]
        pred.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        data = json.loads(receipt.read_text());data['predictions_sha256'] = sha(pred);write_json(receipt, data)
        search.score(task_id, pred, receipt)


def refit_inputs(tmp_path, prepared):
    refit = {'fit_scope': 'train_plus_validation', 'seed_train_files': {}, 'seed_model_paths': {}}
    for seed in SEEDS:
        rows = [json.loads(line) for line in (prepared / 'real961_rl_refit_train_validation.jsonl').read_text().splitlines()]
        for row in rows:
            row['extra_info']['visual_z'] = row['reward_model']['extra_info']['visual_z'] = [0., 1.]
        path = tmp_path / f'refit-{seed}.jsonl';path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        model = tmp_path / f'refit-sft-{seed}';model.mkdir()
        refit['seed_train_files'][str(seed)] = str(path)
        refit['seed_model_paths'][str(seed)] = str(model)
    path = tmp_path / 'refit.json';write_json(path, refit)
    return path, refit


def test_prepared_loco_split_counts_search_and_final_scope(tmp_path):
    cfg, prepared = prepared_loco_configuration(tmp_path)
    search = Search(cfg, tmp_path / 's.db')
    assert search.identity['train_patients'] == 6 and search.identity['validation_patients'] == 2
    assert search.identity['preparation_audit_sha256'] == sha(prepared / 'audit.json')
    with pytest.raises(ValueError, match='Complete every'):
        search.finalize(tmp_path / 'early.json')
    complete_prepared_loco(search)
    selected = search.finalize(tmp_path / 'selected_models.json')
    assert selected['status'] == 'synthetic_selected_references'
    assert selected['fit_scope'] == 'selection_train_only' and len(selected['models']) == 5
    assert all(Path(row['checkpoint']).is_dir() for row in selected['models'])
    refit_path, _ = refit_inputs(tmp_path, prepared)
    result = search.finalize(tmp_path / 'final_refit.json', refit_path)
    assert result['status'] == 'synthetic_refit_commands_only' and result['models'] == []
    assert result['class_counts'] == {'HCC': 8} and len(result['refit_tasks']) == 5
    for task in result['refit_tasks']:
        assert task['training_executed'] is False
        command = task['command']
        assert Path(command[command.index('--train-file') + 1]).name == f"refit-{task['seed']}.jsonl"
        assert Path(command[command.index('--model-path') + 1]).name == f"refit-sft-{task['seed']}"
        assert not Path(command[command.index('--output-dir') + 1]).exists()


@pytest.mark.parametrize('fault', ['counts', 'audit', 'validation'])
def test_preparation_binding_rejects_changes(tmp_path, fault):
    cfg, prepared = prepared_loco_configuration(tmp_path)
    if fault == 'counts':
        for path in cfg['seed_train_files'].values():
            rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
            for row in rows:
                row['extra_info']['class_counts'] = row['reward_model']['extra_info']['class_counts'] = {'HCC': 999}
            Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif fault == 'audit':
        audit = json.loads((prepared / 'audit.json').read_text());audit['loco']['source_center'] = 'B';write_json(prepared / 'audit.json', audit)
    else:
        manifest = json.loads(Path(cfg['validation_manifest']).read_text());manifest['patients'][0]['patient_id'] = 'changed'
        for group in manifest['input_groups']:
            if group['patient_id'] != manifest['patients'][1]['patient_id']:group['patient_id'] = 'changed'
        write_json(cfg['validation_manifest'], manifest)
    with pytest.raises(ValueError): Search(cfg, tmp_path / 's.db')


def test_refit_cache_cannot_add_test_patient_or_change_model_identity(tmp_path):
    cfg, prepared = prepared_loco_configuration(tmp_path)
    search = Search(cfg, tmp_path / 's.db');complete_prepared_loco(search)
    refit_path, refit = refit_inputs(tmp_path, prepared)
    path = Path(refit['seed_train_files']['42'])
    original = path.read_text();rows = [json.loads(line) for line in original.splitlines()]
    rows[0]['extra_info']['patient_id'] = 'test-p'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):search.finalize(tmp_path / 'bad.json', refit_path)
    path.write_text(original)
    search.finalize(tmp_path / 'good.json', refit_path)
    changed = tmp_path / 'another_sft';changed.mkdir();refit['seed_model_paths']['42'] = str(changed);write_json(refit_path, refit)
    with pytest.raises(ValueError, match='different model/cache identity'):
        search.finalize(tmp_path / 'changed.json', refit_path)


def test_formal_prepared_cache_requires_model_binding(tmp_path):
    from training.rars_joint_grid import validate_seed_caches, payloads
    cfg, _ = prepared_loco_configuration(tmp_path)
    base, people = payloads(cfg['train_file'])
    with pytest.raises(ValueError, match='model-bound'):
        validate_seed_caches(cfg['seed_train_files'], cfg['seed_model_paths'], base, people, require_contract=True)
