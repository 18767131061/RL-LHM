import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.manifest import Manifest, load_manifest
from evaluation.classification import classification_metrics
from evaluation.evaluate import evaluate
from evaluation.analyze import analyze, bootstrap_classification
from evaluation.detection import evaluate_similarity_matrix
from matching.geometry import box_iou_matrix
from matching.protocols import (CLASS_AGNOSTIC, GEOMETRY_FIRST_CLASS_AWARE, SAME_CLASS_PRIORITY,
                                THRESHOLD_CLASS_AGNOSTIC, THRESHOLD_CLASS_AWARE)
from rl import DACVQState, VGSSampler
from training.checkpoint import capture_state, restore_state
from tests.test_algorithms import completion, triplet, lesion, reward_config
from rewards import compute_score

OUTPUT = ("HCC", "ICC", "FNH", "HH", "NC")
METRIC = ("HCC", "ICC", "HH", "NC")


def loco():
    patients = {str(i): label for i, label in enumerate(METRIC)}
    records = [dict(uid=pid, patient_id=pid, prompt="Find lesions", images={p: "image.png" for p in "apv"},
                    ground_truth=completion(triplet(label))) for pid, label in patients.items()]
    return Manifest(OUTPUT, patients, records, METRIC, {pid: "A" if int(pid) < 2 else "B" for pid in patients})


def predictions(manifest):
    return [dict(uid=r["uid"], completion=r["ground_truth"], nll=1) for r in manifest.records]


def test_loco_fixed_macro_and_out_of_target_prediction(tmp_path):
    manifest = loco()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.payload()))
    manifest = load_manifest(path)
    rows = predictions(manifest)
    result = evaluate(manifest, rows, iou_thresholds=[0.5])
    assert result["classification"]["macro_f1"] == 1
    # FNH is a legal diagnosis, even when other records in this completion are HCC.
    rows[0]["completion"] = completion([lesion("FNH", phase="a"), lesion("FNH", phase="p"), lesion("HCC", phase="v")])
    result = evaluate(manifest, rows, iou_thresholds=[0.5])
    assert result["input_groups"][0]["schema_valid"]
    assert result["patients"][0]["prediction"] == "FNH"
    assert result["classification"]["accuracy"] == 0.75
    assert result["classification"]["macro_f1"] == 0.75
    assert result["classification"]["per_class"]["HCC"]["fn"] == 1
    rows[0]["completion"] = completion([])
    result = evaluate(manifest, rows, iou_thresholds=[0.5])
    assert result["patients"][0]["prediction"] is None
    assert result["classification"]["macro_f1"] == 0.75
    assert result["metric_classes"] == list(METRIC)


def test_bootstrap_retains_absent_classes_and_paired_patients():
    metric = classification_metrics(["HCC"] * 4, ["HCC"] * 4, OUTPUT, metric_classes=METRIC)
    assert metric["macro_f1"] == 0.25
    manifest = loco()
    result = evaluate(manifest, predictions(manifest), iou_thresholds=[0.5])
    analysis = analyze(manifest, result, samples=100, seed=1, comparison=result)
    assert analysis["bootstrap"]["metrics"]["macro_f1"]["percentile_95_ci"][0] < 1
    assert analysis["paired_bootstrap"]["metrics"]["macro_f1"]["percentile_95_ci"] == [0, 0]
    assert analysis["center_class_table"]["counts"]["A"]["HCC"] == 1
    assert analysis["per_center"]["A"]["macro_f1"] == 0.5
    assert analysis["constant_prediction_references"]["HCC"]["accuracy"] == 0.25
    with pytest.raises(ValueError, match="identical patient"):
        bootstrap_classification(result["patients"], OUTPUT, METRIC, samples=2, seed=1, comparison=result["patients"][:-1])


def codebook(size=64, idle=1024, probability=0.3):
    state = DACVQState(max_codes=size, input_dim=size, feature_decay=0.95, reward_decay=0.9,
                      difficulty_beta=3, dead_code_idle_updates=idle, dead_code_replace_probability=probability,
                      revival_noise_std=0.001, diversity_threshold=0.85, reward_min=-2, reward_max=2,
                      mass_epsilon=1e-8, generator=torch.Generator().manual_seed(42))
    for vector in torch.eye(size):
        state.online_init_step(vector)
    return state


def test_64_uniform_easy_codes_never_revive():
    state = codebook()
    vectors = torch.eye(64)
    for step in range(4096):
        state.update(str(step % 64), vectors[step % 64], 2)
    assert state.revival_count == 0
    assert torch.equal(state.visit_count, torch.full((64,), 64))
    assert torch.equal(state.reward_seen, state.visit_count)
    assert torch.equal(state.reward_ema, torch.full((64,), 2., dtype=torch.float64))
    assert (state.ema_cluster_size < 0.25).all()  # low quality mass is not inactivity


def test_long_tail_and_true_inactivity_are_separate():
    state = codebook(4, 64, 1)
    vectors = torch.eye(4)
    for step in range(512):
        code = 3 if step % 48 == 0 else 2 if step % 16 == 0 else 1 if step % 8 == 0 else 0
        state.update(str(code), vectors[code], 2)
    assert state.revival_count == 0
    for _ in range(65):
        state.update("0", vectors[0], 2)
    assert state.revival_count > 0
    assert state.reward_seen[0] > 400
    assert state.last_seen[0] == state.update_step


def test_activity_state_checkpoint_and_next_revival():
    state = codebook(4, 8, 1)
    vectors = list(torch.eye(4))
    sampler = VGSSampler(state, vectors, missing_feature_multiplier=0.5, weight_floor=1e-8,
                         generator=torch.Generator().manual_seed(4), use_vgs=True, epoch_count=1, replacement=True)
    generators = {"memory": state.generator, "sampling": sampler.generator}
    for _ in range(8):
        state.update("0", vectors[0], 2)
    saved = capture_state(generators, None, sampler)
    state.update("0", vectors[0], 2)
    expected = capture_state(generators, None, sampler)["curriculum"]
    draws = sampler.sample_epoch()
    restore_state(saved, generators, None, sampler)
    state.update("0", vectors[0], 2)
    actual = capture_state(generators, None, sampler)["curriculum"]
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            assert torch.equal(expected[key], actual[key])
        else:
            assert expected[key] == actual[key]
    assert torch.equal(draws, sampler.sample_epoch())


BOXES = [[322.7027245272002, 0., 521.3616755413314, 100.],
         [336.1784415232166, 0., 735.6285243113236, 100.],
         [305.2036538558252, 0., 689.5351796603188, 100.],
         [380.4820973395673, 0., 807.5558010001794, 100.],
         [198.712357964685, 0., 468.98278582453884, 100.]]


def test_threshold_matching_preserves_true_positives_after_extra_box():
    for n, legacy_tp in ((2, 2), (3, 1)):
        matrix = box_iou_matrix(BOXES[:2], BOXES[2:2+n])
        legacy = evaluate_similarity_matrix(matrix, ["HCC"]*2, ["HCC"]*n, CLASS_AGNOSTIC, 0.5)
        aware = evaluate_similarity_matrix(matrix, ["HCC"]*2, ["HCC"]*n, THRESHOLD_CLASS_AGNOSTIC, 0.5)
        assert legacy.tp == legacy_tp
        assert aware.tp == 2
        assert aware.fp == n - 2
    gt = completion([lesion(box=box) for box in BOXES[:2]])
    manifest = Manifest(("HCC",), {"p": "HCC"}, [dict(uid="u", patient_id="p", ground_truth=gt)])
    rows = [dict(uid="u", completion=completion([lesion(box=box) for box in BOXES[2:]]), nll=1)]
    old = evaluate(manifest, rows, iou_thresholds=[0.5])
    new = evaluate(manifest, rows, iou_thresholds=[0.5], matching_method="threshold_aware")
    assert new["localization"] == old["localization"]
    assert new["detection"][THRESHOLD_CLASS_AGNOSTIC][0]["tp"] == 2
    assert new["matching_sensitivity"]


def test_threshold_class_constraint_and_legacy_geometry_first():
    matrix = np.array([[0.9, 0.6], [0.6, 0.9]])
    gt, pred = ["A", "B"], ["B", "A"]
    assert evaluate_similarity_matrix(matrix, gt, pred, GEOMETRY_FIRST_CLASS_AWARE, 0.5).tp == 0
    assert evaluate_similarity_matrix(matrix, gt, pred, THRESHOLD_CLASS_AGNOSTIC, 0.5).tp == 2
    assert evaluate_similarity_matrix(matrix, gt, pred, THRESHOLD_CLASS_AWARE, 0.5).tp == 2
    assert evaluate_similarity_matrix(matrix, gt, pred, THRESHOLD_CLASS_AWARE, 0.7).tp == 0
    assert evaluate_similarity_matrix(np.zeros((1,1)), ["A"], ["A"], THRESHOLD_CLASS_AWARE, 0).tp == 0
    assert evaluate_similarity_matrix(np.empty((0,2)), [], pred, THRESHOLD_CLASS_AWARE, 0.5).fp == 2


def test_length_limited_valid_json_is_not_a_complete_prediction():
    manifest = loco()
    rows = predictions(manifest)
    for row in rows:
        row.update(finish_reason="eos", completion_tokens=100, max_new_tokens=512, target_tokens=100)
    rows[0].update(finish_reason="length", completion_tokens=512, target_tokens=600)
    result = evaluate(manifest, rows, iou_thresholds=[0.5], require_termination_metadata=True)
    assert result["input_groups"][0]["schema_valid"]
    assert not result["input_groups"][0]["completion_usable"]
    assert result["patients"][0]["prediction"] is None
    assert result["generation"]["truncation_rate_known"] == 0.25
    assert result["generation"]["target_over_budget_groups"] == 1
    assert compute_score(completion(triplet()), completion(triplet()), completion_nll=1,
                         class_counts={"HCC":1}, config=reward_config(), finish_reason="length")["score"] == -2
    with pytest.raises(ValueError, match="termination metadata"):
        evaluate(manifest, predictions(manifest), iou_thresholds=[0.5], require_termination_metadata=True)


def test_evaluation_and_analysis_cli(tmp_path):
    import subprocess
    import sys
    manifest = loco()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.payload()))
    prediction_path = tmp_path / "predictions.jsonl"
    rows = predictions(manifest)
    for row in rows:
        row.update(finish_reason="eos", completion_tokens=100, max_new_tokens=512)
    prediction_path.write_text("\n".join(json.dumps(r) for r in rows))
    config = dict(manifest=str(manifest_path), predictions=str(prediction_path), iou_thresholds=[0.5],
                  output_dir=str(tmp_path / "metrics"), matching_method="threshold_aware", require_termination_metadata=True)
    config_path = tmp_path / "eval.json"
    config_path.write_text(json.dumps(config))
    subprocess.run([sys.executable, "-m", "evaluation.evaluate", "--config", str(config_path)], check=True, capture_output=True)
    metrics_path = tmp_path / "metrics" / "metrics.json"
    result = json.loads(metrics_path.read_text())
    assert result["classification"]["macro_f1"] == 1
    assert result["matching_method"] == "threshold_aware"
    analysis_path = tmp_path / "analysis.json"
    subprocess.run([sys.executable, "-m", "evaluation.analyze", "--manifest", str(manifest_path),
                    "--metrics", str(metrics_path), "--compare", str(metrics_path),
                    "--split-manifest", f"test={manifest_path}", "--bootstrap-samples", "10", "--output", str(analysis_path)],
                   check=True, capture_output=True)
    assert json.loads(analysis_path.read_text())["paired_bootstrap"]["metrics"]["macro_f1"]["estimate"] == 0


def test_inference_phase_intervention_and_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from inference import run as inference_run
    manifest = loco()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.payload()))
    calls = []
    class Backend:
        processor = object()
        generation = SimpleNamespace(max_new_tokens=512)
        def generate(self, records):
            calls.extend(records)
            assert all(len(set(r["images"].values())) == 1 for r in records)
            return SimpleNamespace(texts=[r["ground_truth"] for r in records], completion_nlls=[1.] * len(records),
                                   finish_reasons=["eos"] * len(records), completion_token_counts=[100] * len(records))
    monkeypatch.setattr(inference_run.HuggingFaceBackend, "from_pretrained", lambda **kwargs: Backend())
    monkeypatch.setattr(inference_run, "target_lengths", lambda processor, records, classes, generation: [
        dict(uid=r["uid"], patient_id=r["patient_id"], patient_label=r["patient_label"], target_tokens=100,
             lesion_records=3, max_lesions_per_phase=1, exceeds_generation_budget=False) for r in records])
    # Give each phase a distinct path, so equality verifies the actual intervention.
    payload = manifest.payload()
    for row in payload["records"]:
        row["images"] = {p: f"{p}.png" for p in "apv"}
    path.write_text(json.dumps(payload))
    from data.config import load_config
    config = load_config("configs/inference_repeat_a.json")
    config.update(manifest=str(path), output=str(tmp_path / "predictions.jsonl"))
    assert inference_run.run(config) == 4
    saved = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert all(row["phase_sources"] == {p:"a" for p in "apv"} for row in saved)
    assert all(row["finish_reason"] == "eos" for row in saved)
    metadata = json.loads((tmp_path / "predictions.jsonl.metadata.json").read_text())
    assert metadata["metric_classes"] == list(METRIC)
    assert len(metadata["predictions_sha256"]) == 64
    assert json.loads(path.read_text()) == payload
