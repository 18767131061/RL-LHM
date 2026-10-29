import json
import math
from dataclasses import replace

import pytest
import torch

from data.manifest import Manifest
from evaluation.evaluate import evaluate
from rewards import RARSParameters, RewardConfig, compute_score
from rl import DACVQState, ShaeBaselineStore, VGSSampler, compute_shae_advantage, compute_grpo_advantage
from training.checkpoint import capture_state, restore_state

CLASSES = ("HCC", "ICC", "NC")


def completion(items):
    return json.dumps({"detections": items})


def lesion(label="HCC", box=(100, 100, 300, 300), phase="a"):
    return {"phase": phase, "label": label, "bbox_2d": list(box)}


def triplet(label="HCC", box=(100, 100, 300, 300)):
    return [lesion(label, box, phase) for phase in "apv"]


def reward_config(rars=True):
    return RewardConfig(CLASSES, {a: {b: 1.5 if "ICC" in (a, b) else 0.5 for b in CLASSES if a != b}
                                 for a in CLASSES}, RARSParameters(0.5, 1.0, 1.0, 1e-6),
                        0.0, 1.0, 1.0, -2.0, 2.0, rars)


def score(pred, truth, rars=True, counts=None):
    return compute_score(completion(pred) if isinstance(pred, list) else pred, completion(truth),
                         completion_nll=1.0, class_counts=counts or dict(HCC=30, ICC=2, NC=10),
                         config=reward_config(rars))["score"]


@pytest.mark.parametrize("rars", [False, True])
@pytest.mark.parametrize("label", CLASSES)
def test_reward_no_abstention_escape(rars, label):
    truth = triplet(label)
    wrong = triplet(next(c for c in CLASSES if c != label))
    assert score(truth, truth, rars) > score(wrong, truth, rars) > score([], truth, rars)
    assert score("not json", truth, rars) <= score([], truth, rars)


@pytest.mark.parametrize("rars", [False, True])
def test_multilesion_omission_and_extra_predictions(rars):
    truth = triplet() + triplet("ICC", (500, 500, 700, 700))
    wrong = triplet() + triplet("NC", (500, 500, 700, 700))
    assert score(wrong, truth, rars) > score(triplet(), truth, rars)
    previous = score(triplet(), triplet(), rars)
    for count in range(1, 20):
        current = score(triplet() + triplet("ICC", (800, 800, 900, 900)) * count, triplet(), rars)
        assert current < previous
        previous = current


def test_background_and_missing_phases():
    background = [lesion("no-lesion", (-1, -1, -1, -1), p) for p in "apv"]
    assert score(background, background) == 2
    assert score([], background) == -2
    assert score(triplet(), background) == -2
    assert score(triplet()[:1], triplet()) < score(triplet(), triplet())


def test_detection_counts_all_background_predictions():
    records = [{"uid": "positive", "patient_id": "p", "ground_truth": completion(triplet())},
               {"uid": "background", "patient_id": "p", "ground_truth": completion([])}]
    manifest = Manifest(CLASSES, {"p": "HCC", "no-input": "NC"}, records)
    result = evaluate(manifest, [{"uid": r["uid"], "completion": completion(triplet()), "nll": 1}
                                 for r in records], iou_thresholds=[0, 0.5, 0.75])
    for protocol, rows in result["detection"].items():
        for row in rows:
            assert (row["tp"], row["fp"], row["fn"]) == (3, 3, 0)
            assert row["precision"] == 0.5
        assert result["detection_positive_only"][protocol][0]["precision"] == 1
    assert result["coverage"]["patients_without_vote"] == 1
    assert result["background_false_positives"]["count"] == 3


def shae(scores, store):
    return compute_shae_advantage(torch.tensor(scores, dtype=torch.float64), torch.ones(len(scores), 1),
                                  ["rid"] * len(scores), sample_index=["uid"] * len(scores), store=store,
                                  gate_mid=0.05, gate_temperature=0.015, noise_std=0,
                                  epsilon=1e-6, generator=None)


@pytest.mark.parametrize("scores", [[1, 1], [1, 1 + 1e-12], [-2, 2], [1] * 8])
def test_shae_is_bounded(scores):
    store = ShaeBaselineStore(beta=0.9)
    store.update("uid", -1)
    advantage = shae(scores, store)
    assert torch.isfinite(advantage).all()
    assert advantage.abs().max() <= math.sqrt(len(scores))


def test_first_visit_and_grpo_scale():
    store = ShaeBaselineStore(beta=0.9)
    assert torch.equal(shae([1, 1], store), torch.zeros(2, 1, dtype=torch.float64))
    store = ShaeBaselineStore(beta=0.9)
    scores = torch.tensor([-1., 2.], dtype=torch.float64)
    expected = compute_grpo_advantage(scores, torch.ones(2, 1), ["rid", "rid"], epsilon=1e-6)
    torch.testing.assert_close(shae(scores.tolist(), store), expected)


def memory():
    return DACVQState(max_codes=3, input_dim=3, feature_decay=0.5, reward_decay=0.9,
                      difficulty_beta=1, dead_code_idle_updates=30, dead_code_replace_probability=0,
                      revival_noise_std=0.01, diversity_threshold=0.9, reward_min=-2, reward_max=2,
                      mass_epsilon=1e-8, generator=torch.Generator().manual_seed(7))


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("use_vgs", [False, True])
def test_sampling_and_resume(replacement, use_vgs):
    state = memory()
    vectors = [torch.eye(3)[i % 3] for i in range(30)]
    sampling = torch.Generator().manual_seed(123)
    sampler = VGSSampler(state, vectors, missing_feature_multiplier=0.5, weight_floor=1e-6,
                         generator=sampling, use_vgs=use_vgs, epoch_count=0, replacement=replacement)
    sampler.sample_epoch()
    for i in range(3):
        state.update(str(i), vectors[i], float(i - 1))
    baseline = ShaeBaselineStore(beta=0.9)
    baseline.update("a", 1)
    generators = {"sampling": sampling, "curriculum": state.generator}
    saved = capture_state(generators, baseline, sampler)
    expected = sampler.sample_epoch()
    state.update("0", vectors[0], -1)
    baseline.update("a", -1)
    restore_state(saved, generators, baseline, sampler)
    assert baseline.entries == {"a": 1.0}
    assert torch.equal(sampler.sample_epoch(), expected)
    if not replacement:
        assert len(set(expected.tolist())) == len(vectors)
    else:
        assert len(set(expected.tolist())) < len(vectors)


def test_unused_codes_decay_and_revival():
    state = memory()
    for i in range(3):
        state.update(str(i), torch.eye(3)[i], 2)
    state.dead_code_replace_probability = 1
    for _ in range(31):
        state.update("0", torch.eye(3)[0], 2)
    assert state.revival_count > 0
    assert state.visit_count[0] == 32
    assert state.reward_seen[0] == 32
    assert state.reward_ema[0] == 2
    torch.testing.assert_close(state.embedding.norm(dim=1), torch.ones(3, dtype=torch.float64))


def test_global_rng_streams_restore():
    import random
    import numpy as np
    random.seed(8)
    np.random.seed(9)
    torch.manual_seed(10)
    saved = capture_state({}, None, None)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_state(saved, {}, None, None)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_configurations_construct_all_components():
    from pathlib import Path
    from data.config import load_config
    from models.huggingface import DecodingConfig
    from training.engine import ComponentSwitches, PolicyParameters, ShaeParameters
    for path in Path("configs").glob("*.json"):
        config = load_config(path)
        if "generation" in config:
            DecodingConfig(**config["generation"])
        if "reward" in config:
            settings = dict(config["reward"])
            settings["rars_parameters"] = RARSParameters(**settings["rars_parameters"])
            RewardConfig(classes=tuple(settings["severity_matrix"]), rars_enabled=True, **settings)
            ComponentSwitches(**config["components"])
            PolicyParameters(**config["policy"])
            ShaeParameters(**{name: config["shae"][name] for name in ("gate_mid", "gate_temperature", "noise_std")})
            DACVQState(**config["curriculum"]["state"], generator=torch.Generator())


def test_patient_voting_and_training_counts_share_tie_order():
    from aggregation.patient import VoteRecord, aggregate_patient, group_prediction
    from data.manifest import training_class_counts
    from rewards.schema import parse_completion
    truth = completion([lesion("HCC"), lesion("ICC", (500, 500, 700, 700))])
    parsed = parse_completion(truth, CLASSES)
    assert group_prediction(parsed, CLASSES) == "HCC"
    manifest = Manifest(CLASSES, {"p": "HCC"}, [{"uid": "a", "patient_id": "p", "ground_truth": truth}])
    assert training_class_counts(manifest) == {"HCC": 1}
    assert aggregate_patient([VoteRecord("HCC", 2), VoteRecord("ICC", 1), VoteRecord(None, 0)], CLASSES).prediction == "ICC"
    assert aggregate_patient([VoteRecord("HCC", None), VoteRecord("ICC", None)], CLASSES).prediction == "HCC"
    assert aggregate_patient([VoteRecord(None, None)], CLASSES).prediction is None
