"""UID bridge tests with CPU Torch and minimal VeRL type substitutes.

Set RL_LHM_VERL_RAY_TRAINER to the unmodified ray_trainer.py from the documented
reference commit for the source-level carrier integration test. It executes the
actual UID assignment and patched compute_advantage with local source files.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import types
import uuid
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rl.core import ShaeBaselineStore, compute_shae_advantage as core_shae
from rl.patch_verl_sample_uid import patch_source, require_sample_uid_patch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def extension(monkeypatch):
    # Only VeRL's import-time DataProto / AbstractCurriculumSampler types are
    # unavailable. All advantage, state, quantization and sampling math is real.
    for name in ("verl", "verl.experimental", "verl.experimental.dataset",
                 "verl.experimental.dataset.sampler"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["verl"].DataProto = object
    sys.modules["verl.experimental.dataset.sampler"].AbstractCurriculumSampler = object
    spec = importlib.util.spec_from_file_location("verl_lhm_extensions", ROOT / "rl/verl_lhm_extensions.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "verl_lhm_extensions", module)
    spec.loader.exec_module(module)
    return module


def shae_config():
    return OmegaConf.create({"shae_gate_mid": 100.0, "shae_gate_temp": 1.0,
                            "shae_baseline_beta": 0.5, "shae_noise_std": 0.0,
                            "shae_save_every": 0})


def sampler(extension):
    rows = [{"uid": "unused-row-" + key,
             "extra_info": {"sample_uid": key, "visual_z": vector}}
            for key, vector in (("a", [1.0, 0.0]), ("b", [0.0, 1.0]))]
    return extension.VLGSHybridSampler(rows, OmegaConf.create({
        "seed": 4, "sampler": {"vq_code_capacity": 2, "vq_reward_decay": 0.5,
                               "vq_dead_code_replace_prob": 0.0, "save_every": 0}}))


def batch(rewards, visits, samples):
    scores = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
    return types.SimpleNamespace(
        batch={"token_level_rewards": scores, "token_level_scores": scores,
               "response_mask": torch.ones_like(scores)},
        non_tensor_batch={"uid": np.asarray(visits, dtype=object),
                          "extra_info": np.asarray([{"sample_uid": key} for key in samples], dtype=object)})


def calculate(extension, data):
    return extension.compute_shae_advantage(
        data.batch["token_level_rewards"], data.batch["response_mask"],
        data.non_tensor_batch["uid"], shae_config(),
        sample_index=extension.persistent_sample_keys(data.non_tensor_batch, len(data.batch["token_level_scores"])))[0]


def test_repeated_visits_keep_g_and_reuse_cross_batch_memory(extension):
    data = batch([1, 3, 5, 7], ["v1", "v1", "v2", "v2"], ["a"] * 4)
    advantage = calculate(extension, data)
    # Each visit has std=1, not the std across all four responses. The second
    # visit reads the first visit's persistent baseline (2), then updates to 4.
    torch.testing.assert_close(advantage[:, 0], torch.tensor([-1., 1., 3., 5.]), atol=1e-5, rtol=0)
    store = extension._get_shae_store(shae_config())
    assert store.entries == {"a": {"ema": 4.0, "seen": 2}}
    later = batch([7, 9], ["new-random-uid"] * 2, ["a"] * 2)
    torch.testing.assert_close(calculate(extension, later)[:, 0], torch.tensor([3., 5.]), atol=1e-5, rtol=0)
    assert store.entries == {"a": {"ema": 6.0, "seen": 3}}
    assert data.non_tensor_batch["uid"].tolist() == ["v1", "v1", "v2", "v2"]


def test_vgs_uses_persistent_key_but_updates_once_per_visit(extension):
    curriculum = sampler(extension)
    data = batch([1, 3, 5, 7, -5, -3], ["v1", "v1", "v2", "v2", "v3", "v3"],
                 ["a"] * 4 + ["b"] * 2)
    curriculum.update(data)
    assert curriculum.vq.sample_to_code == {"a": 0, "b": 1}
    assert curriculum.vq.reward_seen.tolist() == [2., 1.]
    assert curriculum.vq.reward_ema.tolist() == [4., -4.]
    weights = curriculum._build_weights()
    assert weights[1] > weights[0]
    curriculum.update(batch([7, 9], ["v4"] * 2, ["a"] * 2))
    assert curriculum.vq.reward_seen.tolist() == [3., 1.]
    assert curriculum.vq.reward_ema.tolist() == [6., -4.]


def test_dataset_identity_must_be_explicit_and_unique(extension):
    config = OmegaConf.create({"sampler": {}})
    with pytest.raises(ValueError, match="sample_uid"):
        extension.VLGSHybridSampler([{"uid": "ambiguous"}], config)
    duplicate = {"extra_info": {"sample_uid": "same", "visual_z": [1., 0.]}}
    with pytest.raises(ValueError, match="Duplicate"):
        extension.VLGSHybridSampler([duplicate, duplicate], config)


def test_noncontiguous_balanced_groups_and_core_agree(extension):
    data = batch([1, 5, 3, 7], ["v1", "v2", "v1", "v2"], ["a"] * 4)
    expected = core_shae(data.batch["token_level_rewards"], data.batch["response_mask"],
                         data.non_tensor_batch["uid"], sample_index=["a"] * 4,
                         store=ShaeBaselineStore(beta=0.5), gate_mid=100., gate_temperature=1., noise_std=0.)
    torch.testing.assert_close(calculate(extension, data), expected)
    torch.testing.assert_close(expected[:, 0], torch.tensor([-1., 3., 1., 5.]), atol=1e-5, rtol=0)


@pytest.mark.parametrize("failure", ["missing_metadata", "mixed_group", "wrong_length", "unknown_sample"])
def test_identity_errors_fail_before_curriculum_state_changes(extension, failure):
    curriculum = sampler(extension)
    data = batch([1, 3], ["v1", "v1"], ["a", "a"])
    if failure == "missing_metadata":
        data.non_tensor_batch.pop("extra_info")
    elif failure == "mixed_group":
        data.non_tensor_batch["extra_info"][1] = {"sample_uid": "b"}
    elif failure == "wrong_length":
        data.non_tensor_batch["uid"] = np.asarray(["v1"])
    else:
        data.non_tensor_batch["extra_info"] = [{"sample_uid": "missing"}] * 2
    with pytest.raises(ValueError):
        curriculum.update(data)
    assert curriculum.vq.sample_to_code == {}
    assert curriculum.update_count == 0
    if failure != "unknown_sample":
        with pytest.raises(ValueError):
            calculate(extension, data)
        assert all(not store.entries for store in extension._SHAE_STORES.values())


@pytest.fixture
def reference_source():
    path = os.environ.get("RL_LHM_VERL_RAY_TRAINER")
    if not path:
        pytest.skip("Set RL_LHM_VERL_RAY_TRAINER for the pinned-source integration test")
    return Path(path).read_text(encoding="utf-8")


def test_reference_patch_is_exact_idempotent_and_fail_closed(reference_source, tmp_path):
    updated, was_applied = patch_source(reference_source)
    assert not was_applied
    assert patch_source(updated) == (updated, True)
    from rl.patch_verl_sample_uid import ADDITION
    assert updated.replace(ADDITION, "", 1) == reference_source
    target = tmp_path / "ray_trainer.py"
    target.write_text(reference_source)
    with pytest.raises(RuntimeError, match="missing"):
        require_sample_uid_patch(target)
    target.write_text(updated)
    assert require_sample_uid_patch(target)["patch_version"] == 1
    with pytest.raises(RuntimeError, match="supported reference"):
        patch_source(reference_source.replace("gamma: float = 1.0", "gamma: float = 0.9", 1))


def test_pinned_carrier_uid_repeat_advantage_sampler_integration(extension, reference_source):
    updated, _ = patch_source(reference_source)
    tree = ast.parse(updated)
    # Execute the unchanged upstream UID assignment from fit, plus the complete
    # actual compute_response_mask and patched compute_advantage functions.
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    fit = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    assignment = next(n for n in ast.walk(fit) if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "batch.non_tensor_batch['uid']")
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"compute_advantage", "compute_response_mask"}]
    class Estimator(str, Enum):
        GAE = "gae"
        GRPO = "grpo"
    namespace = {"np": np, "uuid": uuid, "AdvantageEstimator": Estimator}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, "pinned_ray_trainer.py", "exec"), namespace)
    class TensorBatch(dict):
        def __len__(self):
            return next(iter(self.values())).shape[0]
    def visit_data(samples, rewards):
        data = types.SimpleNamespace(batch=TensorBatch(input_ids=torch.zeros(len(samples), 2)),
                                     non_tensor_batch={"extra_info": np.array(
                                         [{"sample_uid": key} for key in samples], dtype=object)})
        namespace["batch"] = data
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "pinned_uid.py", "exec"), namespace)
        before = data.non_tensor_batch["uid"].copy()
        # The DataProto container is unavailable locally; mirror its interleaved
        # repeat with numpy/Torch. No rollout engine or trainer is simulated.
        data.non_tensor_batch = {k: np.repeat(v, 2) for k, v in data.non_tensor_batch.items()}
        scores = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        data.batch = TensorBatch(token_level_rewards=scores, token_level_scores=scores,
                                 responses=torch.zeros_like(scores), attention_mask=torch.ones(len(scores), 3))
        return data, before
    curriculum = sampler(extension)
    first, original_visits = visit_data(["a", "a", "b"], [1, 3, 5, 7, -5, -3])
    assert len(set(original_visits)) == 3  # two visits to a still have distinct group IDs
    out = namespace["compute_advantage"](first, "shae", config=shae_config(), num_repeat=2)
    curriculum.update(out)
    torch.testing.assert_close(out.batch["advantages"][:, 0], torch.tensor([-1., 1., 3., 5., -1., 1.]), atol=1e-5, rtol=0)
    assert out.non_tensor_batch["uid"].tolist() == np.repeat(original_visits, 2).tolist()
    assert curriculum.vq.reward_seen.tolist() == [2., 1.]
    second, later_visits = visit_data(["a"], [7, 9])
    assert not set(original_visits) & set(later_visits)
    namespace["compute_advantage"](second, "shae", config=shae_config(), num_repeat=2)
    curriculum.update(second)
    assert extension._get_shae_store(shae_config()).entries["a"] == {"ema": 6., "seen": 3}
    assert curriculum.vq.reward_ema.tolist() == [6., -4.]
    assert curriculum._build_weights()[1] > curriculum._build_weights()[0]


def test_launcher_checks_bridge_before_starting_shae(extension, reference_source, tmp_path, monkeypatch):
    from training import verl_lhm_launcher as launcher
    target = tmp_path / "ray_trainer.py"
    target.write_text(reference_source)
    audit = tmp_path / "runtime_audit.json"
    monkeypatch.setenv("RL_LHM_UID_AUDIT_PATH", str(audit))
    monkeypatch.setenv("PYTHONPATH", "existing-path")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["launcher", "algorithm.adv_estimator=shae"])
    monkeypatch.setattr(launcher, "_check_policy_contract", lambda: None)  # independently covered by policy tests
    original_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: (
        types.SimpleNamespace(origin=str(target)) if name == "verl.trainer.ppo.ray_trainer" else original_find(name)))
    calls = []
    monkeypatch.setattr(launcher.runpy, "run_module", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(RuntimeError, match="missing"):
        launcher.main()
    assert calls == []
    target.write_text(patch_source(reference_source)[0])
    launcher.main()
    assert len(calls) == 1
    assert str(ROOT / "rl") in os.environ["PYTHONPATH"].split(os.pathsep)
    assert "existing-path" in os.environ["PYTHONPATH"].split(os.pathsep)
    assert json.loads(audit.read_text())["patch_version"] == 1


def test_curriculum_resume_rejects_same_size_changed_dataset(extension):
    original = sampler(extension)
    saved = original.state_dict()
    changed = sampler(extension)
    changed.dataset_signature = "different"
    with pytest.raises(ValueError, match="identities"):
        changed.load_state_dict(saved)


def test_nonfinite_reward_and_feature_fail_before_state_update(extension):
    curriculum = sampler(extension)
    data = batch([1, float("nan")], ["v1", "v1"], ["a", "a"])
    with pytest.raises(ValueError, match="finite"):
        curriculum.update(data)
    with pytest.raises(ValueError, match="finite"):
        calculate(extension, data)
    assert curriculum.vq.sample_to_code == {}
    with pytest.raises(ValueError, match="finite"):
        extension._safe_float_list([float("inf"), 1.])


def test_shae_corrupt_resume_is_not_silently_reset(extension, tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{broken")
    with pytest.raises(ValueError, match="Unreadable"):
        extension.ShaeBaselineStore(.95, path)
