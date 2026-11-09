"""CPU contracts using the real pinned VeRL DataProto and configuration schema."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.manifest import Manifest
from training.verl_backend.config import build_config, validate_recipe, verify_upstream
from training.verl_backend.dataset import ManifestDataset, occurrence_rows
from training.verl_backend.driver import update_once
from training.verl_backend.state import PaperState, completion_nll


ROOT = Path(__file__).resolve().parents[1]


def recipe():
    settings = json.loads((ROOT / "configs/rl_verl_4xa800.json").read_text())
    settings["curriculum"]["state"].update(input_dim=3, max_codes=4)
    return settings


def text(label="HCC"):
    return json.dumps({"detections": [dict(phase=p, label=label, bbox_2d=[100, 100, 300, 300]) for p in "apv"]})


def manifest():
    records = [dict(uid=f"uid{i}", patient_id=f"patient{i}", patient_label="HCC",
                    ground_truth=text(), prompt="diagnose", images={p: f"/{i}/{p}.png" for p in "apv"},
                    visual_vector=torch.eye(3)[i % 3].tolist()) for i in range(5)]
    return Manifest(tuple(recipe()["reward"]["severity_matrix"]),
                    {r["patient_id"]: "HCC" for r in records}, records)


def requires_verl():
    if importlib.util.find_spec("verl") is None:
        pytest.skip("pinned official VeRL source is not installed in this environment")
    verify_upstream()


def test_nll_keeps_eos_uses_double_mean_and_ignores_padding():
    values = torch.tensor([[-2, -4, float("nan")], [-1, -3, -5]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    actual = completion_nll(values, mask)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, torch.tensor([3., 3.], dtype=torch.float64))
    with pytest.raises(ValueError, match="at least one"):
        completion_nll(values, torch.zeros_like(mask))


def test_occurrences_keep_uid_distinct_rid_and_pad_only_tail():
    data = manifest()
    dataset = ManifestDataset(data.records, recipe()["model"]["processor_kwargs"])
    rows, records, rids, real = occurrence_rows(dataset, [1, 1, 3], epoch=2, offset=4,
                                              batch_size=4, group_size=2, seeds=list(range(8)))
    assert [r["uid"] for r in records] == ["uid1"] * 4 + ["uid3"] * 2 + ["uid1"] * 2
    assert rids == ["2:4"] * 2 + ["2:5"] * 2 + ["2:6"] * 2 + ["padding:2:7"] * 2
    assert real == [True] * 6 + [False] * 2
    assert [r["extra_info"]["rl_lhm_seed"] for r in rows] == list(range(8))
    assert [v["image"] for v in rows[0]["raw_prompt"][0]["content"][:3]] == ["/1/a.png", "/1/p.png", "/1/v.png"]
    assert all("min_pixels" in image and "max_pixels" in image for image in rows[0]["raw_prompt"][0]["content"][:3])


def score_tail(state):
    data = state.manifest
    dataset = ManifestDataset(data.records, state.configuration["model"]["processor_kwargs"])
    _, records, rids, real = occurrence_rows(dataset, [0, 0, 1], epoch=0, offset=0,
                                            batch_size=4, group_size=2, seeds=list(range(8)))
    texts = [text(), text("ICC"), text(), text(), text("ICC"), text(), "invalid", "invalid"]
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]] * 4)
    scored = state.score_and_advantage(texts=texts, old_log_probs=torch.full((8, 3), -2.),
                                       response_mask=mask, records=records, rids=rids, real=real)
    return scored


def test_tail_reward_and_uid_memory_exclude_padding_and_update_cvq_after_policy():
    state = PaperState(recipe(), manifest())
    scored = score_tail(state)
    assert set(state.baseline.entries) == {"uid0", "uid1"}
    rewards = scored["rewards"]
    expected = .95 * float(rewards[:2].mean()) + .05 * float(rewards[2:4].mean())
    assert state.baseline.entries["uid0"] == pytest.approx(expected)
    assert scored["response_mask"].sum() == 15
    assert scored["advantages"][-2:].eq(0).all()
    assert scored["response_mask"][-2:].eq(0).all()
    assert state.sampler.state.update_step == 0
    state.after_policy_update(scored)
    assert state.sampler.state.update_step == 3  # two distinct occurrences of uid0, one uid1


def test_epoch_resume_restores_sampling_jitter_and_curriculum_streams():
    settings, data = recipe(), manifest()
    original = PaperState(settings, data)
    assert sorted(original.epoch_order(0)) == list(range(len(data.records)))
    original.sampling_seeds(8)
    original.after_policy_update(score_tail(original))
    original.completed_epochs = 1
    saved = copy.deepcopy(original.state_dict())
    resumed = PaperState(settings, data)
    resumed.load_state_dict(saved)
    assert original.epoch_order(1) == resumed.epoch_order(1)
    assert original.sampling_seeds(16) == resumed.sampling_seeds(16)
    a, b = score_tail(original), score_tail(resumed)
    torch.testing.assert_close(a["advantages"], b["advantages"], rtol=0, atol=0)
    original.after_policy_update(a)
    resumed.after_policy_update(b)
    assert original.baseline.entries == resumed.baseline.entries
    torch.testing.assert_close(original.sampler.state.embedding, resumed.sampler.state.embedding, rtol=0, atol=0)


@pytest.mark.parametrize("replacement", [False, True])
def test_uniform_ablation_uses_order_permutations_and_restores_next_epoch(replacement):
    settings = recipe()
    settings["components"]["curriculum"] = False
    settings["sampling"]["replacement"] = replacement
    state = PaperState(settings, manifest())
    reference = torch.Generator().manual_seed(settings["order_seed"])
    resumed = None
    for epoch in range(3):
        expected = torch.randperm(len(state.records), generator=reference).tolist()
        actual = state.epoch_order(epoch)
        assert actual == expected
        assert sorted(actual) == list(range(len(state.records)))
        assert torch.equal(state.generators["order"].get_state(), reference.get_state())
        if resumed is not None:
            assert resumed.epoch_order(epoch) == expected
            resumed.completed_epochs = epoch + 1
        state.completed_epochs = epoch + 1
        if epoch == 0:
            saved = copy.deepcopy(state.state_dict())
            resumed = PaperState(settings, manifest())
            resumed.load_state_dict(saved)


def test_reject_unsupported_carrier_semantics():
    settings = recipe()
    validate_recipe(settings)
    settings["generation"]["temperature"] = .7
    with pytest.raises(ValueError, match="temperature"):
        validate_recipe(settings)
    settings = recipe()
    settings["policy"]["kl_bound"] = 20
    with pytest.raises(ValueError, match="bounds"):
        validate_recipe(settings)


def test_actual_upstream_config_composition_and_validation():
    requires_verl()
    from verl.utils.config import validate_config
    native = build_config(recipe())
    validate_config(native, use_reference_policy=True, use_critic=False)
    actor, rollout = native.actor_rollout_ref.actor, native.actor_rollout_ref.rollout
    assert actor.ppo_mini_batch_size == 4 and rollout.n == 2
    assert actor.loss_agg_mode == "token-mean" and actor.kl_loss_type == "low_var_kl"
    assert actor.kl_loss_coef == .01 and actor.ppo_epochs == 1
    assert native.algorithm.rollout_correction is None and not native.algorithm.use_kl_in_reward
    assert rollout.calculate_log_probs is False
    assert actor.checkpoint.save_contents == ["model", "optimizer", "extra", "hf_model"]
    assert "DeferredRewardManager" in rollout.agent.agent_loop_manager_class


def test_real_dataproto_update_contract_raw_logprob_reward_order_and_tail():
    requires_verl()
    from verl import DataProto
    paper = PaperState(recipe(), manifest())
    dataset = ManifestDataset(paper.records, recipe()["model"]["processor_kwargs"])
    events, observed = [], {}

    class Carrier:
        global_steps = 0
        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: text("ICC" if tokens[0] == 2 else "HCC"))

        def __init__(self):
            self.async_rollout_manager = SimpleNamespace(generate_sequences=self.generate)
            self.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: events.append("sleep"))

        def generate(self, prompts):
            events.append("generate")
            responses = torch.tensor([[1, 151645, 151643], [2, 3, 151645]] * 4)
            mask = torch.tensor([[1, 1, 0], [1, 1, 1]] * 4)
            p = torch.tensor([[7, 8]] * 8)
            output = DataProto.from_single_dict(dict(prompts=p, responses=responses, response_mask=mask,
                input_ids=torch.cat([p, responses], dim=-1), attention_mask=torch.cat([torch.ones_like(p), mask], dim=-1),
                position_ids=torch.arange(5).repeat(8, 1), rollout_log_probs=torch.full((8, 3), -99.)))
            output.non_tensor_batch.update(prompts.non_tensor_batch)
            output.non_tensor_batch["multi_modal_inputs"] = np.array([{"images_seqlens": [1, 1, 1]}] * 8, dtype=object)
            return output

        def _compute_old_log_prob(self, batch):
            events.append("old")
            assert not paper.baseline.entries
            return DataProto.from_single_dict(dict(old_log_probs=torch.full((8, 3), -2.),
                                                   entropys=torch.zeros(8, 3))), 0

        def _compute_ref_log_prob(self, batch):
            events.append("reference")
            return DataProto.from_single_dict(dict(ref_log_prob=torch.full((8, 3), -3.)))

        def _update_actor(self, batch):
            events.append("actor")
            assert set(paper.baseline.entries) == {"uid0"}
            assert paper.sampler.state.update_step == 0
            observed["batch"] = batch

    report = update_once(Carrier(), paper, dataset, [0], epoch=0, offset=0)
    assert events == ["generate", "sleep", "old", "reference", "actor"]
    assert report["mean_completion_nll"] == 2  # never the transformed rollout logprob 99
    assert report["retained_tokens"] == 5 and report["completion_count"] == 2
    batch = observed["batch"]
    assert batch.batch["response_mask"][2:].sum() == 0
    assert batch.batch["attention_mask"][2:].sum() > 0
    assert batch.batch["advantages"][2:].sum() == 0
    assert paper.sampler.state.update_step == 1


def test_actual_trainer_epoch_checkpoint_calls_and_state_restore(tmp_path):
    requires_verl()
    from training.verl_backend.trainer import PaperVeRLTrainer
    settings = recipe()
    settings["output_dir"] = str(tmp_path)
    paper = PaperState(settings, manifest())
    paper.epoch_order(0)
    paper.after_policy_update(score_tail(paper))
    paper.sampling_seeds(8)
    paper.completed_epochs = 1
    calls = []

    class Actor:
        def __init__(self):
            self.value = torch.tensor([3.])

        def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
            calls.append(("save", Path(local_path).parent.name, global_step))
            Path(local_path).mkdir()
            torch.save(self.value, Path(local_path) / "actor.pt")

        def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
            calls.append(("load", Path(local_path).parent.name))
            self.value = torch.load(Path(local_path) / "actor.pt", weights_only=True)

    def make(state):
        trainer = object.__new__(PaperVeRLTrainer)
        trainer.paper, trainer.identity = state, "contract-test"
        trainer.config = build_config(state.configuration)
        trainer.actor_rollout_wg = Actor()
        trainer.global_steps, trainer.history = 2, [{"epoch": 0}]
        return trainer

    original = make(paper)
    original._save_epoch()
    checkpoint = tmp_path / "epoch_0001"
    assert calls[0] == ("save", "epoch_0001.incomplete", 2)
    assert checkpoint.exists() and not (tmp_path / "epoch_0001.incomplete").exists()
    assert json.loads((checkpoint / "verl_config.json").read_text())["actor_rollout_ref"]["actor"]["strategy"] == "fsdp2"
    settings["resume_from"] = str(checkpoint)
    resumed = make(PaperState(settings, manifest()))
    resumed.actor_rollout_wg.value.zero_()
    resumed._restore_epoch()
    assert calls[-1] == ("load", "epoch_0001")
    assert resumed.actor_rollout_wg.value.item() == 3
    assert resumed.paper.completed_epochs == 1 and resumed.global_steps == 2
    assert resumed.paper.baseline.entries == paper.baseline.entries
    assert resumed.paper.epoch_order(1) == paper.epoch_order(1)
    assert resumed.paper.sampling_seeds(8) == paper.sampling_seeds(8)


def test_raw_agent_loop_retains_only_generated_tokens_and_request_seed():
    requires_verl()
    import asyncio
    from omegaconf import OmegaConf
    from training.verl_backend.rollout import RawSingleTurnLoop
    observed = {}
    loop = object.__new__(RawSingleTurnLoop)
    settings = recipe()
    settings["generation"]["suppress_token_ids"] = []
    loop.config = OmegaConf.create({"rl_lhm": settings})
    loop.rollout_config = SimpleNamespace(prompt_length=1024)

    async def multimodal(messages):
        return {"images": ["a", "p", "v"]}

    async def initial(messages, **kwargs):
        return [7, 8]

    async def generate(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(token_ids=[42, 151645, 88])

    loop.process_multi_modal_info = multimodal
    loop.ct_build_initial_tokens = initial
    loop._assert_mm_supported = lambda value: None
    loop._get_mm_processor_kwargs = lambda value: {"min_pixels": 65536}
    loop.build_sglang_video_payload = lambda *args: None
    loop.server_manager = SimpleNamespace(generate=generate)
    result = asyncio.run(loop.run({"temperature": 1., "top_p": .95}, raw_prompt=[],
                                  extra_info={"rid": "2:3", "member": 1, "rl_lhm_seed": 123}))
    assert result.response_ids == [42, 151645] and result.response_mask == [1, 1]
    assert result.reward_score is None and result.response_logprobs is None
    assert observed["sampling_params"]["seed"] == 123
    assert observed["sampling_params"]["max_tokens"] == 256
    assert observed["image_data"] == ["a", "p", "v"]


def test_actual_trainer_fit_tail_epoch_checkpoint_and_resume_order(tmp_path, monkeypatch):
    requires_verl()
    import training.verl_backend.trainer as module
    calls = []

    class Actor:
        def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
            calls.append(("save", global_step))
            Path(local_path).mkdir()

        def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
            calls.append(("load", Path(local_path).parent.name))

    def update(carrier, paper, dataset, indices, *, epoch, offset):
        calls.append(("update", epoch, offset, len(indices)))
        paper.sampling_seeds(8)
        return dict(epoch=epoch, step=offset // 4)

    monkeypatch.setattr(module, "update_once", update)

    def make(settings):
        paper = PaperState(settings, manifest())
        trainer = object.__new__(module.PaperVeRLTrainer)
        trainer.paper, trainer.identity = paper, "fit-contract"
        trainer.config = build_config(settings)
        trainer.actor_rollout_wg = Actor()
        trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: calls.append(("sync", step)))
        trainer.global_steps, trainer.history = 0, []
        trainer.train_dataset = ManifestDataset(paper.records, settings["model"]["processor_kwargs"])
        trainer._shutdown_dump_executor = lambda: calls.append(("shutdown",))
        return trainer

    settings = recipe()
    settings["output_dir"] = str(tmp_path)
    settings["loop"]["epochs"] = 1
    trainer = make(settings)
    trainer.fit()
    assert calls == [("sync", 0), ("update", 0, 0, 4), ("sync", 1),
                     ("update", 0, 4, 1), ("save", 2), ("sync", 2), ("shutdown",)]
    assert trainer.paper.completed_epochs == 1 and trainer.global_steps == 2
    calls.clear()
    settings["loop"]["epochs"] = 2
    settings["resume_from"] = str(tmp_path / "epoch_0001")
    resumed = make(settings)
    resumed.fit()
    assert calls == [("load", "epoch_0001"), ("sync", 2), ("update", 1, 0, 4), ("sync", 3),
                     ("update", 1, 4, 1), ("save", 4), ("sync", 4), ("shutdown",)]
    assert resumed.paper.completed_epochs == 2 and len(resumed.history) == 4
