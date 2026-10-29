"""Four Gloo processes exercise the same global-batch loops used by ZeRO."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from training.distributed import Runtime
from training.engine import ComponentSwitches, PolicyParameters, ShaeParameters, fit_sft, fit_rl
from training.checkpoint import save_checkpoint, restore_checkpoint
from rl import ShaeBaselineStore, VGSSampler
from tests.test_algorithms import completion, triplet, reward_config, memory


class TinyBackend:
    def __init__(self, distributed=False):
        torch.manual_seed(19)
        self.model = torch.nn.Linear(3, 2, bias=False).double()
        self.forward_model = DistributedDataParallel(self.model) if distributed else self.model
        self.visits = {}

    def prepare_sft(self, records):
        record = records[0]
        tokens = int(record["uid"]) % 4 + 1
        return {"x": torch.tensor([[1., int(record["uid"]) / 10, -0.5]], dtype=torch.float64),
                "tokens": tokens}, torch.ones(1, tokens, dtype=torch.bool)

    def sft_prepared_loss_sum(self, inputs, mask):
        logits = self.forward_model(inputs["x"])
        return -logits.log_softmax(-1)[0, 0] * mask.sum()

    def generate(self, records):
        record = records[0]
        member = self.visits.get(record["uid"], 0) % 2
        self.visits[record["uid"]] = member + 1
        inputs, mask = self.prepare_sft(records)
        # Different completion lengths test global token weighting.
        tokens = inputs.pop("tokens") + member
        inputs["target"] = torch.tensor(member)
        mask = torch.ones(1, tokens, dtype=torch.bool)
        rollout = SimpleNamespace(texts=[completion(triplet("ICC" if member else "HCC"))],
                                  model_inputs=inputs, response_mask=mask, completion_nlls=[1.0])
        rollout.old_log_probabilities = self.log_probabilities(rollout).detach()
        return rollout

    def log_probabilities(self, rollout):
        forward = self.forward_model if torch.is_grad_enabled() else self.model
        logits = forward(rollout.model_inputs["x"])
        return logits.log_softmax(-1)[:, rollout.model_inputs["target"]].reshape(1, 1).expand_as(rollout.response_mask)

    def save_pretrained(self, directory):
        torch.save(self.model.state_dict(), Path(directory) / "tiny.pt")


def objects(distributed=False):
    backend, reference = TinyBackend(distributed), TinyBackend()
    runtime = Runtime(backend.model, torch.optim.AdamW(backend.model.parameters(), lr=1e-3))
    generator = torch.Generator().manual_seed(3)
    state = memory()
    vectors = [torch.eye(3)[i % 3] for i in range(7)]
    sampling = torch.Generator().manual_seed(5)
    sampler = VGSSampler(state, vectors, generator=sampling, use_vgs=True, epoch_count=0,
                         replacement=True, missing_feature_multiplier=0.5, weight_floor=1e-6)
    baseline = ShaeBaselineStore(beta=0.9)
    generators = {"order": generator, "curriculum": state.generator, "sampling": sampling}
    return backend, reference, runtime, sampler, baseline, generators


def fit(stage, obj, start, end, callback=None):
    backend, reference, runtime, sampler, baseline, generators = obj
    records = [{"uid": str(i), "ground_truth": completion(triplet())} for i in range(7)]
    common = dict(runtime=runtime, epochs=end, start_epoch=start, batch_size=5,
                  microbatch_size=1, max_grad_norm=1, on_epoch=callback)
    if stage == "sft":
        return fit_sft(backend, records, generator=generators["order"], **common)
    return fit_rl(backend, reference, records, group_size=2, policy_updates=2,
                  advantage_epsilon=1e-6, switches=ComponentSwitches(True, True, True),
                  policy=PolicyParameters(0.2, 3, 0.01, 10, 10), reward_config=reward_config(),
                  class_counts={"HCC": 3, "ICC": 2, "NC": 1}, baseline_store=baseline,
                  shae_parameters=ShaeParameters(0.05, 0.015, 0), curriculum_sampler=sampler,
                  visual_vectors={str(i): torch.eye(3)[i % 3] for i in range(7)},
                  order_generator=generators["order"], shae_generator=None,
                  sampling_replacement=True, **common)


def worker(rank, world, rendezvous, output, stage):
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        obj = objects(world > 1)
        history = fit(stage, obj, 0, 2)
        full = deepcopy(obj[0].model.state_dict())
        baseline = dict(obj[4].entries)
        embedding = obj[3].state.embedding.clone()
        # Save after one epoch, then initialize new model/optimizer and resume.
        obj = objects(world > 1)
        def save(epoch, metrics):
            save_checkpoint(obj[0], obj[2], directory=Path(output) / f"{stage}_{world}",
                            stage=stage, completed_epoch=epoch, identity="tiny", configuration={},
                            history=metrics, generators=obj[5], baseline_store=obj[4], curriculum_sampler=obj[3])
        fit(stage, obj, 0, 1, save)
        restored = objects(world > 1)
        directory = Path(output) / f"{stage}_{world}"
        restored[0].model.load_state_dict(torch.load(directory / "tiny.pt", weights_only=True))
        start, _ = restore_checkpoint(restored[2], directory=directory, stage=stage, identity="tiny",
                                      generators=restored[5], baseline_store=restored[4], curriculum_sampler=restored[3])
        fit(stage, restored, start, 2)
        for name, value in full.items():
            torch.testing.assert_close(value, restored[0].model.state_dict()[name], rtol=1e-12, atol=1e-12)
        assert baseline == restored[4].entries
        torch.testing.assert_close(embedding, restored[3].state.embedding, rtol=0, atol=0)
        all_baselines = obj[2].gather(baseline)
        assert all(value == baseline for value in all_baselines)
        if rank == 0:
            torch.save(dict(weights=full, baseline=baseline, embedding=embedding, history=history),
                       Path(output) / f"{stage}_{world}.pt")
    finally:
        if world > 1:
            dist.destroy_process_group()


@pytest.mark.parametrize("stage", ["sft", "rl"])
def test_four_rank_equivalence_and_resume(tmp_path, stage):
    worker(0, 1, str(tmp_path / "unused"), str(tmp_path), stage)
    mp.spawn(worker, args=(4, str(tmp_path / "rendezvous"), str(tmp_path), stage), nprocs=4, join=True)
    one = torch.load(tmp_path / f"{stage}_1.pt", weights_only=True)
    four = torch.load(tmp_path / f"{stage}_4.pt", weights_only=True)
    for key in one["weights"]:
        torch.testing.assert_close(one["weights"][key], four["weights"][key], rtol=1e-10, atol=1e-10)
    assert one["baseline"] == four["baseline"]
    torch.testing.assert_close(one["embedding"], four["embedding"], rtol=0, atol=0)
    for first, second in zip(one["history"], four["history"]):
        assert first.keys() == second.keys()
        for key in first:
            assert first[key] == pytest.approx(second[key], abs=1e-10)
