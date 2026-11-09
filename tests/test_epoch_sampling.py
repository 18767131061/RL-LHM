"""Exercise epoch sampling through the actual training loop and VeRL state."""
from copy import deepcopy

import pytest
import torch

from data.manifest import Manifest
from training.checkpoint import restore_checkpoint, save_checkpoint
from training.distributed import Runtime
from training.engine import ComponentSwitches, PolicyParameters, fit_rl
from training.verl_backend.state import PaperState
from tests.test_distributed import TinyBackend
from tests.test_verl_backend import manifest, recipe


class RecordingBackend(TinyBackend):
    def __init__(self):
        super().__init__()
        self.generated_uids = []

    def generate(self, records):
        self.generated_uids.append(records[0]["uid"])
        return super().generate(records)


def training_objects(settings, data):
    backend, reference = RecordingBackend(), TinyBackend()
    runtime = Runtime(backend.model, torch.optim.AdamW(backend.model.parameters(), lr=1e-3))
    return backend, reference, runtime, PaperState(settings, data)


def run_epoch(objects, epoch):
    backend, reference, runtime, paper = objects
    backend.generated_uids.clear()
    history = fit_rl(
        backend, reference, paper.records, runtime=runtime, epochs=epoch + 1,
        start_epoch=epoch, batch_size=3, microbatch_size=1, group_size=2,
        policy_updates=1, max_grad_norm=1., advantage_epsilon=1e-6,
        switches=ComponentSwitches(**paper.configuration["components"]),
        policy=PolicyParameters(**paper.configuration["policy"]),
        reward_config=paper.reward_config, class_counts=paper.class_counts,
        baseline_store=None, shae_parameters=None, curriculum_sampler=paper.sampler,
        visual_vectors=paper.visual_vectors, order_generator=paper.generators["order"],
        shae_generator=None, on_epoch=None,
        sampling_replacement=paper.configuration["sampling"]["replacement"],
    )
    # Each dataset occurrence produces exactly one pair, including the short tail.
    assert backend.generated_uids[::2] == backend.generated_uids[1::2]
    assert len(backend.generated_uids) == 2 * len(paper.records)
    assert [item["input_count"] for item in history] == [3, 2]
    paper.completed_epochs = epoch + 1
    return [int(uid) for uid in backend.generated_uids[::2]], history


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("curriculum", [False, True])
def test_backends_share_epoch_sampling_and_checkpoint_resume(tmp_path, replacement, curriculum):
    settings = recipe()
    settings["components"].update(shae=False, curriculum=curriculum)
    settings["sampling"]["replacement"] = replacement
    settings["curriculum"]["state"]["max_codes"] = 3
    data = manifest()
    records = [dict(record, uid=str(index)) for index, record in enumerate(data.records)]
    # The missing-feature multiplier makes later VGS probabilities nonuniform.
    records[-1]["visual_vector"] = None
    data = Manifest(data.output_classes, data.patients, records)
    original = training_objects(settings, data)
    paper = original[3]
    order_rng = torch.Generator().manual_seed(settings["order_seed"])
    sampling_rng = torch.Generator().manual_seed(settings["curriculum"]["sampling_seed"])
    expected_orders = []
    checkpoint = tmp_path / "epoch_0001"

    for epoch in range(3):
        # Both backends see the same completed-epoch state; VeRL owns its own copy.
        verl = PaperState(settings, data)
        verl.load_state_dict(deepcopy(paper.state_dict()))
        verl_order = verl.epoch_order(epoch)
        if curriculum:
            if epoch == 0:
                expected = torch.randperm(len(records), generator=sampling_rng).tolist()
            else:
                assert paper.sampler.state.is_initialized
                probabilities = paper.sampler.probabilities()
                assert not torch.allclose(probabilities, torch.full_like(probabilities, 1 / len(records)))
                expected = torch.multinomial(probabilities, len(records), replacement=replacement,
                                             generator=sampling_rng).tolist()
        else:
            expected = torch.randperm(len(records), generator=order_rng).tolist()
        actual, history = run_epoch(original, epoch)
        assert actual == verl_order == expected
        if not curriculum or not replacement or epoch == 0:
            assert sorted(actual) == list(range(len(records)))
        assert torch.equal(paper.generators["order"].get_state(), order_rng.get_state())
        assert torch.equal(paper.generators["order"].get_state(), verl.generators["order"].get_state())
        if curriculum:
            assert paper.sampler.epoch_count == epoch + 1
            assert torch.equal(paper.generators["sampling"].get_state(), sampling_rng.get_state())
        expected_orders.append(actual)
        if epoch == 0:
            save_checkpoint(
                original[0], original[2], directory=checkpoint, stage="rl", completed_epoch=epoch,
                identity="epoch-order", configuration=settings, history=history,
                generators=paper.generators, baseline_store=None, curriculum_sampler=paper.sampler,
            )

    resumed = training_objects(settings, data)
    resumed[0].model.load_state_dict(torch.load(checkpoint / "tiny.pt", weights_only=True))
    start, _ = restore_checkpoint(
        resumed[2], directory=checkpoint, stage="rl", identity="epoch-order",
        generators=resumed[3].generators, baseline_store=None, curriculum_sampler=resumed[3].sampler,
    )
    assert start == 1
    resumed[3].completed_epochs = start
    for epoch in range(start, 3):
        actual, _ = run_epoch(resumed, epoch)
        assert actual == expected_orders[epoch]
    assert torch.equal(resumed[3].generators["order"].get_state(), paper.generators["order"].get_state())
