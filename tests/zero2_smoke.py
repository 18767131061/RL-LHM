"""Four-CUDA ZeRO-2 accumulation and checkpoint smoke test.

Run: torchrun --standalone --nproc_per_node=4 -m tests.zero2_smoke --output /tmp/rl-lhm-zero-test
"""
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from tests.test_backend import tiny_backend
from training.distributed import initialize, zero2_runtime
from training.checkpoint import save_checkpoint, restore_checkpoint


class GradientProbe(torch.nn.Module):
    """One vector spans all four optimizer partitions with a known gradient."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8))

    def forward(self, coefficients):
        return (self.weight * coefficients).sum()


def probe_objects():
    model = GradientProbe().to(device=torch.cuda.current_device(), dtype=torch.bfloat16)
    backend = SimpleNamespace(model=model)
    runtime = zero2_runtime(backend, dict(learning_rate=0.01, betas=[0.9, 0.999], epsilon=1e-8,
                                         weight_decay=0.01, amsgrad=False), 1.0)
    reference = torch.nn.Parameter(torch.zeros(8))
    optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0.01)
    return runtime, reference, optimizer


def probe_update(runtime, reference, optimizer, rows, denominator):
    from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param, safe_get_full_optimizer_state

    optimizer.zero_grad(set_to_none=True)
    # One backward over the complete, unsplit global batch is the reference.
    ((rows * reference).sum() / denominator).backward()
    slots = runtime.slots(list(rows))
    runtime.begin(len(slots))
    for index, (_, row, real) in enumerate(slots):
        runtime.boundary(index == len(slots) - 1)
        loss = runtime.engine(row.to(device=runtime.device, dtype=torch.bfloat16))
        runtime.backward(loss * int(real) / denominator)
    parameter = runtime.model.weight
    gradient = safe_get_full_grad(parameter).detach().float().cpu()
    torch.testing.assert_close(gradient, reference.grad, rtol=0, atol=0)
    expected_norm = torch.nn.utils.clip_grad_norm_([reference], 1.0).item()
    norm = runtime.step(1.0)
    torch.testing.assert_close(torch.tensor(norm), torch.tensor(expected_norm), rtol=1e-6, atol=1e-6)
    optimizer.step()
    master = safe_get_full_fp32_param(parameter).detach().cpu()
    torch.testing.assert_close(master, reference.detach(), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(parameter.detach().cpu(), reference.detach().to(torch.bfloat16),
                               rtol=0, atol=1e-4)
    for key in ("exp_avg", "exp_avg_sq"):
        actual = safe_get_full_optimizer_state(parameter, key).detach().cpu()
        torch.testing.assert_close(actual, optimizer.state[reference][key], rtol=1e-5, atol=1e-7)
    replicas = runtime.gather(parameter.detach().cpu().tolist())
    assert all(value == replicas[0] for value in replicas)
    return master.clone()


def probe_cases():
    direction = torch.tensor([1., -1., 2., -2., 0., 1., 0., -1.])

    def ranked(pattern):
        return torch.stack([row * (rank + 1) for row in pattern for rank in range(4)])

    vector = torch.tensor([[1., 0., 0., 1., 0., 0., 0., 0.],
                           [0., 10., 0., -4., 0., 1., 0., 0.],
                           [-1., 0., 0., 1., 0., 0., 0., 0.]])
    return [
        (ranked([direction * i for i in (1, 2, 4)]), 4),
        (ranked([direction * i for i in (1, -4, 1)]), 4),
        (ranked([direction] * 32), 128),
        (ranked(vector), 4),
        (ranked(vector)[:9], 16),  # three slots, uneven rank counts and zero-weight padding
        (direction[None], 1),      # one-slot tail, three ranks contribute zero
        (ranked([direction, -direction * 2]), 8),
    ]


def check_gradient_equivalence(output):
    # Start fresh per case so Adam's moments cannot conceal a wrong first update.
    for rows, denominator in probe_cases():
        runtime, reference, optimizer = probe_objects()
        probe_update(runtime, reference, optimizer, rows, denominator)
        runtime.engine.destroy()
        del runtime

    runtime, reference, optimizer = probe_objects()
    cases = probe_cases()
    probe_update(runtime, reference, optimizer, *cases[2])
    checkpoint = str(Path(output) / "gradient_probe")
    runtime.engine.save_checkpoint(checkpoint, tag="state", save_latest=False)
    saved_reference = reference.detach().clone()
    saved_optimizer = deepcopy(optimizer.state_dict())
    continuation = [cases[4], cases[5], cases[1], cases[2]]
    expected = [probe_update(runtime, reference, optimizer, *case) for case in continuation]
    runtime.engine.destroy()
    del runtime
    runtime, reference, optimizer = probe_objects()
    loaded, _ = runtime.engine.load_checkpoint(checkpoint, tag="state", load_optimizer_states=True)
    assert loaded is not None
    with torch.no_grad():
        reference.copy_(saved_reference)
    optimizer.load_state_dict(saved_optimizer)
    for case, value in zip(continuation, expected):
        actual = probe_update(runtime, reference, optimizer, *case)
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    runtime.engine.destroy()
    if dist.get_rank() == 0:
        print("PASS: four-rank gradients, clipped CPUAdam updates, 3/32 slots, mixed signs, tails and resume")


def objects():
    backend, inputs = tiny_backend()
    backend.model.to(device=torch.cuda.current_device(), dtype=torch.bfloat16)
    backend.enable_training(128)
    runtime = zero2_runtime(backend, dict(learning_rate=1e-3, betas=[0.9, 0.999], epsilon=1e-8,
                                         weight_decay=0.01, amsgrad=False), 1.0)
    return backend, inputs, runtime


def update(backend, inputs, runtime, slots):
    backend.model.train()
    runtime.begin(slots)
    count = int(runtime.sum(sum(i + 1 for i in range(slots))))
    for i in range(slots):
        runtime.boundary(i == slots - 1)
        logp = backend._token_log_probabilities(inputs)
        # Variable local token counts and nonidentical losses across ranks.
        loss = -logp[:, -(i + 1):].sum() * (1 + runtime.rank / 10) / count
        runtime.backward(loss)
    norm = runtime.step(1.0)
    assert norm > 0
    for parameter in backend.model.parameters():
        assert torch.isfinite(parameter).all()
    return norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    initialize(4)
    try:
        check_gradient_equivalence(args.output)
        backend, inputs, runtime = objects()
        initial = {name: p.detach().clone() for name, p in backend.model.named_parameters()}
        update(backend, inputs, runtime, 3)
        assert any(not torch.equal(initial[name], p) for name, p in backend.model.named_parameters())
        directory = Path(args.output) / "epoch_0001"
        generators = {"order": torch.Generator().manual_seed(10)}
        save_checkpoint(backend, runtime, directory=directory, stage="sft", completed_epoch=0,
                        identity="zero-smoke", configuration={}, history=[], generators=generators,
                        baseline_store=None, curriculum_sampler=None)
        # One-slot partial batch exercises CPU offload's non-accumulating path.
        update(backend, inputs, runtime, 1)
        expected = {name: p.detach().cpu().clone() for name, p in backend.model.named_parameters()}
        del backend, runtime
        torch.cuda.empty_cache()
        backend, inputs, runtime = objects()
        restore_checkpoint(runtime, directory=directory, stage="sft", identity="zero-smoke",
                           generators=generators, baseline_store=None, curriculum_sampler=None)
        update(backend, inputs, runtime, 1)
        for name, parameter in backend.model.named_parameters():
            torch.testing.assert_close(parameter.cpu(), expected[name], rtol=0, atol=0)
        # Every rank must have identical updated policy parameters.
        checksum = sum(float(p.float().sum()) for p in backend.model.parameters())
        assert len(set(runtime.gather(checksum))) == 1
        if runtime.rank == 0:
            print("PASS: four-rank BF16 ZeRO-2, accumulated update, partial batch, checkpoint continuation")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
