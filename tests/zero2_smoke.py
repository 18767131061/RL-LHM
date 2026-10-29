"""Four-CUDA ZeRO-2 accumulation and checkpoint smoke test.

Run: torchrun --standalone --nproc_per_node=4 -m tests.zero2_smoke --output /tmp/rl-lhm-zero-test
"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from tests.test_backend import tiny_backend
from training.distributed import initialize, zero2_runtime
from training.checkpoint import save_checkpoint, restore_checkpoint


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
