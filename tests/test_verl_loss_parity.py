"""Numerical parity with the actual pinned VeRL objective, on CPU tensors.

These tests exercise upstream functions without copying or rewriting them.
They do not run the distributed FSDP/vLLM training stack.
"""
from pathlib import Path
import hashlib
import inspect
import json

import pytest
import torch

pytest.importorskip("verl", reason="the optional pinned VeRL carrier is not installed")

from verl.trainer.ppo import core_algos
from verl.workers.config.actor import ActorConfig

from rl import grpo_token_loss


@pytest.fixture(scope="module", autouse=True)
def require_pinned_objective():
    manifest = Path(__file__).parents[1] / "training/verl_backend/upstream_api.json"
    expected = json.loads(manifest.read_text())["sha256"]["trainer/ppo/core_algos.py"]
    source = Path(inspect.getfile(core_algos))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, "VeRL objective is not the pinned source"


def actor_config(token_count, dp_size=1):
    return ActorConfig(
        strategy="fsdp2", rollout_n=2, ppo_micro_batch_size_per_gpu=1,
        clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=3.,
        global_batch_info={"batch_num_tokens": token_count, "dp_size": dp_size},
    )


def upstream_loss(current, old, reference, advantage, mask, *, token_count, dp_size=1, beta=.07):
    configuration = actor_config(token_count, dp_size)
    policy, metrics = core_algos.compute_policy_loss_vanilla(
        old_log_prob=old, log_prob=current, advantages=advantage,
        response_mask=mask, loss_agg_mode="token-mean", config=configuration,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    kl = core_algos.kl_penalty(current, reference, "low_var_kl")
    reference_loss = core_algos.agg_loss(
        kl, mask, "token-mean", batch_num_tokens=token_count, dp_size=dp_size,
    )
    return policy + beta * reference_loss


def local_loss(current, old, reference, advantage, mask, *, token_count, beta=.07):
    return grpo_token_loss(
        current, old, reference, advantage, mask,
        clip_ratio=.2, dual_clip=3., kl_coefficient=beta,
        log_ratio_bound=20., kl_bound=10., total_token_count=token_count,
    )


@pytest.mark.parametrize("current,old,reference,advantage,mask", [
    pytest.param(
        [[-.7, -1.5, -1.9], [-1.3, -.9, -2.1]],
        [[-.8, -1.4, -2.0], [-1.4, -1., -2.2]],
        [[-.8, -1.2, -1.7], [-1.5, -.8, -2.0]],
        [[1., -.4, -.7], [1., -.3, .6]],
        [[1, 1, 1], [1, 1, 1]], id="ordinary-positive-negative"),
    pytest.param(
        [[-.1, -2., -1.05, -.7]], [[-4., -.8, -1., -.9]],
        [[-.4, -1.5, -1.2, -.8]], [[-1., -1., 1., .5]], [[1, 1, 1, 1]],
        id="negative-dual-clip"),
    pytest.param(
        [[-1., -30., -21.1, -1., -2.]], [[-30., -1., -1., -21.1, -2.1]],
        [[-1.1, -29.9, -21., -.9, -1.8]], [[-1., 1., 1., -1., .4]],
        [[1, 1, 1, 1, 1]], id="log-ratio-both-bounds"),
    pytest.param(
        [[-3., -3., -30., -1., -2.]], [[-3., -3., -30., -1., -2.]],
        [[-2., -5., -1., -30., -2.0000001]], [[0., 0., 0., 0., 0.]],
        [[1, 1, 1, 1, 1]], id="reference-kl-and-clipping"),
    pytest.param(
        [[-1.1, -1.8, -30., -1.], [-5., -3., -2., -1.], [-1.2, -8., -1., -1.]],
        [[-1., -2., -1., -30.], [-1., -30., -3., -4.], [-1.1, -1., -30., -1.]],
        [[-.9, -2.1, -1., -30.], [-30., -1., -3., -4.], [-1.4, -30., -1., -1.]],
        [[1., -.7, 99., 99.], [99., 99., 99., 99.], [-.8, 99., 99., 99.]],
        [[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], id="partial-and-wholly-padded-rows"),
])
def test_actual_upstream_loss_and_gradient(current, old, reference, advantage, mask):
    current = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    old, reference, advantage = [torch.tensor(value, dtype=torch.float64) for value in (old, reference, advantage)]
    mask = torch.tensor(mask, dtype=torch.float64)
    token_count = int(mask.sum())
    native = upstream_loss(current, old, reference, advantage, mask, token_count=token_count)
    ours = local_loss(current, old, reference, advantage, mask, token_count=token_count)
    native_gradient, = torch.autograd.grad(native, current, retain_graph=True)
    local_gradient, = torch.autograd.grad(ours, current)
    torch.testing.assert_close(native, ours, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(native_gradient, local_gradient, rtol=1e-12, atol=1e-12)
    assert torch.equal(native_gradient[mask == 0], torch.zeros_like(native_gradient[mask == 0]))


def test_global_token_normalization_survives_padding_rank_and_microbatches():
    """Model the FSDP average after upstream dp_size scaling, without GPUs."""
    current = torch.tensor([[-1.1, -2.0, -1.5], [-1., -3., -4.],
                            [-2., -1., -3.], [-.9, -1.8, -1.3]],
                           dtype=torch.float64, requires_grad=True)
    old = current.detach() + torch.tensor([[.05, -.1, .08], [.04, -.2, -.1],
                                          [.2, -.3, .1], [-.06, .03, .2]], dtype=torch.float64)
    reference = current.detach() - .15
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=torch.float64)
    advantage = torch.tensor([[1., -.4, .3], [-1., 2., 3.], [4., 5., 6.], [.7, -.8, 9.]], dtype=torch.float64)
    count = int(mask.sum())
    reference_loss = local_loss(current, old, reference, advantage, mask, token_count=count)
    # Four ranks receive one row each; rank 2 contains attention tokens but no
    # retained loss tokens, exactly as a padded final prompt batch can produce.
    rank_losses = [upstream_loss(current[i:i+1], old[i:i+1], reference[i:i+1],
                                 advantage[i:i+1], mask[i:i+1], token_count=count, dp_size=4)
                   for i in range(4)]
    averaged = torch.stack(rank_losses).mean()
    # Splitting the full batch into two microbatches must use the SAME global
    # denominator, rather than normalizing each microbatch independently.
    microbatch_sum = sum(upstream_loss(current[i:i+2], old[i:i+2], reference[i:i+2],
                                       advantage[i:i+2], mask[i:i+2], token_count=count)
                         for i in (0, 2))
    expected_gradient, = torch.autograd.grad(reference_loss, current, retain_graph=True)
    for loss in (averaged, microbatch_sum):
        gradient, = torch.autograd.grad(loss, current, retain_graph=True)
        torch.testing.assert_close(loss, reference_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert rank_losses[2].item() == 0
