"""Manuscript equations, EMA lifecycle, and epoch sampling contracts."""
import math

import pytest
import torch

from rl.core import ShaeBaselineStore, compute_grpo_advantage, compute_shae_advantage
from rl.curriculum import DACVQState, VGSSampler
from rl.policy_contract import grpo_token_loss


def codebook(**overrides):
    settings = dict(max_codes=2, input_dim=2, feature_decay=.5, reward_decay=.9,
                    difficulty_beta=3., dead_code_threshold=.25,
                    dead_code_replace_probability=1., revival_noise_std=0.,
                    diversity_threshold=.85, reward_min=-6., reward_max=3.,
                    mass_epsilon=1e-8, generator=torch.Generator().manual_seed(8))
    settings.update(overrides)
    return DACVQState(**settings)


def test_shae_history_uses_previous_baseline_jitter_and_population_std():
    rewards = torch.tensor([1., 1.01], dtype=torch.float64)
    mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
    store = ShaeBaselineStore(beta=.95)
    store.update('case', .5)
    rng = torch.Generator().manual_seed(23)
    reference_rng = torch.Generator().manual_seed(23)
    actual = compute_shae_advantage(
        rewards, mask, ['occurrence'] * 2, sample_index=['case'] * 2, store=store,
        gate_mid=.05, gate_temperature=.015, noise_std=.001, epsilon=1e-6,
        generator=rng,
    )
    jitter = torch.randn((2,), dtype=torch.float64, generator=reference_rng) * .001
    sigma = .005
    group = torch.tensor([-.005, .005], dtype=torch.float64) / (sigma + 1e-6)
    residual = rewards - .5 + jitter
    # For two values the population SD equals half the absolute difference.
    historical = residual / ((residual[1] - residual[0]).abs() / 2 + 1e-6)
    gate = 1 / (1 + math.exp(-((sigma - .05) / .015)))
    expected = gate * group + (1 - gate) * historical
    torch.testing.assert_close(actual, expected[:, None] * mask)
    assert store.get('case') == pytest.approx(.95 * .5 + .05 * 1.005)
    assert torch.equal(rng.get_state(), reference_rng.get_state())


def test_grpo_off_branch_uses_sample_standard_deviation():
    actual = compute_grpo_advantage(
        torch.tensor([1., 3.], dtype=torch.float64), torch.ones((2, 1)),
        ['group', 'group'], epsilon=1e-6,
    )
    expected = torch.tensor([-1., 1.], dtype=torch.float64) / (math.sqrt(2) + 1e-6)
    torch.testing.assert_close(actual[:, 0], expected)


def test_high_reward_low_mass_code_can_revive_and_inactive_high_mass_cannot():
    state = codebook()
    state.online_init_step(torch.tensor([1., 0.]))
    state.online_init_step(torch.tensor([0., 1.]))
    # Each maximum-reward visit has zero difficulty weight. After three visits,
    # assigned mass is 1/8 < 1/4 and that first code is revived.
    for _ in range(3):
        state.update('active', torch.tensor([1., 0.]), 3.)
    assert state.revival_count == 1
    assert state.ema_cluster_size[0] == 1
    assert state.reward_seen[0] == 0
    # Unassigned state is not decayed solely because it was not visited.
    assert state.ema_cluster_size[1] == 1
    assert state.reward_seen[1] == 0
    assert torch.equal(state.embedding[1], torch.tensor([0., 1.], dtype=torch.float64))


def test_revival_chooses_first_low_mass_code_not_oldest_activity():
    state = codebook(feature_decay=0.)
    state.online_init_step(torch.tensor([1., 0.]))
    state.online_init_step(torch.tensor([0., 1.]))
    state.ema_cluster_size[:] = .1
    state.last_seen[:] = torch.tensor([100, 0])
    state.update('case', torch.tensor([1., 0.]), 3.)
    assert state.ema_cluster_size[0] == 1
    assert state.ema_cluster_size[1] == pytest.approx(.1)


@pytest.mark.parametrize('use_vgs', [False, True])
def test_first_uninitialized_epoch_permutation_then_all_one_replacement(use_vgs):
    rng = torch.Generator().manual_seed(5)
    reference_rng = torch.Generator().manual_seed(5)
    sampler = VGSSampler(codebook(), [None] * 10, missing_feature_multiplier=1.5,
                         weight_floor=1e-4, generator=rng, use_vgs=use_vgs,
                         epoch_count=0, replacement=True)
    first = sampler.sample_epoch()
    assert torch.equal(first, torch.randperm(10, generator=reference_rng))
    second = sampler.sample_epoch()
    expected = torch.multinomial(torch.full((10,), .1, dtype=torch.float64),
                                 10, replacement=True, generator=reference_rng)
    assert torch.equal(second, expected)
    assert sampler.epoch_count == 2


def test_appendix_kl_cap_is_ten_separate_from_log_ratio_cap():
    current = torch.tensor([[-20.]], requires_grad=True)
    actual = grpo_token_loss(
        current, torch.tensor([[-20.]]), torch.tensor([[0.]]),
        torch.tensor([0.]), torch.ones((1, 1)), clip_ratio=.2, dual_clip=3.,
        kl_coefficient=.01, log_ratio_bound=20., kl_bound=10., total_token_count=None,
    )
    assert actual.item() == pytest.approx(.1)
    actual.backward()
    assert current.grad.item() == 0
