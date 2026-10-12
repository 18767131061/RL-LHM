import pytest
import torch

from rl.completion_likelihood import (
    completion_likelihood,
    extract_valid_rollout_log_probabilities,
)
from rl.core import DACVQState, ShaeBaselineStore, compute_shae_advantage, vgs_sampling_weights


def test_shae_group_advantage_and_baseline_update():
    rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    store = ShaeBaselineStore(beta=0.5)
    advantage = compute_shae_advantage(
        rewards,
        mask,
        ["a", "a", "b", "b"],
        store=store,
        gate_mid=0.05,
        gate_temperature=0.015,
    )
    assert advantage.shape == rewards.shape
    assert advantage[:2, 0].tolist() == pytest.approx([-1.0, 1.0], abs=1e-4)
    assert advantage[:, 1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert store.get("a") == pytest.approx(2.0)


def test_dacvq_initialization_quantization_and_state_roundtrip():
    state = DACVQState(max_codes=2, input_dim=2, seed=7, dead_code_replace_probability=0.0)
    assert state.online_init_step(torch.tensor([1.0, 0.0]))
    assert state.online_init_step(torch.tensor([0.0, 1.0]))
    assert state.is_initialized
    assert state.quantize(torch.tensor([0.9, 0.1]))[0] == 0
    state.update("hard", torch.tensor([1.0, 0.0]), mean_reward=-4.0)
    state.update("easy", torch.tensor([0.0, 1.0]), mean_reward=2.0)
    assert state.difficulty_weights()[0] > state.difficulty_weights()[1]

    restored = DACVQState(max_codes=2, input_dim=2, seed=1)
    restored.load_state_dict(state.state_dict())
    assert restored.sample_to_code == state.sample_to_code
    assert torch.allclose(restored.reward_ema, state.reward_ema)


def test_vgs_weights_keep_unknown_vectors_exploratory():
    state = DACVQState(max_codes=2, input_dim=2, seed=7, dead_code_replace_probability=0.0)
    state.online_init_step(torch.tensor([1.0, 0.0]))
    state.online_init_step(torch.tensor([0.0, 1.0]))
    state.reward_ema[:] = torch.tensor([-2.0, 1.0])
    weights = vgs_sampling_weights(
        state,
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), None],
    )
    assert weights[0] > weights[1]
    assert weights[2] > weights[0]


def test_completion_likelihood_and_mask_alignment():
    ids, log_probabilities = extract_valid_rollout_log_probabilities(
        torch.tensor([10, 11, 0]),
        torch.tensor([1, 1, 1, 1, 0]),
        torch.tensor([-0.2, -0.4, -9.0]),
    )
    assert ids.tolist() == [10, 11]
    nll, confidence = completion_likelihood(log_probabilities, 2)
    assert nll == pytest.approx(0.3)
    assert confidence == pytest.approx(torch.exp(torch.tensor(-0.3)).item())


def test_single_generated_token_is_not_squeezed_to_scalar():
    ids, probabilities = extract_valid_rollout_log_probabilities(
        torch.tensor([[10]]), torch.tensor([[1, 1]]), torch.tensor([[-.2]]))
    assert ids.tolist() == [10]
    assert completion_likelihood(probabilities, 1)[0] == pytest.approx(.2)
