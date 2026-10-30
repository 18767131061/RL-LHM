"""SHAE and group-relative advantages from already clipped scalar rewards."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class ShaeBaselineStore:
    """Input-UID-keyed reward EMA; keep this object across rollout visits.

    ``beta`` is the baseline decay. State is deliberately held in memory;
    persistence and distributed ownership belong to the caller.
    """

    def __init__(self, *, beta: float):
        if not math.isfinite(beta) or not 0.0 <= beta < 1.0:
            raise ValueError("beta must be finite and in [0, 1)")
        self.beta = float(beta)
        self.entries: dict[str, float] = {}

    def get(self, sample_key: str) -> float | None:
        return self.entries.get(sample_key)

    def update(self, sample_key: str, reward_mean: float) -> None:
        if not isinstance(sample_key, str) or not sample_key.strip():
            raise ValueError("sample keys must be nonempty strings")
        if not math.isfinite(reward_mean):
            raise ValueError("reward_mean must be finite")
        previous = self.get(sample_key)
        self.entries[sample_key] = (
            float(reward_mean) if previous is None
            else self.beta * previous + (1.0 - self.beta) * float(reward_mean)
        )


def rollout_groups(index: Sequence[str], sample_index: Sequence[str]) -> list[tuple[str, list[int]]]:
    """Return (persistent UID, completion positions) in first-RID order.

    A distinct RID identifies each sampled occurrence, including repeated
    occurrences of the same UID. IDs are explicit strings to avoid implicit
    conversions accidentally merging different identifiers.
    """
    if len(index) != len(sample_index):
        raise ValueError("RID and UID lengths must match")
    groups: dict[str, tuple[str, list[int]]] = {}
    for position, (rid, uid) in enumerate(zip(index, sample_index)):
        if not isinstance(rid, str) or not rid.strip() or not isinstance(uid, str) or not uid.strip():
            raise ValueError("RIDs and UIDs must be nonempty strings")
        if rid not in groups:
            groups[rid] = (uid, [])
        if groups[rid][0] != uid:
            raise ValueError(f"Rollout group {rid!r} contains multiple input UIDs")
        groups[rid][1].append(position)
    return list(groups.values())


def _advantage_inputs(rewards: torch.Tensor, response_mask: torch.Tensor, index: Sequence[str], epsilon: float):
    if rewards.ndim != 1 or response_mask.ndim != 2 or rewards.shape[0] != response_mask.shape[0]:
        raise ValueError("rewards must be [batch] and response_mask [batch, tokens]")
    if rewards.numel() == 0 or len(index) != rewards.shape[0]:
        raise ValueError("one rollout ID is required per reward in a nonempty batch")
    if not rewards.is_floating_point() or not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite floating-point values")
    if response_mask.device != rewards.device:
        raise ValueError("rewards and response_mask must share a device")
    if not torch.all((response_mask == 0) | (response_mask == 1)) or not response_mask.bool().any(dim=1).all():
        raise ValueError("response_mask must be binary with at least one retained token per completion")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    dtype = torch.float64 if rewards.dtype == torch.float64 else torch.float32
    return rewards.detach().to(dtype=dtype), response_mask.detach().to(dtype=dtype)


@torch.no_grad()
def compute_shae_advantage(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[str],
    *,
    sample_index: Sequence[str],
    store: ShaeBaselineStore,
    gate_mid: float,
    gate_temperature: float,
    noise_std: float,
    epsilon: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Mix group-centered Std normalization with history-residual RMS.

    Both branches use population moments. RMS bounds historical offsets without
    dividing them by near-zero within-group variance. No jitter is injected:
    first-visit equal rewards carry zero information and yield zero advantage.
    The noise_std argument must be zero.
    """
    scores, mask = _advantage_inputs(rewards, response_mask, index, epsilon)
    groups = rollout_groups(index, sample_index)
    if not isinstance(store, ShaeBaselineStore):
        raise TypeError("store must be a persistent ShaeBaselineStore")
    if not all(math.isfinite(value) for value in (gate_mid, gate_temperature, noise_std)):
        raise ValueError("gate and noise parameters must be finite")
    if gate_temperature <= 0 or noise_std != 0:
        raise ValueError("gate_temperature must be positive and noise_std must be zero")
    outcome = torch.zeros_like(scores)
    for sample_key, positions in groups:
        position_tensor = torch.tensor(positions, device=scores.device, dtype=torch.long)
        sample_scores = scores.index_select(0, position_tensor)
        mean = sample_scores.mean()
        standard_deviation = sample_scores.std(unbiased=False)
        group_advantage = (sample_scores - mean) / (standard_deviation + epsilon)
        previous = store.get(sample_key)
        baseline = float(mean.item()) if previous is None else previous
        ema_raw = sample_scores - baseline
        ema_advantage = ema_raw / (ema_raw.square().mean().sqrt() + epsilon)
        alpha = torch.sigmoid((standard_deviation - gate_mid) / gate_temperature)
        outcome.index_copy_(0, position_tensor, alpha * group_advantage + (1 - alpha) * ema_advantage)
        store.update(sample_key, float(mean.item()))
    return outcome.unsqueeze(-1) * mask


@torch.no_grad()
def compute_grpo_advantage(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[str],
    *,
    epsilon: float,
) -> torch.Tensor:
    """SHAE-off group normalization using population Std (G divisor).

    Each group must contain G > 1 completions.
    This function has no baseline state and performs no EMA or RNG updates.
    """
    scores, mask = _advantage_inputs(rewards, response_mask, index, epsilon)
    groups = rollout_groups(index, index)
    if any(len(positions) < 2 for _, positions in groups):
        raise ValueError("GRPO requires at least two completions per group")
    outcome = torch.zeros_like(scores)
    for _, positions in groups:
        position_tensor = torch.tensor(positions, device=scores.device, dtype=torch.long)
        sample_scores = scores.index_select(0, position_tensor)
        advantage = (sample_scores - sample_scores.mean()) / (sample_scores.std(unbiased=False) + epsilon)
        outcome.index_copy_(0, position_tensor, advantage)
    return outcome.unsqueeze(-1) * mask
