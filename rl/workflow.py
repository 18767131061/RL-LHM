"""Connect clipped rollout rewards to the separate visual curriculum memory."""
from collections.abc import Mapping, Sequence
import torch
from .curriculum import DACVQState
from .core import rollout_groups


@torch.no_grad()
def update_curriculum(
    state: DACVQState,
    clipped_rewards: torch.Tensor,
    rollout_ids: Sequence[str],
    input_uids: Sequence[str],
    visual_vectors: Mapping[str, torch.Tensor | None],
) -> list[int | None]:
    """After the policy step, update memory once per rollout occurrence.

    rewards is [completions]. Each RID denotes one sampled occurrence with one
    persistent UID. Different RIDs sharing a UID stay separate and are processed
    in first-appearance order. An explicit None visual vector skips that group's
    curriculum update. Values are clipped RARS rewards, before SHAE normalization.
    The return value gives the assigned code (or None) for each rollout group.
    """
    if clipped_rewards.ndim != 1 or not clipped_rewards.is_floating_point():
        raise ValueError("clipped_rewards must be a floating-point vector")
    size = len(clipped_rewards)
    if len(rollout_ids) != size or len(input_uids) != size:
        raise ValueError("one rollout ID and input UID are required per completion")
    if not torch.isfinite(clipped_rewards).all():
        raise ValueError("clipped rewards must be finite")
    if torch.any(clipped_rewards < state.reward_min) or torch.any(clipped_rewards > state.reward_max):
        raise ValueError("rewards must lie within the curriculum reward bounds")
    groups = rollout_groups(rollout_ids, input_uids)
    if any(uid not in visual_vectors for uid, _ in groups):
        raise ValueError("each input UID needs an explicit visual vector or None")
    for uid, _ in groups:
        vector = visual_vectors[uid]
        if vector is not None:
            if not isinstance(vector, torch.Tensor) or vector.shape != (state.input_dim,):
                raise ValueError("visual vectors must have shape [input_dim]")
            if not vector.is_floating_point() or not torch.isfinite(vector).all() or not torch.any(vector != 0):
                raise ValueError("available visual vectors must be finite and non-zero")
    codes = []
    for uid, positions in groups:
        vector = visual_vectors[uid]
        if vector is None:
            codes.append(None)
        else:
            mean_reward = clipped_rewards[positions].to(torch.float64).mean().item()
            codes.append(state.update(uid, vector, mean_reward))
    return codes
