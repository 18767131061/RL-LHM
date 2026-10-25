"""Framework-independent masked token GRPO surrogate and reference KL loss."""

from __future__ import annotations

import math

import torch


def grpo_token_loss(
    current_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    reference_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_ratio: float,
    dual_clip: float,
    kl_coefficient: float,
    log_ratio_bound: float,
    kl_bound: float,
    total_token_count: int | None,
) -> torch.Tensor:
    """Return the scalar minimization loss -J in the paper's token objective.

    Log-probabilities and the binary mask have shape [batch, tokens]. Advantages
    may be [batch] (explicitly broadcast) or [batch, tokens]. Rollout/reference
    log-probabilities and advantages are detached; only current-policy values
    receive gradients. Prompt/padding values are excluded before arithmetic.

    Pass ``total_token_count=None`` for a complete update batch. For additive
    microbatch contributions, pass the same full-update retained-token count to
    every call and sum their losses. A distributed caller must account for its
    framework's gradient averaging; this function performs no collectives.
    All clipping bounds and coefficients are explicit mathematical inputs.
    """
    current = current_log_probabilities
    if current.ndim != 2 or current.shape != response_mask.shape:
        raise ValueError("current log probabilities and response_mask must be [batch, tokens]")
    if any(value.shape != current.shape for value in (old_log_probabilities, reference_log_probabilities)):
        raise ValueError("all token log probabilities must have identical shapes")
    tensors = (current, old_log_probabilities, reference_log_probabilities, advantages)
    if any(not value.is_floating_point() for value in tensors):
        raise ValueError("log probabilities and advantages must be floating point")
    if any(value.device != current.device for value in (*tensors, response_mask)):
        raise ValueError("all tensors must share a device")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must be binary")
    mask = response_mask.bool()
    retained_count = int(mask.sum().item())
    if retained_count <= 0:
        raise ValueError("at least one generated token must be retained")
    if total_token_count is None:
        total_token_count = retained_count
    if type(total_token_count) is not int or total_token_count < retained_count:
        raise ValueError("total_token_count must be an integer at least as large as the retained-token count")
    if advantages.shape == (current.shape[0],):
        advantages = advantages[:, None].expand_as(current)
    elif advantages.shape != current.shape:
        raise ValueError("advantages must be [batch] or [batch, tokens]")
    for value in (current, old_log_probabilities, reference_log_probabilities):
        retained = value[mask]
        if not torch.isfinite(retained).all() or torch.any(retained > 0):
            raise ValueError("retained token log probabilities must be finite and nonpositive")
    if not torch.isfinite(advantages[mask]).all():
        raise ValueError("retained advantages must be finite")
    parameters = (clip_ratio, dual_clip, kl_coefficient, log_ratio_bound, kl_bound)
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("objective parameters must be finite")
    if not 0 <= clip_ratio < 1 or dual_clip <= 1 or kl_coefficient < 0 or log_ratio_bound <= 0 or kl_bound <= 0:
        raise ValueError("invalid clipping bound or KL coefficient")
    dtype = torch.float64 if current.dtype == torch.float64 else torch.float32
    if log_ratio_bound >= math.log(torch.finfo(dtype).max):
        raise ValueError("log_ratio_bound is too large for the objective's floating-point dtype")
    current = torch.where(mask, current.to(dtype=dtype), 0)
    old = torch.where(mask, old_log_probabilities.detach().to(dtype=dtype), 0)
    reference = torch.where(mask, reference_log_probabilities.detach().to(dtype=dtype), 0)
    advantage = torch.where(mask, advantages.detach().to(dtype=dtype), 0)
    ratio = torch.exp((current - old).clamp(-log_ratio_bound, log_ratio_bound))
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
    surrogate = torch.where(advantage < 0, torch.maximum(surrogate, dual_clip * advantage), surrogate)
    reference_difference = (reference - current).clamp(-log_ratio_bound, log_ratio_bound)
    # expm1(d)-d is exp(d)-d-1, with improved accuracy close to d=0.
    kl = (torch.expm1(reference_difference) - reference_difference).clamp(-kl_bound, kl_bound)
    token_loss = -surrogate + kl_coefficient * kl
    return token_loss[mask].sum() / total_token_count
