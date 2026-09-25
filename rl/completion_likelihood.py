"""Generated-token likelihood utilities used by the VeRL reward adapter."""

from __future__ import annotations

import math

import torch


def completion_likelihood(log_probabilities: torch.Tensor, valid_length: int) -> tuple[float, float]:
    values = torch.as_tensor(log_probabilities).reshape(-1)[:valid_length].detach().float()
    if valid_length <= 0 or values.numel() != valid_length or not torch.isfinite(values).all():
        raise ValueError("valid generated-token log probabilities are required")
    nll = float((-values.mean()).item())
    confidence = math.exp(-max(0.0, nll))
    return nll, max(0.0, min(1.0, float(confidence)))


def extract_valid_rollout_log_probabilities(
    responses: torch.Tensor,
    attention_mask: torch.Tensor,
    rollout_log_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    response_ids = torch.as_tensor(responses).squeeze()
    mask = torch.as_tensor(attention_mask).squeeze()
    log_probabilities = torch.as_tensor(rollout_log_probabilities).squeeze()
    if response_ids.ndim != 1 or mask.ndim != 1 or log_probabilities.ndim != 1:
        raise ValueError("responses, mask, and rollout log probabilities must be one-dimensional")
    response_length = int(response_ids.numel())
    if response_length <= 0 or mask.numel() < response_length:
        raise ValueError("attention mask does not contain the response span")
    response_mask = mask[-response_length:].to(dtype=torch.bool)
    valid_ids = response_ids[response_mask]
    if log_probabilities.numel() == response_length:
        valid_log_probabilities = log_probabilities[response_mask]
    elif log_probabilities.numel() == valid_ids.numel():
        valid_log_probabilities = log_probabilities
    else:
        raise ValueError("rollout log-probability length does not match padded or valid response")
    if valid_ids.numel() <= 0:
        raise ValueError("response has no valid generated tokens")
    return valid_ids, valid_log_probabilities
