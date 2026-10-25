"""Completion-only NLL and the sequence-likelihood proxy used by RARS."""

from __future__ import annotations

import math

import torch


def completion_likelihood(
    log_probabilities: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[float, float]:
    """Return (-mean(log p), exp(mean(log p))) for one completion.

    Both inputs are one-dimensional and aligned. The binary mask explicitly
    excludes prompt and padding positions; retained positions need not form a
    prefix. Non-finite values at excluded positions do not contribute. The
    likelihood is the geometric mean of the retained token probabilities.
    """
    if log_probabilities.ndim != 1 or log_probabilities.shape != response_mask.shape:
        raise ValueError("log_probabilities and response_mask must have the same [tokens] shape")
    if not log_probabilities.is_floating_point() or log_probabilities.device != response_mask.device:
        raise ValueError("log_probabilities must be floating point on the mask's device")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must be binary")
    values = log_probabilities.detach()[response_mask.bool()].to(dtype=torch.float64)
    if values.numel() == 0 or not torch.isfinite(values).all() or torch.any(values > 0):
        raise ValueError("retained token log probabilities must be finite, nonpositive, and nonempty")
    nll = float((-values.mean()).item())
    return nll, math.exp(-nll)
