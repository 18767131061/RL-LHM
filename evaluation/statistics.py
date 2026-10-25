"""Paired patient-level uncertainty and multiple-testing utilities."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest


def exact_mcnemar(
    truth: Sequence[str],
    reference: Sequence[str | None],
    candidate: Sequence[str | None],
) -> dict[str, int | float]:
    if not (len(truth) == len(reference) == len(candidate)):
        raise ValueError("Paired sequences must have identical lengths")
    reference_only = sum(a == b and a != c for a, b, c in zip(truth, reference, candidate))
    candidate_only = sum(a != b and a == c for a, b, c in zip(truth, reference, candidate))
    discordant = reference_only + candidate_only
    p_value = float(binomtest(candidate_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "reference_correct_candidate_wrong": reference_only,
        "reference_wrong_candidate_correct": candidate_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def paired_bootstrap_difference(
    truth: Sequence[str],
    reference: Sequence[str | None],
    candidate: Sequence[str | None],
    metric: Callable[[Sequence[str], Sequence[str | None]], float],
    repeats: int = 10000,
    seed: int = 2026,
) -> dict[str, float | int]:
    if repeats <= 0 or not (len(truth) == len(reference) == len(candidate)) or not truth:
        raise ValueError("Paired bootstrap needs non-empty equal sequences and positive repeats")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(truth), len(truth))
        sampled_truth = [truth[int(index)] for index in indices]
        sampled_reference = [reference[int(index)] for index in indices]
        sampled_candidate = [candidate[int(index)] for index in indices]
        values.append(metric(sampled_truth, sampled_candidate) - metric(sampled_truth, sampled_reference))
    lower, upper = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    point = metric(truth, candidate) - metric(truth, reference)
    return {
        "difference": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "repeats": repeats,
        "seed": seed,
    }


def paired_sign_flip_permutation(
    paired_differences: Sequence[float],
    repeats: int = 10000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Two-sided Monte Carlo sign-flip test for paired patient contributions."""

    values = np.asarray(paired_differences, dtype=float)
    if repeats <= 0 or values.size == 0:
        raise ValueError("Sign-flip permutation needs non-empty differences and positive repeats")
    if not np.all(np.isfinite(values)):
        raise ValueError("Paired differences must be finite")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(repeats):
        signs = rng.choice((-1.0, 1.0), size=values.size)
        if abs(float(np.mean(values * signs))) >= observed - 1e-15:
            extreme += 1
    return {
        "mean_difference": float(np.mean(values)),
        "two_sided_p": float((extreme + 1) / (repeats + 1)),
        "repeats": repeats,
        "seed": seed,
    }


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    if any(not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("P-values must be in [0, 1]")
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 1.0
    for reverse_rank, key in enumerate(reversed(ordered), 1):
        rank = len(ordered) - reverse_rank + 1
        running = min(running, p_values[key] * len(ordered) / rank)
        adjusted[key] = min(1.0, running)
    return adjusted
