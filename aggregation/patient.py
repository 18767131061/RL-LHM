"""Deterministic lesion-to-group and group-to-patient diagnostic voting."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence
from rewards.schema import ParsedCompletion


@dataclass(frozen=True)
class VoteRecord:
    prediction: str | None
    nll: float | None


@dataclass(frozen=True)
class AggregationResult:
    prediction: str | None
    valid_votes: int
    tied: bool


def _ordered_classes(classes: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(classes)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("classes must be a non-empty sequence of label strings")
    if len(set(labels)) != len(labels):
        raise ValueError("classes must be unique and ordered")
    return labels


def group_prediction(parsed: ParsedCompletion, classes: Sequence[str]) -> str | None:
    """Vote over all active lesion labels across a/p/v, using class-order ties.

    Background records contribute no vote. Invalid completions are represented
    by the caller as undefined group predictions.
    """
    labels = _ordered_classes(classes)
    counts = Counter(item.label for item in parsed.detections if item.label in labels)
    if not counts:
        return None
    return max(labels, key=lambda label: counts[label])


def aggregate_patient(records: Iterable[VoteRecord], classes: Sequence[str]) -> AggregationResult:
    """Majority vote; ties use mean finite completion NLL, then class order.

    Each record represents one input group, not one lesion or phase. Undefined
    predictions do not vote. No eligible votes produces a None prediction.
    """
    labels = _ordered_classes(classes)
    valid = [record for record in records if record.prediction in labels]
    if not valid:
        return AggregationResult(None, 0, False)
    counts = Counter(record.prediction for record in valid)
    maximum = max(counts.values())
    tied = [label for label in labels if counts[label] == maximum]
    if len(tied) == 1:
        return AggregationResult(tied[0], len(valid), False)
    mean_nll = {}
    for label in tied:
        values = [float(record.nll) for record in valid
                  if record.prediction == label and record.nll is not None
                  and math.isfinite(float(record.nll))]
        mean_nll[label] = math.fsum(value / len(values) for value in values) if values else math.inf
    selected = min(tied, key=lambda label: mean_nll[label])
    return AggregationResult(selected, len(valid), True)
