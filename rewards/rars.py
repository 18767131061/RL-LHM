"""RARS scarcity, severity, and completion-likelihood factors.

All mathematical parameters, class counts, and severity weights come from the
caller. Counts follow the paper's training-input-group metadata convention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RARSParameters:
    scarcity_alpha: float
    confidence_eta: float
    severity_kappa: float
    count_epsilon: float

    def __post_init__(self) -> None:
        for name in ("scarcity_alpha", "confidence_eta", "severity_kappa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.count_epsilon) or self.count_epsilon <= 0:
            raise ValueError("count_epsilon must be finite and positive")


def validate_risk_matrix(
    matrix: Mapping[str, Mapping[str, float]], classes: Sequence[str]
) -> None:
    """Check the complete off-diagonal prior (rows=true, columns=predicted)."""
    labels = set(classes)
    if set(matrix) != labels:
        raise ValueError("Severity matrix rows must equal the active classes")
    for truth in classes:
        if set(matrix[truth]) != labels - {truth}:
            raise ValueError(f"Severity row {truth!r} must include all other classes")
        for prediction, weight in matrix[truth].items():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Severity weights must be finite and non-negative")
            if matrix[prediction].get(truth) != weight:
                raise ValueError("The paper's severity prior uses symmetric pair weights")


def normalize_class_counts(
    class_counts: Mapping[str, int], classes: Sequence[str]
) -> dict[str, int]:
    """Apply the appendix's lower bound while retaining every supplied key.

    Missing/zero active entries become one. Extra keys remain in the total;
    these metadata counts are not patient counts. No dataset is inferred.
    """
    if not isinstance(class_counts, Mapping):
        raise ValueError("class_counts must be an explicit count dictionary")
    try:
        counts = {key: int(value) for key, value in class_counts.items()}
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("class_counts values must be convertible to finite integers") from error
    for label in classes:
        counts[label] = max(1, counts.get(label, 1))
    return counts


def completion_likelihood(completion_nll: float) -> float:
    """Return exp(-max(0, NLL)) from the generated-token-only mean NLL."""
    if not math.isfinite(completion_nll):
        raise ValueError("completion_nll must be finite")
    return math.exp(-max(0.0, completion_nll))


def build_rars_confusion_cost(
    predicted_label: str,
    true_label: str,
    confidence: float,
    class_counts: Mapping[str, int],
    matrix: Mapping[str, Mapping[str, float]],
    parameters: RARSParameters,
) -> tuple[float, float]:
    """Return (confusion cost, severity factor); correct labels have zero cost.

    ``confidence`` is one completion likelihood, shared by all its lesion
    pairs. ``class_counts`` must already have passed normalize_class_counts.
    """
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a finite completion likelihood in [0,1]")
    if true_label not in matrix or predicted_label not in matrix:
        raise ValueError("Both labels must belong to the severity matrix")
    if predicted_label == true_label:
        return 0.0, 1.0
    if predicted_label not in matrix[true_label]:
        raise ValueError("A severity weight is missing for the matched confusion")
    total = max(1, sum(int(value) for value in class_counts.values()))
    true_count = max(1, int(class_counts.get(true_label, 1)))
    weight = float(matrix[true_label][predicted_label])
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Severity weights must be finite and non-negative")
    try:
        scarcity = (total / (true_count + parameters.count_epsilon)) ** parameters.scarcity_alpha
        severity = math.exp(parameters.severity_kappa * weight)
        cost = scarcity * severity * (1.0 + parameters.confidence_eta * confidence)
    except OverflowError as error:
        raise ValueError("The supplied RARS parameters overflow the confusion cost") from error
    if not math.isfinite(cost):
        raise ValueError("The supplied RARS parameters produce a non-finite confusion cost")
    return float(cost), float(severity)
