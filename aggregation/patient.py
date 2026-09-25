"""Deterministic patient aggregation for multiple model outputs."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from evaluation.classification import classification_metrics


MAJORITY_NLL_TIEBREAK = "majority_nll_tiebreak"
MAJORITY_FIXED_TIEBREAK = "majority_fixed_tiebreak"
LIKELIHOOD_WEIGHTED_VOTE = "likelihood_weighted_vote"
BEST_LIKELIHOOD_INPUT = "best_likelihood_input"
SUPERMAJORITY_ABSTAIN = "supermajority_abstain"


@dataclass(frozen=True)
class VoteRecord:
    prediction: str | None
    nll: float | None = None
    order: int = 0


@dataclass(frozen=True)
class AggregationResult:
    prediction: str | None
    valid_votes: int
    tied: bool
    abstained: bool


def _finite_nll(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def aggregate_patient(
    records: Iterable[VoteRecord],
    classes: Sequence[str],
    strategy: str = MAJORITY_NLL_TIEBREAK,
) -> AggregationResult:
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Classes must be a non-empty unique sequence")
    class_rank = {label: index for index, label in enumerate(classes)}
    valid = [record for record in records if record.prediction in class_rank]
    if not valid:
        return AggregationResult(None, 0, False, True)
    counts = Counter(record.prediction for record in valid)
    best_count = max(counts.values())
    tied_labels = [label for label in classes if counts[label] == best_count]
    tied = len(tied_labels) > 1

    if strategy == SUPERMAJORITY_ABSTAIN:
        if tied or best_count / len(valid) < 2.0 / 3.0:
            return AggregationResult(None, len(valid), tied, True)
        return AggregationResult(tied_labels[0], len(valid), False, False)

    if strategy == MAJORITY_FIXED_TIEBREAK:
        return AggregationResult(tied_labels[0], len(valid), tied, False)

    if strategy == MAJORITY_NLL_TIEBREAK:
        if not tied:
            return AggregationResult(tied_labels[0], len(valid), False, False)
        mean_nll = {}
        for label in tied_labels:
            values = [
                nll
                for record in valid
                if record.prediction == label and (nll := _finite_nll(record.nll)) is not None
            ]
            mean_nll[label] = sum(values) / len(values) if values else float("inf")
        selected = min(tied_labels, key=lambda label: (mean_nll[label], class_rank[label]))
        return AggregationResult(selected, len(valid), True, False)

    if strategy == LIKELIHOOD_WEIGHTED_VOTE:
        finite = [nll for record in valid if (nll := _finite_nll(record.nll)) is not None]
        fallback = sorted(finite)[len(finite) // 2] if finite else 0.0
        totals = {label: 0.0 for label in classes}
        for record in valid:
            nll = _finite_nll(record.nll)
            numeric = fallback if nll is None else nll
            totals[record.prediction] += math.exp(-max(-50.0, min(50.0, numeric)))
        best = max(totals.values())
        weighted_ties = [label for label in classes if abs(totals[label] - best) <= 1e-12]
        return AggregationResult(weighted_ties[0], len(valid), len(weighted_ties) > 1, False)

    if strategy == BEST_LIKELIHOOD_INPUT:
        selected = min(
            valid,
            key=lambda record: (
                float("inf") if _finite_nll(record.nll) is None else _finite_nll(record.nll),
                record.order,
                class_rank[record.prediction],
            ),
        )
        return AggregationResult(selected.prediction, len(valid), False, False)

    raise ValueError(f"Unknown aggregation strategy: {strategy}")


def aggregate_patient_rows(
    rows: Iterable[dict],
    classes: Sequence[str],
    strategy: str = MAJORITY_NLL_TIEBREAK,
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patient_id"])].append(row)
    details = []
    for patient_id in sorted(grouped):
        patient_rows = grouped[patient_id]
        truths = {str(row["truth"]) for row in patient_rows}
        if len(truths) != 1:
            raise ValueError(f"Inconsistent truth for patient {patient_id}")
        result = aggregate_patient(
            [
                VoteRecord(
                    prediction=row.get("prediction"),
                    nll=row.get("nll"),
                    order=int(row.get("order", index)),
                )
                for index, row in enumerate(patient_rows)
            ],
            classes,
            strategy,
        )
        details.append(
            {
                "patient_id": patient_id,
                "truth": next(iter(truths)),
                "prediction": result.prediction,
                "valid_votes": result.valid_votes,
                "tied": result.tied,
                "abstained": result.abstained,
            }
        )
    truth = [row["truth"] for row in details]
    prediction = [row["prediction"] for row in details]
    covered = [index for index, value in enumerate(prediction) if value is not None]
    overall = classification_metrics(truth, prediction, classes)
    selective_accuracy = (
        sum(truth[index] == prediction[index] for index in covered) / len(covered)
        if covered
        else 0.0
    )
    return {
        "strategy": strategy,
        "patients": len(details),
        "coverage": len(covered) / len(details) if details else 0.0,
        "selective_accuracy": selective_accuracy,
        "metrics": overall,
        "details": details,
    }
