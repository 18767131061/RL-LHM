import math

import pytest

from aggregation.patient import (
    BEST_LIKELIHOOD_INPUT,
    LIKELIHOOD_WEIGHTED_VOTE,
    MAJORITY_NLL_TIEBREAK,
    SUPERMAJORITY_ABSTAIN,
    VoteRecord,
    aggregate_patient,
    aggregate_patient_rows,
)


CLASSES = ["A", "B", "C"]


def test_majority_and_nll_tiebreak() -> None:
    majority = aggregate_patient(
        [VoteRecord("A", 2.0), VoteRecord("A", 3.0), VoteRecord("B", 0.1)],
        CLASSES,
    )
    assert majority.prediction == "A"
    tied = aggregate_patient(
        [VoteRecord("A", 2.0), VoteRecord("B", 1.0)],
        CLASSES,
        MAJORITY_NLL_TIEBREAK,
    )
    assert tied.prediction == "B"
    assert tied.tied


def test_nonfinite_nll_is_missing_and_class_order_is_final_tiebreak() -> None:
    result = aggregate_patient(
        [VoteRecord("A", math.nan), VoteRecord("B", math.inf)],
        CLASSES,
        MAJORITY_NLL_TIEBREAK,
    )
    assert result.prediction == "A"


def test_likelihood_vote_best_input_and_supermajority_abstention() -> None:
    weighted = aggregate_patient(
        [VoteRecord("A", 5.0), VoteRecord("A", 5.0), VoteRecord("B", 0.0)],
        CLASSES,
        LIKELIHOOD_WEIGHTED_VOTE,
    )
    assert weighted.prediction == "B"
    best = aggregate_patient(
        [VoteRecord("A", 1.0, 2), VoteRecord("B", 0.5, 1)],
        CLASSES,
        BEST_LIKELIHOOD_INPUT,
    )
    assert best.prediction == "B"
    abstained = aggregate_patient(
        [VoteRecord("A"), VoteRecord("B"), VoteRecord("C")],
        CLASSES,
        SUPERMAJORITY_ABSTAIN,
    )
    assert abstained.prediction is None
    assert abstained.abstained


def test_patient_table_reports_coverage_and_selective_accuracy() -> None:
    rows = [
        {"patient_id": "p1", "truth": "A", "prediction": "A", "order": 0},
        {"patient_id": "p1", "truth": "A", "prediction": "A", "order": 1},
        {"patient_id": "p1", "truth": "A", "prediction": "B", "order": 2},
        {"patient_id": "p2", "truth": "B", "prediction": "A", "order": 0},
        {"patient_id": "p2", "truth": "B", "prediction": "B", "order": 1},
    ]
    result = aggregate_patient_rows(rows, CLASSES, SUPERMAJORITY_ABSTAIN)
    assert result["coverage"] == 0.5
    assert result["selective_accuracy"] == 1.0
    assert result["metrics"]["accuracy"] == 0.5


def test_inconsistent_patient_truth_fails_closed() -> None:
    with pytest.raises(ValueError, match="Inconsistent"):
        aggregate_patient_rows(
            [
                {"patient_id": "p1", "truth": "A", "prediction": "A"},
                {"patient_id": "p1", "truth": "B", "prediction": "B"},
            ],
            CLASSES,
        )


@pytest.mark.parametrize("records, expected, tied", [
    ([VoteRecord("B", 50.)], "B", False),
    ([VoteRecord("A", 50.), VoteRecord("B", 49.)], "B", False),
    ([VoteRecord("B", 50.), VoteRecord("C", 50.)], "B", True),
    ([VoteRecord("B", math.nan), VoteRecord("B", 50.), VoteRecord("A", 50.)], "B", False),
    ([VoteRecord("B", math.inf)], "B", False),
])
def test_weighted_vote_handles_tiny_weights_without_unvoted_or_false_ties(records, expected, tied):
    result = aggregate_patient(records, CLASSES, LIKELIHOOD_WEIGHTED_VOTE)
    assert result.prediction == expected
    assert result.tied is tied
