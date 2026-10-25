import pytest

from evaluation.classification import classification_metrics
from evaluation.statistics import (
    benjamini_hochberg,
    exact_mcnemar,
    paired_bootstrap_difference,
    paired_sign_flip_permutation,
)


def test_classification_metrics_match_hand_computation() -> None:
    result = classification_metrics(
        ["A", "A", "B", "B"],
        ["A", "B", "B", None],
        ["A", "B"],
    )
    assert result["accuracy"] == 0.5
    assert result["macro_precision"] == 0.75
    assert result["macro_recall"] == 0.5
    assert result["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.5) / 2.0)
    assert result["per_class"]["A"]["support"] == 2


def test_fixed_class_macro_includes_absent_class_as_zero() -> None:
    result = classification_metrics(["A", "B"], ["A", "B"], ["A", "B", "C"])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(2.0 / 3.0)


def test_unknown_truth_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside"):
        classification_metrics(["C"], ["A"], ["A", "B"])


def test_exact_mcnemar_and_bh_adjustment() -> None:
    test = exact_mcnemar(
        ["A", "A", "B", "B"],
        ["A", "B", "B", "A"],
        ["A", "A", "A", "B"],
    )
    assert test["reference_correct_candidate_wrong"] == 1
    assert test["reference_wrong_candidate_correct"] == 2
    assert test["discordant_pairs"] == 3
    adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.04})


def test_paired_bootstrap_is_seeded_and_reports_candidate_minus_reference() -> None:
    truth = ["A", "A", "B", "B"]
    reference = ["A", "B", "A", "B"]
    candidate = list(truth)

    def accuracy(current_truth, current_prediction):
        return sum(a == b for a, b in zip(current_truth, current_prediction)) / len(current_truth)

    first = paired_bootstrap_difference(
        truth, reference, candidate, accuracy, repeats=100, seed=7
    )
    second = paired_bootstrap_difference(
        truth, reference, candidate, accuracy, repeats=100, seed=7
    )
    assert first == second
    assert first["difference"] == 0.5


def test_paired_sign_flip_permutation_is_seeded() -> None:
    first = paired_sign_flip_permutation([1.0, 1.0, 1.0, 1.0], repeats=2000, seed=42)
    second = paired_sign_flip_permutation([1.0, 1.0, 1.0, 1.0], repeats=2000, seed=42)
    assert first == second
    assert first["mean_difference"] == 1.0
    assert 0.0 < first["two_sided_p"] <= 1.0


def test_paired_sign_flip_permutation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_sign_flip_permutation([], repeats=100)
    with pytest.raises(ValueError, match="finite"):
        paired_sign_flip_permutation([float("nan")], repeats=100)
