import pytest

from evaluation.loco import FOUR_CLASS_LOCO_CLASSES, evaluate_four_class_loco


def test_four_class_loco_excludes_absent_class_from_macro_but_counts_prediction_error() -> None:
    truth = list(FOUR_CLASS_LOCO_CLASSES)
    predictions = ["HCC", "FNH", "HA_Hemangioma", "nodule_CYST"]
    result = evaluate_four_class_loco(truth, predictions)
    assert result["n"] == 4
    assert result["accuracy"] == 0.75
    assert result["macro_average_denominator"] == 4
    assert result["target_absent_class"] == "FNH"
    assert result["predictions_of_absent_class"] == 1
    assert result["per_class"]["ICC"]["recall"] == 0.0


def test_four_class_loco_rejects_fnh_ground_truth() -> None:
    with pytest.raises(ValueError, match="unexpectedly"):
        evaluate_four_class_loco(["FNH"], ["HCC"])


def test_four_class_loco_rejects_other_unknown_ground_truth() -> None:
    with pytest.raises(ValueError, match="outside"):
        evaluate_four_class_loco(["OTHER"], ["HCC"])
