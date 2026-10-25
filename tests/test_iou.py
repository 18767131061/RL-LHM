import numpy as np
import pytest

from matching.geometry import box_iou


def test_box_iou_identical_disjoint_and_partial() -> None:
    assert box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert box_iou([0, 0, 1, 1], [1, 1, 2, 2]) == 0.0
    assert box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1.0 / 7.0)


def test_box_iou_zero_area_and_invalid_order() -> None:
    assert box_iou([0, 0, 0, 2], [0, 0, 2, 2]) == 0.0
    with pytest.raises(ValueError, match="ordered"):
        box_iou([2, 0, 1, 2], [0, 0, 2, 2])
