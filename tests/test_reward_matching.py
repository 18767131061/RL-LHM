"""Eligible GIoU assignment and structured-reward ordering regressions."""
import itertools
from dataclasses import replace

import numpy as np
import pytest

from matching.geometry import eligible_giou_assignment, generalized_box_iou
from rewards import RARSParameters, RewardConfig, compute_score
from tests.test_algorithms import completion, triplet


CLASSES = ("HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST")
GT = [[675, 100, 815, 200], [435, 100, 475, 200]]
PRED = [[525, 100, 820, 200], [785, 100, 985, 200]]
EXTRA = [470, 100, 610, 200]
CONFIG = RewardConfig(CLASSES, {a: {b: 1. for b in CLASSES if a != b} for a in CLASSES},
                      RARSParameters(.35, .75, .5, 1e-6), .3, 1., 1., -2., 2., True)


def score(items, config=CONFIG):
    truth = [item for box in GT for item in triplet(box=box)]
    return compute_score(completion(items), completion(truth), completion_nll=1.,
                         class_counts=dict.fromkeys(CLASSES, 10), config=config)


@pytest.mark.parametrize("rars", [False, True])
def test_pure_ineligible_false_positive_cannot_unlock_reward(rars):
    assert all(generalized_box_iou(gt, EXTRA) < CONFIG.giou_for_classification for gt in GT)
    items = [item for box in PRED for item in triplet(box=box)]
    config = replace(CONFIG, rars_enabled=rars)
    before = score(items, config)
    after = score(items + triplet(box=EXTRA), config)
    assert before["classification_matches"] == after["classification_matches"] == 3
    assert before["missed_lesions"] == after["missed_lesions"] == 3
    assert before["false_positives"] == 3 and after["false_positives"] == 6
    assert before["score"] == pytest.approx(-0.8418079096045198)
    assert after["score"] == pytest.approx(-1.1313559322033897)
    # Pure FPs penalize both components monotonically, with fixed NLL and GT.
    for _ in range(5):
        items += triplet(box=EXTRA)
        current = score(items, config)
        assert current["detection_reward"] < before["detection_reward"]
        assert current["classification_reward"] < before["classification_reward"]
        before = current


@pytest.mark.parametrize("rars", [False, True])
def test_omitting_difficult_lesion_keeps_gt_penalty(rars):
    config = replace(CONFIG, rars_enabled=rars)
    easy = triplet(box=PRED[0])
    hard_wrong = triplet("ICC", GT[1])
    wrong = score(easy + hard_wrong, config)
    omitted = score(easy, config)
    empty = score([], config)
    assert wrong["score"] > omitted["score"] > empty["score"]
    assert omitted["missed_lesions"] == 3 and empty["missed_lesions"] == 6
    assert score(easy + triplet(box=GT[1]), config)["score"] > wrong["score"]


def test_giou_negative_edges_and_isolated_vertex_ties():
    # Negative GIoU is eligible at a negative cutoff, unlike IoU evaluation.
    pairs = eligible_giou_assignment(np.array([[-.2, -.8], [-.9, -.3]]), -.4)
    assert [(p.gt_index, p.prediction_index) for p in pairs] == [(0, 0), (1, 1)]
    base = np.array([[.4, .4], [.4, .4]])
    expected = eligible_giou_assignment(base, .3)
    for position in range(3):
        expanded = np.insert(base, position, -.7, axis=1)
        actual = eligible_giou_assignment(expanded, .3)
        assert [(p.gt_index, p.prediction_index - (p.prediction_index > position)) for p in actual] == [
            (p.gt_index, p.prediction_index) for p in expected]
    assert eligible_giou_assignment(np.full((3, 4), -.8), .3) == []
    for matrix in (np.array([[np.nan]]), np.array([[1.1]]), np.array([.3])):
        with pytest.raises(ValueError, match="GIoU"):
            eligible_giou_assignment(matrix, .3)


def test_giou_assignment_matches_exhaustive_partial_matching():
    rng = np.random.default_rng(123)
    # Enumerate every legal partial matching for small signed, tied matrices.
    for rows, columns, threshold in itertools.product(range(5), range(5), (-1., -.3, .3, 1.)):
        for _ in range(5):
            matrix = rng.choice([-1., -.3, .0, .3, .8, 1.], size=(rows, columns))
            best = (0, 0.)
            for choices in itertools.product(range(-1, columns), repeat=rows):
                used = [c for c in choices if c >= 0]
                if len(set(used)) != len(used):
                    continue
                values = [matrix[r, c] for r, c in enumerate(choices) if c >= 0]
                if any(v < threshold for v in values):
                    continue
                best = max(best, (len(values), sum(values)))
            pairs = eligible_giou_assignment(matrix, threshold)
            assert len(pairs) == best[0]
            assert sum(p.similarity for p in pairs) == pytest.approx(best[1])
            assert len({p.gt_index for p in pairs}) == len(pairs)
            assert len({p.prediction_index for p in pairs}) == len(pairs)
            assert all(p.similarity >= threshold for p in pairs)
