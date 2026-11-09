"""Paper reward equations and separately exposed eligible-matching utility."""
import itertools
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from matching.geometry import eligible_giou_assignment, generalized_box_iou
from rewards import RARSParameters, RewardConfig, compute_score


CLASSES = ("HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST")
GT = [[675, 100, 815, 200], [435, 100, 475, 200]]
PRED = [[525, 100, 820, 200], [785, 100, 985, 200]]
EXTRA = [470, 100, 610, 200]
CONFIG = RewardConfig(CLASSES, {a: {b: 1. for b in CLASSES if a != b} for a in CLASSES},
                      RARSParameters(.35, .75, .5, 1e-6), .3, 1., 1., -6., 3., True,
                      missing_match_penalty=.1)


def completion(items):
    return json.dumps({"detections": items})


def triplet(label="HCC", box=(100, 100, 300, 300)):
    return [dict(phase=p, label=label, bbox_2d=list(box)) for p in "apv"]


def score(items, config=CONFIG, truth=None, **kwargs):
    truth = truth if truth is not None else [item for box in GT for item in triplet(box=box)]
    return compute_score(completion(items) if isinstance(items, list) else items,
                         completion(truth), completion_nll=1.,
                         class_counts=dict.fromkeys(CLASSES, 10), config=config, **kwargs)


@pytest.mark.parametrize("rars", [False, True])
def test_reward_assigns_maximum_total_giou_before_thresholding(rars):
    # The max-total assignment chooses two subthreshold pairs here. Screening
    # edges before assignment would instead admit the 0.4746 pair and change Eq.2.
    config = replace(CONFIG, rars_enabled=rars)
    items = [item for box in PRED for item in triplet(box=box)]
    result = score(items, config)
    assert result["classification_matches"] == 0
    assert result["classification_reward"] == 0
    # Localization has its own unthresholded IoU assignment and GT denominator.
    assert result["detection_reward"] == pytest.approx((140 / 295) / 2)
    assert result["missing_match_reward"] == -.1
    assert result["score"] == pytest.approx((140 / 295) / 2 - .1)

    expanded = score(items + triplet(box=EXTRA), config)
    assert expanded["classification_matches"] == 3
    assert expanded["classification_reward"] == 1
    assert expanded["detection_reward"] == pytest.approx((140 / 295 + 5 / 175) / 2)


def test_classification_averages_only_eligible_pairs_and_uses_raw_rars():
    truth = triplet("HCC", GT[0]) + triplet("ICC", GT[1])
    # One of two GT lesions per phase is predicted, so localization is 0.5.
    correct = score(triplet("HCC", GT[0]), truth=truth)
    assert correct["detection_reward"] == .5
    assert correct["classification_reward"] == 1
    assert correct["score"] == 1.5
    wrong = score(triplet("ICC", GT[0]), truth=truth)
    raw_cost = (50 / (10 + 1e-6)) ** .35 * math.exp(.5) * (1 + .75 * math.exp(-1))
    assert wrong["classification_reward"] == pytest.approx(-raw_cost)
    assert wrong["score"] == pytest.approx(.5 - raw_cost)
    symmetric = score(triplet("ICC", GT[0]), replace(CONFIG, rars_enabled=False), truth=truth)
    assert symmetric["classification_reward"] == -1
    assert symmetric["score"] == -.5


def test_unmatched_predictions_add_no_separate_reward_penalty():
    truth = triplet()
    correct = score(truth, truth=truth)
    extra = score(truth + triplet("ICC", (800, 800, 900, 900)), truth=truth)
    assert correct["score"] == extra["score"] == 2
    assert extra["false_positives"] == 3


def test_invalid_empty_background_clipping_and_termination_contract():
    truth = triplet()
    assert score("not json", truth=truth)["score"] == 0
    assert not score("not json", truth=truth)["parse_ok"]
    assert score([], truth=truth)["score"] == -.1
    background = triplet("no-lesion", (-1, -1, -1, -1))
    result = score(background, truth=background)
    assert result["background_correct_phases"] == 3
    assert result["missing_match_reward"] == 0
    assert result["score"] == 1
    # Termination is recorded, while schema validity determines scoring.
    assert score(truth, truth=truth, finish_reason="length")["score"] == 2
    assert score("not json", truth=truth, finish_reason="length")["score"] == 0
    with pytest.raises(ValueError, match="finish_reason"):
        score(truth, truth=truth, finish_reason="unknown")
    clipped = score(truth, replace(CONFIG, classification_weight=4), truth=truth)
    assert clipped["reward_raw"] == 5 and clipped["score"] == 3
    wrong = score(triplet("ICC"), replace(CONFIG, classification_weight=4), truth=truth)
    assert wrong["reward_raw"] < -6 and wrong["score"] == -6


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
