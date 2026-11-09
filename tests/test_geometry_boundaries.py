"""Geometry thresholds use exact integer oracles, never a relaxed tolerance."""
from fractions import Fraction
import json
import math

import numpy as np
import pytest

from data.manifest import Manifest
from evaluation.detection import evaluate_2d_boxes, evaluate_similarity_matrix
from evaluation.evaluate import evaluate
from matching.geometry import box_iou, generalized_box_iou, eligible_giou_assignment
from matching.protocols import ALL_PROTOCOLS
from rewards import RARSParameters, RewardConfig, compute_score


def exact_overlaps(first, second):
    a, b = [tuple(Fraction(v) for v in box) for box in (first, second)]
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    enclosure = (max(a[2], b[2])-min(a[0], b[0])) * (max(a[3], b[3])-min(a[1], b[1]))
    iou = intersection / union if union else Fraction(0)
    return iou, iou - (enclosure-union)/enclosure if enclosure else iou


def completion(box):
    return json.dumps({'detections': [dict(phase=p, label='HCC', bbox_2d=box) for p in 'apv']})


@pytest.mark.parametrize('threshold', [.25, .5, .75])
def test_exact_detection_boundary_survives_translation(threshold):
    for width in range(40, 401, 40):
        for offset in range(0, 601, 25):
            gt = [offset, 100, offset+width, 300]
            pred = [offset, 100, offset+int(width*threshold), 300]
            expected, _ = exact_overlaps(gt, pred)
            assert expected == Fraction(threshold)
            assert box_iou(gt, pred) == threshold
            for protocol in ALL_PROTOCOLS:
                result = evaluate_2d_boxes([gt], [pred], ['HCC'], ['HCC'], protocol, threshold)
                assert (result.tp, result.fp, result.fn) == (1, 0, 0)


def test_full_evaluator_and_reward_translation_regressions():
    config = RewardConfig(('HCC', 'ICC'), {'HCC': {'ICC': 1.1}, 'ICC': {'HCC': 1.1}},
                          RARSParameters(.35, .75, .5, 1e-6), .3, 1., 1., -2., 2., True)
    for offset in (0, 50, 100, 125, 500):
        gt = [offset, offset, offset+200, offset+200]
        pred = [offset, offset, offset+100, offset+200]
        manifest = Manifest(config.classes, {'p': 'HCC'},
                            [dict(uid='u', patient_id='p', ground_truth=completion(gt))])
        result = evaluate(manifest, [dict(uid='u', completion=completion(pred), nll=1.)],
                          iou_thresholds=[.5])
        for method in result['detection_by_matching'].values():
            for rows in method.values():
                assert (rows[0]['tp'], rows[0]['fp'], rows[0]['fn']) == (3, 0, 0)
        pred[2] = offset+60
        assert generalized_box_iou(gt, pred) == .3
        score = compute_score(completion(pred), completion(gt), completion_nll=1.,
                              class_counts=dict.fromkeys(config.classes, 10), config=config)
        assert score['classification_matches'] == 3
        assert score['score'] == 1.3


@pytest.mark.parametrize('threshold', [.25, .3, .5, .75])
def test_immediate_float_neighbors_are_not_promoted_to_threshold(threshold):
    gt = [0., 0., 1., 1.]
    for value, accepted in ((math.nextafter(threshold, 0.), False),
                            (threshold, True), (math.nextafter(threshold, 1.), True)):
        pred = [0., 0., value, 1.]
        assert box_iou(gt, pred) == value
        assert generalized_box_iou(gt, pred) == value
        for protocol in ALL_PROTOCOLS:
            result = evaluate_2d_boxes([gt], [pred], ['HCC'], ['HCC'], protocol, threshold)
            assert result.tp == int(accepted)
        assert bool(eligible_giou_assignment(np.array([[value]]), threshold)) == accepted


def test_signed_and_zero_thresholds_keep_their_exact_rules():
    for value, accepted in ((math.nextafter(-.3, -1.), False), (-.3, True),
                            (math.nextafter(-.3, 0.), True)):
        assert bool(eligible_giou_assignment(np.array([[value]]), -.3)) == accepted
    for protocol in ALL_PROTOCOLS:
        for overlap, expected in ((0., 0), (math.nextafter(0., 1.), 1)):
            result = evaluate_similarity_matrix(np.array([[overlap]]), ['HCC'], ['HCC'], protocol, 0.)
            assert result.tp == expected


def test_integer_boxes_match_rational_iou_and_giou_oracles():
    # Neither box contains the other: I=72, U=180, C=200 gives GIoU=3/10.
    for offset in (0, 50, 500):
        first = [20+offset, 20+offset, 30+offset, 30+offset]
        second = [10+offset, 20+offset, 29+offset, 28+offset]
        assert exact_overlaps(first, second)[1] == Fraction(3, 10)
        assert generalized_box_iou(first, second) == .3
    rng = np.random.default_rng(20261030)
    for _ in range(1000):
        boxes = []
        for _ in range(2):
            xs = sorted(map(int, rng.integers(0, 1001, 2)))
            ys = sorted(map(int, rng.integers(0, 1001, 2)))
            boxes.append([xs[0], ys[0], xs[1], ys[1]])
        iou, giou = exact_overlaps(*boxes)
        assert box_iou(*boxes) == float(iou)
        assert generalized_box_iou(*boxes) == float(giou)


def test_extreme_scales_and_thin_boxes_remain_finite():
    for exponent in (-1070, -1000, -500, 0, 500, 1000):
        for first, second in (([0, 0, 8, 8], [0, 0, 4, 8]),
                              ([-8, -8, 0, 0], [0, 0, 8, 8])):
            boxes = [[math.ldexp(v, exponent) for v in box] for box in (first, second)]
            iou, giou = exact_overlaps(*boxes)
            assert box_iou(*boxes) == float(iou)
            assert generalized_box_iou(*boxes) == float(giou)
    largest = math.nextafter(math.inf, 0.)
    first, second = [-largest, -largest, largest, largest], [0., -largest, largest, largest]
    assert box_iou(first, second) == .5
    assert generalized_box_iou(first, second) == .5
    smallest = math.nextafter(0., 1.)
    point_box = [0., 0., smallest, smallest]
    assert box_iou(point_box, point_box) == generalized_box_iou(point_box, point_box) == 1.
    # Union*enclosure is subnormal although each area is representable.
    first, second = [0., 0., 1., 1e-320], [0., 0., 1e-320, 1.]
    assert box_iou(first, second) == 0.
    assert generalized_box_iou(first, second) == -1.
