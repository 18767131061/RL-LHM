import json

import pytest

from rewards.full_reward import RewardConfig, compute_score, generalized_box_iou


def completion(label="HCC", phase="a", box=(0, 0, 10, 10)):
    return json.dumps(
        {"detections": [{"phase": phase, "label": label, "bbox_2d": list(box)}]}
    )


def ground_truth(label="HCC", phase="a", box=(0, 0, 10, 10)):
    return {
        "gt_dets": [{"phase": phase, "label": label, "bbox": list(box)}],
        "background_phases": [],
    }


def test_generalized_iou_matches_hand_cases():
    assert generalized_box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert generalized_box_iou((0, 0, 10, 10), (20, 0, 30, 10)) == pytest.approx(-1 / 3)


def test_perfect_prediction_combines_detection_and_classification_reward():
    result = compute_score(
        completion(),
        ground_truth(),
        extra_info={"class_counts": {"HCC": 10, "ICC": 10}},
    )
    assert result["parse_ok"] == 1
    assert result["detection_reward"] == pytest.approx(1 / 3)
    assert result["classification_reward"] == pytest.approx(1.0)
    assert result["reward_raw"] == pytest.approx(4 / 3)


def test_rars_wrong_class_penalty_increases_with_sequence_confidence():
    config = RewardConfig(data_mode="private_5class", reward_mode="rars")
    metadata = {"class_counts": {"HCC": 100, "ICC": 10}}
    low = compute_score(
        completion(label="HCC"),
        ground_truth(label="ICC"),
        extra_info={**metadata, "sequence_confidence": 0.1},
        config=config,
    )
    high = compute_score(
        completion(label="HCC"),
        ground_truth(label="ICC"),
        extra_info={**metadata, "sequence_confidence": 0.9},
        config=config,
    )
    assert high["reward_raw"] < low["reward_raw"]
    assert high["severity_factor_max"] == pytest.approx(low["severity_factor_max"])


def test_background_and_invalid_completion_contracts():
    truth = {"gt_dets": [], "background_phases": ["a", "p", "v"]}
    background = json.dumps(
        {
            "detections": [
                {"phase": phase, "label": "background", "bbox_2d": [-1, -1, -1, -1]}
                for phase in ("a", "p", "v")
            ]
        }
    )
    result = compute_score(background, truth)
    assert result["parse_ok"] == 1
    assert result["score"] == pytest.approx(1.0)
    assert result["background_correct_phases"] == 3
    assert compute_score("not-json", truth)["parse_ok"] == 0
