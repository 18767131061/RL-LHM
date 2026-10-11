"""Entry-point regressions for the shared training/evaluation JSON contract."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation.schema import SchemaError, parse_completion
from rewards.full_reward import RewardConfig, compute_score as canonical_score
from rewards.verl_liver_reward import (
    CLASSES_PRIVATE,
    canon_label,
    compute_score,
    get_setting,
    parse_completion_from_text,
)


BOX = [0, 0, 1000, 1000]


def record(label="HCC", phase="a", box=None):
    return {"phase": phase, "label": label, "bbox_2d": BOX if box is None else box}


def completion(*records):
    return json.dumps({"detections": list(records)})


def truth(label="HCC"):
    return {"gt_dets": [{"phase": "a", "label": label, "bbox": BOX}]}


@pytest.fixture(autouse=True)
def reward_environment(monkeypatch):
    monkeypatch.setenv("RL_LHM_REWARD_MODE", "rars")
    monkeypatch.setenv("RL_LHM_RISK_MATRIX_MODE", "ordinal")
    monkeypatch.setenv("RL_LHM_RISK_MATRIX_SCALE", "1.0")


@pytest.mark.parametrize(
    "label,canonical",
    [
        ("HCC", "HCC"), (" hcc ", "HCC"), ("ICC", "ICC"), ("FNH", "FNH"),
        ("HA", "HA_Hemangioma"), ("HH", "HA_Hemangioma"),
        ("HA_Hemangioma", "HA_Hemangioma"), ("ha-hemangioma", "HA_Hemangioma"),
        ("NC", "nodule_CYST"), ("nodule_CYST", "nodule_CYST"),
        ("BCLM", "BCLM"), ("CRLM", "CRLM"),
    ],
)
def test_exact_aliases_receive_the_same_diagnosis_and_reward(label, canonical):
    mode = "5cls" if canonical in {"BCLM", "CRLM"} else "private"
    classes = get_setting(mode)
    raw = completion(record(label))
    parsed = parse_completion(raw, classes)
    adapted = parse_completion_from_text(raw, classes)
    scored = compute_score("test", raw, truth(canonical), {"data_mode": mode})
    assert canon_label(label) == canonical
    assert parsed.detections[0].label == adapted["lesions"][0]["label"] == canonical
    assert scored["parse_ok"] == 1
    assert scored["cls_matched_cnt"] == 1
    assert scored["score"] == pytest.approx(4 / 3)


@pytest.mark.parametrize(
    "label",
    [
        "not HCC", "HCC or ICC", "HCC/ICC", "possible HCC", "HCC negative",
        "HCC-like", "non-HCC", "ICC?", "FNH versus HCC", "BCLM and CRLM",
        "not nodule_CYST", "nodule", "cyst", "hemangioma", "unknown", "", None,
    ],
)
def test_non_whitelist_label_invalidates_whole_completion_in_both_entry_points(label):
    # A valid first lesion must not conceal a later invalid/ambiguous label.
    raw = completion(record(), record(label, "p"))
    with pytest.raises(SchemaError):
        parse_completion(raw, CLASSES_PRIVATE)
    assert canon_label(label) is None
    assert parse_completion_from_text(raw, CLASSES_PRIVATE) is None
    scored = compute_score("test", raw, truth(), {"data_mode": "private"})
    assert scored["parse_ok"] == 0
    assert scored["score"] == scored["reward_raw"] == 0.0


@pytest.mark.parametrize(
    "raw,valid",
    [
        (completion(), True),
        (completion(record("HCC", " A ")), True),
        (completion(record(box={"x1": 0, "y1": 0, "x2": 1000, "y2": 1000})), True),
        (completion(record(box={"xmin": 0, "ymin": 0, "xmax": 1000, "ymax": 1000})), True),
        (completion(record(), record("ICC", "a", [10, 10, 20, 20])), True),
        (completion(record(), record("no-lesion", "p", [-1, -1, -1, -1])), True),
        (completion(*(record("background", phase, [-1, -1, -1, -1]) for phase in "apv")), True),
        ("not-json", False),
        ('[]', False),
        ('{"detections": {}}', False),
        (completion("HCC"), False),
        (completion({"phase": "a", "label": "HCC"}), False),
        (completion(record(phase="arterial")), False),
        (completion(record(box=[0, 0, 10])), False),
        (completion(record(box=[0, 0, "bad", 10])), False),
        (completion(record(box=[0, 0, float("nan"), 10])), False),
        (completion(record(box=[0, 0, float("inf"), 10])), False),
        (completion(record(box=[-1, 0, 10, 10])), False),
        (completion(record(box=[0, 0, 1001, 10])), False),
        (completion(record(box=[10, 0, 0, 10])), False),
        (completion(record(box=[0, 0, 0, 10])), False),
        (completion(record(box={"cx": 10, "cy": 10, "w": 4, "h": 4})), False),
        (completion(record("background", "a", BOX)), False),
        (completion(record(), record("background", "a", [-1, -1, -1, -1])), False),
        (completion(record("background", "a", [-1, -1, -1, -1]),
                    record("bg", "a", [-1, -1, -1, -1])), False),
    ],
)
def test_json_phase_background_and_xyxy_contract_match(raw, valid):
    if valid:
        parsed = parse_completion(raw, CLASSES_PRIVATE)
        adapted = parse_completion_from_text(raw, CLASSES_PRIVATE)
        assert adapted["lesions"] == [
            {"phase": d.phase, "label": d.label, "bbox": list(d.box)}
            for d in parsed.detections
        ]
        assert adapted["background_phases"] == list(parsed.background_phases)
    else:
        with pytest.raises(SchemaError):
            parse_completion(raw, CLASSES_PRIVATE)
        assert parse_completion_from_text(raw, CLASSES_PRIVATE) is None
    scored = compute_score("test", raw, truth(), {"data_mode": "private"})
    assert scored["parse_ok"] == int(valid)
    if not valid:
        assert scored["score"] == 0.0


@pytest.mark.parametrize("mode,label", [("private", "BCLM"), ("5cls", "FNH"), ("3cls", "CRLM")])
def test_label_known_in_another_cohort_is_still_invalid(mode, label):
    raw = completion(record(label))
    with pytest.raises(SchemaError):
        parse_completion(raw, get_setting(mode))
    assert parse_completion_from_text(raw, get_setting(mode)) is None
    assert compute_score("test", raw, truth(), {"data_mode": mode})["parse_ok"] == 0


@pytest.mark.parametrize("label", ["HCC", "ICC", "not HCC", "HCC or ICC"])
def test_deployed_reward_and_canonical_reward_agree_on_matching_fixture(label):
    raw = completion(record(label))
    metadata = {"data_mode": "private", "seq_confidence": 0.8}
    deployed = compute_score("test", raw, truth(), dict(metadata))
    canonical = canonical_score(raw, truth(), extra_info=metadata,
                                config=RewardConfig(data_mode="private_5class"))
    assert deployed["parse_ok"] == canonical["parse_ok"]
    assert deployed["score"] == pytest.approx(canonical["score"])
    assert deployed["reward_raw"] == pytest.approx(canonical["reward_raw"])


def test_background_and_empty_completion_keep_distinct_reward_behavior():
    raw = completion(*(record("no-lesion", phase, [-1, -1, -1, -1]) for phase in "apv"))
    background_truth = {"gt_dets": [], "background_phases": ["a", "p", "v"]}
    background = compute_score("test", raw, background_truth, {"data_mode": "private"})
    assert background["parse_ok"] == 1
    assert background["background_correct_phases"] == 3
    assert background["score"] == 1.0
    empty = compute_score("test", completion(), truth(), {"data_mode": "private"})
    assert empty["parse_ok"] == 1
    assert empty["score"] == -0.1


@pytest.mark.parametrize("module_name", ["verl_reward_fixture", "callbacks.reward_fixture"])
def test_reward_callback_loads_by_absolute_path_from_unrelated_working_directory(tmp_path, module_name):
    callback = Path(__file__).resolve().parents[1] / "rewards" / "verl_liver_reward.py"
    script = """
import importlib.util
import json
from pathlib import Path
import sys
assert str(Path(sys.argv[1]).resolve().parents[1]) not in sys.path
spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raw = json.dumps({"detections": [{"phase": "a", "label": "not HCC", "bbox_2d": [0, 0, 10, 10]}]})
print(json.dumps(module.compute_score("test", raw, {"gt_dets": []}, {"data_mode": "private"})))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-E", "-c", script, str(callback), module_name],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    scored = json.loads(completed.stdout)
    assert scored["parse_ok"] == 0
    assert scored["score"] == 0.0
