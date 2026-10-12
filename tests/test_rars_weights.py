import copy
import json
import math

import pytest

from rewards.rars import (
    PRIVATE_CLASSES,
    PUBLIC_CLASSES,
    build_rars_confusion_cost,
    default_matrix,
    freeze_risk_selection,
    generate_risk_grid,
    uniform_matrix,
    validate_risk_freeze,
)


def test_correct_and_vanilla_confusion_costs() -> None:
    matrix = default_matrix("private")
    counts = {label: 10 for label in PRIVATE_CLASSES}
    assert build_rars_confusion_cost("HCC", "HCC", 0.8, counts, matrix) == (0.0, 1.0)
    assert build_rars_confusion_cost(
        "HCC", "ICC", 0.8, counts, matrix, reward_mode="vanilla"
    ) == (1.0, 1.0)


def test_rarer_true_class_and_higher_severity_increase_cost() -> None:
    matrix = default_matrix("private")
    balanced = {label: 20 for label in PRIVATE_CLASSES}
    rare_icc = dict(balanced, ICC=2)
    common_cost, _ = build_rars_confusion_cost("HCC", "ICC", 0.5, balanced, matrix)
    rare_cost, high_severity = build_rars_confusion_cost("HCC", "ICC", 0.5, rare_icc, matrix)
    low_severity_cost, low_severity = build_rars_confusion_cost(
        "nodule_CYST", "HCC", 0.5, balanced, matrix
    )
    assert rare_cost > common_cost
    assert high_severity > low_severity
    assert common_cost > low_severity_cost


def test_default_matrices_match_frozen_selected_profiles() -> None:
    private = default_matrix("private")
    assert private["HCC"]["ICC"] == private["ICC"]["HCC"] == 1.10
    assert private["HCC"]["FNH"] == private["FNH"]["HCC"] == 0.90
    assert private["HCC"]["HA_Hemangioma"] == 0.50
    assert private["ICC"]["HA_Hemangioma"] == 0.70
    assert private["FNH"]["nodule_CYST"] == 0.40

    public = default_matrix("public")
    assert public["HCC"]["ICC"] == 1.10
    assert public["HCC"]["HA_Hemangioma"] == 0.60
    assert public["ICC"]["HA_Hemangioma"] == 0.80
    assert public["ICC"]["BCLM"] == 1.00
    assert public["ICC"]["CRLM"] == 1.10
    assert public["BCLM"]["CRLM"] == 0.90

    assert all(value == 0.4 for row in uniform_matrix("private").values() for value in row.values())
    assert all(value == 0.4 for row in uniform_matrix("public").values() for value in row.values())
    assert set(public) == set(PUBLIC_CLASSES)


def test_risk_grid_is_cartesian_and_deterministic() -> None:
    parameters = {"HCC->ICC": [0.5, 1.0], "HCC->FNH": [1.0, 1.5]}
    first = generate_risk_grid(PRIVATE_CLASSES, parameters, [0.1, 0.2])
    second = generate_risk_grid(PRIVATE_CLASSES, parameters, [0.1, 0.2])
    assert len(first) == 8
    assert first == second
    assert len({row["profile_id"] for row in first}) == 8


def test_conflicting_directional_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="unordered class pair HCC<->ICC"):
        generate_risk_grid(
            PRIVATE_CLASSES,
            {"HCC->ICC": [0.5], "ICC->HCC": [1.0]},
            [0.4],
        )


def test_grid_keeps_non_scanned_selected_weights_fixed() -> None:
    profiles = generate_risk_grid(
        PRIVATE_CLASSES,
        {"HCC->ICC": [0.1, 0.2], "HCC->FNH": [0.1, 0.2]},
        [0.4],
        fixed_pair_weights={"HCC->HA_Hemangioma": 0.5, "ICC->HA_Hemangioma": 0.7},
    )
    assert len(profiles) == 4
    for profile in profiles:
        matrix = profile["matrix"]
        assert matrix["HCC"]["HA_Hemangioma"] == 0.5
        assert matrix["HA_Hemangioma"]["HCC"] == 0.5
        assert matrix["ICC"]["HA_Hemangioma"] == 0.7
        assert matrix["FNH"]["nodule_CYST"] == 0.4


def test_risk_freeze_hashes_and_rejects_tampering(tmp_path) -> None:
    profile = generate_risk_grid(
        PRIVATE_CLASSES,
        {"HCC->ICC": [0.5], "HCC->FNH": [1.0]},
        [0.1],
    )[0]
    path = tmp_path / "risk_freeze.json"
    freeze = freeze_risk_selection(profile, "a" * 64, path)
    validate_risk_freeze(freeze)
    assert path.is_file()
    tampered = copy.deepcopy(freeze)
    tampered["selected_profile"]["shared_other"] = 9.0
    with pytest.raises(ValueError, match="profile hash"):
        validate_risk_freeze(tampered)


@pytest.mark.parametrize("cohort,data_mode,classes", [
    ("private", "private", PRIVATE_CLASSES), ("public", "5cls", PUBLIC_CLASSES),
])
def test_uniform_control_actual_reward_uses_point_four(cohort, data_mode, classes, monkeypatch):
    from rewards.verl_liver_reward import compute_score
    monkeypatch.setenv("RL_LHM_REWARD_MODE", "rars")
    monkeypatch.setenv("RL_LHM_RISK_MATRIX_MODE", "uniform")
    monkeypatch.setenv("RL_LHM_RISK_MATRIX_SCALE", "1.0")
    counts = {label: 10 for label in classes}
    truth = {"gt_dets": [{"phase": phase, "label": "ICC", "bbox": [0,0,10,10]}
                         for phase in ("a", "p", "v")]}
    completion = json.dumps({"detections": [{"phase": phase, "label": "HCC", "bbox_2d": [0,0,10,10]}
                                           for phase in ("a", "p", "v")]})
    active = compute_score(None, completion, truth,
                           {"data_mode": data_mode, "class_counts": counts, "seq_confidence": .5})
    penalty, multiplier = build_rars_confusion_cost("HCC", "ICC", .5, counts,
                                                    default_matrix(cohort), matrix_mode="uniform")
    assert multiplier == pytest.approx(math.exp(.5 * .4))
    assert active["severity_factor_max"] == pytest.approx(multiplier)
    assert active["cls_score"] == pytest.approx(-penalty)
    monkeypatch.setenv("RL_LHM_REWARD_MODE", "vanilla")
    assert compute_score(None, completion, truth,
                         {"data_mode": data_mode, "class_counts": counts, "seq_confidence": .5})["cls_score"] == -1.


@pytest.mark.parametrize('outside_count', [17, -1])
def test_class_count_metadata_is_copied_and_all_keys_contribute(outside_count, monkeypatch):
    from rewards.verl_liver_reward import compute_score as active_reward
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    monkeypatch.setenv('RL_LHM_REWARD_MODE', 'rars')
    monkeypatch.setenv('RL_LHM_RISK_MATRIX_MODE', 'ordinal')
    monkeypatch.setenv('RL_LHM_RISK_MATRIX_SCALE', '1.0')
    monkeypatch.delenv('RL_LHM_RISK_PROFILE', raising=False)
    metadata = {'data_mode': 'private', 'class_counts': {'HCC': 10, 'ICC': 2, 'outside': outside_count},
                'seq_confidence': .5}
    original = copy.deepcopy(metadata)
    truth = {'gt_dets': [{'phase': phase, 'label': 'ICC', 'bbox': [0,0,10,10]} for phase in ('a','p','v')]}
    completion = json.dumps({'detections': [{'phase': phase, 'label': 'HCC', 'bbox_2d': [0,0,10,10]}
                                           for phase in ('a','p','v')]})
    active = active_reward(None, completion, truth, metadata)
    reference = reference_reward(completion, truth, extra_info=metadata,
                                 config=RewardConfig(data_mode='private_5class'))
    # Three missing active classes each get count one; extra keys stay in the sum.
    expected = -((15 + outside_count) / (2 + 1e-6)) ** .35 * math.exp(.5 * 1.1) * (1 + .75 * .5)
    assert active['cls_score'] == pytest.approx(expected)
    assert reference['classification_reward'] == pytest.approx(expected)
    assert reference['reward_raw'] == pytest.approx(active['reward_raw'])
    assert metadata == original


def _loco_reward_setup(tmp_path, monkeypatch, classes=None):
    from training.rars_joint_grid import candidate
    profile = candidate({'cohort': 'private', 'protocol': 'loco',
                         'classes': list(classes or ('HCC', 'ICC', 'HA_Hemangioma', 'nodule_CYST')),
                         'loco_pairs': [['HCC', 'ICC']], 'values': [.1]}, 0)
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(profile))
    monkeypatch.setenv('RL_LHM_RISK_PROFILE', str(path))
    monkeypatch.setenv('RL_LHM_REWARD_MODE', 'rars')
    monkeypatch.setenv('RL_LHM_RISK_MATRIX_MODE', 'ordinal')
    monkeypatch.setenv('RL_LHM_RISK_MATRIX_SCALE', '1.0')
    return profile, path


def _loco_reward_inputs(truth='ICC', prediction='HCC'):
    gt = {'gt_dets': [{'phase': phase, 'label': truth, 'bbox': [0, 0, 10, 10]}
                      for phase in ('a', 'p', 'v')]}
    raw = json.dumps({'detections': [{'phase': phase, 'label': prediction, 'bbox_2d': [0, 0, 10, 10]}
                                    for phase in ('a', 'p', 'v')]})
    return gt, raw


@pytest.mark.parametrize('extra_count', [None, 0, 7])
def test_loco_four_class_reward_does_not_invent_fnh_count(tmp_path, monkeypatch, extra_count):
    from rewards.verl_liver_reward import compute_score as active_reward, get_setting
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    profile, _ = _loco_reward_setup(tmp_path, monkeypatch)
    counts = {label: 10 for label in profile['active_classes']}
    if extra_count is not None:
        counts['FNH'] = extra_count
    metadata = {'data_mode': 'private', 'class_counts': counts, 'seq_confidence': .5}
    before = copy.deepcopy(metadata)
    gt, raw = _loco_reward_inputs()
    active = active_reward(None, raw, gt, metadata)
    reference = reference_reward(raw, gt, extra_info=metadata,
        config=RewardConfig(data_mode='private_5class', risk_profile=profile))
    expected = -((40 + (extra_count or 0)) / (10 + 1e-6)) ** .35 * math.exp(.5 * .1) * 1.375
    assert get_setting('private') == profile['active_classes']
    assert active['parse_ok'] == reference['parse_ok'] == 1
    assert active['cls_score'] == pytest.approx(expected)
    assert reference['classification_reward'] == pytest.approx(expected)
    assert reference['reward_raw'] == pytest.approx(active['reward_raw'])
    assert metadata == before


def test_loco_counts_still_protect_missing_active_classes(tmp_path, monkeypatch):
    from rewards.verl_liver_reward import compute_score as active_reward
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    profile, _ = _loco_reward_setup(tmp_path, monkeypatch)
    metadata = {'data_mode': 'private', 'class_counts': {'HCC': 10, 'ICC': 2, 'nodule_CYST': 0, 'extra': 17}}
    gt, raw = _loco_reward_inputs()
    active = active_reward(None, raw, gt, metadata)
    reference = reference_reward(raw, gt, extra_info=metadata,
        config=RewardConfig(data_mode='private_5class', risk_profile=profile))
    # The missing HH and zero NC each become one; no absent FNH entry is inserted.
    expected = -(31 / (2 + 1e-6)) ** .35 * math.exp(.5 * .1)
    assert active['cls_score'] == pytest.approx(expected)
    assert reference['classification_reward'] == pytest.approx(expected)


@pytest.mark.parametrize('side', ['prediction', 'truth'])
def test_loco_schema_rejects_fnh_outside_active_classes(tmp_path, monkeypatch, side):
    from rewards.verl_liver_reward import compute_score as active_reward
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    profile, _ = _loco_reward_setup(tmp_path, monkeypatch)
    gt, raw = _loco_reward_inputs(**{side: 'FNH'})
    metadata = {'data_mode': 'private', 'class_counts': {c: 10 for c in profile['active_classes']}}
    assert active_reward(None, raw, gt, metadata)['parse_ok'] == 0
    assert reference_reward(raw, gt, extra_info=metadata,
        config=RewardConfig(data_mode='private_5class', risk_profile=profile))['parse_ok'] == 0


@pytest.mark.parametrize('fault', ['cohort', 'empty', 'duplicate', 'outside', 'unordered', 'not_mapping'])
def test_loco_invalid_profile_fails_before_correct_label_reward(tmp_path, monkeypatch, fault):
    from rewards.verl_liver_reward import compute_score as active_reward
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    profile, path = _loco_reward_setup(tmp_path, monkeypatch)
    if fault == 'cohort':
        profile['cohort'] = 'public'
        profile['active_classes'] = ['HCC', 'ICC', 'HA_Hemangioma']
    elif fault == 'empty': profile['active_classes'] = []
    elif fault == 'duplicate': profile['active_classes'].append('HCC')
    elif fault == 'outside': profile['active_classes'].append('BCLM')
    elif fault == 'unordered': profile['active_classes'] = {c: True for c in profile['active_classes']}
    elif fault == 'not_mapping': profile = []
    path.write_text(json.dumps(profile))
    gt, raw = _loco_reward_inputs(truth='HCC')
    with pytest.raises(RuntimeError, match='risk profile'):
        active_reward(None, raw, gt, {'data_mode': 'private'})
    with pytest.raises(RuntimeError, match='risk profile'):
        reference_reward(raw, gt, config=RewardConfig(data_mode='private_5class', risk_profile=profile))


def test_five_class_loco_profile_keeps_fnh_active(tmp_path, monkeypatch):
    from rewards.verl_liver_reward import compute_score as active_reward, get_setting
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    profile, _ = _loco_reward_setup(tmp_path, monkeypatch, PRIVATE_CLASSES)
    gt, raw = _loco_reward_inputs(truth='FNH', prediction='FNH')
    assert get_setting('private') == list(PRIVATE_CLASSES)
    assert active_reward(None, raw, gt, {'data_mode': 'private'})['cls_score'] == 1.
    assert reference_reward(raw, gt,
        config=RewardConfig(data_mode='private_5class', risk_profile=profile))['classification_reward'] == 1.


def test_loco_profile_must_fit_the_selected_data_mode(tmp_path, monkeypatch):
    from training.rars_joint_grid import candidate
    from rewards.verl_liver_reward import compute_score as active_reward, get_setting
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    profile = candidate({'cohort': 'public', 'protocol': 'loco',
                         'classes': ['HCC', 'ICC', 'HA_Hemangioma', 'BCLM'],
                         'loco_pairs': [['HCC', 'ICC']], 'values': [.1]}, 0)
    path = tmp_path / 'public_profile.json'
    path.write_text(json.dumps(profile))
    monkeypatch.setenv('RL_LHM_RISK_PROFILE', str(path))
    gt, raw = _loco_reward_inputs(truth='HCC')
    assert get_setting('5cls') == profile['active_classes']
    with pytest.raises(RuntimeError, match='data_mode vocabulary'):
        active_reward(None, raw, gt, {'data_mode': '3cls'})
    with pytest.raises(RuntimeError, match='data_mode vocabulary'):
        reference_reward(raw, gt, config=RewardConfig(data_mode='public_3class', risk_profile=profile))


@pytest.mark.parametrize('mode,helper_mode,classes', [
    ('private', 'private_5class', PRIVATE_CLASSES),
    ('5cls', 'public_5class', PUBLIC_CLASSES),
    ('3cls', 'public_3class', ('HCC', 'ICC', 'HA_Hemangioma')),
])
def test_no_profile_reward_vocabularies_and_penalties_unchanged(monkeypatch, mode, helper_mode, classes):
    from rewards.verl_liver_reward import compute_score as active_reward, get_setting
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    monkeypatch.delenv('RL_LHM_RISK_PROFILE', raising=False)
    monkeypatch.setenv('RL_LHM_REWARD_MODE', 'rars')
    monkeypatch.setenv('RL_LHM_RISK_MATRIX_MODE', 'ordinal')
    monkeypatch.setenv('RL_LHM_RISK_MATRIX_SCALE', '1.0')
    gt, raw = _loco_reward_inputs()
    metadata = {'data_mode': mode, 'class_counts': {c: 10 for c in classes}}
    expected = -(len(classes) * 10 / (10 + 1e-6)) ** .35 * math.exp(.5 * 1.1)
    assert get_setting(mode) == list(classes)
    assert active_reward(None, raw, gt, metadata)['cls_score'] == pytest.approx(expected)
    assert reference_reward(raw, gt, extra_info=metadata,
        config=RewardConfig(data_mode=helper_mode))['classification_reward'] == pytest.approx(expected)


@pytest.mark.parametrize('cohort,mode,helper_mode,classes', [
    ('private', 'private', 'private_5class', PRIVATE_CLASSES),
    ('public', '5cls', 'public_5class', PUBLIC_CLASSES),
    ('public', '3cls', 'public_3class', ('HCC', 'ICC', 'HA_Hemangioma')),
])
def test_main_profile_preserves_default_class_space(tmp_path, monkeypatch, cohort, mode, helper_mode, classes):
    from training.rars_joint_grid import candidate
    from rewards.verl_liver_reward import compute_score as active_reward, get_setting
    from rewards.full_reward import compute_score as reference_reward, RewardConfig
    profile = candidate({'cohort': cohort, 'protocol': 'main', 'values': [.1]}, 0)
    path = tmp_path / 'main_profile.json'
    path.write_text(json.dumps(profile))
    monkeypatch.setenv('RL_LHM_RISK_PROFILE', str(path))
    monkeypatch.setenv('RL_LHM_REWARD_MODE', 'rars')
    monkeypatch.setenv('RL_LHM_RISK_MATRIX_MODE', 'ordinal')
    monkeypatch.setenv('RL_LHM_RISK_MATRIX_SCALE', '1.0')
    gt, raw = _loco_reward_inputs()
    metadata = {'data_mode': mode, 'class_counts': {c: 10 for c in classes}}
    expected = -(len(classes) * 10 / (10 + 1e-6)) ** .35 * math.exp(.5 * .1)
    assert get_setting(mode) == list(classes)
    assert active_reward(None, raw, gt, metadata)['cls_score'] == pytest.approx(expected)
    assert reference_reward(raw, gt, extra_info=metadata,
        config=RewardConfig(data_mode=helper_mode, risk_profile=profile))['classification_reward'] == pytest.approx(expected)
