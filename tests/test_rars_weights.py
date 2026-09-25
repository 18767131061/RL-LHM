import copy

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

    assert all(value == 0.5600 for row in uniform_matrix("private").values() for value in row.values())
    assert all(value == 0.7100 for row in uniform_matrix("public").values() for value in row.values())
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
