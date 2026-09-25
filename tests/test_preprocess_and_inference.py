import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from inference.generation import completion_nll
from preprocess.center_split import audit_center_isolation, stratified_tuning_split
from visualization.rars_surface import load_rars_grid


def test_rars_plot_cli_is_directly_importable():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "plot_rars.py"), "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--pair-id" in result.stdout


def test_center_split_is_deterministic_and_stratified():
    rows = [
        {"patient_id": f"{label}{index}", "class": label, "center_id": "A"}
        for label in ("X", "Y")
        for index in range(5)
    ]
    first = stratified_tuning_split(
        rows,
        source_center_id="A",
        tuning_fraction=0.2,
        seed=2026,
    )
    second = stratified_tuning_split(
        rows,
        source_center_id="A",
        tuning_fraction=0.2,
        seed=2026,
    )
    assert first == second
    assert {row["class"] for row in first[1]} == {"X", "Y"}
    assert not ({row["patient_id"] for row in first[0]} & {row["patient_id"] for row in first[1]})


def test_center_isolation_rejects_patient_overlap():
    rows = [
        {"patient_id": "p1", "center_id": "A", "split": "train"},
        {"patient_id": "p1", "center_id": "B", "split": "test"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        audit_center_isolation(rows, source_center_id="A", target_center_id="B")


def test_completion_nll_matches_hand_calculation():
    scores = (
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[2.0, 0.0]]),
    )
    tokens = torch.tensor([[1, 0]])
    expected = float(
        (-torch.log_softmax(scores[0], dim=-1)[0, 1]
         - torch.log_softmax(scores[1], dim=-1)[0, 0]) / 2
    )
    assert completion_nll(scores, tokens) == pytest.approx(expected)


def test_rars_loader_requires_direct_15_by_15_grid(tmp_path):
    path = tmp_path / "grid.csv"
    fields = [
        "pair_id",
        "x_parameter",
        "y_parameter",
        "x_weight",
        "y_weight",
        "macro_f1",
        "energy",
        "selected_cell",
        "test_set_used",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for x in range(1, 16):
            for y in range(1, 16):
                score = 0.5 + 0.001 * x - 0.001 * y
                writer.writerow(
                    {
                        "pair_id": "P1",
                        "x_parameter": "a",
                        "y_parameter": "b",
                        "x_weight": x / 10,
                        "y_weight": y / 10,
                        "macro_f1": score,
                        "energy": 1 - score,
                        "selected_cell": "yes" if (x, y) == (5, 5) else "no",
                        "test_set_used": "false",
                    }
                )
    grid = load_rars_grid(path, "P1")
    assert grid["macro_f1"].shape == (15, 15)
    assert grid["selected"] == (0.5, 0.5)
