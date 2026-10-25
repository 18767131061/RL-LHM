from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_COLUMNS = {
    "pair_id",
    "x_parameter",
    "y_parameter",
    "x_weight",
    "y_weight",
    "macro_f1",
    "energy",
    "selected_cell",
    "test_set_used",
}


def load_rars_grid(path: str | Path, pair_id: str) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"RARS grid is missing columns: {sorted(missing)}")
        rows = [row for row in reader if row["pair_id"] == pair_id]
    if len(rows) != 225:
        raise ValueError(f"Expected 225 cells for {pair_id}, found {len(rows)}")
    if any(row["test_set_used"].strip().lower() != "false" for row in rows):
        raise ValueError("RARS tuning grid must exclude the Test set")

    x_values = sorted({float(row["x_weight"]) for row in rows})
    y_values = sorted({float(row["y_weight"]) for row in rows})
    if len(x_values) != 15 or len(y_values) != 15:
        raise ValueError("RARS pair must be a 15x15 grid")
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    macro_f1 = np.full((15, 15), np.nan, dtype=float)
    energy = np.full((15, 15), np.nan, dtype=float)
    selected: tuple[float, float] | None = None
    for row in rows:
        x_value = float(row["x_weight"])
        y_value = float(row["y_weight"])
        position = (y_index[y_value], x_index[x_value])
        if np.isfinite(macro_f1[position]):
            raise ValueError(f"Duplicate RARS cell at {x_value}, {y_value}")
        macro_f1[position] = float(row["macro_f1"])
        energy[position] = float(row["energy"])
        if row["selected_cell"].strip().lower() == "yes":
            if selected is not None:
                raise ValueError("RARS pair has multiple selected cells")
            selected = (x_value, y_value)
    if selected is None or not np.allclose(energy, 1.0 - macro_f1, atol=1e-6):
        raise ValueError("RARS selected cell or energy closure is invalid")
    return {
        "pair_id": pair_id,
        "x_parameter": rows[0]["x_parameter"],
        "y_parameter": rows[0]["y_parameter"],
        "x_values": np.asarray(x_values),
        "y_values": np.asarray(y_values),
        "macro_f1": macro_f1,
        "energy": energy,
        "selected": selected,
    }


def plot_rars_pair(grid: dict[str, Any], output_path: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("RARS plotting requires matplotlib") from error

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    extent = [
        float(grid["x_values"][0]),
        float(grid["x_values"][-1]),
        float(grid["y_values"][0]),
        float(grid["y_values"][-1]),
    ]
    for axis, field, title in zip(
        axes,
        ("macro_f1", "energy"),
        ("Macro-F1", "Energy = 1 - Macro-F1"),
    ):
        image = axis.imshow(
            grid[field],
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="auto",
        )
        contours = axis.contour(
            grid["x_values"],
            grid["y_values"],
            grid[field],
            levels=6,
            colors="black",
            linewidths=0.5,
            alpha=0.55,
        )
        axis.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
        best_index = np.unravel_index(np.argmax(grid["macro_f1"]), grid["macro_f1"].shape)
        best = (grid["x_values"][best_index[1]], grid["y_values"][best_index[0]])
        axis.scatter(
            *grid["selected"],
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=1.5,
            s=90,
            label="Frozen selected",
        )
        axis.scatter(*best, marker="*", color="gold", edgecolors="black", s=95, label="Pair best")
        axis.set_xlabel(grid["x_parameter"])
        axis.set_ylabel(grid["y_parameter"])
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output
