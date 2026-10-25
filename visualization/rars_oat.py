"""Validate and plot one-dimensional RARS seed-mean sensitivity curves."""
from __future__ import annotations

import math
from pathlib import Path

from training.rars_joint_grid import SEEDS, VALUES, sha, write_json
from training.rars_oat import METRICS, exact_seed_bootstrap, oat_design, read_json


def load_oat_selection(path):
    data = read_json(path)
    if data.get("workflow") != "rars_oat_v2" or data.get("status") not in ("validation_complete", "synthetic_complete"):
        raise ValueError("Require a complete OAT v2 selection")
    if type(data.get("synthetic_test")) is not bool or data["synthetic_test"] != (data["status"] == "synthetic_complete"):
        raise ValueError("Synthetic status is inconsistent")
    axes, profiles, points = oat_design(data["cohort"])
    if data.get("seeds") != SEEDS or data.get("values") != VALUES or set(data.get("curves", {})) != set(axes):
        raise ValueError("Selection must contain every axis, all 15 weights and the five fixed seeds")
    for axis in axes:
        curve = data["curves"][axis]
        expected = [point for point in points if point["axis"] == axis]
        if len(curve) != 15:
            raise ValueError("Every OAT axis requires 15 complete points")
        for point, identity in zip(curve, expected):
            if any(point.get(key) != value for key, value in identity.items()):
                raise ValueError("Axis point/profile/task identity differs from OAT design")
            rows = point.get("seed_metrics", [])
            if [row.get("seed") for row in rows] != SEEDS:
                raise ValueError("Every OAT point requires the five fixed seed metrics")
            for key in METRICS:
                values = [row.get(key) for row in rows]
                if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                    raise ValueError("Seed metrics must be finite proportions")
                mean = math.fsum(values) / 5
                if not math.isclose(point.get(f"mean_validation_{key}", -1), mean, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Curve mean does not match its five seed values")
            expected_ci = exact_seed_bootstrap([row["macro_f1"] for row in rows])
            if point.get("macro_f1_seed_bootstrap_95") != expected_ci:
                raise ValueError("Macro-F1 interval must use all 3125 ordered seed resamples and linear percentiles")
        selected = min(curve, key=lambda p: (-p["mean_validation_macro_f1"], p["value"]))
        if data["selected_axis_points"].get(axis) != selected:
            raise ValueError("Selected weight differs from the complete five-seed curve")
    return data


def plot_oat(selection_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    data = load_oat_selection(selection_path)
    axes_names = list(data["curves"])
    ncols = 3 if len(axes_names) == 5 else 4
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 6.5), sharex=True, sharey=True)
    colors = {"accuracy": "#0072B2", "macro_recall": "#009E73", "macro_f1": "#CC79A7"}
    labels = {"accuracy": "Accuracy", "macro_recall": "Macro-Recall", "macro_f1": "Macro-F1"}
    for ax, name in zip(axes.flat, axes_names):
        curve = data["curves"][name]
        x = [point["value"] for point in curve]
        for metric in METRICS:
            ax.plot(x, [point[f"mean_validation_{metric}"] for point in curve], color=colors[metric],
                    marker="o", markersize=2.8, linewidth=1.5, label=labels[metric])
        ax.fill_between(x, [point["macro_f1_seed_bootstrap_95"]["lower"] for point in curve],
                        [point["macro_f1_seed_bootstrap_95"]["upper"] for point in curve],
                        color=colors["macro_f1"], alpha=.18, linewidth=0)
        ax.axvline(data["selected_axis_points"][name]["value"], color="#C62828", linestyle="--", linewidth=1.2)
        ax.set_title(name.replace("HA_Hemangioma", "HH").replace("nodule_CYST", "NC").replace("--", " – ") if name != "other" else "Shared other-pair weight")
        ax.set_xlim(.06, 1.54)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([.1, .3, .5, .7, .9, 1.1, 1.3, 1.5])
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(axes_names):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=colors[key], label=labels[key]) for key in METRICS]
    handles.append(Line2D([], [], color="#C62828", linestyle="--", label="Selected weight"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .005))
    fig.supxlabel("Severity weight (all other weights fixed at 1.0)", y=.07)
    fig.supylabel("Validation metric")
    suffix = " — SYNTHETIC TEST" if data["synthetic_test"] else ""
    fig.suptitle(f'{data["cohort"].title()} {data["split_id"]}: five-seed OAT sensitivity{suffix}', fontsize=13)
    fig.tight_layout(rect=(.015, .10, 1, .95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_json(output.with_suffix(output.suffix + ".metadata.json"), {
        "schema_version": 1, "selection_sha256": sha(selection_path), "output_sha256": sha(output),
        "plot_source_sha256": sha(__file__),
        "statistics_source_sha256": sha(Path(__file__).resolve().parents[1] / "training/rars_oat.py"),
        "synthetic_test": data["synthetic_test"], "seeds": SEEDS, "values": VALUES,
        "curves": list(METRICS), "band": "macro_f1", "resampling_unit": "training_seed",
        "ordered_resamples": 3125, "percentiles": [2.5, 97.5], "quantile": "linear",
    })
    return output
