#!/usr/bin/env python3
"""Render existing patient-sampling intervals; never estimate or resample data."""
import argparse
import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

KEY_FIELDS = ("record_type", "cohort", "direction", "class_name", "model",
              "comparator", "seed", "metric")
REQUIRED = KEY_FIELDS + ("summary_statistic", "estimate", "ci_lower", "ci_upper",
                         "ci_method", "raw_p", "bh_q", "test_method", "bh_family",
                         "test_patients", "source_table")
SEEDS = ("42", "2026", "3407", "1234", "5678")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
MARKERS = ("o", "s", "^", "D", "P")
MODELS = ("SFT", "vanilla GRPO", "Full RL-LHM")
COMPARATORS = MODELS[:2]
PANEL_SPECS = (("A", "private", ""), ("B", "public", ""),
               ("C", "private", "ICC"), ("D", "private", "FNH"),
               ("E", "public", "ICC"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(row, field):
    try:
        value = Decimal(row[field])
    except (InvalidOperation, KeyError):
        raise ValueError("Missing or nonnumeric {} in {}".format(field, record_key(row)))
    if not value.is_finite():
        raise ValueError("Nonfinite {} in {}".format(field, record_key(row)))
    return value


def record_key(row):
    return tuple(row.get(k, "") for k in KEY_FIELDS)


def load_records(path, expected_rows=None, expected_sha256=None):
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Source SHA256 does not match the supplied frozen source")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(REQUIRED).issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing required S1 fields")
        rows = list(reader)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError("Unexpected source row count")
    seen = set()
    for index, row in enumerate(rows, 1):
        key = record_key(row)
        if key in seen:
            raise ValueError("Duplicate source key: {}".format(key))
        seen.add(key)
        row["source_record"] = index
    return rows, digest


def validate_interval(row, is_recall):
    lower, estimate, upper = (number(row, k) for k in ("ci_lower", "estimate", "ci_upper"))
    if not lower <= estimate <= upper:
        raise ValueError("Interval does not contain its estimate: {}".format(record_key(row)))
    if is_recall and not Decimal(0) <= lower <= upper <= Decimal(1):
        raise ValueError("Recall interval is outside [0, 1]")
    if not is_recall and not Decimal(-1) <= lower <= upper <= Decimal(1):
        raise ValueError("Paired proportion-difference interval is outside [-1, 1]")
    expected_method = "patient_percentile_bootstrap" if is_recall else "paired_patient_percentile_bootstrap"
    if row["ci_method"] != expected_method:
        raise ValueError("Unexpected interval method: {}".format(record_key(row)))
    if row["summary_statistic"] != "single_seed_estimate":
        raise ValueError("A plotted row is not a single-seed estimate")


def select_panels(rows):
    """Return the exact 20 paired and 45 class rows in deterministic display order."""
    by_key = {record_key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate source key")
    panels = {}
    for panel, cohort, class_name in PANEL_SPECS:
        selected = []
        groups = MODELS if class_name else COMPARATORS
        for group_index, group in enumerate(groups):
            for seed_index, seed in enumerate(SEEDS):
                key = (("class_performance", cohort, "", class_name, group, "", seed, "recall")
                       if class_name else
                       ("paired_difference", cohort, "", "", "Full RL-LHM", group, seed, "macro_f1"))
                if key not in by_key:
                    raise ValueError("Missing required plotted record: {}".format(key))
                row = dict(by_key[key])
                validate_interval(row, bool(class_name))
                row.update(panel=panel, plot_group=group,
                           plot_y=group_index * 6 + seed_index,
                           seed_color=COLORS[seed_index], seed_marker=MARKERS[seed_index])
                selected.append(row)
        counts = {row["test_patients"] for row in selected}
        if len(counts) != 1 or not next(iter(counts)).isdigit() or int(next(iter(counts))) < 1:
            raise ValueError("Panel patient counts must be consistent positive integers")
        panels[panel] = selected
    return panels


def validate_paired_family(rows):
    paired = [row for row in rows if row["record_type"] == "paired_difference"]
    expected = {(cohort, seed, comparator, metric)
                for cohort in ("private", "public") for seed in SEEDS
                for comparator in COMPARATORS for metric in ("accuracy", "macro_recall", "macro_f1")}
    observed = {(r["cohort"], r["seed"], r["comparator"], r["metric"]) for r in paired}
    if len(paired) != 60 or observed != expected:
        raise ValueError("Paired family must contain all 60 original endpoints")
    families = {row["bh_family"] for row in paired}
    if len(families) != 1 or "" in families:
        raise ValueError("Paired records do not share one nonempty BH family label")
    for row in paired:
        if row["model"] != "Full RL-LHM" or row["direction"] or row["class_name"]:
            raise ValueError("Unexpected paired family identity")
        for field in ("raw_p", "bh_q"):
            if not Decimal(0) <= number(row, field) <= Decimal(1):
                raise ValueError("{} is outside [0, 1]".format(field))
    return paired


def paired_limits(panels):
    rows = panels["A"] + panels["B"]
    low = min(0., min(float(r["ci_lower"]) for r in rows))
    high = max(0., max(float(r["ci_upper"]) for r in rows))
    padding = max(.01, .04 * (high - low))
    return (math.floor((low - padding) * 20) / 20,
            math.ceil((high + padding) * 20) / 20)


def render(panels, output_prefix, dpi=600):
    """All horizontal lines are supplied CI endpoints, with no statistical calls."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "patient-uncertainty-v1",
        "axes.linewidth": .65, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(6.3, 5.35), facecolor="white")
    # Shared row labels leave more width for the three lower interval panels.
    top = fig.add_gridspec(1, 2, left=.205, right=.985, bottom=.635, top=.875, wspace=.24)
    bottom = fig.add_gridspec(1, 3, left=.205, right=.985, bottom=.09, top=.475, wspace=.25)
    axes = [fig.add_subplot(top[0, i]) for i in range(2)]
    axes += [fig.add_subplot(bottom[0, i]) for i in range(3)]
    effect_limits = paired_limits(panels)
    for ax, (panel, cohort, class_name) in zip(axes, PANEL_SPECS):
        rows = panels[panel]
        n = rows[0]["test_patients"]
        title = ("{}  {} (n = {})".format(panel, cohort.capitalize(), n) if not class_name
                 else "{}  {} {} (n = {})".format(panel, cohort.capitalize(), class_name, n))
        ax.set_title(title, loc="left", pad=8, fontweight="bold", fontsize=8.5)
        for row in rows:
            lo, value, hi = (float(row[k]) for k in ("ci_lower", "estimate", "ci_upper"))
            y = row["plot_y"]
            # Draw endpoints directly: no rounding, percentile calculation or CI clipping.
            ax.plot([lo, hi], [y, y], color=row["seed_color"], lw=1.05, clip_on=False, zorder=2)
            ax.plot([lo, lo], [y-.13, y+.13], color=row["seed_color"], lw=.8, clip_on=False)
            ax.plot([hi, hi], [y-.13, y+.13], color=row["seed_color"], lw=.8, clip_on=False)
            ax.plot(value, y, marker=row["seed_marker"], ms=3.8, color=row["seed_color"],
                    markeredgewidth=.55, markeredgecolor="white", linestyle="none", clip_on=False, zorder=3)
        if class_name:
            ax.set_xlim(0, 1)
            ax.set_ylim(17, -1)
            ax.set_xticks([0, .5, 1], labels=["0", "0.5", "1"])
            ax.set_xlabel("Recall")
            ax.set_yticks([2, 8, 14], labels=["SFT", "vanilla GRPO", "Full RL-LHM"])
            separators = [5, 11]
        else:
            ax.set_xlim(*effect_limits)
            ticks = [i / 10 for i in range(math.ceil(effect_limits[0] * 10), math.floor(effect_limits[1] * 10) + 1)]
            ax.set_xticks(ticks if len(ticks) >= 2 else effect_limits)
            ax.set_ylim(11, -1)
            ax.set_xlabel("Macro-F1 difference")
            ax.set_yticks([2, 8], labels=["Full − SFT", "Full − vanilla GRPO"])
            ax.axvline(0, color="0.35", lw=.8, linestyle=(0, (3, 2)), zorder=1)
            separators = [5]
        for y in separators:
            ax.axhline(y, color="0.87", lw=.65, zorder=0)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=3, pad=3)
        if panel not in ("A", "C"):
            ax.tick_params(labelleft=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="0.93", linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
    handles = [Line2D([0], [0], color=color, marker=marker, markersize=4,
                      linewidth=1, label=seed) for seed, color, marker in zip(SEEDS, COLORS, MARKERS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.58, .99),
               ncol=5, frameon=False, title="Training seed", title_fontsize=8,
               columnspacing=1.5, handlelength=1.25, handletextpad=.45)
    fig.canvas.draw()
    min_font = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text())
    if min_font < 8:
        raise ValueError("Unexpected font below 8 pt")
    renderer = fig.canvas.get_renderer()
    visible_text = [t for t in fig.findobj(Text) if t.get_text() and t.get_visible()]
    for i, text in enumerate(visible_text):
        bounds = text.get_window_extent(renderer)
        for other in visible_text[i+1:]:
            if bounds.overlaps(other.get_window_extent(renderer)):
                raise ValueError("Overlapping figure labels: {!r} / {!r}".format(text.get_text(), other.get_text()))
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_prefix)+".pdf", metadata={"Creator": "plot_patient_uncertainty.py",
                                                    "CreationDate": None, "ModDate": None})
    fig.savefig(str(output_prefix)+".svg", metadata={"Date": None, "Creator": "plot_patient_uncertainty.py"})
    fig.savefig(str(output_prefix)+".png", dpi=dpi)
    plt.close(fig)
    return {"width_inches": 6.3, "height_inches": 5.35, "minimum_font_pt": min_font,
            "png_dpi": dpi, "paired_axis_limits": effect_limits, "recall_axis_limits": [0, 1],
            "all_ci_endpoints_within_axes": True, "endpoint_clipping_disabled": True,
            "legend_seed_order": list(SEEDS), "visible_text_bbox_overlaps": 0}


def range_strings(rows, field):
    values = [number(row, field) for row in rows]
    return [str(min(values)), str(max(values))]


def write_audit(source, digest, rows, panels, paired, layout, output_prefix, audit_dir):
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    selected = [row for group in panels.values() for row in group]
    mapping_fields = ["panel", "source_record", "plot_group", "plot_y", "seed_color", "seed_marker"]
    mapping_fields += [key for key in rows[0] if key not in mapping_fields]
    with (audit_dir / "plotted_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mapping_fields)
        writer.writeheader()
        writer.writerows(selected)
    (audit_dir / "plotted_records.json").write_text(json.dumps(selected, indent=2)+"\n")
    after = sha256(source)
    if after != digest:
        raise ValueError("Source bytes changed during rendering")
    audit = {
        "source_file": Path(source).name, "source_sha256_before": digest,
        "source_sha256_after": after, "source_unchanged": True,
        "source_rows": len(rows), "unique_key_fields": KEY_FIELDS,
        "unique_source_keys": len({record_key(row) for row in rows}),
        "plotted_rows": len(selected), "paired_macro_f1_rows": 20, "class_recall_rows": 45,
        "panel_rows": {k: len(v) for k, v in panels.items()},
        "panel_groups": {k: dict(Counter(r["plot_group"] for r in v)) for k, v in panels.items()},
        "panel_test_patients": {k: v[0]["test_patients"] for k, v in panels.items()},
        "plotted_ci_methods": dict(Counter(row["ci_method"] for row in selected)),
        "paired_macro_f1_estimate_range": range_strings(panels["A"]+panels["B"], "estimate"),
        "paired_macro_f1_ci_range": [str(min(number(r, "ci_lower") for r in panels["A"]+panels["B"])),
                                     str(max(number(r, "ci_upper") for r in panels["A"]+panels["B"]))],
        "paired_macro_f1_q_range": range_strings(panels["A"]+panels["B"], "bh_q"),
        "original_bh_family_rows": len(paired), "original_bh_family_q_range": range_strings(paired, "bh_q"),
        "original_bh_family_q_below_0_05": sum(number(r, "bh_q") < Decimal("0.05") for r in paired),
        "paired_ci_include_zero": sum(number(r, "ci_lower") <= 0 <= number(r, "ci_upper")
                                      for r in panels["A"]+panels["B"]),
        "numeric_mapping": "Source decimal strings copied unchanged; floats used only for plot coordinates.",
        "statistical_operations": "None: no resampling, pooling, seed aggregation, tests or BH recalculation.",
        "layout": layout,
        "runtime": {"matplotlib": matplotlib.__version__},
        "asset_sha256": {ext: sha256(str(output_prefix)+"."+ext) for ext in ("pdf", "png", "svg")},
    }
    (audit_dir / "validation.json").write_text(json.dumps(audit, indent=2)+"\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True,
                        help="Local nonpublic directory for exact source mappings and validation")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    rows, digest = load_records(args.input, args.expected_rows, args.expected_sha256)
    panels = select_panels(rows)
    paired = validate_paired_family(rows)
    layout = render(panels, args.output_prefix, args.dpi)
    audit = write_audit(args.input, digest, rows, panels, paired, layout, args.output_prefix, args.audit_dir)
    print(json.dumps({"source_rows": len(rows), "plotted_rows": 65, "source_unchanged": True,
                      "asset_sha256": audit["asset_sha256"], "layout": layout}, indent=2))


if __name__ == "__main__":
    main()
