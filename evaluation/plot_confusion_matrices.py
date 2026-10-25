#!/usr/bin/env python3
"""Plot complete model × five-seed confusion panels with an abstention column."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.export_paper_tables import load_cohorts
from evaluation.joint_seed_statistics import require


def confusion_records(cohort):
    """Count every patient, including absent classes and undefined predictions."""
    classes = cohort.spec["classes"]
    rows = cohort.spec["macro_classes"]
    records = []
    matrices = {}
    for model in cohort.spec["model_order"]:
        for seed, payload in zip(cohort.seeds, cohort.payloads[model]):
            matrix = np.zeros((len(rows), len(classes)+1), dtype=np.int64)
            for patient in payload["patients"]:
                predicted = patient["prediction"]
                matrix[rows.index(patient["truth"]), len(classes) if predicted is None else classes.index(predicted)] += 1
            require(int(matrix.sum()) == len(cohort.population[2]), "Confusion panel lost a patient")
            supports = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, supports, out=np.zeros_like(matrix, dtype=float), where=supports != 0)
            matrices[model, seed] = (matrix, normalized)
            for i, truth in enumerate(rows):
                for j, predicted in enumerate([*classes, None]):
                    records.append({"cohort": cohort.spec["id"], "cohort_label": cohort.spec["label"], "model": model, "seed": seed,
                        "truth": truth, "prediction": predicted, "abstention": predicted is None, "count": int(matrix[i, j]),
                        "truth_support": int(supports[i, 0]), "row_proportion": float(normalized[i, j])})
    return records, matrices


def plot_cohort(cohort, matrices, output_dir, normalization, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    cfg = cohort.spec
    aliases = cfg.get("class_labels", {label: label for label in cfg["classes"]})
    xticks = [aliases[label] for label in cfg["classes"]] + ["Abstention"]
    yticks = [aliases[label] for label in cfg["macro_classes"]]
    selection = 1 if normalization == "row" else 0
    vmax = 1 if normalization == "row" else max(int(value[0].max()) for value in matrices.values())
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "svg.hashsalt": "rl-lhm-confusion-v1", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(len(cfg["model_order"]), 5, squeeze=False,
                                 figsize=(18, 3.8*len(cfg["model_order"])), layout="constrained")
        fig.get_layout_engine().set(h_pad=.12, w_pad=.08, hspace=.08)
        try:
            for i, model in enumerate(cfg["model_order"]):
                for j, seed in enumerate(cohort.seeds):
                    ax = axes[i, j]
                    values = matrices[model, seed][selection]
                    color = ax.imshow(values, vmin=0, vmax=max(1, vmax), cmap="Blues", interpolation="nearest")
                    ax.set_xticks(range(len(xticks)), labels=xticks, rotation=45, ha="right")
                    ax.set_yticks(range(len(yticks)), labels=yticks)
                    ax.set_title(textwrap.fill(model, width=30) + f"\nSeed {seed}")
                    ax.set_xlabel("Predicted class")
                    ax.set_ylabel("True class")
                    for r in range(values.shape[0]):
                        for c in range(values.shape[1]):
                            text = f"{values[r, c]:.2f}" if normalization == "row" else str(int(values[r, c]))
                            ax.text(c, r, text, ha="center", va="center", fontsize=7,
                                    color="white" if values[r, c] > .55*max(1, vmax) else "black")
                    ax.axvline(len(xticks)-1.5, color="black", linewidth=.8)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
            fig.suptitle(f"{cfg['label']} · {len(cohort.population[2])} patients per panel · " +
                         ("Row proportions (all patients retained)" if normalization == "row" else "Patient counts"), fontsize=12)
            colorbar = fig.colorbar(color, ax=axes.ravel().tolist(), shrink=.6, label="Row proportion" if normalization == "row" else "Patients")
            if normalization == "counts":
                colorbar.locator = MaxNLocator(integer=True)
                colorbar.update_ticks()
            base = output_dir / f"{cfg['id']}_confusion_{normalization}"
            for extension, metadata in (("png", {"Software": "RL-LHM"}),
                                        ("svg", {"Date": None, "Creator": "RL-LHM"}),
                                        ("pdf", {"CreationDate": None, "ModDate": None, "Creator": "RL-LHM"})):
                fig.savefig(base.with_suffix("."+extension), dpi=dpi, metadata=metadata)
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="The same explicit cohort/run spec used by export_paper_tables")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalize", choices=("counts", "row", "both"), default="counts")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    try:
        require(72 <= args.dpi <= 1200, "dpi must be in [72,1200]")
        require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Use an empty output directory")
        _, cohorts = load_cohorts(args.spec)
        data = [(cohort, *confusion_records(cohort)) for cohort in cohorts]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        records = [row for _, rows, _ in data for row in rows]
        result = {"schema_version": 1, "normalization": args.normalize,
            "spec": {"path": str(args.spec.resolve()), "sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest()},
            "cohorts": [{"id": c.spec["id"], "label": c.spec["label"], "seeds": c.seeds, "model_order": c.spec["model_order"],
                         "row_classes": c.spec["macro_classes"], "predicted_classes": [*c.spec["classes"], None],
                         "patients_per_panel": len(c.population[2]), "sources": c.sources} for c in cohorts], "cells": records}
        (args.output_dir / "confusion_matrices.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        with (args.output_dir / "confusion_matrices.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        for cohort, _, matrices in data:
            for normalization in (("counts", "row") if args.normalize == "both" else (args.normalize,)):
                plot_cohort(cohort, matrices, args.output_dir, normalization, args.dpi)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "panels": sum(len(c.payloads)*5 for c in cohorts)}))


if __name__ == "__main__":
    main()
