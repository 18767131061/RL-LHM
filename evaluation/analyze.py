"""Patient bootstrap, center stratification and paired evaluation comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.manifest import load_manifest
from data.provenance import file_digest, source_revision
from .classification import classification_metrics, class_spaces

METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def bootstrap_classification(rows, output_classes, metric_classes, *, samples, seed, comparison=None):
    """Resample paired patients, always retaining the predeclared macro classes."""
    class_spaces(output_classes, metric_classes)
    if type(samples) is not int or samples < 1 or not rows:
        raise ValueError("bootstrap requires patients and a positive sample count")
    ids = [row["patient_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("bootstrap rows must contain each patient once")
    other = None
    if comparison is not None:
        lookup = {row["patient_id"]: row for row in comparison}
        if len(lookup) != len(comparison) or set(lookup) != set(ids):
            raise ValueError("paired comparisons need identical patient populations")
        other = [lookup[uid] for uid in ids]
        if any(a["truth"] != b["truth"] for a, b in zip(rows, other)):
            raise ValueError("paired comparisons must have identical patient truth")
    def metrics(selection):
        return classification_metrics([r["truth"] for r in selection], [r["prediction"] for r in selection],
                                      output_classes, metric_classes=metric_classes)
    point = metrics(rows)
    if other is not None:
        base = metrics(other)
        point = {key: point[key] - base[key] for key in METRICS}
    draws = {key: [] for key in METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        indices = rng.integers(0, len(rows), size=len(rows))
        result = metrics([rows[i] for i in indices])
        baseline = metrics([other[i] for i in indices]) if other is not None else None
        for key in METRICS:
            draws[key].append(result[key] - baseline[key] if baseline is not None else result[key])
    return dict(unit="patient", sampling="unstratified_with_replacement", samples=samples, seed=seed,
                output_classes=list(output_classes), metric_classes=list(metric_classes),
                estimand="paired_difference" if other is not None else "metric",
                metrics={key: dict(estimate=point[key], percentile_95_ci=np.quantile(draws[key], [0.025, 0.975]).tolist())
                         for key in METRICS})


def center_class_table(manifest):
    centers = sorted(set(manifest.patient_centers.values()))
    table = {center: dict.fromkeys(manifest.output_classes, 0) for center in centers}
    for patient, center in manifest.patient_centers.items():
        table[center][manifest.patients[patient]] += 1
    return dict(patients=len(manifest.patients), patients_with_center=len(manifest.patient_centers),
                output_classes=list(manifest.output_classes), counts=table)


def analyze(manifest, evaluation, *, samples, seed, comparison=None):
    if (evaluation["output_classes"] != list(manifest.output_classes)
            or evaluation["metric_classes"] != list(manifest.metric_classes)):
        raise ValueError("manifest and evaluation class spaces must match")
    rows = evaluation["patients"]
    if len({r["patient_id"] for r in rows}) != len(rows) or {r["patient_id"]: r["truth"] for r in rows} != manifest.patients:
        raise ValueError("evaluation must match every manifest patient and truth")
    if comparison is not None and any(evaluation[key] != comparison[key] for key in ("output_classes", "metric_classes")):
        raise ValueError("comparison evaluations must use identical class spaces")
    def score(subset, predictions=None):
        return classification_metrics([row["truth"] for row in subset],
                                      predictions if predictions is not None else [row["prediction"] for row in subset],
                                      manifest.output_classes, metric_classes=manifest.metric_classes)
    per_center = {}
    for center in sorted(set(manifest.patient_centers.values())):
        subset = [row for row in rows if manifest.patient_centers[row["patient_id"]] == center]
        per_center[center] = score(subset)
    result = dict(center_class_table=center_class_table(manifest), per_center=per_center,
                  constant_prediction_references={label: score(rows, [label] * len(rows)) for label in manifest.output_classes},
                  bootstrap=bootstrap_classification(rows, manifest.output_classes, manifest.metric_classes,
                                                     samples=samples, seed=seed),
                  generation=evaluation.get("generation"), matching_sensitivity=evaluation.get("matching_sensitivity"))
    if comparison is not None:
        result["paired_bootstrap"] = bootstrap_classification(rows, manifest.output_classes, manifest.metric_classes,
                                                              samples=samples, seed=seed, comparison=comparison["patients"])
        if evaluation["matching_method"] != comparison["matching_method"]:
            raise ValueError("detection comparisons must use the same matching method")
        delta = {}
        for protocol, actual in evaluation["detection"].items():
            baseline = {row["iou_threshold"]: row for row in comparison["detection"][protocol]}
            if {row["iou_threshold"] for row in actual} != set(baseline):
                raise ValueError("detection thresholds must match")
            delta[protocol] = [{"iou_threshold": row["iou_threshold"],
                                **{key: row[key] - baseline[row["iou_threshold"]][key]
                                   for key in ("tp", "fp", "fn", "precision", "recall", "f1")}} for row in actual]
        result["detection_difference"] = delta
        result["localization_difference"] = (evaluation["localization"]["mean_iou_class_agnostic"]
                                             - comparison["localization"]["mean_iou_class_agnostic"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--compare", help="Baseline metrics.json for paired differences")
    parser.add_argument("--split-manifest", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    evaluation = json.loads(Path(args.metrics).read_text())
    comparison = json.loads(Path(args.compare).read_text()) if args.compare else None
    result = analyze(manifest, evaluation, samples=args.bootstrap_samples, seed=args.seed, comparison=comparison)
    splits = {}
    split_patients = {}
    for argument in args.split_manifest:
        name, path = argument.split("=", 1)
        if name in splits:
            parser.error("split names must be unique")
        split = load_manifest(path)
        splits[name] = {**center_class_table(split), "manifest_sha256": file_digest(path)}
        split_patients[name] = set(split.patients)
    result["split_center_class_tables"] = splits
    result["split_patient_overlap"] = {f"{a}:{b}": len(split_patients[a] & split_patients[b])
                                       for a in splits for b in splits if a < b}
    result["provenance"] = dict(source_revision=source_revision(), manifest_sha256=file_digest(args.manifest),
                                metrics_sha256=file_digest(args.metrics),
                                comparison_sha256=file_digest(args.compare) if args.compare else None)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
