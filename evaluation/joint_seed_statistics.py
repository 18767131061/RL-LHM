#!/usr/bin/env python3
"""Patient-paired inference conditional on five fixed model seeds.

Read canonical per-patient exports, recompute each seed's metrics, then average
those metrics. One patient swap or bootstrap multiplicity is shared by all seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.patient import MAJORITY_NLL_TIEBREAK
from evaluation.evaluate_paper_protocol import AGGREGATION_RULES, PROTOCOL_VERSION, _read_json
from evaluation.schema import SchemaError
from matching.protocols import PROTOCOLS


class PairingError(ValueError):
    """The supplied files do not identify the same evaluation population."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PairingError(message)


def _integer(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value


def _labels(value: Any, name: str) -> tuple[str, ...]:
    require(isinstance(value, list) and bool(value), f"{name} must be a nonempty list")
    require(all(isinstance(x, str) and x and x == x.strip() for x in value), f"Invalid {name}")
    require(len(value) == len(set(value)), f"Duplicate {name}")
    return tuple(value)


def validate_run_identity(payload: dict) -> dict:
    identity = payload.get("run_identity")
    require(isinstance(identity, dict) and set(identity) == {"model_name", "seed"},
            "Require complete run_identity (model_name and seed); rerun the canonical evaluator with both identity flags")
    name = identity["model_name"]
    require(isinstance(name, str) and bool(name) and name == name.strip(), "Invalid run_identity model_name")
    require(type(identity["seed"]) is int, "Run identity seed must be an integer")
    return identity


def indexed_rows(value: Any, name: str) -> dict[str, dict]:
    require(isinstance(value, list) and bool(value), f"Missing nonempty {name}")
    result = {}
    for row in value:
        require(isinstance(row, dict), f"Invalid {name} row")
        patient = row.get("patient_id")
        require(isinstance(patient, str) and patient and patient == patient.strip(), f"Invalid {name} patient_id")
        require(patient not in result, f"Duplicate {name} patient_id: {patient}")
        result[patient] = row
    return result


def detection_rows(value: Any) -> dict[tuple[str, float], dict]:
    require(isinstance(value, dict) and set(value) == set(PROTOCOLS), "Detection protocols must explicitly be CA, GFCA and SCF canonical keys")
    result = {}
    for protocol in PROTOCOLS:
        require(isinstance(value[protocol], list) and bool(value[protocol]), "Missing detection thresholds")
        for row in value[protocol]:
            require(isinstance(row, dict), "Invalid detection row")
            threshold = row.get("iou_threshold")
            require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1, "Invalid detection threshold")
            key = protocol, float(threshold)
            require(key not in result, f"Duplicate detection threshold: {key}")
            result[key] = row
    threshold_sets = [{key[1] for key in result if key[0] == protocol} for protocol in PROTOCOLS]
    require(all(t == threshold_sets[0] for t in threshold_sets), "Threshold sets differ across protocols")
    return result


@dataclass(frozen=True)
class MetricLayout:
    classes: tuple[str, ...]
    macro_classes: tuple[str, ...]
    detection_keys: tuple[tuple[str, float], ...]

    @property
    def width(self) -> int:
        return 2 + 3 * len(self.macro_classes) + 4 * len(self.detection_keys)

    @property
    def names(self) -> list[str]:
        names = ["classification:accuracy", "classification:macro_precision", "classification:macro_recall", "classification:macro_f1"]
        names += [f"class:{label}:{metric}" for label in self.macro_classes for metric in ("precision", "recall", "f1")]
        for protocol, threshold in self.detection_keys:
            value = float(threshold)
            # General-format :g rounds to six significant digits and can merge
            # distinct valid thresholds, silently overwriting endpoint results.
            threshold_name = str(int(value)) if value.is_integer() else repr(value)
            names += [f"detection:{protocol}:iou={threshold_name}:{metric}" for metric in ("precision", "recall", "f1")]
        names += [f"detection:{protocol}:mean_iou" for protocol in PROTOCOLS]
        require(len(names) == len(set(names)), "Metric endpoint names must be unique")
        return names

    def metrics(self, totals: np.ndarray) -> np.ndarray:
        """Map additive totals (..., features) to non-linear metrics (..., metrics)."""
        def ratio(a, b):
            return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

        accuracy = ratio(totals[..., 0], totals[..., 1])
        c = len(self.macro_classes)
        cls = totals[..., 2:2 + 3*c].reshape(*totals.shape[:-1], c, 3)
        tp, predicted, support = (cls[..., i] for i in range(3))
        precision, recall, f1 = ratio(tp, predicted), ratio(tp, support), ratio(2*tp, predicted+support)
        result = [accuracy, precision.mean(axis=-1), recall.mean(axis=-1), f1.mean(axis=-1)]
        for i in range(c):
            result.extend([precision[..., i], recall[..., i], f1[..., i]])
        det = totals[..., 2 + 3*c:].reshape(*totals.shape[:-1], len(self.detection_keys), 4)
        for i in range(len(self.detection_keys)):
            tp, predicted, gt = (det[..., i, j] for j in range(3))
            result.extend([ratio(tp, predicted), ratio(tp, gt), ratio(2*tp, predicted+gt)])
        for protocol in PROTOCOLS:
            i = next(i for i, key in enumerate(self.detection_keys) if key[0] == protocol)
            result.append(ratio(det[..., i, 3], det[..., i, 2]))
        return np.stack(result, axis=-1)


def extract_run(payload: dict) -> tuple[MetricLayout, tuple, np.ndarray, np.ndarray]:
    require(isinstance(payload, dict) and payload.get("protocol_version") == PROTOCOL_VERSION, "Require canonical evaluate_paper_protocol metrics.json")
    classes = _labels(payload.get("classes"), "classes")
    macro = _labels(payload.get("macro_classes"), "macro_classes")
    require(set(macro).issubset(classes), "Macro classes are not an output-vocabulary subset")
    identity = payload.get("evaluation_identity")
    require(isinstance(identity, dict), "Missing evaluation_identity; rerun the canonical evaluator")
    digest = identity.get("manifest_sha256")
    rule = identity.get("aggregation_rule")
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Missing manifest content SHA256")
    require(isinstance(rule, str) and bool(rule), "Missing aggregation rule identity")
    if "aggregation_strategy" in payload:
        require(payload["aggregation_strategy"] in AGGREGATION_RULES
                and AGGREGATION_RULES[payload["aggregation_strategy"]] == rule,
                "Aggregation strategy and rule identity disagree")
    patients = indexed_rows(payload.get("patients"), "patients")
    detection = indexed_rows(payload.get("patient_detection"), "patient_detection")
    require(set(patients) == set(detection), "Patient detection and classification populations differ")
    keys = tuple(sorted(patients))
    truth = tuple(patients[p].get("truth") for p in keys)
    require(all(t in macro for t in truth), "Patient truth outside macro class space")
    aggregate = detection_rows(payload.get("detection"))
    det_keys = tuple(sorted(aggregate, key=lambda k: (PROTOCOLS.index(k[0]), k[1])))
    layout = MetricLayout(classes, macro, det_keys)
    units = np.zeros((len(keys), layout.width), dtype=float)
    gt_counts = []
    for i, patient in enumerate(keys):
        row = patients[patient]
        prediction = row.get("prediction")
        require("prediction" in row and (prediction is None or prediction in classes), f"Invalid prediction for {patient}")
        units[i, :2] = (row["truth"] == prediction, 1)
        for j, label in enumerate(macro):
            units[i, 2+3*j:5+3*j] = (row["truth"] == label and prediction == label, prediction == label, row["truth"] == label)
        per_patient = detection_rows(detection[patient].get("detection"))
        require(set(per_patient) == set(aggregate), f"Patient {patient} detection protocol/threshold mismatch")
        gt_for_patient, pred_for_patient = set(), set()
        iou_by_protocol = {}
        tp_by_protocol = {protocol: [] for protocol in PROTOCOLS}
        for j, key in enumerate(det_keys):
            counts = per_patient[key]
            tp, fp, fn, gt, pred = [_integer(counts.get(k), k) for k in ("tp", "fp", "fn", "gt_instances", "predicted_instances")]
            require(tp+fp == pred and tp+fn == gt, f"Detection count identities fail for {patient}, {key}")
            iou = counts.get("iou_sum")
            require(type(iou) in (int, float) and math.isfinite(iou) and 0 <= iou <= gt+1e-9, f"Invalid ungated iou_sum for {patient}")
            require(iou <= min(gt, pred)+1e-9, f"IoU sum exceeds matched object bound for {patient}")
            gt_for_patient.add(gt); pred_for_patient.add(pred)
            if key[0] in iou_by_protocol:
                require(math.isclose(iou, iou_by_protocol[key[0]], abs_tol=1e-9), "Ungated IoU changes across thresholds")
            iou_by_protocol[key[0]] = iou
            tp_by_protocol[key[0]].append(tp)
            units[i, 2+3*len(macro)+4*j:6+3*len(macro)+4*j] = (tp, pred, gt, iou)
        require(len(gt_for_patient) == len(pred_for_patient) == 1, "Detection population changes across protocols/thresholds")
        require(all(all(a >= b for a, b in zip(v, v[1:])) for v in tp_by_protocol.values()), "TP increases with IoU threshold")
        gt_counts.append(next(iter(gt_for_patient)))
    # The per-patient units, rather than pasted aggregate ratios, drive inference.
    recalculated = layout.metrics(units.sum(axis=0))
    existing = payload.get("classification", {})
    require(existing.get("n") == len(keys), "Classification patient denominator differs")
    for name, value in zip(layout.names[:4], recalculated[:4]):
        reported = existing.get(name.split(":")[-1])
        require(type(reported) in (int, float) and math.isclose(value, reported, abs_tol=1e-10), f"Aggregate {name} does not match patient predictions")
    for j, key in enumerate(det_keys):
        values = units[:, 2+3*len(macro)+4*j:6+3*len(macro)+4*j].sum(axis=0)
        tp, pred, gt, iou = values
        for name, value in (("tp", tp), ("fp", pred-tp), ("fn", gt-tp), ("predicted_instances", pred), ("gt_instances", gt)):
            require(aggregate[key].get(name) == value, f"Aggregate detection {key}/{name} differs from patient totals")
        mean = iou/gt if gt else 0.
        require(math.isclose(aggregate[key].get("mean_iou_with_misses_zero", float("nan")), mean, abs_tol=1e-10), "Aggregate Mean IoU differs from patient totals")
    population = (digest, rule, keys, truth, classes, macro, det_keys, payload.get("detection_scope", {}).get("scope"))
    return layout, population, units, np.asarray(gt_counts)


@dataclass
class Study:
    seeds: list[int]
    reference: str
    layout: MetricLayout
    patient_ids: tuple[str, ...]
    units: dict[str, np.ndarray]
    sources: list[dict]
    population: tuple
    comparison_mode: str = "models"
    arms: dict = field(default_factory=dict)


def _aggregation_arms(spec: dict, models: dict, reference: str) -> tuple[str, dict]:
    mode = spec.get("comparison_mode", "models")
    require(mode in ("models", "aggregation", "aggregation_table"), "Unknown comparison_mode")
    if mode == "models":
        require("arms" not in spec, "Aggregation arms require an explicit aggregation comparison_mode")
        return mode, {}
    arms = spec.get("arms")
    require(isinstance(arms, dict) and set(arms) == set(models), "Declare every aggregation arm explicitly")
    combinations = set()
    for arm in arms.values():
        require(isinstance(arm, dict) and set(arm) == {"model_name", "aggregation_strategy"}, "Invalid aggregation arm declaration")
        name, strategy = arm["model_name"], arm["aggregation_strategy"]
        require(isinstance(name, str) and name and name == name.strip(), "Invalid aggregation arm model_name")
        require(strategy in AGGREGATION_RULES, "Unknown aggregation strategy")
        require((name, strategy) not in combinations, "Duplicate model/strategy arm")
        combinations.add((name, strategy))
    names = {name for name, _ in combinations}
    require(mode != "aggregation" or len(names) == 1, "Aggregation mode requires the same model across strategies")
    require(all((name, MAJORITY_NLL_TIEBREAK) in combinations for name in names), "Every model requires its default aggregation anchor")
    require(arms[reference]["aggregation_strategy"] == MAJORITY_NLL_TIEBREAK, "Aggregation reference must use the default strategy")
    return mode, arms


def validate_aggregation_pair(anchor: dict, alternative: dict) -> None:
    """Validate a readout change against its same-model, same-seed default."""
    require(validate_run_identity(anchor) == validate_run_identity(alternative), "Aggregation changes must retain the same model and seed")
    for payload in (anchor, alternative):
        source = payload.get("sources", {}).get("predictions", {})
        digest = source.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "Aggregation comparisons require prediction source SHA256 identities")
    require(anchor["sources"]["predictions"]["sha256"] == alternative["sources"]["predictions"]["sha256"],
            "Aggregation prediction source identities differ")
    require(anchor["evaluation_identity"]["manifest_sha256"] == alternative["evaluation_identity"]["manifest_sha256"],
            "Aggregation manifest identities differ")
    # Saved group records include completion NLL and manifest input order.
    require(isinstance(anchor.get("input_groups"), list) and anchor["input_groups"] == alternative.get("input_groups"),
            "Aggregation input groups, predictions or completion NLL differ")
    for key in ("coverage", "detection_scope", "detection"):
        require(key in anchor and anchor[key] == alternative.get(key), f"Aggregation changes cannot alter {key}")
    require(indexed_rows(anchor.get("patient_detection"), "patient_detection")
            == indexed_rows(alternative.get("patient_detection"), "patient_detection"),
            "Aggregation changes cannot alter patient detection counts")
    left, right = [indexed_rows(payload.get("patients"), "patients") for payload in (anchor, alternative)]
    require(set(left) == set(right), "Aggregation changes cannot alter the patient denominator")
    for patient in left:
        require(all(key in left[patient] and left[patient][key] == right[patient].get(key)
                    for key in ("truth", "input_groups", "valid_votes", "abstained")),
                "Aggregation changes cannot alter patient truth, inputs, eligible votes or coverage")


def load_study(spec_path: Path) -> Study:
    spec = _read_json(spec_path.read_text(encoding="utf-8"))
    require(isinstance(spec, dict) and spec.get("schema_version") == 1, "Spec requires schema_version=1")
    seeds = spec.get("seeds")
    require(isinstance(seeds, list) and len(seeds) == 5 and all(type(s) is int for s in seeds) and len(set(seeds)) == 5, "Exactly five distinct integer seeds are required")
    models = spec.get("models")
    require(isinstance(models, dict) and len(models) >= 2, "At least two named models are required")
    require(all(isinstance(name, str) and name and name == name.strip() for name in models), "Invalid model name")
    reference = spec.get("reference")
    require(reference in models, "Reference model is absent")
    mode, arms = _aggregation_arms(spec, models, reference)
    baseline = baseline_gt = layout = None
    units, sources = {}, []
    used_files = set()
    payloads = {}
    for name, runs in models.items():
        require(isinstance(runs, list) and len(runs) == 5, f"Model {name} needs exactly five runs")
        by_seed = {}
        for run in runs:
            require(isinstance(run, dict) and type(run.get("seed")) is int, f"Invalid run seed for {name}")
            seed = run["seed"]
            require(seed in seeds and seed not in by_seed, f"Duplicate or mismatched seed for {name}: {seed}")
            require(isinstance(run.get("metrics"), str) and bool(run["metrics"]), "Missing metrics path")
            path = (spec_path.parent/run["metrics"]).resolve()
            require(path not in used_files, "A metrics file cannot represent more than one run")
            used_files.add(path)
            raw = path.read_bytes()
            payload = _read_json(raw.decode("utf-8"))
            require(isinstance(payload, dict), f"Canonical metrics must be an object: {path}")
            declared_model = arms[name]["model_name"] if arms else name
            require(validate_run_identity(payload) == {"model_name": declared_model, "seed": seed}, f"Run identity disagrees with spec: {path}")
            if arms:
                require(payload.get("aggregation_strategy") == arms[name]["aggregation_strategy"], f"Aggregation strategy disagrees with declared arm: {path}")
            current_layout, population, current, gt = extract_run(payload)
            if baseline is None:
                baseline, baseline_gt, layout = population, gt, current_layout
            # Ordinary model comparisons retain the full identity check. Only
            # explicitly declared strategy arms may differ at the rule slot.
            comparable = population if not arms else population[:1] + population[2:]
            expected = baseline if not arms else baseline[:1] + baseline[2:]
            require(comparable == expected, f"Patient/GT/class/manifest/evaluation identity mismatch: {path}")
            require(np.array_equal(gt, baseline_gt), f"Per-patient lesion GT counts differ: {path}")
            by_seed[seed] = current
            payloads[name, seed] = payload
            sources.append({"model": name, "seed": seed, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        units[name] = np.stack([by_seed[seed] for seed in seeds])
    if arms:
        anchors = {arm["model_name"]: name for name, arm in arms.items()
                   if arm["aggregation_strategy"] == MAJORITY_NLL_TIEBREAK}
        for name, arm in arms.items():
            for seed in seeds:
                validate_aggregation_pair(payloads[anchors[arm["model_name"]], seed], payloads[name, seed])
    return Study(seeds, reference, layout, baseline[2], units, sources, baseline, mode, arms)


def weighted_totals(units: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weights axis is patient, shared across every seed, not seed×patient."""
    return np.einsum("bn,snf->bsf", weights, units, optimize=True)


def permutation_differences(layout: MetricLayout, reference: np.ndarray, comparator: np.ndarray, masks: np.ndarray) -> np.ndarray:
    require(masks.ndim == 2 and masks.shape[1] == reference.shape[1], "Swap masks must have shape (replicates, patients)")
    moved = weighted_totals(comparator-reference, masks)
    left = reference.sum(axis=1)[None, ...] + moved
    right = comparator.sum(axis=1)[None, ...] - moved
    return (layout.metrics(left)-layout.metrics(right)).mean(axis=1)


def paired_statistics(layout: MetricLayout, reference: np.ndarray, comparator: np.ndarray, *, permutations: int = 100000, bootstrap: int = 10000, random_seed: int = 2026, exact: bool = False, batch_size: int = 256) -> dict:
    require(reference.shape == comparator.shape and reference.ndim == 3 and reference.shape[0] == 5, "Paired arrays require the same five seeds and patients")
    require(reference.shape[1] > 0 and reference.shape[2] == layout.width, "Empty or invalid statistical units")
    require(type(permutations) is int and permutations > 0 and type(bootstrap) is int and bootstrap > 0, "Positive permutation/bootstrap repeats required")
    require(type(batch_size) is int and batch_size > 0, "Positive batch size required")
    n = reference.shape[1]
    require(not exact or n <= 16, "Exact enumeration is limited to 16 patients")
    ref_values, cmp_values = layout.metrics(reference.sum(axis=1)), layout.metrics(comparator.sum(axis=1))
    observed = (ref_values-cmp_values).mean(axis=0)
    streams = np.random.SeedSequence(random_seed).spawn(2)
    perm_rng, boot_rng = [np.random.Generator(np.random.PCG64(stream)) for stream in streams]
    repeats = 2**n if exact else permutations
    extreme = np.zeros(len(layout.names), dtype=np.int64)
    for start in range(0, repeats, batch_size):
        size = min(batch_size, repeats-start)
        masks = (((np.arange(start, start+size, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(bool)
                 if exact else perm_rng.integers(0, 2, (size, n), dtype=np.int8).astype(bool))
        difference = permutation_differences(layout, reference, comparator, masks)
        extreme += (np.abs(difference) >= np.abs(observed)[None, :]-1e-12).sum(axis=0)
    p = extreme/repeats if exact else (extreme+1)/(repeats+1)
    boot_ref, boot_cmp = [], []
    for start in range(0, bootstrap, batch_size):
        size = min(batch_size, bootstrap-start)
        # Multinomial patient multiplicities are equivalent to drawing N IDs
        # with replacement, synchronized over models, seeds and all endpoints.
        weights = boot_rng.multinomial(n, np.full(n, 1/n), size=size)
        boot_ref.append(layout.metrics(weighted_totals(reference, weights)).mean(axis=1))
        boot_cmp.append(layout.metrics(weighted_totals(comparator, weights)).mean(axis=1))
    boot_ref, boot_cmp = np.concatenate(boot_ref), np.concatenate(boot_cmp)
    intervals = [np.quantile(values, [.025, .975], axis=0, method="linear") for values in (boot_ref, boot_cmp, boot_ref-boot_cmp)]
    metrics = {}
    for j, name in enumerate(layout.names):
        metrics[name] = {
            "reference_values": ref_values[:, j].tolist(), "comparator_values": cmp_values[:, j].tolist(),
            "reference_mean": float(ref_values[:, j].mean()), "reference_sample_sd": float(ref_values[:, j].std(ddof=1)),
            "comparator_mean": float(cmp_values[:, j].mean()), "comparator_sample_sd": float(cmp_values[:, j].std(ddof=1)),
            "reference_minus_comparator": float(observed[j]), "p_raw": float(p[j]), "extreme_permutations": int(extreme[j]),
            "reference_patient_bootstrap_95_ci": intervals[0][:, j].tolist(),
            "comparator_patient_bootstrap_95_ci": intervals[1][:, j].tolist(),
            "difference_patient_bootstrap_95_ci": intervals[2][:, j].tolist(),
        }
    return {"permutation_repeats": repeats, "permutation_method": "exact_enumeration" if exact else "monte_carlo_plus_one", "bootstrap_repeats": bootstrap, "metrics": metrics}


def benjamini_hochberg(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    require(p.ndim == 1 and len(p) > 0 and bool(np.all(np.isfinite(p))) and bool(np.all((p >= 0) & (p <= 1))), "Invalid p-value family")
    order = np.argsort(p, kind="stable")
    q = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(q); result[order] = np.minimum(q, 1)
    return result.tolist()


def analyze(study: Study, *, adjustment: str = "none", **kwargs) -> dict:
    require(adjustment in ("none", "bh"), "Unknown multiplicity adjustment")
    comparisons = []
    for name, units in study.units.items():
        if name != study.reference:
            comparisons.append({"reference": study.reference, "comparator": name,
                **paired_statistics(study.layout, study.units[study.reference], units, **kwargs)})
    family = [metric for pair in comparisons for metric in pair["metrics"].values()]
    q = benjamini_hochberg([row["p_raw"] for row in family]) if adjustment == "bh" else [None]*len(family)
    for row, value in zip(family, q):
        row["q_bh"] = value
    return {"schema_version": 1, "analysis": "five_fixed_seed_patient_paired", "seeds": study.seeds,
        "patients": len(study.patient_ids), "patient_ids": list(study.patient_ids),
        "classes": list(study.layout.classes), "macro_classes": list(study.layout.macro_classes),
        "evaluation_identity": {"manifest_sha256": study.population[0],
            **({"aggregation_rules": {name: AGGREGATION_RULES[arm["aggregation_strategy"]] for name, arm in study.arms.items()}}
               if study.arms else {"aggregation_rule": study.population[1]})},
        "comparison_mode": study.comparison_mode, "arms": study.arms,
        "statistic": "mean over five within-seed reference-minus-comparator metric differences",
        "resampling_unit": "patient; entire five-seed vector synchronized across endpoints and models",
        "interval": "paired patient percentile bootstrap; linear 2.5/97.5 percentiles; fixed seeds",
        "adjustment": adjustment, "family": "all comparator-by-metric rows in this output; no omitted rows", "family_size": len(family),
        "random_seed": kwargs.get("random_seed", 2026), "sources": study.sources, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--adjustment", choices=("none", "bh"), default="none")
    parser.add_argument("--exact-permutations", action="store_true", help="Enumerate all swaps for at most 16 patients; no Monte Carlo plus-one correction")
    args = parser.parse_args()
    try:
        paths = [args.output_dir/name for name in ("joint_seed_statistics.json", "joint_seed_statistics.csv")]
        require(not any(path.exists() for path in paths), "Output files already exist; use a fresh directory")
        study = load_study(args.spec.resolve())
        result = analyze(study, permutations=args.permutations, bootstrap=args.bootstrap, random_seed=args.random_seed, exact=args.exact_permutations, adjustment=args.adjustment)
        result["spec"] = {"path": str(args.spec.resolve()), "sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest()}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        fields = ["reference", "comparator", "metric", *next(iter(result["comparisons"][0]["metrics"].values())).keys()]
        with paths[1].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for pair in result["comparisons"]:
                for metric, values in pair["metrics"].items():
                    writer.writerow({"reference": pair["reference"], "comparator": pair["comparator"], "metric": metric,
                        **{key: json.dumps(value) if isinstance(value, list) else value for key, value in values.items()}})
    except (OSError, ValueError, SchemaError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "patients": result["patients"], "family_size": result["family_size"]}))


if __name__ == "__main__":
    main()
