#!/usr/bin/env python3
"""Generate paper tables from five-seed canonical patient exports and paired tests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.summarize_seed_results import summarize_runs
from evaluation.evaluate_paper_protocol import _read_json
from evaluation.joint_seed_statistics import extract_run, require, validate_run_identity, validate_aggregation_pair
from matching.protocols import CLASS_AGNOSTIC, GEOMETRY_FIRST_CLASS_AWARE, PROTOCOLS, SAME_CLASS_PRIORITY


CLASSIFICATION = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
PRF = ("precision", "recall", "f1")
PROTOCOL_LABELS = {CLASS_AGNOSTIC: "CA", GEOMETRY_FIRST_CLASS_AWARE: "GFCA", SAME_CLASS_PRIORITY: "SCF"}


def read_json(path):
    return _read_json(Path(path).read_text(encoding="utf-8"))


def labels(value, name):
    require(isinstance(value, list) and bool(value), f"{name} must be a nonempty list")
    require(all(isinstance(x, str) and x and x == x.strip() for x in value), f"Invalid {name}")
    require(len(value) == len(set(value)), f"Duplicate {name}")
    return value


def number(value, name, low=-math.inf, high=math.inf):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high, f"Invalid {name}")
    return value


def close(actual, expected, name):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and bool(np.all(np.isfinite(a))) and bool(np.allclose(a, b, rtol=0, atol=1e-10)), f"Statistics disagree with canonical exports: {name}")


@dataclass
class Cohort:
    spec: dict
    seeds: list
    payloads: dict
    sources: list
    layout: object
    population: tuple
    gt: np.ndarray
    summaries: dict
    raw_values: dict


def load_cohorts(spec_path: Path) -> tuple[dict, list[Cohort]]:
    """Validate the complete model × seed population before writing any output."""
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    require(isinstance(spec, dict) and spec.get("schema_version") == 1, "Spec requires schema_version=1")
    seeds = spec.get("seeds")
    require(isinstance(seeds, list) and len(seeds) == 5 and all(type(s) is int for s in seeds) and len(set(seeds)) == 5, "Exactly five distinct integer seeds are required")
    cohorts = spec.get("cohorts")
    require(isinstance(cohorts, list) and bool(cohorts), "An ordered cohorts list is required")
    used_ids, used_paths, result = set(), set(), []
    for item in cohorts:
        require(isinstance(item, dict), "Invalid cohort spec")
        cid = item.get("id")
        require(isinstance(cid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", cid) is not None and cid not in used_ids, "Cohort ids must be distinct filename-safe names")
        used_ids.add(cid)
        labels([item.get("label")], "cohort label")
        classes, macro = labels(item.get("classes"), "classes"), labels(item.get("macro_classes"), "macro_classes")
        order = labels(item.get("model_order"), "model_order")
        models = item.get("models")
        require(isinstance(models, dict) and set(models) == set(order), "model_order must include every model exactly once")
        aliases = item.get("class_labels", {label: label for label in classes})
        require(isinstance(aliases, dict) and set(aliases) == set(classes), "class_labels must cover the full output vocabulary")
        labels(list(aliases.values()), "display class labels")
        require("Abstention" not in aliases.values(), "Abstention is reserved for undefined predictions")
        rounding = item.get("classification_seed_decimals", {})
        require(isinstance(rounding, dict) and set(rounding).issubset(models), "Unknown model in classification_seed_decimals")
        require(all(value is None or (type(value) is int and value == 4) for value in rounding.values()), "classification_seed_decimals supports explicit 4 or null")
        payloads, sources, summaries, raw_values = {}, [], {}, {}
        baseline = baseline_gt = layout = None
        for model in order:
            runs = models[model]
            require(isinstance(runs, list) and len(runs) == 5, f"{model} requires five runs")
            by_seed = {}
            units_by_seed = {}
            for run in runs:
                require(isinstance(run, dict) and type(run.get("seed")) is int and run["seed"] in seeds and run["seed"] not in by_seed, f"Duplicate or incompatible seed for {model}")
                require(isinstance(run.get("metrics"), str) and bool(run["metrics"]), "Missing metrics path")
                path = (spec_path.parent / run["metrics"]).resolve()
                require(path not in used_paths, "A metrics file cannot represent multiple runs or cohorts")
                used_paths.add(path)
                raw = path.read_bytes()
                payload = _read_json(raw.decode("utf-8"))
                require(validate_run_identity(payload) == {"model_name": model, "seed": run["seed"]}, f"Run identity mismatch: {path}")
                current_layout, population, units, gt = extract_run(payload)
                require(list(current_layout.classes) == classes and list(current_layout.macro_classes) == macro, f"Explicit class order disagrees with canonical export: {path}")
                if baseline is None:
                    baseline, baseline_gt, layout = population, gt, current_layout
                require(population == baseline and np.array_equal(gt, baseline_gt), f"Patient/GT/manifest/class/aggregation/threshold identity mismatch: {path}")
                by_seed[run["seed"]] = payload
                units_by_seed[run["seed"]] = units
                sources.append({"model": model, "seed": run["seed"], "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
            payloads[model] = [by_seed[seed] for seed in seeds]
            raw_values[model] = layout.metrics(np.stack([units_by_seed[seed].sum(axis=0) for seed in seeds]))
            kwargs = {"classification_seed_decimals": 4} if rounding.get(model) == 4 else {}
            summaries[model] = summarize_runs(payloads[model], seeds, macro, **kwargs)["summary"]
        result.append(Cohort(item, seeds, payloads, sources, layout, baseline, baseline_gt, summaries, raw_values))
    return spec, result


def significance(p):
    number(p, "raw p-value", 0, 1)
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def validate_statistics(payload, cohort, *, reference=None, arms=None):
    """Bind every inferential row to the exact runs, order, and signed contrast."""
    reference = reference or cohort.spec.get("reference_model")
    require(reference in cohort.payloads, "reference_model must name a configured model")
    require(payload.get("schema_version") == 1 and payload.get("analysis") == "five_fixed_seed_patient_paired", "Require joint_seed_statistics JSON")
    require(type(payload.get("random_seed")) is int and payload["random_seed"] >= 0, "Missing statistical random seed")
    require(payload.get("adjustment") in ("none", "bh"), "Unknown statistical multiplicity adjustment")
    require(payload.get("seeds") == cohort.seeds and payload.get("patient_ids") == list(cohort.population[2]) and payload.get("patients") == len(cohort.population[2]), "Statistical seeds/patient identity differs")
    require(payload.get("classes") == list(cohort.layout.classes) and payload.get("macro_classes") == list(cohort.layout.macro_classes), "Statistical class order differs")
    identity = {"manifest_sha256": cohort.population[0]}
    if arms is None:
        identity["aggregation_rule"] = cohort.population[1]
        require(payload.get("comparison_mode", "models") == "models" and not payload.get("arms"), "Model table requires model-comparison statistics")
    else:
        identity["aggregation_rules"] = {arm: cohort.payloads[arm][0]["evaluation_identity"]["aggregation_rule"] for arm in arms}
        mode = "aggregation_table" if len({arm["model_name"] for arm in arms.values()}) > 1 else "aggregation"
        require(payload.get("comparison_mode") == mode and payload.get("arms") == arms, "Aggregation statistical arms differ")
    require(payload.get("evaluation_identity") == identity, "Statistical manifest/aggregation identity differs")
    expected_sources = {(x["model"], x["seed"]): x["sha256"] for x in cohort.sources}
    actual_sources = {}
    require(isinstance(payload.get("sources"), list), "Statistical sources are missing")
    for source in payload["sources"]:
        key = (source.get("model"), source.get("seed"))
        require(key not in actual_sources, "Duplicate statistical source")
        actual_sources[key] = source.get("sha256")
    require(actual_sources == expected_sources, "Statistics source hashes do not match configured canonical runs")
    comparisons = payload.get("comparisons")
    require(isinstance(comparisons, list), "Missing comparisons")
    pairs = {}
    for pair in comparisons:
        model = pair.get("comparator")
        require(pair.get("reference") == reference and model in cohort.payloads and model != reference and model not in pairs, "Missing, duplicate, or reversed comparison identity")
        metrics = pair.get("metrics")
        require(isinstance(metrics, dict) and set(metrics) == set(cohort.layout.names), "Missing or unexpected comparison endpoint")
        repeats = pair.get("permutation_repeats")
        require(type(repeats) is int and repeats > 0, "Invalid permutation count")
        require(type(pair.get("bootstrap_repeats")) is int and pair["bootstrap_repeats"] > 0, "Invalid bootstrap count")
        for j, endpoint in enumerate(cohort.layout.names):
            row = metrics[endpoint]
            ref, cmp = cohort.raw_values[reference][:, j], cohort.raw_values[model][:, j]
            for side, values in (("reference", ref), ("comparator", cmp)):
                close(row.get(side + "_values"), values, endpoint + "/" + side)
                close(row.get(side + "_mean"), values.mean(), endpoint + "/mean")
                close(row.get(side + "_sample_sd"), values.std(ddof=1), endpoint + "/sample_sd")
            close(row.get("reference_minus_comparator"), (ref-cmp).mean(), endpoint + "/signed difference")
            p = number(row.get("p_raw"), "raw p-value", 0, 1)
            extreme = row.get("extreme_permutations")
            require(type(extreme) is int and 0 <= extreme <= repeats, "Invalid extreme permutation count")
            method = pair.get("permutation_method")
            require(method in ("exact_enumeration", "monte_carlo_plus_one"), "Unknown permutation method")
            expected_p = extreme/repeats if method == "exact_enumeration" else (extreme+1)/(repeats+1)
            require(p == expected_p, "Raw p-value/count identity differs")
            for key, low in (("reference_patient_bootstrap_95_ci", 0), ("comparator_patient_bootstrap_95_ci", 0), ("difference_patient_bootstrap_95_ci", -1)):
                bounds = row.get(key)
                require(isinstance(bounds, list) and len(bounds) == 2, "Missing bootstrap interval")
                number(bounds[0], key, low, 1); number(bounds[1], key, low, 1)
                require(bounds[0] <= bounds[1], "Reversed bootstrap interval")
        pairs[model] = pair
    require(set(pairs) == set(cohort.payloads)-{reference}, "Missing required model comparisons")
    require(payload.get("family_size") == sum(len(pair["metrics"]) for pair in pairs.values()), "Statistical family size differs")
    return pairs


def statistical_settings(payload):
    return {"random_seed": payload.get("random_seed"), "analysis": payload["analysis"],
            "adjustment": payload.get("adjustment"), "family_size": payload["family_size"],
            "statistic": payload.get("statistic"), "resampling_unit": payload.get("resampling_unit"), "interval": payload.get("interval"),
            "comparisons": [{key: pair[key] for key in ("reference", "comparator", "permutation_repeats", "permutation_method", "bootstrap_repeats")} for pair in payload["comparisons"]]}


def tex_escape(value):
    return "".join({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(c, c) for c in str(value))


def metric_cell(summary, mark, decimals, *, best=False):
    mean = f"{summary['mean']:.{decimals}f}"
    if best:
        mean = r"\mathbf{" + mean + "}"
    suffix = "" if mark is None else "^{" + (r"\mathrm{ns}" if mark == "ns" else mark) + "}"
    return "$" + mean + r"\,\pm\," + f"{summary['sd']:.{decimals}f}" + suffix + "$"


def threshold_name(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def det_endpoint(protocol, threshold, metric):
    return f"detection:{protocol}:iou={threshold_name(threshold)}:{metric}"


def table_block(headers, rows, note, *, multipage=False):
    environment = "longtable" if multipage else "tabular"
    header = " & ".join(map(tex_escape, headers)) + r" \\"
    lines = ["% Generated from canonical per-patient exports.", r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             "\\begin{" + environment + "}{@{}" + "l"*len(headers) + "@{}}", r"\toprule", header, r"\midrule"]
    if multipage:
        lines += [r"\endfirsthead", r"\toprule", header, r"\midrule", r"\endhead", r"\midrule",
                  r"\multicolumn{" + str(len(headers)) + r"}{r}{Continued on next page} \\", r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    if not multipage:
        lines.append(r"\bottomrule")
    lines += ["\\end{" + environment + "}", "", r"\par\smallskip\noindent\footnotesize " + note, r"\par\endgroup", ""]
    return "\n".join(lines)


def export_cohort(cohort, joint, decimals):
    cfg = cohort.spec
    pairs = validate_statistics(joint, cohort)
    views = cfg.get("views")
    require(isinstance(views, dict) and set(views) == {"overview", "per_class", "thresholds"}, "views must explicitly provide overview, per_class, and thresholds model orders")
    for view, models in views.items():
        labels(models, view + " model order")
        require(set(models).issubset(cohort.payloads), f"Unknown {view} model")
    threshold_order = cfg.get("threshold_order")
    require(isinstance(threshold_order, list) and bool(threshold_order) and len(threshold_order) == len(set(threshold_order)), "Explicit distinct threshold_order is required")
    for value in threshold_order:
        number(value, "IoU threshold", 0, 1)
    require(set(threshold_order) == {key[1] for key in cohort.layout.detection_keys}, "threshold_order must include every canonical threshold")
    protocol, threshold = cfg.get("primary_protocol"), cfg.get("primary_threshold")
    number(threshold, "primary_threshold", 0, 1)
    require(protocol in PROTOCOLS and threshold in threshold_order, "Explicit valid primary_protocol/primary_threshold required")
    aliases = cfg.get("class_labels", {label: label for label in cfg["classes"]})
    support = {label: cohort.population[3].count(label) for label in cfg["macro_classes"]}
    records = []
    for model in cfg["model_order"]:
        for endpoint, summary in cohort.summaries[model].items():
            infer = pairs[model]["metrics"][endpoint] if model in pairs else None
            records.append({"cohort": cfg["id"], "cohort_label": cfg["label"], "model": model, "metric": endpoint, "seeds": cohort.seeds,
                "seed_values": summary["values"], "unrounded_seed_values": cohort.raw_values[model][:, cohort.layout.names.index(endpoint)].tolist(),
                "mean": summary["mean"], "sample_sd": summary["sd"],
                "classification_seed_decimals": cfg.get("classification_seed_decimals", {}).get(model),
                "reference": cfg["reference_model"], "stars": significance(infer["p_raw"]) if infer else None,
                "comparison": infer})
    def cells(model, endpoints, models, *, rank=True):
        out = []
        for endpoint in endpoints:
            mark = significance(pairs[model]["metrics"][endpoint]["p_raw"]) if model in pairs else None
            best = rank and cohort.summaries[model][endpoint]["mean"] == max(cohort.summaries[m][endpoint]["mean"] for m in models)
            out.append(metric_cell(cohort.summaries[model][endpoint], mark, decimals, best=best))
        return out
    note = (tex_escape(cfg["label"]) + ": five seeds; mean $\\pm$ sample SD. All manifest patients remain in classification denominators, including abstentions. "
            "Superscripts compare each row with " + tex_escape(cfg["reference_model"]) + ". Raw two-sided patient-paired $p$: $***<0.001$, $**<0.01$, $*<0.05$, ns $\\geq0.05$. "
            "Marks indicate differences in either direction; the reference has no self-comparison. Bold denotes the highest unrounded displayed-summary mean within each table block. "
            "Full-precision signed reference-minus-comparator differences, raw $p$ values and patient-bootstrap intervals accompany these tables in JSON/CSV.")
    if any(value == 4 for value in cfg.get("classification_seed_decimals", {}).values()):
        note += " For explicitly configured models, classification summaries aggregate four-decimal seed metrics; detection and inference use unrounded values."
    endpoints = [det_endpoint(protocol, threshold, metric) for metric in PRF] + [f"detection:{protocol}:mean_iou"] + ["classification:" + x for x in CLASSIFICATION]
    names = [f"{PROTOCOL_LABELS[protocol]} {m}@{threshold_name(threshold)}" for m in ("P", "R", "F1")] + [f"{PROTOCOL_LABELS[protocol]} Mean IoU", "Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]
    models = views["overview"]
    tables = {"overview": table_block(["Model", *names], [[tex_escape(model), *cells(model, endpoints, models)] for model in models], note)}
    models = views["per_class"]
    rows = [[tex_escape(model), tex_escape(aliases[label]), str(support[label]), *cells(model, [f"class:{label}:{m}" for m in PRF], models)] for label in cfg["macro_classes"] for model in models]
    tables["per_class"] = table_block(["Model", "Class", "Test patients", "Precision", "Recall", "F1"], rows, note, multipage=True)
    models = views["thresholds"]
    rows = []
    for mode in (CLASS_AGNOSTIC, GEOMETRY_FIRST_CLASS_AWARE, SAME_CLASS_PRIORITY):
        for value in threshold_order:
            endpoints = [det_endpoint(mode, value, m) for m in PRF]
            rows += [[PROTOCOL_LABELS[mode], "$>0$" if value == 0 else "$\\geq " + threshold_name(value) + "$", tex_escape(model), *cells(model, endpoints, models)] for model in models]
    tables["thresholds"] = table_block(["Protocol", "IoU", "Model", "Precision", "Recall", "F1"], rows, note + " IoU zero denotes strictly positive overlap. CA, GFCA and SCF preserve their distinct assignments.", multipage=True)
    factorial = cfg.get("factorial")
    if factorial is not None:
        factors = labels(factorial.get("factors"), "factor names")
        require(len(factors) == 3, "Factorial requires three factors")
        frows = factorial.get("rows")
        require(isinstance(frows, list) and len(frows) == 8, "Factorial requires all eight configurations")
        bitsets, fmodels = set(), []
        for row in frows:
            bits = row.get("bits")
            require(isinstance(bits, list) and len(bits) == 3 and all(type(b) is int and b in (0, 1) for b in bits), "Factorial bits must be three binary integers")
            bitsets.add(tuple(bits)); fmodels.append(row.get("model"))
        require(bitsets == set(itertools.product((0, 1), repeat=3)) and len(set(fmodels)) == 8 and set(fmodels).issubset(cohort.payloads), "Duplicate or missing factorial configuration/model")
        uniform = factorial.get("uniform_model")
        require(uniform in cohort.payloads and uniform not in fmodels, "Uniform must identify a separate model outside the eight factorial configurations")
        endpoints = [det_endpoint(protocol, threshold, m) for m in PRF] + [f"detection:{protocol}:mean_iou"] + ["classification:" + x for x in CLASSIFICATION]
        tables["factorial"] = "\n".join(
            r"\noindent\textbf{" + title + "}\n" + table_block(
                ["Configuration", *factors, *names[span]],
                [[tex_escape(row["model"]), *map(str, row["bits"]), *cells(row["model"], endpoints[span], fmodels)] for row in frows], note)
            for title, span in (("Lesion-level detection", slice(0, 4)), ("Patient-level classification", slice(4, 8))))
        tables["uniform"] = table_block(["Configuration", *names], [[tex_escape(uniform), *cells(uniform, endpoints, [uniform], rank=False)]], note + " Uniform is a severity-weight comparison separate from the three binary component factors.")
    return {"id": cfg["id"], "label": cfg["label"], "patients": len(cohort.population[2]),
            "classes": cfg["classes"], "macro_classes": cfg["macro_classes"], "class_labels": aliases,
            "seeds": cohort.seeds, "model_order": cfg["model_order"], "class_support": support,
            "primary_protocol": protocol, "primary_threshold": threshold, "threshold_order": threshold_order,
            "evaluation_identity": {"manifest_sha256": cohort.population[0], "aggregation_rule": cohort.population[1]},
            "classification_seed_rounding": "decimal_half_even_for_explicit_four_decimal_models",
            "classification_seed_decimals": cfg.get("classification_seed_decimals", {}),
            "statistical_settings": statistical_settings(joint), "sources": cohort.sources, "records": records}, tables


def write_csv(path, records):
    fields = ["cohort", "cohort_label", "model", "metric", "seeds", "seed_values", "unrounded_seed_values", "mean", "sample_sd", "classification_seed_decimals", "reference", "stars", "p_raw", "reference_minus_comparator", "reference_patient_bootstrap_95_ci", "comparator_patient_bootstrap_95_ci", "difference_patient_bootstrap_95_ci", "comparison"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for record in records:
            infer = record.get("comparison") or {}
            row = {**record, **{key: infer.get(key) for key in fields if key not in record}}
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def export_aggregation(cohort, options, base_dir, decimals):
    """Read all three saved readouts of the same model/seed prediction source."""
    from aggregation.patient import BEST_LIKELIHOOD_INPUT, LIKELIHOOD_WEIGHTED_VOTE, MAJORITY_NLL_TIEBREAK
    from evaluation.evaluate_paper_protocol import AGGREGATION_RULES

    require(isinstance(options, dict), "aggregation_comparison requires path, model_order and strategy_order")
    path = options.get("path")
    require(isinstance(path, str) and bool(path), "Missing aggregation comparison path")
    path = (base_dir / path).resolve()
    report = read_json(path)
    models = labels(options.get("model_order"), "aggregation model_order")
    strategies = labels(options.get("strategy_order"), "aggregation strategy_order")
    require(set(models).issubset(cohort.payloads) and cohort.spec["reference_model"] in models, "Aggregation models must include the cohort reference and use configured runs")
    require(set(strategies) == set(AGGREGATION_RULES), "Aggregation table requires all three strategies")
    require(report.get("schema_version") == 1 and report.get("cohort") == cohort.spec["id"] and report.get("seeds") == cohort.seeds, "Aggregation report cohort/seed identity differs")
    require(report.get("reference_model") == cohort.spec["reference_model"], "Aggregation report reference model differs")
    declared = report.get("arms")
    require(isinstance(declared, list) and len(declared) == len(models)*3, "Aggregation report requires every model × strategy arm")
    arm_by_pair, declarations, payloads, raw_values, summaries, sources = {}, {}, {}, {}, {}, []
    used_paths = set()
    for arm in declared:
        key, model, strategy = arm.get("arm"), arm.get("model_name"), arm.get("aggregation_strategy")
        require(isinstance(key, str) and bool(key) and key not in declarations and model in models and strategy in strategies, "Duplicate or unknown aggregation arm")
        require((model, strategy) not in arm_by_pair, "Duplicate aggregation model/strategy")
        arm_by_pair[model, strategy] = key
        declarations[key] = {"model_name": model, "aggregation_strategy": strategy}
        runs = arm.get("runs")
        require(isinstance(runs, list) and len(runs) == 5, "Every aggregation arm requires five runs")
        by_seed, units = {}, {}
        for run in runs:
            seed = run.get("seed")
            require(type(seed) is int and seed in cohort.seeds and seed not in by_seed, "Duplicate or incompatible aggregation seed")
            require(isinstance(run.get("metrics"), str) and bool(run["metrics"]), "Missing aggregation metrics path")
            metrics_path = (path.parent / run["metrics"]).resolve()
            require(metrics_path not in used_paths, "Duplicate aggregation metrics file")
            used_paths.add(metrics_path)
            raw = metrics_path.read_bytes()
            payload = _read_json(raw.decode("utf-8"))
            require(validate_run_identity(payload) == {"model_name": model, "seed": seed} and payload.get("aggregation_strategy") == strategy, "Aggregation run identity disagrees with report")
            layout, population, current, gt = extract_run(payload)
            require(population[:1] + population[2:] == cohort.population[:1] + cohort.population[2:] and np.array_equal(gt, cohort.gt), "Aggregation population/GT/layout differs from the cohort")
            by_seed[seed], units[seed] = payload, current
            sources.append({"model": key, "seed": seed, "path": str(metrics_path), "sha256": hashlib.sha256(raw).hexdigest()})
        payloads[key] = [by_seed[seed] for seed in cohort.seeds]
        raw_values[key] = layout.metrics(np.stack([units[seed].sum(axis=0) for seed in cohort.seeds]))
        kwargs = {"classification_seed_decimals": 4} if cohort.spec.get("classification_seed_decimals", {}).get(model) == 4 else {}
        summaries[key] = summarize_runs(payloads[key], cohort.seeds, list(layout.macro_classes), **kwargs)["summary"]
        supplied = arm.get("summary")
        require(isinstance(supplied, dict), "Missing aggregation summary")
        for metric in ("classification:accuracy", "classification:macro_f1"):
            values = raw_values[key][:, layout.names.index(metric)]
            require(metric in supplied, "Aggregation summary lacks requested endpoint")
            close(supplied[metric].get("values"), values, "aggregation summary seed values")
            close(supplied[metric].get("mean"), values.mean(), "aggregation summary mean")
            close(supplied[metric].get("sample_sd"), values.std(ddof=1), "aggregation summary sample SD")
    require(set(arm_by_pair) == set(itertools.product(models, strategies)), "Missing aggregation strategy arm")
    reference = arm_by_pair[cohort.spec["reference_model"], MAJORITY_NLL_TIEBREAK]
    require(report.get("reference_arm") == reference, "Aggregation reference must be the configured model's default readout")
    extra = {}
    for model in models:
        anchor = arm_by_pair[model, MAJORITY_NLL_TIEBREAK]
        for i, seed in enumerate(cohort.seeds):
            # Bind the regenerated default readout to the supplied ordinary model run.
            validate_aggregation_pair(cohort.payloads[model][i], payloads[anchor][i])
            require(cohort.payloads[model][i]["patients"] == payloads[anchor][i]["patients"], "Default aggregation predictions differ from the configured model run")
        for strategy in strategies:
            arm = arm_by_pair[model, strategy]
            values = {"changed_patients": [], "coverage": [], "selective_accuracy": []}
            for i, seed in enumerate(cohort.seeds):
                default, current = payloads[anchor][i], payloads[arm][i]
                validate_aggregation_pair(default, current)
                left = {row["patient_id"]: row for row in default["patients"]}
                right = {row["patient_id"]: row for row in current["patients"]}
                eligible = [row for row in right.values() if row["prediction"] is not None]
                values["changed_patients"].append(sum(left[p]["prediction"] != right[p]["prediction"] for p in left))
                values["coverage"].append(len(eligible)/len(right))
                values["selective_accuracy"].append(sum(row["truth"] == row["prediction"] for row in eligible)/len(eligible) if eligible else None)
            extra[arm] = {name: {"values": series, "mean": float(np.mean(series)) if None not in series else None,
                                 "sample_sd": float(np.std(series, ddof=1)) if None not in series else None} for name, series in values.items()}
    joint_path = report.get("joint_statistics")
    require(isinstance(joint_path, str) and bool(joint_path), "Aggregation report requires joint_statistics")
    joint_path = (path.parent / joint_path).resolve()
    arm_cohort = Cohort({"reference_model": reference}, cohort.seeds, payloads, sources, cohort.layout, cohort.population, cohort.gt, summaries, raw_values)
    joint = read_json(joint_path)
    pairs = validate_statistics(joint, arm_cohort, reference=reference, arms=declarations)
    strategy_labels = {MAJORITY_NLL_TIEBREAK: "Majority vote + NLL tiebreak", LIKELIHOOD_WEIGHTED_VOTE: "NLL-weighted vote", BEST_LIKELIHOOD_INPUT: "Best single input group"}
    rows, sensitivity, records = [], [], []
    endpoints = ("classification:accuracy", "classification:macro_f1")
    for model in models:
        for strategy in strategies:
            arm = arm_by_pair[model, strategy]
            cells = []
            for metric in endpoints:
                infer = pairs[arm]["metrics"][metric] if arm in pairs else None
                summary = summaries[arm][metric]
                mark = significance(infer["p_raw"]) if infer else None
                best = summary["mean"] == max(values[metric]["mean"] for values in summaries.values())
                cells.append(metric_cell(summary, mark, decimals, best=best))
                records.append({"cohort": cohort.spec["id"], "cohort_label": cohort.spec["label"], "model": arm, "metric": metric,
                    "seeds": cohort.seeds, "seed_values": summary["values"], "unrounded_seed_values": raw_values[arm][:, cohort.layout.names.index(metric)].tolist(),
                    "mean": summary["mean"], "sample_sd": summary["sd"], "classification_seed_decimals": cohort.spec.get("classification_seed_decimals", {}).get(model),
                    "reference": reference, "stars": mark, "comparison": infer})
            for metric, entry in extra[arm].items():
                records.append({"cohort": cohort.spec["id"], "cohort_label": cohort.spec["label"], "model": arm,
                    "metric": "aggregation:" + metric, "seeds": cohort.seeds, "seed_values": entry["values"], "unrounded_seed_values": entry["values"],
                    "mean": entry["mean"], "sample_sd": entry["sample_sd"], "classification_seed_decimals": None,
                    "reference": arm_by_pair[model, MAJORITY_NLL_TIEBREAK], "stars": None, "comparison": None})
            rows.append([tex_escape(model), tex_escape(strategy_labels[strategy]), *cells])
            sensitivity.append([tex_escape(model), tex_escape(strategy_labels[strategy]), *[
                metric_cell({"mean": entry["mean"], "sd": entry["sample_sd"]}, None, decimals) if entry["mean"] is not None else "--"
                for entry in extra[arm].values()]])
    note = (tex_escape(cohort.spec["label"]) + ": the same saved predictions and completion NLL values support all three readouts within each model and seed. "
            "Mean $\\pm$ sample SD over five seeds. All patients, including abstentions, remain in the Accuracy and macro-F1 denominators. "
            "Raw two-sided patient-paired comparisons use " + tex_escape(cohort.spec["reference_model"]) + " with majority vote + NLL tiebreak as the reference for every row. "
            "$***$: $p<0.001$; $**$: $p<0.01$; $*$: $p<0.05$; ns: $p\\geq0.05$. Marks indicate differences in either direction. "
            "Bold denotes the highest unformatted mean across all model and strategy rows within this cohort and metric.")
    tables = {"aggregation": table_block(["Model", "Strategy", "Accuracy", "Macro-F1"], rows, note),
        "aggregation_sensitivity": table_block(["Model", "Strategy", "Changed patients", "Coverage", "Selective accuracy"], sensitivity,
            "Changed patients compare each strategy with its own model's default readout. Coverage is the fraction with a diagnosis; selective accuracy uses that fraction's patient count as its denominator. "
            "If any seed has no diagnosed patients, its selective accuracy and the five-seed selective-accuracy summary are undefined (shown as --); no seed is dropped.")}
    return {"sources": sources, "arms": declarations, "reference_arm": reference, "records": records, "sensitivity": extra,
            "statistical_settings": statistical_settings(joint),
            "comparison_source": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "joint_statistics": {"path": str(joint_path), "sha256": hashlib.sha256(joint_path.read_bytes()).hexdigest()}}, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--display-decimals", type=int, default=4)
    args = parser.parse_args()
    try:
        require(0 <= args.display_decimals <= 10, "display-decimals must be in [0,10]")
        require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Use an empty output directory")
        _, cohorts = load_cohorts(args.spec)
        output, files = [], {}
        for cohort in cohorts:
            path = cohort.spec.get("joint_statistics")
            require(isinstance(path, str) and bool(path), "joint_statistics is required for paper tables")
            joint_path = (args.spec.resolve().parent / path).resolve()
            row, tables = export_cohort(cohort, read_json(joint_path), args.display_decimals)
            row["joint_statistics"] = {"path": str(joint_path), "sha256": hashlib.sha256(joint_path.read_bytes()).hexdigest()}
            if "aggregation_comparison" in cohort.spec:
                row["aggregation"], aggregation_tables = export_aggregation(cohort, cohort.spec["aggregation_comparison"], args.spec.resolve().parent, args.display_decimals)
                tables.update(aggregation_tables)
            output.append(row)
            files.update({f"{cohort.spec['id']}_{view}.tex": text for view, text in tables.items()})
        result = {"schema_version": 1, "analysis": "canonical_five_seed_paper_tables", "display_decimals": args.display_decimals,
            "spec": {"path": str(args.spec.resolve()), "sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest()}, "cohorts": output}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "paper_tables.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        write_csv(args.output_dir / "paper_tables.csv", [record for cohort in output for record in cohort["records"] + cohort.get("aggregation", {}).get("records", [])])
        for name, text in files.items():
            (args.output_dir / name).write_text(text, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "cohorts": len(output), "tex_blocks": list(files)}))


if __name__ == "__main__":
    main()
