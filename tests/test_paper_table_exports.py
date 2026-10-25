"""Synthetic-only canonical end-to-end table/figure checks; no study results."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.evaluate_paper_protocol import evaluate
from evaluation.export_paper_tables import export_cohort, load_cohorts, significance, validate_statistics
from evaluation.joint_seed_statistics import PairingError, analyze, load_study
from evaluation.plot_confusion_matrices import confusion_records

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 22, 33, 44, 55]
# Canonical vocabulary, entirely synthetic patient observations; FNH is absent.
CLASSES = ["HCC", "ICC", "FNH"]


def make_fixture(tmp_path, *, constant=False, models=None, truth=None, unequal_protocols=False):
    truth = truth or [CLASSES[0], CLASSES[0], CLASSES[1], CLASSES[1]]
    manifest = {"schema_version": 1, "classes": CLASSES,
                "patients": [{"patient_id": f"p{i}", "truth": actual} for i, actual in enumerate(truth)],
                "input_groups": [{"group_id": f"g{i}", "patient_id": f"p{i}", "phases": {
                    phase: {"image_ref": f"p{i}-{phase}.png", "ground_truth": [{"label": actual, "bbox_2d": [0, 0, 10, 10]}] if phase == "a" else []}
                    for phase in ("a", "p", "v")}} for i, actual in enumerate(truth)]}
    if unequal_protocols:
        for group in manifest["input_groups"]:
            group["phases"]["a"]["ground_truth"] = [{"label": "HCC", "bbox_2d": [0,0,10,10]}, {"label": "ICC", "bbox_2d": [2,0,12,10]}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    models = models or ["synthetic_Full", "synthetic_Base"]
    runs = {}
    for model in models:
        runs[model] = []
        for s, seed in enumerate(SEEDS):
            if model == models[0]:
                predictions = truth if constant else [actual if i < len(truth)-s else None for i, actual in enumerate(truth)]
            else:
                predictions = [CLASSES[1] if actual == CLASSES[0] else CLASSES[0] for actual in truth]
            rows = [{"group_id": f"g{i}", "completion_nll": .5,
                     "raw": json.dumps({"detections": [] if predicted is None else [{"phase": "a", "label": predicted, "bbox_2d": [0, 0, 10, 10]}]})}
                    for i, predicted in enumerate(predictions)]
            if unequal_protocols:
                for row in rows:
                    row["raw"] = json.dumps({"detections": [{"phase": "a", "label": "ICC", "bbox_2d": [0,0,10,10]}, {"phase": "a", "label": "HCC", "bbox_2d": [8,0,18,10]}]})
            prediction_path = tmp_path / f"{model}-{seed}.jsonl"
            prediction_path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            payload = evaluate(manifest, prediction_path, [0, .25, .5, .75])
            payload["run_identity"] = {"model_name": model, "seed": seed}
            payload["sources"] = {"predictions": {"sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()}}
            metrics_path = tmp_path / f"{model}-{seed}.json"
            metrics_path.write_text(json.dumps(payload))
            runs[model].append({"seed": seed, "metrics": metrics_path.name})
    study_spec = {"schema_version": 1, "seeds": SEEDS, "reference": models[0], "models": runs}
    study_path = tmp_path / "joint_spec.json"
    study_path.write_text(json.dumps(study_spec))
    joint = analyze(load_study(study_path), exact=True, bootstrap=40, random_seed=2026)
    (tmp_path / "joint.json").write_text(json.dumps(joint))
    cohort = {"id": "synthetic", "label": "Synthetic cohort", "classes": CLASSES, "macro_classes": CLASSES,
        "model_order": models, "models": runs, "reference_model": models[0], "joint_statistics": "joint.json",
        "threshold_order": [0, .25, .5, .75], "primary_protocol": "class_agnostic", "primary_threshold": .5,
        "views": {"overview": models, "per_class": models, "thresholds": models}}
    config = {"schema_version": 1, "seeds": SEEDS, "cohorts": [cohort]}
    spec_path = tmp_path / "paper.json"; spec_path.write_text(json.dumps(config))
    return spec_path, config, joint


def test_mean_sample_sd_abstentions_and_absent_class_are_hand_checkable(tmp_path):
    path, _, joint = make_fixture(tmp_path)
    _, cohorts = load_cohorts(path)
    cohort = cohorts[0]
    table, blocks = export_cohort(cohort, joint, 4)
    accuracy = next(row for row in table["records"] if row["model"] == "synthetic_Full" and row["metric"] == "classification:accuracy")
    assert accuracy["seed_values"] == [1, .75, .5, .25, 0]
    assert accuracy["mean"] == .5
    assert accuracy["sample_sd"] == pytest.approx(math.sqrt(sum((v-.5)**2 for v in [1,.75,.5,.25,0])/4))
    assert table["class_support"][CLASSES[2]] == 0
    assert set(blocks) == {"overview", "per_class", "thresholds"}
    assert "GFCA" in blocks["thresholds"] and "SCF" in blocks["thresholds"]
    assert "$>0$" in blocks["thresholds"]
    cells, matrices = confusion_records(cohort)
    assert len(cells) == 2*5*3*4
    for matrix, normalized in matrices.values():
        assert matrix.sum() == 4
        assert matrix.shape == (3, 4)
        assert matrix[2].sum() == 0 and normalized[2].sum() == 0
        assert normalized[:2].sum(axis=1).tolist() == [1, 1]
    assert matrices["synthetic_Full", SEEDS[-1]][0][:, -1].tolist() == [2, 2, 0]


def test_constant_bootstrap_interval_signed_contrast_and_raw_p(tmp_path):
    path, _, joint = make_fixture(tmp_path, constant=True)
    _, cohorts = load_cohorts(path)
    table, _ = export_cohort(cohorts[0], joint, 4)
    row = next(row for row in table["records"] if row["model"] == "synthetic_Base" and row["metric"] == "classification:accuracy")
    # Of 16 exact four-patient swaps, all-or-none swaps alone reach |delta|=1.
    assert row["comparison"]["p_raw"] == 2/16
    assert row["comparison"]["reference_minus_comparator"] == 1
    assert row["comparison"]["difference_patient_bootstrap_95_ci"] == [1, 1]
    assert row["stars"] == "ns"
    assert row["comparison"]["reference_patient_bootstrap_95_ci"] == [1, 1]
    assert row["comparison"]["comparator_patient_bootstrap_95_ci"] == [0, 0]


def test_ca_gfca_scf_labels_identify_distinct_numerical_results(tmp_path):
    path, _, joint = make_fixture(tmp_path, constant=True, truth=["HCC"], unequal_protocols=True)
    _, cohorts = load_cohorts(path)
    cohort = cohorts[0]
    result, blocks = export_cohort(cohort, joint, 4)
    # At 0.25 the geometric assignment has two wrong-label pairs (IoU 1,.25).
    # Same-class assignment has IoUs 1/9,2/3 and accepts one; GFCA accepts none.
    expected = [("CA", "class_agnostic", 1.0), ("SCF", "same_class_priority", .5), ("GFCA", "geometry_first_class_aware", 0.0)]
    for label, protocol, recall in expected:
        row = next(record for record in result["records"] if record["model"] == "synthetic_Full" and record["metric"] == f"detection:{protocol}:iou=0.25:recall")
        assert row["mean"] == recall
        tex = next(line for line in blocks["thresholds"].splitlines() if line.startswith(label + " & $\\geq 0.25$ & synthetic\\_Full & "))
        assert f"\\mathbf{{{recall:.4f}}}" in tex.split(" & ")[4]
        cohort.spec["primary_protocol"] = protocol
        cohort.spec["primary_threshold"] = .25
        _, primary = export_cohort(cohort, joint, 4)
        assert f"{label} R@0.25" in primary["overview"]
        model_row = next(line for line in primary["overview"].splitlines() if line.startswith("synthetic\\_Full & "))
        assert f"\\mathbf{{{recall:.4f}}}" in model_row.split(" & ")[2]


def test_significant_negative_reference_difference_keeps_star_and_correct_best(tmp_path):
    path, config, _ = make_fixture(tmp_path, constant=True, truth=["HCC","ICC"]*3)
    config["cohorts"][0]["reference_model"] = "synthetic_Base"
    path.write_text(json.dumps(config))
    study_path = tmp_path/"joint_spec.json"
    study_spec = json.loads(study_path.read_text()); study_spec["reference"] = "synthetic_Base"
    study_path.write_text(json.dumps(study_spec))
    joint = analyze(load_study(study_path), exact=True, bootstrap=20)
    _, cohorts = load_cohorts(path)
    result, blocks = export_cohort(cohorts[0], joint, 4)
    record = next(row for row in result["records"] if row["model"] == "synthetic_Full" and row["metric"] == "classification:accuracy")
    # Six synchronized patient swaps: 2/64 extremes, reference is worse by one.
    assert record["comparison"]["p_raw"] == 2/64
    assert record["comparison"]["reference_minus_comparator"] == -1
    assert record["comparison"]["difference_patient_bootstrap_95_ci"] == [-1,-1]
    assert record["stars"] == "*"
    tex_row = next(line for line in blocks["overview"].splitlines() if line.startswith("synthetic\\_Full & "))
    assert r"\mathbf{1.0000}\,\pm\,0.0000^{*}" in tex_row.split(" & ")[5]


@pytest.mark.parametrize("p,mark", [(0,"***"),(.0009999,"***"),(.001,"**"),(.009999,"**"),(.01,"*"),(.049999,"*"),(.05,"ns"),(1,"ns")])
def test_raw_p_boundaries_do_not_round_before_stars(p, mark):
    assert significance(p) == mark


@pytest.mark.parametrize("mutation", ["hash", "direction", "seed_values", "missing_pair", "missing_metric", "duplicate_pair", "denominator", "p_count", "reversed_ci"])
def test_statistics_must_identify_exact_runs_and_endpoints(tmp_path, mutation):
    path, _, joint = make_fixture(tmp_path)
    _, cohorts = load_cohorts(path)
    broken = copy.deepcopy(joint)
    metric = broken["comparisons"][0]["metrics"]["classification:accuracy"]
    if mutation == "hash": broken["sources"][0]["sha256"] = "0"*64
    elif mutation == "direction": broken["comparisons"][0]["reference"] = "synthetic_Base"
    elif mutation == "seed_values": metric["reference_values"][0] = .123
    elif mutation == "missing_pair": broken["comparisons"] = []
    elif mutation == "missing_metric": del broken["comparisons"][0]["metrics"]["classification:accuracy"]
    elif mutation == "duplicate_pair": broken["comparisons"].append(copy.deepcopy(broken["comparisons"][0]))
    elif mutation == "denominator": broken["patients"] = 3
    elif mutation == "p_count": metric["p_raw"] = .123
    elif mutation == "reversed_ci": metric["difference_patient_bootstrap_95_ci"] = [1,-1]
    with pytest.raises(PairingError):
        validate_statistics(broken, cohorts[0])


def test_rounding_is_per_model_classification_only_and_statistics_remain_raw(tmp_path):
    path, config, joint = make_fixture(tmp_path, truth=[CLASSES[0], CLASSES[0], CLASSES[1]])
    config["cohorts"][0]["classification_seed_decimals"] = {"synthetic_Full": 4}
    path.write_text(json.dumps(config))
    _, cohorts = load_cohorts(path)
    table, _ = export_cohort(cohorts[0], joint, 4)
    row = next(row for row in table["records"] if row["model"] == "synthetic_Full" and row["metric"] == "classification:accuracy")
    assert row["seed_values"] == [1, .6667, .3333, 0, 0]
    detection = next(row for row in table["records"] if row["model"] == "synthetic_Full" and row["metric"] == "detection:class_agnostic:iou=0.5:recall")
    assert detection["seed_values"][1] == 2/3
    assert joint["comparisons"][0]["metrics"]["classification:accuracy"]["reference_values"][1] == 2/3


def test_factorial_requires_all_eight_and_uniform_is_separate(tmp_path):
    import itertools
    names = ["synthetic_Full"] + [f"synthetic_factorial_{i}" for i in range(7)] + ["synthetic_Uniform"]
    path, config, joint = make_fixture(tmp_path, constant=True, models=names)
    config["cohorts"][0]["factorial"] = {"factors": ["SHAE", "RARS", "DA-CVQ/VGS"],
        "rows": [{"model": model, "bits": list(bits)} for model, bits in zip(names, itertools.product((0,1), repeat=3))],
        "uniform_model": names[-1]}
    path.write_text(json.dumps(config))
    _, cohorts = load_cohorts(path)
    _, blocks = export_cohort(cohorts[0], joint, 4)
    assert "synthetic\\_Uniform" not in blocks["factorial"]
    assert "synthetic\\_Uniform" in blocks["uniform"]
    cohorts[0].spec["factorial"]["rows"][-1]["bits"] = [0,0,0]
    with pytest.raises(PairingError, match="factorial"):
        export_cohort(cohorts[0], joint, 4)


def test_executable_tables_and_deterministic_all_format_plots(tmp_path):
    pytest.importorskip("matplotlib")
    path, _, _ = make_fixture(tmp_path)
    tables = tmp_path / "tables"
    command = [sys.executable, str(ROOT/"evaluation/export_paper_tables.py"), "--spec", str(path), "--output-dir", str(tables)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads((tables/"paper_tables.json").read_text())
    assert result["cohorts"][0]["patients"] == 4
    assert (tables/"paper_tables.csv").exists()
    assert len(list(tables.glob("*.tex"))) == 3
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0
    outputs = []
    for name in ("figures_one", "figures_two"):
        output = tmp_path/name; outputs.append(output)
        subprocess.run([sys.executable, str(ROOT/"evaluation/plot_confusion_matrices.py"), "--spec", str(path), "--output-dir", str(output), "--normalize", "both", "--dpi", "72"], cwd=ROOT, check=True, capture_output=True, text=True)
        data = json.loads((output/"confusion_matrices.json").read_text())
        assert sum(row["count"] for row in data["cells"]) == 40
        assert len(list(output.glob("*.png"))) == len(list(output.glob("*.pdf"))) == len(list(output.glob("*.svg"))) == 2
    for first in outputs[0].iterdir():
        assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256((outputs[1]/first.name).read_bytes()).digest()


def test_duplicate_seed_fails_before_any_output(tmp_path):
    path, config, _ = make_fixture(tmp_path)
    config["cohorts"][0]["models"]["synthetic_Full"][1]["seed"] = SEEDS[0]
    path.write_text(json.dumps(config))
    with pytest.raises(PairingError, match="Duplicate"):
        load_cohorts(path)


def test_three_aggregation_readouts_link_to_exact_same_predictions_and_cli(tmp_path):
    from aggregation.summarize_seed_results import summarize_runs
    from evaluation.evaluate_paper_protocol import AGGREGATION_RULES
    from evaluation.export_paper_tables import export_aggregation

    path, config, _ = make_fixture(tmp_path)
    manifest = json.loads((tmp_path/"manifest.json").read_text())
    agg_dir = tmp_path / "aggregation"; agg_dir.mkdir()
    strategies = list(AGGREGATION_RULES)
    models = config["cohorts"][0]["model_order"]
    arm_spec, all_runs, arms = {}, {}, []
    for model in models:
        for strategy in strategies:
            arm = model + "::" + strategy
            arm_spec[arm] = {"model_name": model, "aggregation_strategy": strategy}
            all_runs[arm] = []
            payloads = []
            for seed in SEEDS:
                prediction = tmp_path/f"{model}-{seed}.jsonl"
                payload = evaluate(manifest, prediction, [0,.25,.5,.75], aggregation_strategy=strategy)
                payload["run_identity"] = {"model_name": model, "seed": seed}
                payload["sources"] = {"predictions": {"sha256": hashlib.sha256(prediction.read_bytes()).hexdigest()}}
                payloads.append(payload)
                metrics_path = agg_dir/f"{model}-{seed}-{strategy}.json"
                metrics_path.write_text(json.dumps(payload))
                all_runs[arm].append({"seed": seed, "metrics": metrics_path.name})
            summary = summarize_runs(payloads, SEEDS)["summary"]
            arms.append({"arm": arm, "model_name": model, "aggregation_strategy": strategy, "runs": all_runs[arm],
                         "summary": {metric: {"values": value["values"], "mean": value["mean"], "sample_sd": value["sd"]} for metric, value in summary.items()}})
    reference = models[0]+"::majority_nll_tiebreak"
    joint_spec = {"schema_version":1,"seeds":SEEDS,"comparison_mode":"aggregation_table","reference":reference,"models":all_runs,"arms":arm_spec}
    (agg_dir/"joint_spec.json").write_text(json.dumps(joint_spec))
    (agg_dir/"joint_seed_statistics.json").write_text(json.dumps(analyze(load_study(agg_dir/"joint_spec.json"), exact=True, bootstrap=20)))
    report = {"schema_version":1,"cohort":"synthetic","seeds":SEEDS,"reference_model":models[0],"reference_arm":reference,
              "arms":arms,"joint_statistics":"joint_seed_statistics.json"}
    report_path = agg_dir/"aggregation_comparison.json"; report_path.write_text(json.dumps(report))
    options = {"path":"aggregation/aggregation_comparison.json","model_order":models,"strategy_order":strategies}
    config["cohorts"][0]["aggregation_comparison"] = options
    path.write_text(json.dumps(config))
    output = tmp_path/"complete_tables"
    subprocess.run([sys.executable, str(ROOT/"evaluation/export_paper_tables.py"), "--spec", str(path), "--output-dir", str(output)], cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads((output/"paper_tables.json").read_text())["cohorts"][0]["aggregation"]
    assert len(result["arms"]) == 6
    assert len(result["records"]) == 30
    for series in result["sensitivity"].values():
        assert series["changed_patients"]["values"] == [0]*5
    assert result["sensitivity"][reference]["coverage"]["values"] == [1,.75,.5,.25,0]
    assert result["sensitivity"][reference]["selective_accuracy"]["values"] == [1,1,1,1,None]
    assert result["sensitivity"][reference]["selective_accuracy"]["mean"] is None
    assert (output/"synthetic_aggregation.tex").exists()
    assert (output/"synthetic_aggregation_sensitivity.tex").exists()
    assert r"\mathbf{0.5000}" in (output/"synthetic_aggregation.tex").read_text()
    # A comparator can exceed the reference: ranking follows the mean, not identity.
    negative_report = copy.deepcopy(report)
    negative_report["reference_model"] = models[1]
    negative_report["reference_arm"] = models[1]+"::majority_nll_tiebreak"
    negative_spec = copy.deepcopy(joint_spec); negative_spec["reference"] = negative_report["reference_arm"]
    (agg_dir/"negative_spec.json").write_text(json.dumps(negative_spec))
    (agg_dir/"negative_statistics.json").write_text(json.dumps(analyze(load_study(agg_dir/"negative_spec.json"), exact=True, bootstrap=20)))
    negative_report["joint_statistics"] = "negative_statistics.json"
    (agg_dir/"negative_report.json").write_text(json.dumps(negative_report))
    _, cohorts = load_cohorts(path)
    negative_cohort = copy.deepcopy(cohorts[0]); negative_cohort.spec["reference_model"] = models[1]
    negative_options = {**options,"path":"aggregation/negative_report.json"}
    negative, negative_blocks = export_aggregation(negative_cohort, negative_options, tmp_path, 4)
    comparison = next(row for row in negative["records"] if row["model"] == reference and row["metric"] == "classification:accuracy")
    assert comparison["comparison"]["reference_minus_comparator"] == -.5
    higher_row = next(line for line in negative_blocks["aggregation"].splitlines() if line.startswith(r"synthetic\_Full & Majority"))
    assert r"\mathbf{0.5000}" in higher_row
    lower_row = next(line for line in negative_blocks["aggregation"].splitlines() if line.startswith(r"synthetic\_Base & Majority"))
    assert r"\mathbf" not in lower_row
    # Missing one strategy cannot silently yield a smaller comparison table.
    complete_report = copy.deepcopy(report)
    report["arms"].pop(); report_path.write_text(json.dumps(report))
    _, cohorts = load_cohorts(path)
    with pytest.raises(PairingError, match="every model"):
        export_aggregation(cohorts[0], options, tmp_path, 4)
    report_path.write_text(json.dumps(complete_report))
    alternative = agg_dir/complete_report["arms"][1]["runs"][0]["metrics"]
    saved = alternative.read_text()
    changed = json.loads(saved)
    changed["sources"]["predictions"]["sha256"] = "0"*64
    alternative.write_text(json.dumps(changed))
    with pytest.raises(PairingError, match="prediction source identities"):
        export_aggregation(cohorts[0], options, tmp_path, 4)


def test_all_tex_views_compile_with_special_labels_without_overflow(tmp_path):
    import itertools
    compiler = shutil.which("pdflatex")
    if compiler is None and Path("/Library/TeX/texbin/pdflatex").exists():
        compiler = "/Library/TeX/texbin/pdflatex"
    if compiler is None:
        pytest.skip("pdflatex is required for the optional rendered TeX check")
    symbols = "& % _ # $ { } ~ ^ " + chr(92)
    models = ["Synthetic Full " + symbols] + [f"Synthetic variant {i} " + symbols for i in range(1,8)] + ["Synthetic Uniform " + symbols]
    case = tmp_path/"special"; case.mkdir()
    path, config, _ = make_fixture(case, constant=True, models=models)
    cohort = config["cohorts"][0]
    cohort["label"] = "Synthetic cohort " + symbols
    cohort["class_labels"] = {label: label + " " + symbols for label in CLASSES}
    cohort["factorial"] = {"factors": ["SHAE","RARS","DA-CVQ/VGS"],
        "rows": [{"model": model, "bits": list(bits)} for model, bits in zip(models,itertools.product((0,1),repeat=3))], "uniform_model": models[-1]}
    path.write_text(json.dumps(config))
    subprocess.run([sys.executable, str(ROOT/"evaluation/export_paper_tables.py"), "--spec", str(path), "--output-dir", str(case/"tables")], cwd=ROOT, check=True, capture_output=True, text=True)
    agg = tmp_path/"aggregation"; agg.mkdir()
    test_three_aggregation_readouts_link_to_exact_same_predictions_and_cli(agg)
    blocks = sorted((case/"tables").glob("*.tex")) + sorted((agg/"complete_tables").glob("*aggregation*.tex"))
    assert len(blocks) == 7
    wrapper = "\\documentclass[10pt]{article}\n\\usepackage[T1]{fontenc}\n\\usepackage[a4paper,landscape,margin=12mm]{geometry}\n\\usepackage{booktabs,longtable}\n\\begin{document}\n"
    wrapper += "\n".join("\\clearpage\n\\input{" + str(block.relative_to(tmp_path)) + "}" for block in blocks)
    wrapper += "\n\\end{document}\n"
    (tmp_path/"validation.tex").write_text(wrapper)
    for _ in range(2):
        subprocess.run([compiler,"-interaction=nonstopmode","-halt-on-error","validation.tex"],cwd=tmp_path,check=True,capture_output=True,text=True)
    log = (tmp_path/"validation.log").read_text()
    assert "Overfull" not in log
    assert "LaTeX Error" not in log
    assert (tmp_path/"validation.pdf").stat().st_size > 1000


@pytest.mark.parametrize("model_count", [1,2])
def test_real_aggregation_cli_feeds_paper_table_cli(tmp_path, model_count):
    from evaluation.evaluate_paper_protocol import AGGREGATION_RULES
    path, config, _ = make_fixture(tmp_path)
    models = config["cohorts"][0]["model_order"][:model_count]
    aggregation_spec = {"schema_version":1,"cohort":"synthetic","manifest":"manifest.json","seeds":SEEDS,
        "reference_model":models[0],"iou_thresholds":[0,.25,.5,.75],
        "models":{model:[{"seed":seed,"predictions":f"{model}-{seed}.jsonl"} for seed in SEEDS] for model in models}}
    (tmp_path/"aggregation_spec.json").write_text(json.dumps(aggregation_spec))
    subprocess.run([sys.executable,str(ROOT/"evaluation/compare_aggregation_strategies.py"),"--spec",str(tmp_path/"aggregation_spec.json"),
                    "--output-dir",str(tmp_path/"real_aggregation"),"--bootstrap","20","--exact-permutations"],cwd=ROOT,check=True,capture_output=True,text=True)
    config["cohorts"][0]["aggregation_comparison"] = {"path":"real_aggregation/aggregation_comparison.json","model_order":models,"strategy_order":list(AGGREGATION_RULES)}
    path.write_text(json.dumps(config))
    subprocess.run([sys.executable,str(ROOT/"evaluation/export_paper_tables.py"),"--spec",str(path),"--output-dir",str(tmp_path/"real_tables")],cwd=ROOT,check=True,capture_output=True,text=True)
    output = json.loads((tmp_path/"real_tables/paper_tables.json").read_text())["cohorts"][0]["aggregation"]
    assert len(output["arms"]) == model_count*3
    assert output["sensitivity"][output["reference_arm"]]["selective_accuracy"]["mean"] is None
    assert (tmp_path/"real_tables/synthetic_aggregation.tex").exists()
