"""Artificial aggregate fixtures only; none are reported patient results."""
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aggregation.patient import aggregate_patient_rows
from aggregation.summarize_seed_results import summarize
from evaluation.table_presentation import (
    COHORTS, COMPARATORS, ENDPOINTS, SEEDS,
    aggregation_view, factorial_sections, paired_summary,
)


def synthetic_paired():
    return [dict(cohort=cohort, seed=seed, comparator=comparator, endpoint=endpoint,
                 candidate="full_rl_lhm", delta=f"{(i - 2) / 100:.4f}",
                 ci_lower=f"{-0.12 + i / 100:.4f}", ci_upper=f"{0.20 + i / 100:.4f}",
                 p=f"{0.20 + i / 10:.4f}", q=f"{0.44 + i / 10:.4f}")
            for cohort, comparator, endpoint in itertools.product(COHORTS, COMPARATORS, ENDPOINTS)
            for i, seed in enumerate(SEEDS)]


def test_paired_ranges_keep_five_seed_identity_values_and_cis():
    source = synthetic_paired()
    original = copy.deepcopy(source)
    result = paired_summary(source)
    assert source == original == result["source_rows"]
    assert result["family_size"] == 60 and len(result["rows"]) == 12
    for summary in result["rows"]:
        expected = [row for row in source if all(row[key] == summary[key]
                    for key in ("cohort", "comparator", "endpoint"))]
        assert summary["seed_details"] == expected
        assert summary["delta_range"] == ["-0.0200", "0.0200"]
        assert summary["q_range"] == ["0.4400", "0.8400"]
        assert not any(key in summary for key in ("mean_p", "mean_q", "pooled_ci", "ci_lower_range", "ci_upper_range"))
    result["rows"][0]["seed_details"][0]["q"] = 0
    assert result["source_rows"] == original == source


@pytest.mark.parametrize("error", ["missing", "duplicate", "precision", "reverse", "nan", "bad_ci"])
def test_paired_summary_rejects_changed_family_or_invalid_records(error):
    rows = synthetic_paired()
    if error == "missing": rows.pop()
    elif error == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    elif error == "precision": rows[0]["endpoint"] = "macro_precision"
    elif error == "reverse": rows[0]["candidate"] = "sft"
    elif error == "nan": rows[0]["q"] = float("nan")
    else: rows[0]["ci_lower"] = 99
    with pytest.raises(ValueError): paired_summary(rows)


def synthetic_factorial():
    rows = [dict(section="factorial", rars=r, shae=s, vgs=v,
                 accuracy={"mean": 0.20 + i / 100, "sd": 0.01})
            for i, (r, s, v) in enumerate(itertools.product((False, True), repeat=3))]
    rows.append(dict(section="additional_severity_control", control="uniform",
                     rars=True, shae=True, vgs=True, macro_precision=None,
                     accuracy={"mean": 0.99, "sd": 0.03}))
    return rows


def test_uniform_remains_separate_from_exact_eight_factorial_rows():
    source = synthetic_factorial()
    original = copy.deepcopy(source)
    result = factorial_sections(source)
    assert len(result["factorial_rows"]) == 8
    assert result["factorial_rows"] == source[:8]
    assert result["additional_severity_control"] == source[8:]
    assert result["additional_severity_control"][0]["macro_precision"] is None
    assert result["source_rows"] == original == source
    source[-1]["section"] = "factorial"
    with pytest.raises(ValueError): factorial_sections(source)


def synthetic_aggregation():
    # Artificial full-coverage inputs exercise the actual public aggregator.
    evaluation = aggregate_patient_rows([
        {"patient_id": "synthetic-a", "truth": "HCC", "prediction": "HCC"},
        {"patient_id": "synthetic-b", "truth": "ICC", "prediction": "HCC"},
    ], ["HCC", "ICC"])
    row = dict(cohort="private", model="synthetic_model", strategy=evaluation["strategy"],
               coverage={str(seed): evaluation["coverage"] for seed in SEEDS},
               accuracy={"mean": evaluation["metrics"]["accuracy"], "sd": 0.0},
               selective_accuracy={"mean": evaluation["selective_accuracy"], "sd": 0.0},
               macro_f1={"mean": evaluation["metrics"]["macro_f1"], "sd": 0.0},
               changed_patients={"mean": 0.0, "sd": 0.0})
    return evaluation, [row]


def test_aggregation_hides_only_display_fields_preserves_actual_results():
    evaluation, source = synthetic_aggregation()
    originals = copy.deepcopy((evaluation, source))
    view = aggregation_view(source)
    assert view["source_rows"] == source
    assert "coverage" not in view["rows"][0] and "selective_accuracy" not in view["rows"][0]
    assert view["rows"][0]["accuracy"] == source[0]["accuracy"]
    assert view["rows"][0]["changed_patients"] == source[0]["changed_patients"]
    assert (evaluation, source) == originals
    assert evaluation["coverage"] == 1 and "selective_accuracy" in evaluation
    source[0]["coverage"][str(SEEDS[0])] = 0.99
    with pytest.raises(ValueError, match="every seed"): aggregation_view(source)


def test_aggregation_does_not_infer_full_coverage_from_rounded_means():
    _, rows = synthetic_aggregation()
    rows[0]["coverage"] = {"mean": 1.0, "sd": 0.0}
    with pytest.raises(ValueError, match="each of the five seeds"): aggregation_view(rows)
    _, rows = synthetic_aggregation()
    rows[0]["selective_accuracy"]["sd"] = 0.01
    with pytest.raises(ValueError, match="cells differ"): aggregation_view(rows)


def test_seed_summary_uses_unrounded_values_and_sample_sd():
    summary = summarize([0.10001, 0.10003, 0.10005, 0.10007, 0.10009])
    assert summary["mean"] == pytest.approx(0.10005)
    assert summary["sd"] == pytest.approx(0.000031622776601683795)
    assert summary["values"] == [0.10001, 0.10003, 0.10005, 0.10007, 0.10009]


def test_presentation_cli_keeps_sources_and_refuses_overwrite(tmp_path):
    source = tmp_path / "synthetic.json"
    output = tmp_path / "view.json"
    source.write_text(json.dumps(synthetic_paired()))
    before = source.read_bytes()
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "evaluation/table_presentation.py"),
               "--kind", "paired", "--input", str(source), "--output", str(output)]
    subprocess.run(command, check=True)
    assert source.read_bytes() == before
    assert len(json.loads(output.read_text())["source_rows"]) == 60
    assert subprocess.run(command, capture_output=True).returncode != 0
