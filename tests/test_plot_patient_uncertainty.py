"""Plotting contract tests with synthetic interval records."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.plot_patient_uncertainty import (KEY_FIELDS, REQUIRED, SEEDS, load_records,
    select_panels, validate_paired_family, paired_limits, render, write_audit, sha256)


def synthetic_records():
    """Synthetic schema-compatible fixture."""
    rows = []
    def add(**values):
        row = {field: "" for field in REQUIRED}
        row.update(summary_statistic="single_seed_estimate", estimate="0.123456",
                   ci_lower="-0.012345", ci_upper="0.234567", test_patients="100",
                   source_table="synthetic_fixture")
        row.update(values)
        rows.append(row)
    for cohort in ("private", "public"):
        for seed in SEEDS:
            for comparator in ("SFT", "vanilla GRPO"):
                for metric in ("accuracy", "macro_recall", "macro_f1"):
                    add(record_type="paired_difference", cohort=cohort, seed=seed,
                        model="Full RL-LHM", comparator=comparator, metric=metric,
                        ci_method="paired_patient_percentile_bootstrap", raw_p="0.2000",
                        bh_q="0.4000", test_method="synthetic", bh_family="synthetic 60-endpoint family")
    for cohort, class_name in (("private", "ICC"), ("private", "FNH"), ("public", "ICC")):
        for model in ("SFT", "vanilla GRPO", "Full RL-LHM"):
            for seed in SEEDS:
                add(record_type="class_performance", cohort=cohort, class_name=class_name,
                    model=model, seed=seed, metric="recall", estimate="0.400001",
                    ci_lower="0.000000", ci_upper="1.000000", test_patients="25",
                    ci_method="patient_percentile_bootstrap")
    # An unrelated row must not be mistaken for a plotted patient interval.
    add(record_type="unused_synthetic", cohort="private", metric="macro_f1")
    for index, row in enumerate(rows, 1):
        row["source_record"] = index
    return rows


class PlotContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_records()

    def test_exact_selection_and_original_decimal_strings(self):
        panels = select_panels(self.rows)
        self.assertEqual([len(panels[k]) for k in "ABCDE"], [10, 10, 15, 15, 15])
        mapped = [r for panel in panels.values() for r in panel]
        self.assertEqual(len({tuple(r[k] for k in KEY_FIELDS) for r in mapped}), 65)
        by_index = {r["source_record"]: r for r in self.rows}
        for r in mapped:
            original = by_index[r["source_record"]]
            for field in REQUIRED:
                self.assertEqual(r[field], original[field])
        self.assertEqual(panels["A"][0]["estimate"], "0.123456")
        self.assertEqual(panels["A"][0]["ci_lower"], "-0.012345")

    def test_missing_seed_and_duplicate_key_are_errors(self):
        rows = self.rows[1:]
        # The first required macro-F1 row, not the removed Accuracy row.
        rows = [r for r in rows if not (r["cohort"] == "private" and r["seed"] == "42"
                and r["comparator"] == "SFT" and r["metric"] == "macro_f1")]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            select_panels(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_panels(self.rows + [copy.deepcopy(self.rows[0])])

    def test_invalid_intervals_cannot_be_silently_clipped(self):
        for field, value in (("ci_lower", "-0.1"), ("ci_upper", "1.1"),
                             ("estimate", "NaN"), ("ci_lower", ".9"),
                             ("ci_method", "Wilson")):
            rows = copy.deepcopy(self.rows)
            rows[60][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                select_panels(rows)

    def test_patient_counts_must_agree_within_panel(self):
        self.rows[60]["test_patients"] = "26"
        with self.assertRaisesRegex(ValueError, "patient counts"):
            select_panels(self.rows)

    def test_full_bh_family_is_checked_without_recalculation(self):
        self.assertEqual(len(validate_paired_family(self.rows)), 60)
        with self.assertRaisesRegex(ValueError, "60 original"):
            validate_paired_family(self.rows[1:])
        self.rows[0]["bh_q"] = "1.2"
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_paired_family(self.rows)

    def test_axes_cover_supplied_endpoints(self):
        panels = select_panels(self.rows)
        low, high = paired_limits(panels)
        self.assertLess(low, -.012345)
        self.assertGreater(high, .234567)

    def test_file_to_asset_mapping_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "synthetic.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REQUIRED, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
            frozen_bytes = source.read_bytes()
            rows, digest = load_records(source, expected_rows=106, expected_sha256=sha256(source))
            panels = select_panels(rows)
            prefix = tmp / "figure"
            layout = render(panels, prefix, dpi=80)
            audit = write_audit(source, digest, rows, panels, validate_paired_family(rows),
                                layout, prefix, tmp / "audit")
            self.assertEqual(source.read_bytes(), frozen_bytes)
            self.assertEqual(layout["minimum_font_pt"], 8)
            self.assertEqual(layout["recall_axis_limits"], [0, 1])
            self.assertTrue(audit["source_unchanged"])
            self.assertEqual(audit["plotted_rows"], 65)
            loaded = json.loads((tmp / "audit/plotted_records.json").read_text())
            self.assertEqual(loaded[0]["ci_upper"], "0.234567")
            for ext in ("pdf", "png", "svg"):
                self.assertGreater(Path(str(prefix)+"."+ext).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
