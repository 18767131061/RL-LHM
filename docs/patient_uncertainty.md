# Patient-sampling uncertainty figure

This script draws existing confidence intervals from a schema-compatible Supplementary Data S1 CSV. It performs no statistical estimation, bootstrap resampling, permutation tests, BH correction, pooling, averaging across seeds, or selection based on significance. Estimates and interval endpoints are read directly from their original rows. Source decimal strings are preserved verbatim in the local mapping files.

## Reproduce

Python 3.9+ and Matplotlib are required. The validated environment uses Matplotlib 3.9.4 (install with `python -m pip install matplotlib==3.9.4`). Run with explicit input and output locations:

```sh
python evaluation/plot_patient_uncertainty.py \
  --input /path/to/Supplementary_statistical_data.csv \
  --output-prefix /path/to/figures/patient_uncertainty \
  --audit-dir /path/to/patient_uncertainty_audit
```

Optional `--expected-rows N` and `--expected-sha256 SHA256` lock the input to an independently recorded row count and digest. The source is opened read-only; its SHA256 is checked again after rendering. The script produces a vector PDF, an editable vector SVG, and a 600 dpi PNG. Output size is exactly 6.3 × 5.35 inches, with fonts of at least 8 pt at that width. Do not apply additional downscaling when placing the figure in a manuscript.

## Display and selection contract

Panels A and B display all five training seeds for both Full RL-LHM − SFT and Full RL-LHM − vanilla GRPO macro-F1 differences, separately for the private and public cohorts: 20 point-and-interval records. The same horizontal limits apply to both panels, with a dashed zero line. Full denotes Full RL-LHM.

Panels C, D and E display Recall for private ICC, private FNH and public ICC, respectively. Every panel contains all three models and all five seeds: 45 records. The Recall axis is 0–1. Patient counts in each title are read from the source records and checked for agreement within the panel. Methods occupy the same rows across all three lower panels. Shared method/comparison labels appear at the left of each panel row.

Each training seed has a fixed color from the Okabe–Ito palette, a distinct marker, and its own vertically offset row within every method group. One legend applies to all panels; marker redundancy and fixed row positions retain interpretation in grayscale. Lines and end caps show the supplied 95% patient-bootstrap interval. The paired panels use paired-patient intervals; the class panels use individual-model patient intervals. These intervals describe patient sampling conditional on the trained model(s), not cross-seed variability. The plot does not infer significance from interval overlap or seed ordering.

Keys are `record_type, cohort, direction, class_name, model, comparator, seed, metric`. The entire input is checked for unique keys. The required plotted key set is exact; missing seeds, duplicate keys, nonfinite estimates, reversed intervals, incorrect CI methods, inconsistent patient counts and out-of-domain endpoints fail rather than being silently dropped or clipped. The complete original 60-endpoint paired family is checked, and its existing q-value range is recorded without recalculation. Other records are neither plotted nor modified.

## Mapping and validation outputs

The audit directory contains `plotted_records.csv` and `.json` (all 65 source records plus panel/row/color/marker mapping) and `validation.json` (source hash, unique keys, counts, CI methods, existing q ranges, axis limits, font size, asset hashes). All numeric coordinates are derived from the supplied source.

## Tests

```sh
python -m unittest -v tests.test_plot_patient_uncertainty
```


Supply a schema-compatible S1 CSV and explicit figure and audit output paths.
