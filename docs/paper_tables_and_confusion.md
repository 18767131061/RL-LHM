# Paper tables and confusion panels

`evaluation/export_paper_tables.py` calculates descriptive tables from canonical
`metrics.json` patient exports and joins the corresponding fixed-five-seed
patient-paired inference. `evaluation/plot_confusion_matrices.py` uses the same
run specification to render every model and seed. Install the core requirements
and `matplotlib>=3.8` for plotting.

## Inputs and identities

Start with `configs/paper_tables.example.json` for the trained-model overview or
`configs/factorial_tables.example.json` for the eight component combinations and
the separate Uniform severity comparison. Replace the example paths and model
names with the exact identities passed to the canonical evaluator. The examples
declare five training seeds; a zero-shot block can instead declare its five
inference seeds in a separate specification. The numbered model directories in
the examples are path placeholders, not an implied checkpoint naming convention.
For the example threshold set, run the canonical evaluator with
`--iou-thresholds 0 0.25 0.5 0.75`; its default single threshold does not supply
the full sweep. If exports contain additional thresholds, list those in
`threshold_order` as well.

Each cohort declares:

- A unique filename-safe `id`, a display `label`, an ordered full output
  `classes` list and an ordered `macro_classes` list. These must agree exactly
  with the canonical exports. Optional `class_labels` changes display labels
  only; it covers every output class and cannot merge distinct classes.
- `model_order` and `models`, with exactly one canonical file for each model and
  each of the five distinct integer `seeds`. Every model name and seed must
  agree with the file's `run_identity`.
- `reference_model` and `joint_statistics`, the output of
  `evaluation/joint_seed_statistics.py`. Its study must contain exactly the same
  model and seed runs as this cohort. Paths resolve relative to the specification
  containing them. The joint statistics can reside in a different directory;
  source hashes, rather than absolute path spellings, bind its inputs.
- `primary_protocol` and `primary_threshold` for the overview; `threshold_order`
  must contain every threshold present in the exports. `views` explicitly lists
  the model order for `overview`, `per_class`, and `thresholds`.
- Optional `classification_seed_decimals`, a model-to-`4` mapping for the
  auxiliary classification reporting convention. Omitted models use full
  precision. A `null` value also retains full precision.

All runs in a cohort must share the frozen manifest digest, patient identities,
patient truths, output vocabulary, macro class space, aggregation identity,
per-patient ground-truth lesion counts and detection protocol/threshold sets.
Different cohorts remain separate even when model names are the same. A file
cannot supply more than one declared run. Statistical input requires each
non-reference comparison and every endpoint; missing, duplicate, reversed,
stale-source and incompatible records fail before tables are written.

The exporter verifies each joint statistic's five unrounded seed values, mean,
sample SD, signed difference, raw p-value/count identity and interval bounds.
The paired intervals themselves come from the joint-statistics computation;
the presentation step does not recompute bootstrap draws.

## Generate the tables

```bash
python evaluation/export_paper_tables.py \
  --spec configs/paper_tables.example.json \
  --output-dir results/paper-tables
```

The output directory must be empty. `paper_tables.json` and `paper_tables.csv`
contain every computed endpoint, all five seed values, unrounded seed values,
means, sample SDs and the complete linked comparison records. P-values retain
their original precision. `--display-decimals` controls TeX display only and
defaults to four.

The per-cohort TeX blocks are:

| File suffix | Contents |
| --- | --- |
| `_overview.tex` | Primary-protocol detection P/R/F1, ungated Mean IoU, patient Accuracy and macro P/R/F1 |
| `_per_class.tex` | Fixed class support and five-seed per-class P/R/F1 |
| `_thresholds.tex` | All declared thresholds for CA, GFCA and SCF, each retaining its own assignment protocol |
| `_factorial.tex` | All eight binary component combinations in separate detection and classification panels, when configured |
| `_uniform.tex` | The separate Uniform severity-weight comparison, when configured |
| `_aggregation.tex` | Accuracy and macro-F1 for all three same-prediction aggregation strategies, when configured |
| `_aggregation_sensitivity.tex` | Changed patients, coverage and selective accuracy for those strategies |

The blocks use `booktabs` and `longtable`
(`\usepackage{booktabs,longtable}`). Insert each with `\input{...}`. The
`per_class` and `thresholds` blocks use a multipage `longtable` with repeated
headers and belong directly in the document flow. The other blocks use compact
`tabular` panels and can be placed in the manuscript's chosen table environment.
The factorial view separates detection and classification to keep the three
factor columns readable. The blocks scope their small font and column spacing
locally. View lists can select different model blocks while the JSON/CSV keeps
all configured models and endpoints.

Metrics are recomputed within each seed before calculating the arithmetic mean
and sample SD (`ddof=1`). Predictions are never pooled across training seeds.
Absent classes remain in the fixed macro class space. A patient with no eligible
diagnostic vote stays in the full denominator, contributes a false negative to
its true class and no diagnostic-class false positive.

For models explicitly assigned `classification_seed_decimals: 4`, descriptive
overall and per-class classification seed metrics use decimal half-even
four-place rounding before mean/SD aggregation. Detection counts/ratios and all
inferential tests remain unrounded. Both descriptive and unrounded seed series
are available in JSON/CSV.

Stars use the **unadjusted, unrounded, two-sided** patient-paired p-value:
`***` for `p<0.001`, `**` for `p<0.01`, `*` for `p<0.05`, otherwise `ns`.
Equality belongs to the next less significant category. Marks describe a
difference in either direction; the signed contrast is always reference minus
comparator. The reference has no self-comparison superscript. Bold marks the
highest descriptive mean in the displayed view, before final display rounding.
The thresholds block ranks within each protocol and threshold, and the
per-class block ranks within each class and endpoint. Raw p-values, signed
differences and bootstrap intervals remain available in the machine-readable
output, even when display rounding would conceal a threshold boundary.

## Factorial and Uniform views

`factorial` is an optional cohort field:

```json
{
  "factors": ["SHAE", "RARS", "DA-CVQ/VGS"],
  "rows": [
    {"model": "vanilla GRPO", "bits": [0, 0, 0]},
    {"model": "+SHAE", "bits": [1, 0, 0]},
    {"model": "+RARS", "bits": [0, 1, 0]},
    {"model": "+DA-CVQ/VGS", "bits": [0, 0, 1]},
    {"model": "RARS-off", "bits": [1, 0, 1]},
    {"model": "SHAE-off", "bits": [0, 1, 1]},
    {"model": "DA-CVQ/VGS-off", "bits": [1, 1, 0]},
    {"model": "Full RL-LHM", "bits": [1, 1, 1]}
  ],
  "uniform_model": "Uniform"
}
```

All eight bit patterns and eight distinct models are required. The factor names
and row order are explicit. `uniform_model` must be a ninth distinct model:
Uniform retains the selected configuration's components while changing severity
weights and is not a ninth binary component combination. Every named run must
also appear in `models` and in the paired-statistics input.

## Three-strategy aggregation tables

Produce an `aggregation_comparison.json` with
`evaluation/compare_aggregation_strategies.py`, then add this optional field to
the corresponding cohort:

```json
"aggregation_comparison": {
  "path": "../results/private/aggregation/aggregation_comparison.json",
  "model_order": ["vanilla GRPO", "Full RL-LHM"],
  "strategy_order": [
    "majority_nll_tiebreak",
    "likelihood_weighted_vote",
    "best_likelihood_input"
  ]
}
```

The comparison report's `cohort` equals this cohort's `id`. Its model list is a
subset of the configured models and includes the cohort reference. All three
strategies and all five seeds are mandatory. Relative run and joint-statistics
paths within the comparison report resolve against that report's directory.

The exporter reads the canonical files for every arm, verifies the common
prediction-source SHA256, saved group outputs, completion NLL, patient coverage
and lesion detection, and binds each model's default readout to the ordinary
model runs supplied in the paper specification. The reported arm summaries and
joint statistics must agree with those canonical files. Every inferential mark
uses the configured reference model's **default** readout, including comparison
against that same model's two alternative readouts. Changed-patient counts use
each model's **own default** readout as their descriptive baseline.

Coverage is the fraction of patients with a diagnostic prediction. Selective
accuracy uses only diagnosed patients in its denominator; it is supplementary
to the full-denominator classification metrics. If a seed has no diagnosed
patients, its selective accuracy is `null`. The five-seed selective-accuracy
mean/SD is then also `null` and is displayed as `--`; the exporter does not drop
that seed. JSON/CSV retain all per-seed values, arm declarations, source hashes
and the signed reference-minus-comparator inference.

## Generate confusion panels

```bash
python evaluation/plot_confusion_matrices.py \
  --spec configs/paper_tables.example.json \
  --output-dir results/confusion \
  --normalize both --dpi 300
```

`--normalize counts`, `row`, or `both` controls the heatmap. A separate figure
for each cohort contains one row per model and five columns in the declared
seed order. True-class rows preserve `macro_classes`; predicted columns preserve
the complete output vocabulary and add an explicit **Abstention** column.
This also retains output classes outside a transfer cohort's truth vocabulary.
Classes with no true patients have zero rows and zero normalized proportions.
Row normalization divides by the complete true-class support, including
abstentions. Every panel sums to the same frozen patient count.

`confusion_matrices.json` and `.csv` always contain both raw counts and row
proportions, independent of the plot option. Each selected view is saved as
PNG, SVG and PDF with a common within-cohort color scale. The plotting process
uses the noninteractive Agg backend, closes figures and fixes date/hash
metadata for deterministic repeated output in a fixed runtime.

The plot command needs no paired-statistics file: only the cohort label, class
orders, model order, seeds and canonical run declarations are needed from the
specification. It validates complete model/seed identities before plotting.

