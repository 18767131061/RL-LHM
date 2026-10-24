# Legacy presentation of supplied statistical results

The paired-comparison views on this page preserve the legacy family of 60
separate per-seed tests. They are not the joint five-seed inference implemented
by `evaluation/joint_seed_statistics.py`; see [joint_seed_statistics.md](joint_seed_statistics.md)
for that protocol. Do not substitute one family's p/q values or intervals for
the other's. These exporters describe supplied records and do not certify any
manuscript's current results or presentation.

`evaluation/table_presentation.py` creates display views from supplied aggregate
results. It never recomputes predictions, metrics, p-values, q-values, confidence
intervals, or factorial effects. Every output retains the complete `source_rows`.
The input files belong with the experiment outputs and are not bundled here.

```bash
python evaluation/table_presentation.py --kind paired \
  --input /path/to/paired_endpoint_rows.json --output ./outputs/paired_view.json
```

The CLI requires a new output path. Its input is a JSON list of records; its
output is a JSON object containing both the presentation and source records.
The same presentation can supply manuscript and response tables. The helper
does not parse a manuscript or certify which historical experiment produced it.

## Paired classification comparisons

Each input record has `cohort` (`private` or `public`), `seed` (42, 2026, 3407,
1234, or 5678), `comparator` (`sft` or `vanilla_grpo`),
`candidate: "full_rl_lhm"`, and `endpoint` (`accuracy`, `macro_recall`, or
`macro_f1`). Supply its existing `delta`, `ci_lower`, `ci_upper`, `p`, and `q`.
Numeric strings retain their original precision. These must come from the same
fixed patient-level evaluation and the supplied legacy family of 60 comparisons.

The view rejects missing/duplicate identities, other endpoints, and reversed
candidate labels. It returns 12 cohort/comparator/endpoint summaries. Each has
the minimum and maximum of its five existing `delta` and `q` values and all
five seed-specific source records in `seed_details`. Delta means Full minus
the comparator. A range is a descriptive range across seeds; it is not a
confidence interval. The actual patient-bootstrap difference CIs remain
identified by seed. These `seed_details` are an interchange view for preserving
existing statistics and can accompany compact summaries in a statistical
supplement. No cross-seed CI,
mean p/q, or new Macro-Precision test is generated.

The q-values originate from the existing 60-comparison BH correction. A
presentation change must consume those values without rerunning tests or BH.
The display helper checks the 60 identities but cannot authenticate supplied
q-values or reconstruct their patient-level evidence. Preserve each result's existing precision and aggregation
convention. State whether each summary was computed from full-precision seed
results or retained rounded values; the exporter cannot recover that history.
`aggregation/summarize_seed_results.py` uses `ddof=1` on the
supplied values without rounding them. Do not reconstruct unavailable precision
from displayed table cells or replace an established summary with a different
rounding convention.

## Legacy locked summaries and statistical-detail CSV

`evaluation/export_locked_summary_tables.py` exports shared LaTeX body rows
from locked summaries and the existing statistical-detail CSV:

```bash
python evaluation/export_locked_summary_tables.py \
  --summary /path/to/results_summary.json \
  --supplement /path/to/Supplementary_statistical_data.csv \
  --output-dir ./outputs/tables
```

Use `--check` to compare generated fragments without writing. Inputs are
supplied separately; manuscript sources and real study data are not bundled
in this repository. This script does not invoke the evaluator, train models,
bootstrap, permute labels, or apply BH.

The summary JSON has `training_seeds` and `tables`. Each table specifies its
ordered `metrics` and `rows`; each row has a `model` and metric mappings
containing decimal-string `mean`, `sd`, and optional `best`/`seed_range` fields.
Class tables additionally retain `class` and `test_patients`. Locked strings
retain their original precision. The exporter does not reconstruct a primary
summary from rounded seed-level table cells. If `sd` represents sample SD
across five training seeds, its computation uses denominator n−1; this is
distinct from a patient interval. The exporter accepts the supplied SD and
cannot verify that denominator from a summary cell alone.

Primary `classification_private`/`classification_public` tables have five
columns: Model/training, Accuracy, macro-Precision, macro-Recall and macro-F1.
Their groups are Qwen3-VL-4B (SFT, vanilla GRPO and RL-LHM), Auxiliary VLM
baselines (three models), and Task-specific detectors (four models). Missing,
extra or duplicated models raise an error rather than silently dropping a row.
The corresponding `_response` fragments retain the same four endpoints for the
three primary training methods. They contain no significant-seed column.

The CSV `paired_difference` records must contain all 60 unique
cohort/comparator/endpoint/seed identities against `model: Full RL-LHM`.
`validate_paired_family()` validates five distinct seeds, all 60 identities and
existing q values; it does not change them. `paired_group_summaries()` returns
six cohort-specific comparator/endpoint groups, each retaining five existing
tests. English and Chinese notes report the number with `q < 0.05`, the number
with `q = 0.05`, the number with `q > 0.05`, and the minimum–maximum q range for
each group. The strict-below count uses `q < 0.05`; equality is shown separately.
All three outcomes are accepted. Range endpoints preserve their supplied
strings so a value such as `0.04999` is not displayed as `0.0500`.

The notes explicitly identify the legacy 60-test family and do not claim a
joint five-seed result. Neither counts nor ranges recalculate BH or alter any
p/q value. Missing, duplicate or unexpected family identities still fail.
The tested endpoints are Accuracy, macro-Recall and macro-F1. Macro-Precision
and auxiliary models are outside that family. Bold formatting follows the
supplied `best` marker; the exporter does not infer winners or significance.

All performance means and seed-range endpoints are validated in [0,1], and a
range must be ordered and contain its mean. These checks also inspect ranges
omitted from the main display. `metric_cell(..., domain="signed_difference")`
explicitly permits negative differences of proportions and preserves their
signs. It never applies an absolute value. `parse_locked_tex_summary()` parses
an explicit bracketed `--` separator separately from endpoint signs; a generic
numeric regex over the entire cell would incorrectly interpret `[0.8--1.0]`
as an upper bound of −1.0. The range renderer separates the TeX dash command
from endpoint text, including when a legitimate endpoint is negative.

Per-class fragments display mean ± supplied SD and test-patient counts.
The locked JSON retains any class-specific seed ranges; this exporter does not
create a separate `perclass_seed_ranges.tex` fragment. If presented separately,
seed ranges and patient-bootstrap intervals must be identified as different
summaries with different estimands.

Per-class and rare-class fragments should use their corresponding supplied
class means/SDs and actual test-patient counts. LOCO tables can retain their
existing direction-specific Accuracy, macro-Recall and macro-F1 summaries.
Do not borrow primary paired tests or invent an unreported Macro-Precision.
Different cohorts, classes, thresholds, configurations and directions remain
separate scientific groups.

The legacy statistical-detail CSV can hold original seed-level estimates,
patient-bootstrap/Wilson intervals, p/q values and discordant/correct-patient
counts once, under their full identities. It adds statistical detail to the
main mean/SD tables rather than duplicating the summary performance tables.
Its record types are `paired_difference`, `model_performance`,
`class_performance`, `loco_performance`, `auxiliary_performance`, and
`loco_class_summary`. The last type preserves the separately reported LOCO
class Recall/F1 means with `summary_statistic: five_seed_mean` and empty seed,
interval and inferential fields; no unreported class SD should be invented.
Blank statistical fields mean inapplicable or unreported, never zero or
nonsignificant. Existing values are preserved, including adverse comparisons.
No p-value, q-value or interval endpoint is averaged, pooled or substituted
for a mean-level inference.

`tests/test_export_locked_summary_tables.py` uses artificial inputs to verify
exact decimal preservation, legal 0/1 boundaries, negative-difference sign
preservation, malformed and reversed range rejection, q values below, equal to
and above 0.05, endpoint-specific counts and unrounded q ranges,
paired identity completeness, model ordering, five-column output, and separate
retention of class ranges. Its TeX integration test compiles grouped rows and
signed ranges with `pdflatex` when the engine is available and checks for
alignment errors and horizontal overflow. These CPU tests validate the export
interface, not the paper's experiments.

## Factorial configurations and severity control

Use `--kind factorial` separately for each fixed cohort/protocol. Supply eight
records marked `section: "factorial"`, each with unique Boolean `rars`, `shae`,
and `vgs` settings. Supply Uniform once with
`section: "additional_severity_control"` and `control: "uniform"`. Other metric
fields pass through verbatim. An unreported Macro-Precision remains null or the
source's explicit missing-data marker.

The returned `factorial_rows` contains exactly the eight component settings.
Uniform appears only in `additional_severity_control`; it does not enter
factorial main-effect calculations. Full/Selected and the RARS-off anchor can
be referenced from their factorial rows without duplicating separate controls.
The helper performs no main-effect calculation.

When presenting factorial detection results, six intermediate component
configurations can accompany the vanilla-GRPO and Full endpoint rows without
duplicating those endpoints across main and supplementary tables.
Class-specific results, model-specific seed details and complete threshold
series remain separate because their experimental identities or statistical
levels differ. Distinct CA, GFCA and same-class-first protocols retain their
own results; coinciding numerical cells do not make those analyses duplicates.

## Aggregation readouts

Use `--kind aggregation` with one record per `cohort`, `model`, and `strategy`.
Supply `accuracy`, `selective_accuracy`, `macro_f1`, and `changed_patients`
as the existing summary cells, retaining their SD fields. Supply `coverage` as
a mapping from each of the five string seed IDs to its actual coverage value.
The view requires every seed to equal 1 and the two accuracy cells to be equal
before hiding the redundant display columns. It retains Strategy, Accuracy,
Macro-F1 and Changed patients and adds the full-denominator caption statement.

The original coverage, selective accuracy, and other source fields remain in
`source_rows`; the evaluation and aggregation APIs remain unchanged. When any
seed has incomplete coverage, this compact view raises an error: the display
must then retain the relevant denominator distinction.

Alternative readouts such as NLL-weighted voting and best-single-input
selection can be displayed alongside the declared default reference.
Any stated changed-patient count or equality of metrics must follow the
supplied records; the display helper cannot establish a manuscript's results.

`tests/test_table_presentation.py` uses artificial aggregate fixtures for these
contracts, including preservation of source values and seed identity. These
software tests do not constitute new patient results or reproduction of the
paper's experiments.

## Patient-bootstrap interval figure

`evaluation/plot_patient_uncertainty.py` reads supplied Supplementary Data S1
records and plots existing estimates and intervals without resampling or
performing inference. Panels A–B select all 20 same-seed paired macro-F1
differences (two cohorts, two comparators, five seeds). Panels C–E select all
45 small-class Recall estimates (three classes, three models, five seeds).
The axes distinguish paired differences from Recall proportions. A single
seed legend uses both colors and marker shapes; the difference panels retain
negative bounds and show a zero reference line.

See `docs/patient_uncertainty.md` for usage and validation. The repository
contains code and synthetic tests only, not S1, manuscript files, or real-data
mapping outputs. Preserve the underlying statistical-detail file with its
protocol identity and source evidence when sharing these legacy plots.
