# Presenting existing statistical results

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
fixed patient-level evaluation and the prespecified family of 60 comparisons.

The view rejects missing/duplicate identities, other endpoints, and reversed
candidate labels. It returns 12 cohort/comparator/endpoint summaries. Each has
the minimum and maximum of its five existing `delta` and `q` values and all
five seed-specific source records in `seed_details`. Delta means Full minus
the comparator. A range is a descriptive range across seeds; it is not a
confidence interval. The actual patient-bootstrap difference CIs remain
identified by seed. These `seed_details` are an interchange view for preserving
existing statistics; the manuscript presents compact five-seed summaries and
provides the individual records in Supplementary Data S1. No cross-seed CI,
mean p/q, or new Macro-Precision test is generated.

The q-values originate from the existing 60-comparison BH correction. A
presentation change must consume those values without rerunning tests or BH.
The display helper checks the 60 identities but cannot authenticate supplied
q-values or reconstruct their patient-level evidence. Preserve each result's existing precision and aggregation
convention: the main summaries use full-precision seed results, while some
intermediate or auxiliary results explicitly summarize retained four-decimal
seed metrics. `aggregation/summarize_seed_results.py` uses `ddof=1` on the
supplied values without rounding them. Do not reconstruct unavailable precision
from displayed table cells or replace an established summary with a different
rounding convention.

## Locked five-seed summaries and Supplementary Data S1

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
summary from rounded seed-level table cells. Sample SD describes five training
seeds and uses denominator n−1; it is distinct from a patient interval.

Primary `classification_private`/`classification_public` tables render one
row per model: Accuracy, macro-Precision, macro-Recall, macro-F1, and significant-
seed counts. For SFT and vanilla GRPO, the CSV `paired_difference` records must
contain exactly the five unique seed IDs for each tested endpoint against
`model: Full RL-LHM`. The exporter counts their existing `bh_q < 0.05`, in
Accuracy/Recall/F1 order. The current study has `0/5, 0/5, 0/5` in each comparator
row. Other models and the Full reference use a dash. Macro-Precision is never
included in these counts. Neither these counts nor `best` formatting denote
a test on the means; nonsignificance does not establish equivalence.

Per-class and rare-class fragments share the canonical class means/SDs.
The patient counts 19 (private ICC), 9 (private FNH) and 20 (public ICC) identify
class-specific test patients. LOCO direction tables retain three model rows
and their existing Accuracy, macro-Recall and macro-F1 summaries. They do not
borrow the primary paired tests or fabricate an unreported Macro-Precision.
Different cohorts, classes, thresholds, configurations and directions remain
separate scientific groups.

The formal Supplementary Data S1 CSV holds original seed-level estimates,
patient-bootstrap/Wilson intervals, p/q values and discordant/correct-patient
counts once, under their full identities. It adds statistical detail to the
main mean/SD tables rather than duplicating the summary performance tables.
Its record types are `paired_difference`, `model_performance`,
`class_performance`, `loco_performance`, and `auxiliary_performance`.
Blank statistical fields mean inapplicable or unreported, never zero or
nonsignificant. Existing values are preserved, including adverse comparisons.
No p-value, q-value or interval endpoint is averaged, pooled or substituted
for a mean-level inference.

`tests/test_export_locked_summary_tables.py` uses artificial inputs to verify
seed grouping, preservation of decimal precision, the q < 0.05 boundary,
missing/duplicate seed rejection, and the separation of untested models.
These CPU tests validate the export interface, not the paper's experiments.

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

The manuscript's supplementary factorial detection tables
(`tab:app_factorial_detection` and `tab:app_factorial_detection_3`) retain the
six intermediate component configurations. The vanilla-GRPO and Full endpoint
rows remain in the corresponding main detection tables
(`tab:private_detection_audit` and `tab:public_detection_audit`). The appendix
therefore supplies additional configurations without repeating those endpoints.
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

In the manuscript, `tab:app_aggregation` displays the two alternative readouts,
NLL-weighted voting and best-single-input selection. The default majority-plus-
NLL classification summaries remain in `tab:unified_private` and
`tab:unified_public`; the aggregation caption records the default reference's
zero changed-patient count (0.0000 ± 0.0000). This removes duplicate default
performance rows while keeping the alternatives and their variability.

`tests/test_table_presentation.py` uses artificial aggregate fixtures for these
contracts, including preservation of source values and seed identity. These
software tests do not constitute new patient results or reproduction of the
paper's experiments.
