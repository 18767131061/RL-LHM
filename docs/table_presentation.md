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
identified by seed, either beside the main metrics or in one detailed table.
No cross-seed CI, mean p/q, or new Macro-Precision test is generated.

Compute the BH correction upstream with `statistics/apply_r1_1_bh.py`, once
both cohorts' required paired records are available. The display helper checks
the 60 identities but cannot authenticate the supplied q-values or replace that
upstream calculation. Preserve each result's existing precision and aggregation
convention: the main summaries use full-precision seed results, while some
intermediate or auxiliary results explicitly summarize retained four-decimal
seed metrics. `aggregation/summarize_seed_results.py` uses `ddof=1` on the
supplied values without rounding them. Do not reconstruct unavailable precision
from displayed table cells or replace an established summary with a different
rounding convention.

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

`tests/test_table_presentation.py` uses artificial aggregate fixtures for these
contracts, including preservation of source values and seed identity. These
software tests do not constitute new patient results or reproduction of the
paper's experiments.
