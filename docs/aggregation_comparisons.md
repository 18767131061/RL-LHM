# Patient aggregation comparisons

`evaluation/compare_aggregation_strategies.py` applies three patient readouts to the same saved group predictions and completion NLL values for every model and seed. It writes canonical evaluator exports, coverage and changed-patient summaries, and joint five-seed patient-paired statistics for Table 7 and supplementary aggregation comparisons.

## Readouts

Every schema-valid group containing an active diagnostic lesion label contributes one group vote. Within-group ties use the manifest class order. All eligible groups enter the following patient readouts:

- `majority_nll_tiebreak`: plurality count, then lowest mean finite completion NLL among tied classes, then fixed class order. A tied class without finite NLL receives an infinite mean for this comparison.
- `likelihood_weighted_vote`: sum `exp(-clip(NLL, -50, 50))` per class. Missing or non-finite values use the upper median of the patient's finite eligible-group NLL values, or zero if none are finite. Only classes receiving votes are candidates. Totals within `1e-12 * maximum_total` use fixed class order.
- `best_likelihood_input`: choose the eligible group with the lowest finite NLL, then earliest manifest input order, then fixed class order. Missing/non-finite NLL ranks as positive infinity; if all are missing, input order determines the selection.

Patients without an eligible vote retain `null` predictions and remain in the full classification denominator. These readouts preserve each patient's eligible votes, coverage and lesion predictions. Detection counts and matching results therefore remain identical across strategies within a model/seed.

For a single canonical export, select the readout explicitly:

```bash
python evaluation/evaluate_paper_protocol.py \
  --manifest data/private_test_manifest.json \
  --predictions runs/full_42/predictions.jsonl \
  --model-name Full --model-seed 42 \
  --aggregation-strategy likelihood_weighted_vote \
  --output-dir runs/full_42/weighted_evaluation
```

The default is `majority_nll_tiebreak`. Each export records `aggregation_strategy` and a corresponding distinct `evaluation_identity.aggregation_rule`.

## Complete five-seed comparison

Create one specification per cohort. Paths are relative to the specification file. Each model needs exactly one saved prediction JSONL per declared seed; a file cannot represent multiple model/seed runs. The same prediction source is reused internally for all three strategies.

```json
{
  "schema_version": 1,
  "cohort": "private",
  "manifest": "data/private_test_manifest.json",
  "seeds": [42, 2026, 3407, 1234, 5678],
  "reference_model": "Full",
  "iou_thresholds": [0.5],
  "models": {
    "GRPO": [
      {"seed": 42, "predictions": "runs/grpo_42/predictions.jsonl"},
      {"seed": 2026, "predictions": "runs/grpo_2026/predictions.jsonl"},
      {"seed": 3407, "predictions": "runs/grpo_3407/predictions.jsonl"},
      {"seed": 1234, "predictions": "runs/grpo_1234/predictions.jsonl"},
      {"seed": 5678, "predictions": "runs/grpo_5678/predictions.jsonl"}
    ],
    "Full": [
      {"seed": 42, "predictions": "runs/full_42/predictions.jsonl"},
      {"seed": 2026, "predictions": "runs/full_2026/predictions.jsonl"},
      {"seed": 3407, "predictions": "runs/full_3407/predictions.jsonl"},
      {"seed": 1234, "predictions": "runs/full_1234/predictions.jsonl"},
      {"seed": 5678, "predictions": "runs/full_5678/predictions.jsonl"}
    ]
  }
}
```

```bash
python evaluation/compare_aggregation_strategies.py \
  --spec aggregation_private.json \
  --output-dir results/aggregation_private \
  --permutations 100000 --bootstrap 10000 --random-seed 2026
```

The three repeat/seed flags above are the defaults. The output directory must be new. All predictions and source identities are validated before writing evaluator outputs. One-model specifications are also supported. `--exact-permutations` enumerates every patient mask for cohorts with at most 16 patients.

The statistical reference is **Full with `majority_nll_tiebreak`** when `reference_model` is `Full`. All other model/strategy rows, including GRPO alternatives, are compared with that reference. This supplies the Table 7 inferential baseline. Changed-patient counts instead compare each strategy with **its own model and seed's default**; the default count is zero. A changed patient means any change in the final prediction, not only a change in correctness.

Coverage is the number of determinate predictions divided by all manifest patients. Selective accuracy is correct predictions divided by determinate predictions; it is `null` if no prediction is determinate. Overall accuracy uses every manifest patient. All five per-seed values are retained. If any seed's selective accuracy is undefined, its five-seed mean and SD are `null`; `defined_seed_count` reports the number of defined values without dropping seeds from the reported list.

## Outputs and table input

- `runs/model-N/seed-N/STRATEGY/` contains `metrics.json`, `patient_predictions.csv` and `input_group_audit.csv` for every model/seed/strategy. Model indices follow specification order. Canonical run identities retain the actual model name and seed.
- `aggregation_comparison.json` contains `cohort`, `seeds`, `reference_model`, `reference_arm`, `arms`, and relative `joint_statistics`/`joint_spec` paths. Each arm has `arm`, `model_name`, `aggregation_strategy`, `runs` and `summary`. Arm names are `MODEL::STRATEGY`. Each run records `seed`, its relative canonical `metrics` path, `changed_patients`, `coverage` and `selective_accuracy`.
- Each arm's `summary` maps endpoint names to `values`, `mean`, `sample_sd` and `defined_seed_count`. Classification names include `classification:accuracy` and `classification:macro_f1`; other classification and detection endpoints follow the joint statistics naming convention. Coverage, selective accuracy and changed counts use `coverage`, `selective_accuracy` and `changed_patients`.
- `aggregation_summary.csv` contains the same descriptive series. Means and SDs use unrounded seed metrics with `ddof=1`. The paper table exporter separately applies any declared classification seed rounding convention.
- `joint_seed_spec.json`, `joint_seed_statistics.json` and `joint_seed_statistics.csv` contain the complete declared comparison and all endpoint statistics. Canonical source file SHA256 identities bind statistics to each arm's files. Raw p-values are unadjusted; there is no reference self-comparison.

Use `aggregation_comparison.json` as the table exporter's `aggregation_comparison.path`, with the same cohort ID and model names as the main table specification. The table exporter verifies the canonical default anchors and the paired-statistics sources. See [paper table instructions](paper_tables_and_confusion.md) for the table specification.

## Pairing contracts

The generated joint specification uses `comparison_mode: "aggregation"` for one model and `"aggregation_table"` for multiple models. Its `arms` object explicitly declares every named arm's `model_name` and `aggregation_strategy`, and its `models` object supplies the five canonical files per arm. Each model must include a default anchor. A table comparison can cross model identities only after each alternative is validated against its own default anchor.

Within a model and seed, strategy comparisons require the same prediction source SHA256, manifest identity, ordered group records including completion NLL, patient truth, eligible votes, input counts, coverage, detection scope and per-patient detection counts. Across all models and seeds, manifest/GT, patient cohort, ordered output/macro classes and detection protocol/threshold definitions must match. The ordinary `models` mode retains its shared aggregation-rule requirement; simply mixing strategies in an ordinary model comparison is rejected.

Joint inference follows [joint seed statistics](joint_seed_statistics.md): one patient swap or bootstrap multiplicity applies to the patient's entire five-seed vector, with metrics recomputed within each seed before averaging. The default uses 100,000 two-sided permutations with the plus-one correction and 10,000 paired patient bootstrap replicates. Across-seed sample SD and patient-bootstrap uncertainty remain separate summaries.

Run the focused checks with:

```bash
python -m pytest -q tests/test_aggregation_comparisons.py
```
