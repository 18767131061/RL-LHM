# Joint statistics over five fixed seeds

`evaluation/joint_seed_statistics.py` computes patient-paired comparisons from canonical evaluator exports. It evaluates each seed separately, then averages the five metric differences. It does not ensemble predictions, pool five confusion matrices into one metric, average p-values, or treat five observations of one patient as independent patients.

## 1. Export each run

Use the same frozen evaluation manifest for every seed and model in a comparison. Keep run identities explicit:

```bash
python evaluation/evaluate_paper_protocol.py \
  --manifest data/private_test_manifest.json \
  --predictions runs/full_42/predictions.jsonl \
  --model-name Full --model-seed 42 \
  --iou-thresholds 0.25 0.50 0.75 \
  --output-dir runs/full_42/evaluation
```

Repeat for the other seeds and comparison models. The ordinary evaluator accepts omitted identity flags; joint statistics requires both flags and five independently trained runs per model.

The existing `metrics.json`, `patient_predictions.csv` and `input_group_audit.csv` outputs remain available. `metrics.json` includes:

- `run_identity`: the declared model name and training/inference seed.
- `evaluation_identity`: SHA256 of the supplied manifest's canonical JSON content and the aggregation rule identifier. This hash includes the supplied phase-wise GT annotations and input definitions.
- `patient_detection`: every manifest patient, including patients with no complete input group. For each named matching protocol and IoU threshold, records contain TP, FP, FN, GT, predicted-object counts, and ungated assigned `iou_sum`.

Statistics needs these per-patient records. Old exports containing only cohort totals cannot be converted to paired detection tests by distributing totals over patients. Rerun evaluation on the actual saved predictions and original manifest.

## 2. Declare the complete comparison set

Save a specification such as `comparison.json`. The paths are relative to that specification file. The names and seeds must match the identities embedded in each evaluator output.

```json
{
  "schema_version": 1,
  "seeds": [42, 2026, 3407, 1234, 5678],
  "reference": "Full",
  "models": {
    "Full": [
      {"seed": 42, "metrics": "runs/full_42/evaluation/metrics.json"},
      {"seed": 2026, "metrics": "runs/full_2026/evaluation/metrics.json"},
      {"seed": 3407, "metrics": "runs/full_3407/evaluation/metrics.json"},
      {"seed": 1234, "metrics": "runs/full_1234/evaluation/metrics.json"},
      {"seed": 5678, "metrics": "runs/full_5678/evaluation/metrics.json"}
    ],
    "SFT": [
      {"seed": 42, "metrics": "runs/sft_42/evaluation/metrics.json"},
      {"seed": 2026, "metrics": "runs/sft_2026/evaluation/metrics.json"},
      {"seed": 3407, "metrics": "runs/sft_3407/evaluation/metrics.json"},
      {"seed": 1234, "metrics": "runs/sft_1234/evaluation/metrics.json"},
      {"seed": 5678, "metrics": "runs/sft_5678/evaluation/metrics.json"}
    ]
  }
}
```

Add GRPO, detector baselines, or ablations as further model entries. The program compares the reference against every other supplied model and reports every supported endpoint. A specification must contain exactly five distinct seeds per model; arbitrary five-seed sets are supported, and the example uses the manuscript's set. A separate specification is needed for a different cohort, split, aggregation rule, output/macro class space, or evaluation protocol.

The loader rejects duplicate/missing patients, different truths, different manifest hashes, class order changes, differing threshold sets, seed identity mismatches, and per-patient GT count differences. Count identities and agreement between per-patient units and aggregate classification/count summaries are checked. Patient row order can differ; pairing uses `patient_id`. Reports must preserve abstentions as JSON `null`; a patient without a vote remains in the denominator.

## 3. Run paired inference

```bash
python evaluation/joint_seed_statistics.py \
  --spec comparison.json \
  --output-dir statistics/private \
  --permutations 100000 --bootstrap 10000 \
  --random-seed 2026 --adjustment bh
```

The defaults are 100,000 permutations, 10,000 bootstrap replicates, random seed 2026, and `--adjustment none`. Raw p-values are always retained. `--adjustment bh` additionally applies Benjamini–Hochberg to **all comparator × metric rows in that invocation**, including class-wise and threshold-specific endpoints. Both the family definition and its size are recorded. The correction family is limited to that invocation. Include all intended comparators before defining the family, and prespecify a different primary family externally if needed.

Outputs are `joint_seed_statistics.json` and `joint_seed_statistics.csv`; an existing output is not overwritten. They contain per-seed values, mean and sample SD (`ddof=1`), reference-minus-comparator differences, raw p-values, optional BH q-values, reference/comparator/difference patient-bootstrap intervals, sample sizes, identities, and file hashes.

### Endpoints

- Patient classification: Accuracy, macro-Precision, macro-Recall, macro-F1, and each macro class's Precision/Recall/F1.
- Detection: pooled lesion Precision/Recall/F1 at each supplied threshold, separately for `class_agnostic`, `geometry_first_class_aware`, and `same_class_priority`.
- Mean IoU: ungated IoU sum divided by all GT lesions, once per matching protocol. It is threshold independent and is not counted repeatedly as a new hypothesis at every threshold.

The fixed macro class list is retained even when a bootstrap sample has no member of a class; zero denominators yield zero. An output class excluded from macro averaging, such as FNH in one LOCO direction, remains a possible prediction and counts as an error for a shared-class truth. Detection scope remains the evaluator's lesion-positive-phase scope; this procedure does not add false positives from excluded negative phases.

### Resampling definition

For a reference model A, comparator B, seed s and metric M, the observed statistic is

`mean_s [M(A_s) - M(B_s)]`.

One permutation draws one binary mask over patients. If patient p is swapped, that patient's full five-seed classification vector and detection sufficient statistics are exchanged between A and B. Each metric is recomputed within each seed, including nonlinear F1 and pooled detection ratios, before taking the mean difference. Two-sided Monte Carlo p-values use `(extreme + 1) / (repeats + 1)`, with the absolute tail and a `1e-12` comparison tolerance.

One bootstrap replicate draws N patients with replacement; the implementation uses equivalent multinomial patient multiplicities. The same multiplicities apply to both models, every seed and every metric. Reference means, comparator means and paired mean differences receive 95% percentile intervals using NumPy's explicit linear 2.5th/97.5th quantiles. Permutations and bootstraps use separate PCG64 streams derived from the recorded random seed; each comparator uses the same resampling streams over the fixed population. These intervals describe patient-sampling variation **conditional on the fixed five trained seeds**. Across-seed SD describes variation among the five trained models.

For cohorts with at most 16 patients, `--exact-permutations` enumerates every patient mask. Its output says `exact_enumeration` and uses the exact extreme fraction, without the Monte Carlo plus-one adjustment.

## 4. Seed summaries and the legacy interface

```bash
python aggregation/summarize_seed_results.py \
  --metrics runs/full_42/evaluation/metrics.json \
            runs/full_2026/evaluation/metrics.json \
            runs/full_3407/evaluation/metrics.json \
            runs/full_1234/evaluation/metrics.json \
            runs/full_5678/evaluation/metrics.json \
  --seeds 42 2026 3407 1234 5678 \
  --output-json statistics/full_summary.json \
  --output-csv statistics/full_summary.csv
```

Canonical summaries require complete, matching model/seed identities, reject repeated input paths, recompute ratios from patient exports, and validate the shared population. For exports missing an identity, evaluate the saved predictions with their corresponding model and seed identifiers. They preserve all three matching protocol names. The older `classification_overall`/`lesion_detection` schema remains supported as a distinctly labelled legacy schema; its `class_aware` key is not inferred to mean GFCA or SCF, and it cannot be mixed with canonical exports. The summary command calculates descriptive seed statistics only.

For auxiliary classification summaries, add `--classification-seed-decimals 4`.
Each seed's classification and class-wise metrics are rounded to four decimal
places using decimal round-to-nearest, ties-to-even before their mean and sample
SD are calculated. Detection ratios retain full precision. The JSON records the
rounding convention and unrounded classification values. Patient-paired
permutation and bootstrap calculations continue to use the per-patient counts.

`evaluation/paired_patient_tests.py` remains a **legacy single-seed** tool. It accepts `logical_key` or canonical `patient_id` CSV keys, and labels its output accordingly. It is not the joint-five-seed test and its outputs must not be averaged to approximate that test.

## Tests

