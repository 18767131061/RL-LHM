# Metrics pipeline

`evaluation/evaluate_paper_protocol.py` evaluates the explicit 2D protocol in the
revised manuscript. It calls `evaluation/schema.py` for output validation,
`evaluation/detection.py` and `matching/protocols.py` for all three matching
protocols, and `aggregation/patient.py` for patient voting. It requires only the
core NumPy/SciPy environment; it does not load images, models, or Torch.

```bash
python evaluation/evaluate_paper_protocol.py \
  --manifest /path/to/evaluation_manifest.json \
  --predictions /path/to/predictions.jsonl \
  --output-dir ./outputs/paper_protocol_evaluation \
  --iou-thresholds 0 0.5
```

Use a fresh output directory. `0` means strictly positive IoU; positive thresholds
mean IoU greater than or equal to the threshold. Every requested threshold is
evaluated separately for `class_agnostic`, `same_class_priority` (the manuscript's
same-class-first), and `geometry_first_class_aware`.

## Evaluation manifest

The JSON manifest defines the entire patient denominator separately from its
complete input groups. Only already prepared, complete, corresponding-position
three-phase groups belong in `input_groups`. Retain every evaluation patient in
`patients`, including patients with no complete group. Exclude incomplete groups
during documented input preparation; the evaluator rejects incomplete group
records and does not fill or copy phase images or annotations.

```json
{
  "schema_version": 1,
  "classes": ["HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"],
  "patients": [
    {"patient_id": "synthetic-p1", "truth": "HCC"},
    {"patient_id": "synthetic-p2", "truth": "ICC"}
  ],
  "input_groups": [
    {
      "group_id": "synthetic-p1-slice-001",
      "patient_id": "synthetic-p1",
      "phases": {
        "a": {
          "image_ref": "synthetic/a.png",
          "ground_truth": [
            {"label": "HCC", "bbox_2d": [100, 200, 300, 400]},
            {"label": "ICC", "bbox_2d": [500, 600, 700, 800]}
          ]
        },
        "p": {
          "image_ref": "synthetic/p.png",
          "ground_truth": [
            {"label": "HCC", "bbox_2d": [110, 210, 320, 430]}
          ]
        },
        "v": {"image_ref": "synthetic/v.png", "ground_truth": []}
      }
    }
  ]
}
```

- `classes` is the frozen active class order used for macro metrics and fixed
  tie-breaking. Exact aliases accepted by `evaluation/schema.py`, such as `HH`
  and `NC`, are normalized; aliases cannot create duplicate active classes.
- `patients` must list each patient exactly once, with the patient-level reference
  diagnosis in `truth`. This label is used only to score the patient endpoint.
- `group_id` is unique across the evaluation. `patient_id` must occur in `patients`.
- Each `phases.a/p/v` has a non-empty `image_ref` and its own `ground_truth` list.
  The evaluator treats image references as identifiers and does not verify files,
  registration, correspondence, or slice-selection completeness.
- Each GT lesion has its own active-vocabulary `label` and finite normalized
  `[x1,y1,x2,y2]` `bbox_2d` in `[0,1000]`, with positive width and height. Multiple
  lesions and different lesion labels within one phase are supported.
- `ground_truth: []` explicitly means a lesion-negative phase. It does not mean
  unknown or missing annotation. No GT background sentinel is used. Missing
  annotations are errors; patient diagnosis and another phase's boxes are never
  substituted for them.

## Prediction records

Supply one JSON object per line, with exactly one output record for every manifest
group. Preserve the generated completion as a string in `raw`:

```json
{"group_id":"synthetic-p1-slice-001","raw":"{\"detections\":[{\"phase\":\"a\",\"label\":\"HCC\",\"bbox_2d\":[100,200,300,400]}]}","completion_nll":0.3}
```

`completion_nll` is the generated-token average NLL. The evaluator consumes it; it
does not compute or calibrate model likelihoods. A missing field or `null` denotes
a missing score. Explicit strings `"NaN"`, `"Infinity"`, and `"-Infinity"` can
record non-finite scores in valid JSON; they are excluded from class means without
removing the diagnostic vote. Other strings are rejected.

Duplicate, unknown, or missing group records stop evaluation before writing
results. An absent output record is not imputed as an invalid completion. Keep
actual malformed or empty generated text in `raw`; an observed invalid completion
is different from a missing inference record.

## Endpoint rules

1. Validate the entire completion with the active vocabulary. Invalid completions
   contribute no predictions or diagnostic vote; their annotated positive-phase
   GT lesions remain false negatives. A missing output phase alone does not make
   a completion invalid.
2. Each valid completion with at least one diagnostic lesion label yields one
   group vote: largest label count, then frozen class order. Background sentinels
   do not vote. GT positivity does not determine diagnostic voting eligibility.
3. Retain every eligible group vote for a patient. Patient ties use the mean
   finite completion NLL for each tied class; classes with no finite score receive
   infinity. Fixed class order breaks remaining ties only among minimum-NLL
   candidates. No top-k selection or likelihood-weighted voting occurs.
4. Every manifest patient remains in the classification denominator. A patient
   with no eligible vote has an undefined prediction and counts as incorrect.
5. Detection includes only phase slices with non-empty GT lesion lists. All valid
   predictions on those phases enter one-to-one matching; phase-negative
   predictions are counted separately in the audit, outside detection FP and its
   denominators. Their valid group diagnostic vote is still retained.
6. CA maximizes total IoU; GFCA freezes that assignment and rejects wrong-label
   pairs without rematching; same-class-first maximizes the number of same-class
   pairs first and total IoU second. Threshold-rejected pairs produce both FP and
   FN. Count invariants are `TP+FP=eligible predictions` and `TP+FN=GT lesions`.

## Outputs and validation limits

- `metrics.json`: fixed-label patient metrics, all three detection protocols at
  each threshold, GT-denominator Mean IoU, input/patient coverage, excluded-phase
  counts, per-group/per-patient details, and SHA-256 hashes of the two input files.
- `patient_predictions.csv`: every manifest patient, including undefined
  predictions, with input-group and eligible-vote counts.
- `input_group_audit.csv`: every manifest group, schema validity, diagnostic
  eligibility, NLL status, and the group vote. Raw generated text is not copied.

These are single-run endpoint summaries, not new bootstrap or paired statistical
results. Run separately for each frozen cohort/model/seed. The command checks
coverage against the supplied manifest; it cannot establish complete original CT
coverage or validate patient annotation provenance. Old
`evaluate_sim999_predictions.py` and `evaluate_real961_predictions.py` utilities
remain legacy paths and must not be substituted for this interface without
reconciling their contracts.

The synthetic CLI tests establish the repaired interface's behavior. They do not
prove that this code produced the manuscript's historical results or that its
published tables have been reproduced. That requires confirming the historical
evaluation procedure and running the corresponding frozen prediction records.

```bash
python -m pytest tests/test_paper_protocol_cli.py -q
```

Statistical post-processing is available separately in `evaluation/statistics.py`,
`evaluation/paired_patient_tests.py`, and `statistics/`. Use the same patient
identifiers, cohort definition, and frozen prediction records when constructing
paired inputs; an aggregate metric alone cannot reconstruct patient-level pairs.
