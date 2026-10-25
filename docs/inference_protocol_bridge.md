# Inference output and canonical evaluation

`inference/infer_qwen_real961.py` and `inference/infer_hf_vlm_revision.py` produce one CSV record for each `(logical_key, slice_index)` in their input JSONL. `evaluation/import_inference_predictions.py` connects these records to the canonical evaluator through an explicit group map and an independent evaluation manifest.

Prepare and freeze the manifest and group map from the dataset's patient identities, complete input groups, and phase-specific lesion annotations before inspecting model predictions. The manifest supplies ground truth independently of the inference CSV's `class` column. Input preparation defines the eligible CT groups; the converter checks their coverage and consistency against the supplied manifest.

## Explicit group map

The CSV map requires the following columns:

```csv
logical_key,slice_index,group_id,patient_id,source_patient_id
source-series-7,12,canonical-group-0042,center-A-patient-9,patient-9
```

`logical_key` and `slice_index` identify the inference input. `group_id` and `patient_id` must match a group in the evaluation manifest. Optional `source_patient_id` states the exact patient ID written by inference when it differs from the canonical ID. Without that column, or with an empty value, the inference patient ID must equal the canonical patient ID. This is an explicit identity mapping, not an automatic name or path heuristic.

Each manifest group must appear exactly once in the map, and each mapped source key must appear exactly once in inference CSV. Unknown groups, duplicated source keys or groups, inconsistent patient IDs, and missing prediction records fail conversion. Duplicate or empty CSV headers and inconsistent row widths also fail. The converter requires a new output path and does not overwrite predictions or their provenance file.

```bash
python evaluation/import_inference_predictions.py \
  --csv results/inference.csv \
  --manifest data/evaluation_manifest.json \
  --group-map data/inference_group_map.csv \
  --output results/canonical_predictions.jsonl

python evaluation/evaluate_paper_protocol.py \
  --manifest data/evaluation_manifest.json \
  --predictions results/canonical_predictions.jsonl \
  --output-dir results/canonical_evaluation \
  --iou-thresholds 0 0.5 0.9
```

The prediction JSONL contains `group_id`, the unchanged CSV `raw` string, and `completion_nll`. Malformed JSON and empty raw strings remain explicit prediction records. A missing raw field caused by a malformed CSV row is an input error. Empty NLL becomes JSON `null`; explicit nonfinite NLL becomes a string such as `"nan"` or `"inf"`, keeping the JSON valid. Negative finite NLL is rejected. The inference entries currently write nonfinite NLL as an empty CSV field, so those files do not distinguish missing from nonfinite NLL.

Canonical parsing decides whether each completion supplies a diagnostic vote. Nonfinite or missing NLL does not remove an otherwise eligible vote. NLL is only used in the declared tie rule; an unavailable NLL cannot defeat a finite value in a tie. Patients with no eligible votes, including patients present in the manifest with no complete input groups, remain in the classification denominator. The converter never takes an intersection of available rows to obtain a smaller cohort.

`canonical_predictions.jsonl.provenance.json` records input CSV, group-map and manifest SHA-256 digests, the output digest, and the number of converted groups. Preserve this file alongside the evaluation outputs and the original inference CSV and resume sidecar.

## Decoding and resuming

Both entries default to `do_sample=False`, `num_beams=1`, and `num_return_sequences=1`. Sampling requires an explicit option, for example:

```bash
python inference/infer_qwen_real961.py \
  --model-dir checkpoints/model \
  --json-path data/test.jsonl \
  --data-root data \
  --output results/sample-seed-2026.csv \
  --decoding sample --temperature 0.7 --top-p 0.9 --top-k 20 --seed 2026
```

Temperature must be finite and positive, `top_p` must be in `(0, 1]`, and `top_k` must be a nonnegative integer. Non-default sampling parameters with greedy mode are rejected rather than silently ignored. Declare the settings and seeds used by each experiment; using a new sampling setting is a new experiment.

Sampling resets the random generators immediately before each generation using a stable hash of the run seed and input key. With the same model, prepared inputs and execution environment, skipping completed inputs on resume therefore does not advance the random stream used for another input. Record the hardware, library versions and any stochastic image-preprocessing settings with the run.

The `.csv.resume.json` sidecar binds output bytes to model, processor, input and image hashes, runtime versions, entry/helper code hashes, and settings including decoding and NLL definition. HF also records its optional `--limit`; this limit is for canary runs and must not be used for formal cohort metrics. A changed identity or modified/uncommitted partial output is rejected. Keep the partial CSV and sidecar together when resuming.

## Completion NLL

The two single-sequence entries request `output_logits=True` and calculate mean token negative log-probability from the model's unprocessed logits. This prevents sampling temperature and truncation from redefining the NLL used to break ties. A model wrapper that does not return these logits fails; the entries do not fall back to processed generation scores. Because these calls generate one sequence without batch padding, the generated EOS token is included even when its token ID equals the tokenizer's padding ID. The recorded definition is `unprocessed_logits_single_sequence_v1`.

Processed-score NLL or NLL excluding EOS uses a different definition and can produce different patient ties. Use a consistent NLL definition across a comparison and record checkpoint identities and decoding settings for each training seed.

