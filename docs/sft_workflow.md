# Preparing five independent SFT runs

`training/prepare_sft_runs.py` prepares LLaMA-Factory datasets, configurations and commands for five separate training runs. Its output status is `prepared_not_trained`. The preparation step does not launch training or supply pretrained experiment checkpoints.

## Inputs

Supply an existing local model directory containing `config.json`, a fitting source, a separate validation source, and the root containing their already prepared three-phase images. Source files may be JSON arrays, JSONL/NDJSON, or their gzip-compressed forms.

Each source record has:

- `metadata.logical_key`, `metadata.slice_index`, `metadata.patient_id`, `metadata.statistical_patient_id` and `metadata.positive`.
- One user/assistant exchange, optionally preceded by one system message.
- Exactly three image segments in the user content, ordered a/p/v, and the user instruction in a text segment. The files must exist. Relative image paths resolve against `--data-root`.
- Exactly one assistant text segment containing a valid `{"detections": [...]}` target. Every phase has explicit lesion records or a `no-lesion` record with `bbox_2d: [-1, -1, -1, -1]`. Lesion boxes use normalized xyxy coordinates from 0 to 1000.

Classes are selected by `--data-mode`: `private`, `5cls` or `3cls`, as defined in `preprocess/build_verl_real961_dataset.py`. The preparation command validates targets against that class space, rejects duplicate input-group identities, and checks that fitting and validation sources have no overlapping source patient IDs, statistical patient IDs or resolved image paths. Invalid records cause failure; they are not silently dropped. Background input groups remain in the exported training data.

The validation file is exported separately. The generated default training configuration uses only the fitting dataset and selects the final training epoch. It does not automatically search hyperparameters, add validation examples to fitting data, or select a checkpoint using test results.

## Prepare the configurations

Run from the repository root, with paths to the actual prepared data and local base model:

```bash
python training/prepare_sft_runs.py \
  --model /path/to/local-model \
  --model-family qwen --template qwen3_vl \
  --data-mode private \
  --train-json /path/to/fitting.jsonl \
  --validation-json /path/to/validation.jsonl \
  --data-root /path/to/prepared-images \
  --artifacts /path/to/new-sft-artifacts \
  --checkpoints /path/to/new-sft-checkpoints
```

Defaults are:

| Setting | Value |
| --- | --- |
| Independent seeds and matching `data_seed` | 42, 2026, 3407, 1234, 5678 |
| Fine-tuning | Full parameters |
| Precision | BF16 |
| Learning rate / scheduler / warmup | 1e-5 / constant / 0 |
| Epochs | 6 |
| Token cutoff | 2048 |
| Per-device group batch | 2 |
| Gradient accumulation | 16 |
| Data-parallel processes | 4 on one node |
| Effective group batch | 2 × 16 × 4 = 128 |
| Checkpoint selection | Final epoch |

All five seeds are explicit in each configuration. `--seeds` accepts a different set of five distinct integer seeds. Precision, batch dimensions, epochs and the remaining schedule parameters have explicit CLI options; changes are recorded in the generated configuration and `sft_runs.json`.

The artifacts directory must be new. Before preparing any run, the command also refuses an existing `seed_<seed>` directory under the requested checkpoint root. Training configurations set `overwrite_output_dir: false` and `overwrite_cache: false`.

The output layout for the default Qwen/private setting is:

```text
new-sft-artifacts/
  sft_runs.json
  qwen/private/seed_42/
    llamafactory_data/private_sft_train.json
    llamafactory_data/private_sft_val.json
    llamafactory_data/dataset_info.json
    configs/llamafactory_sft.yaml
    commands/run_sft.sh
    README.md
  ... one directory for each remaining seed
```

`sft_runs.json` records source/config SHA256 values, exported source counts, seed assignments, checkpoint paths, effective batch, precision and discoverable preparation-environment package versions. These are preparation records. Retain the actual training environment, logs and checkpoints separately when training is executed.

## Launch training

Use an environment with a LLaMA-Factory installation and model/template support compatible with the supplied model. This repository's preparation command does not install or validate that external training runtime. For each seed, run its generated command script, for example:

```bash
bash /path/to/new-sft-artifacts/qwen/private/seed_42/commands/run_sft.sh
```

The script sets `FORCE_TORCHRUN=1`, `NNODES=1`, `NODE_RANK=0` and `NPROC_PER_NODE` to the configured data-parallel size before invoking `llamafactory-cli train`. Thus the requested process count is explicit, including when ambient shell variables have different values. Provide that many visible, suitable GPUs. Run each seed independently; the preparation command does not schedule GPU allocation between simultaneous training jobs.

For another model family, choose its compatible LLaMA-Factory `--template`. Qwen family names `qwen`, `qwen3vl` and `qwen3_vl` use the dedicated Qwen inference entry point. Other family names use the generic Hugging Face inference entry point; model support still depends on the installed model/runtime implementation.

## Connect actual predictions to canonical evaluation

To prepare evaluation commands as well, add all three options to the preparation command:

```bash
  --eval-json /path/to/frozen-test-inputs.jsonl \
  --eval-manifest /path/to/frozen-test-manifest.json \
  --eval-group-map /path/to/frozen-test-group-map.csv
```

The manifest independently supplies the entire patient cohort, patient reference diagnoses and explicit phase-wise GT annotations. The CSV map supplies `logical_key`, `slice_index`, `group_id` and canonical `patient_id`; add `source_patient_id` if inference uses a different source patient identifier. Freeze these inputs and the evaluation split before comparing methods. Preparation checks that all three files are supplied and exist; the conversion/evaluation steps validate their contents and complete group coverage.

After each actual checkpoint has been produced, run its `commands/run_eval_sft.sh`. The generated script runs, in order:

1. `inference/infer_qwen_real961.py` or `inference/infer_hf_vlm_revision.py` on that seed's checkpoint, using greedy decoding, 256 new tokens and the same seed.
2. `evaluation/import_inference_predictions.py` with the explicit frozen group map. Every manifest group must have a prediction record. Invalid or empty completions remain as records.
3. `evaluation/evaluate_paper_protocol.py`, attaching `--model-name "qwen SFT"` (or the chosen family name plus ` SFT`) and `--model-seed <seed>`.

The final outputs are under each run's `eval/metrics/`, including `metrics.json`, patient predictions and group audit. The canonical evaluator retains patients without valid votes in classification denominators and exports the per-patient detection sufficient statistics required for paired comparisons.

Use the actual five `metrics.json` paths with the summarizer or with [joint seed statistics](joint_seed_statistics.md). The comparison spec's model name must match the exported identity, such as `qwen SFT`; do not relabel copies of one run as five independent seeds. Re-evaluating saved predictions does not substitute for missing independent training runs.

## Verification scope

Run the focused workflow checks with:

```bash
python -m pytest tests/test_sft_workflow.py -q
```

The tests exercise the preparation CLI in a subprocess without site packages, use valid small PNG files, check five complete exports and seed/batch settings, reject malformed records and fitting/validation overlap, and check preservation of existing outputs. A shell executable double captures the configured launcher environment and quoted paths. It does not launch torchrun workers or perform GPU training.

With the optional inference dependencies installed, the two inference entry points are also exercised with their real `--help` parsers. Synthetic CSV predictions are passed through the actual generated conversion and canonical evaluation commands, including an invalid completion that must remain in the patient denominator. These tests validate interfaces and bookkeeping. No GPU training, model-quality reproduction, or recovery of historical experimental results is claimed by this test suite.
