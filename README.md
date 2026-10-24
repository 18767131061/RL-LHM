# RL-LHM

Training components and evaluation tools for three-phase, two-dimensional liver lesion analysis. The repository includes prepared-data conversion, SFT configuration, RL integration, structured inference, lesion matching, patient aggregation, and five-seed statistics. Clinical images, trained checkpoints and per-run study predictions are not included.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The metric and data-validation tools use the core dependencies. Install `requirements-optional.txt` for model inference, plotting and RL components. SFT additionally needs LLaMA-Factory; RL needs compatible VeRL, vLLM and CUDA installations. Record the installed versions and applied patches for each run: these requirements files are dependency ranges, not a complete training environment lock.

## Prepare data

Supply approved, prepared three-phase inputs with patient IDs, disjoint train/validation/test partitions and explicit annotations for each phase. The converters do not reconstruct CT registration, slice correspondence or image selection from volumes.

- [Training inputs](docs/training_pipeline.md): JSONL conversion, class counts, eight component combinations and visual-cache inputs.
- [Evaluation manifest](docs/metrics_pipeline.md): the complete patient denominator, input groups and phase-specific lesions.
- [Detector export](docs/detector_export.md): separate train/validation/test phase images by default. `--refit-trainval` explicitly combines the development partitions and supplies no validation set.

Validate identities and freeze the evaluation manifest before inspecting predictions. Patients without a diagnostic vote remain in the classification denominator. Detection uses lesion-positive phase slices as specified by the evaluation protocol; negative-phase false positives are outside that endpoint.

## SFT and RL

Prepare five SFT runs from explicit fitting and validation inputs:

```bash
python training/prepare_sft_runs.py \
  --model /path/to/base-model --model-family qwen --template qwen3_vl --data-mode private \
  --train-json /path/to/train.jsonl \
  --validation-json /path/to/validation.jsonl --data-root /path/to/images \
  --artifacts ./outputs/sft-configs --checkpoints ./checkpoints/sft
```

This prepares configurations and launch scripts; it does not train a model. Defaults use seeds 42, 2026, 3407, 1234 and 5678, learning rate 1e-5, six epochs, device batch two, accumulation 16 and four data-parallel processes. Specify hardware and precision explicitly for your environment. The [SFT workflow](docs/sft_workflow.md) documents checkpoint paths and optional inference/evaluation scripts.

RL configurations can be inspected before launching a GPU run:

```bash
python training/run_verl_lhm_variant.py \
  --train-file /path/to/train.jsonl --model-path /path/to/sft-checkpoint \
  --output-dir ./outputs/rl-run --variant full_rl_lhm \
  --seed 2026 --scripts-dir . --n-gpus 2 --config-only
```

Use the [training integration instructions](docs/training_pipeline.md), [runtime preflight](docs/runtime_preflight.md), and [visual-feature extractor](docs/visual_feature_extraction.md) before actual training. Configuration-only execution does not exercise the external policy loss or a GPU model.

The [OAT planner and collector](docs/rars_oat.md) implement one severity parameter at a time, including the shared other-pair parameter. The [Cartesian search](docs/rars_joint_search.md) is a separate optional experiment. Neither interface contains the paper's validation scores or reconstructs past selections.

## Inference to evaluation

Run a local Qwen checkpoint, then convert its CSV with an explicit input-group map:

```bash
python inference/infer_qwen_real961.py \
  --model-dir /path/to/checkpoint --json-path /path/to/test.jsonl \
  --data-root /path/to/images --output ./outputs/inference.csv \
  --seed 2026 --decoding greedy

python evaluation/import_inference_predictions.py \
  --csv ./outputs/inference.csv --manifest /path/to/test_manifest.json \
  --group-map /path/to/input_group_map.csv \
  --output ./outputs/predictions.jsonl

python evaluation/evaluate_paper_protocol.py \
  --manifest /path/to/test_manifest.json --predictions ./outputs/predictions.jsonl \
  --model-name Full --model-seed 2026 --iou-thresholds 0 0.5 \
  --output-dir ./outputs/evaluation
```

The map must cover every manifest group exactly once. Empty or invalid generated text is retained as an explicit output. Missing, extra or duplicated records fail conversion. See the [inference bridge](docs/inference_protocol_bridge.md) for schemas, sampling, resume identities and completion NLL.

The evaluator exports classification, three distinct matching protocols (class-agnostic, geometry-first class-aware and same-class-priority), ungated Mean IoU and per-patient sufficient statistics. Threshold zero means strictly positive overlap; positive thresholds use `>=`. The full contract is in [metrics_pipeline.md](docs/metrics_pipeline.md).

## Five-seed summaries and paired statistics

Evaluate each actual checkpoint against the same frozen manifest with explicit model/seed identities. Then use a comparison specification listing all five exports per model:

```bash
python evaluation/joint_seed_statistics.py \
  --spec /path/to/comparison.json --output-dir ./outputs/statistics \
  --permutations 100000 --bootstrap 10000 --random-seed 2026
```

One patient-level permutation swaps that patient's complete five-seed vector. Metrics are recomputed within each seed before averaging their differences. Bootstrap multiplicities are synchronized across seeds and models. The default reports raw p-values; optional `--adjustment bh` corrects all comparator × metric rows in that invocation and records the family. See [joint_seed_statistics.md](docs/joint_seed_statistics.md) for the specification and uncertainty definitions. `aggregation/summarize_seed_results.py` also accepts canonical exports for descriptive mean/sample-SD summaries.

## Earlier interfaces

These utilities retain different input contracts and are not the current five-seed analysis:

| Utility | Scope |
| --- | --- |
| `evaluation/evaluate_sim999_predictions.py`, `evaluate_real961_predictions.py` | Older evaluation definitions; require `--allow-legacy-protocol`. |
| `evaluation/paired_patient_tests.py` | Single-seed paired classification; accepts `patient_id` or `logical_key`. |
| `statistics/finalize_revision_statistics.py`, `evaluation/postprocess_verl_checkpoint.py` | Older experiment layouts, helper scripts and metric schemas; require `--allow-legacy-protocol`. |
| `statistics/apply_r1_1_bh.py` | Earlier 60-comparison, seed-specific BH family. |
| Table and interval-figure exporters | Presentation of separately supplied results; see [table_presentation.md](docs/table_presentation.md). |

Older `class_aware` fields are not automatically equivalent to GFCA or SCF. Seed ensembles and averages of p-values are not the joint five-seed statistic. Presentation utilities do not recompute or authenticate results.

## Verification and availability

```bash
./scripts/run_required_tests.sh --cpu-only
./scripts/run_required_tests.sh
```

The full suite includes PyTorch/Transformers-dependent checks; neither command starts clinical inference or GPU training. Synthetic fixtures test implementations and interfaces. The [implementation map](docs/implementation_consistency_matrix.md) lists coverage, and [release status](docs/release_status.md) lists required experiment artifacts and remaining integration limits. Changes to rewards, inference NLL or statistical definitions require the corresponding retraining or reevaluation before updating reported results.

## Citation and license

See `CITATION.cff` for citation metadata and `LICENSE` for the MIT license. Dependencies retain their upstream licenses.
