# RL-LHM

This repository contains the minimal public implementation accompanying the RL-LHM paper. It covers the final two-dimensional workflow: data-manifest checks, preprocessing, model loading, reinforcement-learning components, structured inference, lesion matching, detection metrics, patient-level classification, and Mean IoU reporting. No clinical data, patient records, checkpoints, or private infrastructure are included.

## Repository layout

- `data/`, `preprocess/`: manifest validation and deterministic patient-level splits.
- `models/`, `training/`, `rl/`, `rewards/`: model loading and the RL-LHM training components.
- `inference/`: structured Qwen/Hugging Face VLM inference and completion-likelihood helpers.
- `matching/`, `evaluation/`, `aggregation/`, `statistics/`: 2D matching, detection/classification metrics, patient aggregation, and statistical tests.
- `configs/`: path-free JSON templates.
- `tests/`: CPU-testable contracts for the metric and algorithmic core.
- `docs/implementation_consistency_matrix.md`: paper-to-code map and validation boundaries.

The public scope is intentionally two-dimensional. Historical 3D post-processing, experiment schedulers, figure-generation sources, audit logs, and temporary analysis files are excluded.

## Installation

Python 3.10 or newer is recommended. The metric and preprocessing core requires:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Install `requirements-optional.txt` for model inference, plotting, and RL runtime integration. The training launcher additionally requires a compatible VeRL/vLLM environment. The dependency lists are not a validated environment lock: reproducing a training run also requires the exact external runtime version, any agent-loop or policy-loss patches, and the feature-extraction configuration. See `docs/reproducibility_guide.md`.

## Data preparation

Prepare your own approved dataset and model files. Nothing in this repository grants access to the clinical data used in the paper. Use relative asset references in a dataset manifest with the fields `sample_id`, `patient_id`, `center_id`, `split`, `label`, and `asset_ref`; validate it with `data/manifest.py`. Replace every `${...}` value in `configs/` with a path on your machine.

The evaluation manifest has a separate schema that lists the complete patient denominator, input groups, and ground-truth lesions for each individual phase. Its explicit phase-level annotations must not be replaced by copying a single slice's boxes or a patient-level label across all phases. See `docs/metrics_pipeline.md`. Creating that manifest from CT volumes requires a specified selection and preprocessing procedure; this evaluator does not select slices from volumes.

## Training

The launcher exposes the paper variants and accepts explicit data, model, output, and script paths. A configuration-only dry run is useful before starting a GPU job:

```bash
python training/run_verl_lhm_variant.py \
  --train-file /path/to/train.jsonl \
  --model-path /path/to/base-model \
  --output-dir ./outputs/example \
  --variant full_rl_lhm \
  --seed 2026 \
  --scripts-dir . \
  --n-gpus 2 \
  --config-only
```

Use `--smoke` or `--formal-single-gpu` only when the hardware and experiment protocol allow it. The launcher does not download data or weights. Configuration-only execution checks the generated configuration, not the external policy loss or a GPU training run. The eight component combinations, required stable sample identifiers, and explicit VeRL integration patches are documented in `docs/training_pipeline.md`.

## Inference

For local Qwen3-VL inference, provide a model directory, an input JSONL, an image root, and an output CSV:

```bash
python inference/infer_qwen_real961.py \
  --model-dir /path/to/checkpoint \
  --json-path /path/to/input.jsonl \
  --data-root /path/to/images \
  --output ./outputs/predictions.csv
```

The input JSONL and output CSV schemas are defined by the inference source and must be supplied or adapted by the user. This command requires the optional GPU/model dependencies.

## Evaluation

The main protocol evaluation entry point consumes an explicit evaluation manifest and keyed completion records. It uses the shared strict parser, all eligible diagnostic votes, and the three canonical matching protocols. Detection is evaluated only on phase slices with at least one ground-truth lesion. Patient classification retains every patient listed in the manifest, including patients with no eligible diagnostic vote.

```bash
python evaluation/evaluate_paper_protocol.py \
  --manifest /path/to/evaluation_manifest.json \
  --predictions /path/to/predictions.jsonl \
  --output-dir ./outputs/evaluation \
  --iou-thresholds 0 0.5
```

Threshold `0` denotes strictly positive overlap, and positive thresholds use `>=`. Missing, duplicate, or extra prediction keys are rejected instead of silently dropping inputs. Each record's `group_id` must match the manifest. Adapt the inference helper's CSV output to the documented completion-record schema when using that helper; phase-specific ground truth must come from the evaluation annotations.

See `docs/metrics_pipeline.md` for the complete input/output contract. The CLI tests use synthetic inputs without images, model weights, or private data.

`evaluation/evaluate_sim999_predictions.py` is retained as a **legacy utility**, not the protocol evaluation entry point. Its metadata layout, GT-positive vote selection, background handling, and old `class_aware` mode differ from the protocol above. Do not use its output interchangeably with the three canonical protocols.

Use `evaluation/classification.py`, `evaluation/detection.py`, `aggregation/patient.py`, and `statistics/` directly when integrating another pipeline. `scripts/plot_rars.py` renders a supplied RARS grid; it does not contain paper data.

## Reproducibility and limitations

Run the full test suite after installing its optional dependencies:

```bash
./scripts/run_required_tests.sh
```

The full test script includes tests requiring PyTorch and OmegaConf from `requirements-optional.txt`. The CPU-only contract tests can be run separately with `./scripts/run_required_tests.sh --cpu-only`. Tests cover deterministic matching, IoU, classification, patient aggregation, RARS weighting, strict parsing at the actual reward entry point, and the evaluation and configuration CLIs. Full training and inference require approved medical data, model weights, CUDA, and a compatible VeRL runtime. Passing synthetic and CPU contract tests does not establish reproduction of reported numerical results.

## Data availability

The clinical datasets and trained weights are not redistributed. Researchers must obtain approved access independently and adapt the path placeholders in `configs/`.

## Citation and license

Please cite the RL-LHM paper and see `CITATION.cff` for machine-readable citation metadata. The code written for this repository is released under the MIT License in `LICENSE`. Dependencies are installed separately and remain under their respective upstream licenses; they are not redistributed here.
