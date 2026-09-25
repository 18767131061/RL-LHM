# RL-LHM

This repository contains the minimal public implementation accompanying the RL-LHM paper. It covers the final two-dimensional workflow: data-manifest checks, preprocessing, model loading, reinforcement-learning components, structured inference, lesion matching, detection metrics, patient-level classification, and Mean IoU reporting. No clinical data, patient records, checkpoints, or private infrastructure are included.

## Repository layout

- `data/`, `preprocess/`: manifest validation and deterministic patient-level splits.
- `models/`, `training/`, `rl/`, `rewards/`: model loading and the RL-LHM training components.
- `inference/`: structured Qwen/Hugging Face VLM inference and completion-likelihood helpers.
- `matching/`, `evaluation/`, `aggregation/`, `statistics/`: 2D matching, detection/classification metrics, patient aggregation, and statistical tests.
- `configs/`: path-free JSON templates.
- `tests/`: CPU-testable contracts for the metric and algorithmic core.
- `docs/METHOD_MAPPING.md`: paper-to-code map and validation boundaries.

The public scope is intentionally two-dimensional. Historical 3D post-processing, experiment schedulers, figure-generation sources, audit logs, and temporary analysis files are excluded.

## Installation

Python 3.10 or newer is recommended. The metric and preprocessing core requires:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Install `requirements-optional.txt` for model inference, plotting, and RL runtime integration. The training launcher additionally requires a compatible VeRL/vLLM environment; use the upstream version used for the paper.

## Data preparation

Prepare your own approved dataset and model files. Nothing in this repository grants access to the clinical data used in the paper. Use relative asset references in a manifest with the fields `sample_id`, `patient_id`, `center_id`, `split`, `label`, and `asset_ref`; validate it with `data/manifest.py`. Replace every `${...}` value in `configs/` with a path on your machine.

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

Use `--smoke` or `--formal-single-gpu` only when the hardware and experiment protocol allow it. The launcher does not download data or weights.

## Inference

For local Qwen3-VL inference, provide a model directory, an input CSV, an image root, and an output CSV:

```bash
python inference/infer_qwen_real961.py \
  --model-dir /path/to/checkpoint \
  --json-path /path/to/input.jsonl \
  --data-root /path/to/images \
  --output ./outputs/predictions.csv
```

The input JSONL and output CSV schemas are defined by the inference source and must be supplied or adapted by the user. This command requires the optional GPU/model dependencies.

## Evaluation

The main evaluation entry point consumes model predictions and metadata and writes JSON/CSV summaries, including 2D detection, patient-level classification, matching-protocol comparisons, and Mean IoU:

```bash
python evaluation/evaluate_sim999_predictions.py \
  --predictions /path/to/predictions.csv \
  --metadata /path/to/metadata.csv \
  --output-dir ./outputs/evaluation \
  --iou-threshold 0.5
```

Use `evaluation/classification.py`, `evaluation/detection.py`, `aggregation/patient.py`, and `statistics/` directly when integrating the metrics into another pipeline. `scripts/plot_rars.py` renders a supplied RARS grid; it does not contain paper data.

## Reproducibility and limitations

Run the CPU-testable suite with:

```bash
./scripts/run_required_tests.sh
```

The tests cover deterministic matching, IoU, classification, patient aggregation, RARS weighting, preprocessing, and RL-core contracts. Full training and inference require private/approved medical data, model weights, CUDA, and a compatible VeRL runtime, so those paths are statically checked here rather than claimed as executable in a clean CPU environment.

## Data availability

The clinical datasets and trained weights are not redistributed. Researchers must obtain approved access independently and adapt the path placeholders in `configs/`.

## Citation and license

Please cite the RL-LHM paper and see `CITATION.cff` for machine-readable citation metadata. The code written for this repository is released under the MIT License in `LICENSE`. Dependencies are installed separately and remain under their respective upstream licenses; they are not redistributed here.
