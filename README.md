# RL-LHM

Code for **Risk-aware lesion-grounded reinforcement learning for structured vision–language diagnosis of liver lesions in multi-phase CT**.

The repository implements supervised fine-tuning, grouped reinforcement learning, frozen visual feature extraction, inference, and lesion-level and patient-level evaluation. Training uses a single-process PyTorch loop and Hugging Face causal image/text models. The visual feature extractor uses Qwen3-VL's final vision-block patch states.

## Installation

Use Python 3.10 or newer, then install from the repository root:

```sh
python -m pip install -r requirements.txt
```

## Workflow

1. Prepare separate training and evaluation manifests with complete three-phase image groups, structured lesion annotations, prompts, and patient labels. See the [manifest specification](docs/configuration.md#manifest).
2. Run SFT using the selected base model and processor.
3. Extract training visual vectors from that run's SFT checkpoint with one shared fixed projection.
4. Run RL using the SFT checkpoint as both the initial policy and the frozen reference. Component switches select SHAE, RARS, and DA-CVQ/VGS.
5. Generate one completion and its mean token NLL for each evaluation group, then compute patient and lesion metrics.

Supply the configuration paths and run these commands from the repository root:

```sh
python -m training.run sft --config "$SFT_CONFIG"
python -m inference.visual_features --config "$FEATURE_CONFIG"
python -m training.run rl --config "$RL_CONFIG"
python -m inference.run --config "$INFERENCE_CONFIG"
python -m evaluation.evaluate --manifest "$EVALUATION_MANIFEST" --predictions "$PREDICTIONS" --iou-thresholds "$IOU_THRESHOLD" --output-dir "$METRICS_DIR"
```

[Configuration fields and outputs](docs/configuration.md) describe each stage. [Algorithm contracts](docs/algorithms.md) define rewards, advantages, policy loss, curriculum updates, and voting.

## Source layout

| Directory | Purpose |
| --- | --- |
| `data/` | Manifest loading, patient population, and training-group class counts |
| `models/` | Three-image chat inputs, completion probabilities, and frozen Qwen3-VL features |
| `training/` | SFT/RL optimization, component switches, epoch checkpoints, and continuation |
| `inference/` | Feature-manifest creation and keyed completion generation |
| `rewards/` | Structured output parsing and RARS reward construction |
| `rl/` | SHAE, GRPO, token objective, visual projection, DA-CVQ, and VGS |
| `matching/` | IoU/GIoU geometry and three detection matching protocols |
| `aggregation/` | Group voting and patient diagnosis with NLL tie-breaking |
| `evaluation/` | Fixed-population patient classification and lesion metrics |

The [file guide](docs/code_map.md) explains every source file. Code is distributed under the MIT License.
