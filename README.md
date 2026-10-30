# RL-LHM

Training and evaluation code for **Risk-aware lesion-grounded reinforcement learning for structured vision–language diagnosis of liver lesions in multi-phase CT**.

SFT and RL run on **four NVIDIA A800 GPUs on one Linux host**, using Qwen3-VL-4B, BF16, DeepSpeed ZeRO-2, CPU AdamW state offload, and activation checkpointing. The pipeline includes frozen visual features, structured lesion rewards, SHAE/GRPO, DA-CVQ/VGS, patient aggregation, and detection evaluation over every input phase slice.

## Install

Use Python 3.10–3.12, a CUDA-enabled PyTorch installation, a CUDA toolkit compatible with PyTorch, and a C++ compiler for DeepSpeed CPUAdam. Allow at least 128 GB of host RAM for model loading and offloaded optimizer state.

```sh
python -m pip install -r requirements-training.txt
```

## Four-GPU workflow

Prepare patient-disjoint manifests at `datasets/train.json` and `datasets/test.json` using the [data specification](docs/configuration.md#manifest). Edit the manifest paths and diagnostic vocabulary/severity matrix in `configs/` for your dataset. Image paths are relative to their manifest; configuration paths are relative to the repository root.

```sh
bash scripts/train_4gpu.sh sft configs/sft_4xa800.json
python -m inference.visual_features --config configs/visual_features.json
bash scripts/train_4gpu.sh rl configs/rl_4xa800.json
python -m inference.run --config configs/inference.json
python -m evaluation.evaluate \
  --manifest datasets/test.json --predictions predictions/test.jsonl \
  --iou-thresholds 0 0.25 0.5 0.75 --output-dir metrics/test
```

The equivalent direct training command is:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m training.run rl --config configs/rl_4xa800.json
```

The supplied SFT configuration uses a global batch of 128 inputs. RL uses four input occurrences and two completions per occurrence, giving eight completions per update. Both use one sequence per GPU forward pass and normalize by the **global number of retained target tokens**, including partial final batches. Each image is limited to 262,144 pixels; the total sequence cap is 4,096 tokens and generation is capped at 512 tokens.

Checkpoints are saved after each epoch. To continue, set `resume_from` to an `epoch_XXXX` directory and retain the original `model` source. `loop.epochs` is the total desired epoch count. Checkpoints contain the Hugging Face model/processor, ZeRO optimizer shards, per-rank random states, UID baselines, and curriculum state.

## Algorithms and metrics

[Algorithm contracts](docs/algorithms.md) define the structured reward, RMS-normalized SHAE history branch, sampling policy, and global token objective. Reward utilities explicitly include missed lesions and unmatched predictions. Main detection metrics include background false positives; `detection_positive_only` reports the separately named positive-slice conditional metrics.

[Configuration fields](docs/configuration.md) describe the supplied settings. [Source layout](docs/code_map.md) maps the implementation.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

On the four-GPU training host, check BF16 ZeRO-2 accumulation and checkpoint continuation with a small Qwen3-VL model:

```sh
torchrun --standalone --nproc_per_node=4 -m tests.zero2_smoke --output /tmp/rl-lhm-zero-test
```

Code is distributed under the MIT License.

## Evaluation and controlled analyses

Manifests declare `output_classes` for parsing/voting and `metric_classes` for the fixed macro average. For WMU→ZJU, keep all five output labels and set the macro classes to HCC, ICC, HA_Hemangioma and nodule_CYST. FNH outputs remain valid predictions and count as errors. `configs/evaluation_loco.json` provides this evaluation setup.

Check target capacity using the actual chat template and tokenizer before training:

```sh
python -m data.check_lengths --config configs/sft_4xa800.json --output target_lengths.json
python -m evaluation.evaluate --config configs/evaluation_loco.json
```

Training also runs this preflight and rejects GT targets exceeding the completion budget. Predictions record EOS versus length-limit termination, token counts and phase sources. Evaluations report truncation by class and lesion count. Both maximum-total-IoU and threshold-aware detection results are emitted under separate names; `--matching-method` selects the primary metrics.

For center-stratified results, fixed-class patient bootstrap intervals and paired comparisons:

```sh
python -m evaluation.analyze --manifest datasets/test.json \
  --metrics metrics/test/metrics.json --compare metrics/baseline/metrics.json \
  --split-manifest train=datasets/train.json --split-manifest test=datasets/test.json \
  --bootstrap-samples 2000 --seed 42 --output metrics/analysis.json
```

Add `patient_centers` to manifests for center×class tables and per-center metrics. `configs/inference_repeat_a.json` repeats the arterial image in all three input slots while preserving patients, prompts and original evaluation targets. Generate and evaluate it alongside `configs/inference.json`, then use the paired comparison command to measure the effect of this input intervention. Use the same model and decoding settings in both configurations.
