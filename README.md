# RL-LHM

Training and evaluation code for **Risk-aware lesion-grounded reinforcement learning for structured vision–language diagnosis of liver lesions in multi-phase CT**.

Training is configured for **four NVIDIA A800 GPUs on one Linux host** with Qwen3-VL-4B and BF16. SFT uses Hugging Face Transformers and DeepSpeed ZeRO-2; RL offers DeepSpeed and VeRL FSDP2/vLLM backends. Both RL interfaces use the same structured rewards, SHAE/GRPO and DA-CVQ/VGS modules. The pipeline also includes frozen visual features, patient aggregation, and lesion-positive phase-slice detection with separate all-input diagnostics.

## Install

For DeepSpeed, use Python 3.10–3.12, a CUDA-enabled PyTorch installation, a compatible CUDA toolkit, and a C++ compiler for CPUAdam. Allow at least 128 GB of host RAM for model loading and offloaded optimizer state.

```sh
python -m pip install -r requirements-training.txt
```

For VeRL, use a Linux host with a CUDA 13-compatible driver, Git and `uv`. From this repository, create a separate runtime in a new absolute directory and activate it:

```sh
bash scripts/setup_verl.sh /absolute/path/to/verl-runtime
source /absolute/path/to/verl-runtime/.venv/bin/activate
```

The script checks out the pinned VeRL source, synchronizes its official `uv.lock` with the FSDP/vLLM extras and Python 3.12, installs this project's supplemental dependencies, and checks the recipe configuration. Run the training and inference commands below from this repository.

For optional DICOM intensity preparation, install `requirements-preprocessing.txt` and use `data.dicom.read_dicom_series`. It preserves native geometry and supports window clipping with optional uint8 normalization; see the [DICOM interface](docs/configuration.md#optional-dicom-intensity-preparation). Phase correspondence, slice selection and lesion annotations remain manifest inputs.

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

The equivalent direct DeepSpeed training command is:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m training.run rl --config configs/rl_4xa800.json
```

For VeRL RL, use the local SFT checkpoint and extracted-feature manifest in `configs/rl_verl_4xa800.json`, then launch one driver process in the VeRL environment:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m training.run rl \
  --backend verl --config configs/rl_verl_4xa800.json
```

The dedicated entry point is `python -m training.verl_run --config configs/rl_verl_4xa800.json`. Add `--check-config` to that command to validate and print the pinned VeRL configuration. The shared entry point selects `--backend`, then the JSON `backend` field, then `deepspeed` by default. SFT uses the DeepSpeed backend.

The supplied SFT configuration uses two inputs per device and a global batch of 128 inputs on four GPUs, giving 16 accumulation steps. RL uses four input occurrences and two completions per occurrence, giving eight completions per update. Both normalize by the **global number of retained target tokens**, including partial final batches. Each image is limited to 262,144 pixels; the total sequence cap is 2,048 tokens and generation is capped at 256 tokens. RL uses temperature 1.0 and top-p 0.95.

Checkpoints are saved after each epoch. To continue with the same backend, set `resume_from` to its `epoch_XXXX` directory and retain the original `model` source. `loop.epochs` is the total desired epoch count. DeepSpeed stores the Hugging Face model/processor at the epoch root and optimizer shards under `zero/`. VeRL stores actor/optimizer shards under `actor/`, the Hugging Face export under `actor/huggingface/`, and driver state in `rl_lhm_state.pt`. For inference after VeRL training, set both model and processor paths to that Hugging Face export. See [checkpoint configuration](docs/configuration.md#sft-and-rl-training) for state and continuation fields.

The VeRL inference recipe selects `checkpoints/rl_verl/epoch_0006/actor/huggingface` and writes a separate prediction file:

```sh
python -m inference.run --config configs/inference_verl.json
python -m evaluation.evaluate \
  --manifest datasets/test.json --predictions predictions/test_verl.jsonl \
  --iou-thresholds 0 0.25 0.5 0.75 --output-dir metrics/test_verl
```

## Algorithms and metrics

[Algorithm contracts](docs/algorithms.md) define the separately matched localization/diagnosis reward, SHAE history branch with Gaussian jitter and population-standard-deviation normalization, mass-based codebook maintenance, sampling policy, and global token objective. The reward is clipped to `[-6,3]`; invalid outputs receive zero. The default `detection_scope=positive_phase_slices` evaluates the paper's lesion-positive phase slices. `detection_all_input` and background false-positive summaries are reported separately; select `--detection-scope all_input_phase_slices` to make that scope primary.

[Configuration fields](docs/configuration.md) describe the supplied settings. [Source layout](docs/code_map.md) maps the implementation.

## Tests

Run the core and DeepSpeed tests in the DeepSpeed environment:

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
