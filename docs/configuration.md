# Configuration and data

Stage configuration files are JSON objects. Supply the fields below explicitly. Configuration paths are interpreted from the current working directory; image paths are interpreted from the manifest's directory. Use patient-disjoint training, validation, and test manifests for a split-based experiment.

## Manifest

The input JSON declares the output vocabulary, fixed evaluation classes, patient population and input records:

| Field | Meaning |
| --- | --- |
| `output_classes` | Ordered unique canonical output labels; this order resolves voting ties |
| `metric_classes` | Fixed nonempty subset for macro metrics; defaults to all output classes |
| `patient_centers` | Optional mapping of every patient ID to its center |
| `patients` | Object mapping every patient ID in the population to its true diagnostic label |
| `records` | List of complete three-phase input groups |

Every record contains `uid` (unique group ID), `patient_id`, `images` (an object with `a`, `p`, `v` paths), `prompt` (the complete instruction), and `ground_truth` (a serialized structured JSON completion). Images enter the model in arterial, portal-venous, delayed order, followed by the prompt. Image files must be readable by Pillow during model execution. Image preparation, phase correspondence, and normalized lesion annotations are supplied in this interface.

An optional `visual_vector` is a finite, nonzero numeric list with the same dimension across records; a missing value or `null` denotes an unavailable vector. Feature extraction writes this field. `patients` may include patients with zero complete groups; evaluation retains them in its population.

A structured completion contains a `detections` list. Each detection has `phase` (`a`, `p`, or `v`), `label` (a canonical class), and `bbox_2d` (`[x1,y1,x2,y2]`, numeric coordinates on `[0,1000]`, positive width and height). Background uses `label="no-lesion"` and `bbox_2d=[-1,-1,-1,-1]`, once per background phase. A phase contains either background or lesions. Multiple lesions in one phase are retained. An omitted output phase is allowed.

Training class counts are computed once from the complete fitting manifest. Each positive group contributes one count to its most frequent ground-truth lesion label across phases; ties use the lexicographically greatest internal label, as specified in Appendix C.2. This metadata rule differs from inference voting, whose ties use output class order. Background-only groups contribute no count. Counts remain fixed during sampling and across seeds on the same manifest.

## Optional DICOM intensity preparation

Install `requirements-preprocessing.txt` to use `data.dicom.read_dicom_series(directory, series_uid=None, window_center=None, window_width=None, normalize=False)`. This SimpleITK utility reads one scalar 3D series in GDCM order. Supply `series_uid` when the directory contains multiple series. Both explicit window parameters must be supplied together; otherwise the first DICOM's WindowCenter/WindowWidth are used, taking the first component of multivalued tags. Missing or unusable window metadata falls back to the decoded volume's intensity range.

Clipping uses `center ± width/2`. The reader applies no additional slope/intercept after SimpleITK decoding. The default returns clipped float64 values; `normalize=True` maps the window linearly to `[0,255]` and truncates to uint8, with a constant-range volume mapped to zero. `DicomVolume.values` uses z/y/x indexing; spacing and origin use x/y/z, and direction is a flattened row-major 3×3 matrix. Native spacing, origin and direction are preserved without reorientation or resampling. The returned object records the selected series, source files and window bounds/source. Cross-phase correspondence, slice selection, mask-to-box conversion and manifest construction remain separate data-preparation inputs.

## Shared model and decoding objects

The `model` object contains:

| Field | Meaning |
| --- | --- |
| `model_name_or_path` | Hugging Face model identifier or checkpoint directory |
| `processor_name_or_path` | Corresponding processor identifier or directory |
| `revision` | Revision string, or `null` for the selected source's default revision |
| `device` | Explicit PyTorch device, such as `cpu` or `cuda:0` |
| `dtype` | `float32`, `float16`, or `bfloat16` |
| `attn_implementation` | Attention implementation supported by the model and device |
| `processor_kwargs` | Explicit processor-loading arguments, including image settings when required |

The DeepSpeed backend and inference tools load `AutoModelForImageTextToText` and `AutoProcessor`. VeRL loads the local post-SFT model and processor through its FSDP2/vLLM interfaces. The model requires a causal decoder and a chat template whose supervised assistant sequence extends the generation prompt. Hugging Face checkpoint exports contain the model and processor together.

The `generation` object contains all `DecodingConfig` fields:

| Field | Meaning |
| --- | --- |
| `max_new_tokens` | Positive completion length cap |
| `do_sample` | Boolean; `true` for RL rollouts |
| `temperature`, `top_p`, `top_k` | Explicit sampling settings |
| `repetition_penalty` | Positive repetition penalty |
| `num_beams` | `1`, one completion per supplied input row |
| `eos_token_ids` | Nonempty list of stopping token IDs from the selected model/tokenizer |
| `pad_token_id` | Tokenizer's padding token ID |
| `suppress_token_ids` | Input-only image/video/vision marker IDs, plus a distinct non-EOS padding ID when applicable, excluded from text completions |
| `use_cache` | Boolean generation cache setting |

The first generated EOS contributes to the completion score; subsequent padding is excluded. Completion NLL and old/current/reference policy log-probabilities use raw model logits and exclude prompt tokens. Token suppression applies to generation only, not to these scores. The paper rollout configuration uses temperature=1, top_p=0.95, top_k=0, repetition_penalty=1 and do_sample=true. Sampling settings and the model probabilities used by the objective are distinct; see the algorithm contract.

## SFT and RL training

The shared CLI is `python -m training.run {sft,rl} --config ... --backend {deepspeed,verl}`. An explicit `--backend` takes precedence over JSON `backend`; without either, DeepSpeed is selected. SFT uses DeepSpeed. The VeRL interface supports RL.

For DeepSpeed, install `requirements-training.txt`, start from `configs/sft_4xa800.json` or `configs/rl_4xa800.json`, and run `bash scripts/train_4gpu.sh sft [config.json]` or `bash scripts/train_4gpu.sh rl [config.json]`. These scripts launch four workers with torchrun.

For VeRL, use a Linux host with a CUDA 13-compatible driver, Git and `uv`. Run `bash scripts/setup_verl.sh /absolute/path/to/verl-runtime` with a new runtime directory, then `source /absolute/path/to/verl-runtime/.venv/bin/activate`. The script checks out official VeRL commit `a911bca25d6adb26f5bef2dc57e7ba62e99b7ade`, runs `uv sync --frozen --all-packages --extra fsdp --extra vllm --python 3.12` against its official lock and package sources, installs the project supplements in `requirements-verl.txt`, and checks the recipe. Run subsequent commands from this repository.

Launch one driver process with `python -m training.run rl --backend verl --config configs/rl_verl_4xa800.json`. The dedicated equivalent is `python -m training.verl_run --config configs/rl_verl_4xa800.json`. The adapter verifies the pinned upstream source files. `python -m training.verl_run --config configs/rl_verl_4xa800.json --check-config` validates and prints the composed configuration without starting training.

Both stages require:

| Field | Meaning |
| --- | --- |
| `manifest` | Fitting-manifest path |
| `output_dir` | Epoch checkpoint output directory |
| `resume_from` | `null` to start, or an epoch checkpoint directory to continue |
| `backend` | Optional `deepspeed` or `verl`; CLI override takes precedence |
| `seed` | Base training seed; DeepSpeed seeds each worker with seed + rank, while VeRL seeds its driver and configures its worker engines |
| `distributed.world_size` | 4 workers on one host |
| `order_seed` | Dataset-order random stream seed |
| `deterministic_algorithms` | Boolean passed to PyTorch |
| `model`, `generation` | Shared objects above |
| `optimizer` | AdamW: `learning_rate`, `betas` (two numbers), `epsilon`, `weight_decay`, `amsgrad` |
| `loop` | `epochs` (total target count), `batch_size` (global input occurrences per update), `microbatch_size`, `max_grad_norm`, `max_sequence_length`, optional `max_prompt_length` |

SFT optimizes assistant completion tokens. `microbatch_size` counts input groups in SFT and completions in RL; RL requires this value to be one. Each update retains one normalization over its full retained-token population. A partial final batch uses its own token count. The supplied SFT settings use two inputs per device and a global batch of 128 on four devices, giving 16 accumulation steps for a full update. RL uses one completion per device forward. The sequence cap is 2048, the prompt cap is 1024, and the completion cap is 256. Oversized prompts/targets raise an error instead of silently dropping image tokens or annotations.

DeepSpeed uses Hugging Face Transformers, BF16 model parameters, and ZeRO-2 with FP32 CPU AdamW master parameters and optimizer state. `LOCAL_RANK` selects the CUDA device, and rollout tensors are cached on CPU. VeRL uses FSDP2 actor/reference workers and vLLM generation, with the offload settings below. Both RL backends enable activation checkpointing, disable scoring dropout and use a frozen reference policy. The supplied SFT batch is 128 inputs; RL uses four inputs × two completions. Remote training model revisions are immutable; local source weights are fingerprinted for continuation.

VeRL requires a local post-SFT model and processor, `model.revision=null`, `model.attn_implementation="flash_attention_2"`, four visible CUDA GPUs, and RL batch/group/microbatch sizes 4/2/1. Its `verl` object contains:

| Field | Meaning |
| --- | --- |
| `upstream_commit` | Exact commit selected by `scripts/setup_verl.sh` and checked by the adapter |
| `rollout_seed` | Independent driver-owned stream generating an explicit seed for each vLLM request |
| `tensor_parallel_size` | vLLM tensor parallelism: 1, 2 or 4; supplied value 1 |
| `gpu_memory_utilization` | vLLM memory fraction, strictly between zero and one; supplied value 0.35 |
| `agent_workers` | Number of rollout agent workers, 1–8; supplied value 4 |
| `actor_param_offload`, `actor_optimizer_offload`, `reference_param_offload` | Offload switches; all enabled in the supplied recipe |

RL also requires:

| Field | Meaning |
| --- | --- |
| `loop.group_size` | At least two sampled completions per input occurrence |
| `loop.policy_updates` | Positive number of updates on each fixed rollout batch |
| `loop.advantage_epsilon` | Positive advantage normalization constant |
| `sampling.replacement` | Boolean controlling later curriculum draws; the paper configuration uses `true`, with epoch-zero uninitialized sampling as a permutation. Ignored when `components.curriculum=false`, which uses a random permutation every epoch |
| `components` | Booleans `shae`, `rars`, `curriculum` |
| `policy` | `clip_ratio`, `dual_clip`, `kl_coefficient`, `log_ratio_bound`, `kl_bound` |
| `reward` | Reward object described below |
| `shae` | When enabled: `baseline_beta`, `gate_mid`, `gate_temperature`, `noise_std` (paper value 0.001), `seed` |
| `curriculum` | When enabled: `state`, `state_seed`, `sampling_seed`, `missing_feature_multiplier`, `weight_floor` |

For RL, `model` points to the run's post-SFT checkpoint. A new run loads separate current and frozen reference policies from this source. Continuation restores the current policy and optimizer from the selected backend's `resume_from` epoch directory and retains the original `model` source for the reference. Keep the original source available.

The `reward` object contains `severity_matrix` (true-label rows, predicted-label columns, complete symmetric off-diagonal entries), `rars_parameters` (`scarcity_alpha`, `confidence_eta`, `severity_kappa`, `count_epsilon`), `giou_for_classification`, `detection_weight`, `classification_weight`, `missing_match_penalty`, `clip_min`, and `clip_max`. The supplied settings use GIoU 0.30, weights 1, missing-match magnitude 0.1 and clipping `[-6,3]`. Invalid completions receive zero. Localization and classification use independent IoU and GIoU assignments; diagnostic pairs are thresholded after assignment.

`curriculum.state` contains `max_codes`, `input_dim`, `feature_decay`, `reward_decay`, `difficulty_beta`, `dead_code_threshold`, `dead_code_replace_probability`, `revival_noise_std`, `diversity_threshold`, `reward_min`, `reward_max`, and `mass_epsilon`. `input_dim` equals the vector length; reward bounds equal reward clipping bounds. `difficulty_beta` is inverse sampling temperature. `dead_code_threshold` is the EMA feature-mass cutoff (0.25), not an inactivity count. During initialized updates, the first code below this mass threshold is revived when the probability draw permits. Activity counters are diagnostics and do not determine eligibility.

Each occurrence has its own RID and G completions. DeepSpeed keeps the occurrence on one worker and applies global occurrence-ordered UID baseline and DA-CVQ updates on every worker. VeRL keeps one copy of that state on the Ray driver, independently of rollout worker assignment. SHAE uses population Std for the centered group branch and for the uncentered history residual plus Gaussian jitter. SHAE-off uses sample Std with divisor G−1 and no jitter, baseline or gate. RARS-off uses unit diagnostic error costs. With curriculum off, both backends use the independent `order_seed` generator to draw a random permutation of all inputs each epoch, regardless of `sampling.replacement`.

DeepSpeed epoch directories contain a Hugging Face model/processor, collective ZeRO shards under `zero/`, per-rank state files, `checkpoint.json` (format version 3), `training_config.json`, and `training_history.json`. Continuation restores optimizer, per-rank RNG, shared CPU generators, UID baseline, codebook and sampler epoch.

VeRL epoch directories contain official actor/FSDP optimizer and RNG shards under `actor/`, a Hugging Face export under `actor/huggingface/`, `rl_lhm_state.pt`, `checkpoint.json` (format version 1, `backend="verl"`), `training_config.json`, `training_history.json`, and the composed `verl_config.json`. The driver state includes completed epochs, UID baselines, codebook, sampler and random streams, including the per-request rollout-seed stream. Set `resume_from` to the epoch directory; use `actor/huggingface/` as the model and processor source for inference.

`configs/inference_verl.json` selects `checkpoints/rl_verl/epoch_0006/actor/huggingface` for both model and processor and writes `predictions/test_verl.jsonl`. Its decoding and evaluation input manifest match `configs/inference.json`.

Both backends publish each completed epoch directory by atomic rename. Keep its files on shared local storage and resume with the same code revision, backend, sampling policy and four-GPU configuration. The source model, manifest and mathematical settings identify the run; output directory, continuation path and total target epoch count may change. An interrupted `.incomplete` directory is not a completed checkpoint.

## Frozen visual vectors

Run `python -m inference.visual_features --config ...` with `manifest`, `output`, `batch_size`, `model`, `generation`, and `projection`. Select the same seed's SFT checkpoint.

`projection` is either `{"path": "projection.npy"}` to load a matrix, or an object with explicit `seed`, `dimension`, and `output` to construct and save a Gaussian-QR orthonormal projection. Its row dimension equals the vision hidden dimension. The extractor reads final Qwen3-VL visual-block patches before the final merger and normalization, pools each phase, applies the shared projection, concatenates a/p/v, and normalizes the vector. It writes a manifest containing vectors and resolved image paths. Reuse the same projection for inputs sharing this visual space.

## Inference

Run `python -m inference.run --config ...` with `manifest`, `output`, `batch_size`, `seed`, `model`, and `generation`. The supplied configuration sets `max_sequence_length=2048`, `max_prompt_length=1024`, and a 256-token greedy completion cap. Optional `phase_sources` maps each a/p/v input slot to an original phase; the default is the identity map. Repeating one phase changes image inputs only and retains the original targets for intervention analysis. Select the trained checkpoint and its saved processor. Output JSONL contains `uid`, `patient_id`, `completion`, `nll`, `finish_reason` (`eos` or `length`), `completion_tokens`, `max_new_tokens`, `target_tokens`, and `phase_sources`. An EOS at the last allowed position is an EOS termination. A sidecar `.metadata.json` records configuration, code revision, checkpoint identity, manifest/prediction hashes and target lengths.

## Evaluation

Run `python -m evaluation.evaluate` with `--manifest`, `--predictions`, `--iou-thresholds` (one or more values in `[0,1]`), and `--output-dir`. Prediction UIDs must cover groups exactly once. Threshold zero means `IoU > 0`; positive thresholds mean `IoU >= threshold`.

Outputs are `metrics.json`, `patient_predictions.jsonl`, and `group_predictions.jsonl`. Patient accuracy and macro precision/recall/F1 use the fixed patient and class populations. Groups vote by lesion-label majority; patient ties use mean finite NLL, then class order. Patients without votes remain in the metrics with undefined predictions.

By default, `detection_scope="positive_phase_slices"` selects the paper's lesion-positive phase-slice metrics. Set `--detection-scope all_input_phase_slices` for all-input primary metrics. `detection` and `detection_by_matching` follow the selected scope. `detection_positive_only` and `detection_positive_only_by_matching` always retain the positive-only results; `detection_all_input` and `detection_all_input_by_matching` always retain the all-input results. `background_false_positives` reports background FP count and burden. CA maximizes total IoU. GFCA filters that same assignment by class agreement. Same-class-first maximizes the number of same-class pairs, then total IoU. Precision/recall/F1 use TP/FP/FN at every threshold. Mean IoU uses unthresholded CA divided by all GT lesions. Acc@IoU>0 uses accepted positive-overlap matches divided by all GT lesions.


## Fixed class spaces, termination and matching

Manifests use `output_classes`; `classes` is accepted as an output-vocabulary alias. Supply only one of these fields. `metric_classes` is fixed before evaluating or resampling, even if a bootstrap sample omits a class. Parsing and group/patient voting always use all output classes. Classification accuracy retains all patients, including legal out-of-target predictions and patients without votes.

The evaluator accepts direct CLI arguments or `--config configs/evaluation_loco.json`; explicit CLI fields override configuration fields. `--metric-classes` can override the manifest's fixed macro list. `--require-termination-metadata` rejects prediction files lacking EOS/length metadata. Without this flag, files lacking termination metadata are readable with termination counted as `unknown`, never included in the known-termination truncation denominator. Termination is reported independently of schema validity. A length-limited output follows the same schema and diagnostic-eligibility checks as any other output; an invalid completion provides no lesion records or diagnostic vote and receives zero RL reward. Every patient and GT lesion remains in the evaluation denominator.

`--matching-method max_total_iou` selects maximum-total-IoU class-agnostic, geometry-first class-aware, and same-class-priority protocols. `--matching-method threshold_aware` selects threshold-aware class-agnostic and class-aware protocols. Both families are always returned for the selected scope in `detection_by_matching`, and for the explicit scopes in `detection_positive_only_by_matching` and `detection_all_input_by_matching`. `matching_sensitivity` lists phase/threshold cases whose TP counts differ between corresponding protocols. Mean IoU always uses unthresholded maximum-total-IoU class-agnostic assignment, independently of this selection.

`python -m data.check_lengths --config ... --output ...` loads only the processor, computes the chat-template target suffix including EOS, and exits with code 1 if any target exceeds `max_new_tokens`. Training performs the same check across ranks before any optimizer update and saves `target_lengths.json`; SFT never truncates annotations. Generation/evaluation statistics include per-class and per-lesion-count breakdowns.

`evaluation.analyze` computes fixed-class patient bootstrap confidence intervals, paired classification differences, per-center classification, center×class counts, descriptive constant-prediction references and detection/localization differences. Paired comparisons require the same patients, truths, class spaces, primary matching method, detection scope and IoU thresholds. The bootstrap unit is one patient, not a lesion, image or training seed. Optional named split manifests produce center×class tables and patient-overlap counts.
