# Configuration and data

Stage configuration files are JSON objects. Supply the fields below explicitly. Configuration paths are interpreted from the current working directory; image paths are interpreted from the manifest's directory. Use patient-disjoint training, validation, and test manifests for a split-based experiment.

## Manifest

The input JSON has exactly three top-level fields:

| Field | Meaning |
| --- | --- |
| `classes` | Ordered unique canonical labels; this order resolves diagnostic voting ties |
| `patients` | Object mapping every patient ID in the population to its true diagnostic label |
| `records` | List of complete three-phase input groups |

Every record contains `uid` (unique group ID), `patient_id`, `images` (an object with `a`, `p`, `v` paths), `prompt` (the complete instruction), and `ground_truth` (a serialized structured JSON completion). Images enter the model in arterial, portal-venous, delayed order, followed by the prompt. Image files must be readable by Pillow during model execution. Image preparation, phase correspondence, and normalized lesion annotations are supplied in this interface.

An optional `visual_vector` is a finite, nonzero numeric list with the same dimension across records; a missing value or `null` denotes an unavailable vector. Feature extraction writes this field. `patients` may include patients with zero complete groups; evaluation retains them in its population.

A structured completion contains a `detections` list. Each detection has `phase` (`a`, `p`, or `v`), `label` (a canonical class), and `bbox_2d` (`[x1,y1,x2,y2]`, numeric coordinates on `[0,1000]`, positive width and height). Background uses `label="no-lesion"` and `bbox_2d=[-1,-1,-1,-1]`, once per background phase. A phase contains either background or lesions. Multiple lesions in one phase are retained. An omitted output phase is allowed.

Training class counts are computed once from the complete fitting manifest. Each positive group contributes one count to its most frequent ground-truth lesion label across phases; ties use the lexicographically greatest label. Background-only groups contribute no count. Counts remain fixed during sampling and across seeds on the same manifest.

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

Loading uses `AutoModelForImageTextToText` and `AutoProcessor`. The backend requires a causal decoder and a chat template whose supervised assistant sequence extends the generation prompt. Checkpoints save the model and processor together.

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
| `suppress_token_ids` | Input-only image/video/vision marker IDs and a distinct non-EOS padding ID excluded from text completions |
| `use_cache` | Boolean generation cache setting |

The first generated EOS contributes to the completion score; subsequent padding is excluded. NLL uses the model's original logits and excludes prompt tokens. Old, current, and reference policy scores use the same saved token sequences.

## SFT and RL training

Run `python -m training.run sft --config ...` or `python -m training.run rl --config ...`.

Both stages require:

| Field | Meaning |
| --- | --- |
| `manifest` | Fitting-manifest path |
| `output_dir` | Epoch checkpoint output directory |
| `resume_from` | `null` to start, or an epoch checkpoint directory to continue |
| `seed` | Python, NumPy, and PyTorch seed |
| `order_seed` | Dataset-order random stream seed |
| `deterministic_algorithms` | Boolean passed to PyTorch |
| `model`, `generation` | Shared objects above |
| `optimizer` | AdamW: `learning_rate`, `betas` (two numbers), `epsilon`, `weight_decay`, `amsgrad` |
| `loop` | `epochs` (total target count), `batch_size` (input groups per update), `microbatch_size`, `max_grad_norm` |

SFT optimizes assistant completion tokens. `microbatch_size` counts input groups in SFT and completions in RL; each update retains one normalization over its full retained-token population. A partial final batch uses its own token count.

Training uses `model.dtype="float32"` for model parameters and AdamW state, on a `cpu` or `cuda` device. Inference and frozen feature extraction accept all three listed dtypes. Each training process places its model and optimizer on the configured device.

RL also requires:

| Field | Meaning |
| --- | --- |
| `loop.group_size` | At least two sampled completions per input occurrence |
| `loop.policy_updates` | Positive number of updates on each fixed rollout batch |
| `loop.advantage_epsilon` | Positive advantage normalization constant |
| `components` | Booleans `shae`, `rars`, `curriculum` |
| `policy` | `clip_ratio`, `dual_clip`, `kl_coefficient`, `log_ratio_bound`, `kl_bound` |
| `reward` | Reward object described below |
| `shae` | When enabled: `baseline_beta`, `gate_mid`, `gate_temperature`, `noise_std`, `seed` |
| `curriculum` | When enabled: `state`, `state_seed`, `sampling_seed`, `missing_feature_multiplier`, `weight_floor` |

For RL, `model` points to the run's post-SFT checkpoint. A new run loads separate current and frozen reference policies from this source. Continuation loads the current policy from `resume_from` and the reference from the original `model` source. Keep the original source available.

The `reward` object contains `severity_matrix` (true-label rows, predicted-label columns, complete symmetric off-diagonal entries), `rars_parameters` (`scarcity_alpha`, `confidence_eta`, `severity_kappa`, `count_epsilon`), `giou_for_classification`, `detection_weight`, `classification_weight`, `missing_match_penalty`, `clip_min`, and `clip_max`.

`curriculum.state` contains `max_codes`, `input_dim`, `feature_decay`, `reward_decay`, `difficulty_beta`, `dead_code_threshold`, `dead_code_replace_probability`, `revival_noise_std`, `diversity_threshold`, `reward_min`, `reward_max`, and `mass_epsilon`. `input_dim` equals the vector length; reward bounds equal reward clipping bounds. `difficulty_beta` is inverse sampling temperature.

Each sampled occurrence has its own rollout ID. SHAE maintains a persistent UID baseline. DA-CVQ receives occurrence-level mean clipped rewards after policy optimization and maintains separate code-level memory. SHAE-off uses group-centered sample-standard-deviation GRPO. RARS-off sets error multipliers to one while retaining geometry, correct-class reward, and the missing-match term. Curriculum-off uniformly shuffles every epoch and performs no visual-memory update.

Epoch directories (`epoch_0001`, `epoch_0002`, ...) contain model/processor files, `training_state.pt`, `training_config.json`, and `training_history.json`. Continuation restores optimizer, SHAE baseline, codebook, sampler epoch, and random streams. The loaded manifest and mathematical configuration identify the run; output directory, continuation path, and total target epoch count may change.

## Frozen visual vectors

Run `python -m inference.visual_features --config ...` with `manifest`, `output`, `batch_size`, `model`, `generation`, and `projection`. Select the same seed's SFT checkpoint.

`projection` is either `{"path": "projection.npy"}` to load a matrix, or an object with explicit `seed`, `dimension`, and `output` to construct and save a Gaussian-QR orthonormal projection. Its row dimension equals the vision hidden dimension. The extractor reads final Qwen3-VL visual-block patches before the final merger and normalization, pools each phase, applies the shared projection, concatenates a/p/v, and normalizes the vector. It writes a manifest containing vectors and resolved image paths. Reuse the same projection for inputs sharing this visual space.

## Inference

Run `python -m inference.run --config ...` with `manifest`, `output`, `batch_size`, `seed`, `model`, and `generation`. Select the trained checkpoint and its saved processor. Output is JSONL with one row per group: `uid`, `patient_id`, `completion`, and `nll`.

## Evaluation

Run `python -m evaluation.evaluate` with `--manifest`, `--predictions`, `--iou-thresholds` (one or more values in `[0,1]`), and `--output-dir`. Prediction UIDs must cover groups exactly once. Threshold zero means `IoU > 0`; positive thresholds mean `IoU >= threshold`.

Outputs are `metrics.json`, `patient_predictions.jsonl`, and `group_predictions.jsonl`. Patient accuracy and macro precision/recall/F1 use the fixed patient and class populations. Groups vote by lesion-label majority; patient ties use mean finite NLL, then class order. Patients without votes remain in the metrics with undefined predictions.

Detection uses lesion-positive phase slices. CA maximizes total IoU. GFCA filters that same assignment by class agreement. Same-class-first maximizes the number of same-class pairs, then total IoU. Precision/recall/F1 use TP/FP/FN at every threshold. Mean IoU uses unthresholded CA divided by all GT lesions. Acc@IoU>0 uses accepted positive-overlap matches divided by all GT lesions.
