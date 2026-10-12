# Training pipeline

`training/run_verl_lhm_variant.py` exposes all eight combinations of RARS, SHAE, and the coupled DA-CVQ/VGS curriculum. `training/verl_lhm_launcher.py` adapts the launcher to a local VeRL installation. These commands generate configurations or start new runs; configuration generation does not reproduce the paper's reported measurements.

The reward path is `rewards/verl_liver_reward.py`; SHAE and DA-CVQ/VGS state are implemented in `rl/core.py` and `rl/verl_lhm_extensions.py`. The code does not bundle VeRL, vLLM, Ray, CUDA, data, or weights.

## Convert train and validation sources

For the reported primary private-cohort experiments, model training used the 654 training patients. The 117 validation patients were used for parameter selection and were not added to the final model's training set. To preserve this protocol, use the train-only output below; the optional combined-refit output describes a different data-use policy.

The converter accepts already prepared three-image groups and structured targets. It checks that the three referenced image files exist; it does not select CT slices, register phases, or extract visual embeddings.

```bash
python preprocess/build_verl_real961_dataset.py \
  --data-root /path/to/images \
  --train-jsonl /path/to/train.jsonl \
  --validation-jsonl /path/to/validation.jsonl \
  --output-dir ./outputs/prepared
```

The default outputs are:

| File | Source and role |
| --- | --- |
| `real961_rl_train.jsonl` | Only the supplied training records. |
| `real961_rl_validation.jsonl` | Only the supplied validation records; written when that source is provided. |
| `real961_rl_monitor.jsonl` | Validation records when available; otherwise at most the first 64 training records for training diagnostics, not held-out validation. |
| `audit.json` | Contract version 2: source/output SHA256 values, patient/group counts, output roles, and the current conversion's class-count definitions. |

The converter rejects overlap in either supplied `patient_id` or `statistical_patient_id`, duplicate input-group UIDs, empty training sources, and explicitly supplied empty validation sources. It preserves the caller's source partition; it does not establish how that partition was created.

For this converter, `class_counts` counts **positive three-phase input groups by `gt_major`**, not patients or individual lesion records. `gt_major` is the most frequent active lesion-record label in a group; equal counts use the lexical maximum label. Background groups and zero-count classes are omitted. The default training, validation, and monitor files all carry counts derived only from the supplied training source. The metadata documents this conversion policy; it does not establish the original cohort-specific dictionary values or their source split.

To explicitly prepare a combined training/validation refit dataset after model selection:

```bash
python preprocess/build_verl_real961_dataset.py \
  --data-root /path/to/images \
  --train-jsonl /path/to/train.jsonl \
  --validation-jsonl /path/to/validation.jsonl \
  --output-dir ./outputs/refit_prepared \
  --refit-train-validation
```

This adds `real961_rl_refit_train_validation.jsonl`; it never substitutes combined data into `real961_rl_train.jsonl`. The additional file carries class counts computed from both supplied sources. `audit.json` marks the explicit refit role and that the validation source is no longer held out from this combined file. Starting a refit requires passing that exact additional filename as `--train-file`. The flag requires a nonempty validation source and does not decide whether a refit is appropriate for an experiment.

## Eight component combinations

| `--variant` | RARS | SHAE | DA-CVQ/VGS |
| --- | --- | --- | --- |
| `vanilla_grpo` | off | off | off |
| `shae_grpo` | off | on | off |
| `dacvq_vgs_grpo` | off | off | on |
| `shae_dacvq_vgs` | off | on | on |
| `rars_grpo` | on | off | off |
| `rars_shae` | on | on | off |
| `rars_dacvq_vgs` | on | off | on |
| `full_rl_lhm` | on | on | on |

RARS off selects the existing `vanilla` reward mode; SHAE off selects the existing GRPO estimator; DA-CVQ/VGS off omits the custom curriculum sampler. DA-CVQ and VGS remain one coupled factor. The previously exposed five names keep their component settings and numeric hyperparameters. All combinations retain the existing learning rate `1e-6`, clipping `0.20`, KL coefficient `0.01`, two completions per prompt, and six formal epochs.

The following loop writes one manifest per combination without importing VeRL, checking GPUs, or loading model weights:

```bash
for variant in vanilla_grpo shae_grpo dacvq_vgs_grpo shae_dacvq_vgs \
  rars_grpo rars_shae rars_dacvq_vgs full_rl_lhm; do
  python training/run_verl_lhm_variant.py \
    --train-file ./outputs/prepared/real961_rl_train.jsonl \
    --validation-file ./outputs/prepared/real961_rl_validation.jsonl \
    --model-path /path/to/base-model \
    --output-dir "./outputs/configs/${variant}" \
    --variant "$variant" \
    --seed 2026 \
    --scripts-dir . \
    --n-gpus 4 \
    --config-only
done
```

Even in `--config-only` mode, the training file must exist and contain at least four records, and the model directory must exist. This mode writes `run_manifest.json` only; it does not validate weights, visual embeddings, or external runtime compatibility. The `--n-gpus` value describes the generated configuration.

`--validation-file` is optional and binds the supplied path and hash in the manifest. It must differ from the training path; the launcher does not inspect patient identities, so use the converter's split checks or an equivalent check for other inputs. The launcher retains `trainer.val_before_train=False` and `trainer.test_freq=-1`, so merely supplying this file does not enable validation or model selection. Without the option, the existing `data.val_files` placeholder points to the training file and the manifest explicitly marks it as not held-out validation.

Remove `--config-only` to start training in a prepared GPU environment. Curriculum-enabled variants require extracted `extra_info.visual_z`; this converter leaves it `null`. The launcher checks the first record before launch; sampler initialization validates all supplied vectors for finite, nonzero and consistent-dimensional values. Missing vectors follow the explicitly documented sampler fallback. A grid search requires a complete feature cache for every seed. For the reported protocol, each seed uses features extracted once from its own SFT checkpoint and held fixed during RL. The released reference extractor `scripts/extract_qwen_visual_latents.py` provides the complete local checkpoint, per-phase pooling, fixed projection and cache interface; see [visual_feature_extraction.md](visual_feature_extraction.md). It maps the native 1024-wide visual states to the 1152-dimensional cache without claiming equivalence to an earlier private extractor. Prepare one complete feature cache per SFT seed before launching. GPU `--smoke` runs use one step without curriculum or two with curriculum so the second epoch can consume learned sampling state. Neither configuration generation nor a smoke run establishes the full training result.

## Persistent input identity and rollout groups

Every training row used by SHAE or DA-CVQ/VGS must provide a nonempty
`extra_info.sample_uid` that uniquely identifies that input record across visits
and epochs. Dataset rows must have distinct persistent keys. VeRL's `uid` remains
the identifier for one sampled occurrence and its `G` completions. Sampling the
same input twice must produce two rollout groups, each with its own statistics.

SHAE groups by `uid` and reads/writes its EMA under `extra_info.sample_uid`.
DA-CVQ/VGS uses the same persistent key to look up the dataset vector and consumes
one mean reward per rollout group. Repeated visits are processed in their order
of first appearance in the current batch; their groups are never pooled. Missing
or inconsistent identity metadata, unknown sampler keys, and duplicate dataset
keys raise an error instead of silently disabling memory updates.

Apply the following additional bridge in the same separately selected VeRL
checkout as the reward-loop patch below:

```bash
python rl/patch_verl_sample_uid.py \
  --ray-trainer /path/to/verl/trainer/ppo/ray_trainer.py \
  --audit ./outputs/sample_uid_patch.json
```

The bridge supports the exact `compute_advantage` function at VeRL commit
`fc3a5acae155425909bc30c982c3cea0784ce3c2`. It adds the persistent key only to the
SHAE dispatch, preserves upstream rollout `uid` generation, and leaves GRPO and
policy losses unchanged. It backs up the selected file, is idempotent, and rejects
unrecognized source. SHAE-enabled launches verify the bridge before starting and
write `sample_uid_runtime_audit.json`; configuration-only mode does not perform
this runtime check. The launcher also passes the extension import path into Ray
workers. Deploy the same code and patched VeRL installation to every participating
node.

This correction affects training state. Continuing an old checkpoint does not
reconstruct past EMA or curriculum updates that were missed because of transient
identifiers. Historical results require a separate check against their actual
runtime; new runs should use the corrected identity path from initialization.

## Explicit token-level policy and runtime checks

The current launcher fixes `policy_loss.loss_mode=vanilla`,
`loss_agg_mode=token-mean`, lower/upper PPO clipping `0.20`, negative-advantage
dual clipping `3.0`, low-variance KL `0.01`, gradient clipping `1.0`, one PPO
epoch, the legacy FSDP worker and sequence-parallel size one. Rollout correction
weights and reward-side KL are disabled. The exact vanilla-loss and KL source
functions at VeRL commit `fc3a5acae155425909bc30c982c3cea0784ce3c2` are checked
before training. The source fixes log-ratio bounds `[-20,20]` and the low-variance
KL numerical bounds; `rl/policy_contract.py` records this contract.

The reference FSDP actor averages microbatch token means. With unequal response
lengths this is not the token mean over the complete optimizer minibatch. Apply
this additional version-restricted patch to the selected VeRL installation:

```bash
python rl/patch_verl_token_mean.py \
  --dp-actor /path/to/verl/workers/actor/dp_actor.py \
  --audit ./outputs/token_mean_patch.json
```

The patch counts retained tokens over the entire distributed PPO minibatch once.
Policy and KL numerators use that same denominator. The data-parallel size
compensates for FSDP gradient averaging; microbatch gradients are summed without
a second microbatch-count division. It rejects unknown source and is idempotent.
The launcher requires the patch and writes `policy_runtime_audit.json`, including
the explicit overrides and source hashes, before entering VeRL. `--config-only`
records the required configuration but does not perform that runtime check.

These checks establish the current supported code path, not the version used for
historical experiments or compatibility of an arbitrary external GPU stack.
Pin the full VeRL checkout at the stated commit and record the installed Torch,
Ray, vLLM and Transformers versions for each real deployment.

Independent SHAE state files are not atomically synchronized to VeRL actor
checkpoints. The current launcher therefore starts fresh (`resume_mode=disable`)
and refuses directories containing prior actor or EMA state. The joint-grid
runner retains interrupted run directories and restarts that unfinished seed
from its SFT model; completed candidate/seed tasks are reused without retraining.
This is search-task recovery, not a claim of exact mid-training checkpoint resume.

## CPU entry-point checks

The CLI checks use synthetic image groups and Python's standard library. Every converter and launcher subprocess runs with `-S`, which disables installed site packages:

```bash
python -S -m unittest discover -s tests -p test_training_data_cli.py -v
```

They exercise all eight configuration-only variants, separate/default/refit datasets, train-only count propagation, source/output manifests, patient overlap rejection, duplicate input groups, empty validation, monitor roles, and optional validation-file binding. They do not run GPU training or download any runtime or model.

## Reward-loop integration

`rl/patch_verl_agent_loop_logprobs.py` supplies the reward manager's
`rollout_log_probs` field from the agent loop's response log probabilities.
It recognizes one supported source block, backs up the original file as
`.pre_rl_lhm`, and records the before/after hashes. Reapplying it is idempotent;
unrecognized source layouts fail rather than receiving a guessed patch.

```bash
python rl/patch_verl_agent_loop_logprobs.py \
  --agent-loop /path/to/verl/experimental/agent_loop/agent_loop.py \
  --audit ./outputs/agent_loop_patch.json
```

Use this only in a separate, explicitly selected VeRL checkout. The integration
reference is [VeRL commit
`fc3a5acae155425909bc30c982c3cea0784ce3c2`](https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/verl/experimental/agent_loop/agent_loop.py); verify the actual source block before
applying it. This reference and a successful text patch are not a complete
runtime lock or a training validation. The helper requires upstream `torch`
and `TensorDict` bindings, aligns log probabilities to the response width,
and does not modify the policy loss. Verify generated-token masks and lengths
against the installed reward-loop runtime before training.

Deploy this complete package, including `evaluation/schema.py`, with the reward
callback. Copying only the reward file into an unrelated flat directory omits
its shared strict parser dependency.

## Joint weight selection

See [rars_joint_search.md](rars_joint_search.md) for the full joint-grid task, validation, recovery and freeze interfaces. The independent single-axis sensitivity protocol is not a joint-search substitute.
