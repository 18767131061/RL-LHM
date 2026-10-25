# RL-LHM

Core algorithms for **Risk-aware lesion-grounded reinforcement learning for structured vision–language diagnosis of liver lesions in multi-phase CT**.

The functions operate on structured completions, supplied annotations and class metadata, generated-token log probabilities and masks, and fixed visual features.

## Modules

| Module | Input → output | Paper correspondence |
| --- | --- | --- |
| `rewards/schema.py` | JSON completion + ordered classes → lesion records and background phases | Structured output contract |
| `matching/geometry.py` | Boxes / similarity matrix → IoU, GIoU, maximum-sum one-to-one assignment | Reward matching |
| `rewards/rars.py` | True/predicted classes + counts + severity prior + completion likelihood → confusion cost | Scarcity × severity × likelihood |
| `rewards/full_reward.py` | Completion + reference + explicit reward parameters → component scores and clipped reward | RARS reward equations |
| `rl/core.py` | Scalar clipped rewards + RID/UID + token mask + baseline state → token advantages | SHAE and GRPO replacement |
| `rl/policy_contract.py` | Current/rollout/reference token log probabilities + advantages + mask → scalar loss | Token-ratio surrogate, dual clipping and KL |
| `rl/completion_likelihood.py` | Token log probabilities + completion mask → mean NLL and likelihood proxy | Completion-only likelihood |
| `rl/visual.py` | Three phase patch tensors + masks + fixed projection → unit visual latent | Phase pooling and projection |
| `rl/curriculum.py` | Visual latents + group reward means → prototypes, reward memory and epoch indices | DA-CVQ / VGS |
| `rl/workflow.py` | Clipped rewards + RID/UID + UID-to-vector mapping → curriculum updates | Reward-to-memory connection |
| `aggregation/patient.py` | Parsed lesions → group vote; group votes + NLL → patient diagnosis | Deterministic aggregation |

## Inputs and parameters

- A structured completion is a JSON object with a `detections` list. Each record has `phase` (`a`, `p`, or `v`), `label`, and numeric `bbox_2d` coordinates `[x1, y1, x2, y2]` on `[0,1000]`. Background uses a unique `no-lesion` record with `[-1,-1,-1,-1]` in that phase. Classes and optional aliases are supplied by the caller.
- `RewardConfig` and `RARSParameters` require explicit coefficients, clipping bounds, the diagnostic GIoU threshold, and a complete symmetric off-diagonal severity mapping (rows: true class; columns: predicted class). `class_counts` contains training-input-group counts. The appendix's active-class lower-bound rule is applied, and supplied extra keys remain in the total.
- Localization uses an IoU assignment per phase and divides by its ground-truth lesion count. Diagnostic reward uses a separate GIoU assignment followed by thresholding. Only malformed model completions receive the schema-invalid zero reward; reference and parameter errors raise exceptions.
- RID identifies one sampled occurrence; UID identifies the persistent input. Repeated occurrences of a UID retain separate rollout groups. Keep the same `ShaeBaselineStore` across visits. SHAE reads the prior baseline, constructs the advantage, then updates the baseline once per RID. Its standard deviations use the population convention; the SHAE-off GRPO replacement uses sample standard deviation.
- Token log probabilities and binary masks must align. The mask retains generated tokens only. `grpo_token_loss` minimizes the negative surrogate plus KL. For a complete batch, explicitly pass `total_token_count=None`; for microbatches, pass the same full-update token count and sum the losses.
- `project_visual_latent` takes phase features in arterial, portal-venous, delayed order and a caller-supplied fixed column-orthonormal projection. `DACVQState` takes explicit EMA, diversity, revival, reward-bound and numerical settings. `difficulty_beta` is the inverse sampling temperature. Random draws use supplied generators.
- `VGSSampler` takes explicit epoch state, missing-vector multiplier and weight floor. Epoch zero with an uninitialized codebook uses a permutation; later epochs use replacement sampling. `use_vgs=False` uses a permutation every epoch.
- Group votes use all active lesion labels and fixed class-order ties. Patient votes use one vote per eligible group, then mean finite completion NLL and fixed class order for ties. No eligible votes yields `None`.

## Algorithm sequence

1. Use `compute_score` to obtain one clipped `score` per completion. Keep zero rewards from invalid outputs in their rollout groups.
2. Pass those scalar rewards, RID/UID and response masks to `compute_shae_advantage`, or use `compute_grpo_advantage` with SHAE disabled.
3. Compute `grpo_token_loss` from the generated-token probabilities, advantages and masks.
4. After the policy update, call `update_curriculum` with the same clipped rewards and fixed visual vectors. It updates separate code-level reward memory using each RID's mean reward. An explicit `None` vector skips its memory update.
5. Call `VGSSampler.sample_epoch()` for the next epoch's input positions. At inference, parse completions and apply the two-stage voting functions.

Function signatures and docstrings specify tensor shapes and state transitions. All experiment-dependent settings are caller inputs.

## Dependencies

Python 3.10 or newer; NumPy, SciPy and PyTorch:

```sh
python -m pip install -r requirements.txt
```

Code is distributed under the MIT License.
