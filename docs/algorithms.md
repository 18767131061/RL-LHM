# Algorithm contracts

The reward, advantage and curriculum definitions below follow the manuscript equations and Appendices C–D. The DeepSpeed and VeRL RL backends share these modules and settings. DeepSpeed uses Hugging Face generation and ZeRO-2 optimization; VeRL uses its vLLM rollout, FSDP2 actor/reference scoring and policy-update interfaces.

## Structured reward

Completions use the explicit a/p/v schema. An invalid completion receives reward zero. A missing phase is schema-valid but receives no localization credit. A correct background phase requires the explicit `no-lesion` sentinel and no predicted lesion. Termination metadata is reported separately; schema validity controls reward eligibility.

Localization and diagnosis use separate one-to-one assignments within each phase:

- Localization maximizes total IoU without an acceptance threshold. Sum assigned IoUs and divide by the number of GT lesions; unmatched GT lesions contribute zero. A phase with explicit GT and predicted background receives one, otherwise a lesion-free phase receives zero. Average the three phase scores.
- Diagnosis maximizes total GIoU, then retains assigned pairs with `GIoU >= giou_for_classification`. Each correct diagnosis contributes one. Each wrong diagnosis contributes minus its RARS cost. Average over all retained lesion pairs across the completion, or use zero if none qualify.
- Add `-missing_match_penalty` only when there are no retained diagnostic pairs and no correctly predicted background phases. There is no separate per-FP term in this objective.

RARS cost is scarcity × severity × completion-likelihood multiplier. With `N` the protected sum of the supplied count dictionary and `N_true` its protected true-class count:

```text
scarcity = (N / (N_true + count_epsilon)) ** scarcity_alpha
severity = exp(severity_kappa * severity_matrix[true][predicted])
likelihood = exp(-max(0, completion_nll))
cost = scarcity * severity * (1 + confidence_eta * likelihood)
raw_reward = detection_weight * localization + classification_weight * diagnosis + miss_term
reward = clip(raw_reward, clip_min, clip_max)
```

The supplied settings use component weights one, missing-match magnitude 0.1, classification GIoU threshold 0.30, and reward clipping `[-6,3]`. Costs are not divided by a maximum-risk normalization. RARS-off uses one for each wrong-label cost while retaining the other terms and matching rules.

Class counts are derived once from the complete fitting manifest and held fixed across epochs and repeated sampling. Each positive input group contributes one count for its most frequent GT lesion label; ties use the lexicographically greatest internal label. This training metadata rule is separate from inference's fixed class-order tie rule. Counts describe input groups, not patients. Background-only groups contribute no count. The reward API retains extra dictionary keys in the total and protects missing/zero active-class counts with a lower bound of one, as specified in Appendix C.2.

## SHAE and GRPO

RID identifies a sampled occurrence; UID identifies its persistent input. Each occurrence has `G >= 2` completions. For reward vector `r`, let `mu = mean(r)` and `s = population_std(r)`:

```text
group = (r - mu) / (s + epsilon)
b = previous_UID_EMA, or mu on first visit
history_raw = r - b + Normal(0, noise_std^2)
history = history_raw / (population_std(history_raw) + epsilon)
alpha = sigmoid((s - gate_mid) / gate_temperature)
advantage = alpha * group + (1 - alpha) * history
```

The history branch is not re-centered by the current group mean. The supplied jitter standard deviation is 0.001. Read the previous UID baseline before forming advantages, then initialize/update it once per occurrence with the group mean. Jitter is training-only; a zero configured scale disables it. The epsilon is a numerical denominator guard and does not impose an advantage magnitude bound.

SHAE-off uses `(r - mu) / (sample_std(r) + epsilon)` with divisor `G-1`, and no baseline, jitter or gate. This differs from SHAE's population-standard-deviation convention; with `G=2` and negligible epsilon, the nonidentical-reward group branch differs by a factor of `sqrt(2)`, as stated in Appendix C.1.

UID updates follow global occurrence order. DeepSpeed workers receive the global scalar rewards and each maintain the same baseline state; the VeRL Ray driver owns one baseline store. Repeated UIDs share their history baseline but preserve distinct rollout-group statistics.

## Policy probabilities and objective

The paper rollout configuration uses temperature one, `top_p=0.95`, `top_k=0`, and no repetition penalty. Raw model token log-probabilities define the policy objective; generation-only token suppression does not renormalize the scored model distribution; completion-only average NLL separately supplies the RARS likelihood factor and patient-level tie-break. The first EOS is retained; later padding and all prompt tokens are excluded.

The PPO token-ratio objective includes ratio clipping at 0.20, negative-advantage dual clipping at 3.0, and a reference KL term with coefficient 0.01. Token log-ratios and reference-minus-current log-probabilities are bounded to `[-20,20]`; the low-variance KL estimate `exp(d)-d-1` is clipped to `[-10,10]`. Rollout tokens, old probabilities, reference probabilities and advantages stay fixed across `policy_updates`. Dropout is disabled during differentiable scoring. Each independent multimodal forward reconstructs Qwen3-VL rotary offsets for its own input grid.

Each microbatch contributes a token loss sum divided by the real global token count. The DeepSpeed runtime compensates for ZeRO's averaged gradients once; VeRL uses the FSDP carrier's global loss-mask token count. Short final batches use zero-weight work slots to align collectives; those slots never enter reward populations, baselines, curriculum updates or token counts. The VeRL driver computes RARS after raw actor log-probabilities are available, supplies the shared SHAE/GRPO advantages, and applies KL only through the actor loss.

## DA-CVQ and sampling

Visual vectors come from the frozen SFT vision encoder: pool phase patches, apply one fixed orthonormal projection, concatenate a/p/v, and normalize. The supplied 384-dimensional phase projection produces 1,152-dimensional vectors.

After policy updates, each real occurrence updates DA-CVQ using its mean clipped reward. New prototypes are admitted while capacity is available and the diversity condition is met. Initialization ends when capacity is reached, or at the next epoch boundary if at least one prototype exists. Assignments use the current nearest active codeword.

The assigned feature mass and accumulator decay per assigned visit and receive the difficulty-weighted feature, with `difficulty = 1 - (mean_reward - reward_min)/(reward_max - reward_min)`. Unassigned code states remain unchanged unless revived. Code-level reward memory is initialized by its first reward observation, then updated by its own EMA. With an initialized codebook, a draw below the revival probability resets the first active code whose feature EMA mass is below `dead_code_threshold` (0.25). Revival uses the current vector plus Gaussian noise, resets mass to one and reward memory/count to zero. Activity counters may be stored for diagnostics but do not determine revival.

VGS computes active-code `softmax(-reward_memory / temperature)`, maps each available vector to its nearest prototype, assigns missing vectors `1.5 * max(code_probability)`, floors each input weight at `1e-4`, then normalizes over inputs. Missing vectors skip memory updates. Disabled or uninitialized VGS uses all-one weights.

With curriculum enabled, an epoch-zero sampler with an uninitialized codebook draws a random permutation. Subsequent epochs draw directly from normalized weights, with replacement in the paper configuration, including the all-one case. `sampling.replacement` controls these curriculum draws. With curriculum disabled, every epoch is a random permutation of all inputs drawn from the independent order generator, regardless of `sampling.replacement`. DeepSpeed uses shared CPU generators and globally ordered updates on all workers; VeRL maintains these generators and curriculum state on its Ray driver. Both save the state after the epoch's policy and curriculum updates.

## Inference and evaluation

Each input group votes by the most frequent lesion label. Background records do not vote; class-order ties are deterministic. Each eligible group contributes one patient vote. Patient ties use mean finite completion NLL, then fixed class order among minimum-NLL candidates. Patients without valid votes remain in classification denominators. Eligibility follows the structured schema and presence of a diagnostic lesion label; termination statistics are separately reported.

Paper lesion-level detection uses **lesion-positive phase slices**. Background phase slices are excluded from those TP/FP/FN denominators. The default `detection_scope=positive_phase_slices` selects this scope for `detection`. `detection_all_input` and background FP summaries are separate diagnostic outputs; an explicit all-input scope is also available. Unthresholded class-agnostic mean IoU retains all GT lesions in its denominator. Prediction UIDs must cover the manifest exactly once.

Geometry uses power-of-two axis scaling and compensated union-area summation. GIoU uses a common denominator when its product is in the normal floating-point range, with divide-first evaluation for extremely thin boxes. Integer boxes in the `[0,1000]` schema retain exact area arithmetic before the final ratio. Comparisons use `>=` for positive IoU thresholds and the classification GIoU gate, and `> 0` for the positive-overlap endpoint; no acceptance tolerance is added.

Macro classification averages only the explicitly fixed `metric_classes`; the complete `output_classes` vocabulary remains legal for parsing and all voting stages. An out-of-target prediction contributes an error and a false negative to its true class. Missing classes in bootstrap draws retain zero per-class metrics in the fixed macro denominator.

The paper's CA protocol maximizes total IoU before thresholding. Geometry-first class-aware filtering uses that same assignment; same-class-priority matching optimizes class agreement before total IoU. Optional threshold-aware protocols instead maximize eligible pair count first, then eligible IoU sum. These alternatives are separately named and do not replace the paper's primary matching convention.
