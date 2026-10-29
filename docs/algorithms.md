# Algorithm contracts

## Structured reward

Parse all three input phases. A phase without GT lesions is background; an explicit `no-lesion` output is required to receive background credit. Invalid output receives the minimum structured utility, `-(detection_weight + classification_weight)`.

Within each phase, maximize total GIoU in a one-to-one assignment and retain pairs meeting `giou_for_classification`. The same eligible pairs define localization and diagnosis. Let `N` be the GT lesion count, `F` unmatched predictions, `M` unmatched GT lesions, and `D = max(N, 1) + F`.

- Localization utility: `(sum(matched IoU) - M - F) / D`.
- Diagnosis utility: `(number correct - sum(normalized error costs) - M - F) / D`.
- For a background phase, add `+1` to each numerator for an explicit correct background output, otherwise `-1`.

RARS error costs retain scarcity × severity × completion-likelihood factors and are divided by the maximum possible cost across the fixed active class space at likelihood one. This shared scale preserves relative risk weights and bounds each diagnostic error by one. RARS-off uses unit error costs. Each component lies in `[-1,1]`; average each component over phases, then apply its positive weight. Clipping bounds must enclose this entire range. Consequently, dropping an eligible wrong diagnosis cannot improve the score, and extra unmatched predictions reduce utility unless it is already minimal. Multi-lesion GT remains in the denominator when predictions omit lesions.

Class counts use the entire training manifest and the same class-order majority rule as group inference. They remain fixed across epochs and repeated sampling. They count input groups, not patients.

## SHAE and GRPO

RID identifies a sampled occurrence; UID identifies its persistent input. Each occurrence has `G >= 2` completions. For reward vector `r`, let `mu = mean(r)` and `s = population_std(r)`:

```text
group = (r - mu) / (s + epsilon)
b = previous_UID_EMA, or mu on first visit
history = (r - b) / (sqrt(mean((r - b)^2)) + epsilon)
alpha = sigmoid((s - gate_mid) / gate_temperature)
advantage = alpha * group + (1 - alpha) * history
```

Update the UID baseline once per occurrence after reading it. Both branches have magnitude at most `sqrt(G)`. First-visit equal rewards yield zero advantage. Jitter is disabled (`noise_std=0`), so random noise cannot create the first learning signal. SHAE-off uses the same population-standard-deviation `group` normalization. A historical mean shift can still produce same-sign advantages, with RMS-controlled magnitude.

Every worker receives the global scalar rewards in occurrence order and performs the same UID updates. Repeated UIDs on different workers therefore have deterministic sequential baseline semantics.

## Policy probabilities and objective

RL samples at temperature one, with full support (`top_p=1`, `top_k=0`) and no repetition penalty. Input-only multimodal markers are suppressed. Old, current, and reference policy probabilities use this same fixed suppression and renormalization. Raw model completion NLL is computed separately for the reward's likelihood factor and inference tie-breaking. The first EOS is retained; later padding and all prompt tokens are excluded.

The PPO token-ratio objective includes ratio clipping, negative-advantage dual clipping, and a reference KL term. Rollout tokens, old probabilities, reference probabilities and advantages stay fixed across `policy_updates`. Dropout is disabled; gradient checkpointing is active during differentiable scoring. Each independent multimodal forward reconstructs Qwen3-VL rotary offsets for its own input grid.

Each microbatch contributes a token loss sum divided by the real global token count. The training runtime multiplies by world size once to compensate for ZeRO's averaged gradients. Short final batches use zero-weight work slots to align collectives; those slots never enter reward populations, baselines, curriculum updates or token counts.

## DA-CVQ and sampling

Visual vectors come from the frozen SFT vision encoder: pool phase patches, apply one fixed orthonormal projection, concatenate a/p/v, and normalize. The supplied 384-dimensional phase projection produces 1,152-dimensional vectors.

After policy updates, each real occurrence updates DA-CVQ using its mean clipped reward. All active feature masses and accumulators decay on each update; the assigned code receives the new difficulty-weighted feature. Code-level reward EMA changes only on assignment. Low-mass codes can be revived. UID assignments are recomputed from the current codebook.

With curriculum enabled, epoch zero initializes from a permutation; later epochs use VGS probabilities. `sampling.replacement` explicitly controls later draws. Without curriculum it selects uniform replacement sampling or a uniform permutation, allowing matched sampling controls. Weighted sampling without replacement visits every input once and changes order only. Shared CPU generators and globally ordered updates make curriculum state identical on all workers. Missing visual vectors skip memory updates and use the configured sampling weight.

## Inference and evaluation

Each input group votes by lesion-label majority. Background records do not vote; class-order ties are deterministic. Each group contributes one patient vote. Patient ties use mean finite completion NLL, then class order. Patients without valid votes remain in classification denominators.

Main `detection` statistics count TP/FP/FN across **all input phase slices**, including predictions on background. `detection_positive_only` exposes conditional positive-slice metrics separately. Background FP count, FP per background phase, and FP per patient are reported. Unthresholded class-agnostic mean IoU retains all GT lesions in its denominator. Prediction UIDs must cover the manifest exactly once.
