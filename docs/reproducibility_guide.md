# Reproducibility guide

The evaluation CLI and shared parser implement explicit, testable contracts. This is distinct from establishing which code revision and settings produced a particular experimental table.

For a reproducible training and evaluation run, specify the following without distributing patient-identifiable data:

- **Training objective:** the actual policy-loss implementation, token- or completion-level probability ratio, clipping, masking and aggregation. The launcher delegates policy optimization to an external VeRL runtime. Its configuration manifest is not a substitute for checking that runtime's loss implementation.
- **Runtime:** the exact VeRL, vLLM, PyTorch, Ray and Transformers versions, plus the patches used to propagate rollout log probabilities and persistent sample identity, and any changes to policy optimization.
- **Visual features:** encoder checkpoint, feature layer, pooling, combination of phases, normalization, and extraction/caching schedule for `visual_z`. The dataset converter does not extract these vectors, and the curriculum-enabled variants require them.
- **Input construction:** CT preprocessing and position-selection rules, patient and group identifiers, per-phase ground-truth annotations, and the complete intended input-key set.
- **Class counts:** actual `class_counts`, counting unit, source split, extra keys and freeze point. Metadata emitted by a new conversion describes that conversion; it does not establish what was used in an earlier experiment.
- **Model selection:** the candidates actually evaluated, how each candidate maps to a trained policy, any reuse or refitting, and the held-out data used for selection and final evaluation.

The main evaluation CLI rejects incomplete output coverage. An invalid completion that was actually generated is represented by its record and contributes no diagnostic vote; a missing output file or record is not silently treated as an invalid completion. Patients with no eligible votes stay in the classification denominator.

When replacing a legacy parser or evaluator, first establish the input and implementation used for the results being reproduced. If an evaluation difference occurred, reassess the affected metrics from the corresponding outputs. If the difference affected training rewards or the optimization objective, evaluation-only reruns may not establish equivalence.

Synthetic fixtures demonstrate interface behavior and regression coverage. They are not clinical evidence, historical run records, or a substitute for comparing against the actual experimental inputs and model outputs.

## Executable checks

Run `./scripts/run_required_tests.sh --cpu-only` after installing
`requirements.txt`. Omit `--cpu-only` when PyTorch and OmegaConf are also available to include
the RL-core, sample-identity, and preprocessing/inference helper tests. Neither command starts
model inference, GPU training, or a scheduler. The agent-loop helper's synthetic
checks cover source replacement and refusal behavior, not an installed VeRL
reward-loop execution.

The manifest and prediction schemas are documented in `metrics_pipeline.md`;
training and the explicit reward-loop patch are documented in
`training_pipeline.md`. Keep datasets, weights, generated outputs, test logs,
and environment files outside the public repository.

The identity integration tests additionally accept a local copy of the supported
VeRL `ray_trainer.py` (they do not download it or import a complete VeRL runtime):

```bash
RL_LHM_VERL_RAY_TRAINER=/path/to/verl/trainer/ppo/ray_trainer.py \
  ./scripts/run_required_tests.sh
```

These tests execute the reference UID assignment and patched advantage function
with real PyTorch tensors and synthetic batches. They check cross-batch memory,
distinct groups for repeated samples, sampler updates and nonuniform weights.
Without the reference source, the source-integration cases are explicitly skipped.
Passing them does not reproduce a distributed training run or historical results.
