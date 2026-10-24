# RARS one-axis-at-a-time validation search

`training/rars_oat.py` prepares a **new** OAT search and verifies its supplied
validation results. It neither supplies historical scores nor establishes that a
new run reproduces the paper's published Selected weights. It does not execute
training or inference.

## Design and counts

The private cohort has four named pair axes plus `other`; the public cohort has
six named pair axes plus `other`. Pair names and class vocabularies are shared
with `training/rars_joint_grid.py`. Each axis takes all 15 values
`0.1, 0.2, ..., 1.5`; **every other weight is held at 1.0**. All pairs are
unordered and symmetric. `other` assigns the same weight to every off-diagonal
pair outside the explicitly named axes. Diagonal penalties remain zero.

Every profile uses seeds `42, 2026, 3407, 1234, 5678`. The all-one profile appears
on every axis but is trained once per seed:

| Cohort | Axes | Axis points | Axis × seed records | Unique profiles | Unique training tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Private | 5 | 75 | 375 | 71 | 355 |
| Public | 7 | 105 | 525 | 99 | 495 |

`plan.json` preserves every axis-point-to-task mapping. Counts describe the
planned workflow, not a claim that these experiments have already been run.

The risk JSON retains `cohort`, `pair_weights` (for example `HCC--ICC`), and
`other_weight`, with explicit `protocol: "oat"`. The shared profile validator,
launcher `--risk-profile`, and reward callback accept this protocol. Existing
joint-grid and LOCO profiles retain their fixed `other_weight: 0.4` rule. The
combined profile also uses `protocol: "oat"`; this records how it was selected
and does not imply that only one of its coordinates differs from one.

## Plan

Prepare five seed-specific SFT checkpoints and matching training feature caches,
an explicit held-out source-validation manifest, and an identity-only reserved
test manifest. Use the existing joint-grid config fields with `protocol: "main"`
and the full values list. Private selection uses P1; public selection uses U1.
`synthetic_test` must be false for actual experimental results.

This is an illustrative configuration; replace paths and choose the training
command explicitly for your environment:

```json
{
  "cohort": "private",
  "protocol": "main",
  "split_id": "P1",
  "selection_split": "validation",
  "formal_test_used_for_selection": false,
  "synthetic_test": false,
  "seeds": [42, 2026, 3407, 1234, 5678],
  "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
  "train_file": "cache/seed42.jsonl",
  "seed_train_files": {
    "42": "cache/seed42.jsonl",
    "2026": "cache/seed2026.jsonl",
    "3407": "cache/seed3407.jsonl",
    "1234": "cache/seed1234.jsonl",
    "5678": "cache/seed5678.jsonl"
  },
  "seed_model_paths": {
    "42": "sft/seed42", "2026": "sft/seed2026", "3407": "sft/seed3407",
    "1234": "sft/seed1234", "5678": "sft/seed5678"
  },
  "validation_manifest": "validation.json",
  "test_identity_manifest": "reserved_test_identities.json",
  "output_root": "oat_runs",
  "training_command_argv": [
    "python", "training/run_verl_lhm_variant.py",
    "--train-file", "{train_file}", "--model-path", "{model_path}",
    "--output-dir", "{output_dir}", "--variant", "full_rl_lhm",
    "--seed", "{seed}", "--risk-profile", "{profile_path}",
    "--scripts-dir", ".", "--n-gpus", "4"
  ]
}
```

```bash
python training/rars_oat.py plan --config /absolute/path/oat_config.json
```

The config's data, model, and output paths resolve relative to the config file.
The example command is intended to run from the repository root; argument values
are passed as an argv list, without shell expansion. There is no default command
and the planner never executes the command. All five task placeholders are
required. Unknown placeholders, unavailable model files, invalid feature caches,
and overlapping train/validation/test identities fail planning.

The printed `plan_path` identifies an immutable input plan. It contains the
current data and relevant source-file hashes, checkpoint-content hashes, profile
JSONs, and one `task.json`, `result_template.json`, and
`inference_receipt_template.json` for each unique task.
`task.json` provides the fully substituted `training_command_argv`. Checkpoint
hashing reads file contents, including weights, so planning can take time on
large checkpoints. Repeating the exact plan is idempotent; changed inputs create
a separately identified plan. Finish code and data preparation before planning.

## Train, infer, and record actual evidence

Run each task's explicitly provided training command in your training environment.
The collector requires the existing launcher's `run_manifest.json` and
`training_complete.json` under that task's training directory, a completed
`full_rl_lhm` run, the intended seed/cache/profile/SFT checkpoint, and its final
checkpoint. Configuration-only and smoke runs are rejected. A custom command
must produce that same completion/provenance contract. The run's saved runtime
environment and its actual VeRL command must both bind `RL_LHM_RISK_PROFILE`
to this task's profile; a profile hash in a summary field alone is insufficient.

Generate **held-out validation** predictions from each completed final checkpoint
using the schema accepted by `evaluation/evaluate_paper_protocol.py`: one row per
validation `group_id`, the model's actual `raw` completion, and its completion NLL
when available. Compute Macro-F1 with that evaluator and fill the corresponding
`result_template.json` with real artifact paths, hashes, and the metric. The
template's null fields are deliberately not valid results. In each filled record:

| Field | Required content |
| --- | --- |
| `task_id`, `profile_id`, `seed`, `input_hash` | Exact values from its generated task |
| `partition`, `split_id`, `validation_sha256` | The plan's validation identity |
| `synthetic_test` | The same explicit Boolean as the plan |
| `source`, `source_sha256` | Absolute predictions JSONL path and SHA-256 of its bytes |
| `checkpoint`, `checkpoint_sha256` | Absolute completed final checkpoint directory and its content digest |
| `inference_receipt`, `inference_receipt_sha256` | Absolute inference receipt path and SHA-256 of its bytes |
| `metrics` | Exactly `{"macro_f1": number}` in `[0,1]`, using proportions, not percentages |

Hashes are computed by the shared `sha(path)` for files and
`training.rars_oat.checkpoint_digest(path)` for checkpoint directories. The latter
hashes a canonical mapping of relative file names to their SHA-256 content hashes.
Do not use a directory's name or modification time as its digest.

The inference producer must emit an independent receipt after its validation
inference completes, using the generated `inference_receipt_template.json`
schema. Populate it from the task/checkpoint/input actually used and the output
actually produced. It must contain the exact task identity fields from the table
above, `kind: "oat_validation_inference"`, `status: "inference_complete"`, and
the `source`, `source_sha256`, `checkpoint`, and `checkpoint_sha256` fields.
The generated template has `status: "pending"` and null artifact fields, so it
cannot be accepted as completed evidence. The collector verifies this receipt's
file hash and its binding of task, profile, seed, input hash, validation split,
checkpoint, and prediction artifact. Substituting one task's predictions or
receipt into another task fails. Two independently recorded runs may produce
identical greedy predictions; equal output-content hashes alone are not treated
as duplicate experiments.

Combine the filled records into one JSONL file with exactly one record per unique
task. Shared all-one tasks appear once in this file; the plan maps their five
results to every applicable axis point. Do not copy them into extra result rows.

## Collect and select

```bash
python training/rars_oat.py collect \
  --plan /absolute/path/oat_runs/oat_HASH/plan.json \
  --results /absolute/path/actual_validation_results.jsonl \
  --output /absolute/path/new_oat_selection
```

The output directory must be absent or empty. Collection fails with a nonzero
exit code if a task is missing, repeated, or unexpected; if identifiers or input
hashes differ; if training is unfinished; if inference provenance is unbound;
if a checkpoint/source is missing or
changed; or if metrics are absent, nonfinite, or out of range. It also rejects
test partition identities, incorrect patient identities, and unknown/duplicate
prediction groups. The canonical evaluator checks complete prediction coverage
and recomputes every Macro-F1; supplied values must agree within `1e-12`. A file
named “validation” is insufficient by itself: split identities, patient/group
coverage, and hashes must match the plan.

`selection.json` contains the verified result references and all 75/105 axis
points. Each axis chooses the largest five-seed mean Macro-F1. **An exact mean
tie chooses the smaller weight.** Means use the five prescribed seeds equally;
neither the best seed nor a partial seed set is used. `combined_profile.json`
combines these axis winners and is accepted by the normal launcher risk-profile
argument.

The combined weights are **not established as a joint optimum**. Each axis was
measured while other weights equalled 1.0; interactions in the combined profile
have not been measured. Run a separately identified final five-seed training and
validation for the combined profile before reporting its performance. The tool
does not assign the axis means to this final combination, and writes
`combined_profile_validation_metric: null` and
`combined_profile_requires_training: true`.

## Verification scope

`tests/test_rars_oat.py` uses clearly marked synthetic checkpoint and prediction
fixtures to test count arithmetic, background weights, legacy profile behavior,
all-five-seed averaging, ties, evidence rejection, input drift, and a complete
355-task synthetic collection through the actual evaluator. No GPU training is
performed by these tests. Synthetic plans/outputs remain labeled synthetic.

Hash and receipt checks establish consistency of the supplied artifacts; they do
not independently establish how a user-produced file or model was created. Real
training logs, model files, data provenance, and inference evidence remain
necessary experimental records. The workflow always records
`historical_results_verified: false` and never manufactures earlier logs or
claims to reproduce previously reported Selected weights.
