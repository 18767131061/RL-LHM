# RARS one-at-a-time validation workflow

`training/rars_oat.py` prepares, executes and collects an OAT validation scan,
then trains and validates the combined Selected profile separately across all
five seeds. `scripts/plot_rars_oat.py` renders the one-dimensional curves.

## Design and metrics

The private P1 scan contains four named pair axes and a shared `other` axis. The
public U1 scan contains six named pair axes and `other`. Every axis has the 15
values `0.1, 0.2, ..., 1.5`, with **all other severity weights fixed at 1.0**.
Named pairs are unordered and symmetric; `other` applies to every unlisted
off-diagonal pair. Diagonal penalties are excluded.

All points use seeds `42, 2026, 3407, 1234, 5678`. The all-one point belongs to
every axis and shares one training task per seed:

| Cohort | Axes | Axis points | Axis × seed records | Unique profiles | Scan training tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Private | 5 | 75 | 375 | 71 | 355 |
| Public | 7 | 105 | 525 | 99 | 495 |

The collector retains patient-level **Accuracy, macro-Recall and macro-F1 for
every seed**. Curves use the arithmetic mean of the five values. Each macro-F1
band is the 95% percentile interval from all `5^5 = 3,125` ordered bootstrap
resamples of those five seed values, using **linearly interpolated 2.5th and
97.5th percentiles**. The resampling unit is the independently trained seed.

An axis selects the weight with the greatest five-seed mean macro-F1; an exact
tie selects the smaller weight. Combining the independently selected weights
produces an additional configuration requiring **five separate training and
validation tasks**, even if it equals a scanned profile. It does not establish a
joint optimum. Existing two-dimensional RARS surface plotting is a separate
interface.

## Inputs

Supply the original seed-specific SFT checkpoints and matching training feature
caches, held-out validation manifest, reserved test identities, validation
inference input JSONL, image root and frozen source-key/group map. The config
uses the shared `training/rars_joint_grid.py` input fields. The fitting population
and its count metadata must remain the same across seed caches. Main selection
uses `protocol: "main"`, P1 or U1, and the full 15 values. Risk profiles emitted
by this workflow use `protocol: "oat"`.

Paths in the configuration resolve relative to that configuration. Commands run
from the repository root and use argv lists without shell expansion. Set the
Python executable paths for the installed training, VeRL merger and inference
environments explicitly. No input model, data file or inference artifact is
created when its required path is missing.

Example private configuration (replace the data/model/runtime paths):

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
    "42": "cache/seed42.jsonl", "2026": "cache/seed2026.jsonl",
    "3407": "cache/seed3407.jsonl", "1234": "cache/seed1234.jsonl",
    "5678": "cache/seed5678.jsonl"
  },
  "seed_model_paths": {
    "42": "sft/seed42", "2026": "sft/seed2026", "3407": "sft/seed3407",
    "1234": "sft/seed1234", "5678": "sft/seed5678"
  },
  "validation_manifest": "validation_manifest.json",
  "test_identity_manifest": "reserved_test_identities.json",
  "inference_input": "validation_inputs.jsonl",
  "image_root": "/absolute/path/images",
  "group_map": "validation_group_map.csv",
  "inference_format": "inference_csv",
  "output_root": "oat_runs",
  "training_command_argv": [
    "/absolute/training/env/bin/python", "training/run_verl_lhm_variant.py",
    "--train-file", "{train_file}", "--model-path", "{model_path}",
    "--output-dir", "{output_dir}", "--variant", "full_rl_lhm",
    "--seed", "{seed}", "--risk-profile", "{profile_path}",
    "--scripts-dir", ".", "--n-gpus", "4"
  ],
  "inference_command_argv": [
    "/absolute/evaluation/env/bin/python", "training/rars_oat_infer.py",
    "--checkpoint", "{checkpoint}", "--validation-manifest", "{validation_manifest}",
    "--inference-input", "{inference_input}", "--data-root", "{image_root}",
    "--processor-source", "{sft_model}",
    "--output", "{output_path}", "--seed", "{seed}",
    "--merge-python", "/absolute/verl/env/bin/python",
    "--inference-python", "/absolute/inference/env/bin/python",
    "--device", "cuda", "--decoding", "greedy", "--max-new-tokens", "256"
  ]
}
```

The inference adapter exports the completed VeRL `actor` with
`verl.model_merger`, validates the exported Hugging Face model, and runs
`inference/infer_hf_vlm_revision.py` with explicitly supplied decoding settings.
The adapter requires both `--decoding greedy|sample` and `--max-new-tokens`.
The example chooses greedy decoding and 256 maximum new tokens; set these to
the intended evaluation recipe. Sampling options are `--temperature`, `--top-p`
and `--top-k`, with named defaults `1.0`, `1.0` and `0`. All effective settings
are forwarded to inference and recorded in `.oat_identity.json`; changes reject
reuse of the same adapter output. Greedy mode rejects nondefault sampling
settings through the shared decoding validator.

The example explicitly uses the corresponding seed's SFT checkpoint as
`--processor-source`. The adapter loads its processor with
`AutoProcessor.from_pretrained(..., local_files_only=True)`, saves processor and
tokenizer assets to a staging directory, and copies them into the model export.
This runs in the specified inference Python environment. Staged model weights,
`config.json` and `generation_config.json` are rejected before copying; trained
model weights and configuration remain those of the completed checkpoint.
The processor source's non-weight files are content-hashed in the adapter
identity, and the OAT plan also hashes the complete source SFT checkpoint.
Changed assets invalidate resume. Without `--processor-source`, the adapter
uses the assets emitted by the configured merger.

The pinned VeRL merger attempts to save a processor and tokenizer from the
actor checkpoint's `huggingface` directory, but its processor loader can return
`None` after an error; supplying the explicit source makes that dependency
concrete. See the pinned [merger](https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/verl/model_merger/base_model_merger.py)
and [processor loader](https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/verl/utils/tokenizer.py).
The export validator checks model configuration and nonempty weight files; the
inference engine then loads the exported processor locally. Its inference input
and CSV group map must describe the same validation groups. The map uses
`logical_key,slice_index,group_id,patient_id` columns; optional `source_patient_id`
handles the source file's patient identifier. The canonical importer enforces
complete coverage and preserves all raw completions, including empty or invalid
ones. See `evaluation/import_inference_predictions.py` for that input contract.

An alternate backend can use `inference_format: "canonical_jsonl"`, emitting
one row for each manifest `group_id`, its actual `raw` completion and
`completion_nll`. Training commands must bind all five placeholders shown above.
Inference commands must bind `checkpoint`, `validation_manifest`, `seed` and
`output_path`; optional placeholders are `task_path`, `profile_path`,
`inference_input`, `image_root`, `group_map` and `sft_model` (the task's seed SFT checkpoint). Custom commands follow the same
training completion and inference-output contracts. Local `.py`/`.sh` command
operands are hashed automatically; list additional imported wrapper sources in
`execution_source_files` so they are included in the plan's input identity.

## Execute and resume

```bash
python training/rars_oat.py plan --config /absolute/path/oat_config.json
python training/rars_oat.py run --config /absolute/path/oat_config.json
```

`plan` only writes immutable profiles, task files and pending result/receipt
templates. `run` performs the complete sequence: plan, training command, final
checkpoint validation, validation inference, canonical import, canonical
patient evaluation, inference receipt, all-task collection, then separate
five-seed training/validation of the combined Selected profile.

For bounded runs or a scan-only stage:

```bash
python training/rars_oat.py run --config /absolute/path/oat_config.json --max-tasks 10
python training/rars_oat.py run --config /absolute/path/oat_config.json --stage scan
python training/rars_oat.py run --config /absolute/path/oat_config.json --stage combined
```

`--max-tasks` limits newly executed tasks and returns `status: "incomplete"`
without selecting from a partial scan. Repeat the same command to continue.
`--stage combined` requires all scan results and verifies them before deriving
the combined profile. A file lock permits one executor per plan.

Each completed task is validated before it is skipped. Planning binds the
configuration, fitting/validation data, reserved test identities, source files
and the contents of all five starting checkpoints. Images explicitly referenced
by the prepared training and inference records are bound by their file contents;
their inventory is checked during execution. Changed inputs define a new
plan; a collector cannot use an old plan with changed bound inputs. During a run,
file-inventory checks detect fitting-data, model and validation changes. Input
and task-checkpoint digests are reused while their complete inventory and file
metadata remain unchanged. A metadata change triggers content verification
against the bound plan or checkpoint digest; unchanged content refreshes the
metadata cache, while changed content or file names stop execution. File
snapshots must also remain stable throughout verification. A failing subprocess
stops execution and produces no
completed result for that task. A backend may resume its own partial training or
inference under the unchanged task identity.

The output training directory must contain the launcher's `run_manifest.json`
and `training_complete.json`, with a completed `full_rl_lhm` run, the specified
seed/cache/profile/SFT checkpoint, and its final checkpoint. Configuration-only
and smoke runs cannot satisfy completion. Both the runtime environment and
actual training command must bind the task's `RL_LHM_RISK_PROFILE`.

## Outputs and manual collection

Under `oat_HASH/` the workflow writes:

| Output | Content |
| --- | --- |
| `plan.json`, `profiles/`, `tasks/` | Full axis mapping, unique task inputs and commands |
| `tasks/.../training/` | Training manifest, completion marker and final checkpoint |
| `tasks/.../inference_predictions.*` | Backend's validation output |
| `tasks/.../predictions.jsonl` | Canonical predictions |
| `tasks/.../evaluation.json` | Canonical evaluator output |
| `tasks/.../inference_receipt.json`, `result.json` | Task/checkpoint/data/source identities and three metrics |
| `results.jsonl` | Exactly one result per unique scan task |
| `selection/selection.json` | Verified seed metrics, all curves, exact seed intervals and selected weights |
| `selection/combined_profile.json` | Combined Selected profile accepted by the training launcher |
| `combined/plan.json`, `combined/tasks/` | Separate five-seed configuration and execution outputs |
| `combined/validation_summary.json` | Five combined-profile results, means, sample SDs and seed macro-F1 interval |

Every result has `task_id`, `profile_id`, `seed`, `input_hash`,
`partition: "validation"`, `split_id`, `validation_sha256`, `synthetic_test`,
absolute `source`, `checkpoint` and `inference_receipt` paths with their hashes,
and exactly `metrics: {"accuracy": ..., "macro_recall": ..., "macro_f1": ...}`
using proportions. Receipts bind those same identities and
`status: "inference_complete"`. Executor receipts also bind the actual command,
canonical import and evaluation files. Pending templates are not valid results.

Externally executed tasks can supply matching completion artifacts and a JSONL
of completed records to the same collector:

```bash
python training/rars_oat.py collect \
  --plan /absolute/path/oat_HASH/plan.json \
  --results /absolute/path/results.jsonl \
  --output /absolute/path/selection
```

Every unique profile × seed must occur exactly once. The collector recomputes
all three metrics with the canonical evaluator and requires agreement within
`1e-12`; missing/duplicate groups, unexpected patients, test identities, changed
artifacts and incomplete seeds are rejected. An identical existing collection
is reusable; conflicting output is rejected. Manual collection selects a profile
but does not execute its five combined-profile tasks.

## Plotting and tests

```bash
python scripts/plot_rars_oat.py \
  --selection /absolute/path/oat_HASH/selection/selection.json \
  --output /absolute/path/oat_curves.pdf
```

Plotting requires Matplotlib and accepts PDF, SVG or PNG output. Every panel
shows mean Accuracy, macro-Recall and macro-F1, the seed-bootstrap macro-F1 band,
and a red dashed line at the Selected weight. Input validation independently
recomputes the means and all seed intervals before rendering. A companion
`OUTPUT.metadata.json` records input/output hashes, plotting/statistics source
hashes, metric definitions and bootstrap settings.

