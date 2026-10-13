# Optional joint-grid RARS search tool

`training/rars_joint_grid.py` implements a finite Cartesian search over four
private or six public unordered pair weights. Each pair shares its weight in
both directions; unlisted off-diagonal pairs stay at 0.4 in this tool. A formal
tool configuration supplies 15 levels from 0.1 through 1.5 and five seeds. The
candidate/seed coverage required by `freeze` is computed from that configuration.
This is an optional executable search interface, not the confirmed procedure
that generated the paper's Selected profiles.

The paper separates the final Selected profiles, OAT sensitivity, and trained
Uniform/RARS-off controls. OAT also scans the shared other-pair weight: five
private axes and seven public axes, 15 values per axis, with other severity
weights at 1.0. Its 75/105 axis-counted points and 375/525 seed evaluations are
not the joint tool's candidate count. All axes pass through the same all-1.0
profile. The Selected values coincide with observed OAT peaks; that observation
does not specify candidate generation or selection order.

Copy `configs/rars_grid.example.json` outside the repository and replace its
paths. The five `seed_train_files` contain the same training patient/group
population and the visual cache extracted from the corresponding seed's SFT
checkpoint. Each persistent UID must retain the same patient, images, targets and
other non-feature fields across all five caches; visual feature values and cache
identifiers may differ by seed, but their dimensions must agree. The five `seed_model_paths` point to those SFT checkpoints. The
feature layer, pooling and phase-fusion recipe must be supplied from the actual
extraction implementation; this runner does not invent or perform extraction.

Set `preparation_audit` to the converter's `audit.json` to bind selection to its
actual fitting population and counts. The runner verifies the audit's output
hashes and matches every fitting payload against every seed cache, excluding
only feature/cache fields. Formal prepared searches require the model-bound
visual-cache contract from `scripts/extract_qwen_visual_latents.py`; cache-only
validation verifies its source model and extraction settings. The optional
audit field preserves compatibility for existing manually prepared inputs;
without it the runner still checks cross-seed payload identity, but cannot
claim the converter-to-search binding.

The validation file uses the paper-protocol schema in `metrics_pipeline.md`,
with `split: "validation"` and `split_id: "P1"` (private) or `"U1"` (public).
The reserved-test identity manifest contains only `split: "test"` and a
`patients` list of `{"patient_id": "..."}` objects; no test labels or scores are
read. The runner rejects cross-split patients, training/validation group overlap,
duplicate keys, wrong labels, empty required splits and changed input hashes.

## Tasks, sharding and recovery

```bash
python training/rars_joint_grid.py --config /path/to/search.json \
  --database /path/to/search.sqlite tasks --shard 0 --shards 64
python training/rars_joint_grid.py --config /path/to/search.json \
  --database /path/to/search.sqlite prepare --task 0
python training/rars_joint_grid.py --config /path/to/search.json \
  --database /path/to/search.sqlite run --task 0
```

Task IDs are mixed-radix candidate indices times five plus the seed index. Tasks
are yielded one at a time; the full task universe is never expanded in memory.
Disjoint modulo shards can be assigned to workers; use a shared SQLite database
on a filesystem with supported locking. `prepare` writes the exact task,
unordered pair profile and training command. `run` calls the real
`run_verl_lhm_variant.py --risk-profile ...` entry point. The profile path is
propagated into Ray reward workers; the callback checks the cohort and applies
the symmetric pair weight. Each task has an independent model, optimizer,
SHAE/curriculum state and output directory.

Only completed candidate/seed results are skipped on recovery. A changed result
for the same task fails; an exact replay is idempotent. An interrupted training
directory is retained with an `incomplete` suffix before the task is restarted
from its SFT checkpoint. Exact mid-training checkpoint recovery is deliberately
unsupported because the external EMA state and actor checkpoint are not atomic.
Do not dispatch the same unfinished task concurrently to multiple workers.

## Validation scoring

After a real task completes, generate predictions on its validation manifest
from the final-epoch checkpoint. Supply all manifest groups, including actual
invalid/no-vote completions. The inference adapter must write a JSON receipt:

```json
{
  "task_digest": "SHA256 of canonical task.json",
  "validation_sha256": "SHA256 of the validation manifest",
  "predictions_sha256": "SHA256 of the prediction JSONL",
  "partition": "validation",
  "checkpoint": "/path/to/this/task/training/global_step_N"
}
```

`digest(json.loads(task_path.read_text()))` from `training.rars_joint_grid`
computes the task digest. The receipt binds supplied inference artifacts; it is
not an independent proof that an external model generated them. The runner uses
the actual `evaluation.evaluate_paper_protocol.evaluate` implementation to
compute macro-F1, rather than accepting a caller's scalar score.

```bash
python training/rars_joint_grid.py --config /path/to/search.json \
  --database /path/to/search.sqlite score --task 0 \
  --predictions /path/to/predictions.jsonl --receipt /path/to/receipt.json
python training/rars_joint_grid.py --config /path/to/search.json \
  --database /path/to/search.sqlite summary
python training/rars_joint_grid.py --config /path/to/search.json \
  --database /path/to/search.sqlite freeze --output /path/to/selected.json
```

Scoring requires a matching completed training manifest, seed, train-file hash,
SFT model path, risk-profile hash, final checkpoint and validation identity. Dry
runs, smoke runs, wrong/duplicate/missing predictions, non-finite scores and
conflicting duplicate task results fail. Summary rechecks stored evidence hashes.
Partial searches expose only `best_among_completed_candidates` and set
`global_best_in_grid` to null. A formal exhaustive freeze requires every
candidate and all five seeds. Exact ties select the lowest candidate index.
`freeze` writes an audit record and `selected.profile.json`; pass the latter to
`--risk-profile` when training with the frozen weights. Keep both records with
the experiment, and do not use the test endpoint to revise them. Reward workers
cache validated profiles by path for the process lifetime (up to four paths);
keep each supplied profile immutable during a run. Start new workers for a
changed profile, using a new file path.

## LOCO and synthetic tests

LOCO is a separate configuration and database. It requires `protocol: "loco"`,
its own `split_id`, `classes`, explicit unordered `loco_pairs`, `training_center`
and `held_out_center`. Training rows' `extra_info.center_id` and validation
patients' `center_id` must equal the training center; reserved-test patients'
centers must equal the held-out center. The axes must be within the active class
rule. Main-grid dimension counts, validation sizes and selected weights are not
inferred for LOCO. The reward callback uses the same explicit `active_classes`
for completion parsing, ground-truth labels and the active-class count lower bounds.
A class omitted from the profile is not added to `class_counts`; an explicitly
supplied extra metadata key remains in the total under the documented count rule.
The pure-Python `RewardConfig` helper accepts the same profile through its explicit
`risk_profile` argument. These class rules do not constrain model token generation;
the inference prompt and evaluation manifest define the diagnostic output interface.
The [converter's LOCO mode](training_pipeline.md#prepare-a-source-center-loco-split)
preserves source-center metadata and generates the source fitting rows,
validation manifest, validation inference source and audit together. Its
explicit split fraction and seed stratify patient reference diagnoses, never
group-level lesion majorities. Set `validation_manifest` and
`test_identity_manifest` to the exact files bound by that audit. All candidates
and seeds then use that fixed split and fitting-count dictionary. Source
validation `macro_classes` must equal `classes` (or be omitted, which has the
same meaning). Thus a five-class WMU source optimizes five-class validation
macro-F1, while target ZJU evaluation may separately use the four shared
`macro_classes` and retain all five output classes. A ZJU source uses four
classes for both source validation and model output. Do not use a target
four-class macro list as a five-class source's selection objective.

The current formal reference runner requires the same explicit 15 levels and
five seeds for LOCO; `loco_pairs` supplies the direction-specific axes. This
constraint defines a new executable configuration interface, not a recovery of
the historical direction-specific candidate domain. Historical LOCO internal
split parameters are not bundled.

## Final fitting scope

After every candidate and all five seeds have completed, export the selected
candidate's five existing train-only checkpoint references:

```bash
python training/rars_joint_grid.py --config /path/to/search.json \
  --database /path/to/search.sqlite finalize --output /path/to/final_models.json
```

The output has `fit_scope: "selection_train_only"` and
`status: "selected_models_ready"`; checkpoint directories must still exist.
This retains the selected models and does not start another training run.
The reported primary private protocol remains train-only (654 patients).

For an explicitly chosen LOCO source train-plus-validation refit, prepare the
converter's extra combined file at the outset with `--refit-train-validation`.
Create a separate JSON config with `fit_scope: "train_plus_validation"` and
`seed_train_files` and `seed_model_paths` mappings for all five seeds, using the
same key structure as the search config. The feature caches must be extracted
from the combined file with its recomputed counts and the explicitly chosen,
seed-matched SFT checkpoints. These choices are inputs; no historical SFT
refit policy is inferred from an existing checkpoint path.

```bash
python training/rars_joint_grid.py --config /path/to/search.json \
  --database /path/to/search.sqlite finalize --output /path/to/refit_plan.json \
  --refit-config /path/to/refit.json
```

The runner checks the unchanged preparation audit, the exact source
fitting/validation group and patient union, source-center isolation, every
cache payload and the formal model-bound feature contracts. It then writes
five separate `final_refit/seed_*/task.json`, selected profiles and
`train_command.json` argument arrays for `training/run_verl_lhm_variant.py`.
Execute those commands in the prepared training environment to train refit
models. Finalization itself reports `refit_commands_prepared_not_trained`,
`training_executed: false`, and an empty model list; it does not claim refit
results. Existing refit tasks with a different cache/model identity are
rejected. Synthetic test runs use explicitly synthetic status names.

## Synthetic test scope

For tests only, `synthetic_test: true` permits a smaller subset of the same levels.
It prohibits launching model training and a formal freeze. The integration test
uses a 2^4-by-five grid and synthetic training/inference receipts; scoring,
coverage checks, sharding, resume and five-seed selection execute normally.
These tests validate the optional tool interface and do not generate the paper's experimental results.
