# Full joint RARS selection

`training/rars_joint_grid.py` implements the finite Cartesian grid independently
of single-axis sensitivity plots. Main private selection uses HCC–ICC, HCC–FNH,
HCC–HH and ICC–HH. Main public selection uses HCC–ICC, HCC–HH, ICC–HH, ICC–BCLM,
ICC–CRLM and BCLM–CRLM. HH is `HA_Hemangioma` in the schema. Each pair shares one
weight in both directions; other off-diagonal pairs stay at 0.4.

Every axis has the 15 values 0.1 through 1.5. There are 50,625 private and
11,390,625 public configurations. Each uses seeds 42, 2026, 3407, 1234 and 5678,
for 253,125 and 56,953,125 seed trainings (57,206,250 total). The selection score
is the arithmetic mean of five patient-level validation macro-F1 scores. This is
an argmax within the finite specified grid; it makes no continuous/global
population-optimality claim and supplies no historical runtime or total wall time.

Copy `configs/rars_grid.example.json` outside the repository and replace its
paths. The five `seed_train_files` contain the same training patient/group
population and the visual cache extracted from the corresponding seed's SFT
checkpoint. The five `seed_model_paths` point to those SFT checkpoints. The
feature layer, pooling and phase-fusion recipe must be supplied from the actual
extraction implementation; this runner does not invent or perform extraction.

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
the experiment, and do not use the test endpoint to revise them.

## LOCO and synthetic tests

LOCO is a separate configuration and database. It requires `protocol: "loco"`,
its own `split_id`, `classes`, explicit unordered `loco_pairs`, `training_center`
and `held_out_center`. Training rows' `extra_info.center_id` and validation
patients' `center_id` must equal the training center; reserved-test patients'
centers must equal the held-out center. The axes must be within the active class
rule. Main-grid dimension counts, validation sizes and selected weights are not
inferred for LOCO. The actual historical LOCO internal split is not bundled.

For tests only, `synthetic_test: true` permits a smaller subset of the same levels.
It prohibits launching model training and a formal freeze. The integration test
uses a 2^4-by-five grid and synthetic training/inference receipts; scoring,
coverage checks, sharding, resume and five-seed selection execute normally.
No current script, dry run or CPU test is evidence that the reported historical
full grid was executed. Historical search records and training versions remain
separate evidence requirements.
