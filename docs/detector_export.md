# Detector dataset export

`preprocess/build_real961_detector_dataset.py` exports prepared three-phase
annotations to phase images and YOLO-format labels. Despite the historical
filename, it supports private, public five-class, and shared three-class data.
It does not select CT slices, train a detector, or define detector-to-patient
aggregation.

## Preserve fitting, validation and test partitions

The existing `--root` layout remains supported:

```bash
python preprocess/build_real961_detector_dataset.py --root /path/to/experiment
```

The command reads `ROOT/data/train.jsonl`, `validation.jsonl`, and `test.jsonl`
and writes `ROOT/baselines_revision/detectors/data`. The output directory must
be new or empty, preventing old merged exports from contaminating a new run.
Use `--data-root` and `--output-dir` to select other locations. All three source
partitions must be nonempty. The YAML uses separate paths:

```yaml
train: images/train
val: images/validation
test: images/test
```

Patient IDs and input-group identities must be disjoint across the supplied
partitions. When both `patient_id` and `statistical_patient_id` are supplied,
their mapping must be one-to-one; both identity fields are checked. A source
image path also cannot occur in different partitions. Optional metadata `split`
must agree with the source file. Counts are not fixed to a private cohort.
Assertions are available when the intended counts are known:

```bash
python preprocess/build_real961_detector_dataset.py --root /path/to/experiment \
  --data-mode 5cls \
  --expected-train-patients 330 \
  --expected-validation-patients 83 \
  --expected-test-patients 104
```

`--data-mode private` is the default. `5cls` means HCC, ICC, HA_Hemangioma,
BCLM, CRLM; `3cls` means HCC, ICC, HA_Hemangioma. `--classes` can instead
specify an ordered comma-separated vocabulary using canonical label aliases.
The order determines integer detector labels and is recorded in the export.

## Explicit refitting

Only `--refit-trainval` combines the supplied train and validation partitions:

```bash
python preprocess/build_real961_detector_dataset.py --root /path/to/experiment \
  --output-dir /path/to/new-refit-export --refit-trainval
```

The original partitions must still be disjoint. The export uses
`train: images/trainval`, `val: null`, and a separate `test: images/test`.
No validation set is supplied, and the original source partition of every image
remains in the mapping. Configure the downstream trainer for a fixed training
schedule without validation or checkpoint selection. Some detector frameworks
require a validation path even when validation is disabled; this exporter does
not provide a framework-specific training adapter. Do not substitute the test
set or the fitting union as a validation set to satisfy such a requirement.

## Annotation contract

Each source row contains `metadata` and `messages`. Metadata requires nonempty
`logical_key`, `statistical_patient_id`, a nonnegative integer `slice_index`,
and a `positive` flag (0/1). The user message contains exactly three image
segments in a/p/v order. The assistant message supplies JSON with a `detections`
list, either as a string or a text segment.

Ground truth passes the same `evaluation.schema.parse_completion` used by the
canonical evaluation. Boxes must be finite normalized xyxy coordinates in
[0,1000] with positive extent. Unknown labels, reversed/out-of-range boxes,
duplicate background sentinels and lesion/background conflicts within a phase
are rejected. Coordinates are never clipped, sorted or silently repaired.

For this fully supervised detector export, each of a/p/v must explicitly have
lesion records or a `no-lesion` sentinel `[-1,-1,-1,-1]`. A missing phase annotation
is not converted to an empty detector label. This is stricter than the allowed
omitted phases in model completions, because an unknown training target is not
known background. An arterial lesion with explicit p/v background is valid;
`positive` describes whether the entire group contains any lesion, not whether
every phase is positive. Background phases produce empty label files.

All source partitions and targets are validated before export files are written.

## Outputs and provenance

- `dataset.yaml`: ordered classes and explicit fitting/validation/test paths.
- `phase_mapping.csv`: original `source_split`, exported split, patient/group/
  slice/phase identity, lesion count, normalized-label class list, image and label
  paths, and SHA-256 hashes of source image bytes and emitted label files.
- `dataset_audit.json`: format `detector-export-v2`; original patient IDs/counts,
  zero observed source-split overlap, refit choice, per-export-split counts of
  patients/groups/phase images/positive phases/background phases/lesions, source
  JSONL and exporter hashes, and output hashes.

Images are symlinks. Check `source_image_sha256` before downstream fitting if
source files may have changed. The audit establishes export provenance and
partition checks; it does not attest to future training, model selection, or
historical paper experiments. The old unconditional assertions that test was
never used for selection and validation was disabled are no longer emitted.

## Historical evaluation entry points

`evaluation/evaluate_sim999_predictions.py` and its `evaluate_real961_predictions.py`
wrapper now require `--allow-legacy-protocol` and print a warning. This allows
explicit reproduction of historical GT-positive voting/shared-box/class-aware
semantics without confusing them with the current canonical protocols. The old
metric formulas are unchanged. Use `evaluation/evaluate_paper_protocol.py` for
the current protocol; historical orchestration scripts must explicitly opt in
if their intended purpose is to rerun the old behavior.
