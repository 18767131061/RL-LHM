# Paper method and comparison settings

## RARS (Methods; Appendix B.10 and C.2)

The final Selected profiles use private pairs HCC–ICC=1.1, HCC–FNH=0.9,
HCC–HH=0.5, ICC–HH=0.7; public pairs HCC–ICC=1.1, HCC–HH=0.6,
ICC–HH=0.8, ICC–BCLM=1.0, ICC–CRLM=1.1, BCLM–CRLM=0.9. Other
off-diagonal pairs share 0.4. An unordered pair's severity is symmetric;
true-class scarcity and completion likelihood separately enter the error cost.
`--risk-profile` accepts an explicit profile through the training launcher.

OAT varies one severity parameter at a time from 0.1 to 1.5 by 0.1,
including the shared other-pair parameter, with other severity weights at 1.0.
The five private/seven public axes give 75/105 axis points and 375/525
seed evaluations. Reference points at 1.0 share a weight vector. These counts
are not distinct complete-profile candidate counts. The [OAT workflow](rars_oat.md) prepares those axes and collects complete
five-seed validation outputs. Shared all-one vectors can reuse one run per
seed, yielding 71/99 distinct profiles and 355/495 runs. Axis-wise selection
does not establish a jointly optimal combined profile; train and evaluate the
combined selection separately. The optional `training/rars_joint_grid.py`
implements a different, explicit Cartesian search interface. Neither tool
contains the original validation observations.

Uniform sets all off-diagonal severity weights to 0.4, retaining the other
Selected settings. RARS-off neutralizes scarcity, severity and
completion-likelihood error multipliers, retaining the localization reward,
correct-class reward, miss penalty, matching, SHAE and DA-CVQ/VGS. The full
RARS × SHAE × DA-CVQ/VGS factorial remains the component comparison.

## Group frequencies and visual cache (C.2, C.4, C.8)

`preprocess/build_verl_real961_dataset.py` exports counts from supplied fitting
groups. Each positive group contributes once to its most frequent cross-phase
GT lesion-record label (plurality); ties choose the lexicographically greatest
internal label. Different groups of a patient count separately; background-only
groups contribute none. Only observed dominant-label keys are emitted.
Validation receives the training dictionary unchanged. Explicit combined refit
recomputes counts. The optional search holds fitting records and counts fixed
across candidates/seeds, while visual caches may differ by SFT seed.

Private vocabulary: HCC, ICC, FNH, HA_Hemangioma, nodule_CYST.
Public vocabulary (`5cls`): HCC, ICC, HA_Hemangioma, BCLM, CRLM.
HH and NC are aliases. Vocabulary size does not specify dictionary key count.
`rewards/verl_liver_reward.py` retains extra caller-supplied keys in the total;
its existing active-class lower bounds are unchanged. The matched GT lesion's
true class supplies the class-specific reward count.

Each seed's SFT visual vectors are cached once at dimension 1152, normalized
as complete vectors, and held fixed during RL; prototypes and reward memory
update separately. The [reference extractor](visual_feature_extraction.md)
uses final 1024-wide visual-block states before merger/normalization, mean
pooling per phase, a shared fixed 1024×384 projection, phase concatenation,
and whole-vector L2 normalization. Cache metadata records the reference extractor identity; the cache contract
checks the model and extraction settings before use.

## Prepared input construction (D.1)

The public converters consume prepared three-phase images and per-phase boxes.
They do not implement the complete CT-volume-to-input procedure. An experiment
must specify series selection, HU/window handling, spatial correspondence,
resampling if any, and eligibility rules for complete three-phase positions.
Do not infer those operations from a converter's ability to read the prepared
files. Record complete patient/group coverage in the evaluation manifest.

The paper describes all eligible corresponding three-phase positions without
GT-guided key-slice selection or an additional top-k limit. Missing phases are
not filled. The evaluator retains patients with undefined diagnoses. Training
background sampling, test-input coverage, and lesion-positive-phase detection
are distinct operations and need separately recorded counts.

## Comparison models and detector stages (C.7; D)

The paper's configuration table distinguishes primary Qwen3-VL-4B-Instruct,
LLaVA-OneVision-1.5-4B, Qwen2.5-VL-3B, InternVL3.5-4B, YOLOv8-X,
YOLOv11-X, RT-DETR-L, and Faster R-CNN R50-FPN. Five training seeds and
final-epoch checkpoints describe training variability. The Qwen3-VL-4B primary SFT uses learning rate 1e-5, 6 epochs, per-device
batch 2, and accumulation 16. Configure the optimizer, schedule, device batch,
accumulation, and data-parallel size for each auxiliary model explicitly.
Effective batch size is device batch × accumulation × data-parallel size,
with units of three-phase groups for VLMs and phase images for detectors.

`training/prepare_sft_runs.py` prepares explicit five-seed SFT configurations;
see [sft_workflow.md](sft_workflow.md). It records the device batch,
accumulation, data-parallel size, precision and checkpoint policy.

`preprocess/build_real961_detector_dataset.py` now preserves separate fitting,
validation and test inputs by default, with distinct YAML paths and patient,
group and image-path overlap checks. An explicit `--refit-trainval` export
combines fitting and validation and sets `val: null`; it must follow a fixed
training schedule without further checkpoint selection. Counts are configurable
for private and public cohorts. The export records image/label hashes and
`phase_mapping.csv`; see [detector_export.md](detector_export.md).

The VLM adapter uses group plurality, fixed-order group ties, patient plurality,
then mean finite completion NLL for patient ties, with fixed-order fallback
only among minimum-NLL classes. Nonfinite NLL does not discard valid votes.

Detector processing has four distinct stages:
1. Phase-image boxes/classes and patient/position/phase association.
2. Model-specific confidence filtering and NMS, if used.
3. Three-phase group diagnosis from the retained lesion outputs.
4. Patient diagnosis from the group outputs.

A detector integration must explicitly supply confidence definition/threshold/
stage; NMS method, class scope, IoU and maximum boxes; group/patient aggregation
and tie rules; and no-box/no-diagnosis handling. Completion NLL is not a detector
score. The evaluator's prediction–GT IoU=0.50 is separate from confidence and
NMS thresholds. Undefined diagnoses retain the full denominator;
absence of a box does not imply NC. The repository supplies canonical matching
and patient-metric modules; model-specific detector-to-patient readout adapters
must be provided by the integrating application.

## Statistical interfaces

The current joint-five-seed endpoint is the mean of within-seed metric
differences. One patient-level permutation exchanges that patient's entire
five-seed vector between the compared models; a bootstrap uses the same patient
multiplicities across both models and all five seeds. Nonlinear metrics are
recomputed before averaging. The executable implementation is
`evaluation/joint_seed_statistics.py`; its default uses 100,000 permutations,
10,000 bootstrap replicates and raw p-values. Optional BH correction has an
explicit comparator-by-metric family within one invocation. See
[joint_seed_statistics.md](joint_seed_statistics.md) for inputs and supported
classification, per-class and detection endpoints.

The earlier `evaluation/paired_patient_tests.py` and
`statistics/apply_r1_1_bh.py` implement seed-specific comparisons and an older
60-test family. Their p/q values describe a different statistical object.
The presentation tools in [table_presentation.md](table_presentation.md)
consume supplied legacy summaries; they do not calculate the current joint
statistic. Across-seed sample SD, patient-bootstrap intervals conditional on
five fixed runs, and adjusted versus raw p-values must remain identified
separately. No inference about actual significance is encoded in these docs.
