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
are not distinct complete-profile candidate counts. Selected values coincide
with observed OAT peaks. The optional `training/rars_joint_grid.py` implements
a different, explicit Cartesian search interface.

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

## DICOM reading and prepared input construction (D.1)

The author-supplied volume-reading utility uses SimpleITK ImageSeriesReader
and GDCM discovery. It records spacing/origin/direction, transposes D×H×W
arrays to H×W×D, and computes z positions as origin_z + k × spacing_z ×
direction_zz while retaining native spatial sampling. This position calculation
is not inter-phase registration.

A complete explicit window-center/width pair takes precedence. If either is
missing, the code reads both values from the first DICOM (first value when
multivalued); missing metadata falls back to the volume intensity range.
Windowing clips the intensities returned by SimpleITK. Normalization maps the
window to 0–255 and uint8. The example enables normalization without a fixed
window. The manual RescaleSlope/RescaleIntercept multiplication is commented
out; the code does not apply a second manual HU conversion.

The utility reads complete series and demonstrates contours on nonzero-mask
slices. Its calculated maximum-area slice index is not used for final input
selection. It does not implement registration, resampling, interpolation,
three-phase matching, box generation, or the final manifest. The private
path-bearing original and mask script are not redistributed.

The prepared interface retains all corresponding positions with complete
three-phase inputs, without GT-guided/manual key-slice selection or extra top-k.
Missing phases are not filled. Patient evaluation retains all 190/104 patients,
including undefined diagnoses. Training background sampling, test-input
coverage, and lesion-positive phase-slice detection are distinct operations.
The conversion entry points consume already prepared images and annotations;
they do not reconstruct upstream volume-to-input processing.

## Comparison models and detector stages (C.7; D)

The paper's configuration table distinguishes primary Qwen3-VL-4B-Instruct,
LLaVA-OneVision-1.5-4B, Qwen2.5-VL-3B, InternVL3.5-4B, YOLOv8-X,
YOLOv11-X, RT-DETR-L, and Faster R-CNN R50-FPN. Five training seeds and
final-epoch checkpoints describe training variability. The Qwen3-VL-4B primary SFT uses learning rate 1e-5, 6 epochs, per-device
batch 2, and accumulation 16. Configure the optimizer, schedule, device batch,
accumulation, and data-parallel size for each auxiliary model explicitly.
Effective batch size is device batch × accumulation × data-parallel size,
with units of three-phase groups for VLMs and phase images for detectors.

`preprocess/build_real961_detector_dataset.py` exports the private 771-patient
development union and a separate 190-patient test set. Both YAML train and val
point to that development export. Prepare distinct fitting and validation
inputs when required by the intended training protocol. The preparation
metadata describes the exported dataset. `phase_mapping.csv` records the supplied
patient/position/phase relationship for downstream adapters.

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

## Classification statistics and manuscript table locations

The primary within-cohort comparisons are Full RL-LHM minus SFT and Full
RL-LHM minus vanilla GRPO, using the same test patients and training seed.
The main private/public classification tables (`tab:unified_private` and
`tab:unified_public`) show model means ± sample SD in Panel A and individual
seed/comparator Δ [patient-bootstrap 95% CI] and BH q values for Accuracy,
macro-Recall and macro-F1 in Panel B. These cover the original family of
2 cohorts × 5 seeds × 2 comparators × 3 endpoints = 60 comparisons. BH q values
all exceed 0.05. Macro-Precision and auxiliary comparisons remain outside this
paired family.

The appendix tables (`tab:app_paired` and `tab:app_paired_public`) supply the
complementary raw p values and Accuracy's discordant b/c counts, without
repeating Panel B. [Table presentation](table_presentation.md) describes the
record identities and preservation rules. Cross-seed SD and same-seed
patient-bootstrap CIs describe different sources of uncertainty. The LOCO
Wilson/bootstrap intervals are model-specific intervals and do not belong to
the paired 60-comparison family.
