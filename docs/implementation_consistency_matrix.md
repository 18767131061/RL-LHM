# Implementation consistency matrix

| Concept | Code | CPU contract |
| --- | --- | --- |
| Fixed reference visual cache | `scripts/extract_qwen_visual_latents.py`, `models/assets/qwen3vl_projection_1024x384_v1.npy` | `tests/test_visual_feature_cache.py` (PyTorch; real small HF vision forward with Transformers 4.57.1) |
| SHAE and DA-CVQ/VGS | `rl/core.py`, `rl/verl_lhm_extensions.py` | `tests/test_rl_core.py` |
| Persistent sample identity and per-visit rollout groups | `rl/patch_verl_sample_uid.py`, `rl/verl_lhm_extensions.py`, `training/verl_lhm_launcher.py` | `tests/test_verl_sample_uid.py` (PyTorch; exact-reference integration when supplied) |
| RARS | `rewards/rars.py` | `tests/test_rars_weights.py` |
| Full reward and strict parsing | `rewards/full_reward.py`, `rewards/verl_liver_reward.py`, `evaluation/schema.py` | `tests/test_full_reward.py`, `tests/test_reward_schema_contract.py` |
| 2D matching and Mean IoU | `matching/`, `evaluation/detection.py` | `tests/test_iou.py`, `tests/test_detection_metrics.py` |
| Patient classification and aggregation | `evaluation/classification.py`, `aggregation/patient.py` | `tests/test_classification_metrics.py`, `tests/test_patient_aggregation.py` |
| Data isolation | `data/manifest.py`, `preprocess/center_split.py` | `tests/test_data_manifest.py`, `tests/test_preprocess_and_inference.py` |
| Paper-protocol evaluation CLI | `evaluation/evaluate_paper_protocol.py` | `tests/test_paper_protocol_cli.py` |
| Separate train/validation preparation and eight configurations | `preprocess/build_verl_real961_dataset.py`, `training/run_verl_lhm_variant.py` | `tests/test_training_data_cli.py` |
| Reward-loop source patch | `rl/patch_verl_agent_loop_logprobs.py` | `tests/test_agent_logprob_patch.py` |
| Legacy 60-comparison display, eight factorial settings plus Uniform, full-coverage aggregation view | `evaluation/table_presentation.py` | `tests/test_table_presentation.py` (artificial aggregate fixtures; original fields retained) |
| Configuration helpers without optional Torch | `rl/__init__.py`, `rl/policy_contract.py` | `tests/test_optional_rl_import.py` (same exported objects when Torch is installed) |
| Five-seed SFT preparation and current evaluation commands | `training/prepare_sft_runs.py`, `training/framework_sft_common.py` | `tests/test_sft_workflow.py` (configuration and stub launcher, no training) |
| Split-preserving detector export | `preprocess/build_real961_detector_dataset.py` | `tests/test_detector_export_contract.py` |
| Explicit inference/evaluation mapping, raw-logit NLL and sampling resume | `evaluation/import_inference_predictions.py`, `inference/decoding.py`, `inference/resume.py` | `tests/test_inference_protocol_bridge.py`, `tests/test_inference_resume.py` (stub models) |
| Joint five-seed paired statistics | `evaluation/joint_seed_statistics.py`, `aggregation/summarize_seed_results.py` | `tests/test_joint_seed_statistics.py` (exact small-population enumeration and nonlinear recomputation) |
| Response logprob coverage and runtime preflight | `rl/patch_verl_agent_loop_logprobs.py`, `training/rl_preflight_check.py` | `tests/test_agent_logprob_patch.py`, `tests/test_runtime_preflight.py` |
| One-at-a-time severity experiments | `training/rars_oat.py`, `training/rars_oat_execution.py`, `training/rars_oat_infer.py` | `tests/test_rars_oat.py`, `tests/test_rars_oat_execution.py`, `tests/test_rars_oat_infer.py` |
| OAT seed-bootstrap curves | `visualization/rars_oat.py`, `scripts/plot_rars_oat.py` | `tests/test_rars_oat_execution.py` (independent enumeration of all 3,125 samples) |
| Canonical paper tables and confusion panels | `evaluation/export_paper_tables.py`, `evaluation/plot_confusion_matrices.py` | `tests/test_paper_table_exports.py` (patient counts, raw-p direction, protocol labels and figure exports) |
| Auxiliary classification seed rounding | `aggregation/summarize_seed_results.py` | `tests/test_seed_summary_rounding.py` (hand-computed means/SD and unchanged detection counts) |
| Cross-strategy patient aggregation | `evaluation/compare_aggregation_strategies.py`, `evaluation/evaluate_paper_protocol.py`, `evaluation/joint_seed_statistics.py` | `tests/test_aggregation_comparisons.py` (same-source identities, all-patient denominators and explicit reference arms) |

Install the core and optional dependencies for the corresponding test groups.
Training inputs and runtime setup are described in `training_pipeline.md`.
