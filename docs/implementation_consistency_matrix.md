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
| Reward-loop source patch | `rl/patch_verl_agent_loop_logprobs.py` | `tests/test_training_data_cli.py` (synthetic source only) |
| Existing 60-comparison display, eight factorial settings plus Uniform, full-coverage aggregation view | `evaluation/table_presentation.py` | `tests/test_table_presentation.py` (artificial aggregate fixtures; original fields retained) |
| Configuration helpers without optional Torch | `rl/__init__.py`, `rl/policy_contract.py` | `tests/test_optional_rl_import.py` (same exported objects when Torch is installed) |

The table describes the final public 2D scope. GPU training and model inference remain environment-dependent.
