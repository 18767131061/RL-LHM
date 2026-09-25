# Method-to-code mapping

| Paper component | Public implementation | Notes |
| --- | --- | --- |
| Model loading | `models/loading.py` | Local Hugging Face loading with explicit device and dtype settings. |
| Data manifest and patient split | `data/manifest.py`, `preprocess/center_split.py` | Validates relative asset references and split isolation. |
| Structured generation and completion likelihood | `inference/generation.py`, `rl/completion_likelihood.py` | Used as the inference/ranking signal described in the paper. |
| SHAE and DA-CVQ/VGS | `rl/core.py`, `rl/verl_lhm_extensions.py` | Core advantage and sampler state logic. |
| Reward orchestration and RARS | `rewards/full_reward.py`, `rewards/rars.py`, `rewards/verl_liver_reward.py` | Final 2D reward path used by the launcher. |
| RL training entry point | `training/run_verl_lhm_variant.py`, `training/verl_lhm_launcher.py` | Requires a compatible external VeRL/vLLM environment. |
| 2D matching and IoU | `matching/geometry.py`, `matching/protocols.py`, `evaluation/detection.py` | Class-agnostic, same-class-priority, and geometry-first class-aware protocols. |
| Patient-level classification | `evaluation/classification.py`, `evaluation/loco.py`, `aggregation/patient.py` | Includes four-class LOCO and abstention/coverage contracts. |
| Main evaluation command | `evaluation/evaluate_sim999_predictions.py` | Produces detection, classification, aggregation, and bootstrap summaries. |
| Statistical tests | `evaluation/statistics.py`, `evaluation/paired_patient_tests.py`, `statistics/` | Post-hoc paired and multiple-testing procedures. |

The public package contains no 3D evaluation path. It also excludes clinical data, patient-level exports, model weights, server schedulers, and internal audit artifacts.
