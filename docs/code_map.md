# Source file guide

| File | Function |
| --- | --- |
| `data/config.py` | Read stage JSON configurations with unique keys and finite values |
| `data/manifest.py` | Read three-phase groups, resolve image paths, validate patient/UID relationships, derive fitting-group counts |
| `models/huggingface.py` | Load VLM/processor; build image conversations; calculate SFT loss, generate completions, score tokens |
| `models/visual_features.py` | Extract frozen pre-merger Qwen3-VL patches; create/load the orthonormal projection |
| `training/run.py` | Read configuration, construct policies and optimizer, launch SFT or RL |
| `training/engine.py` | Sample inputs, construct rollout groups, optimize losses, update curriculum memory |
| `training/checkpoint.py` | Save/restore optimizer, random streams, baseline, and codebook at epoch boundaries |
| `inference/visual_features.py` | Produce a manifest with fixed visual vectors |
| `inference/run.py` | Produce keyed completions and completion NLL for every input group |
| `rewards/schema.py` | Parse lesion/background records under the phase, label, and box contract |
| `rewards/rars.py` | Compute scarcity, severity, and likelihood-based diagnostic error costs |
| `rewards/full_reward.py` | Combine localization, diagnostic, and missing-match terms; clip scalar rewards |
| `matching/geometry.py` | Compute IoU/GIoU and maximum-sum Hungarian assignments |
| `matching/protocols.py` | Apply CA, geometry-first class-aware, and same-class-first assignment rules |
| `rl/core.py` | Maintain UID baselines; compute SHAE or GRPO advantages |
| `rl/completion_likelihood.py` | Compute completion NLL and geometric-mean likelihood |
| `rl/policy_contract.py` | Compute token-ratio loss, dual clipping, and reference-policy KL |
| `rl/visual.py` | Pool patches, project, concatenate phases, normalize the latent |
| `rl/curriculum.py` | Maintain DA-CVQ prototypes/reward EMAs; construct VGS probabilities and draws |
| `rl/workflow.py` | Connect group mean clipped rewards to visual vectors and codebook updates |
| `aggregation/patient.py` | Compute group votes and patient diagnoses with deterministic ties |
| `evaluation/classification.py` | Compute patient accuracy and macro/per-class precision, recall, F1 |
| `evaluation/detection.py` | Count lesion matches; calculate precision, recall, F1, overlap totals |
| `evaluation/evaluate.py` | Align UIDs, retain patient denominators, run protocols, write metrics |

Package `__init__.py` files define public imports or package descriptions. `requirements.txt` specifies dependencies, `CITATION.cff` provides citation metadata, and `LICENSE` provides the MIT license. `README.md` describes the workflow; `docs/configuration.md` defines inputs/settings; `docs/algorithms.md` defines mathematical interfaces. `.gitignore` excludes local Python caches and environments.
