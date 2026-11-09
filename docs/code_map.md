# Source file guide

| File | Function |
| --- | --- |
| `data/config.py` | Read stage JSON configurations with unique keys and finite values |
| `data/manifest.py` | Read three-phase groups, resolve image paths, validate patient/UID relationships, derive fitting-group counts |
| `data/dicom.py` | Read a DICOM series with native geometry; apply window clipping and optional uint8 normalization |
| `models/huggingface.py` | Load VLM/processor; build image conversations; calculate SFT loss, generate completions, score tokens |
| `models/visual_features.py` | Extract frozen pre-merger Qwen3-VL patches; create/load the orthonormal projection |
| `training/run.py` | Select DeepSpeed or VeRL; construct the DeepSpeed SFT/RL workflow |
| `training/engine.py` | Run DeepSpeed input sampling, grouped rollouts, policy updates and curriculum updates |
| `training/checkpoint.py` | Save/restore DeepSpeed epoch checkpoints; provide shared state serialization and run identities |
| `training/verl_run.py` | Launch the VeRL RL driver or check its configuration |
| `training/verl_backend/config.py` | Validate the pinned VeRL source interface and translate the shared JSON recipe |
| `training/verl_backend/dataset.py` | Convert manifest inputs to multimodal VeRL records with occurrence IDs and request seeds |
| `training/verl_backend/rollout.py` | Generate through VeRL/vLLM, preserve response tokens and defer reward calculation |
| `training/verl_backend/driver.py` | Connect VeRL actor/reference scoring, shared rewards/advantages, actor update and curriculum update |
| `training/verl_backend/state.py` | Own UID baselines, DA-CVQ/VGS and random streams on the Ray driver |
| `training/verl_backend/trainer.py` | Construct official VeRL workers and save/restore epoch checkpoints |
| `inference/visual_features.py` | Produce a manifest with fixed visual vectors |
| `inference/run.py` | Produce keyed completions and completion NLL for every input group |
| `rewards/schema.py` | Parse lesion/background records under the phase, label, and box contract |
| `rewards/rars.py` | Compute scarcity, severity, and likelihood-based diagnostic error costs |
| `rewards/full_reward.py` | Combine independent IoU/GIoU reward terms, raw RARS costs, missing-match penalty and clipping |
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

Package `__init__.py` files define public imports or package descriptions. `requirements.txt` specifies core dependencies and `requirements-training.txt` adds DeepSpeed. `scripts/setup_verl.sh` creates a separate pinned VeRL environment from its official `uv.lock` and FSDP/vLLM extras, then installs the project supplements in `requirements-verl.txt`. `requirements-preprocessing.txt` supplies optional SimpleITK DICOM support. `CITATION.cff` provides citation metadata, and `LICENSE` provides the MIT license. `README.md` describes the workflow; `docs/configuration.md` defines inputs/settings; `docs/algorithms.md` defines mathematical interfaces. `.gitignore` excludes local Python caches and environments.

`training/distributed.py` owns DeepSpeed rank assignment, global reductions, ZeRO-2 CPU offload and accumulation boundaries. `training/verl_backend/upstream_api.json` records the required VeRL source hashes, and `agent_loop.yaml` registers the single-turn rollout adapter. `configs/` contains four-A800 DeepSpeed SFT/RL, VeRL RL, and feature/inference configurations. `tests/` covers algorithm boundaries, four-process equivalence, model probabilities and checkpoint continuation.

`data/lengths.py` and `data/check_lengths.py` check chat-template target budgets. `data/provenance.py` supplies artifact identities. `evaluation/analyze.py` performs center stratification, fixed-class patient bootstrap and paired comparisons.
