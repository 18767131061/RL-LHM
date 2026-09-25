# Implementation consistency matrix

| Concept | Code | CPU contract |
| --- | --- | --- |
| SHAE and DA-CVQ/VGS | `rl/core.py`, `rl/verl_lhm_extensions.py` | `tests/test_rl_core.py` |
| RARS | `rewards/rars.py` | `tests/test_rars_weights.py` |
| Full reward | `rewards/full_reward.py` | `tests/test_full_reward.py` |
| 2D matching and Mean IoU | `matching/`, `evaluation/detection.py` | `tests/test_iou.py`, `tests/test_detection_metrics.py` |
| Patient classification and aggregation | `evaluation/classification.py`, `aggregation/patient.py` | `tests/test_classification_metrics.py`, `tests/test_patient_aggregation.py` |
| Data isolation | `data/manifest.py`, `preprocess/center_split.py` | `tests/test_data_manifest.py`, `tests/test_preprocess_and_inference.py` |

The table describes the final public 2D scope. GPU training and model inference remain environment-dependent.
