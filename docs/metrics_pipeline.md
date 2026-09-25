# Metrics pipeline

Structured outputs are parsed by `evaluation/schema.py`. The final two-dimensional matching protocols are in `matching/protocols.py`, with box IoU and Hungarian assignment in `matching/geometry.py`. `evaluation/detection.py` computes lesion-level counts and GT-denominator Mean IoU.

`evaluation/evaluate_sim999_predictions.py` joins prediction and metadata CSV files and writes detection, patient classification, bootstrap, and aggregation summaries. `aggregation/patient.py` contains the deterministic patient-level voting and abstention rules. Statistical post-processing is in `evaluation/statistics.py`, `evaluation/paired_patient_tests.py`, and `statistics/`.
