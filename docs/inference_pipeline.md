# Inference pipeline

`inference/infer_qwen_real961.py` loads a local Qwen3-VL checkpoint, accepts
prepared three-phase input groups, generates completions, and exports raw
text and average generated-token NLL in keyed CSV rows.
`inference/infer_hf_vlm_revision.py` provides a generic Hugging Face VLM path;
`inference/generation.py` contains deterministic generation and NLL helpers.

Supply `--model-dir`, `--json-path`, `--data-root`, and `--output` explicitly.
These commands require compatible optional model dependencies and local
weights. They do not construct a CT-position selection procedure.

For evaluation, convert each inference row to the `group_id`, `raw`, and
`completion_nll` record defined in `metrics_pipeline.md`. Join the identifier
to an explicit evaluation manifest with separate phase-wise lesion annotations.
Preserve all manifest keys and actual invalid completions. Do not infer missing
phase annotations or replace missing output records with generated empty text.

The older `evaluate_sim999_predictions.py`, its real961 wrapper, and
`postprocess_verl_checkpoint.py` use the legacy metadata/evaluation route.
Their completion markers do not establish equivalence to
`evaluate_paper_protocol.py`. The legacy vote filter, detection scope, and
class-aware matching must be reconciled before comparing their results with
the three canonical protocols.

Completion NLL is a sequence-likelihood proxy for tie-breaking. It is not
per-box confidence or calibrated diagnostic probability.
