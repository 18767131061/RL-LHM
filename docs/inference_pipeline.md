# Inference pipeline

`inference/infer_qwen_real961.py` is the paper-facing local Qwen3-VL entry point. `inference/infer_hf_vlm_revision.py` provides the generic Hugging Face VLM path, while `inference/generation.py` contains deterministic generation and completion-NLL helpers.

Provide a model directory, input CSV, image root, and output CSV through command-line arguments. The model runtime is optional and requires a compatible `torch`, `transformers`, Pillow, and vision-utility installation.
