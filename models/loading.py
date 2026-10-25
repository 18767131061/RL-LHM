from __future__ import annotations

from pathlib import Path
from typing import Any


def load_hf_vision_language_model(
    model_dir: str | Path,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    local_files_only: bool = True,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a local Hugging Face VLM and processor with explicit trust controls."""
    try:
        import torch
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = None
    except ImportError as error:
        raise RuntimeError(
            "Model loading requires the optional torch and transformers dependencies"
        ) from error

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    loaders = [
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoModelForCausalLM,
        AutoModel,
    ]
    errors: list[str] = []
    for loader in dict.fromkeys(item for item in loaders if item is not None):
        for dtype_key in ("dtype", "torch_dtype"):
            try:
                model = loader.from_pretrained(
                    str(model_path),
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    **{dtype_key: dtype_map[dtype]},
                )
                return model.to(device).eval(), processor
            except Exception as error:  # loader compatibility is model-specific
                errors.append(f"{loader.__name__}/{dtype_key}: {error}")
    raise RuntimeError("Unable to load model:\n" + "\n".join(errors))
