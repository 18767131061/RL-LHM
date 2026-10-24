from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from inference.decoding import DecodingConfig, unprocessed_logits


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def completion_nll(
    scores: Sequence[Any],
    generated_tokens: Any,
    *,
    pad_token_id: int | None = None,
) -> float:
    """Mean token NLL for a generated completion, used only as a ranking proxy."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("completion_nll requires torch") from error

    tokens = generated_tokens.reshape(-1)
    if len(scores) != int(tokens.numel()):
        raise ValueError(f"Score/token length mismatch: {len(scores)} != {tokens.numel()}")
    values = []
    for index, logits in enumerate(scores):
        token = int(tokens[index].item())
        if pad_token_id is not None and token == pad_token_id:
            continue
        values.append(float(-torch.log_softmax(logits[0].float(), dim=-1)[token].item()))
    return float(np.mean(values)) if values else float("inf")


def generate_one(
    model: Any,
    processor: Any,
    *,
    messages: list[dict[str, Any]],
    images: list[Any],
    device: str,
    max_new_tokens: int = 512,
    decoding: DecodingConfig | None = None,
) -> tuple[str, float]:
    """Generate one structured completion and return text plus completion NLL."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("generate_one requires torch") from error
    decoding = decoding or DecodingConfig()
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    template_args = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        prompt = processor.apply_chat_template(messages, **template_args)
    except TypeError:
        template_args.pop("enable_thinking")
        prompt = processor.apply_chat_template(messages, **template_args)

    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **decoding.generation_kwargs(),
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_logits=True,
        )
    generated = output.sequences[:, input_length:]
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    nll = completion_nll(
        unprocessed_logits(output),
        generated,
        pad_token_id=None,
    )
    return text, nll
