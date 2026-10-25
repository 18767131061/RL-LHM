#!/usr/bin/env python3
"""Deterministic, resumable inference for supported Hugging Face VLM baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - depends on the installed transformers version
    AutoModelForVision2Seq = None


try:
    from .resume import ResumeGuard, file_sha256, inference_identity, output_center
except ImportError:
    from resume import ResumeGuard, file_sha256, inference_identity, output_center


FIELDNAMES = [
    "logical_key",
    "patient_id",
    "class",
    "slice_index",
    "positive",
    "analysis_center",
    "a",
    "p",
    "v",
    "avg_nll",
    "raw",
]


try:
    from .decoding import add_decoding_arguments, config_from_args, input_seed, unprocessed_logits
except ImportError:
    from decoding import add_decoding_arguments, config_from_args, input_seed, unprocessed_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    if "metadata" in row:
        return row["metadata"]
    case_id = str(row["case_id"])
    labels = {item["label"] for item in row.get("detections", []) if item.get("label")}
    if not labels:
        raise RuntimeError(f"Public row has no class label: {case_id}")
    return {
        "logical_key": case_id,
        "patient_id": case_id,
        "class": str(next(iter(labels))),
        "slice_index": int(row["slice_index"]),
        "positive": 1,
        "analysis_center": "MCT-LTDiag",
    }


def sample_key(row: dict[str, Any]) -> tuple[str, int]:
    metadata = row_metadata(row)
    return str(metadata["logical_key"]), int(metadata["slice_index"])


def read_completed(path: Path, expected: set[tuple[str, int]]) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError(f"Unexpected columns in resumable output: {reader.fieldnames}")
        keys = [(row["logical_key"], int(row["slice_index"])) for row in reader]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate samples in resumable output: {path}")
    unknown = set(keys) - expected
    if unknown:
        raise RuntimeError(f"Unknown samples in resumable output: {sorted(unknown)[:5]}")
    return set(keys)


def completion_nll(
    scores: tuple[torch.Tensor, ...],
    generated: torch.Tensor,
    pad_id: int | None,
) -> float:
    if len(scores) != generated.shape[1]:
        raise RuntimeError(f"Generation score/token mismatch: {len(scores)} vs {generated.shape[1]}")
    values = []
    for index, logits in enumerate(scores):
        token = int(generated[0, index].item())
        if pad_id is not None and token == pad_id:
            continue
        values.append(float(-torch.log_softmax(logits[0].float(), dim=-1)[token].item()))
    return float(np.mean(values)) if values else float("inf")


def load_model(model_dir: Path, dtype: torch.dtype, device: str):
    common = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    candidates = [AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM, AutoModel]
    errors = []
    seen = set()
    for loader in candidates:
        if loader is None or loader in seen:
            continue
        seen.add(loader)
        for dtype_key in ("dtype", "torch_dtype"):
            try:
                model = loader.from_pretrained(
                    str(model_dir),
                    **common,
                    **{dtype_key: dtype},
                )
                model = model.to(device)
                model.eval()
                return model, loader.__name__
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                errors.append(f"{loader.__name__}/{dtype_key}: {type(error).__name__}: {error}")
    raise RuntimeError("Unable to load VLM:\n" + "\n".join(errors[-10:]))


def render_prompt(processor, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return processor.apply_chat_template(messages, **kwargs)


def prepare_inputs(
    processor,
    prompt: str,
    images: list[Image.Image],
    image_paths: list[str],
) -> dict[str, Any]:
    path_messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": path} for path in image_paths],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rich_messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    errors = []
    for messages in (rich_messages, path_messages):
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }
        try:
            return dict(processor.apply_chat_template(messages, **kwargs))
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            errors.append(f"multimodal chat template: {type(error).__name__}: {error}")
            kwargs.pop("enable_thinking")
            try:
                return dict(processor.apply_chat_template(messages, **kwargs))
            except (AttributeError, TypeError, ValueError, RuntimeError) as fallback_error:
                errors.append(
                    f"multimodal chat template fallback: "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                )

    rendered = render_prompt(processor, path_messages)
    for text_value in ([rendered], rendered):
        for image_value in (images, [images]):
            try:
                return dict(
                    processor(
                        text=text_value,
                        images=image_value,
                        padding=True,
                        return_tensors="pt",
                    )
                )
            except (TypeError, ValueError, RuntimeError) as error:
                errors.append(f"processor text/images: {type(error).__name__}: {error}")
    raise RuntimeError("Unable to prepare multimodal inputs:\n" + "\n".join(errors[-10:]))


def to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--image-max-pixels", type=int, default=512 * 512)
    parser.add_argument("--image-min-pixels", type=int, default=32 * 32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, help="Canary-only row limit; never use for formal metrics.")
    add_decoding_arguments(parser)
    args = parser.parse_args()
    decoding = config_from_args(args)
    set_seed(args.seed)

    model_dir = Path(args.model_dir).resolve()
    json_path = Path(args.json_path).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_sha = file_sha256(json_path)
    with json_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if args.limit is not None:
        if args.limit <= 0:
            raise RuntimeError("--limit must be positive")
        rows = rows[: args.limit]
    expected_keys = [sample_key(row) for row in rows]
    if len(expected_keys) != len(set(expected_keys)):
        raise RuntimeError("Duplicate logical_key/slice_index samples in test JSONL")
    expected = set(expected_keys)
    centers = {sample_key(row): output_center(row_metadata(row)) for row in rows}
    settings = {"seed": args.seed, "device": args.device,
                "max_new_tokens": args.max_new_tokens,
                "image_max_pixels": args.image_max_pixels,
                "image_min_pixels": args.image_min_pixels, "decoding": decoding.identity(), "limit": args.limit}
    identity = inference_identity(model_dir, json_path, data_root, rows, settings, entry_path=__file__)
    if identity["input_sha256"] != source_sha:
        raise RuntimeError("Input changed while it was read")
    guard = ResumeGuard(output, identity)
    if output.is_file():
        completed_output = read_completed(output, expected)
        if completed_output == expected:
            guard.checkpoint(output, complete=True)
            guard.close()
            print(f"[inference] complete output already exists: {output}")
            return
        raise RuntimeError(f"Incomplete final output must not be resumed in place: {output}")

    completed = read_completed(partial, expected)
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    for target in (processor, getattr(processor, "image_processor", None)):
        if target is None:
            continue
        for name, value in (
            ("image_min_pixels", args.image_min_pixels),
            ("image_max_pixels", args.image_max_pixels),
            ("min_pixels", args.image_min_pixels),
            ("max_pixels", args.image_max_pixels),
        ):
            if hasattr(target, name):
                setattr(target, name, value)
    dtype = (
        torch.bfloat16
        if args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model, loader_name = load_model(model_dir, dtype, args.device)
    print(f"[inference] model_loader={loader_name} rows={len(rows)}")

    mode = "a" if completed else "w"
    with partial.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        if not completed:
            writer.writeheader()
            stream.flush()
            os.fsync(stream.fileno())
            guard.checkpoint(partial)
        for row in tqdm(rows, desc=f"{model_dir.name} inference"):
            metadata = row_metadata(row)
            key = sample_key(row)
            if key in completed:
                continue
            user = next(message for message in row["messages"] if message["role"] == "user")
            prompt = next(segment["text"] for segment in user["content"] if segment["type"] == "text")
            image_paths = [
                str((data_root / segment["image"]).resolve())
                for segment in user["content"]
                if segment["type"] == "image"
            ]
            if len(image_paths) != 3 or not all(Path(path).is_file() for path in image_paths):
                raise RuntimeError(f"Expected three existing images for {metadata['logical_key']}")
            with Image.open(image_paths[0]) as first, Image.open(image_paths[1]) as second, Image.open(
                image_paths[2]
            ) as third:
                images = [first.convert("RGB"), second.convert("RGB"), third.convert("RGB")]
            inputs = to_device(prepare_inputs(processor, prompt, images, image_paths), args.device)
            if "input_ids" not in inputs:
                raise RuntimeError(f"Processor did not return input_ids for {key}: {sorted(inputs)}")
            if decoding.mode == "sample":
                set_seed(input_seed(args.seed, key))
            with torch.inference_mode():
                generated_output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    **decoding.generation_kwargs(),
                    return_dict_in_generate=True,
                    output_logits=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            generated = generated_output.sequences
            if generated.shape[1] >= prompt_length:
                generated = generated[:, prompt_length:]
            raw = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            avg_nll = completion_nll(
                tuple(unprocessed_logits(generated_output)),
                generated,
                None,
            )
            writer.writerow(
                {
                    "logical_key": metadata["logical_key"],
                    "patient_id": metadata["patient_id"],
                    "class": metadata["class"],
                    "slice_index": metadata["slice_index"],
                    "positive": int(metadata["positive"]),
                    "analysis_center": centers[key],
                    "a": image_paths[0],
                    "p": image_paths[1],
                    "v": image_paths[2],
                    "avg_nll": "" if not math.isfinite(avg_nll) else f"{avg_nll:.8f}",
                    "raw": raw,
                }
            )
            stream.flush()
            os.fsync(stream.fileno())
            guard.checkpoint(partial)

    final_keys = read_completed(partial, expected)
    missing = expected - final_keys
    if missing:
        raise RuntimeError(f"Inference ended with missing samples: {sorted(missing)[:5]}")
    if inference_identity(model_dir, json_path, data_root, rows, settings, entry_path=__file__) != identity:
        raise RuntimeError("Model, input or images changed during inference")
    shutil.move(str(partial), str(output))
    guard.checkpoint(output, complete=True)
    guard.close()


if __name__ == "__main__":
    main()
