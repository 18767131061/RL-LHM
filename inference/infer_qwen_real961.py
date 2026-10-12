#!/usr/bin/env python3
"""Deterministic, resumable Qwen3-VL inference for the frozen real961 test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from .resume import ResumeGuard, file_sha256, inference_identity
except ImportError:
    from resume import ResumeGuard, file_sha256, inference_identity


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render_prompt(processor, messages: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return processor.apply_chat_template(messages, **kwargs)


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


def sample_key(row: dict) -> tuple[str, int]:
    metadata = row["metadata"]
    return str(metadata["logical_key"]), int(metadata["slice_index"])


def completion_nll(
    scores: tuple[torch.Tensor, ...],
    generated: torch.Tensor,
    pad_id: int | None,
) -> float:
    if len(scores) != generated.shape[1]:
        raise RuntimeError(
            f"Generation score/token mismatch: {len(scores)} vs {generated.shape[1]}"
        )
    values = []
    for index, logits in enumerate(scores):
        token = int(generated[0, index].item())
        if pad_id is not None and token == pad_id:
            continue
        values.append(float(-torch.log_softmax(logits[0].float(), dim=-1)[token].item()))
    return float(np.mean(values)) if values else float("inf")


def read_completed(path: Path, expected: set[tuple[str, int]]) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError(f"Unexpected columns in resumable output: {reader.fieldnames}")
        keys = []
        for row in reader:
            keys.append((row["logical_key"], int(row["slice_index"])))
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate samples in resumable output: {path}")
    unknown = set(keys) - expected
    if unknown:
        raise RuntimeError(f"Unknown samples in resumable output: {sorted(unknown)[:5]}")
    return set(keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--image-max-pixels", type=int, default=512 * 512)
    parser.add_argument("--image-min-pixels", type=int, default=32 * 32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
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
    expected_keys = [sample_key(row) for row in rows]
    if len(expected_keys) != len(set(expected_keys)):
        raise RuntimeError("Duplicate logical_key/slice_index samples in test JSONL")
    expected = set(expected_keys)
    settings = {"seed": args.seed, "device": args.device,
                "max_new_tokens": args.max_new_tokens,
                "image_max_pixels": args.image_max_pixels,
                "image_min_pixels": args.image_min_pixels, "do_sample": False}
    identity = inference_identity(model_dir, json_path, data_root, rows, settings)
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
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    processor.image_min_pixels = args.image_min_pixels
    processor.image_max_pixels = args.image_max_pixels
    dtype = (
        torch.bfloat16
        if args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
    ).to(args.device)
    model.eval()
    from qwen_vl_utils import process_vision_info

    mode = "a" if completed else "w"
    with partial.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        if not completed:
            writer.writeheader()
            stream.flush()
            os.fsync(stream.fileno())
            guard.checkpoint(partial)
        for row in tqdm(rows, desc="real961 inference"):
            metadata = row["metadata"]
            key = sample_key(row)
            if key in completed:
                continue
            user = next(message for message in row["messages"] if message["role"] == "user")
            content = []
            image_paths = []
            for segment in user["content"]:
                if segment["type"] == "image":
                    path = str((data_root / segment["image"]).resolve())
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
                    content.append({"type": "image", "image": path})
                    image_paths.append(path)
                elif segment["type"] == "text":
                    content.append({"type": "text", "text": segment["text"]})
            if len(image_paths) != 3:
                raise RuntimeError(f"Expected three images for {metadata['logical_key']}")
            messages = [{"role": "user", "content": content}]
            rendered = render_prompt(processor, messages)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[rendered],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(args.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated_output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            generated = generated_output.sequences[:, prompt_length:]
            raw = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            avg_nll = completion_nll(
                generated_output.scores,
                generated,
                getattr(processor.tokenizer, "pad_token_id", None),
            )
            writer.writerow(
                {
                    "logical_key": metadata["logical_key"],
                    "patient_id": metadata["patient_id"],
                    "class": metadata["class"],
                    "slice_index": metadata["slice_index"],
                    "positive": int(metadata["positive"]),
                    "analysis_center": metadata["analysis_center"],
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
    if inference_identity(model_dir, json_path, data_root, rows, settings) != identity:
        raise RuntimeError("Model, input or images changed during inference")
    shutil.move(str(partial), str(output))
    guard.checkpoint(output, complete=True)
    guard.close()


if __name__ == "__main__":
    main()
