#!/usr/bin/env python3
"""Check a prepared RL-LHM record's images, actual prompt budget and reward schema.

This does not run a model rollout or verify an external VeRL GPU installation.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.schema import DEFAULT_LABEL_ALIASES, parse_completion


def load_jsonl_sample(path: Path, sample_idx: int) -> tuple[list[str], dict[str, Any]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 <= sample_idx < len(lines):
        raise ValueError(f"sample index {sample_idx} is outside {len(lines)} prepared records")
    sample = json.loads(lines[sample_idx])
    if not isinstance(sample, dict):
        raise ValueError("Prepared record must be a JSON object")
    return lines, sample


def image_path_from_item(item: Any) -> str:
    value = item.get("image") if isinstance(item, dict) else item
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every image must provide a nonempty local path")
    return value


def build_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Restore converter chat messages, replacing each image marker once.

    Image order and processing metadata (such as max_pixels) are preserved.
    A legacy plain text prompt without markers prepends the three images.
    Chat-list prompts must explicitly identify their image slots.
    """
    images = sample.get("images")
    if not isinstance(images, list) or len(images) != 3:
        raise ValueError("Prepared RL-LHM records require exactly three images")
    image_content = []
    for item in images:
        path = image_path_from_item(item)
        segment = copy.deepcopy(item) if isinstance(item, dict) else {"image": path}
        if segment.get("type", "image") != "image":
            raise ValueError("Only phase-image inputs are supported")
        segment["type"] = "image"
        image_content.append(segment)
    prompt = sample.get("prompt")
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("Prompt must be nonempty")
        if "<image>" not in prompt:
            return [{"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}]
        prompt = [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Prompt must be a nonempty chat-message list or text")
    messages, image_index = [], 0
    for message in prompt:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or not message["role"].strip() or not isinstance(message.get("content"), str)):
            raise ValueError("Prepared chat messages require a role and string content")
        if "<video>" in message["content"]:
            raise ValueError("Video placeholders are outside the three-phase image contract")
        content = []
        for part in re.split(r"(<image>)", message["content"]):
            if part == "<image>":
                if image_index >= len(image_content):
                    raise ValueError("Prompt has more image placeholders than image inputs")
                content.append(image_content[image_index])
                image_index += 1
            elif part:
                content.append({"type": "text", "text": part})
        restored = copy.deepcopy(message)
        restored["content"] = content
        messages.append(restored)
    if image_index != len(image_content):
        raise ValueError("Prompt image-placeholder count does not match image inputs")
    return messages


def ground_truth_object(sample: dict[str, Any]) -> dict[str, Any]:
    source = sample.get("ground_truth")
    nested = sample.get("reward_model", {}).get("ground_truth")
    if source is None:
        source = nested
    decoded = json.loads(source) if isinstance(source, str) else copy.deepcopy(source)
    if not isinstance(decoded, dict):
        raise ValueError("ground_truth must decode once to an object")
    if nested is not None:
        nested_decoded = json.loads(nested) if isinstance(nested, str) else nested
        if nested_decoded != decoded:
            raise ValueError("Top-level and reward_model ground truth disagree")
    return decoded


def ground_truth_completion(sample: dict[str, Any]) -> str:
    """Convert gt_dets/bbox plus explicit backgrounds to the prediction schema."""
    truth = ground_truth_object(sample)
    if "gt_dets" in truth:
        if not isinstance(truth["gt_dets"], list) or not isinstance(truth.get("background_phases", []), list):
            raise ValueError("gt_dets and background_phases must be lists")
        records = []
        for item in truth["gt_dets"]:
            if not isinstance(item, dict):
                raise ValueError("Ground-truth lesion must be an object")
            converted = copy.deepcopy(item)
            if "bbox_2d" not in converted and "bbox" in converted:
                converted["bbox_2d"] = converted.pop("bbox")
            records.append(converted)
        records.extend({"phase": phase, "label": "no-lesion", "bbox_2d": [-1, -1, -1, -1]}
                       for phase in truth.get("background_phases", []))
        payload = {"detections": records}
    elif "detections" in truth:
        payload = truth
    else:
        raise ValueError("Ground truth needs gt_dets or detections")
    completion = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    parsed = parse_completion(completion, set(DEFAULT_LABEL_ALIASES.values()))
    if not parsed.detections and not parsed.background_phases:
        raise ValueError("Ground truth has neither lesion records nor explicit background phases")
    return completion


def import_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reward module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_reward_safely(reward_module, response_text: str, sample: dict[str, Any]) -> Any:
    """Call the released VeRL callback without hiding or retrying exceptions."""
    return reward_module.compute_score(
        data_source=sample.get("data_source", "preflight"),
        solution_str=response_text,
        ground_truth=ground_truth_object(sample),
        extra_info=sample.get("extra_info", {}),
    )


def check_reward_contract(reward_module, sample: dict[str, Any]) -> dict[str, Any]:
    valid = call_reward_safely(reward_module, ground_truth_completion(sample), sample)
    invalid = call_reward_safely(reward_module, "not valid JSON", sample)
    for name, result in (("valid", valid), ("invalid", invalid)):
        if not isinstance(result, dict) or not isinstance(result.get("score"), (int, float)) or not math.isfinite(result["score"]):
            raise RuntimeError(f"{name} reward probe did not return a finite score dictionary")
    if valid.get("parse_ok") != 1 or valid["score"] <= 0:
        raise RuntimeError(f"Ground-truth completion did not receive a valid positive reward: {valid}")
    if invalid.get("parse_ok") != 0 or invalid["score"] != 0:
        raise RuntimeError(f"Invalid completion did not satisfy the zero-reward schema contract: {invalid}")
    return {"valid": valid, "invalid": invalid}


def prompt_token_count(sample, processor, process_vision_info) -> int:
    messages = build_messages(sample)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 0:
        raise RuntimeError("Processor must return one nonempty token sequence")
    return int(ids.shape[1])


def check_token_budget(total_tokens: int, max_prompt_len: int, max_model_len: int, max_response_len: int) -> None:
    if min(total_tokens, max_prompt_len, max_model_len, max_response_len) <= 0:
        raise ValueError("Token counts and configured limits must be positive")
    if total_tokens > max_prompt_len:
        raise ValueError(f"Prompt has {total_tokens} tokens, exceeding data.max_prompt_length={max_prompt_len}")
    remaining = max_model_len - total_tokens
    if remaining < max_response_len:
        raise ValueError(f"Remaining context {remaining} is less than max_response_length={max_response_len}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-data-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reward-path", type=Path, required=True)
    parser.add_argument("--rl-dataset-path", type=Path, help="Optional external dataset source; informational inspection only")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-response-len", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    lines, sample = load_jsonl_sample(args.rl_data_path.expanduser().resolve(), args.sample_idx)
    build_messages(sample)  # Reject malformed messages before loading the processor.
    for image in sample["images"]:
        path = Path(image_path_from_item(image)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
    if not args.model_path.expanduser().is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")
    reward = import_module_from_file("preflight_reward", args.reward_path.expanduser().resolve())
    reward_results = check_reward_contract(reward, sample)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(str(args.model_path.expanduser().resolve()),
                                             local_files_only=True, trust_remote_code=args.trust_remote_code)
    count = prompt_token_count(sample, processor, process_vision_info)
    check_token_budget(count, args.max_prompt_len, args.max_model_len, args.max_response_len)
    if args.rl_dataset_path is not None:
        dataset_path = args.rl_dataset_path.expanduser().resolve()
        text = dataset_path.read_text(encoding="utf-8")
        print(json.dumps({"external_dataset_source": str(dataset_path), "inspection_only": True,
                          "keywords": {key: key in text for key in ("max_pixels", "process_vision_info", "raw_prompt", "_build_messages")}}))
    print(json.dumps({"status": "passed_selected_record", "records_in_file": len(lines),
                      "sample_index": args.sample_idx, "prompt_tokens_including_images": count,
                      "max_prompt_length": args.max_prompt_len, "max_model_length": args.max_model_len,
                      "max_response_length": args.max_response_len,
                      "reward_probes": reward_results, "gpu_rollout_verified": False}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"[FATAL] {error}") from error
