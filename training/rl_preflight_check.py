# -*- coding: utf-8 -*-
"""
RL Preflight Check

Run before starting VeRL to check:
1. RL data format
2. image paths
3. token budget
4. reward function stability
5. rl_dataset.py image handling support
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict

from transformers import AutoProcessor


def print_title(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_jsonl_sample(path: Path, sample_idx: int) -> tuple[list[str], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    sample = json.loads(lines[sample_idx])
    return lines, sample


def image_path_from_item(item: Any) -> str:
    if isinstance(item, dict):
        return str(item["image"])
    return str(item)


def build_messages(sample: Dict[str, Any]) -> list[Dict[str, Any]]:
    content = []
    for item in sample["images"]:
        if isinstance(item, dict):
            content.append({"type": "image", **dict(item)})
        else:
            content.append({"type": "image", "image": str(item)})
    content.append({"type": "text", "text": str(sample["prompt"])})
    return [{"role": "user", "content": content}]


def import_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call_reward_safely(reward_module, response_text: str, sample: Dict[str, Any]) -> Any:
    gt = sample["ground_truth"]
    extra_info = sample.get("extra_info", {})
    data_source = sample.get("data_source", "preflight")

    try:
        return reward_module.compute_score(
            data_source=data_source,
            solution_str=response_text,
            ground_truth=gt,
            extra_info=extra_info,
        )
    except TypeError:
        try:
            return reward_module.compute_score(response_text, gt)
        except Exception as e:
            return e
    except Exception as e:
        return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-data-path", type=str, required=True)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--reward-path", type=str, required=True)
    ap.add_argument("--rl-dataset-path", type=str, default="")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-response-len", type=int, default=256)
    args = ap.parse_args()

    rl_data_path = Path(args.rl_data_path).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    reward_path = Path(args.reward_path).expanduser().resolve()
    rl_dataset_path = Path(args.rl_dataset_path).expanduser().resolve() if args.rl_dataset_path else None

    print_title("RL PREFLIGHT CHECK")

    # 1. Check RL data format
    print_title("[1] RL data format")
    lines, sample = load_jsonl_sample(rl_data_path, args.sample_idx)
    print(f"RL records: {len(lines)}")
    print(f"Fields: {list(sample.keys())}")
    assert "prompt" in sample, "Missing prompt field"
    assert "images" in sample, "Missing images field"
    assert "ground_truth" in sample, "Missing ground_truth field"
    print(f"Image count: {len(sample['images'])}")
    print(f"images[0] type: {type(sample['images'][0])}")
    print(f"Contains reward_model: {'reward_model' in sample}")
    if "reward_model" in sample and isinstance(sample["reward_model"], dict):
        print(f"reward_model keys: {list(sample['reward_model'].keys())}")
    if isinstance(sample["images"][0], dict):
        print(f"images[0] keys: {list(sample['images'][0].keys())}")
        if "max_pixels" in sample["images"][0]:
            print(f"max_pixels: {sample['images'][0]['max_pixels']}")

    # 2. Check image existence
    print_title("[2] Check image paths")
    for i, item in enumerate(sample["images"]):
        img_path = image_path_from_item(item)
        exists = os.path.exists(img_path)
        print(f"Image {i}: exists={exists}, path={img_path}")
        assert exists, f"Image does not exist: {img_path}"

    # 3. Token budget
    print_title("[3] Token budget")
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    from qwen_vl_utils import process_vision_info

    messages = build_messages(sample)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    total_tokens = int(inputs["input_ids"].shape[1])
    remaining = int(args.max_model_len - total_tokens)
    print(f"processor class: {processor.__class__.__name__}")
    print(f"processor keys: {list(inputs.keys())}")
    print(f"Total prompt tokens (including vision): {total_tokens}")
    print(f"max_model_len: {args.max_model_len}")
    print(f"Remaining generation space: {remaining}")
    print(f"max_response_length: {args.max_response_len}")
    if remaining < args.max_response_len:
        print(f"[FATAL] Remaining space {remaining} < max_response_length {args.max_response_len}")
        print(f"Recommended max_model_len >= {total_tokens + args.max_response_len + 100}")
    else:
        print("[OK] Token space is sufficient")

    # 4. Reward
    print_title("[4] Reward function")
    reward_module = import_module_from_file("reward_module", reward_path)
    ok_reward = call_reward_safely(reward_module, json.dumps(sample["ground_truth"], ensure_ascii=False), sample)
    print(f"Reward for valid output: {ok_reward}")
    bad_reward = call_reward_safely(reward_module, "not valid JSON", sample)
    print(f"Reward for invalid output: {bad_reward}")

    # 5. rl_dataset compatibility
    print_title("[5] rl_dataset compatibility")
    if rl_dataset_path and rl_dataset_path.exists():
        text_src = rl_dataset_path.read_text(encoding="utf-8", errors="ignore")
        image_related = ("images" in text_src) or ("image" in text_src)
        print(f"rl_dataset_path = {rl_dataset_path}")
        print(f"Contains image/images keywords: {image_related}")
        for needle in ["max_pixels", "process_vision_info", "raw_prompt", "_build_messages"]:
            print(f"  contains {needle}: {needle in text_src}")
    else:
        print("No readable rl_dataset.py path was supplied; skipping static source check.")

    print("\n" + "=" * 60)
    print("PREFLIGHT CHECK complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
