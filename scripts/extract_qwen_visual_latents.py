#!/usr/bin/env python3
"""Released, image-only Qwen3-VL reference extractor for fixed 1152-D caches.

This explicitly versioned implementation instantiates the public cache interface;
it does not claim equivalence to an unavailable earlier extraction implementation.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "qwen3vl_three_phase_reference_v1"
PROJECTION = ROOT / "models/assets/qwen3vl_projection_1024x384_v1.npy"
PROJECTION_SHA256 = "76d1919c689850c70e0a7f2d481c71a7ee06d3587f27d434f716abfba9a7082b"
PHASES = ["arterial", "portal_venous", "delayed"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_rows(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("extra_info", {}).get("sample_uid")
            if not isinstance(uid, str) or not uid.strip() or uid in seen:
                raise ValueError("Each input must have a unique nonempty extra_info.sample_uid")
            seen.add(uid)
            rows.append(row)
    if not rows:
        raise ValueError("Empty input JSONL")
    return rows


def image_specs(row, image_root):
    entries = row.get("images")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("Expected exactly three images in arterial, portal-venous, delayed order")
    specs = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"image": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("image"), str):
            raise ValueError("Each image must be a local path or an image-path object")
        if entry.get("phase", PHASES[index]) != PHASES[index]:
            raise ValueError("Explicit image phase contradicts the AP/PVP/DP ordering contract")
        path = Path(entry["image"])
        if not path.is_absolute():
            path = Path(image_root) / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Image file does not exist: {path}")
        maximum = entry.get("max_pixels", 262144)
        if type(maximum) is not int or maximum < 1024:
            raise ValueError("max_pixels must be an integer of at least 1024")
        specs.append({"path": str(path), "max_pixels": maximum})
    return specs


def checkpoint_files(model_dir):
    directory = Path(model_dir).resolve()
    if not directory.is_dir():
        raise ValueError("--model-dir must be an existing local checkpoint directory")
    if (directory / "adapter_config.json").exists():
        raise ValueError("Adapter-only/PEFT directories are not accepted: merge the SFT adapter "
                         "into its exact local base model and save a complete checkpoint plus "
                         "image processor first. No base model is downloaded automatically.")
    required = [directory / "config.json", directory / "preprocessor_config.json"]
    if any(not path.is_file() for path in required):
        raise ValueError("Local checkpoint requires config.json and preprocessor_config.json")
    safe = sorted(directory.glob("*.safetensors"))
    binary = sorted(directory.glob("pytorch_model*.bin"))
    weights = safe or binary
    if not weights:
        raise ValueError("No local safetensors or pytorch_model*.bin checkpoint weights")
    return sorted(set(required + weights + list(directory.glob("*.index.json"))))


def build_identity(model_dir, input_path, rows, image_root, device, dtype):
    files = checkpoint_files(model_dir)
    config = json.loads((Path(model_dir) / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl" or config.get("vision_config", {}).get("hidden_size") != 1024:
        raise ValueError("Reference v1 requires Qwen3-VL with 1024-wide visual hidden states")
    if sha(PROJECTION) != PROJECTION_SHA256:
        raise ValueError("Bundled projection checksum mismatch")
    memo, image_records = {}, []
    for row in rows:
        specs = image_specs(row, image_root)
        for spec in specs:
            if spec["path"] not in memo:
                memo[spec["path"]] = sha(spec["path"])
        image_records.append({"sample_uid": row["extra_info"]["sample_uid"],
                              "images": [{**spec, "sha256": memo[spec["path"]]} for spec in specs]})
    runtime = {}
    for package in ("torch", "transformers", "numpy", "Pillow", "safetensors"):
        try:
            runtime[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            runtime[package] = None
    return {"contract": CONTRACT, "extractor_sha256": sha(__file__), "runtime_packages": runtime,
            "input_sha256": sha(input_path), "input_path": str(Path(input_path).resolve()),
            "image_root": str(Path(image_root).resolve()), "model_dir": str(Path(model_dir).resolve()),
            "checkpoint_files": {path.name: sha(path) for path in files},
            "input_images_sha256": digest(image_records), "phase_order": PHASES,
            "projection_sha256": PROJECTION_SHA256, "feature_dim": 1152,
            "device": device, "dtype": dtype, "pooling_dtype": "float32",
            "label_or_text_used_for_features": False}


def valid_vector(vector):
    return (isinstance(vector, list) and len(vector) == 1152
            and all(type(v) in (int, float) and math.isfinite(v) for v in vector)
            and abs(sum(v * v for v in vector) - 1.0) < 1e-4)


def add_vector(row, vector, identity_hash):
    if not valid_vector(vector):
        raise ValueError("Feature must be finite, nonzero, L2-normalized and 1152-dimensional")
    result = copy.deepcopy(row)
    result["extra_info"].update(visual_z=vector, visual_z_contract=CONTRACT,
                                 visual_z_cache_id=identity_hash)
    if isinstance(result.get("reward_model"), dict):
        result["reward_model"]["extra_info"] = copy.deepcopy(result["extra_info"])
    return result


def pool_features(hidden, grid, projection):
    """Mean each phase's raw final-block tokens, project, concatenate, normalize."""
    import torch
    if tuple(grid.shape) != (3, 3) or bool((grid <= 0).any()):
        raise ValueError("Expected three positive image grids")
    sizes = [int(v) for v in grid.prod(dim=1).tolist()]
    if hidden.ndim != 2 or hidden.shape != (sum(sizes), 1024):
        raise ValueError("Unexpected pre-merger visual tensor shape")
    if tuple(projection.shape) != (1024, 384) or not torch.isfinite(hidden).all():
        raise ValueError("Invalid projection or nonfinite visual activations")
    pooled = torch.stack([part.float().mean(dim=0) for part in hidden.split(sizes)])
    vector = (pooled @ projection.to(device=hidden.device, dtype=torch.float32)).reshape(1152)
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Nonfinite or zero visual feature norm")
    result = (vector / norm).cpu().tolist()
    if not valid_vector(result):
        raise ValueError("Invalid normalized visual feature")
    return result


class QwenVisualExtractor:
    def __init__(self, model_dir, device="cuda", dtype="bfloat16"):
        import numpy as np
        import torch
        import transformers
        from transformers import AutoImageProcessor
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        if transformers.__version__ != "4.57.1":
            raise ValueError("Reference v1 is validated with transformers==4.57.1; install that version")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu --dtype float32 for a CPU check")
        if device == "cpu" and dtype != "float32":
            raise ValueError("CPU reference extraction requires --dtype float32")
        files = checkpoint_files(model_dir)
        configuration = json.loads((Path(model_dir) / "config.json").read_text())
        if configuration.get("model_type") != "qwen3_vl" or configuration["vision_config"]["hidden_size"] != 1024:
            raise ValueError("Reference v1 requires a Qwen3-VL 1024-wide vision encoder")
        vision_config = Qwen3VLVisionConfig(**configuration["vision_config"])
        vision_config._attn_implementation = "eager"
        self.device, self.dtype = torch.device(device), getattr(torch, dtype)
        self.vision = Qwen3VLVisionModel(vision_config).to(dtype=self.dtype)
        state = {}
        for path in files:
            if path.suffix == ".safetensors":
                from safetensors import safe_open
                with safe_open(str(path), framework="pt", device="cpu") as source:
                    for key in source.keys():
                        if key.startswith("model.visual."):
                            state[key.removeprefix("model.visual.")] = source.get_tensor(key)
            elif path.suffix == ".bin":
                source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                state.update({key.removeprefix("model.visual."): value for key, value in source.items()
                              if key.startswith("model.visual.")})
        if not state:
            raise ValueError("Checkpoint has no model.visual.* parameters; save a full Qwen3-VL SFT checkpoint")
        self.vision.load_state_dict(state, strict=True)
        del state
        self.vision.to(self.device).eval().requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        if sha(PROJECTION) != PROJECTION_SHA256:
            raise ValueError("Bundled projection checksum mismatch")
        self.projection = torch.from_numpy(np.load(PROJECTION, allow_pickle=False).copy()).to(self.device)
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                         "numpy": np.__version__, "attention": "eager"}

    def __call__(self, specs):
        import torch
        from PIL import Image
        pixels, grids = [], []
        for spec in specs:
            with Image.open(spec["path"]) as image:
                encoded = self.processor(images=[image.convert("RGB")], return_tensors="pt",
                                         max_pixels=spec["max_pixels"])
            pixels.append(encoded["pixel_values"])
            grids.append(encoded["image_grid_thw"])
        grid = torch.cat(grids).to(self.device)
        captured = []
        handle = self.vision.merger.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            with torch.inference_mode():
                self.vision(torch.cat(pixels).to(self.device, dtype=self.dtype), grid_thw=grid)
                if len(captured) != 1:
                    raise ValueError("Expected exactly one final pre-merger feature tensor")
                return pool_features(captured[0], grid, self.projection)
        finally:
            handle.remove()


def extract_cache(input_path, output_path, model_dir, image_root=None, device="cuda",
                  dtype="bfloat16", backend_factory=QwenVisualExtractor, validate_only=False):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("Output must differ from the source JSONL")
    image_root = Path(image_root).resolve() if image_root else input_path.parent
    input_sha = sha(input_path)
    rows = read_rows(input_path)
    if sha(input_path) != input_sha:
        raise ValueError("Input JSONL changed while it was read")
    identity = build_identity(model_dir, input_path, rows, image_root, device, dtype)
    if identity["input_sha256"] != input_sha:
        raise ValueError("Input JSONL changed while its identity was prepared")
    identity_hash = digest(identity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    state_path = output_path.with_suffix(output_path.suffix + ".state.json")
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another extractor owns this output cache") from error
        if output_path.exists():
            if not audit_path.is_file():
                raise ValueError("Existing cache has no audit sidecar; use a new output path")
            audit = json.loads(audit_path.read_text())
            if (audit.get("cache_id") != identity_hash or audit.get("status") != "complete"
                    or audit.get("output_sha256") != sha(output_path)):
                raise ValueError("Existing cache identity/content conflicts; use a new output path")
            cached = read_rows(output_path)
            if len(cached) != len(rows) or any(item != add_vector(row, item.get("extra_info", {}).get("visual_z"), identity_hash)
                                               for item, row in zip(cached, rows)):
                raise ValueError("Existing cache row contract mismatch")
            return audit
        if validate_only:
            raise ValueError("No completed cache exists for validation")
        if state_path.exists():
            if json.loads(state_path.read_text()).get("cache_id") != identity_hash:
                raise ValueError("Partial cache conflicts with current input/checkpoint; use a new output path")
        else:
            if partial.exists():
                raise ValueError("Partial cache without its identity sidecar")
            write_json(state_path, {"cache_id": identity_hash, "identity": identity})
        completed = []
        if partial.exists():
            # Only a torn final line can be discarded; every complete row is revalidated.
            with partial.open("rb+") as stream:
                data = stream.read()
                if data and not data.endswith(b"\n"):
                    data = data[:data.rfind(b"\n") + 1]
                    stream.seek(0); stream.write(data); stream.truncate()
            completed = [json.loads(line) for line in data.splitlines()]
            if len(completed) > len(rows):
                raise ValueError("Partial cache has extra records")
            for item, row in zip(completed, rows):
                if item != add_vector(row, item.get("extra_info", {}).get("visual_z"), identity_hash):
                    raise ValueError("Partial cache contains a changed source row or invalid feature")
        backend = backend_factory(model_dir, device=device, dtype=dtype)
        versions = backend.versions
        with partial.open("a", encoding="utf-8") as stream:
            for row in rows[len(completed):]:
                vector = backend(image_specs(row, image_root))
                stream.write(canonical(add_vector(row, vector, identity_hash)) + "\n")
                stream.flush(); os.fsync(stream.fileno())
        # Reject concurrent checkpoint/input/image mutation before committing the cache.
        if build_identity(model_dir, input_path, rows, image_root, device, dtype) != identity:
            raise ValueError("Input, checkpoint or images changed during extraction")
        audit = {"status": "complete", "cache_id": identity_hash, "identity": identity,
                 "output_sha256": sha(partial), "rows": len(rows), "runtime": versions}
        write_json(audit_path, audit)
        os.replace(partial, output_path)
        state_path.unlink(missing_ok=True)
        return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-root", help="Relative image paths default to the input JSONL directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--validate-only", action="store_true", help="Check a completed cache without loading the model")
    args = parser.parse_args()
    audit = extract_cache(args.input_jsonl, args.output_jsonl, args.model_dir,
                          args.image_root, args.device, args.dtype, validate_only=args.validate_only)
    print(json.dumps({"status": audit["status"], "rows": audit["rows"], "cache_id": audit["cache_id"]}))


if __name__ == "__main__":
    main()
