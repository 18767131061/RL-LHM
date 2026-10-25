#!/usr/bin/env python3
"""Export a completed VeRL actor and infer the explicit validation input JSONL."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_paper_protocol import validate_manifest
from evaluation.postprocess_verl_checkpoint import validate_merged_model
from inference.decoding import config_from_args
from training.rars_joint_grid import digest, sha, write_json
from training.rars_oat import checkpoint_digest, read_json
from training.rars_oat_execution import immutable_json, run_logged


def is_model_weight(path):
    name = path.name.lower()
    return (path.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack", ".onnx")
            or name.endswith((".safetensors.index.json", ".bin.index.json")))


def processor_source_digest(source):
    """Bind config/tokenizer/processor assets without rehashing unused base weights."""
    source = Path(source)
    if not source.is_dir():
        raise ValueError("processor-source must be an existing local model directory")
    files = {str(path.relative_to(source)): sha(path) for path in sorted(source.rglob("*"))
             if path.is_file() and not is_model_weight(path)}
    if not files:
        raise ValueError("processor-source contains no processor/config/tokenizer assets")
    return digest(files)


def processor_assets_main(argv):
    """Run inside the explicitly selected inference Python environment."""
    parser = argparse.ArgumentParser(description="Export processor assets from an explicit local source")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args(argv)
    if processor_source_digest(args.source) != args.expected_source_sha256:
        raise ValueError("Processor source changed before export")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(args.source), local_files_only=True, trust_remote_code=True)
    staging = args.export.parent / ".processor_assets.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        processor.save_pretrained(str(staging))
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        if not files:
            raise ValueError("AutoProcessor.save_pretrained produced no assets")
        if any(path.is_symlink() or not path.resolve().is_relative_to(staging.resolve()) for path in files):
            raise ValueError("Processor asset export contains a symlink outside its staging directory")
        # Validate the entire staging directory before copying any files.
        if any(is_model_weight(path) or path.name in ("config.json", "generation_config.json") for path in files):
            raise ValueError("Processor assets cannot overwrite model weights or model/generation config")
        if processor_source_digest(args.source) != args.expected_source_sha256:
            raise ValueError("Processor source changed during export")
        args.export.mkdir(parents=True, exist_ok=True)
        for path in files:
            target = args.export / path.relative_to(staging)
            if target.is_symlink() or not target.resolve().is_relative_to(args.export.resolve()):
                raise ValueError("Processor asset destination must remain inside its model export")
        for path in files:
            target = args.export / path.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    finally:
        shutil.rmtree(staging)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "processor-assets":
        return processor_assets_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--inference-input", required=True, type=Path)
    parser.add_argument("--processor-source", type=Path, help="Explicit seed SFT directory supplying processor/tokenizer assets")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--merge-python", default=sys.executable)
    parser.add_argument("--inference-python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--decoding", choices=("greedy", "sample"), required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    decoding = config_from_args(args)
    manifest = read_json(args.validation_manifest)
    if manifest.get("split") != "validation":
        raise ValueError("OAT inference requires a validation manifest")
    validate_manifest(manifest)
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "actor").is_dir():
        raise ValueError("Completed VeRL checkpoint must contain its actor directory")
    output = args.output.resolve()
    folder = output.parent / "model_export"
    processor_source = args.processor_source.resolve() if args.processor_source else None
    processor_hash = processor_source_digest(processor_source) if processor_source else None
    identity = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_digest(checkpoint),
                "validation_sha256": sha(args.validation_manifest), "inference_input_sha256": sha(args.inference_input),
                "seed": args.seed, "data_root": str(args.data_root.resolve()),
                "merge_python": args.merge_python, "inference_python": args.inference_python,
                "device": args.device, "max_new_tokens": args.max_new_tokens, "decoding": decoding.identity(),
                "processor_source": str(processor_source) if processor_source else None,
                "processor_source_sha256": processor_hash}
    immutable_json(output.with_suffix(output.suffix + ".oat_identity.json"), identity)
    receipt = output.parent / "model_export.json"
    if folder.exists() and not receipt.exists():
        # A model export is reusable only together with its complete identity.
        # Recreate this task-owned derived directory from the bound checkpoint.
        shutil.rmtree(folder)
    if not folder.exists():
        temporary = output.parent / ".model_export.partial"
        if temporary.exists():
            shutil.rmtree(temporary)
        command = [args.merge_python, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                   "--use_cpu_initialization", "--local_dir", str(checkpoint / "actor"), "--target_dir", str(temporary)]
        run_logged(command, output.parent / "model_export.log")
        if processor_source is not None:
            run_logged([args.inference_python, str(Path(__file__).absolute()), "processor-assets",
                        "--source", str(processor_source), "--export", str(temporary),
                        "--expected-source-sha256", processor_hash], output.parent / "processor_export.log")
        validate_merged_model(temporary)
        write_json(receipt, {"identity": identity, "model_sha256": checkpoint_digest(temporary)})
        os.replace(temporary, folder)
    else:
        exported = read_json(receipt)
        if exported.get("identity") != identity or exported.get("model_sha256") != checkpoint_digest(folder):
            raise ValueError("Existing model export differs from its checkpoint or inputs")
    validate_merged_model(folder)
    command = [args.inference_python, str(ROOT / "inference/infer_hf_vlm_revision.py"),
               "--model-dir", str(folder), "--json-path", str(args.inference_input.resolve()),
               "--data-root", str(args.data_root.resolve()), "--output", str(output),
               "--seed", str(args.seed), "--device", args.device,
               "--max-new-tokens", str(args.max_new_tokens), "--decoding", decoding.mode,
               "--temperature", str(decoding.temperature), "--top-p", str(decoding.top_p), "--top-k", str(decoding.top_k)]
    run_logged(command, output.parent / "vlm_inference.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
