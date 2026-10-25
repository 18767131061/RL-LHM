"""Generate one structured completion and completion NLL per input group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.manifest import load_manifest
from data.config import load_config
from models import DecodingConfig, HuggingFaceBackend


def run(config: dict) -> int:
    manifest = load_manifest(config["manifest"])
    batch_size = config["batch_size"]
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    torch.manual_seed(config["seed"])
    backend = HuggingFaceBackend.from_pretrained(
        **config["model"], generation=DecodingConfig(**config["generation"])
    )
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for start in range(0, len(manifest.records), batch_size):
            records = manifest.records[start:start + batch_size]
            batch = backend.generate(records)
            for record, text, nll in zip(records, batch.texts, batch.completion_nlls):
                stream.write(json.dumps({
                    "uid": record["uid"], "patient_id": record["patient_id"],
                    "completion": text, "nll": nll,
                }, ensure_ascii=False, allow_nan=False) + "\n")
    return len(manifest.records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    count = run(config)
    print(json.dumps({"groups": count, "output": config["output"]}))


if __name__ == "__main__":
    main()
