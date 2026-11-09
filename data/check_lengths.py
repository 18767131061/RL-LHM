"""Check assistant target capacity with the configured processor, without weights."""
import argparse
import json
from pathlib import Path

from transformers import AutoProcessor
from data.config import load_config
from data.manifest import load_manifest
from data.lengths import target_lengths, length_summary
from models.huggingface import DecodingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = load_manifest(config["manifest"])
    model = config["model"]
    processor = AutoProcessor.from_pretrained(model["processor_name_or_path"], revision=model["revision"],
                                             trust_remote_code=False, **model["processor_kwargs"])
    generation = DecodingConfig(**config["generation"])
    report = length_summary(target_lengths(processor, manifest.records, manifest.output_classes, generation),
                            generation.max_new_tokens)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))
    if report["over_budget_groups"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
