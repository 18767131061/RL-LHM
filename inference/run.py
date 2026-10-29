"""Generate one structured completion and completion NLL per input group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.manifest import load_manifest
from data.config import load_config
from data.lengths import target_lengths, length_summary
from data.provenance import file_digest, source_revision
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
    lengths = target_lengths(backend.processor, manifest.records, manifest.output_classes, backend.generation)
    lengths_by_uid = {row["uid"]: row for row in lengths}
    phase_sources = config.get("phase_sources", {p: p for p in ("a", "p", "v")})
    if set(phase_sources) != {"a", "p", "v"} or any(p not in {"a", "p", "v"} for p in phase_sources.values()):
        raise ValueError("phase_sources must map each input slot a/p/v to a source phase")
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for start in range(0, len(manifest.records), batch_size):
            records = manifest.records[start:start + batch_size]
            inputs = [dict(record, images={p: record["images"][phase_sources[p]] for p in phase_sources})
                      for record in records]
            batch = backend.generate(inputs)
            for index, (record, text, nll) in enumerate(zip(records, batch.texts, batch.completion_nlls)):
                stream.write(json.dumps({
                    "uid": record["uid"], "patient_id": record["patient_id"],
                    "completion": text, "nll": nll,
                    "finish_reason": batch.finish_reasons[index],
                    "completion_tokens": batch.completion_token_counts[index],
                    "max_new_tokens": backend.generation.max_new_tokens,
                    "target_tokens": lengths_by_uid[record["uid"]]["target_tokens"],
                    "phase_sources": phase_sources,
                }, ensure_ascii=False, allow_nan=False) + "\n")
    checkpoint_path = Path(config["model"]["model_name_or_path"]) / "checkpoint.json"
    metadata = dict(configuration=config, source_revision=source_revision(),
                    manifest_sha256=file_digest(config["manifest"]), predictions_sha256=file_digest(output),
                    output_classes=manifest.output_classes, metric_classes=manifest.metric_classes,
                    target_lengths=length_summary(lengths, backend.generation.max_new_tokens),
                    checkpoint=json.loads(checkpoint_path.read_text()) if checkpoint_path.is_file() else None)
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2))
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
