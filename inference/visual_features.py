"""Write fixed three-phase visual vectors into a prepared manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.manifest import load_manifest
from data.config import load_config
from models import DecodingConfig, HuggingFaceBackend
from models.visual_features import extract_qwen3_visual_vectors, load_projection, orthonormal_projection


def run(config: dict) -> int:
    manifest = load_manifest(config["manifest"])
    batch_size = config["batch_size"]
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    backend = HuggingFaceBackend.from_pretrained(
        **config["model"], generation=DecodingConfig(**config["generation"])
    )
    backend.model.requires_grad_(False)
    backend.model.eval()
    settings = config["projection"]
    if set(settings) == {"path"}:
        projection = load_projection(settings["path"])
    elif set(settings) == {"seed", "dimension", "output"}:
        projection = orthonormal_projection(
            backend.model.config.vision_config.hidden_size, settings["dimension"], settings["seed"]
        )
        destination = Path(settings["output"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.save(stream, projection.numpy(), allow_pickle=False)
    else:
        raise ValueError("projection needs path, or seed/dimension/output")
    records = []
    for start in range(0, len(manifest.records), batch_size):
        batch = manifest.records[start:start + batch_size]
        vectors = extract_qwen3_visual_vectors(backend, batch, projection)
        for record, vector in zip(batch, vectors):
            output_record = {key: value for key, value in record.items() if key != "patient_label"}
            output_record["visual_vector"] = vector.cpu().tolist()
            records.append(output_record)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump({"classes": list(manifest.classes), "patients": manifest.patients, "records": records},
                  stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    count = run(config)
    print(json.dumps({"groups": count, "output": config["output"]}))


if __name__ == "__main__":
    main()
