"""Bind resumable inference output to its exact inputs and local model."""
from __future__ import annotations

import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inference_identity(model_dir, input_path, data_root, rows, settings, *, entry_path=None):
    model_dir, input_path, data_root = map(lambda p: Path(p).resolve(),
                                          (model_dir, input_path, data_root))
    # Include weights, model/processor/generation configuration, tokenizer files
    # and any local custom Python model implementation used by Transformers.
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja", ".tiktoken", ".py"}
    model_files = sorted(p for p in model_dir.rglob("*") if p.is_file() and p.suffix in suffixes)
    if not model_files or not (model_dir / "config.json").is_file():
        raise ValueError("Local model configuration and files are required for inference identity")
    images = {}
    for row in rows:
        user = next(message for message in row["messages"] if message["role"] == "user")
        paths = [(data_root / segment["image"]).resolve()
                 for segment in user["content"] if segment["type"] == "image"]
        if len(paths) != 3:
            raise ValueError("Inference requires exactly three image paths per input")
        for path in paths:
            if str(path) not in images:
                images[str(path)] = file_sha256(path)
    versions = {}
    for package in ("torch", "transformers", "qwen-vl-utils", "numpy", "Pillow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"schema_version": 1, "model_dir": str(model_dir),
            "model_files": {str(p.relative_to(model_dir)): file_sha256(p) for p in model_files},
            "input_path": str(input_path), "input_sha256": file_sha256(input_path),
            "data_root": str(data_root), "images": images, "settings": settings,
            "runtime_packages": versions,
            "entry_sha256": file_sha256(entry_path or Path(__file__).with_name("infer_qwen_real961.py")),
            "resume_sha256": file_sha256(__file__)}


def output_center(metadata: dict) -> str | None:
    """Accept the converter's two center aliases without inventing a center."""
    values = [metadata[key] for key in ("center_id", "analysis_center")
              if metadata.get(key) is not None]
    if not values:
        return None
    if any(not isinstance(value, str) or not value.strip() or value != value.strip()
           for value in values) or len(set(values)) != 1:
        raise ValueError("Inconsistent or invalid center_id/analysis_center metadata")
    return values[0]


class ResumeGuard:
    """Verify saved bytes; commit only newly appended bytes after each CSV row."""
    def __init__(self, output, identity):
        self.output = Path(output)
        self.partial = self.output.with_suffix(self.output.suffix + ".partial")
        self.sidecar = self.output.with_suffix(self.output.suffix + ".resume.json")
        self.identity = identity
        self._lock = self.sidecar.with_suffix(self.sidecar.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            raise ValueError("Another inference process owns this output") from error
        self.hasher = hashlib.sha256()
        self.size = 0
        existing = self.output if self.output.exists() else self.partial
        if self.output.exists() and self.partial.exists():
            raise ValueError("Both final and partial inference outputs exist")
        if self.sidecar.exists():
            state = json.loads(self.sidecar.read_text(encoding="utf-8"))
            if state.get("identity") != identity:
                raise ValueError("Inference model, seed, settings or input changed; use a new output path")
            if existing.exists():
                self._read_new_bytes(existing)
            if state.get("bytes") != self.size or state.get("sha256") != self.hasher.hexdigest():
                raise ValueError("Inference output changed or has uncommitted bytes; use a new output path")
        elif existing.exists():
            raise ValueError("Existing inference output lacks its identity sidecar; use a new output path")
        else:
            self._save(False)

    def _read_new_bytes(self, path):
        with Path(path).open("rb") as stream:
            if stream.seek(0, os.SEEK_END) < self.size:
                raise ValueError("Inference output was truncated")
            stream.seek(self.size)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                self.hasher.update(block)
                self.size += len(block)

    def _save(self, complete):
        temporary = self.sidecar.with_suffix(self.sidecar.suffix + ".tmp")
        state = {"identity": self.identity, "bytes": self.size,
                 "sha256": self.hasher.hexdigest(), "complete": bool(complete)}
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.sidecar)

    def checkpoint(self, path, complete=False):
        self._read_new_bytes(path)
        if complete and file_sha256(path) != self.hasher.hexdigest():
            raise ValueError("Inference output changed before finalization")
        self._save(complete)

    def close(self):
        self._lock.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self._lock.close()
