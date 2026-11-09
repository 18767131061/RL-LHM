"""Atomic epoch checkpoints with collective ZeRO shards and per-rank RNG state."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from data.provenance import source_revision

import numpy as np
import torch
import torch.distributed as dist


def training_identity(configuration, manifest_payload):
    settings = json.loads(json.dumps(configuration))
    for name in ("output_dir", "resume_from"):
        settings.pop(name, None)
    settings["loop"].pop("epochs", None)
    payload = {"configuration": settings, "manifest": manifest_payload}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_identity(source):
    """Hash local source weights; remote sources must use immutable revisions."""
    path = Path(source["model_name_or_path"])
    if not path.is_dir():
        revision = source["revision"]
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("remote training model revision must be an immutable 40-character commit")
        return {"model": source["model_name_or_path"], "revision": revision}
    weights = set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("local model source has no Hugging Face weights")
    metadata = {item for pattern in ("*.json", "*.jinja", "*.txt") for item in path.glob(pattern)
                if not item.name.startswith("training_") and item.name != "checkpoint.json"}
    files = sorted(weights | metadata)
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode())
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def capture_state(generators, baseline_store, curriculum_sampler):
    numpy_state = np.random.get_state()
    state = {
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "numpy_rng": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
        "generators": {name: generator.get_state() for name, generator in generators.items()},
        "baseline": None, "curriculum": None,
    }
    if baseline_store is not None:
        state["baseline"] = {"beta": baseline_store.beta, "entries": dict(baseline_store.entries)}
    if curriculum_sampler is not None:
        memory = curriculum_sampler.state
        state["curriculum"] = {
            **{name: getattr(memory, name).clone() for name in
               ("embedding", "ema_cluster_size", "ema_weight", "reward_ema", "reward_seen", "last_seen", "visit_count")},
            "update_step": memory.update_step, "revival_count": memory.revival_count,
            "filled_slots": memory.filled_slots, "is_initialized": memory.is_initialized,
            "epoch_count": curriculum_sampler.epoch_count,
            "replacement": curriculum_sampler.replacement,
        }
    return state


def restore_state(state, generators, baseline_store, curriculum_sampler):
    if set(state["generators"]) != set(generators):
        raise ValueError("checkpoint random streams do not match")
    if (state["baseline"] is None) != (baseline_store is None):
        raise ValueError("checkpoint SHAE component does not match")
    if (state["curriculum"] is None) != (curriculum_sampler is None):
        raise ValueError("checkpoint curriculum component does not match")
    if baseline_store is not None:
        if state["baseline"]["beta"] != baseline_store.beta:
            raise ValueError("checkpoint baseline decay does not match")
        baseline_store.entries = dict(state["baseline"]["entries"])
    if curriculum_sampler is not None:
        memory, saved = curriculum_sampler.state, state["curriculum"]
        if saved["replacement"] != curriculum_sampler.replacement:
            raise ValueError("checkpoint replacement sampling does not match")
        for name in ("embedding", "ema_cluster_size", "ema_weight", "reward_ema", "reward_seen", "last_seen", "visit_count"):
            if saved[name].shape != getattr(memory, name).shape:
                raise ValueError("checkpoint codebook dimensions do not match")
            setattr(memory, name, saved[name].clone())
        memory.update_step = int(saved["update_step"])
        memory.revival_count = int(saved["revival_count"])
        memory.filled_slots = int(saved["filled_slots"])
        memory.is_initialized = bool(saved["is_initialized"])
        curriculum_sampler.epoch_count = int(saved["epoch_count"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name])
    torch.random.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    value = state["numpy_rng"]
    np.random.set_state((value["name"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))


def _barrier(runtime):
    if runtime.world_size > 1:
        dist.barrier()


def save_checkpoint(backend, runtime, *, directory, stage, completed_epoch, identity,
                    configuration, history, generators, baseline_store, curriculum_sampler):
    destination = Path(directory)
    staging = destination.with_name(destination.name + ".incomplete")
    state = capture_state(generators, baseline_store, curriculum_sampler)
    if runtime.rank == 0:
        if destination.exists():
            raise FileExistsError(destination)
        staging.mkdir(parents=True, exist_ok=False)
    _barrier(runtime)
    if runtime.engine is not None:
        runtime.engine.save_checkpoint(str(staging / "zero"), tag="state", save_latest=False)
    else:
        torch.save(runtime.optimizer.state_dict(), staging / f"optimizer_rank_{runtime.rank}.pt")
    torch.save(state, staging / f"rank_{runtime.rank:05d}.pt")
    if runtime.rank == 0:
        backend.save_pretrained(staging)
        metadata = dict(format_version=3, stage=stage, completed_epoch=completed_epoch,
                        identity=identity, world_size=runtime.world_size, source_revision=source_revision())
        (staging / "checkpoint.json").write_text(json.dumps(metadata, indent=2))
        (staging / "training_config.json").write_text(json.dumps(configuration, indent=2))
        (staging / "training_history.json").write_text(json.dumps(history, indent=2))
    _barrier(runtime)
    if runtime.rank == 0:
        staging.rename(destination)
    _barrier(runtime)
    # Checkpoint serialization must not advance the continuation RNG stream.
    restore_state(state, generators, baseline_store, curriculum_sampler)


def restore_checkpoint(runtime, *, directory, stage, identity, generators,
                       baseline_store, curriculum_sampler):
    path = Path(directory)
    saved = json.loads((path / "checkpoint.json").read_text())
    if (saved["format_version"] != 3 or saved["stage"] != stage or saved["identity"] != identity
            or saved["world_size"] != runtime.world_size):
        raise ValueError("checkpoint stage, source, data, settings or world size does not match")
    if runtime.engine is not None:
        loaded, _ = runtime.engine.load_checkpoint(str(path / "zero"), tag="state",
                                                  load_optimizer_states=True, load_lr_scheduler_states=True)
        if loaded is None:
            raise ValueError("ZeRO checkpoint was not loaded")
    else:
        runtime.optimizer.load_state_dict(torch.load(path / f"optimizer_rank_{runtime.rank}.pt",
                                                    map_location="cpu", weights_only=True))
    state = torch.load(path / f"rank_{runtime.rank:05d}.pt", map_location="cpu", weights_only=True)
    restore_state(state, generators, baseline_store, curriculum_sampler)
    return int(saved["completed_epoch"]) + 1, json.loads((path / "training_history.json").read_text())
