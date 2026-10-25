"""Epoch checkpoint state for model, optimizer, curriculum, and random streams."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch

from rl import ShaeBaselineStore, VGSSampler


def training_identity(configuration: dict, manifest_payload: dict) -> str:
    """Hash the data and settings that determine a continued training run."""
    settings = json.loads(json.dumps(configuration))
    settings.pop("output_dir", None)
    settings.pop("resume_from", None)
    settings["loop"].pop("epochs", None)
    payload = {"configuration": settings, "manifest": manifest_payload}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_checkpoint(
    backend, optimizer, *, directory: str | Path, stage: str,
    completed_epoch: int, identity: str, configuration: dict, history: list,
    generators: Mapping[str, torch.Generator],
    baseline_store: ShaeBaselineStore | None,
    curriculum_sampler: VGSSampler | None,
):
    """Save a Hugging Face model/processor and the state needed at the next epoch."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    backend.save_pretrained(destination)
    numpy_state = np.random.get_state()
    state = {
        "stage": stage, "completed_epoch": completed_epoch, "identity": identity,
        "optimizer": optimizer.state_dict(), "history": history,
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": {
            "name": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": numpy_state[2], "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "generators": {name: generator.get_state() for name, generator in generators.items()},
        "baseline": None,
        "curriculum": None,
    }
    if baseline_store is not None:
        state["baseline"] = {"beta": baseline_store.beta, "entries": dict(baseline_store.entries)}
    if curriculum_sampler is not None:
        memory = curriculum_sampler.state
        names = ("embedding", "ema_cluster_size", "ema_weight", "reward_ema", "reward_seen")
        state["curriculum"] = {
            **{name: getattr(memory, name).clone() for name in names},
            "filled_slots": memory.filled_slots,
            "is_initialized": memory.is_initialized,
            "epoch_count": curriculum_sampler.epoch_count,
        }
    torch.save(state, destination / "training_state.pt")
    (destination / "training_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (destination / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def restore_checkpoint(
    optimizer, *, directory: str | Path, stage: str, identity: str,
    generators: Mapping[str, torch.Generator],
    baseline_store: ShaeBaselineStore | None,
    curriculum_sampler: VGSSampler | None,
) -> tuple[int, list]:
    """Restore epoch state after loading the checkpoint's model and processor."""
    state = torch.load(Path(directory) / "training_state.pt", map_location="cpu", weights_only=True)
    if state["stage"] != stage or state["identity"] != identity:
        raise ValueError("checkpoint stage, training data, or mathematical settings do not match")
    if set(state["generators"]) != set(generators):
        raise ValueError("checkpoint random streams do not match the configured components")
    if (state["baseline"] is None) != (baseline_store is None):
        raise ValueError("checkpoint SHAE component does not match")
    if (state["curriculum"] is None) != (curriculum_sampler is None):
        raise ValueError("checkpoint curriculum component does not match")
    if baseline_store is not None:
        if state["baseline"]["beta"] != baseline_store.beta:
            raise ValueError("checkpoint baseline decay does not match")
        baseline_store.entries = dict(state["baseline"]["entries"])
    if curriculum_sampler is not None:
        memory = curriculum_sampler.state
        saved = state["curriculum"]
        for name in ("embedding", "ema_cluster_size", "ema_weight", "reward_ema", "reward_seen"):
            if saved[name].shape != getattr(memory, name).shape:
                raise ValueError("checkpoint codebook dimensions do not match")
            setattr(memory, name, saved[name].clone())
        memory.filled_slots = int(saved["filled_slots"])
        memory.is_initialized = bool(saved["is_initialized"])
        curriculum_sampler.epoch_count = int(saved["epoch_count"])
    optimizer.load_state_dict(state["optimizer"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name])
    torch.random.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count must match the saved training run")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    numpy_state = state["numpy_rng"]
    np.random.set_state((
        numpy_state["name"], np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    return int(state["completed_epoch"]) + 1, list(state["history"])
