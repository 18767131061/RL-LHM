"""Command-line SFT and RL training with Hugging Face multimodal models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from data.manifest import load_manifest, training_class_counts
from data.config import load_config
from models.huggingface import DecodingConfig, HuggingFaceBackend
from rewards import RewardConfig, RARSParameters
from rl import DACVQState, ShaeBaselineStore, VGSSampler
from .checkpoint import restore_checkpoint, save_checkpoint, training_identity
from .engine import ComponentSwitches, PolicyParameters, ShaeParameters, fit_rl, fit_sft


def _optimizer(backend, settings):
    return torch.optim.AdamW(
        [parameter for parameter in backend.model.parameters() if parameter.requires_grad],
        lr=settings["learning_rate"], betas=tuple(settings["betas"]),
        eps=settings["epsilon"], weight_decay=settings["weight_decay"],
        amsgrad=settings["amsgrad"],
    )


def run(stage: str, configuration: dict) -> list:
    """Train from explicit JSON settings and save one checkpoint per epoch."""
    if stage not in {"sft", "rl"}:
        raise ValueError("stage must be sft or rl")
    if configuration["model"]["dtype"] != "float32":
        raise ValueError("Training uses float32 model parameters and AdamW state")
    if torch.device(configuration["model"]["device"]).type not in {"cpu", "cuda"}:
        raise ValueError("Training devices must be CPU or CUDA")
    manifest = load_manifest(configuration["manifest"])
    seed = configuration["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(configuration["deterministic_algorithms"])
    generation = DecodingConfig(**configuration["generation"])
    source_model = dict(configuration["model"])
    current_model = dict(source_model)
    resume_from = configuration["resume_from"]
    if resume_from is not None:
        current_model.update(
            model_name_or_path=resume_from, processor_name_or_path=resume_from, revision=None,
        )
    backend = HuggingFaceBackend.from_pretrained(**current_model, generation=generation)
    optimizer = _optimizer(backend, configuration["optimizer"])
    generators = {"order": torch.Generator().manual_seed(configuration["order_seed"])}
    baseline = sampler = shae = reference = None
    visual_vectors = {record["uid"]: record.get("visual_vector") for record in manifest.records}
    visual_vectors = {
        uid: None if vector is None else torch.tensor(vector, dtype=torch.float64)
        for uid, vector in visual_vectors.items()
    }
    if stage == "rl":
        if not generation.do_sample:
            raise ValueError("RL requires sampled completions")
        switches = ComponentSwitches(**configuration["components"])
        policy = PolicyParameters(**configuration["policy"])
        reward_settings = dict(configuration["reward"])
        reward_settings["rars_parameters"] = RARSParameters(**reward_settings["rars_parameters"])
        reward = RewardConfig(
            classes=manifest.classes, rars_enabled=switches.rars, **reward_settings,
        )
        # Both policies originate from the explicitly selected post-SFT model.
        reference = HuggingFaceBackend.from_pretrained(**source_model, generation=generation)
        reference.model.requires_grad_(False)
        reference.model.eval()
        if switches.shae:
            shae_settings = configuration["shae"]
            baseline = ShaeBaselineStore(beta=shae_settings["baseline_beta"])
            shae = ShaeParameters(**{key: shae_settings[key] for key in ("gate_mid", "gate_temperature", "noise_std")})
            device = next(backend.model.parameters()).device
            generators["shae"] = torch.Generator(device=device).manual_seed(shae_settings["seed"])
        if switches.curriculum:
            curriculum = configuration["curriculum"]
            generators["curriculum"] = torch.Generator().manual_seed(curriculum["state_seed"])
            generators["sampling"] = torch.Generator().manual_seed(curriculum["sampling_seed"])
            memory = DACVQState(**curriculum["state"], generator=generators["curriculum"])
            sampler = VGSSampler(
                memory, [visual_vectors[record["uid"]] for record in manifest.records],
                generator=generators["sampling"], use_vgs=True, epoch_count=0,
                missing_feature_multiplier=curriculum["missing_feature_multiplier"],
                weight_floor=curriculum["weight_floor"],
            )
    identity = training_identity(
        configuration,
        {"classes": list(manifest.classes), "patients": manifest.patients, "records": manifest.records},
    )
    start_epoch, previous_history = 0, []
    if resume_from is not None:
        start_epoch, previous_history = restore_checkpoint(
            optimizer, directory=resume_from, stage=stage, identity=identity,
            generators=generators, baseline_store=baseline, curriculum_sampler=sampler,
        )
    output = Path(configuration["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, configuration["loop"]["epochs"]):
        if (output / f"epoch_{epoch + 1:04d}").exists():
            raise ValueError("An output checkpoint already exists for a requested training epoch")

    def save_epoch(epoch, history):
        save_checkpoint(
            backend, optimizer, directory=output / f"epoch_{epoch + 1:04d}",
            stage=stage, completed_epoch=epoch, identity=identity,
            configuration=configuration, history=previous_history + history,
            generators=generators, baseline_store=baseline, curriculum_sampler=sampler,
        )
        print(json.dumps({**history[-1], "stage": stage, "epoch": epoch + 1}, ensure_ascii=False))

    loop = configuration["loop"]
    common = dict(
        optimizer=optimizer, epochs=loop["epochs"], start_epoch=start_epoch,
        batch_size=loop["batch_size"], microbatch_size=loop["microbatch_size"],
        max_grad_norm=loop["max_grad_norm"], on_epoch=save_epoch,
    )
    if stage == "sft":
        history = fit_sft(backend, manifest.records, generator=generators["order"], **common)
    else:
        history = fit_rl(
            backend, reference, manifest.records, group_size=loop["group_size"],
            policy_updates=loop["policy_updates"], advantage_epsilon=loop["advantage_epsilon"],
            switches=switches, policy=policy, reward_config=reward,
            class_counts=training_class_counts(manifest),
            baseline_store=baseline, shae_parameters=shae, curriculum_sampler=sampler,
            visual_vectors=visual_vectors, order_generator=generators["order"],
            shae_generator=generators.get("shae"), **common,
        )
    return previous_history + history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sft", "rl"))
    parser.add_argument("--config", required=True, help="JSON training configuration")
    arguments = parser.parse_args()
    configuration = load_config(arguments.config)
    run(arguments.stage, configuration)


if __name__ == "__main__":
    main()
