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
from data.lengths import target_lengths, length_summary
from models.huggingface import DecodingConfig, HuggingFaceBackend
from rewards import RewardConfig, RARSParameters
from rl import DACVQState, ShaeBaselineStore, VGSSampler
from .checkpoint import restore_checkpoint, save_checkpoint, training_identity, source_identity
from .distributed import initialize, zero2_runtime
from .engine import ComponentSwitches, PolicyParameters, ShaeParameters, fit_rl, fit_sft


def run(stage: str, configuration: dict) -> list:
    """Train from explicit JSON settings and save one checkpoint per epoch."""
    if stage not in {"sft", "rl"}:
        raise ValueError("stage must be sft or rl")
    if configuration["model"]["dtype"] != "bfloat16":
        raise ValueError("four-GPU training requires bfloat16 parameters")
    local_rank = initialize(configuration["distributed"]["world_size"])
    manifest = load_manifest(configuration["manifest"])
    seed = configuration["seed"] + local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(configuration["deterministic_algorithms"])
    generation = DecodingConfig(**configuration["generation"])
    source_model = dict(configuration["model"])
    source_model["device"] = f"cuda:{local_rank}"
    current_model = dict(source_model)
    resume_from = configuration["resume_from"]
    if resume_from is not None:
        current_model.update(
            model_name_or_path=resume_from, processor_name_or_path=resume_from, revision=None,
        )
    backend = HuggingFaceBackend.from_pretrained(**current_model, generation=generation)
    backend.enable_training(configuration["loop"]["max_sequence_length"])
    runtime = zero2_runtime(backend, configuration["optimizer"], configuration["loop"]["max_grad_norm"])
    local_lengths = target_lengths(backend.processor, manifest.records[runtime.rank::runtime.world_size],
                                   manifest.output_classes, generation)
    all_lengths = sorted((row for shard in runtime.gather(local_lengths) for row in shard), key=lambda row: row["uid"])
    report = length_summary(all_lengths, generation.max_new_tokens)
    if runtime.rank == 0:
        Path(configuration["output_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(configuration["output_dir"]) / "target_lengths.json").write_text(json.dumps(report, indent=2))
    if report["over_budget_groups"]:
        raise ValueError(f"{report['over_budget_groups']} GT targets exceed max_new_tokens; see target_lengths.json")
    generators = {"order": torch.Generator().manual_seed(configuration["order_seed"])}
    baseline = sampler = shae = reference = None
    visual_vectors = {record["uid"]: record.get("visual_vector") for record in manifest.records}
    visual_vectors = {
        uid: None if vector is None else torch.tensor(vector, dtype=torch.float64)
        for uid, vector in visual_vectors.items()
    }
    if stage == "rl":
        if (not generation.do_sample or generation.temperature != 1 or generation.top_p != 1
                or generation.top_k != 0 or generation.repetition_penalty != 1):
            raise ValueError("RL samples the scored policy: do_sample=true, temperature=top_p=1, top_k=0, repetition_penalty=1")
        backend.policy_scoring = True
        switches = ComponentSwitches(**configuration["components"])
        policy = PolicyParameters(**configuration["policy"])
        reward_settings = dict(configuration["reward"])
        reward_settings["rars_parameters"] = RARSParameters(**reward_settings["rars_parameters"])
        reward = RewardConfig(
            classes=manifest.output_classes, rars_enabled=switches.rars, **reward_settings,
        )
        # Both policies originate from the explicitly selected post-SFT model.
        reference = HuggingFaceBackend.from_pretrained(**source_model, generation=generation)
        reference.model.requires_grad_(False)
        reference.model.eval()
        reference.policy_scoring = True
        reference.max_sequence_length = configuration["loop"]["max_sequence_length"]
        if switches.shae:
            shae_settings = configuration["shae"]
            baseline = ShaeBaselineStore(beta=shae_settings["baseline_beta"])
            shae = ShaeParameters(**{key: shae_settings[key] for key in ("gate_mid", "gate_temperature", "noise_std")})
            generators["shae"] = torch.Generator().manual_seed(shae_settings["seed"])
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
                replacement=configuration["sampling"]["replacement"],
            )
    source_fingerprint = runtime.gather(source_identity(source_model) if runtime.rank == 0 else None)[0]
    identity = training_identity(
        configuration,
        {**manifest.payload(), "source": source_fingerprint},
    )
    start_epoch, previous_history = 0, []
    if resume_from is not None:
        start_epoch, previous_history = restore_checkpoint(
            runtime, directory=resume_from, stage=stage, identity=identity,
            generators=generators, baseline_store=baseline, curriculum_sampler=sampler,
        )
    output = Path(configuration["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, configuration["loop"]["epochs"]):
        if (output / f"epoch_{epoch + 1:04d}").exists():
            raise ValueError("An output checkpoint already exists for a requested training epoch")

    def save_epoch(epoch, history):
        save_checkpoint(
            backend, runtime, directory=output / f"epoch_{epoch + 1:04d}",
            stage=stage, completed_epoch=epoch, identity=identity,
            configuration=configuration, history=previous_history + history,
            generators=generators, baseline_store=baseline, curriculum_sampler=sampler,
        )
        if runtime.rank == 0:
            print(json.dumps({**history[-1], "stage": stage, "epoch": epoch + 1}, ensure_ascii=False))

    loop = configuration["loop"]
    common = dict(
        runtime=runtime, epochs=loop["epochs"], start_epoch=start_epoch,
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
            shae_generator=generators.get("shae"),
            sampling_replacement=configuration["sampling"]["replacement"], **common,
        )
    return previous_history + history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sft", "rl"))
    parser.add_argument("--config", required=True, help="JSON training configuration")
    arguments = parser.parse_args()
    configuration = load_config(arguments.config)
    try:
        run(arguments.stage, configuration)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
