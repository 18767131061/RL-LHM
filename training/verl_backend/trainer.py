"""Synchronous VeRL recipe with RL-LHM driver state and atomic epoch saves."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict

from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.skip.skip_manager import SkipManager
from data.manifest import load_manifest
from data.lengths import target_lengths, length_summary
from data.provenance import source_revision
from models.huggingface import DecodingConfig
from training.checkpoint import source_identity, training_identity
from . import UPSTREAM_COMMIT
from .config import verify_upstream
from .dataset import ManifestDataset
from .driver import update_once
from .state import PaperState


class PaperVeRLTrainer(RayPPOTrainer):
    def __init__(self, *, paper, identity, **kwargs):
        self.paper, self.identity = paper, identity
        self.history, self.global_steps = [], 0
        super().__init__(**kwargs)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        # The manifest and epoch order remain driver-owned. The official default
        # dataloader drops tails and does not know the persistent VGS state.
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.total_training_steps = math.ceil(len(train_dataset) / self.config.data.train_batch_size) * self.config.trainer.total_epochs
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps

    def _save_epoch(self):
        settings = self.paper.configuration
        destination = Path(settings["output_dir"]) / f"epoch_{self.paper.completed_epochs:04d}"
        staging = destination.with_name(destination.name + ".incomplete")
        if destination.exists() or staging.exists():
            raise FileExistsError(destination if destination.exists() else staging)
        staging.mkdir(parents=True)
        state = self.paper.state_dict()
        # Real official FSDP actor/optimizer/RNG shards and full HF export.
        self.actor_rollout_wg.save_checkpoint(str(staging / "actor"), None, self.global_steps, max_ckpt_to_keep=None)
        torch.save(state, staging / "rl_lhm_state.pt")
        metadata = dict(format_version=1, backend="verl", upstream_commit=UPSTREAM_COMMIT,
                        identity=self.identity, completed_epochs=self.paper.completed_epochs,
                        global_steps=self.global_steps, world_size=4, source_revision=source_revision())
        (staging / "checkpoint.json").write_text(json.dumps(metadata, indent=2))
        (staging / "training_config.json").write_text(json.dumps(settings, indent=2))
        (staging / "training_history.json").write_text(json.dumps(self.history, indent=2))
        (staging / "verl_config.json").write_text(json.dumps(OmegaConf.to_container(self.config, resolve=True), indent=2))
        staging.rename(destination)
        # Serialization must not consume the driver continuation streams.
        self.paper.load_state_dict(state)

    def _restore_epoch(self):
        path = self.paper.configuration["resume_from"]
        if path is None:
            return
        path = Path(path)
        metadata = json.loads((path / "checkpoint.json").read_text())
        if (metadata["format_version"] != 1 or metadata["backend"] != "verl"
                or metadata["upstream_commit"] != UPSTREAM_COMMIT or metadata["world_size"] != 4
                or metadata["identity"] != self.identity):
            raise ValueError("VeRL checkpoint source, data, settings or carrier does not match")
        state = torch.load(path / "rl_lhm_state.pt", map_location="cpu", weights_only=True)
        if int(state["completed_epochs"]) != int(metadata["completed_epochs"]):
            raise ValueError("incomplete or inconsistent epoch checkpoint")
        self.actor_rollout_wg.load_checkpoint(str(path / "actor"), hdfs_path=None, del_local_after_load=False)
        self.paper.load_state_dict(state)
        self.global_steps = int(metadata["global_steps"])
        self.history = json.loads((path / "training_history.json").read_text())

    def fit(self):
        self._restore_epoch()
        self.checkpoint_manager.update_weights(self.global_steps)
        SkipManager.init(self.config)
        settings = self.paper.configuration
        output = Path(settings["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.paper.completed_epochs, settings["loop"]["epochs"]):
            destination = output / f"epoch_{epoch + 1:04d}"
            if destination.exists() or destination.with_name(destination.name + ".incomplete").exists():
                raise FileExistsError(destination)
        try:
            for epoch in range(self.paper.completed_epochs, settings["loop"]["epochs"]):
                order = self.paper.epoch_order(epoch)
                batch_size = settings["loop"]["batch_size"]
                for offset in range(0, len(order), batch_size):
                    SkipManager.set_step(self.global_steps + 1)
                    report = update_once(self, self.paper, self.train_dataset,
                                         order[offset:offset + batch_size], epoch=epoch, offset=offset)
                    self.global_steps += 1
                    self.history.append(report)
                    if offset + batch_size >= len(order):
                        self.paper.completed_epochs = epoch + 1
                        self._save_epoch()
                    # Synchronize the newly updated actor before another rollout.
                    self.checkpoint_manager.update_weights(self.global_steps)
                    print(json.dumps({**report, "backend": "verl", "global_step": self.global_steps}))
        finally:
            self._shutdown_dump_executor()
        return self.history


@ray.remote
class PaperTaskRunner(BaseTaskRunner):
    def run(self, config):
        verify_upstream()
        settings = OmegaConf.to_container(config.rl_lhm, resolve=True)
        random.seed(settings["seed"])
        np.random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        manifest = load_manifest(settings["manifest"])
        identity = training_identity(settings, {**manifest.payload(), "source": source_identity(settings["model"])})
        paper = PaperState(settings, manifest)
        _, worker_group = self.add_actor_rollout_worker(config)
        validate_config(config, use_reference_policy=True, use_critic=False)
        model_config = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        if model_config.processor is None:
            raise ValueError("VeRL requires the post-SFT multimodal processor")
        if model_config.tokenizer.pad_token_id != settings["generation"]["pad_token_id"]:
            raise ValueError("processor padding token does not match the training recipe")
        generation = DecodingConfig(**settings["generation"])
        report = length_summary(target_lengths(model_config.processor, manifest.records,
                                               manifest.output_classes, generation), generation.max_new_tokens)
        output = Path(settings["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "target_lengths.json").write_text(json.dumps(report, indent=2))
        if report["over_budget_groups"]:
            raise ValueError("ground-truth targets exceed response cap; see target_lengths.json")
        dataset = ManifestDataset(manifest.records, settings["model"]["processor_kwargs"])
        trainer = PaperVeRLTrainer(
            paper=paper, identity=identity, config=config,
            tokenizer=model_config.tokenizer, processor=model_config.processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.init_resource_pool_mgr(config), ray_worker_group_cls=worker_group,
            train_dataset=dataset, val_dataset=dataset,
        )
        trainer.init_workers()
        return trainer.fit()
