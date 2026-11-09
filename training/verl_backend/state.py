"""Driver-owned rewards, persistent UID state and epoch sampling for VeRL."""
from __future__ import annotations

import torch

from data.manifest import training_class_counts
from rewards import RewardConfig, RARSParameters, compute_score
from rl import DACVQState, ShaeBaselineStore, VGSSampler
from rl.core import compute_grpo_advantage, compute_shae_advantage
from rl.workflow import update_curriculum
from training.checkpoint import capture_state, restore_state


def completion_nll(old_log_probs, response_mask):
    """Original-model generated-token NLL, including the retained first EOS."""
    if old_log_probs.shape != response_mask.shape or old_log_probs.ndim != 2:
        raise ValueError("old log probabilities and response mask must be [completions, tokens]")
    mask = response_mask.bool()
    counts = mask.sum(-1)
    if not torch.all((response_mask == 0) | (response_mask == 1)) or torch.any(counts == 0):
        raise ValueError("every generated completion must retain at least one token")
    if not torch.isfinite(old_log_probs[mask]).all():
        raise ValueError("retained old log probabilities must be finite")
    return -torch.where(mask, old_log_probs.double(), 0).sum(-1) / counts


class PaperState:
    """One owner on the Ray driver; workers never update UID or CVQ memory."""

    def __init__(self, configuration, manifest):
        self.configuration, self.manifest = configuration, manifest
        self.records = manifest.records
        if not self.records:
            raise ValueError("the training manifest is empty")
        self.completed_epochs = 0
        self.generators = {
            "order": torch.Generator().manual_seed(configuration["order_seed"]),
            "rollout": torch.Generator().manual_seed(configuration["verl"]["rollout_seed"]),
        }
        self.class_counts = training_class_counts(manifest)
        settings = dict(configuration["reward"])
        settings["rars_parameters"] = RARSParameters(**settings["rars_parameters"])
        self.reward_config = RewardConfig(classes=manifest.output_classes,
                                         rars_enabled=configuration["components"]["rars"], **settings)
        self.visual_vectors = {r["uid"]: None if r.get("visual_vector") is None else
                               torch.tensor(r["visual_vector"], dtype=torch.float64) for r in self.records}
        self.baseline = self.sampler = None
        if configuration["components"]["shae"]:
            self.baseline = ShaeBaselineStore(beta=configuration["shae"]["baseline_beta"])
            self.generators["shae"] = torch.Generator().manual_seed(configuration["shae"]["seed"])
        if configuration["components"]["curriculum"]:
            cfg = configuration["curriculum"]
            self.generators["curriculum"] = torch.Generator().manual_seed(cfg["state_seed"])
            self.generators["sampling"] = torch.Generator().manual_seed(cfg["sampling_seed"])
            memory = DACVQState(**cfg["state"], generator=self.generators["curriculum"])
            if (memory.reward_min, memory.reward_max) != (self.reward_config.clip_min, self.reward_config.clip_max):
                raise ValueError("curriculum and reward clipping bounds must match")
            self.sampler = VGSSampler(
                memory, [self.visual_vectors[r["uid"]] for r in self.records],
                generator=self.generators["sampling"], use_vgs=True, epoch_count=0,
                missing_feature_multiplier=cfg["missing_feature_multiplier"], weight_floor=cfg["weight_floor"],
                replacement=configuration["sampling"]["replacement"],
            )

    def epoch_order(self, epoch):
        if epoch != self.completed_epochs:
            raise ValueError("sampling must start at the next uncompleted epoch")
        if self.sampler is not None:
            if self.sampler.epoch_count != epoch:
                raise ValueError("curriculum sampler epoch does not match the checkpoint")
            return self.sampler.sample_epoch().tolist()
        return torch.randperm(len(self.records), generator=self.generators["order"]).tolist()

    def sampling_seeds(self, count):
        # Explicit request seeds permit epoch resume without relying on an
        # inaccessible vLLM server RNG or its asynchronous scheduling order.
        return torch.randint(0, 2**31 - 1, (count,), generator=self.generators["rollout"]).tolist()

    def score_and_advantage(self, *, texts, old_log_probs, response_mask, records, rids, real):
        nlls = completion_nll(old_log_probs, response_mask)
        if not (len(texts) == len(records) == len(rids) == len(real) == len(nlls)):
            raise ValueError("completion metadata must align with the generated rows")
        positions = [i for i, keep in enumerate(real) if keep]
        if not positions:
            raise ValueError("a global update must contain real completions")
        rewards = torch.tensor([
            compute_score(texts[i], records[i]["ground_truth"], completion_nll=float(nlls[i]),
                          class_counts=self.class_counts, config=self.reward_config)["score"]
            for i in positions
        ], dtype=torch.float32)
        uids = [records[i]["uid"] for i in positions]
        real_rids = [rids[i] for i in positions]
        masks = response_mask[positions].to(torch.float32).cpu()
        if self.baseline is not None:
            cfg = self.configuration["shae"]
            advantages = compute_shae_advantage(
                rewards, masks, real_rids, sample_index=uids, store=self.baseline,
                gate_mid=cfg["gate_mid"], gate_temperature=cfg["gate_temperature"], noise_std=cfg["noise_std"],
                epsilon=self.configuration["loop"]["advantage_epsilon"], generator=self.generators["shae"],
            )
        else:
            advantages = compute_grpo_advantage(rewards, masks, real_rids,
                                               epsilon=self.configuration["loop"]["advantage_epsilon"])
        full_advantages = torch.zeros_like(response_mask, dtype=torch.float32)
        full_advantages[positions] = advantages
        loss_mask = response_mask.clone()
        loss_mask[torch.tensor([not value for value in real], dtype=torch.bool)] = 0
        return dict(advantages=full_advantages, response_mask=loss_mask, rewards=rewards,
                    rids=real_rids, uids=uids, nlls=nlls[positions], retained_tokens=int(loss_mask.sum()))

    def after_policy_update(self, scored):
        if self.sampler is not None:
            update_curriculum(self.sampler.state, scored["rewards"], scored["rids"],
                              scored["uids"], self.visual_vectors)

    def state_dict(self):
        return {"completed_epochs": self.completed_epochs,
                "streams": capture_state(self.generators, self.baseline, self.sampler)}

    def load_state_dict(self, state):
        restore_state(state["streams"], self.generators, self.baseline, self.sampler)
        self.completed_epochs = int(state["completed_epochs"])
        if self.sampler is not None and self.sampler.epoch_count != self.completed_epochs:
            raise ValueError("only completed-epoch curriculum checkpoints can be resumed")
