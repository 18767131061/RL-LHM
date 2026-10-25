"""Single-process supervised and grouped reinforcement-learning loops."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch

from rewards.full_reward import RewardConfig, compute_score
from rl import (
    ShaeBaselineStore, VGSSampler, compute_grpo_advantage,
    compute_shae_advantage, grpo_token_loss, update_curriculum,
)


class ModelBackend(Protocol):
    model: torch.nn.Module

    def sft_loss_sum(self, records: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, int]: ...
    def generate(self, records: Sequence[Mapping[str, Any]]) -> Any: ...
    def log_probabilities(self, rollout: Any) -> torch.Tensor: ...


@dataclass(frozen=True)
class ComponentSwitches:
    shae: bool
    rars: bool
    curriculum: bool

    def __post_init__(self):
        if any(type(value) is not bool for value in (self.shae, self.rars, self.curriculum)):
            raise ValueError("component switches must be booleans")


@dataclass(frozen=True)
class ShaeParameters:
    gate_mid: float
    gate_temperature: float
    noise_std: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.gate_mid, self.gate_temperature, self.noise_std)):
            raise ValueError("SHAE settings must be finite")
        if self.gate_temperature <= 0 or self.noise_std < 0:
            raise ValueError("SHAE temperature must be positive and noise scale nonnegative")


@dataclass(frozen=True)
class PolicyParameters:
    clip_ratio: float
    dual_clip: float
    kl_coefficient: float
    log_ratio_bound: float
    kl_bound: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (
            self.clip_ratio, self.dual_clip, self.kl_coefficient,
            self.log_ratio_bound, self.kl_bound,
        )):
            raise ValueError("policy settings must be finite")
        if not 0 <= self.clip_ratio < 1 or self.dual_clip <= 1 or self.kl_coefficient < 0:
            raise ValueError("invalid policy clipping or KL coefficient")
        if self.log_ratio_bound <= 0 or self.kl_bound <= 0:
            raise ValueError("policy numerical bounds must be positive")


def _positive_integer(name: str, value: int):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _records(records: Sequence[Mapping[str, Any]]):
    if not records:
        raise ValueError("training records must be nonempty")
    uids = [record["uid"] for record in records]
    if any(not isinstance(uid, str) or not uid.strip() for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError("training records must have unique nonempty string UIDs")


def _backward(loss):
    if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss):
        raise ValueError("training loss must be a finite differentiable scalar")
    loss.backward()


def _optimizer_step(backend, optimizer, max_grad_norm, denominator):
    for parameter in backend.model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(denominator)
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in backend.model.parameters() if parameter.requires_grad],
        max_grad_norm, error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach())


def _loop_settings(epochs, batch_size, max_grad_norm):
    _positive_integer("epochs", epochs)
    _positive_integer("batch_size", batch_size)
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")


def fit_sft(
    backend: ModelBackend,
    records: Sequence[Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    start_epoch: int,
    batch_size: int,
    microbatch_size: int,
    max_grad_norm: float,
    generator: torch.Generator,
    on_epoch: Callable[[int, list[dict[str, float | int]]], None] | None,
) -> list[dict[str, float | int]]:
    """Fit assistant completion tokens, shuffling input records each epoch.

    The backend constructs the multimodal inputs and completion-only targets.
    ``batch_size`` is the optimizer's input batch; ``microbatch_size`` is the
    number of inputs per forward pass. Gradients are accumulated from token-NLL
    sums and divided by the full batch's retained-token count before the step.
    An epoch callback can save the model and metrics after its final update.
    """
    _records(records)
    _loop_settings(epochs, batch_size, max_grad_norm)
    _positive_integer("microbatch_size", microbatch_size)
    if type(start_epoch) is not int or not 0 <= start_epoch <= epochs:
        raise ValueError("start_epoch must be an integer between zero and epochs")
    if generator.device.type != "cpu":
        raise ValueError("the dataset-order generator must use CPU")
    history = []
    for epoch in range(start_epoch, epochs):
        backend.model.train()
        order = torch.randperm(len(records), generator=generator).tolist()
        for step, start in enumerate(range(0, len(order), batch_size)):
            batch = [records[i] for i in order[start:start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss_sum, token_count = 0.0, 0
            for offset in range(0, len(batch), microbatch_size):
                total, count = backend.sft_loss_sum(batch[offset:offset + microbatch_size])
                _positive_integer("retained target tokens", count)
                _backward(total)
                loss_sum += float(total.detach())
                token_count += count
            norm = _optimizer_step(backend, optimizer, max_grad_norm, token_count)
            history.append({
                "epoch": epoch, "step": step, "loss": loss_sum / token_count,
                "gradient_norm": norm, "input_count": len(batch),
                "retained_tokens": token_count,
            })
        if on_epoch is not None:
            on_epoch(epoch, history)
    return history


def fit_rl(
    backend: ModelBackend,
    reference: ModelBackend,
    records: Sequence[Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    start_epoch: int,
    batch_size: int,
    microbatch_size: int,
    group_size: int,
    policy_updates: int,
    max_grad_norm: float,
    advantage_epsilon: float,
    switches: ComponentSwitches,
    policy: PolicyParameters,
    reward_config: RewardConfig,
    class_counts: Mapping[str, int],
    baseline_store: ShaeBaselineStore | None,
    shae_parameters: ShaeParameters | None,
    curriculum_sampler: VGSSampler | None,
    visual_vectors: Mapping[str, torch.Tensor | None],
    order_generator: torch.Generator,
    shae_generator: torch.Generator | None,
    on_epoch: Callable[[int, list[dict[str, float | int]]], None] | None,
) -> list[dict[str, float | int]]:
    """Optimize all eight SHAE/RARS/curriculum component combinations.

    Each sampled occurrence has its own RID and G completions. The old-policy
    token log probabilities are fixed throughout its policy updates. The
    post-SFT reference stays frozen. Evaluation mode is used for generation,
    old/reference scoring, and differentiable current-policy scoring so dropout
    does not change the probability model between these operations.

    SHAE updates its UID baseline once when advantages are constructed. After
    all policy updates for the batch, each occurrence's mean clipped reward
    updates DA-CVQ. The next epoch's VGS draws use this separate visual memory.
    ``batch_size`` counts sampled input occurrences; ``microbatch_size`` counts
    completions per generation/forward pass. All policy microbatches use the
    same full-update retained-token denominator and one optimizer step.
    """
    _records(records)
    _loop_settings(epochs, batch_size, max_grad_norm)
    _positive_integer("microbatch_size", microbatch_size)
    if type(start_epoch) is not int or not 0 <= start_epoch <= epochs:
        raise ValueError("start_epoch must be an integer between zero and epochs")
    _positive_integer("group_size", group_size)
    _positive_integer("policy_updates", policy_updates)
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    if not math.isfinite(advantage_epsilon) or advantage_epsilon <= 0:
        raise ValueError("advantage_epsilon must be finite and positive")
    if order_generator.device.type != "cpu":
        raise ValueError("the dataset-order generator must use CPU")
    if switches.shae and (baseline_store is None or shae_parameters is None):
        raise ValueError("SHAE requires its baseline store and parameters")
    if switches.curriculum != (curriculum_sampler is not None):
        raise ValueError("provide a curriculum sampler exactly when curriculum is enabled")
    if switches.curriculum:
        if not curriculum_sampler.use_vgs or len(curriculum_sampler.vectors) != len(records):
            raise ValueError("the enabled VGS sampler must align with the training records")
        if curriculum_sampler.epoch_count != start_epoch:
            raise ValueError("the VGS sampler epoch must equal start_epoch")
        for record, vector in zip(records, curriculum_sampler.vectors):
            uid = record["uid"]
            if uid not in visual_vectors:
                raise ValueError("each training UID needs an explicit visual vector or None")
            expected = visual_vectors[uid]
            if (vector is None) != (expected is None):
                raise ValueError("sampler and curriculum-update vectors must have identical UID alignment")
            if vector is not None and not torch.allclose(vector, curriculum_sampler.state.normalize(expected)):
                raise ValueError("sampler and curriculum-update vectors must agree")
        state = curriculum_sampler.state
        if state.reward_min != reward_config.clip_min or state.reward_max != reward_config.clip_max:
            raise ValueError("reward and curriculum clipping bounds must agree")
    if backend.model is reference.model:
        raise ValueError("current and reference policies must be distinct models")
    current_storage = {(p.device, p.data_ptr()) for p in backend.model.parameters()}
    if any((p.device, p.data_ptr()) in current_storage for p in reference.model.parameters()):
        raise ValueError("current and reference parameters must not share storage")
    reference_parameters = {id(p) for p in reference.model.parameters()}
    if any(id(p) in reference_parameters for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("optimizer must not contain reference-policy parameters")
    reference.model.requires_grad_(False)
    reference.model.eval()
    config = replace(reward_config, rars_enabled=switches.rars)
    history = []
    for epoch in range(start_epoch, epochs):
        if curriculum_sampler is not None:
            order = curriculum_sampler.sample_epoch().tolist()
        else:
            order = torch.randperm(len(records), generator=order_generator).tolist()
        for step, start in enumerate(range(0, len(order), batch_size)):
            sampled = [records[i] for i in order[start:start + batch_size]]
            expanded, rollout_ids, input_uids = [], [], []
            for occurrence, record in enumerate(sampled):
                expanded.extend([record] * group_size)
                rollout_ids.extend([f"{epoch}:{step}:{occurrence}"] * group_size)
                input_uids.extend([record["uid"]] * group_size)
            backend.model.eval()
            chunks, scores, masks = [], [], []
            with torch.no_grad():
                for offset in range(0, len(expanded), microbatch_size):
                    inputs = expanded[offset:offset + microbatch_size]
                    rollout = backend.generate(inputs)
                    old = rollout.old_log_probabilities.detach()
                    mask = rollout.response_mask
                    if len(rollout.texts) != len(inputs) or len(rollout.completion_nlls) != len(inputs):
                        raise ValueError("generation must return one completion per expanded record")
                    if old.shape != mask.shape or old.ndim != 2 or old.shape[0] != len(inputs):
                        raise ValueError("generated token probabilities and masks must align with completions")
                    if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any(dim=1).all():
                        raise ValueError("each completion must retain generated tokens in a binary mask")
                    scores.extend(
                        compute_score(
                            text, record["ground_truth"], completion_nll=float(nll),
                            class_counts=class_counts, config=config,
                        )["score"]
                        for record, text, nll in zip(inputs, rollout.texts, rollout.completion_nlls)
                    )
                    reference_logp = reference.log_probabilities(rollout).detach()
                    chunks.append((offset, rollout, old, reference_logp))
                    masks.extend(mask.unbind(0))
                mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
                rewards = torch.tensor(scores, dtype=torch.float32, device=mask.device)
                if switches.shae:
                    advantages = compute_shae_advantage(
                        rewards, mask, rollout_ids, sample_index=input_uids,
                        store=baseline_store, gate_mid=shae_parameters.gate_mid,
                        gate_temperature=shae_parameters.gate_temperature,
                        noise_std=shae_parameters.noise_std,
                        epsilon=advantage_epsilon, generator=shae_generator,
                    )
                else:
                    advantages = compute_grpo_advantage(
                        rewards, mask, rollout_ids, epsilon=advantage_epsilon,
                    )
            count = int(mask.sum().item())
            for update in range(policy_updates):
                backend.model.eval()
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for offset, rollout, old, reference_logp in chunks:
                    current = backend.log_probabilities(rollout)
                    chunk_advantage = advantages[offset:offset + old.shape[0], :old.shape[1]]
                    loss = grpo_token_loss(
                        current, old, reference_logp, chunk_advantage, rollout.response_mask,
                        clip_ratio=policy.clip_ratio, dual_clip=policy.dual_clip,
                        kl_coefficient=policy.kl_coefficient,
                        log_ratio_bound=policy.log_ratio_bound, kl_bound=policy.kl_bound,
                        total_token_count=count,
                    )
                    _backward(loss)
                    total_loss += float(loss.detach())
                norm = _optimizer_step(backend, optimizer, max_grad_norm, 1)
                history.append({
                    "epoch": epoch, "step": step, "policy_update": update,
                    "loss": total_loss, "gradient_norm": norm,
                    "mean_reward": float(rewards.mean()),
                    "input_count": len(sampled), "completion_count": len(expanded),
                    "retained_tokens": count,
                })
            if curriculum_sampler is not None:
                update_curriculum(
                    curriculum_sampler.state, rewards, rollout_ids,
                    input_uids, visual_vectors,
                )
        if on_epoch is not None:
            on_epoch(epoch, history)
    return history
