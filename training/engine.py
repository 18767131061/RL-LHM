"""Globally normalized SFT and grouped RL with equal distributed work slots."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from rewards.full_reward import RewardConfig, compute_score
from rl import (
    ShaeBaselineStore, VGSSampler, compute_grpo_advantage,
    compute_shae_advantage, grpo_token_loss, update_curriculum,
)


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
        if self.gate_temperature <= 0 or self.noise_std != 0:
            raise ValueError("SHAE temperature must be positive and noise scale zero")


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


def _loop_settings(epochs, start_epoch, batch_size, microbatch_size, max_grad_norm):
    _positive_integer("epochs", epochs)
    _positive_integer("batch_size", batch_size)
    if microbatch_size != 1:
        raise ValueError("training microbatch_size must be 1; batch_size is the global accumulation batch")
    if type(start_epoch) is not int or not 0 <= start_epoch <= epochs:
        raise ValueError("start_epoch must lie between zero and epochs")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")


def fit_sft(backend, records, *, runtime, epochs, start_epoch, batch_size,
            microbatch_size, max_grad_norm, generator, on_epoch):
    """Accumulate token sums over the global batch, including unpadded tails."""
    _records(records)
    _loop_settings(epochs, start_epoch, batch_size, microbatch_size, max_grad_norm)
    history = []
    for epoch in range(start_epoch, epochs):
        order = torch.randperm(len(records), generator=generator).tolist()
        for step, start in enumerate(range(0, len(order), batch_size)):
            batch = [records[i] for i in order[start:start + batch_size]]
            prepared = [(backend.prepare_sft([record]), real)
                        for _, record, real in runtime.slots(batch)]
            count = int(runtime.sum(sum(int(mask.sum()) for (_, mask), real in prepared if real)))
            _positive_integer("global retained tokens", count)
            runtime.begin(len(prepared))
            backend.model.train()
            total_loss = 0.0
            for index, ((inputs, mask), real) in enumerate(prepared):
                runtime.boundary(index == len(prepared) - 1)
                total = backend.sft_prepared_loss_sum(inputs, mask) * int(real)
                runtime.backward(total / count)
                total_loss += float(total.detach())
            norm = runtime.step(max_grad_norm)
            history.append(dict(epoch=epoch, step=step, loss=runtime.sum(total_loss) / count,
                                gradient_norm=norm, input_count=len(batch), retained_tokens=count))
        if on_epoch is not None:
            on_epoch(epoch, history)
    return history


def _cpu_rollout(rollout):
    # Rollouts are reused across policy updates. Cache image tensors, masks and
    # old/reference scores in host memory instead of growing GPU residency.
    rollout.model_inputs = {key: value.cpu() for key, value in rollout.model_inputs.items()}
    rollout.response_mask = rollout.response_mask.cpu()
    rollout.old_log_probabilities = rollout.old_log_probabilities.cpu()
    return rollout


def fit_rl(backend, reference, records, *, runtime, epochs, start_epoch, batch_size,
           microbatch_size, group_size, policy_updates, max_grad_norm, advantage_epsilon,
           switches, policy, reward_config, class_counts, baseline_store, shae_parameters,
           curriculum_sampler, visual_vectors, order_generator, shae_generator, on_epoch,
           sampling_replacement):
    """Keep each G-completion occurrence on one rank; synchronize scalar rewards.

    Every worker processes gathered rewards in global occurrence order. Thus
    repeated UIDs update the same baseline and codebook, independent of their
    worker assignment. Padding slots participate in collectives but never enter
    rewards, state updates, metric populations or token denominators.
    """
    _records(records)
    _loop_settings(epochs, start_epoch, batch_size, microbatch_size, max_grad_norm)
    _positive_integer("policy_updates", policy_updates)
    if type(group_size) is not int or group_size < 2:
        raise ValueError("group_size must be at least two")
    if not math.isfinite(advantage_epsilon) or advantage_epsilon <= 0:
        raise ValueError("advantage_epsilon must be positive")
    if switches.shae and (baseline_store is None or shae_parameters is None):
        raise ValueError("SHAE requires a UID baseline and parameters")
    if switches.curriculum != (curriculum_sampler is not None):
        raise ValueError("curriculum switch and sampler must agree")
    if curriculum_sampler is not None:
        if curriculum_sampler.epoch_count != start_epoch:
            raise ValueError("sampler epoch must equal start_epoch")
        if len(curriculum_sampler.vectors) != len(records):
            raise ValueError("sampler vectors must align with training records")
        if (curriculum_sampler.state.reward_min != reward_config.clip_min
                or curriculum_sampler.state.reward_max != reward_config.clip_max):
            raise ValueError("curriculum and reward bounds must agree")
        for record, vector in zip(records, curriculum_sampler.vectors):
            expected = visual_vectors[record["uid"]]
            if ((vector is None) != (expected is None)
                    or vector is not None and not torch.allclose(vector, curriculum_sampler.state.normalize(expected))):
                raise ValueError("sampler and curriculum-update features must agree")
    if backend.model is reference.model:
        raise ValueError("reference must be a separate frozen model")
    reference.model.requires_grad_(False)
    reference.model.eval()
    config = replace(reward_config, rars_enabled=switches.rars)
    history = []
    for epoch in range(start_epoch, epochs):
        if curriculum_sampler is not None:
            order = curriculum_sampler.sample_epoch().tolist()
        elif sampling_replacement:
            order = torch.randint(len(records), (len(records),), generator=order_generator).tolist()
        else:
            order = torch.randperm(len(records), generator=order_generator).tolist()
        for step, start in enumerate(range(0, len(order), batch_size)):
            sampled = [records[i] for i in order[start:start + batch_size]]
            chunks, local_scores = [], []
            backend.model.eval()
            with torch.no_grad():
                for position, record, real in runtime.slots(sampled):
                    for member in range(group_size):
                        rollout = backend.generate([record])
                        score = compute_score(rollout.texts[0], record["ground_truth"],
                                              completion_nll=rollout.completion_nlls[0],
                                              class_counts=class_counts, config=config,
                                              finish_reason=rollout.finish_reasons[0])["score"]
                        ref = reference.log_probabilities(rollout).detach().cpu()
                        rollout = _cpu_rollout(rollout)
                        ordinal = position * group_size + member
                        chunks.append((ordinal, rollout, ref, real))
                        if real:
                            local_scores.append((ordinal, score, rollout.finish_reasons[0], rollout.completion_token_counts[0]))
            scored = sorted(item for shard in runtime.gather(local_scores) for item in shard)
            if [ordinal for ordinal, *_ in scored] != list(range(len(sampled) * group_size)):
                raise ValueError("distributed rollout population is not complete and unique")
            rewards = torch.tensor([item[1] for item in scored], dtype=torch.float32)
            rids = [f"{epoch}:{step}:{position}" for position in range(len(sampled)) for _ in range(group_size)]
            uids = [record["uid"] for record in sampled for _ in range(group_size)]
            mask = torch.ones(len(rewards), 1)
            if switches.shae:
                advantages = compute_shae_advantage(
                    rewards, mask, rids, sample_index=uids, store=baseline_store,
                    gate_mid=shae_parameters.gate_mid, gate_temperature=shae_parameters.gate_temperature,
                    noise_std=shae_parameters.noise_std, epsilon=advantage_epsilon, generator=shae_generator)
            else:
                advantages = compute_grpo_advantage(rewards, mask, rids, epsilon=advantage_epsilon)
            count = int(runtime.sum(sum(int(rollout.response_mask.sum()) for _, rollout, _, real in chunks if real)))
            _positive_integer("global retained tokens", count)
            for update in range(policy_updates):
                # Dropout is disabled in enable_training; train mode activates
                # non-reentrant gradient checkpointing without changing policy.
                backend.model.train()
                runtime.begin(len(chunks))
                total_loss = 0.0
                for index, (ordinal, rollout, reference_logp, real) in enumerate(chunks):
                    runtime.boundary(index == len(chunks) - 1)
                    current = backend.log_probabilities(rollout)
                    response_mask = rollout.response_mask.to(current.device)
                    advantage = advantages[ordinal if real else 0].to(current.device)
                    # Sum first: dummy completion length may exceed the entire
                    # real population on a short final batch.
                    tokens = int(response_mask.sum())
                    loss = grpo_token_loss(
                        current, rollout.old_log_probabilities.to(current.device),
                        reference_logp.to(current.device), advantage, response_mask,
                        clip_ratio=policy.clip_ratio, dual_clip=policy.dual_clip,
                        kl_coefficient=policy.kl_coefficient, log_ratio_bound=policy.log_ratio_bound,
                        kl_bound=policy.kl_bound, total_token_count=tokens) * (tokens * int(real) / count)
                    runtime.backward(loss)
                    total_loss += float(loss.detach())
                norm = runtime.step(max_grad_norm)
                history.append(dict(epoch=epoch, step=step, policy_update=update,
                                    loss=runtime.sum(total_loss), gradient_norm=norm,
                                    mean_reward=float(rewards.mean()), input_count=len(sampled),
                                    eos_completions=sum(item[2] == "eos" for item in scored),
                                    length_limited_completions=sum(item[2] == "length" for item in scored),
                                    truncation_rate=sum(item[2] == "length" for item in scored) / len(scored),
                                    completion_count=len(rewards), retained_tokens=count))
            if curriculum_sampler is not None:
                update_curriculum(curriculum_sampler.state, rewards, rids, uids, visual_vectors)
        if on_epoch is not None:
            on_epoch(epoch, history)
    return history
