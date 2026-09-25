#!/usr/bin/env python3
"""VERL reward manager that injects generated-token completion likelihood."""

from __future__ import annotations

import inspect
import math

import torch

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


def completion_likelihood(log_probs: torch.Tensor, valid_length: int) -> tuple[float, float]:
    values = log_probs.reshape(-1)[:valid_length].detach().float()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise RuntimeError("Valid generated-token rollout log probabilities are required")
    nll = float((-values.mean()).item())
    if not math.isfinite(nll):
        raise RuntimeError(f"Non-finite completion NLL: {nll}")
    confidence = float(math.exp(-max(0.0, nll)))
    return nll, max(0.0, min(1.0, confidence))


def _single_sequence(tensor: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(tensor)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 1:
        raise RuntimeError(f"{name} must have shape [L] or [1,L], got {tuple(value.shape)}")
    return value


def extract_generated_tokens_and_log_probs(batch) -> tuple[torch.Tensor, torch.Tensor]:
    response_ids = _single_sequence(batch["responses"], "responses")
    attention_mask = _single_sequence(batch["attention_mask"], "attention_mask")
    response_length = int(response_ids.numel())
    if response_length <= 0 or attention_mask.numel() < response_length:
        raise RuntimeError("attention_mask does not contain the generated response span")
    response_mask = attention_mask[-response_length:].to(dtype=torch.bool)
    valid_response_ids = response_ids[response_mask]
    if valid_response_ids.numel() <= 0:
        raise RuntimeError("Generated response has no valid tokens")

    rollout_log_probs = _single_sequence(batch["rollout_log_probs"], "rollout_log_probs")
    if rollout_log_probs.numel() == response_length:
        valid_log_probs = rollout_log_probs[response_mask]
    elif rollout_log_probs.numel() == valid_response_ids.numel():
        valid_log_probs = rollout_log_probs
    else:
        raise RuntimeError(
            "rollout_log_probs length must match the padded or valid response length: "
            f"{rollout_log_probs.numel()} not in ({response_length}, {valid_response_ids.numel()})"
        )
    return valid_response_ids, valid_log_probs


class CompletionLikelihoodRewardManager(RewardManagerBase):
    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data) -> dict:
        if len(data) != 1:
            raise RuntimeError("CompletionLikelihoodRewardManager expects one item")
        data_item = data[0]
        if "rollout_log_probs" not in data_item.batch:
            raise RuntimeError(
                "rollout_log_probs missing; enable calculate_log_probs and apply the RL-LHM agent-loop patch"
            )
        valid_response_ids, valid_log_probs = extract_generated_tokens_and_log_probs(data_item.batch)
        completion_nll, seq_confidence = completion_likelihood(
            valid_log_probs, int(valid_log_probs.numel())
        )
        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields")
        if tool_extra_fields is not None:
            extra_info.update(dict(tool_extra_fields))
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__")
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["completion_nll"] = completion_nll
        extra_info["seq_confidence"] = seq_confidence

        kwargs = {}
        if self.reward_router_address is not None:
            kwargs = {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **kwargs,
                ),
            )

        reward_extra_info = {}
        if isinstance(result, dict):
            score = float(result["score"])
            reward_extra_info.update(result)
        else:
            score = float(result)
            reward_extra_info["score"] = score
        reward_extra_info["completion_nll"] = completion_nll
        reward_extra_info["seq_confidence"] = seq_confidence
        return {"reward_score": score, "reward_extra_info": reward_extra_info}
