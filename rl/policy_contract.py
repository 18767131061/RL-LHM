"""Explicit VeRL policy configuration and supported source-contract checks."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path

REFERENCE_COMMIT = "fc3a5acae155425909bc30c982c3cea0784ce3c2"
POLICY_OVERRIDES = {
    "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
    "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
    "actor_rollout_ref.actor.clip_ratio": "0.20",
    "actor_rollout_ref.actor.clip_ratio_low": "0.20",
    "actor_rollout_ref.actor.clip_ratio_high": "0.20",
    "actor_rollout_ref.actor.clip_ratio_c": "3.0",
    "actor_rollout_ref.actor.use_kl_loss": "True",
    "actor_rollout_ref.actor.kl_loss_coef": "0.01",
    "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
    "actor_rollout_ref.actor.entropy_coeff": "0",
    "actor_rollout_ref.actor.grad_clip": "1.0",
    "actor_rollout_ref.actor.ppo_epochs": "1",
    "actor_rollout_ref.actor.use_dynamic_bsz": "False",
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size": "1",
    "actor_rollout_ref.actor.strategy": "fsdp",
    "algorithm.use_kl_in_reward": "False",
    "algorithm.rollout_correction": "null",
    "trainer.use_legacy_worker_impl": "enable",
}
FUNCTION_HASHES = {
    "agg_loss": "52fd3c3fdc79184a40862f96f4e35160b7cd1c305dbba5916ab51c3cde442e90",
    "compute_policy_loss_vanilla": "9f919f3c96a7a9dfc978faa9b706a8e9327549e2758222339f2ad58e647ae423",
    "kl_penalty": "e2c69b1414fe3b5f57230ebc78d8b8ca1c53200c0cf8a13ece69bd66c152e170",
    "kl_penalty_forward": "acef8c2d2e49688f9d81746d9cae713bf2b8ce38b39c61bc90bb80d553f340f6",
}


def function_source(source, name):
    nodes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"Expected one {name} function")
    n = nodes[0]
    return "".join(source.splitlines(keepends=True)[n.lineno - 1:n.end_lineno])


def verify_core(path: Path):
    source = path.read_text(encoding="utf-8")
    for name, expected in FUNCTION_HASHES.items():
        digest = hashlib.sha256(function_source(source, name).encode()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Unsupported VeRL {name}; expected reference {REFERENCE_COMMIT}")
    return {"reference_commit": REFERENCE_COMMIT, "core_algos_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "functions": FUNCTION_HASHES}


def verify_overrides(arguments):
    parsed = {}
    for argument in arguments:
        if "=" in argument:
            key, value = argument.split("=", 1)
            key = key.lstrip("+")
            if key in parsed:
                raise RuntimeError(f"Duplicate configuration override: {key}")
            parsed[key] = value
    for key, value in POLICY_OVERRIDES.items():
        if parsed.get(key, "").lower() != value.lower():
            raise RuntimeError(f"Required policy override {key}={value}")
    return {key: parsed[key] for key in POLICY_OVERRIDES}


def global_token_batch_info(response_mask, device):
    """Global PPO minibatch denominator, with DP gradient averaging compensation.

    Supported runtime is FSDP, one node and no sequence parallelism. Each rank
    contributes its local minibatch once. All microbatches share this denominator.
    """
    import torch
    mask = response_mask.detach()
    if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("response_mask must be finite binary values")
    count = mask.sum(dtype=torch.float64).to(device)
    dp_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dp_size = torch.distributed.get_world_size()
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
    if count.item() <= 0:
        raise ValueError("PPO minibatch contains no retained generated tokens")
    return {"dp_size": dp_size, "batch_num_tokens": int(count.item())}
