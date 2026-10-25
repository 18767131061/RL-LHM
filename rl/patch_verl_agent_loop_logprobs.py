#!/usr/bin/env python3
"""Idempotently expose rollout log probabilities to VERL's async reward loop."""

# Source-context strings target the Apache-2.0-licensed VeRL agent loop:
# https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/verl/experimental/agent_loop/agent_loop.py
# Upstream license: https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/LICENSE
# Applying this helper does not change the upstream checkout's license.

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path


OLD = '''            batch = TensorDict(
                {
                    "prompts": prompts,  # [1, prompt_length]
                    "responses": responses,  # [1, response_length]
                    "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                    "input_ids": input_ids,  # [1, prompt_length + response_length]
                    "position_ids": position_ids,
                },
                batch_size=1,
            )'''

LEGACY_NEW = '''            reward_batch_fields = {
                "prompts": prompts,  # [1, prompt_length]
                "responses": responses,  # [1, response_length]
                "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                "input_ids": input_ids,  # [1, prompt_length + response_length]
                "position_ids": position_ids,
            }
            if output.response_logprobs is not None:
                reward_batch_fields["rollout_log_probs"] = output.response_logprobs
            batch = TensorDict(reward_batch_fields, batch_size=1)'''

PADDED_NEW = '''            reward_batch_fields = {
                "prompts": prompts,  # [1, prompt_length]
                "responses": responses,  # [1, response_length]
                "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                "input_ids": input_ids,  # [1, prompt_length + response_length]
                "position_ids": position_ids,
            }
            if output.response_logprobs is not None:
                rollout_log_probs = torch.as_tensor(output.response_logprobs, dtype=torch.float32).reshape(1, -1)
                response_length = responses.shape[-1]
                if rollout_log_probs.shape[-1] > response_length:
                    rollout_log_probs = rollout_log_probs[:, :response_length]
                elif rollout_log_probs.shape[-1] < response_length:
                    rollout_log_probs = torch.nn.functional.pad(
                        rollout_log_probs, (0, response_length - rollout_log_probs.shape[-1])
                    )
                reward_batch_fields["rollout_log_probs"] = rollout_log_probs
            batch = TensorDict(reward_batch_fields, batch_size=1)'''


NEW = '''            reward_batch_fields = {
                "prompts": prompts,  # [1, prompt_length]
                "responses": responses,  # [1, response_length]
                "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                "input_ids": input_ids,  # [1, prompt_length + response_length]
                "position_ids": position_ids,
            }
            # RL-LHM logprob bridge v3: padding cannot replace missing token scores.
            if output.response_logprobs is None:
                raise RuntimeError("RL-LHM requires generated-token rollout log probabilities")
            if responses.ndim != 2 or responses.shape[0] != 1 or responses.shape[-1] <= 0:
                raise RuntimeError("RL-LHM reward bridge expects responses with shape [1, L]")
            response_length = responses.shape[-1]
            if attention_mask.ndim != 2 or attention_mask.shape[0] != 1 or attention_mask.shape[-1] < response_length:
                raise RuntimeError("attention_mask does not contain the generated response span")
            if not torch.isfinite(attention_mask).all() or not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise RuntimeError("attention_mask must be finite and binary")
            response_mask = attention_mask[0, -response_length:].to(device=responses.device, dtype=torch.bool)
            valid_count = int(response_mask.sum().item())
            if valid_count <= 0:
                raise RuntimeError("Generated response has no valid tokens")
            source_log_probs = torch.as_tensor(output.response_logprobs, dtype=torch.float32, device=responses.device)
            if source_log_probs.ndim == 2 and source_log_probs.shape[0] == 1:
                source_log_probs = source_log_probs.squeeze(0)
            if source_log_probs.ndim != 1:
                raise RuntimeError("rollout log probabilities must have shape [L] or [1, L]")
            if source_log_probs.numel() == response_length:
                valid_log_probs = source_log_probs[response_mask]
            elif source_log_probs.numel() == valid_count:
                valid_log_probs = source_log_probs
            else:
                raise RuntimeError("rollout log-probability length must equal the padded or valid response length")
            if not torch.isfinite(valid_log_probs).all():
                raise RuntimeError("Every valid generated token requires a finite rollout log probability")
            rollout_log_probs = torch.zeros_like(responses, dtype=torch.float32)
            rollout_log_probs[0, response_mask] = valid_log_probs
            reward_batch_fields["rollout_log_probs"] = rollout_log_probs
            batch = TensorDict(reward_batch_fields, batch_size=1)'''


def patch_source(source: str) -> tuple[str, bool, str]:
    """Replace exactly one known executable block, including v1/v2 upgrades.

    Exact standalone snippets are supported for the patch contract tests. An
    occurrence inside a comment/string, duplicate blocks, altered old patches,
    or invalid Python is never accepted as a source match.
    """
    variants = (("original", OLD), ("legacy-v1", LEGACY_NEW),
                ("legacy-v2", PADDED_NEW), ("current-v3", NEW))
    matches = [(name, block) for name, block in variants for _ in range(source.count(block))]
    if len(matches) != 1:
        raise RuntimeError("VERL agent-loop source must contain exactly one known reward bridge block")
    name, block = matches[0]
    try:
        normalized = textwrap.dedent(source)
        tree = ast.parse(normalized)
    except SyntaxError as error:
        raise RuntimeError("VERL agent-loop source is not valid Python") from error
    position = source.index(block)
    line = source.count("\n", 0, position) + 1
    indentation = len(block) - len(block.lstrip())
    removed = len(source.splitlines()[line - 1]) - len(normalized.splitlines()[line - 1])
    column = position - (source.rfind("\n", 0, position) + 1) + indentation - removed
    variable = "batch" if name == "original" else "reward_batch_fields"
    anchors = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and node.lineno == line and node.col_offset == column and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == variable]
    if len(anchors) != 1:
        raise RuntimeError("Known reward bridge text is not an executable assignment")
    updated = source.replace(block, NEW, 1)
    try:
        ast.parse(textwrap.dedent(updated))
    except SyntaxError as error:
        raise RuntimeError("Patched VERL agent-loop source would not be valid Python") from error
    return updated, name == "current-v3", name


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-loop", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    target = Path(args.agent_loop).resolve()
    audit_path = Path(args.audit).resolve()
    before = sha256(target)
    text, already_applied, source_variant = patch_source(target.read_text(encoding="utf-8"))
    if not already_applied:
        backup = target.with_suffix(target.suffix + ".pre_rl_lhm")
        if not backup.exists():
            shutil.copy2(target, backup)
        target.write_text(text, encoding="utf-8")
    after = sha256(target)
    audit = {
        "status": "passed",
        "target": str(target),
        "already_applied": already_applied,
        "sha256_before": before,
        "sha256_after": after,
        "rollout_log_probs_in_reward_batch": NEW in target.read_text(encoding="utf-8"),
        "patch_version": 3,
        "source_variant": source_variant,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
