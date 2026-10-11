#!/usr/bin/env python3
"""Idempotently expose rollout log probabilities to VERL's async reward loop."""

# Source-context strings target the Apache-2.0-licensed VeRL agent loop:
# https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/verl/experimental/agent_loop/agent_loop.py
# Upstream license: https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/LICENSE
# Applying this helper does not change the upstream checkout's license.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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

NEW = '''            reward_batch_fields = {
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
    text = target.read_text(encoding="utf-8")
    already_applied = NEW in text
    if not already_applied:
        if text.count(LEGACY_NEW) == 1:
            text = text.replace(LEGACY_NEW, NEW)
        elif text.count(OLD) == 1:
            text = text.replace(OLD, NEW)
        else:
            raise RuntimeError("VERL agent-loop source does not match the audited patch context")
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
        "patch_version": 2,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
