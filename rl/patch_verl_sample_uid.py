#!/usr/bin/env python3
"""Pass persistent sample identity to SHAE without changing rollout group uid.

Targets compute_advantage in Apache-2.0-licensed VeRL at:
https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/verl/trainer/ppo/ray_trainer.py
Upstream license:
https://github.com/verl-project/verl/blob/fc3a5acae155425909bc30c982c3cea0784ce3c2/LICENSE
Only the exact reference function (or this patch's exact result) is accepted.
Other functions, including uid generation and policy losses, are unchanged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


REFERENCE_COMMIT = "fc3a5acae155425909bc30c982c3cea0784ce3c2"
REFERENCE_FUNCTION_SHA256 = "314695381c755264971a8f25865625bad4cf46c35d91901ba2a8f5e1ea980a18"
ANCHOR = "    elif adv_estimator == AdvantageEstimator.GRPO:\n"
ADDITION = '''    elif adv_estimator == "shae":
        # RL-LHM sample identity bridge v1. Import here in the Ray driver too.
        from verl_lhm_extensions import compute_shae_advantage, persistent_sample_keys

        advantages, returns = compute_shae_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            sample_index=persistent_sample_keys(
                data.non_tensor_batch, data.batch["token_level_rewards"].shape[0]
            ),
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
'''


def _function_source(source: str) -> tuple[list[str], int, int, str]:
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == "compute_advantage"]
    if len(nodes) != 1:
        raise RuntimeError("Expected exactly one top-level VeRL compute_advantage function")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    function = "".join(lines[node.lineno - 1:node.end_lineno])
    return lines, node.lineno - 1, node.end_lineno, function


def patch_source(source: str) -> tuple[str, bool]:
    lines, start, end, function = _function_source(source)
    already_applied = ADDITION in function
    original = function.replace(ADDITION, "", 1) if already_applied else function
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    if digest != REFERENCE_FUNCTION_SHA256:
        raise RuntimeError("VeRL compute_advantage does not match the supported reference; no guessed patch applied")
    if already_applied:
        return source, True
    if function.count(ANCHOR) != 1:
        raise RuntimeError("VeRL GRPO source anchor is not unique")
    lines[start:end] = [function.replace(ANCHOR, ADDITION + ANCHOR, 1)]
    return "".join(lines), False


def require_sample_uid_patch(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    _, already_applied = patch_source(source)
    if not already_applied:
        raise RuntimeError(
            "VeRL is missing the RL-LHM persistent sample UID bridge. Apply "
            "rl/patch_verl_sample_uid.py --ray-trainer <path/to/ray_trainer.py> "
            "--audit <output.json> before launching SHAE."
        )
    return {"reference_commit": REFERENCE_COMMIT, "patch_version": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-trainer", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    target = args.ray_trainer.resolve()
    original = target.read_text(encoding="utf-8")
    updated, already_applied = patch_source(original)
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    if not already_applied:
        backup = target.with_suffix(target.suffix + ".pre_rl_lhm_sample_uid")
        if not backup.exists():
            shutil.copy2(target, backup)
        target.write_text(updated, encoding="utf-8")
    audit = {"status": "passed", "target": str(target), "already_applied": already_applied,
             "sha256_before": before, "sha256_after": hashlib.sha256(target.read_bytes()).hexdigest(),
             **require_sample_uid_patch(target)}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
