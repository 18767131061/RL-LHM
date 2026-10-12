#!/usr/bin/env python3
"""Patch pinned VeRL FSDP actor to use one global token denominator for PG and KL.

Upstream source/license (Apache-2.0): https://github.com/verl-project/verl/tree/
fc3a5acae155425909bc30c982c3cea0784ce3c2 . This is a current implementation fix,
not evidence that historical experiments used this update.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
try:
    from .policy_contract import function_source, REFERENCE_COMMIT
except ImportError:
    from policy_contract import function_source, REFERENCE_COMMIT

REFERENCE_FUNCTION_SHA256 = "3060cb7cfa96666564b17d69223673bd7a3bb737e7f9a1ce3857042b626c1958"
OLD_INIT = "                self.actor_optimizer.zero_grad()\n"
NEW_INIT = '''                # RL-LHM global token mean v1: sum tokens over the DP minibatch.
                from policy_contract import global_token_batch_info
                if self.ulysses_sequence_parallel_size != 1 or self.config.loss_agg_mode != "token-mean":
                    raise RuntimeError("RL-LHM token mean requires token-mean and sequence parallel size 1")
                self.config.global_batch_info.clear()
                self.config.global_batch_info.update(global_token_batch_info(
                    mini_batch.batch["response_mask"], get_device_id()
                ))
                self.actor_optimizer.zero_grad()
'''
OLD_SCALE = '''                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
'''
NEW_SCALE = '''                    # The global denominator already includes all microbatches.
                    loss_scale_factor = 1.0
'''
OLD_KL = 'kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)'
NEW_KL = 'kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **self.config.global_batch_info)'
REPLACEMENTS = ((OLD_INIT, NEW_INIT), (OLD_SCALE, NEW_SCALE), (OLD_KL, NEW_KL))


def patch_source(source):
    function = function_source(source, "update_policy")
    applied = NEW_INIT in function
    original = function
    if applied:
        for old, new in REPLACEMENTS:
            if original.count(new) != 1:
                raise RuntimeError("Incomplete token mean patch")
            original = original.replace(new, old, 1)
    if hashlib.sha256(original.encode()).hexdigest() != REFERENCE_FUNCTION_SHA256:
        raise RuntimeError("Unsupported VeRL update_policy; refusing guessed token mean patch")
    if applied:
        return source, True
    updated = function
    for old, new in REPLACEMENTS:
        if updated.count(old) != 1:
            raise RuntimeError("Nonunique token mean patch anchor")
        updated = updated.replace(old, new, 1)
    return source.replace(function, updated, 1), False


def require_token_mean_patch(path):
    _, applied = patch_source(path.read_text(encoding="utf-8"))
    if not applied:
        raise RuntimeError("Missing RL-LHM token mean patch; run rl/patch_verl_token_mean.py --dp-actor PATH --audit JSON")
    return {"patch_version": 1, "reference_commit": REFERENCE_COMMIT,
            "dp_actor_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-actor", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    original = args.dp_actor.read_text(encoding="utf-8")
    updated, applied = patch_source(original)
    if not applied:
        backup = args.dp_actor.with_suffix(".py.pre_rl_lhm_token_mean")
        if not backup.exists():
            shutil.copy2(args.dp_actor, backup)
        args.dp_actor.write_text(updated, encoding="utf-8")
    audit = {"already_applied": applied, **require_token_mean_patch(args.dp_actor)}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
