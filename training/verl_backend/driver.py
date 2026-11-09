"""One synchronous RL update using official VeRL carrier interfaces."""
from __future__ import annotations

import torch
from .dataset import occurrence_rows


def update_once(carrier, paper, dataset, indices, *, epoch, offset):
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn

    config = paper.configuration
    loop = config["loop"]
    rows, records, rids, real = occurrence_rows(
        dataset, indices, epoch=epoch, offset=offset, batch_size=loop["batch_size"],
        group_size=loop["group_size"], seeds=paper.sampling_seeds(loop["batch_size"] * loop["group_size"]),
    )
    prompts = DataProto.from_single_dict(collate_fn(rows))
    prompts.meta_info.update(global_steps=carrier.global_steps, temperature=1.0)
    # Rows already contain G completions per occurrence. Do not repeat again.
    batch = carrier.async_rollout_manager.generate_sequences(prompts)
    carrier.checkpoint_manager.sleep_replicas()
    if len(batch) != len(records) or list(batch.non_tensor_batch.get("uid", [])) != rids:
        raise RuntimeError("VeRL generation changed occurrence order or lost RID metadata")
    batch.meta_info["temperature"] = 1.0
    mask = batch.batch["response_mask"]
    if not torch.equal(mask.bool(), batch.batch["attention_mask"][:, -mask.shape[1]:].bool()):
        raise RuntimeError("single-turn rollout mask must retain every generated token, including EOS")
    batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(-1).tolist()
    batch.meta_info["images_seqlens"] = [int(n) for mm in batch.non_tensor_batch["multi_modal_inputs"]
                                         for n in mm.get("images_seqlens", [])]
    # Recompute the proximal policy on the generated sequence. Sampling-engine
    # logprobs (after top-p/suppression) never enter NLL, ratio or reference KL.
    old, _ = carrier._compute_old_log_prob(batch)
    old.batch.pop("entropys")
    batch = batch.union(old)
    batch = batch.union(carrier._compute_ref_log_prob(batch))
    if not torch.isfinite(batch.batch["ref_log_prob"][mask.bool()]).all():
        raise ValueError("retained reference log probabilities must be finite")
    texts = [carrier.tokenizer.decode(tokens[keep.bool()].tolist(), skip_special_tokens=True)
             for tokens, keep in zip(batch.batch["responses"], mask)]
    scored = paper.score_and_advantage(
        texts=texts, old_log_probs=batch.batch["old_log_probs"], response_mask=mask,
        records=records, rids=rids, real=real,
    )
    batch.batch["advantages"] = scored["advantages"]
    batch.batch["returns"] = scored["advantages"].clone()
    batch.batch["response_mask"] = scored["response_mask"]
    scores = torch.zeros_like(mask, dtype=torch.float32)
    real_positions = [i for i, keep in enumerate(real) if keep]
    for row, reward in zip(real_positions, scored["rewards"]):
        last = int(mask[row].sum()) - 1
        scores[row, last] = reward
    batch.batch["token_level_scores"] = scores
    batch.batch["token_level_rewards"] = scores.clone()
    # The inherited method converts response_mask to loss_mask. Official FSDP
    # all-reduces loss_mask.sum BEFORE splitting microbatches; dummy attention
    # stays intact for Qwen position/vision processing, with zero gradient.
    carrier._update_actor(batch)
    paper.after_policy_update(scored)
    lengths = mask[real_positions].sum(-1)
    eos_ids = set(config["generation"]["eos_token_ids"])
    eos_count = sum(int(batch.batch["responses"][row, int(mask[row].sum()) - 1]) in eos_ids
                    for row in real_positions)
    return dict(epoch=epoch, step=offset // loop["batch_size"],
                mean_reward=float(scored["rewards"].mean()), mean_completion_nll=float(scored["nlls"].mean()),
                input_count=len(indices), completion_count=len(real_positions),
                retained_tokens=scored["retained_tokens"], eos_completions=eos_count,
                length_limited_completions=len(real_positions) - eos_count,
                truncation_rate=(len(real_positions) - eos_count) / len(real_positions),
                maximum_completion_tokens=int(lengths.max()), policy_updates=loop["policy_updates"])
