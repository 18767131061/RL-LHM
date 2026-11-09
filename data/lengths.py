"""Tokenizer-exact assistant target budgets and generation termination metadata."""
from collections import Counter

from rewards.schema import PHASES, parse_completion


def target_token_count(processor, record, eos_token_ids):
    """Count the actual chat-template assistant suffix through its first EOS.

    Image placeholders suffice here: image-grid expansion changes the prompt,
    not the assistant suffix. No images or model weights are needed.
    """
    user = {"role": "user", "content": [*({"type": "image"} for _ in PHASES),
                                          {"type": "text", "text": record["prompt"]}]}
    assistant = {"role": "assistant", "content": [{"type": "text", "text": record["ground_truth"]}]}
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
    prompt_ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = processor.tokenizer.encode(full, add_special_tokens=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant target does not extend its generation prompt")
    suffix = full_ids[len(prompt_ids):]
    eos = [i for i, token in enumerate(suffix) if token in eos_token_ids]
    if not eos:
        raise ValueError("assistant template must include a configured EOS")
    # Reject embedded EOS in annotation text rather than silently shorten GT.
    raw_ids = processor.tokenizer.encode(record["ground_truth"], add_special_tokens=False)
    if any(token in eos_token_ids for token in raw_ids):
        raise ValueError(f"ground truth for {record['uid']} contains an EOS token")
    return eos[0] + 1


def target_lengths(processor, records, output_classes, generation):
    rows = []
    for record in records:
        count = target_token_count(processor, record, generation.eos_token_ids)
        parsed = parse_completion(record["ground_truth"], output_classes)
        per_phase = Counter(item.phase for item in parsed.detections)
        rows.append(dict(uid=record["uid"], patient_id=record["patient_id"],
                         patient_label=record.get("patient_label"), target_tokens=count,
                         lesion_records=len(parsed.detections),
                         max_lesions_per_phase=max(per_phase.values(), default=0),
                         exceeds_generation_budget=count > generation.max_new_tokens))
    return rows


def length_summary(rows, max_new_tokens):
    strata = {}
    for row in rows:
        for key in (f"class:{row['patient_label']}", f"max_lesions_per_phase:{row['max_lesions_per_phase']}"):
            bucket = strata.setdefault(key, dict(groups=0, over_budget=0, maximum_target_tokens=0))
            bucket["groups"] += 1
            bucket["over_budget"] += int(row["exceeds_generation_budget"])
            bucket["maximum_target_tokens"] = max(bucket["maximum_target_tokens"], row["target_tokens"])
    return dict(max_new_tokens=max_new_tokens, groups=len(rows),
                maximum_target_tokens=max((r["target_tokens"] for r in rows), default=0),
                target_token_histogram=dict(sorted(Counter(r["target_tokens"] for r in rows).items())),
                over_budget_groups=sum(r["exceeds_generation_budget"] for r in rows), strata=strata, records=rows)
