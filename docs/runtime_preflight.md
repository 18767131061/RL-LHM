# Runtime input checks and rollout log probabilities

These checks validate prepared input, token limits, reward parsing and
response-token log probabilities for the configured runtime.

## Check a prepared record

Use the JSONL emitted by `preprocess/build_verl_real961_dataset.py` and the
local processor saved with the intended SFT checkpoint:

```bash
python training/rl_preflight_check.py \
  --rl-data-path /path/to/prepared/real961_rl_train.jsonl \
  --model-path /path/to/local/sft-checkpoint \
  --reward-path rewards/verl_liver_reward.py \
  --sample-idx 0 \
  --max-prompt-len 1024 --max-response-len 256 --max-model-len 2048
```

The command checks one selected record. Check all records to determine the
dataset's maximum prompt length before training. The converter writes absolute
image paths. A manually prepared relative path resolves from the working
directory, so prefer absolute paths for this interface.

The preflight preserves the chat roles and replaces each `<image>` marker
with the corresponding phase image, including its `max_pixels` metadata.
There must be exactly three inputs and three image slots in a chat-list
prompt. The legacy plain-string prompt interface prepends the three images
when no markers are present. Unsupported prepared message schemas fail
instead of being converted to a Python list's string representation.

Tokenization uses the local processor, `add_generation_prompt=True` and
`enable_thinking=False`, matching the launcher's explicit chat setting. Both
`data.max_prompt_length` and the remaining model context are checked.
Insufficient context, an overlong prompt, malformed metadata, missing image
files or a failed reward probe produce a nonzero exit code. Local model
loading does not download missing processor files. If a separately reviewed
local model requires custom code, pass `--trust-remote-code` explicitly.

The reward probe decodes `ground_truth` once, converts `gt_dets`/`bbox` into
`detections`/`bbox_2d`, and adds the explicitly declared background sentinels.
It retains multiple lesions. The actual callback must return `parse_ok=1`
and a finite positive score for that target, and `parse_ok=0`/score zero for
invalid JSON. Exceptions are propagated. The probe checks the configured callback against the supervised target schema.

Successful output has `status: "passed_selected_record"` and
`gpu_rollout_verified: false`. The optional `--rl-dataset-path` inspects a
supplied external source file for informational keywords only. Keyword
presence is not an integration test of that runtime.

## Install the agent-loop bridge

Apply the helper to the separately selected VeRL agent-loop source:

```bash
python rl/patch_verl_agent_loop_logprobs.py \
  --agent-loop /path/to/verl/experimental/agent_loop/agent_loop.py \
  --audit /path/to/run/agent_loop_patch.json
```

Bridge version 3 accepts exactly one of the known original, v1, v2 or v3
source blocks. It checks Python syntax and that the matched text is an
executable assignment; examples inside strings, duplicate blocks and
modified/unrecognized blocks fail before any file is changed. Reapplication
is idempotent. The original `.py.pre_rl_lhm` backup is kept during an upgrade;
the audit records the previous source variant and before/after hashes.
Use the VeRL reference version specified in [training_pipeline.md](training_pipeline.md).

For a generated response of padded width `L`, the bridge accepts log
probabilities in either of these forms:

- Exactly `L` entries aligned with the padded response.
- Exactly one entry per valid generated token, in mask order.

The attention mask must be finite and binary, and every valid token must
have a finite log probability. Compact values are scattered to their actual
valid positions. Only padding positions receive zeros. Padding-only NaN/Inf
values in a full-width input are discarded; they are never included in NLL.
Missing values, extra values, ambiguous intermediate lengths and a response
without valid tokens fail. No probability sequence is silently truncated.

Inspect generated token masks, lengths and score alignment in the configured
VeRL/vLLM environment before a complete training run.

