"""Exercise the actual bridge code with CPU Torch; no VeRL runtime is installed."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest
import torch

from rl.completion_likelihood import completion_likelihood, extract_valid_rollout_log_probabilities
from rl.patch_verl_agent_loop_logprobs import LEGACY_NEW, NEW, OLD, PADDED_NEW, patch_source

ROOT = Path(__file__).resolve().parents[1]


def execute_bridge(mask, logprobs, *, responses=None):
    width = len(mask)
    namespace = {
        "torch": torch,
        "TensorDict": lambda fields, batch_size: fields,
        "prompts": torch.tensor([[7, 8]]),
        "responses": torch.arange(width).reshape(1, width) if responses is None else responses,
        "attention_mask": torch.tensor([[1, 1, *mask]], dtype=torch.float32),
        "input_ids": torch.arange(width + 2).reshape(1, width + 2),
        "position_ids": torch.arange(width + 2).reshape(1, width + 2),
        "output": types.SimpleNamespace(response_logprobs=logprobs),
    }
    exec(textwrap.dedent(NEW), namespace)
    return namespace["batch"]


@pytest.mark.parametrize("mask,logprobs,expected", [
    ([1, 1, 0, 0], [-.2, -.6], [-.2, -.6, 0, 0]),
    ([1, 1, 0, 0], [[-.2, -.6, float("nan"), float("inf")]], [-.2, -.6, 0, 0]),
    ([0, 1, 0, 1], [-.2, -.6], [0, -.2, 0, -.6]),
    ([0, 1, 0, 1], [float("nan"), -.2, 99, -.6], [0, -.2, 0, -.6]),
    ([1, 1], [-.2, -.6], [-.2, -.6]),
])
def test_bridge_aligns_compact_or_padded_scores_using_actual_mask(mask, logprobs, expected):
    batch = execute_bridge(mask, logprobs)
    assert batch["rollout_log_probs"].tolist()[0] == pytest.approx(expected)
    ids, scores = extract_valid_rollout_log_probabilities(batch["responses"], batch["attention_mask"], batch["rollout_log_probs"])
    assert len(ids) == 2
    nll, confidence = completion_likelihood(scores, len(ids))
    assert nll == pytest.approx(.4)
    assert confidence == pytest.approx(torch.exp(torch.tensor(-.4)).item())


@pytest.mark.parametrize("mask,logprobs", [
    ([1, 1, 1], [-2.]),  # Previously became [-2, 0, 0], fabricating two valid scores.
    ([1, 1, 0, 0], [-.1]),
    ([1, 1, 0, 0], [-.1, -.2, -.3]),  # Neither full padded nor compact valid span.
    ([1, 1, 0, 0], [-.1] * 5),
    ([1, 1], [float("nan"), -.1]),
    ([1, 1], [float("inf"), -.1]),
    ([1, 1], None),
    ([1, 1], [[-.1], [-.2]]),
    ([1, 2], [-.1, -.2]),
    ([1, float("nan")], [-.1, -.2]),
    ([0, 0], []),
])
def test_bridge_rejects_incomplete_ambiguous_or_invalid_generated_span(mask, logprobs):
    with pytest.raises(RuntimeError):
        execute_bridge(mask, logprobs)


@pytest.mark.parametrize("block,version", [(OLD, "original"), (LEGACY_NEW, "legacy-v1"), (PADDED_NEW, "legacy-v2"), (NEW, "current-v3")])
def test_known_code_blocks_upgrade_once_and_remain_idempotent(block, version):
    source = "class Agent:\n    async def reward(self):\n        if True:\n" + block + "\n"
    changed, applied, source_version = patch_source(source)
    assert source_version == version
    assert applied == (block == NEW)
    assert NEW in changed
    repeated, applied, source_version = patch_source(changed)
    assert applied and repeated == changed and source_version == "current-v3"


@pytest.mark.parametrize("source", [
    'documented_example = """\n' + OLD + '\n"""\n',
    'reward_batch_fields = """' + NEW + '\n"""\n',
    OLD + "\n" + OLD,
    PADDED_NEW.replace("response_length - rollout_log_probs.shape[-1]", "99"),
    "not Python: {" + OLD,
])
def test_patch_rejects_strings_duplicate_blocks_modified_legacy_and_invalid_python(source):
    with pytest.raises(RuntimeError):
        patch_source(source)


def test_cli_upgrade_preserves_original_backup_and_audits_v3(tmp_path):
    target = tmp_path / "agent_loop.py"
    target.write_text(PADDED_NEW)
    backup = target.with_suffix(".py.pre_rl_lhm")
    backup.write_text(OLD)
    audit = tmp_path / "patch.json"
    command = [sys.executable, "-S", str(ROOT / "rl/patch_verl_agent_loop_logprobs.py"),
               "--agent-loop", str(target), "--audit", str(audit)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    metadata = json.loads(audit.read_text())
    assert metadata["patch_version"] == 3 and metadata["source_variant"] == "legacy-v2"
    assert target.read_text() == NEW and backup.read_text() == OLD
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    metadata = json.loads(audit.read_text())
    assert metadata["already_applied"] and metadata["sha256_before"] == metadata["sha256_after"]
    assert backup.read_text() == OLD


def test_cli_rejected_source_does_not_mutate_target_or_create_audit(tmp_path):
    target = tmp_path / "agent_loop.py"
    original = 'example = """\n' + PADDED_NEW + '\n"""\n'
    target.write_text(original)
    audit = tmp_path / "patch.json"
    result = subprocess.run([sys.executable, "-S", str(ROOT / "rl/patch_verl_agent_loop_logprobs.py"),
                             "--agent-loop", str(target), "--audit", str(audit)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert target.read_text() == original
    assert not audit.exists() and not target.with_suffix(".py.pre_rl_lhm").exists()
