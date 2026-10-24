"""Prepared converter output and CLI failures with a small processor substitute."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
from PIL import Image

from evaluation.schema import parse_completion
from preprocess.build_verl_real961_dataset import CLASSES, class_counts_for, unpack, write_verl
from rewards import verl_liver_reward
from training import rl_preflight_check as preflight

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_LHM_REWARD_MODE", "rars")
    monkeypatch.setenv("RL_LHM_RISK_MATRIX_MODE", "ordinal")
    monkeypatch.setenv("RL_LHM_RISK_MATRIX_SCALE", "1.0")
    monkeypatch.delenv("RL_LHM_RISK_PROFILE", raising=False)
    for phase in "apv":
        Image.new("RGB", (16, 16)).save(tmp_path / f"{phase}.png")
    records = [{"phase": "a", "label": "HCC", "bbox_2d": [10, 20, 100, 120]},
               *[{"phase": phase, "label": "no-lesion", "bbox_2d": [-1] * 4} for phase in "pv"]]
    source = {
        "metadata": {"positive": 1, "patient_id": "P1", "statistical_patient_id": "P1",
                     "logical_key": "P1", "slice_index": 4, "class": "HCC"},
        "messages": [{"role": "user", "content": [
            *[{"type": "image", "image": f"{phase}.png"} for phase in "apv"],
            {"type": "text", "text": "Compare all three phases and return detections."}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps({"detections": records})}]}],
    }
    row = unpack(source, tmp_path)
    path = tmp_path / "converted.jsonl"
    write_verl([row], path, "synthetic_preflight", class_counts_for([row]), "private")
    return path, json.loads(path.read_text())


def test_actual_converter_output_restores_chat_images_and_reward(prepared):
    _, sample = prepared
    original = copy.deepcopy(sample)
    sample["prompt"].insert(0, {"role": "system", "content": "Read the study."})
    messages = preflight.build_messages(sample)
    assert [m["role"] for m in messages] == ["system", "user"]
    content = messages[1]["content"]
    images = [part for part in content if part["type"] == "image"]
    assert [part["image"] for part in images] == [part["image"] for part in sample["images"]]
    assert [part["max_pixels"] for part in images] == [262144] * 3
    assert "".join(part["text"] for part in content if part["type"] == "text") == "\n\n\nCompare all three phases and return detections."
    assert sample["prompt"][1:] == original["prompt"]
    parsed = parse_completion(preflight.ground_truth_completion(sample), CLASSES)
    assert parsed.detections[0].box == (10, 20, 100, 120)
    assert parsed.background_phases == ("p", "v")
    result = preflight.check_reward_contract(verl_liver_reward, sample)
    assert result["valid"]["parse_ok"] == 1 and result["valid"]["score"] == 2
    assert result["invalid"]["parse_ok"] == 0 and result["invalid"]["score"] == 0


def test_background_and_multilesion_ground_truth_convert_without_double_encoding(prepared):
    _, sample = prepared
    background = {"gt_dets": [], "background_phases": ["a", "p", "v"]}
    sample["ground_truth"] = background
    sample["reward_model"]["ground_truth"] = json.dumps(background)
    assert preflight.check_reward_contract(verl_liver_reward, sample)["valid"]["score"] == 1
    multilesion = {"gt_dets": [
        {"phase": "a", "label": "HCC", "bbox": [0, 0, 10, 10]},
        {"phase": "a", "label": "HCC", "bbox": [30, 30, 40, 40]}],
        "background_phases": ["p", "v"]}
    sample["ground_truth"] = json.dumps(multilesion)
    sample["reward_model"]["ground_truth"] = multilesion
    assert len(json.loads(preflight.ground_truth_completion(sample))["detections"]) == 4
    assert preflight.check_reward_contract(verl_liver_reward, sample)["valid"]["score"] == 2


@pytest.mark.parametrize("mutation", ["too_few_images", "extra_marker", "missing_marker", "list_repr", "nonstring_content"])
def test_prompt_contract_rejects_ambiguous_inputs(prepared, mutation):
    _, sample = prepared
    if mutation == "too_few_images":
        sample["images"].pop()
    elif mutation == "extra_marker":
        sample["prompt"][0]["content"] += "<image>"
    elif mutation == "missing_marker":
        sample["prompt"][0]["content"] = "No image markers"
    elif mutation == "list_repr":
        sample["prompt"] = {"role": "user", "content": "<image>" * 3}
    else:
        sample["prompt"][0]["content"] = [{"type": "text", "text": "unsupported prepared schema"}]
    with pytest.raises(ValueError):
        preflight.build_messages(sample)


def test_plain_prompt_compatibility_has_each_image_once(prepared):
    _, sample = prepared
    sample["prompt"] = "Diagnose this study."
    content = preflight.build_messages(sample)[0]["content"]
    assert len([segment for segment in content if segment["type"] == "image"]) == 3
    assert content[-1] == {"type": "text", "text": "Diagnose this study."}


@pytest.mark.parametrize("fault", ["double_encoded", "disagree", "empty", "invalid_box"])
def test_ground_truth_contract_fails_before_a_false_success(prepared, fault):
    _, sample = prepared
    if fault == "double_encoded":
        sample["ground_truth"] = json.dumps(sample["ground_truth"])
        sample.pop("reward_model")
    elif fault == "disagree":
        sample["reward_model"]["ground_truth"] = {"gt_dets": []}
    else:
        truth = {"gt_dets": [], "background_phases": []}
        if fault == "invalid_box":
            truth["gt_dets"] = [{"phase": "a", "label": "HCC", "bbox": [10, 10, 0, 0]}]
        sample["ground_truth"] = truth
        sample["reward_model"]["ground_truth"] = truth
    with pytest.raises(ValueError):
        preflight.ground_truth_completion(sample)


def test_reward_exceptions_are_not_retried_or_reported_as_success(prepared):
    _, sample = prepared
    calls = []
    def broken(**kwargs):
        calls.append(kwargs)
        raise TypeError("callback internal defect")
    with pytest.raises(TypeError, match="callback internal defect"):
        preflight.check_reward_contract(types.SimpleNamespace(compute_score=broken), sample)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="positive reward"):
        preflight.check_reward_contract(types.SimpleNamespace(compute_score=lambda **kw: {"score": 0, "parse_ok": 0}), sample)
    with pytest.raises(RuntimeError, match="finite score"):
        preflight.check_reward_contract(types.SimpleNamespace(compute_score=lambda **kw: {"score": float("nan"), "parse_ok": 1}), sample)


def test_tokenizer_receives_restored_messages_and_training_template_options(prepared):
    _, sample = prepared
    captured = {}
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"], captured["options"] = messages, kwargs
            return "template with visual placeholders"
        def __call__(self, **kwargs):
            captured["inputs"] = kwargs
            return {"input_ids": types.SimpleNamespace(ndim=2, shape=(1, 813))}
    def vision_info(messages):
        assert messages is captured["messages"]
        return [part["image"] for message in messages for part in message["content"] if part["type"] == "image"], None
    assert preflight.prompt_token_count(sample, Processor(), vision_info) == 813
    assert captured["options"] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert len(captured["inputs"]["images"]) == 3
    preflight.check_token_budget(813, 1024, 2048, 256)


def cli_with_processor_stub(tmp_path, prepared_path, tokens, *overrides):
    """Only processor/vision preprocessing are substitutes; converter/reward/CLI run normally."""
    stubs = tmp_path / "processor_stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / "transformers.py").write_text('''import os
from types import SimpleNamespace
class AutoProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert kwargs["local_files_only"] is True
        return cls()
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert sum(s["type"] == "image" for m in messages for s in m["content"]) == 3
        return "synthetic processor template"
    def __call__(self, **kwargs):
        assert len(kwargs["images"]) == 3
        return {"input_ids": SimpleNamespace(ndim=2, shape=(1, int(os.environ["PREFLIGHT_TEST_TOKENS"])))}
''')
    (stubs / "qwen_vl_utils.py").write_text('''def process_vision_info(messages):
    return [s["image"] for m in messages for s in m["content"] if s["type"] == "image"], None
''')
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(stubs), PREFLIGHT_TEST_TOKENS=str(tokens))
    command = [sys.executable, str(ROOT / "training/rl_preflight_check.py"),
               "--rl-data-path", str(prepared_path), "--model-path", str(model),
               "--reward-path", str(ROOT / "rewards/verl_liver_reward.py"), *overrides]
    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)


def test_cli_actual_converter_record_passes_with_explicit_cpu_test_boundary(tmp_path, prepared):
    path, _ = prepared
    completed = cli_with_processor_stub(tmp_path, path, 800)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "passed_selected_record"
    assert result["prompt_tokens_including_images"] == 800
    assert result["reward_probes"]["valid"]["parse_ok"] == 1
    assert result["gpu_rollout_verified"] is False


@pytest.mark.parametrize("tokens,overrides,reason", [
    (1025, [], "max_prompt_length"),
    (800, ["--max-model-len", "900"], "Remaining context"),
    (800, ["--max-response-len", "0"], "positive"),
    (800, ["--sample-idx", "-1"], "outside"),
])
def test_cli_budget_or_index_failures_have_nonzero_exit(tmp_path, prepared, tokens, overrides, reason):
    path, _ = prepared
    result = cli_with_processor_stub(tmp_path, path, tokens, *overrides)
    assert result.returncode != 0
    assert "[FATAL]" in result.stderr and reason in result.stderr
    assert "passed_selected_record" not in result.stdout
