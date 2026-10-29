"""Optional local processor integration: RL_LHM_PROCESSOR_PATH=/path/to/processor pytest."""
import os
import json

import pytest
from PIL import Image
from transformers import AutoProcessor

from data.config import load_config
from models.huggingface import DecodingConfig, HuggingFaceBackend
from tests.test_backend import tiny_backend


@pytest.mark.skipif(not os.environ.get("RL_LHM_PROCESSOR_PATH"), reason="local Qwen3-VL processor path required")
def test_actual_processor_sft_mask(tmp_path):
    config = load_config("configs/sft_4xa800.json")
    backend, _ = tiny_backend()
    backend.processor = AutoProcessor.from_pretrained(os.environ["RL_LHM_PROCESSOR_PATH"], local_files_only=True,
                                                     **config["model"]["processor_kwargs"])
    backend.generation = DecodingConfig(**config["generation"])
    backend.processor.tokenizer.padding_side = "left"
    backend._encode = HuggingFaceBackend._encode.__get__(backend)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (512, 512), (80, 100, 120)).save(image_path)
    truth = '{"detections":[{"phase":"a","label":"HCC","bbox_2d":[100,100,300,300]}]}'
    record = dict(images={p: str(image_path) for p in "apv"}, prompt="Describe liver lesions.", ground_truth=truth)
    longer = dict(record, prompt=record["prompt"] + " Return all findings in each phase using JSON.")
    inputs, mask = backend.prepare_sft([record, longer])
    for row in range(2):
        tokens = inputs["input_ids"][row, 1:][mask[row]]
        text = backend.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        assert text == truth
        assert tokens[-1].item() in backend.generation.eos_token_ids
        from data.lengths import target_token_count
        assert target_token_count(backend.processor, (record, longer)[row], backend.generation.eos_token_ids) == len(tokens)
    assert inputs["image_grid_thw"].shape[0] == 6


@pytest.mark.skipif(not os.environ.get("RL_LHM_PROCESSOR_PATH"), reason="local Qwen3-VL processor path required")
def test_multilesion_target_budget_includes_eos():
    from types import SimpleNamespace
    from data.lengths import target_lengths, target_token_count
    from tests.test_algorithms import completion, lesion
    config = load_config("configs/sft_4xa800.json")
    processor = AutoProcessor.from_pretrained(os.environ["RL_LHM_PROCESSOR_PATH"], local_files_only=True)
    generation = DecodingConfig(**config["generation"])
    for label in ("HCC", "HA_Hemangioma", "nodule_CYST"):
        records = [dict(uid=str(n), patient_id="p", prompt="Find lesions", ground_truth=json.dumps({"detections": [
            lesion(label, (50 + i*100, 100, 100 + i*100, 200), p) for p in "apv" for i in range(n)]}, separators=(",", ":")))
                   for n in (1, 3, 6)]
        rows = target_lengths(processor, records, (label,), generation)
        assert rows[0]["target_tokens"] < 256 < rows[1]["target_tokens"] < 512 < rows[2]["target_tokens"]
        assert [r["exceeds_generation_budget"] for r in rows] == [False, False, True]
        exact = rows[1]["target_tokens"]
        assert not target_lengths(processor, records[1:2], (label,), SimpleNamespace(eos_token_ids=generation.eos_token_ids, max_new_tokens=exact))[0]["exceeds_generation_budget"]
        assert target_lengths(processor, records[1:2], (label,), SimpleNamespace(eos_token_ids=generation.eos_token_ids, max_new_tokens=exact-1))[0]["exceeds_generation_budget"]


@pytest.mark.skipif(not os.environ.get("RL_LHM_PROCESSOR_PATH"), reason="local Qwen3-VL processor path required")
def test_length_preflight_cli_fails_over_budget_without_loading_model(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from tests.test_algorithms import lesion
    config = load_config("configs/sft_4xa800.json")
    config["model"]["processor_name_or_path"] = str(Path(os.environ["RL_LHM_PROCESSOR_PATH"]).resolve())
    config["model"]["revision"] = None
    config["model"]["model_name_or_path"] = "model-weights-must-not-load"
    truth = json.dumps({"detections": [lesion("HCC", (50+i*100,100,100+i*100,200), p) for p in "apv" for i in range(6)]}, separators=(",", ":"))
    manifest = dict(output_classes=["HCC"], metric_classes=["HCC"], patients={"p":"HCC"}, records=[
        dict(uid="u", patient_id="p", images={p:"unused.png" for p in "apv"}, prompt="Find lesions", ground_truth=truth)])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config["manifest"] = str(path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "lengths.json"
    result = subprocess.run([sys.executable, "-m", "data.check_lengths", "--config", str(config_path), "--output", str(output)], capture_output=True)
    assert result.returncode == 1, result.stderr.decode()
    assert json.loads(output.read_text())["over_budget_groups"] == 1
