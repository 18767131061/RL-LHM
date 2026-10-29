"""Optional local processor integration: RL_LHM_PROCESSOR_PATH=/path/to/processor pytest."""
import os

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
    assert inputs["image_grid_thw"].shape[0] == 6
