"""Reference feature/cache contracts; all inputs are synthetic."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import extract_qwen_visual_latents as extractor


def setup_inputs(tmp_path, n=2):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "vision_config": {"hidden_size": 1024}}))
    (model / "preprocessor_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"synthetic-backend-only")
    images = []
    for i in range(3):
        path = tmp_path / f"phase{i}.png"
        Image.new("RGB", (32, 32), (i * 50, 20, 30)).save(path)
        images.append({"image": path.name, "max_pixels": 1024})
    rows = [{"images": images, "prompt": [{"role": "user", "content": "Do not use text"}],
             "extra_info": {"sample_uid": f"sample-{i}", "patient_id": f"p-{i}", "visual_z": None},
             "ground_truth": f"unused-label-{i}", "reward_model": {"extra_info": {}}} for i in range(n)]
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return model, source, tmp_path / "cache.jsonl", rows


class SyntheticBackend:
    versions = {"backend": "explicit-test-double"}
    calls = 0
    fail_after = None

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, specs):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated interruption")
        type(self).calls += 1
        assert len(specs) == 3
        assert all(set(spec) == {"path", "max_pixels"} for spec in specs)
        return [1.0] + [0.0] * 1151


@pytest.fixture(autouse=True)
def reset_backend():
    SyntheticBackend.calls = 0
    SyntheticBackend.fail_after = None


def run_cache(model, source, output, **kwargs):
    return extractor.extract_cache(source, output, model, device="cpu", dtype="float32",
                                   backend_factory=SyntheticBackend, **kwargs)


def test_projection_is_frozen_orthonormal_and_phase_pooling_is_explicit():
    assert extractor.sha(extractor.PROJECTION) == extractor.PROJECTION_SHA256
    matrix = torch.from_numpy(np.load(extractor.PROJECTION))
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(384), atol=1e-6, rtol=1e-6)
    hidden = torch.arange(7 * 1024).reshape(7, 1024).float() / 1024
    grid = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 1, 4]])
    actual = torch.tensor(extractor.pool_features(hidden, grid, matrix))
    expected = torch.cat([hidden[:1].mean(0) @ matrix, hidden[1:3].mean(0) @ matrix,
                          hidden[3:].mean(0) @ matrix])
    expected /= expected.norm()
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1152,)


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_invalid_features_fail(bad):
    with pytest.raises(ValueError):
        extractor.pool_features(torch.full((12, 1024), bad), torch.tensor([[1, 2, 2]] * 3),
                                torch.from_numpy(np.load(extractor.PROJECTION)))


def test_cache_preserves_targets_and_checks_idempotent_reuse(tmp_path):
    model, source, output, rows = setup_inputs(tmp_path)
    audit = run_cache(model, source, output)
    cached = extractor.read_rows(output)
    assert audit["status"] == "complete" and audit["rows"] == 2
    assert SyntheticBackend.calls == 2
    assert cached[0]["ground_truth"] == rows[0]["ground_truth"]
    assert cached[0]["reward_model"]["extra_info"] == cached[0]["extra_info"]
    assert run_cache(model, source, output, validate_only=True) == audit
    assert SyntheticBackend.calls == 2


@pytest.mark.parametrize("mutate", ["image", "source", "checkpoint", "cache"])
def test_cache_conflicts_fail_closed(tmp_path, mutate):
    model, source, output, _ = setup_inputs(tmp_path)
    run_cache(model, source, output)
    if mutate == "image":
        Image.new("RGB", (32, 32), (255, 255, 255)).save(tmp_path / "phase0.png")
    elif mutate == "source":
        source.write_text(source.read_text().replace("unused-label", "changed-label"))
    elif mutate == "checkpoint":
        (model / "model.safetensors").write_bytes(b"changed")
    else:
        output.write_text(output.read_text().replace("unused-label", "changed-label"))
    with pytest.raises(ValueError, match="conflict"):
        run_cache(model, source, output)


def test_partial_resume_discards_only_torn_last_line(tmp_path):
    model, source, output, _ = setup_inputs(tmp_path)
    SyntheticBackend.fail_after = 1
    with pytest.raises(RuntimeError, match="interruption"):
        run_cache(model, source, output)
    assert not output.exists()
    partial = output.with_suffix(".jsonl.partial")
    with partial.open("a") as stream:
        stream.write('{"torn":')
    SyntheticBackend.fail_after = None
    audit = run_cache(model, source, output)
    assert audit["rows"] == 2 and SyntheticBackend.calls == 2
    assert not partial.exists()


def test_invalid_input_and_adapters_are_rejected(tmp_path):
    model, source, output, rows = setup_inputs(tmp_path)
    rows[1]["extra_info"]["sample_uid"] = rows[0]["extra_info"]["sample_uid"]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="unique"):
        run_cache(model, source, output)
    (model / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="merge"):
        extractor.checkpoint_files(model)
    with pytest.raises(ValueError, match="ordering"):
        extractor.image_specs({"images": [{"image": "phase0.png", "phase": "delayed"}] * 3}, tmp_path)


def test_real_hf_vision_loading_forward_and_label_independence(tmp_path):
    pytest.importorskip("transformers", minversion="4.57.1")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from safetensors.torch import save_file
    model, source, output, rows = setup_inputs(tmp_path)
    torch.manual_seed(17)
    config = Qwen3VLVisionConfig(depth=1, hidden_size=1024, intermediate_size=32, num_heads=16,
                                out_hidden_size=32, num_position_embeddings=16,
                                deepstack_visual_indexes=[], patch_size=16, temporal_patch_size=2)
    config._attn_implementation = "eager"
    vision = Qwen3VLVisionModel(config).eval()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "vision_config": config.to_dict()}))
    save_file({"model.visual." + key: value for key, value in vision.state_dict().items()}, model / "model.safetensors")
    del vision
    Qwen2VLImageProcessor(min_pixels=1024, max_pixels=1024, patch_size=16,
                         temporal_patch_size=2, merge_size=2).save_pretrained(model)
    backend = extractor.QwenVisualExtractor(model, device="cpu", dtype="float32")
    first = backend(extractor.image_specs(rows[0], tmp_path))
    changed = copy.deepcopy(rows[0]); changed["ground_truth"] = "different"; changed["prompt"] = "ignored"
    second = backend(extractor.image_specs(changed, tmp_path))
    assert first == second and extractor.valid_vector(first)
    assert not backend.vision.training and not any(p.requires_grad for p in backend.vision.parameters())
    reversed_row = copy.deepcopy(rows[0]); reversed_row["images"].reverse()
    reversed_feature = backend(extractor.image_specs(reversed_row, tmp_path))
    assert first != reversed_feature


def test_runtime_change_prevents_mixed_partial_cache(tmp_path, monkeypatch):
    model, source, output, _ = setup_inputs(tmp_path)
    SyntheticBackend.fail_after = 1
    with pytest.raises(RuntimeError):
        run_cache(model, source, output)
    original = extractor.metadata.version
    monkeypatch.setattr(extractor.metadata, "version", lambda key: "changed-version" if key == "torch" else original(key))
    SyntheticBackend.fail_after = None
    with pytest.raises(ValueError, match="Partial cache conflicts"):
        run_cache(model, source, output)


def test_grid_binds_reference_cache_to_sft_checkpoint(tmp_path):
    from training.rars_joint_grid import identities
    model, source, output, _ = setup_inputs(tmp_path)
    run_cache(model, source, output)
    other = tmp_path / "other_seed"
    import shutil
    shutil.copytree(model, other)
    config = {"train_file": str(output), "seed_train_files": {"42": str(output)},
              "seed_model_paths": {"42": str(other)}, "protocol": "main"}
    # This must fail before any validation manifest is opened, even when all
    # model bytes are equal: this cache is bound to another seed checkpoint path.
    with pytest.raises(ValueError, match="identity/content conflicts"):
        identities(config)
