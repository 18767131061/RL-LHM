from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from models.huggingface import DecodingConfig, HuggingFaceBackend, _through_first_eos


def tiny_backend():
    torch.manual_seed(42)
    config = Qwen3VLConfig(
        image_token_id=91, video_token_id=94, vision_start_token_id=90, vision_end_token_id=92,
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                         pad_token_id=0, eos_token_id=2, attention_dropout=0.0),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                           out_hidden_size=32, deepstack_visual_indexes=[0, 1], num_position_embeddings=16))
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    generation = DecodingConfig(max_new_tokens=3, do_sample=True, temperature=1, top_p=1, top_k=0,
                                repetition_penalty=1, num_beams=1, eos_token_ids=(2,), pad_token_id=0,
                                use_cache=True, suppress_token_ids=(0, 90, 91, 92, 94))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, padding_side="left"),
                                batch_decode=lambda rows, **kwargs: [str(row.tolist()) for row in rows],
                                save_pretrained=lambda path: None)
    backend = HuggingFaceBackend(model, processor, generation)
    tokens = torch.tensor([[3] + ([90] + [91] * 4 + [92]) * 3 + [4]])
    inputs = dict(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                  pixel_values=torch.randn(48, 24), image_grid_thw=torch.tensor([[1, 4, 4]] * 3))
    backend._encode = lambda records, supervised: {k: v.clone() for k, v in inputs.items()}
    backend.policy_scoring = True
    return backend, inputs


def test_multimodal_sampling_scores_equal_teacher_forcing():
    backend, inputs = tiny_backend()
    with torch.no_grad():
        output = backend.model.generate(**inputs, generation_config=backend.generation.as_transformers(),
                                        return_dict_in_generate=True, output_scores=True)
        suffix = output.sequences[:, inputs["input_ids"].shape[1]:]
        retained = _through_first_eos(suffix, backend.generation.eos_token_ids)
        forward_inputs = dict(inputs, input_ids=output.sequences,
                              attention_mask=torch.cat([inputs["attention_mask"], retained], dim=1))
        forced = backend._token_log_probabilities(forward_inputs, policy=True)[:, -suffix.shape[1]:]
        actual = torch.stack(output.scores, dim=1).log_softmax(-1).gather(-1, suffix[..., None]).squeeze(-1)
        torch.testing.assert_close(forced[retained], actual[retained], rtol=2e-5, atol=2e-5)


def test_rollout_scoring_checkpointed_backward_and_new_image_grid():
    backend, _ = tiny_backend()
    rollout = backend.generate([{}])
    old = rollout.old_log_probabilities[rollout.response_mask].clone()
    backend.enable_training(128)
    backend.model.train()
    current = backend.log_probabilities(rollout)
    torch.testing.assert_close(current[rollout.response_mask], old, rtol=2e-5, atol=2e-5)
    loss = -current[rollout.response_mask].mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in backend.model.parameters() if p.grad is not None)
    # A different image grid must not reuse rope offsets from a previous input.
    with torch.no_grad():
        first = backend.log_probabilities(rollout).clone()
        backend.model.model.rope_deltas = torch.tensor([[999]])
        second = backend.log_probabilities(rollout)
    torch.testing.assert_close(first, second)


def test_eos_and_sequence_budget():
    mask = _through_first_eos(torch.tensor([[4, 2, 0, 0], [5, 6, 7, 8]]), [2])
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    backend, _ = tiny_backend()
    backend.max_sequence_length = 10
    with pytest.raises(ValueError, match="generation budget"):
        backend.generate([{}])


def test_visual_feature_extraction_uses_frozen_phase_patches(tmp_path):
    from PIL import Image
    from models.visual_features import extract_qwen3_visual_vectors, orthonormal_projection
    backend, inputs = tiny_backend()
    backend.processor.image_processor = lambda **kwargs: {
        "pixel_values": inputs["pixel_values"], "image_grid_thw": inputs["image_grid_thw"]}
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    vectors = extract_qwen3_visual_vectors(backend, [{"images": {p: str(path) for p in "apv"}}],
                                          orthonormal_projection(32, 8, 42))
    assert vectors[0].shape == (24,)
    assert vectors[0].norm().item() == pytest.approx(1.0, abs=1e-6)
    assert not vectors[0].requires_grad
