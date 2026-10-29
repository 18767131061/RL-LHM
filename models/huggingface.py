"""Three-phase image/text generation and differentiable token probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig


@dataclass(frozen=True)
class DecodingConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    num_beams: int
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    use_cache: bool
    suppress_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(self.do_sample) is not bool or type(self.use_cache) is not bool:
            raise ValueError("do_sample and use_cache must be booleans")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0,1]")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")
        if self.num_beams != 1:
            raise ValueError("One completion per input uses num_beams=1")
        ids = tuple(self.eos_token_ids)
        if not ids or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("eos_token_ids must contain non-negative token IDs")
        if type(self.pad_token_id) is not int or self.pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative token ID")
        object.__setattr__(self, "eos_token_ids", ids)
        suppressed = tuple(self.suppress_token_ids)
        if any(type(value) is not int or value < 0 for value in suppressed):
            raise ValueError("suppress_token_ids must contain non-negative token IDs")
        if set(suppressed) & set(ids):
            raise ValueError("EOS tokens must remain available during generation")
        object.__setattr__(self, "suppress_token_ids", suppressed)

    def as_transformers(self) -> GenerationConfig:
        settings = dict(
            max_new_tokens=self.max_new_tokens, do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty, num_beams=self.num_beams,
            eos_token_id=list(self.eos_token_ids), pad_token_id=self.pad_token_id,
            use_cache=self.use_cache, return_dict_in_generate=False,
            suppress_tokens=list(self.suppress_token_ids),
        )
        if self.do_sample:
            settings.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        return GenerationConfig(**settings)


@dataclass
class RolloutBatch:
    texts: list[str]
    model_inputs: dict[str, torch.Tensor]
    response_mask: torch.Tensor
    old_log_probabilities: torch.Tensor
    completion_nlls: list[float]
    finish_reasons: list[str]
    completion_token_counts: list[int]


def _through_first_eos(tokens: torch.Tensor, eos_token_ids: Sequence[int]) -> torch.Tensor:
    """Retain each generated token through the first EOS, including EOS."""
    eos = torch.tensor(eos_token_ids, device=tokens.device, dtype=tokens.dtype)
    hits = (tokens.unsqueeze(-1) == eos).any(dim=-1)
    return hits.long().cumsum(dim=-1) - hits.long() == 0


class HuggingFaceBackend:
    """A causal image/text model using a checkpoint's own chat processor.

    Records contain uid, images={a,p,v}, prompt, and (for SFT) ground_truth.
    Images enter the user message in a,p,v order, followed by the record prompt.
    Token matrices use next-token alignment: column t scores input_ids[:,t+1].
    """

    def __init__(self, model, processor, generation: DecodingConfig):
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("The backend requires a causal image/text decoder")
        self.model = model
        self.forward_model = model
        self.policy_scoring = False
        self.max_sequence_length = None
        self.processor = processor
        self.generation = generation
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id != generation.pad_token_id:
            raise ValueError("Decoding pad_token_id must equal the processor's pad token")
        vocabulary_size = min(model.get_input_embeddings().weight.shape[0],
                              model.get_output_embeddings().weight.shape[0])
        declared_ids = (*generation.eos_token_ids, *generation.suppress_token_ids, generation.pad_token_id)
        if any(token_id >= vocabulary_size for token_id in declared_ids):
            raise ValueError("Decoding token IDs must lie within the model vocabulary")
        if (generation.pad_token_id not in generation.eos_token_ids
                and generation.pad_token_id not in generation.suppress_token_ids):
            raise ValueError("Text completions must suppress a distinct padding token")
        input_markers = {
            getattr(model.config, name) for name in
            ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
            if getattr(model.config, name, None) is not None
        }
        if not input_markers <= set(generation.suppress_token_ids):
            raise ValueError("Text completions must suppress the model's image/video input markers")

    @classmethod
    def from_pretrained(
        cls, *, model_name_or_path: str, processor_name_or_path: str,
        revision: str | None, device: str, dtype: str,
        attn_implementation: str, processor_kwargs: Mapping,
        generation: DecodingConfig,
    ) -> "HuggingFaceBackend":
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtypes:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        processor = AutoProcessor.from_pretrained(
            processor_name_or_path, revision=revision, trust_remote_code=False,
            **dict(processor_kwargs),
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path, revision=revision, dtype=dtypes[dtype],
            attn_implementation=attn_implementation, trust_remote_code=False,
        ).to(torch.device(device))
        return cls(model, processor, generation)

    def _messages(self, records: Sequence[Mapping], *, supervised: bool) -> list[list[dict]]:
        if not records:
            raise ValueError("A model batch must contain at least one record")
        conversations = []
        for record in records:
            if not isinstance(record["prompt"], str) or not record["prompt"]:
                raise ValueError("Each record needs a non-empty prompt")
            content = []
            for phase in ("a", "p", "v"):
                with Image.open(record["images"][phase]) as image:
                    content.append({"type": "image", "image": image.convert("RGB")})
            content.append({"type": "text", "text": record["prompt"]})
            messages = [{"role": "user", "content": content}]
            if supervised:
                if not isinstance(record["ground_truth"], str):
                    raise ValueError("ground_truth must be a structured JSON string")
                messages.append({"role": "assistant", "content": [
                    {"type": "text", "text": record["ground_truth"]}
                ]})
            conversations.append(messages)
        return conversations

    def _encode(self, records: Sequence[Mapping], *, supervised: bool) -> dict[str, torch.Tensor]:
        encoded = self.processor.apply_chat_template(
            self._messages(records, supervised=supervised), tokenize=True,
            add_generation_prompt=not supervised, return_dict=True,
            return_tensors="pt", padding=True,
        )
        inputs = {}
        for name, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Processor output {name!r} must be a tensor")
            inputs[name] = value.to(device=self.model.device,
                                   dtype=self.model.dtype if value.is_floating_point() else value.dtype)
        return inputs

    def _move_inputs(self, inputs):
        return {key: value.to(device=self.model.device,
                             dtype=self.model.dtype if value.is_floating_point() else value.dtype)
                for key, value in inputs.items()}

    def _token_log_probabilities(self, inputs: Mapping[str, torch.Tensor], *, policy=False) -> torch.Tensor:
        inputs = self._move_inputs(inputs)
        if self.max_sequence_length is not None and inputs["input_ids"].shape[1] > self.max_sequence_length:
            raise ValueError("Encoded sequence exceeds max_sequence_length; reduce image tokens or prompt length")
        # Qwen3-VL caches multimodal rope_deltas on its inner model. Every new
        # teacher-forced sequence must reconstruct these from its own image grid.
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = None
        forward = self.forward_model if torch.is_grad_enabled() else self.model
        outputs = forward(**inputs, use_cache=False, return_dict=True)
        logits = outputs.logits[:, :-1]
        targets = inputs["input_ids"][:, 1:]
        if logits.shape[:2] != targets.shape:
            raise ValueError("Model logits must align with every next-token target")
        # Chunk the vocabulary projection output in time; avoid a second full
        # [batch, sequence, vocabulary] FP32 allocation.
        values = []
        for start in range(0, targets.shape[1], 64):
            scores = logits[:, start:start + 64].float()
            if policy:
                scores = scores.clone()
                scores[..., list(self.generation.suppress_token_ids)] = -torch.inf
            labels = targets[:, start:start + 64]
            selected = scores.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            values.append(selected - scores.logsumexp(dim=-1))
        return torch.cat(values, dim=1)

    def log_probabilities(self, batch: RolloutBatch) -> torch.Tensor:
        """Evaluate the saved rollout tokens with gradients for this model."""
        return self._token_log_probabilities(batch.model_inputs, policy=self.policy_scoring)

    def sft_loss(self, records: Sequence[Mapping]) -> torch.Tensor:
        """Mean next-token NLL of assistant targets through their first EOS."""
        total, count = self.sft_loss_sum(records)
        return total / count

    def sft_loss_sum(self, records: Sequence[Mapping]) -> tuple[torch.Tensor, int]:
        """Sum target NLL and count tokens for full-batch microbatch normalization."""
        inputs, mask = self.prepare_sft(records)
        return self.sft_prepared_loss_sum(inputs, mask), int(mask.sum())

    def prepare_sft(self, records: Sequence[Mapping]):
        """Encode once and retain CPU inputs/masks before global token counting."""
        prompts = self._encode(records, supervised=False)
        inputs = self._encode(records, supervised=True)
        mask = torch.zeros_like(inputs["attention_mask"][:, 1:], dtype=torch.bool)
        for row in range(len(records)):
            prompt = prompts["input_ids"][row][prompts["attention_mask"][row].bool()]
            full = inputs["input_ids"][row][inputs["attention_mask"][row].bool()]
            if len(full) <= len(prompt) or not torch.equal(full[:len(prompt)], prompt):
                raise ValueError("The assistant training sequence must extend its generation prompt")
            suffix = full[len(prompt):]
            retained = _through_first_eos(suffix, self.generation.eos_token_ids)
            start = inputs["input_ids"].shape[1] - len(full) + len(prompt) - 1
            mask[row, start:start + len(suffix)] = retained
        return {key: value.cpu() for key, value in inputs.items()}, mask.cpu()

    def sft_prepared_loss_sum(self, inputs, mask):
        log_probs = self._token_log_probabilities(inputs)
        return -log_probs[mask.to(log_probs.device)].sum()

    @torch.no_grad()
    def generate(self, records: Sequence[Mapping]) -> RolloutBatch:
        """Generate text and score the retained completion using original logits."""
        training = self.model.training
        self.model.eval()
        try:
            inputs = self._encode(records, supervised=False)
            prefix_width = inputs["input_ids"].shape[1]
            if (self.max_sequence_length is not None
                    and prefix_width + self.generation.max_new_tokens > self.max_sequence_length):
                raise ValueError("Prompt plus generation budget exceeds max_sequence_length")
            inner = getattr(self.model, "model", None)
            if inner is not None and hasattr(inner, "rope_deltas"):
                inner.rope_deltas = None
            sequences = self.model.generate(**inputs, generation_config=self.generation.as_transformers(),
                                            synced_gpus=False)
            generated = sequences[:, prefix_width:]
            if generated.shape[1] == 0:
                raise ValueError("Generation returned no completion tokens")
            retained = _through_first_eos(generated, self.generation.eos_token_ids)
            complete_inputs = dict(inputs)
            complete_inputs["input_ids"] = sequences
            complete_inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"], retained.to(inputs["attention_mask"].dtype)
            ], dim=1)
            for name in ("token_type_ids", "mm_token_type_ids"):
                if name in inputs:
                    complete_inputs[name] = torch.cat([
                        inputs[name], torch.zeros_like(generated, dtype=inputs[name].dtype)
                    ], dim=1)
            mask = torch.zeros_like(sequences[:, 1:], dtype=torch.bool)
            mask[:, prefix_width - 1:] = retained
            raw_log_probs = self._token_log_probabilities(complete_inputs).detach()
            old_log_probs = (self._token_log_probabilities(complete_inputs, policy=True).detach()
                             if self.policy_scoring else raw_log_probs)
            nlls = [float(-raw_log_probs[row][mask[row]].double().mean().item())
                    for row in range(len(records))]
            texts = self.processor.batch_decode(
                [tokens[keep] for tokens, keep in zip(generated, retained)],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            lengths = retained.sum(dim=1).tolist()
            reasons = ["eos" if int(tokens[int(length) - 1]) in self.generation.eos_token_ids else "length"
                       for tokens, length in zip(generated, lengths)]
            return RolloutBatch(list(texts), complete_inputs, mask, old_log_probs, nlls, reasons, lengths)
        finally:
            self.model.train(training)

    def enable_training(self, max_sequence_length: int):
        self.max_sequence_length = max_sequence_length
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if isinstance(getattr(module, "attention_dropout", None), (int, float)):
                module.attention_dropout = 0.0
        for config in (self.model.config, getattr(self.model.config, "text_config", None),
                       getattr(self.model.config, "vision_config", None)):
            if config is not None:
                for name in ("attention_dropout", "hidden_dropout_prob", "dropout"):
                    if hasattr(config, name):
                        setattr(config, name, 0.0)

    def save_pretrained(self, path: str | Path) -> None:
        """Save model and processor together as a Hugging Face checkpoint."""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
