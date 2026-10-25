"""Explicit, resumable single-sequence decoding settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class DecodingConfig:
    mode: str = "greedy"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self):
        if self.mode not in {"greedy", "sample"}:
            raise ValueError("Decoding mode must be greedy or sample")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if self.mode == "greedy" and (self.temperature, self.top_p, self.top_k) != (1.0, 1.0, 0):
            raise ValueError("Sampling settings require --decoding sample")

    def generation_kwargs(self):
        # NLL extraction is for one returned sequence, without beam ancestry.
        kwargs = {"do_sample": self.mode == "sample", "num_beams": 1, "num_return_sequences": 1}
        if self.mode == "sample":
            kwargs.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        return kwargs

    def identity(self):
        return {**asdict(self), **self.generation_kwargs(), "sampling_seed_scope": "input-key-v1",
                "nll_source": "unprocessed_logits_single_sequence_v1"}


def unprocessed_logits(output):
    logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError("Model generation did not return unprocessed logits for completion NLL")
    return logits


def add_decoding_arguments(parser):
    parser.add_argument("--decoding", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)


def config_from_args(args):
    return DecodingConfig(args.decoding, args.temperature, args.top_p, args.top_k)


def input_seed(seed: int, key) -> int:
    """Stable per-input seed, so skipping completed rows cannot change sampling."""
    payload = json.dumps([int(seed), key], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
