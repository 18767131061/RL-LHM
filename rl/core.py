"""Framework-independent SHAE and DA-CVQ/VGS algorithm core."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

EPSILON = 1e-6


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ShaeBaselineStore:
    """Per-sample exponential moving-average baseline used by SHAE."""

    def __init__(self, beta: float = 0.95, path: str | Path | None = None):
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.beta = float(beta)
        self.path = Path(path).expanduser().resolve() if path else None
        self.entries: dict[str, dict[str, float | int]] = {}
        self.update_count = 0
        if self.path and self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for key, value in raw.items():
                self.entries[str(key)] = {
                    "ema": float(value["ema"] if isinstance(value, dict) else value),
                    "seen": int(value.get("seen", 1) if isinstance(value, dict) else 1),
                }

    def get(self, sample_key: str) -> float | None:
        entry = self.entries.get(str(sample_key))
        return float(entry["ema"]) if entry and int(entry["seen"]) > 0 else None

    def update(self, sample_key: str, reward_mean: float) -> None:
        key = str(sample_key)
        if key not in self.entries:
            self.entries[key] = {"ema": float(reward_mean), "seen": 1}
        else:
            entry = self.entries[key]
            entry["ema"] = self.beta * float(entry["ema"]) + (1.0 - self.beta) * float(reward_mean)
            entry["seen"] = int(entry["seen"]) + 1
        self.update_count += 1

    def save(self) -> None:
        if self.path is not None:
            _atomic_write_json(self.path, self.entries)


def _group_positions(index: Iterable[object]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for position, key in enumerate(index):
        grouped[str(key)].append(position)
    return grouped


def compute_shae_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[object] | np.ndarray,
    *,
    sample_index: Sequence[object] | np.ndarray | None = None,
    store: ShaeBaselineStore | None = None,
    gate_mid: float = 0.05,
    gate_temperature: float = 0.015,
    noise_std: float = 1e-3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mix per-visit advantages with a persistent per-sample EMA.

    ``index`` identifies each rollout visit, while ``sample_index`` identifies
    the persistent input. Callers using one visit per input can omit the latter.
    Repeated visits update the EMA in first-occurrence batch order, each using
    its own G responses; they never form one larger normalization group.
    The EMA branch uses Gaussian jitter with standard deviation 0.001. Supply
    a seeded ``generator`` to control the standalone function's random draws.
    """
    if token_level_rewards.shape != response_mask.shape or token_level_rewards.ndim != 2:
        raise ValueError("rewards and response mask must have the same [batch, tokens] shape")
    if len(index) != token_level_rewards.shape[0]:
        raise ValueError("index length must equal batch size")
    sample_index = index if sample_index is None else sample_index
    if len(sample_index) != len(index):
        raise ValueError("sample_index length must equal batch size")
    groups = []
    for visit_key, positions in _group_positions(index).items():
        keys = {str(sample_index[position]) for position in positions}
        if len(keys) != 1:
            raise ValueError(f"Rollout group {visit_key!r} contains multiple sample keys")
        groups.append((keys.pop(), positions))
    if gate_temperature <= 0.0 or noise_std < 0.0:
        raise ValueError("gate_temperature must be positive and noise_std non-negative")
    baseline_store = store or ShaeBaselineStore()
    scores = token_level_rewards.sum(dim=-1).float()
    outcome = torch.zeros_like(scores)
    for sample_key, positions in groups:
        position_tensor = torch.tensor(positions, device=scores.device, dtype=torch.long)
        sample_scores = scores.index_select(0, position_tensor)
        mean = sample_scores.mean()
        standard_deviation = sample_scores.std(unbiased=False)
        group_advantage = (sample_scores - mean) / (standard_deviation + EPSILON)
        previous = baseline_store.get(sample_key)
        baseline = float(previous) if previous is not None else float(mean.item())
        ema_advantage = sample_scores - baseline
        if noise_std:
            noise = torch.randn(
                ema_advantage.shape,
                dtype=ema_advantage.dtype,
                device=ema_advantage.device,
                generator=generator,
            )
            ema_advantage = ema_advantage + noise_std * noise
        ema_advantage = ema_advantage / (ema_advantage.std(unbiased=False) + EPSILON)
        alpha = 1.0 / (
            1.0
            + math.exp(
                -(float(standard_deviation.item()) - gate_mid) / max(gate_temperature, EPSILON)
            )
        )
        outcome.index_copy_(
            0,
            position_tensor,
            alpha * group_advantage + (1.0 - alpha) * ema_advantage,
        )
        baseline_store.update(sample_key, float(mean.item()))
    return outcome.unsqueeze(-1) * response_mask.float()


class DACVQState:
    """Difficulty-aware online vector quantizer used by VGS sampling."""

    def __init__(
        self,
        *,
        max_codes: int,
        input_dim: int,
        feature_decay: float = 0.95,
        reward_decay: float = 0.90,
        difficulty_beta: float = 3.0,
        dead_code_threshold: float = 0.25,
        dead_code_replace_probability: float = 0.3,
        reward_initial_value: float = 0.0,
        seed: int = 42,
        revival_noise_std: float = 1e-3,
    ):
        if max_codes <= 0 or input_dim <= 0:
            raise ValueError("max_codes and input_dim must be positive")
        if revival_noise_std < 0.0:
            raise ValueError("revival_noise_std must be non-negative")
        self.max_codes = int(max_codes)
        self.input_dim = int(input_dim)
        self.feature_decay = float(feature_decay)
        self.reward_decay = float(reward_decay)
        self.difficulty_beta = float(difficulty_beta)
        self.dead_code_threshold = float(dead_code_threshold)
        self.dead_code_replace_probability = float(dead_code_replace_probability)
        self.reward_initial_value = float(reward_initial_value)
        self.revival_noise_std = float(revival_noise_std)
        self.generator = torch.Generator().manual_seed(int(seed))
        initial = torch.randn(self.max_codes, self.input_dim, generator=self.generator)
        self.embedding = torch.nn.functional.normalize(initial, p=2, dim=1)
        self.ema_cluster_size = torch.ones(self.max_codes)
        self.ema_weight = self.embedding.clone()
        self.reward_ema = torch.full((self.max_codes,), self.reward_initial_value)
        self.reward_seen = torch.zeros(self.max_codes)
        self.sample_to_code: dict[str, int] = {}
        self.filled_slots = 0
        self.is_initialized = False

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        values = vector.detach().float().reshape(-1)
        if not torch.isfinite(values).all() or float(values.norm().item()) <= 0.0:
            raise ValueError("DA-CVQ vectors must be finite and non-zero")
        return torch.nn.functional.normalize(values, p=2, dim=0)

    def active_codes(self) -> int:
        if self.filled_slots:
            return min(self.filled_slots, self.max_codes)
        return self.max_codes if self.is_initialized else 0

    def finalize_initialization(self) -> bool:
        if self.active_codes() <= 0:
            return False
        self.filled_slots = self.active_codes()
        self.is_initialized = True
        return True

    def online_init_step(self, vector: torch.Tensor, similarity_threshold: float = 0.85) -> bool:
        if self.is_initialized:
            return False
        if self.filled_slots >= self.max_codes:
            self.finalize_initialization()
            return False
        values = self._normalize(vector)
        if values.numel() != self.input_dim:
            raise ValueError("DA-CVQ vector dimension does not match input_dim")
        if self.filled_slots and float(torch.mv(self.embedding[: self.filled_slots], values).max()) >= similarity_threshold:
            return False
        slot = self.filled_slots
        self.embedding[slot] = values
        self.ema_weight[slot] = values
        self.filled_slots += 1
        if self.filled_slots >= self.max_codes:
            self.finalize_initialization()
        return True

    def quantize(self, vector: torch.Tensor) -> tuple[int, float]:
        values = self._normalize(vector)
        active = self.active_codes()
        if active <= 0:
            return -1, 0.0
        similarities = torch.mv(self.embedding[:active], values)
        code = int(torch.argmax(similarities).item())
        return code, float(similarities[code].item())

    def update(self, sample_key: str, vector: torch.Tensor, mean_reward: float) -> int:
        values = self._normalize(vector)
        if not self.is_initialized:
            self.online_init_step(values)
        code, _ = self.quantize(values)
        if code < 0:
            return -1
        difficulty_weight = 1.0 - max(0.0, min(1.0, (float(mean_reward) + 6.0) / 9.0))
        self.ema_cluster_size[code] = (
            self.feature_decay * self.ema_cluster_size[code]
            + (1.0 - self.feature_decay) * difficulty_weight
        )
        self.ema_weight[code] = (
            self.feature_decay * self.ema_weight[code]
            + (1.0 - self.feature_decay) * values * difficulty_weight
        )
        denominator = float(self.ema_cluster_size[code].item())
        if denominator > EPSILON:
            self.embedding[code] = self._normalize(self.ema_weight[code] / denominator)
        if float(self.reward_seen[code].item()) <= 0.0:
            self.reward_ema[code] = float(mean_reward)
        else:
            self.reward_ema[code] = (
                self.reward_decay * self.reward_ema[code]
                + (1.0 - self.reward_decay) * float(mean_reward)
            )
        self.reward_seen[code] += 1.0
        self.sample_to_code[str(sample_key)] = code
        active = self.active_codes()
        if (
            self.is_initialized
            and active > 0
            and float(torch.rand((), generator=self.generator).item())
            < self.dead_code_replace_probability
        ):
            dead = (self.ema_cluster_size[:active] < self.dead_code_threshold).nonzero(as_tuple=True)[0]
            if len(dead):
                replacement = int(dead[0].item())
                noise = torch.randn(values.shape, generator=self.generator)
                revived = self._normalize(values + self.revival_noise_std * noise)
                self.embedding[replacement] = revived
                self.ema_weight[replacement] = revived
                self.ema_cluster_size[replacement] = 1.0
                self.reward_ema[replacement] = self.reward_initial_value
                self.reward_seen[replacement] = 0.0
        return code

    def difficulty_weights(self) -> torch.Tensor:
        active = self.active_codes()
        if active <= 0:
            return torch.ones(1)
        return torch.softmax(-self.reward_ema[:active] * self.difficulty_beta, dim=0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_codes": self.max_codes,
            "input_dim": self.input_dim,
            "embedding": self.embedding.tolist(),
            "ema_cluster_size": self.ema_cluster_size.tolist(),
            "ema_weight": self.ema_weight.tolist(),
            "reward_ema": self.reward_ema.tolist(),
            "reward_seen": self.reward_seen.tolist(),
            "sample_to_code": self.sample_to_code,
            "filled_slots": self.filled_slots,
            "is_initialized": self.is_initialized,
            "generator_state": self.generator.get_state().tolist(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("max_codes", -1)) != self.max_codes or int(state.get("input_dim", -1)) != self.input_dim:
            raise ValueError("DA-CVQ state/config mismatch")
        expected_shapes = {
            "embedding": (self.max_codes, self.input_dim),
            "ema_cluster_size": (self.max_codes,),
            "ema_weight": (self.max_codes, self.input_dim),
            "reward_ema": (self.max_codes,),
            "reward_seen": (self.max_codes,),
        }
        tensors = {name: torch.tensor(state[name], dtype=torch.float32) for name in expected_shapes}
        for name, values in tensors.items():
            if tuple(values.shape) != expected_shapes[name] or not torch.isfinite(values).all():
                raise ValueError(f"Invalid DA-CVQ state tensor: {name}")
        sample_to_code = {str(key): int(value) for key, value in state.get("sample_to_code", {}).items()}
        if any(code < 0 or code >= self.max_codes for code in sample_to_code.values()):
            raise ValueError("DA-CVQ state contains an invalid code index")
        self.embedding = tensors["embedding"]
        self.ema_cluster_size = tensors["ema_cluster_size"]
        self.ema_weight = tensors["ema_weight"]
        self.reward_ema = tensors["reward_ema"]
        self.reward_seen = tensors["reward_seen"]
        self.sample_to_code = sample_to_code
        self.filled_slots = int(state.get("filled_slots", 0))
        self.is_initialized = bool(state.get("is_initialized", False))
        if not 0 <= self.filled_slots <= self.max_codes:
            raise ValueError("Invalid DA-CVQ filled_slots")
        if "generator_state" in state:
            self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))


def vgs_sampling_weights(
    state: DACVQState,
    vectors: Sequence[torch.Tensor | None],
    *,
    use_vgs: bool = True,
) -> torch.Tensor:
    if not vectors:
        return torch.ones(1)
    if not use_vgs or not state.is_initialized:
        return torch.ones(len(vectors))
    difficulty = state.difficulty_weights()
    weights = torch.full((len(vectors),), float(difficulty.max().item()) * 1.5)
    for position, vector in enumerate(vectors):
        if vector is None:
            continue
        code, _ = state.quantize(vector)
        weights[position] = difficulty[code]
    return weights.clamp_min(1e-4)
