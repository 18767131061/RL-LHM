"""DA-CVQ visual reward memory and VGS sampling, independent of a trainer.

All algorithm settings are caller-supplied. Features are fixed cached vectors;
only prototypes and reward memories change. This module uses detached CPU
float64 tensors and a caller-owned CPU generator for numerical state and draws.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _cpu_generator(generator: torch.Generator) -> torch.Generator:
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("generator must be a CPU torch.Generator")
    return generator


class DACVQState:
    """Online diversity admission, difficulty-weighted feature EMA and reward EMA.

    ``update`` consumes one occurrence's mean *clipped reward*, after the policy
    update, never its normalized advantage. Unassigned codes keep their state;
    low-mass codes remain active until optionally revived in place.
    """

    def __init__(
        self,
        *,
        max_codes: int,
        input_dim: int,
        feature_decay: float,
        reward_decay: float,
        difficulty_beta: float,
        dead_code_threshold: float,
        dead_code_replace_probability: float,
        revival_noise_std: float,
        diversity_threshold: float,
        reward_min: float,
        reward_max: float,
        mass_epsilon: float,
        generator: torch.Generator,
    ):
        if isinstance(max_codes, bool) or not isinstance(max_codes, int) or max_codes <= 0:
            raise ValueError("max_codes must be a positive integer")
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.max_codes, self.input_dim = max_codes, input_dim
        for name, value in (
            ("feature_decay", feature_decay), ("reward_decay", reward_decay),
            ("difficulty_beta", difficulty_beta),
            ("dead_code_threshold", dead_code_threshold),
            ("dead_code_replace_probability", dead_code_replace_probability),
            ("revival_noise_std", revival_noise_std),
            ("diversity_threshold", diversity_threshold),
            ("reward_min", reward_min), ("reward_max", reward_max),
            ("mass_epsilon", mass_epsilon),
        ):
            setattr(self, name, _finite(name, value))
        if not 0 <= self.feature_decay < 1 or not 0 <= self.reward_decay < 1:
            raise ValueError("EMA decays must be in [0, 1)")
        if self.difficulty_beta <= 0 or self.mass_epsilon <= 0:
            raise ValueError("difficulty_beta and mass_epsilon must be positive")
        if self.dead_code_threshold < 0 or self.revival_noise_std < 0:
            raise ValueError("dead_code_threshold and revival_noise_std must be nonnegative")
        if not 0 <= self.dead_code_replace_probability <= 1:
            raise ValueError("dead_code_replace_probability must be in [0, 1]")
        if not -1 <= self.diversity_threshold <= 1:
            raise ValueError("diversity_threshold must be in [-1, 1]")
        if self.reward_min >= self.reward_max or not math.isfinite(self.reward_max - self.reward_min):
            raise ValueError("reward bounds must have a finite, positive width")
        self.generator = _cpu_generator(generator)
        self.embedding = torch.zeros((max_codes, input_dim), dtype=torch.float64)
        self.ema_cluster_size = torch.zeros(max_codes, dtype=torch.float64)
        self.ema_weight = torch.zeros_like(self.embedding)
        self.reward_ema = torch.zeros(max_codes, dtype=torch.float64)
        self.reward_seen = torch.zeros(max_codes, dtype=torch.int64)
        self.filled_slots = 0
        self.is_initialized = False

    def normalize(self, vector: torch.Tensor) -> torch.Tensor:
        """Validate one [input_dim] vector and return its detached unit direction."""
        values = torch.as_tensor(vector).detach()
        if values.is_complex():
            raise ValueError("visual vector must be real-valued")
        values = values.to(device="cpu", dtype=torch.float64)
        if values.shape != (self.input_dim,) or not torch.isfinite(values).all():
            raise ValueError("visual vector must be finite with shape [input_dim]")
        scale = values.abs().max()
        if scale == 0:
            raise ValueError("visual vector must be nonzero")
        # Scaling first avoids overflow/underflow in the Euclidean norm.
        values = values / scale
        return values / torch.linalg.vector_norm(values)

    def active_codes(self) -> int:
        return self.filled_slots

    def finalize_initialization(self) -> bool:
        """Freeze the active set at capacity or at the next epoch boundary."""
        if not self.filled_slots:
            return False
        self.is_initialized = True
        return True

    def _reset_code(self, code: int, vector: torch.Tensor) -> None:
        self.embedding[code] = vector
        self.ema_cluster_size[code] = 1
        self.ema_weight[code] = vector
        self.reward_ema[code] = 0
        self.reward_seen[code] = 0

    def online_init_step(self, vector: torch.Tensor) -> bool:
        """Admit a diverse normalized vector while initialization is open."""
        values = self.normalize(vector)
        if self.is_initialized:
            return False
        if self.filled_slots >= self.max_codes:
            self.finalize_initialization()
            return False
        if self.filled_slots:
            similarities = self.embedding[:self.filled_slots] @ values
            if float(similarities.max()) >= self.diversity_threshold:
                return False
        self._reset_code(self.filled_slots, values)
        self.filled_slots += 1
        if self.filled_slots == self.max_codes:
            self.finalize_initialization()
        return True

    def quantize(self, vector: torch.Tensor) -> tuple[int, float]:
        """Return nearest active code and cosine similarity; (-1, 0) if empty."""
        values = self.normalize(vector)
        if not self.filled_slots:
            return -1, 0.0
        similarities = self.embedding[:self.filled_slots] @ values
        code = int(similarities.argmax())
        return code, float(similarities[code])

    def update(self, sample_key: str, vector: torch.Tensor, mean_reward: float) -> int:
        """Update one RID's visual neighborhood; ``sample_key`` is its stable UID.

        UID assignments are deliberately recomputed, not cached: prototypes may
        have moved or been revived since an input's previous visit. If the feature
        accumulator cancels exactly to zero, its direction is undefined; retain
        the previous unit prototype while keeping the prescribed EMA statistics.
        """
        if not isinstance(sample_key, str) or not sample_key.strip():
            raise ValueError("sample_key must be a nonempty persistent UID string")
        values = self.normalize(vector)
        reward = _finite("mean_reward", mean_reward)
        if not self.reward_min <= reward <= self.reward_max:
            raise ValueError("mean_reward must be the mean of rewards within the clipping bounds")
        if not self.is_initialized:
            self.online_init_step(values)
        code, _ = self.quantize(values)
        weight = 1 - (reward - self.reward_min) / (self.reward_max - self.reward_min)
        self.ema_cluster_size[:self.filled_slots].mul_(self.feature_decay)
        self.ema_weight[:self.filled_slots].mul_(self.feature_decay)
        mass = self.ema_cluster_size[code] + (1 - self.feature_decay) * weight
        accumulator = self.ema_weight[code] + (1 - self.feature_decay) * weight * values
        self.ema_cluster_size[code] = mass
        self.ema_weight[code] = accumulator
        guarded_mean = accumulator / max(float(mass), self.mass_epsilon)
        if guarded_mean.abs().max() > 0:
            self.embedding[code] = self.normalize(guarded_mean)
        elif accumulator.abs().max() > 0:
            # The positive mass normalizer cannot change direction; use mu
            # itself if division underflows for an unusually large guard.
            self.embedding[code] = self.normalize(accumulator)
        if self.reward_seen[code] == 0:
            self.reward_ema[code] = reward
        else:
            self.reward_ema[code] = self.reward_decay * self.reward_ema[code] + (1 - self.reward_decay) * reward
        self.reward_seen[code] += 1
        if self.is_initialized:
            draw = float(torch.rand((), generator=self.generator))
            if draw < self.dead_code_replace_probability:
                dead = (self.ema_cluster_size[:self.filled_slots] < self.dead_code_threshold).nonzero().flatten()
                if dead.numel():
                    noise = torch.randn(values.shape, dtype=values.dtype, generator=self.generator)
                    # Algebraically identical to normalize(z + sigma*noise),
                    # with scaling to keep a large finite sigma from overflowing.
                    scale = max(1.0, self.revival_noise_std)
                    revived = values / scale + (self.revival_noise_std / scale) * noise
                    if revived.abs().max() == 0:
                        revived = values
                    self._reset_code(int(dead[0]), self.normalize(revived))
        return code

    def difficulty_weights(self) -> torch.Tensor:
        """Active-code softmax(-m/tau_s), where difficulty_beta = 1/tau_s."""
        memories = self.reward_ema[:self.filled_slots]
        if not memories.numel():
            return memories.clone()
        # The shift leaves softmax unchanged and gives at least one zero logit.
        logits = -(memories - memories.min()) * self.difficulty_beta
        return torch.softmax(logits, dim=0)


def vgs_sampling_weights(
    state: DACVQState,
    vectors: Sequence[torch.Tensor | None],
    *,
    missing_feature_multiplier: float,
    weight_floor: float,
    use_vgs: bool,
) -> torch.Tensor:
    """Map active-code preferences to sample weights, then apply the floor.

    Missing features get ``missing_feature_multiplier * max(a_k)`` before
    flooring. Disabled VGS or an uninitialized codebook yields all-one weights.
    Empty input yields an empty vector.
    """
    multiplier = _finite("missing_feature_multiplier", missing_feature_multiplier)
    floor = _finite("weight_floor", weight_floor)
    if multiplier <= 0 or floor <= 0:
        raise ValueError("missing_feature_multiplier and weight_floor must be positive")
    if not use_vgs or not state.is_initialized or len(vectors) == 0:
        return torch.ones(len(vectors), dtype=torch.float64)
    difficulty = state.difficulty_weights()
    weights = torch.full((len(vectors),), multiplier * float(difficulty.max()), dtype=torch.float64)
    for position, vector in enumerate(vectors):
        if vector is not None:
            code, _ = state.quantize(vector)
            weights[position] = difficulty[code]
    return weights.clamp_min(floor)


def vgs_sampling_probabilities(
    state: DACVQState,
    vectors: Sequence[torch.Tensor | None],
    *,
    missing_feature_multiplier: float,
    weight_floor: float,
    use_vgs: bool,
) -> torch.Tensor:
    """Normalize VGS sample weights into q_i; empty input stays empty."""
    weights = vgs_sampling_weights(
        state, vectors, missing_feature_multiplier=missing_feature_multiplier,
        weight_floor=weight_floor, use_vgs=use_vgs,
    )
    if not weights.numel():
        return weights
    scaled = weights / weights.max()
    return scaled / scaled.sum()


class VGSSampler:
    """Epoch index draws from fixed cached features, without dataset/trainer IO.

    A new sampler with an uninitialized codebook draws a random permutation.
    Later epochs finalize any nonempty online codebook, then draw exactly N
    positions using the explicit replacement setting and VGS (or uniform weights while empty).
    ``epoch_count`` is explicit so a resumed sampler need not replay epoch zero.
    """

    def __init__(
        self,
        state: DACVQState,
        vectors: Sequence[torch.Tensor | None],
        *,
        missing_feature_multiplier: float,
        weight_floor: float,
        generator: torch.Generator,
        use_vgs: bool,
        epoch_count: int,
        replacement: bool,
    ):
        if isinstance(epoch_count, bool) or not isinstance(epoch_count, int) or epoch_count < 0:
            raise ValueError("epoch_count must be a nonnegative integer")
        self.state = state
        self.vectors = tuple(None if z is None else state.normalize(z) for z in vectors)
        self.missing_feature_multiplier = missing_feature_multiplier
        self.weight_floor = weight_floor
        self.generator = _cpu_generator(generator)
        self.use_vgs = use_vgs
        self.epoch_count = epoch_count
        if type(replacement) is not bool:
            raise ValueError("replacement must be boolean")
        self.replacement = replacement
        # Validate VGS settings before a draw can mutate the epoch count.
        self.probabilities()

    def probabilities(self) -> torch.Tensor:
        return vgs_sampling_probabilities(
            self.state, self.vectors,
            missing_feature_multiplier=self.missing_feature_multiplier,
            weight_floor=self.weight_floor, use_vgs=self.use_vgs,
        )

    def sample_epoch(self) -> torch.Tensor:
        """Return dataset positions; a zero-length dataset consumes no epoch."""
        size = len(self.vectors)
        if not size:
            return torch.empty(0, dtype=torch.int64)
        if self.use_vgs and self.epoch_count > 0 and not self.state.is_initialized:
            self.state.finalize_initialization()
        if not self.use_vgs and not self.replacement:
            sampled = torch.randperm(size, generator=self.generator)
        elif self.use_vgs and self.epoch_count == 0 and not self.state.is_initialized:
            sampled = torch.randperm(size, generator=self.generator)
        else:
            sampled = torch.multinomial(
                self.probabilities(), num_samples=size, replacement=self.replacement,
                generator=self.generator,
            )
        self.epoch_count += 1
        return sampled
