from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import register_adv_est


EPS = 1e-6
STATE_FORMAT_VERSION = 1


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cfg_get(config: Optional[DictConfig], key: str, default: Any) -> Any:
    if config is None:
        return default
    try:
        return config.get(key, default)
    except Exception:
        return default


def _to_path(path_like: Any) -> Optional[Path]:
    if not path_like:
        return None
    return Path(str(path_like)).expanduser().resolve()


def _safe_float_list(values: Any) -> Optional[torch.Tensor]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return None
        return values.detach().float().cpu().view(-1)
    if not isinstance(values, (list, tuple)):
        return None
    if len(values) == 0:
        return None
    try:
        return torch.tensor([float(v) for v in values], dtype=torch.float32)
    except Exception:
        return None


def _group_positions(index: Iterable[Any]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for pos, key in enumerate(index):
        grouped[str(key)].append(pos)
    return grouped


class ShaeBaselineStore:
    def __init__(self, beta: float, path: Optional[Path] = None):
        self.beta = float(beta)
        self.path = path
        self.entries: Dict[str, Dict[str, float]] = {}
        self.update_count = 0
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if self._loaded or self.path is None or (not self.path.exists()):
            self._loaded = True
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._loaded = True
            return
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, dict):
                    ema = float(value.get("ema", 0.0))
                    seen = int(value.get("seen", 0))
                else:
                    ema = float(value)
                    seen = 1
                self.entries[str(key)] = {"ema": ema, "seen": seen}
        self._loaded = True

    def get(self, sample_key: str) -> Optional[float]:
        item = self.entries.get(str(sample_key))
        if not item:
            return None
        seen = int(item.get("seen", 0))
        if seen <= 0:
            return None
        return float(item.get("ema", 0.0))

    def update(self, sample_key: str, reward_mean: float) -> None:
        key = str(sample_key)
        if key not in self.entries:
            self.entries[key] = {"ema": float(reward_mean), "seen": 1}
        else:
            old = float(self.entries[key]["ema"])
            self.entries[key]["ema"] = float(
                self.beta * old + (1.0 - self.beta) * float(reward_mean)
            )
            self.entries[key]["seen"] = int(self.entries[key]["seen"]) + 1
        self.update_count += 1

    def maybe_save(self, save_every: int) -> None:
        if self.path is None:
            return
        if save_every > 0 and self.update_count % int(save_every) != 0:
            return
        _atomic_write_json(self.path, self.entries)


_SHAE_STORES: Dict[str, ShaeBaselineStore] = {}


def _get_shae_store(config: Optional[DictConfig]) -> ShaeBaselineStore:
    beta = float(_cfg_get(config, "shae_baseline_beta", 0.95))
    path = _to_path(_cfg_get(config, "shae_baseline_path", ""))
    key = f"{path or 'memory_only'}::{beta}"
    if key not in _SHAE_STORES:
        _SHAE_STORES[key] = ShaeBaselineStore(beta=beta, path=path)
    return _SHAE_STORES[key]


@register_adv_est("shae")
def compute_shae_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[DictConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1).float()
    response_mask = response_mask.float()

    gate_mid = float(_cfg_get(config, "shae_gate_mid", 0.05))
    gate_temp = float(_cfg_get(config, "shae_gate_temp", 0.015))
    noise_std = float(_cfg_get(config, "shae_noise_std", 1e-3))
    save_every = int(_cfg_get(config, "shae_save_every", 200))

    store = _get_shae_store(config)
    outcome_adv = torch.zeros_like(scores)

    for sample_key, positions in _group_positions(index).items():
        pos_tensor = torch.tensor(positions, device=scores.device, dtype=torch.long)
        sample_scores = scores.index_select(0, pos_tensor)
        sample_mean = float(sample_scores.mean().item())
        sample_std = float(sample_scores.std(unbiased=False).item()) if sample_scores.numel() > 1 else 0.0

        grp_adv = (sample_scores - sample_scores.mean()) / (sample_scores.std(unbiased=False) + EPS)
        baseline = store.get(sample_key)
        baseline_value = float(baseline) if baseline is not None else sample_mean
        ema_adv = sample_scores - baseline_value
        if noise_std > 0:
            ema_adv = ema_adv + noise_std * torch.randn_like(ema_adv)
        ema_adv = ema_adv / (ema_adv.std(unbiased=False) + EPS)

        alpha = float(1.0 / (1.0 + math.exp(-((sample_std - gate_mid) / max(gate_temp, EPS)))))
        mixed_adv = alpha * grp_adv + (1.0 - alpha) * ema_adv
        outcome_adv.index_copy_(0, pos_tensor, mixed_adv)

        store.update(sample_key, sample_mean)

    store.maybe_save(save_every=save_every)
    outcome_adv = outcome_adv.unsqueeze(-1) * response_mask
    return outcome_adv, outcome_adv


class DACVQState:
    def __init__(
        self,
        max_codes: int,
        input_dim: int,
        feat_decay: float,
        reward_decay: float,
        difficulty_beta: float,
        dead_code_thresh: float,
        dead_code_replace_prob: float,
        reward_init: float,
        seed: int,
        revival_noise_std: float = 1e-3,
    ):
        self.max_codes = int(max_codes)
        self.input_dim = int(input_dim)
        self.feat_decay = float(feat_decay)
        self.reward_decay = float(reward_decay)
        self.difficulty_beta = float(difficulty_beta)
        self.dead_code_thresh = float(dead_code_thresh)
        self.dead_code_replace_prob = float(dead_code_replace_prob)
        self.reward_init = float(reward_init)
        self.revival_noise_std = float(revival_noise_std)
        if self.revival_noise_std < 0:
            raise ValueError("revival_noise_std must be non-negative")
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

        emb = torch.randn(
            self.max_codes,
            self.input_dim,
            dtype=torch.float32,
            generator=self.generator,
        )
        self.embedding = torch.nn.functional.normalize(emb, p=2, dim=1)
        self.ema_cluster_size = torch.ones(self.max_codes, dtype=torch.float32)
        self.ema_w = self.embedding.clone()
        self.reward_ema = torch.full((self.max_codes,), self.reward_init, dtype=torch.float32)
        self.reward_seen = torch.zeros(self.max_codes, dtype=torch.float32)
        self.sample_to_code: Dict[str, int] = {}
        self.filled_slots = 0
        self.is_initialized = False

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(z.float().view(-1), p=2, dim=0)

    def active_codes(self) -> int:
        if self.filled_slots > 0:
            return min(self.filled_slots, self.max_codes)
        if self.is_initialized:
            return self.max_codes
        return 0

    def finalize_initialization(self) -> bool:
        active = self.active_codes()
        if active <= 0:
            return False
        self.filled_slots = active
        self.is_initialized = True
        return True

    def online_init_step(self, z: torch.Tensor) -> bool:
        if self.is_initialized:
            return False

        # Capacity is a hard boundary: once all slots are active, subsequent
        # samples must be assigned to an existing code instead of repeatedly
        # overwriting the final slot until the next epoch starts.
        if self.filled_slots >= self.max_codes:
            self.finalize_initialization()
            return False

        vec = self._normalize(z)
        if self.filled_slots == 0:
            self.embedding[0] = vec
            self.ema_w[0] = vec
            self.filled_slots = 1
            if self.filled_slots >= self.max_codes:
                self.finalize_initialization()
            return True

        active = self.embedding[: self.filled_slots]
        sims = torch.mv(active, vec)
        max_sim = float(sims.max().item())
        if max_sim >= 0.85:
            return False

        idx = self.filled_slots
        self.embedding[idx] = vec
        self.ema_w[idx] = vec
        self.filled_slots += 1
        if self.filled_slots >= self.max_codes:
            self.finalize_initialization()
        return True

    def quantize(self, z: torch.Tensor) -> tuple[int, float]:
        z = self._normalize(z)
        active = self.active_codes()
        if active <= 0:
            return -1, 0.0
        sims = torch.mv(self.embedding[:active], z)
        code = int(torch.argmax(sims).item())
        sim = float(sims[code].item())
        return code, sim

    def update(self, sample_key: str, z: torch.Tensor, mean_reward: float) -> int:
        z = self._normalize(z)
        if not self.is_initialized:
            self.online_init_step(z)

        code, _ = self.quantize(z)
        if code < 0:
            return -1

        difficulty_w = 1.0 - max(0.0, min(1.0, (float(mean_reward) + 6.0) / 9.0))

        self.ema_cluster_size[code] = (
            self.ema_cluster_size[code] * self.feat_decay
            + (1.0 - self.feat_decay) * difficulty_w
        )
        self.ema_w[code] = self.ema_w[code] * self.feat_decay + (1.0 - self.feat_decay) * z * difficulty_w
        denom = self.ema_cluster_size[code].item()
        if denom > EPS:
            self.embedding[code] = self._normalize(self.ema_w[code] / max(denom, EPS))

        if self.reward_seen[code].item() <= 0:
            self.reward_ema[code] = float(mean_reward)
        else:
            self.reward_ema[code] = (
                self.reward_decay * self.reward_ema[code] + (1.0 - self.reward_decay) * float(mean_reward)
            )
        self.reward_seen[code] += 1.0
        self.sample_to_code[str(sample_key)] = code

        active = self.active_codes()
        replace_draw = float(torch.rand((), generator=self.generator).item())
        if self.is_initialized and active > 0 and replace_draw < self.dead_code_replace_prob:
            dead = (self.ema_cluster_size[:active] < self.dead_code_thresh).nonzero(as_tuple=True)[0]
            if len(dead) > 0:
                repl = int(dead[0].item())
                noise = torch.randn(z.shape, dtype=z.dtype, generator=self.generator)
                revived = self._normalize(z + self.revival_noise_std * noise)
                self.embedding[repl] = revived
                self.ema_w[repl] = revived
                self.ema_cluster_size[repl] = 1.0
                self.reward_ema[repl] = self.reward_init
                self.reward_seen[repl] = 0.0
        return code

    def difficulty_weights(self) -> torch.Tensor:
        active = self.active_codes()
        if active <= 0:
            return torch.ones(1, dtype=torch.float32)
        logits = -self.reward_ema[:active] * self.difficulty_beta
        return torch.softmax(logits, dim=0)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_codes": self.max_codes,
            "num_codes": self.max_codes,
            "input_dim": self.input_dim,
            "feat_decay": self.feat_decay,
            "reward_decay": self.reward_decay,
            "difficulty_beta": self.difficulty_beta,
            "dead_code_thresh": self.dead_code_thresh,
            "dead_code_replace_prob": self.dead_code_replace_prob,
            "reward_init": self.reward_init,
            "revival_noise_std": self.revival_noise_std,
            "embedding": self.embedding.tolist(),
            "ema_cluster_size": self.ema_cluster_size.tolist(),
            "ema_w": self.ema_w.tolist(),
            "reward_ema": self.reward_ema.tolist(),
            "reward_seen": self.reward_seen.tolist(),
            "sample_to_code": self.sample_to_code,
            "filled_slots": self.filled_slots,
            "is_initialized": self.is_initialized,
            "generator_state": self.generator.get_state().tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        saved_max_codes = int(state.get("max_codes", state.get("num_codes", -1)))
        saved_input_dim = int(state.get("input_dim", -1))
        if saved_max_codes != self.max_codes or saved_input_dim != self.input_dim:
            raise ValueError(
                "DA-CVQ state/config mismatch: "
                f"saved=({saved_max_codes}, {saved_input_dim}) "
                f"configured=({self.max_codes}, {self.input_dim})"
            )

        tensors = {
            "embedding": torch.tensor(state["embedding"], dtype=torch.float32),
            "ema_cluster_size": torch.tensor(state["ema_cluster_size"], dtype=torch.float32),
            "ema_w": torch.tensor(state["ema_w"], dtype=torch.float32),
            "reward_ema": torch.tensor(state["reward_ema"], dtype=torch.float32),
            "reward_seen": torch.tensor(state["reward_seen"], dtype=torch.float32),
        }
        expected_shapes = {
            "embedding": (self.max_codes, self.input_dim),
            "ema_cluster_size": (self.max_codes,),
            "ema_w": (self.max_codes, self.input_dim),
            "reward_ema": (self.max_codes,),
            "reward_seen": (self.max_codes,),
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(
                    f"Invalid DA-CVQ {name} shape: {tuple(tensor.shape)} != {expected_shapes[name]}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"DA-CVQ state contains non-finite values in {name}")

        sample_to_code = {str(k): int(v) for k, v in state.get("sample_to_code", {}).items()}
        if any(code < 0 or code >= self.max_codes for code in sample_to_code.values()):
            raise ValueError("DA-CVQ state contains an out-of-range sample_to_code value")
        filled_slots = int(state.get("filled_slots", 0))
        if not 0 <= filled_slots <= self.max_codes:
            raise ValueError(f"Invalid DA-CVQ filled_slots value: {filled_slots}")

        self.embedding = tensors["embedding"]
        self.ema_cluster_size = tensors["ema_cluster_size"]
        self.ema_w = tensors["ema_w"]
        self.reward_ema = tensors["reward_ema"]
        self.reward_seen = tensors["reward_seen"]
        self.sample_to_code = sample_to_code
        self.filled_slots = filled_slots
        self.is_initialized = bool(state.get("is_initialized", False))
        generator_state = state.get("generator_state")
        if generator_state is not None:
            self.generator.set_state(torch.tensor(generator_state, dtype=torch.uint8))


class VLGSHybridSampler(AbstractCurriculumSampler):
    def __init__(self, data_source, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config
        sampler_cfg = data_config.get("sampler", {}) or {}

        self.use_vgs = bool(sampler_cfg.get("use_vgs", True))
        self.state_path = _to_path(sampler_cfg.get("state_path", ""))
        self.save_every = int(sampler_cfg.get("save_every", 100))
        self.update_count = 0
        self.epoch_count = 0
        self.seed = int(data_config.get("seed", 42))
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.sample_keys: List[str] = []
        self.sample_vectors: List[Optional[torch.Tensor]] = []
        self.uid_to_pos: Dict[str, int] = {}

        input_dim = int(sampler_cfg.get("vq_input_dim", 1152))
        for pos in range(len(self.data_source)):
            row = self.data_source[pos]
            sample_key, z = self._extract_row_info(row=row, fallback_pos=pos)
            self.sample_keys.append(sample_key)
            self.sample_vectors.append(z)
            self.uid_to_pos[sample_key] = pos
            if z is not None:
                input_dim = int(z.numel())

        self.vq = DACVQState(
            max_codes=int(sampler_cfg.get("vq_code_capacity", sampler_cfg.get("vq_num_codes", 64))),
            input_dim=input_dim,
            feat_decay=float(sampler_cfg.get("vq_feat_decay", 0.95)),
            reward_decay=float(sampler_cfg.get("vq_reward_decay", 0.90)),
            difficulty_beta=float(sampler_cfg.get("vq_difficulty_beta", 3.0)),
            dead_code_thresh=float(sampler_cfg.get("vq_dead_code_thresh", 0.25)),
            dead_code_replace_prob=float(sampler_cfg.get("vq_dead_code_replace_prob", 0.3)),
            reward_init=float(sampler_cfg.get("vq_reward_init", 0.0)),
            seed=self.seed,
            revival_noise_std=float(sampler_cfg.get("vq_revival_noise_std", 1e-3)),
        )
        if self.state_path is not None and self.state_path.exists():
            try:
                saved = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.load_state_dict(saved)
            except Exception as error:
                raise RuntimeError(
                    f"Unable to resume DA-CVQ/VGS state from {self.state_path}: {error}"
                ) from error

    def _extract_row_info(self, row: Dict[str, Any], fallback_pos: int) -> tuple[str, Optional[torch.Tensor]]:
        extra_info = row.get("extra_info") or {}
        sample_key = str(row.get("uid") or extra_info.get("sample_uid") or extra_info.get("index") or fallback_pos)
        z = _safe_float_list(extra_info.get("visual_z"))
        return sample_key, z

    def _save(self) -> None:
        if self.state_path is None:
            return
        _atomic_write_json(self.state_path, self.state_dict())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format_version": STATE_FORMAT_VERSION,
            "dataset_size": len(self.data_source),
            "seed": self.seed,
            "update_count": self.update_count,
            "epoch_count": self.epoch_count,
            "generator_state": self.generator.get_state().tolist(),
            "vq": self.vq.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("format_version", -1)) != STATE_FORMAT_VERSION:
            raise ValueError(f"Unsupported sampler state format: {state.get('format_version')}")
        if int(state.get("dataset_size", -1)) != len(self.data_source):
            raise ValueError(
                f"Sampler dataset size changed: {state.get('dataset_size')} != {len(self.data_source)}"
            )
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError(f"Sampler seed changed: {state.get('seed')} != {self.seed}")
        self.vq.load_state_dict(state["vq"])
        self.update_count = int(state.get("update_count", 0))
        self.epoch_count = int(state.get("epoch_count", 0))
        generator_state = state.get("generator_state")
        if generator_state is not None:
            tensor_state = torch.tensor(generator_state, dtype=torch.uint8)
            self.generator.set_state(tensor_state)

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        if n <= 0:
            return iter([])
        if self.epoch_count > 0 and (not self.vq.is_initialized) and self.vq.active_codes() > 0:
            self.vq.finalize_initialization()
        if self.epoch_count == 0 and not self.vq.is_initialized:
            sampled = torch.randperm(n, generator=self.generator).tolist()
        else:
            weights = self._build_weights()
            sampled = torch.multinomial(weights, num_samples=n, replacement=True, generator=self.generator).tolist()
        self.epoch_count += 1
        return iter(sampled)

    def _build_weights(self) -> torch.Tensor:
        n = len(self.data_source)
        if n <= 0:
            return torch.ones(1, dtype=torch.float32)
        if not self.use_vgs or not self.vq.is_initialized:
            return torch.ones(n, dtype=torch.float32)
        diff = self.vq.difficulty_weights()
        explore_weight = float(diff.max().item()) * 1.5
        weights = torch.full((n,), explore_weight, dtype=torch.float32)
        for pos, z in enumerate(self.sample_vectors):
            if z is None:
                continue
            code, _ = self.vq.quantize(z)
            weights[pos] = float(diff[code].item())
        weights = weights.clamp_min(1e-4)
        return weights

    def update(self, batch: DataProto) -> None:
        if "token_level_scores" not in batch.batch:
            return
        rewards = batch.batch["token_level_scores"].sum(dim=-1).detach().float().cpu()
        if "uid" in batch.non_tensor_batch:
            raw_keys = list(batch.non_tensor_batch["uid"])
        else:
            extra_infos = list(batch.non_tensor_batch.get("extra_info", []))
            raw_keys = [
                str((x or {}).get("sample_uid") or (x or {}).get("index") or idx) for idx, x in enumerate(extra_infos)
            ]

        grouped: Dict[str, List[int]] = defaultdict(list)
        for pos, key in enumerate(raw_keys):
            grouped[str(key)].append(pos)

        updated = 0
        for sample_key, positions in grouped.items():
            dataset_pos = self.uid_to_pos.get(str(sample_key))
            if dataset_pos is None:
                continue
            z = self.sample_vectors[dataset_pos]
            if z is None:
                continue
            mean_reward = float(rewards[positions].mean().item())
            self.vq.update(sample_key=sample_key, z=z, mean_reward=mean_reward)
            updated += 1

        if updated > 0:
            self.update_count += 1
        if updated > 0 and self.save_every > 0 and self.update_count % self.save_every == 0:
            self._save()
