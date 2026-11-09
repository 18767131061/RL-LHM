"""Translate the shared JSON recipe into the pinned VeRL Hydra schema."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from . import UPSTREAM_COMMIT


def upstream_directory():
    spec = importlib.util.find_spec("verl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Create the pinned Linux VeRL environment with scripts/setup_verl.sh")
    return Path(next(iter(spec.submodule_search_locations)))


def verify_upstream(package_directory=None):
    """Reject changed carrier interfaces, including unversioned editable installs."""
    root = Path(package_directory) if package_directory is not None else upstream_directory()
    manifest = json.loads(Path(__file__).with_name("upstream_api.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"VeRL source mismatch at {relative}; required commit {UPSTREAM_COMMIT}")
    return root


def validate_recipe(settings):
    from models.huggingface import DecodingConfig
    from training.engine import ComponentSwitches, PolicyParameters
    generation = DecodingConfig(**settings["generation"])
    ComponentSwitches(**settings["components"])
    policy = PolicyParameters(**settings["policy"])
    if settings["verl"]["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("the VeRL recipe requires its exact upstream commit")
    if settings["model"]["dtype"] != "bfloat16" or settings["distributed"]["world_size"] != 4:
        raise ValueError("this VeRL recipe uses one node with four CUDA GPUs and bfloat16")
    if (not generation.do_sample or generation.temperature != 1 or generation.top_k != 0
            or generation.repetition_penalty != 1 or generation.num_beams != 1):
        raise ValueError("VeRL RL uses sampling at temperature 1, unrestricted top-k and no repetition penalty")
    if policy.log_ratio_bound != 20 or policy.kl_bound != 10:
        raise ValueError("VeRL vanilla/low_var_kl implements the manuscript bounds 20 and 10")
    if policy.kl_coefficient <= 0:
        raise ValueError("this recipe requires the frozen reference KL term")
    loop = settings["loop"]
    for key in ("epochs", "batch_size", "microbatch_size", "group_size", "policy_updates", "max_prompt_length", "max_sequence_length"):
        if isinstance(loop[key], bool) or not isinstance(loop[key], int) or loop[key] <= 0:
            raise ValueError(f"loop.{key} must be a positive integer")
    if loop["batch_size"] != 4 or loop["group_size"] != 2 or loop["microbatch_size"] != 1:
        raise ValueError("this recipe uses global prompt batch 4, two completions per prompt and microbatch 1")
    if loop["max_prompt_length"] + generation.max_new_tokens > loop["max_sequence_length"]:
        raise ValueError("prompt plus response budget exceeds maximum model length")
    if settings["model"]["revision"] is not None:
        raise ValueError("VeRL RL starts from an explicit local post-SFT checkpoint; revision must be null")
    if settings["model"]["attn_implementation"] != "flash_attention_2":
        raise ValueError("the pinned Qwen3-VL packed FSDP recipe uses flash_attention_2")
    if set(settings["model"]["processor_kwargs"]) - {"min_pixels", "max_pixels"}:
        raise ValueError("only min_pixels and max_pixels processor overrides are supported")
    verl = settings["verl"]
    required = {"upstream_commit", "rollout_seed", "tensor_parallel_size", "gpu_memory_utilization",
                "agent_workers", "actor_param_offload", "actor_optimizer_offload", "reference_param_offload"}
    if set(verl) != required:
        raise ValueError(f"verl configuration must contain exactly {sorted(required)}")
    if verl["tensor_parallel_size"] not in (1, 2, 4) or not 0 < verl["gpu_memory_utilization"] < 1:
        raise ValueError("invalid rollout tensor parallelism or memory utilization")
    if not isinstance(verl["agent_workers"], int) or not 1 <= verl["agent_workers"] <= 8:
        raise ValueError("agent_workers must be between 1 and 8")


def build_config(settings, package_directory=None):
    """Compose official defaults; assign only verified upstream fields.

    `rl_lhm` is the recipe's own namespace. All other fields are official VeRL
    fields from the immutable source manifest checked by verify_upstream().
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    validate_recipe(settings)
    root = verify_upstream(package_directory)
    settings = copy.deepcopy(settings)
    for key in ("manifest", "output_dir", "resume_from"):
        if settings[key] is not None:
            settings[key] = str(Path(settings[key]).resolve())
    for key in ("model_name_or_path", "processor_name_or_path"):
        settings["model"][key] = str(Path(settings["model"][key]).resolve())
    with initialize_config_dir(config_dir=str(root / "trainer/config"), version_base=None):
        native = compose(config_name="ppo_trainer", overrides=["model_engine=dp"])
    loop, generation, policy, optimizer, carrier = (settings[k] for k in ("loop", "generation", "policy", "optimizer", "verl"))
    values = {
        "trainer.use_v1": False, "trainer.n_gpus_per_node": 4, "trainer.nnodes": 1,
        "trainer.total_epochs": loop["epochs"], "trainer.balance_batch": False,
        "trainer.default_local_dir": settings["output_dir"], "trainer.default_hdfs_dir": None,
        "trainer.resume_mode": "disable", "trainer.val_before_train": False, "trainer.test_freq": -1,
        "trainer.logger": ["console"], "trainer.project_name": "RL-LHM", "trainer.experiment_name": "paper_verl",
        "data.train_files": [settings["manifest"]], "data.val_files": [], "data.train_batch_size": loop["batch_size"],
        "data.max_prompt_length": loop["max_prompt_length"], "data.max_response_length": generation["max_new_tokens"],
        "data.dataloader_num_workers": 0, "data.truncation": "error", "data.filter_overlong_prompts": False,
        "data.mm_processor_kwargs": settings["model"]["processor_kwargs"],
        "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
        "algorithm.rollout_correction": None,
        "actor_rollout_ref.model.path": settings["model"]["model_name_or_path"],
        "actor_rollout_ref.model.tokenizer_path": settings["model"]["processor_name_or_path"],
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.override_config": {"attn_implementation": "flash_attention_2", "attention_dropout": 0.0},
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
        "actor_rollout_ref.actor.fsdp_config.param_offload": carrier["actor_param_offload"],
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": carrier["actor_optimizer_offload"],
        "actor_rollout_ref.actor.fsdp_config.seed": settings["seed"],
        "actor_rollout_ref.actor.fsdp_config.full_determinism": settings["deterministic_algorithms"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": loop["batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": loop["microbatch_size"],
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_epochs": loop["policy_updates"],
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.data_loader_seed": settings["seed"],
        "actor_rollout_ref.actor.clip_ratio": policy["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_low": policy["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_high": policy["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_c": policy["dual_clip"],
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": policy["kl_coefficient"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.grad_clip": loop["max_grad_norm"],
        "actor_rollout_ref.actor.optim.clip_grad": loop["max_grad_norm"],
        "actor_rollout_ref.actor.optim.lr": optimizer["learning_rate"],
        "actor_rollout_ref.actor.optim.betas": optimizer["betas"],
        "actor_rollout_ref.actor.optim.weight_decay": optimizer["weight_decay"],
        "actor_rollout_ref.actor.optim.override_optimizer_config": {"eps": optimizer["epsilon"], "amsgrad": optimizer["amsgrad"]},
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.checkpoint.async_save": False,
        "actor_rollout_ref.ref.strategy": "fsdp2", "actor_rollout_ref.ref.fsdp_config.strategy": "fsdp2",
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "bf16",
        "actor_rollout_ref.ref.fsdp_config.param_offload": carrier["reference_param_offload"],
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.n": loop["group_size"],
        "actor_rollout_ref.rollout.temperature": generation["temperature"],
        "actor_rollout_ref.rollout.top_p": generation["top_p"], "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.seed": carrier["rollout_seed"],
        "actor_rollout_ref.rollout.full_determinism": settings["deterministic_algorithms"],
        "actor_rollout_ref.rollout.max_model_len": loop["max_sequence_length"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": carrier["tensor_parallel_size"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": carrier["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.calculate_log_probs": False,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.limit_images": 3,
        "actor_rollout_ref.rollout.agent.num_workers": carrier["agent_workers"],
        "actor_rollout_ref.rollout.agent.default_agent_loop": "rl_lhm_single_turn",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(Path(__file__).with_name("agent_loop.yaml")),
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class": "training.verl_backend.rollout.DeferredRewardManager",
        "reward.num_workers": 1, "reward.reward_model.enable": False,
    }
    with open_dict(native):
        for key, value in values.items():
            OmegaConf.update(native, key, value, merge=False)
        native.rl_lhm = settings
        # One-node deployment: use the same interpreter and project modules on
        # the driver, rollout actors and training workers.
        project_path = str(Path(__file__).resolve().parents[2])
        python_path = os.pathsep.join(filter(None, [project_path, os.environ.get("PYTHONPATH")]))
        native.ray_kwargs.ray_init.runtime_env = {
            "py_executable": sys.executable, "env_vars": {"PYTHONPATH": python_path},
        }
    return native
