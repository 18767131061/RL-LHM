"""Run RL-LHM through the pinned official VeRL FSDP2/vLLM carrier."""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
from data.config import load_config
from training.verl_backend.config import build_config


def run(configuration: dict):
    native = build_config(configuration)
    if platform.system() != "Linux":
        raise RuntimeError("VeRL training requires a Linux CUDA host; use --check-config for CPU schema checks")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("this VeRL recipe requires four visible CUDA GPUs")
    for key in ("model_name_or_path", "processor_name_or_path"):
        if not Path(configuration["model"][key]).is_dir():
            raise ValueError("VeRL RL requires local post-SFT weights and processor files")
    from verl.trainer.main_ppo import run_ppo
    from training.verl_backend.trainer import PaperTaskRunner
    return run_ppo(native, task_runner_class=PaperTaskRunner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="shared JSON recipe with a verl section")
    parser.add_argument("--check-config", action="store_true", help="compose and validate pinned upstream schema without training")
    args = parser.parse_args()
    settings = load_config(args.config)
    if args.check_config:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(build_config(settings), resolve=True))
    else:
        run(settings)


if __name__ == "__main__":
    main()
