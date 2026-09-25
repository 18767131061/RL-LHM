# Training pipeline

`training/run_verl_lhm_variant.py` is the public launcher for the paper variants `vanilla_grpo`, `rars_grpo`, `rars_shae`, `rars_dacvq_vgs`, and `full_rl_lhm`. It accepts explicit training data, base-model, output, and script directories. `training/verl_lhm_launcher.py` adapts the launcher to a local VeRL installation.

The reward path is `rewards/verl_liver_reward.py`; SHAE and DA-CVQ/VGS state are implemented in `rl/core.py` and `rl/verl_lhm_extensions.py`. The code does not bundle VeRL, vLLM, Ray, CUDA, data, or weights.
