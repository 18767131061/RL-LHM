# Reproducibility guide

Run `python -m compileall` over the repository and execute `./scripts/run_required_tests.sh` after installing `requirements.txt`. The tests cover the CPU-testable metric, preprocessing, reward, and RL-core contracts.

Full training and inference are environment-dependent: they require approved medical data, local checkpoints, CUDA, and the VeRL/vLLM versions used for the paper. The public package therefore makes those paths explicit through CLI arguments and `${...}` configuration placeholders and does not claim a data-free end-to-end run. Figure-generation scripts and image assets are excluded because they depend on private or separately approved image data.
