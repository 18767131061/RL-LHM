# Scripts

- `run_required_tests.sh`: runs the configured implementation and evaluation
  tests and writes their log; use `--cpu-only` for the core dependency suite.
- `plot_rars.py`: validates and renders one confirmed RARS 15x15 parameter pair.
- `plot_rars_oat.py`: renders complete one-dimensional OAT curves with exact
  five-seed bootstrap intervals from the OAT selection output.

- `extract_qwen_visual_latents.py`: local image-only reference extraction, fixed
  1152-D per-seed caches, identity validation and interruption recovery. See
  `../docs/visual_feature_extraction.md`.
