# Experiment configuration

The JSON files in `configs/` are templates. Replace `${...}` tokens with local
paths to datasets, checkpoints, outputs and risk-selection records before use.

[training_pipeline.md](training_pipeline.md) defines the eight combinations of
RARS, SHAE and coupled DA-CVQ/VGS, separate train/validation inputs and generated
configuration manifests. Each manifest records its requested run configuration.

Freeze the cohort, input keys, seed, component settings, runtime, policy loss,
class-count definition, visual-feature extraction and model-selection settings
before evaluation. A training job consumes a frozen risk matrix. A parameter
search additionally records evaluated candidates and their trained checkpoints.
