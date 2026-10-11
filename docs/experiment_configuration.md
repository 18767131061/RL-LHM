# Experiment configuration

The JSON files in `configs/` are templates. Replace `${...}` tokens with local
paths before use. Keep datasets, checkpoints, outputs, and risk-selection
records outside the repository.

`training_pipeline.md` defines the eight executable combinations of RARS,
SHAE, and coupled DA-CVQ/VGS, separate train/validation inputs, and generated
configuration manifests. A manifest records a requested run; it does not
establish that the run completed or produced a particular published table.

Freeze the cohort, input keys, seed, component settings, actual runtime and
policy loss, class-count definition, visual-feature extraction, and model
selection settings before evaluation. The frozen risk matrix consumed by a
training job is distinct from the procedure that selected it: reproducing a
parameter search also requires the evaluated candidates and their mapping to
trained models. See `reproducibility_guide.md` for the remaining inputs.
