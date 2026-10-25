# Parameterized Config Examples

- `evaluation.example.json`
- `loco_four_class.example.json`
- `rars_grid.example.json`
- `training.example.json`
- `inference.example.json`
- `rl_lhm.example.json`
- `paper_tables.example.json`
- `factorial_tables.example.json`

Deployment locations are placeholders (environment-style or /path/to).
The inference and evaluation examples document CLI arguments; their scripts do
not load these JSON files or expand environment placeholders automatically.
Substitute actual paths and pass the values as CLI flags, replacing underscores
with hyphens. The inference example uses prepared JSONL, greedy decoding and
256 generated tokens. The evaluation example consumes canonical prediction
JSONL after the explicit CSV conversion step; phase-wise annotations and class
spaces belong to the evaluation manifest. Patient aggregation is performed
separately from phase-wise lesion matching.

`rars_grid.example.json` is an executable private four-dimensional Cartesian-grid
configuration; the public cohort uses six axes automatically. It is separate
from the OAT protocol in `training/rars_oat.py`. Training and RL examples describe
parameter groups; use the documented launchers to generate executable recipes.
Distributed VeRL, model checkpoints and GPU dependencies must be supplied and
validated in the execution environment.

The paper/factorial table specifications are read by
`evaluation/export_paper_tables.py` and `evaluation/plot_confusion_matrices.py`.
They list explicit canonical five-seed exports and joint-statistics files;
see `docs/paper_tables_and_confusion.md` for the fields and generated views.
