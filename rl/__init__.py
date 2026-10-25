"""RL-LHM exports with Torch optional for configuration-only imports."""

from importlib import import_module
from importlib.util import find_spec

# Preserve the existing exported object identities and submodule import order
# whenever Torch is installed. Configuration-only helpers also work without it.
if find_spec("torch") is not None:
    from .completion_likelihood import completion_likelihood, extract_valid_rollout_log_probabilities
    from .core import DACVQState, ShaeBaselineStore, compute_shae_advantage, vgs_sampling_weights

__all__ = [
    "DACVQState",
    "ShaeBaselineStore",
    "completion_likelihood",
    "compute_shae_advantage",
    "extract_valid_rollout_log_probabilities",
    "vgs_sampling_weights",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = (
        ".completion_likelihood"
        if name in {"completion_likelihood", "extract_valid_rollout_log_probabilities"}
        else ".core"
    )
    module = import_module(module_name, __name__)
    # Cache every export from this module, including completion_likelihood,
    # whose function name also matches the imported submodule name.
    names = (
        ("completion_likelihood", "extract_valid_rollout_log_probabilities")
        if module_name == ".completion_likelihood"
        else ("DACVQState", "ShaeBaselineStore", "compute_shae_advantage", "vgs_sampling_weights")
    )
    globals().update({export: getattr(module, export) for export in names})
    return globals()[name]
