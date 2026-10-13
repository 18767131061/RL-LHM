"""Import compatibility without model execution or real patient inputs."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_policy_configuration_import_does_not_require_torch():
    code = '''
import sys
sys.modules["torch"] = None
from rl.policy_contract import POLICY_OVERRIDES, verify_overrides
assert verify_overrides([f"{k}={v}" for k,v in POLICY_OVERRIDES.items()]) == POLICY_OVERRIDES
import rl
assert "rl.core" not in sys.modules
assert "rl.completion_likelihood" not in sys.modules
try:
    rl.unknown_export
except AttributeError:
    pass
else:
    raise AssertionError("Unknown exports must remain errors")
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_torch_exports_keep_exact_original_object_identity():
    pytest.importorskip("torch")
    import rl
    from importlib import import_module
    core = import_module("rl.core")
    likelihood = import_module("rl.completion_likelihood")
    for name in ("DACVQState", "ShaeBaselineStore", "compute_shae_advantage", "vgs_sampling_weights"):
        assert getattr(rl, name) is getattr(core, name)
    for name in ("completion_likelihood", "extract_valid_rollout_log_probabilities"):
        assert getattr(rl, name) is getattr(likelihood, name)
