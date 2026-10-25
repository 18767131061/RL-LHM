from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import runpy
import sys
from pathlib import Path


def _import_local_extensions() -> Path:
    here = Path(__file__).resolve()
    candidates = (here.with_name("verl_lhm_extensions.py"), here.parents[1] / "rl" / "verl_lhm_extensions.py")
    ext_file = next((path for path in candidates if path.is_file()), None)
    if ext_file is None:
        raise RuntimeError(f"Unable to find VERL extension file in: {candidates}")
    # Ray subprocesses must import the same extension when compute_advantage
    # dispatches to SHAE. A registration in this launcher alone is insufficient.
    extension_dir = str(ext_file.parent)
    sys.path.insert(0, extension_dir)
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = extension_dir + (os.pathsep + existing if existing else "")
    spec = importlib.util.spec_from_file_location("verl_lhm_extensions", ext_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load VERL extension file: {ext_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["verl_lhm_extensions"] = module
    spec.loader.exec_module(module)
    return ext_file


def _check_sample_uid_bridge() -> None:
    from patch_verl_sample_uid import require_sample_uid_patch

    spec = importlib.util.find_spec("verl.trainer.ppo.ray_trainer")
    if spec is None or spec.origin is None:
        raise RuntimeError("Unable to locate VeRL ray_trainer for sample UID bridge verification")
    target = Path(spec.origin).resolve()
    audit = {"target": str(target), **require_sample_uid_patch(target)}
    audit_path = os.environ.get("RL_LHM_UID_AUDIT_PATH")
    if audit_path:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def _check_policy_contract() -> None:
    from policy_contract import verify_core, verify_overrides
    from patch_verl_token_mean import require_token_mean_patch
    resolved = verify_overrides(sys.argv[1:])
    paths = {}
    for role, module in (("core", "verl.trainer.ppo.core_algos"), ("actor", "verl.workers.actor.dp_actor")):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Unable to locate {module}")
        paths[role] = Path(spec.origin).resolve()
    dependencies = {}
    for package in ("verl", "torch", "ray", "vllm", "transformers", "omegaconf"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not installed as a distribution"
    audit = {"status": "verified_current_source_and_explicit_overrides", "overrides": resolved,
             "python": sys.version, "dependency_versions": dependencies,
             **verify_core(paths["core"]), **require_token_mean_patch(paths["actor"])}
    target = os.environ.get("RL_LHM_POLICY_AUDIT_PATH")
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    # VERL may still try to initialize wandb if a generated command forgets to
    # override trainer.logger. Keep the local launcher safe for no-wandb usage.
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    # vLLM and expandable_segments can be incompatible; clear the setting explicitly.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    _import_local_extensions()
    if "algorithm.adv_estimator=shae" in sys.argv[1:]:
        _check_sample_uid_bridge()
    _check_policy_contract()
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


if __name__ == "__main__":
    main()
