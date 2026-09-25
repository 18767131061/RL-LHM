from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path


def _import_local_extensions() -> None:
    here = Path(__file__).resolve()
    candidates = (here.with_name("verl_lhm_extensions.py"), here.parents[1] / "rl" / "verl_lhm_extensions.py")
    ext_file = next((path for path in candidates if path.is_file()), None)
    if ext_file is None:
        raise RuntimeError(f"Unable to find VERL extension file in: {candidates}")
    spec = importlib.util.spec_from_file_location("verl_lhm_extensions", ext_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load VERL extension file: {ext_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["verl_lhm_extensions"] = module
    spec.loader.exec_module(module)


def main() -> None:
    # VERL may still try to initialize wandb if a generated command forgets to
    # override trainer.logger. Keep the local launcher safe for no-wandb usage.
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    # vLLM and expandable_segments can be incompatible; clear the setting explicitly.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    _import_local_extensions()
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


if __name__ == "__main__":
    main()
