"""Backend selection must dispatch before starting distributed workers."""
from types import ModuleType
import sys

import pytest

import training.run as entry


def test_legacy_config_uses_deepspeed(monkeypatch):
    calls = []
    monkeypatch.setattr(entry, "run", lambda stage, config: calls.append((stage, config)))
    configuration = {"manifest": "train.json"}
    entry.run_with_backend("sft", configuration)
    assert calls == [("sft", configuration)]


def test_config_and_explicit_selection_dispatch_verl(monkeypatch):
    module = ModuleType("training.verl_run")
    calls = []
    module.run = lambda config: calls.append(config)
    monkeypatch.setitem(sys.modules, "training.verl_run", module)
    monkeypatch.setattr(entry, "run", lambda *a: pytest.fail("DeepSpeed must not start"))
    configuration = {"backend": "verl"}
    entry.run_with_backend("rl", configuration)
    entry.run_with_backend("rl", {}, backend="verl")
    assert calls == [configuration, {}]


def test_explicit_backend_overrides_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr(entry, "run", lambda stage, config: calls.append(stage))
    entry.run_with_backend("rl", {"backend": "verl"}, backend="deepspeed")
    assert calls == ["rl"]


@pytest.mark.parametrize("stage,backend", [("sft", "verl"), ("rl", "unknown")])
def test_invalid_backend_or_stage_does_not_start_workers(monkeypatch, stage, backend):
    monkeypatch.setattr(entry, "run", lambda *a: pytest.fail("Workers must not start"))
    with pytest.raises(ValueError):
        entry.run_with_backend(stage, {}, backend=backend)
