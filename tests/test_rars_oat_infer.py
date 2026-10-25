"""Adapter command/identity checks with explicitly synthetic merge/inference functions."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_rars_oat import synthetic_config
from training import rars_oat_infer as adapter
from training.rars_oat import read_json


def inputs(tmp_path):
    _, config, _ = synthetic_config(tmp_path)
    checkpoint = tmp_path / "synthetic_checkpoint"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "SYNTHETIC_NOT_WEIGHTS.txt").write_text("synthetic test only")
    output = tmp_path / "adapter" / "SYNTHETIC_predictions.csv"
    return ["--checkpoint", str(checkpoint), "--validation-manifest", config["validation_manifest"],
            "--inference-input", config["train_file"], "--data-root", str(tmp_path), "--output", str(output),
            "--seed", "42", "--merge-python", "/unsymlinked/merge-env/bin/python",
            "--inference-python", "/unsymlinked/infer-env/bin/python", "--device", "cpu"], output


def fake_commands(monkeypatch):
    commands = []
    def run(command, log):
        commands.append(command)
        if "verl.model_merger" in command:
            directory = Path(command[command.index("--target_dir") + 1])
            directory.mkdir(parents=True)
            (directory / "config.json").write_text('{"synthetic_test": true}')
            (directory / "SYNTHETIC.safetensors").write_text("SYNTHETIC TEST: NOT MODEL WEIGHTS")
        elif "processor-assets" in command:
            directory = Path(command[command.index("--export") + 1])
            (directory / "tokenizer_config.json").write_text('{"synthetic_test": true}')
        else:
            Path(command[command.index("--output") + 1]).write_text("SYNTHETIC TEST: NOT PREDICTIONS\n")
    monkeypatch.setattr(adapter, "run_logged", run)
    return commands


@pytest.mark.parametrize("decoding,extra,temperature,top_p,top_k", [
    ("greedy", [], 1.0, 1.0, 0),
    ("sample", ["--temperature", "0.7", "--top-p", "0.9", "--top-k", "30"], .7, .9, 30),
])
def test_adapter_forwards_explicit_decoding_and_records_identity(tmp_path, monkeypatch, decoding, extra, temperature, top_p, top_k):
    argv, output = inputs(tmp_path)
    commands = fake_commands(monkeypatch)
    args = [*argv, "--decoding", decoding, "--max-new-tokens", "256", *extra]
    assert adapter.main(args) == 0
    assert commands[0][0] == "/unsymlinked/merge-env/bin/python"
    infer = commands[1]
    assert infer[0] == "/unsymlinked/infer-env/bin/python"
    for option, expected in {"--decoding": decoding, "--max-new-tokens": "256", "--temperature": str(temperature),
                             "--top-p": str(top_p), "--top-k": str(top_k)}.items():
        assert infer[infer.index(option) + 1] == expected
    identity = read_json(output.with_suffix(".csv.oat_identity.json"))
    assert identity["decoding"]["mode"] == decoding
    assert identity["decoding"]["temperature"] == temperature
    assert identity["decoding"]["top_p"] == top_p
    assert identity["decoding"]["top_k"] == top_k
    assert identity["max_new_tokens"] == 256
    assert adapter.main(args) == 0
    assert len(commands) == 3  # Model export reused; inference engine receives its same identity.
    changed_tokens = args.copy()
    changed_tokens[changed_tokens.index("--max-new-tokens") + 1] = "128"
    with pytest.raises(ValueError, match="conflicts"):
        adapter.main(changed_tokens)
    changed_mode = [*argv, "--decoding", "sample" if decoding == "greedy" else "greedy", "--max-new-tokens", "256"]
    with pytest.raises(ValueError, match="conflicts"):
        adapter.main(changed_mode)
    if decoding == "sample":
        changed_temperature = args.copy()
        changed_temperature[changed_temperature.index("--temperature") + 1] = "0.8"
        with pytest.raises(ValueError, match="conflicts"):
            adapter.main(changed_temperature)
    assert len(commands) == 3


@pytest.mark.parametrize("settings", [[], ["--decoding", "greedy"], ["--max-new-tokens", "256"],
                                      ["--decoding", "greedy", "--max-new-tokens", "0"]])
def test_adapter_requires_explicit_positive_generation_settings(tmp_path, settings):
    argv, output = inputs(tmp_path)
    with pytest.raises(SystemExit):
        adapter.main([*argv, *settings])
    assert not output.exists()


def test_adapter_uses_shared_decoding_validation(tmp_path):
    argv, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="Sampling settings"):
        adapter.main([*argv, "--decoding", "greedy", "--max-new-tokens", "256", "--temperature", "0.7"])


@pytest.mark.parametrize("interruption", ["receipt", "rename", "unreceipted_export"])
def test_model_export_resumes_with_complete_identity(tmp_path, monkeypatch, interruption):
    argv, output = inputs(tmp_path)
    args = [*argv, "--decoding", "greedy", "--max-new-tokens", "256"]
    commands = fake_commands(monkeypatch)
    folder = output.parent / "model_export"
    receipt = output.parent / "model_export.json"
    real_write, real_replace = adapter.write_json, adapter.os.replace
    if interruption == "unreceipted_export":
        folder.mkdir(parents=True)
        (folder / "unbound.txt").write_text("SYNTHETIC UNBOUND EXPORT")
    else:
        def interrupted_write(path, payload):
            if Path(path) == receipt:
                raise OSError("SYNTHETIC receipt interruption")
            return real_write(path, payload)
        def interrupted_replace(source, target):
            if Path(target) == folder:
                raise OSError("SYNTHETIC rename interruption")
            return real_replace(source, target)
        if interruption == "receipt":
            monkeypatch.setattr(adapter, "write_json", interrupted_write)
        else:
            monkeypatch.setattr(adapter.os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="SYNTHETIC"):
            adapter.main(args)
        assert not folder.exists()
        assert receipt.exists() == (interruption == "rename")
        monkeypatch.setattr(adapter, "write_json", real_write)
        monkeypatch.setattr(adapter.os, "replace", real_replace)
    assert adapter.main(args) == 0
    assert output.is_file() and receipt.is_file()
    assert not (folder / "unbound.txt").exists()
    assert read_json(receipt)["model_sha256"] == adapter.checkpoint_digest(folder)
    assert "infer_hf_vlm_revision.py" in commands[-1][1]


def test_explicit_processor_source_is_forwarded_and_hashed_for_resume(tmp_path, monkeypatch):
    argv, output = inputs(tmp_path)
    source = tmp_path / "seed_sft_processor"
    source.mkdir()
    (source / "tokenizer_config.json").write_text('{"synthetic_test": true}')
    (source / "model.safetensors").write_text("SYNTHETIC UNUSED BASE WEIGHTS")
    commands = fake_commands(monkeypatch)
    args = [*argv, "--decoding", "greedy", "--max-new-tokens", "256", "--processor-source", str(source)]
    assert adapter.main(args) == 0
    exported = commands[1]
    assert exported[0] == "/unsymlinked/infer-env/bin/python" and exported[2] == "processor-assets"
    assert exported[exported.index("--source") + 1] == str(source)
    identity = read_json(output.with_suffix(".csv.oat_identity.json"))
    assert identity["processor_source"] == str(source)
    assert identity["processor_source_sha256"] == adapter.processor_source_digest(source)
    assert (output.parent / "model_export/tokenizer_config.json").is_file()
    assert adapter.main(args) == 0
    assert len(commands) == 4
    (source / "tokenizer_config.json").write_text('{"synthetic_test": true, "changed": true}')
    with pytest.raises(ValueError, match="conflicts"):
        adapter.main(args)
    assert len(commands) == 4


@pytest.mark.parametrize("bad_name", [None, "config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin"])
def test_processor_export_stages_assets_and_preserves_model_weights_and_config(tmp_path, monkeypatch, bad_name):
    source = tmp_path / "synthetic_processor_source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text('{"synthetic_test": true}')
    export = tmp_path / "synthetic_model_export"
    export.mkdir()
    for name in ("config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin"):
        (export / name).write_text("SYNTHETIC ORIGINAL MODEL FILE " + name)
    originals = {path.name: path.read_bytes() for path in export.iterdir()}
    calls = []
    class Processor:
        def save_pretrained(self, path):
            path = Path(path)
            (path / "tokenizer_config.json").write_text('{"synthetic_export": true}')
            (path / "chat_template.jinja").write_text("SYNTHETIC TEMPLATE")
            if bad_name:
                (path / bad_name).write_text("FORBIDDEN SYNTHETIC PAYLOAD")
    class AutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return Processor()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=AutoProcessor))
    args = ["processor-assets", "--source", str(source), "--export", str(export),
            "--expected-source-sha256", adapter.processor_source_digest(source)]
    if bad_name:
        with pytest.raises(ValueError, match="cannot overwrite"):
            adapter.main(args)
        assert not (export / "tokenizer_config.json").exists()
    else:
        assert adapter.main(args) == 0
        assert (export / "tokenizer_config.json").is_file() and (export / "chat_template.jinja").is_file()
    assert calls == [(str(source), {"local_files_only": True, "trust_remote_code": True})]
    assert all((export / name).read_bytes() == content for name, content in originals.items())
    assert not (tmp_path / ".processor_assets.partial").exists()
