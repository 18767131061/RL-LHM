"""Inference bridge tests with synthetic CSV records and CPU model doubles."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from evaluation.import_inference_predictions import convert
from inference.decoding import DecodingConfig, add_decoding_arguments, config_from_args, input_seed
from inference.resume import ResumeGuard

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ["logical_key", "slice_index", "patient_id", "avg_nll", "raw", "class"]
MAP_FIELDS = ["logical_key", "slice_index", "group_id", "patient_id", "source_patient_id"]


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def raw_completion(label):
    return json.dumps({"detections": [{"phase": "a", "label": label, "bbox_2d": [0, 0, 10, 10]}]})


def bridge_fixture(tmp_path):
    # Canonical groups deliberately have names that cannot be inferred from the
    # source keys. Two nonfinite-NLL HCC votes must beat one finite ICC vote.
    definitions = [
        ("p1", raw_completion("HCC"), "NaN"),
        ("p1", raw_completion("HCC"), "Infinity"),
        ("p1", raw_completion("ICC"), "0.01"),
        ("p2", raw_completion("HCC"), "-Infinity"),
        ("p2", raw_completion("ICC"), "0.2"),
        ("p3", "not JSON", ""),
        ("p4", "", ""),
    ]
    patients = [{"patient_id": f"p{i}", "truth": "ICC" if i == 2 else "HCC"} for i in range(1, 6)]
    groups, mapping, rows = [], [], []
    for i, (patient, raw, nll) in enumerate(definitions):
        group_id = f"manifest-group-{100 - i}"
        groups.append({"group_id": group_id, "patient_id": patient, "phases": {
            phase: {"image_ref": f"synthetic/{i}-{phase}.png", "ground_truth":
                    [{"label": "HCC", "bbox_2d": [0, 0, 10, 10]}] if phase == "a" and i in (0, 5) else []}
            for phase in ("a", "p", "v")}})
        mapping.append({"logical_key": "source-series", "slice_index": i, "group_id": group_id,
                        "patient_id": patient, "source_patient_id": "source-" + patient})
        rows.append({"logical_key": "source-series", "slice_index": i, "patient_id": "source-" + patient,
                     "avg_nll": nll, "raw": raw, "class": "FNH"})
    manifest = {"schema_version": 1, "classes": ["HCC", "ICC", "FNH"],
                "patients": patients, "input_groups": groups}
    paths = {name: tmp_path / filename for name, filename in {
        "csv": "inference.csv", "manifest": "manifest.json", "map": "group-map.csv", "output": "predictions.jsonl"}.items()}
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    write_csv(paths["csv"], rows, CSV_FIELDS)
    write_csv(paths["map"], mapping, MAP_FIELDS)
    return paths, manifest, mapping, rows


def run_conversion(paths):
    return convert(paths["csv"], paths["manifest"], paths["map"], paths["output"])


def test_csv_bridge_preserves_every_raw_and_nonfinite_vote_through_canonical_cli(tmp_path):
    paths, manifest, mapping, rows = bridge_fixture(tmp_path)
    # Order is not identity; neither input order nor CSV's class column supplies GT.
    write_csv(paths["map"], list(reversed(mapping)), MAP_FIELDS)
    process = subprocess.run([sys.executable, str(ROOT / "evaluation/import_inference_predictions.py"),
        "--csv", str(paths["csv"]), "--manifest", str(paths["manifest"]),
        "--group-map", str(paths["map"]), "--output", str(paths["output"])],
        capture_output=True, text=True, cwd=tmp_path)
    assert process.returncode == 0, process.stderr
    converted = [json.loads(line) for line in paths["output"].read_text().splitlines()]
    assert [row["group_id"] for row in converted] == [row["group_id"] for row in mapping]
    assert [row["raw"] for row in converted] == [row["raw"] for row in rows]
    assert [row["completion_nll"] for row in converted] == ["nan", "inf", 0.01, "-inf", 0.2, None, None]
    provenance = json.loads(paths["output"].with_suffix(".jsonl.provenance.json").read_text())
    assert provenance["groups"] == len(rows) == 7
    assert provenance["completion_filtering"] == "none"
    for name, source in [("inference_csv", paths["csv"]), ("manifest", paths["manifest"]), ("group_map", paths["map"])]:
        assert provenance["sources_sha256"][name] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance["output_sha256"] == hashlib.sha256(paths["output"].read_bytes()).hexdigest()

    output = tmp_path / "evaluation"
    process = subprocess.run([sys.executable, str(ROOT / "evaluation/evaluate_paper_protocol.py"),
        "--manifest", str(paths["manifest"]), "--predictions", str(paths["output"]),
        "--output-dir", str(output), "--iou-thresholds", "0.5"], capture_output=True, text=True, cwd=tmp_path)
    assert process.returncode == 0, process.stderr
    result = json.loads((output / "metrics.json").read_text())
    assert result["classification"]["n"] == 5
    assert result["classification"]["accuracy"] == pytest.approx(2 / 5)
    patients = {row["patient_id"]: row for row in result["patients"]}
    assert (patients["p1"]["prediction"], patients["p1"]["valid_votes"]) == ("HCC", 3)
    assert (patients["p2"]["prediction"], patients["p2"]["valid_votes"]) == ("ICC", 2)
    assert all(patients[p]["prediction"] is None for p in ("p3", "p4", "p5"))
    assert result["coverage"]["patients_without_valid_vote"] == 3
    assert result["coverage"]["patients_without_complete_input_groups"] == 1
    assert result["coverage"]["prediction_records"] == 7
    assert result["coverage"]["diagnostic_eligible_groups"] == 5
    for protocol in result["detection"].values():
        assert (protocol[0]["tp"], protocol[0]["fp"], protocol[0]["fn"]) == (1, 0, 1)


@pytest.mark.parametrize("fault, message", [
    ("missing_map", "cover every"), ("duplicate_map_key", "Duplicate"),
    ("duplicate_map_group", "Duplicate"), ("unknown_map_group", "unknown groups"),
    ("map_patient_mismatch", "patient identities"), ("csv_patient_mismatch", "patient_id"),
    ("missing_csv", "every manifest group"), ("duplicate_csv", "duplicate source keys"),
    ("unknown_csv", "unknown or duplicate"), ("negative_nll", "negative"),
    ("malformed_nll", "convert string to float"), ("padded_key", "logical_key"),
    ("negative_index", "nonnegative integer"),
])
def test_identity_and_coverage_errors_fail_without_output(tmp_path, fault, message):
    paths, _, mapping, rows = bridge_fixture(tmp_path)
    if fault == "missing_map": mapping.pop()
    elif fault == "duplicate_map_key": mapping.append(copy.deepcopy(mapping[0]))
    elif fault == "duplicate_map_group": mapping[1]["group_id"] = mapping[0]["group_id"]
    elif fault == "unknown_map_group": mapping[0]["group_id"] = "unknown"
    elif fault == "map_patient_mismatch": mapping[0]["patient_id"] = "p5"
    elif fault == "csv_patient_mismatch": rows[0]["patient_id"] = "p1"
    elif fault == "missing_csv": rows.pop()
    elif fault == "duplicate_csv": rows.append(copy.deepcopy(rows[0]))
    elif fault == "unknown_csv": rows[0]["logical_key"] = "unknown"
    elif fault == "negative_nll": rows[0]["avg_nll"] = "-0.1"
    elif fault == "malformed_nll": rows[0]["avg_nll"] = "not-a-number"
    elif fault == "padded_key": rows[0]["logical_key"] += " "
    elif fault == "negative_index": rows[0]["slice_index"] = -1
    write_csv(paths["csv"], rows, CSV_FIELDS)
    write_csv(paths["map"], mapping, MAP_FIELDS)
    with pytest.raises(ValueError, match=message):
        run_conversion(paths)
    assert not paths["output"].exists()
    assert not paths["output"].with_suffix(".jsonl.provenance.json").exists()


@pytest.mark.parametrize("target", ["csv", "map"])
@pytest.mark.parametrize("fault", ["duplicate_header", "blank_header", "extra_cell", "missing_cell"])
def test_ambiguous_csv_structure_is_rejected(tmp_path, target, fault):
    paths, _, _, _ = bridge_fixture(tmp_path)
    with paths[target].open(newline="") as stream:
        content = list(csv.reader(stream))
    if fault == "duplicate_header":
        content[0].append(content[0][0])
        for row in content[1:]: row.append(row[0])
    elif fault == "blank_header":
        content[0].append("")
        for row in content[1:]: row.append("ignored")
    elif fault == "extra_cell": content[1].append("ignored")
    elif fault == "missing_cell": content[1].pop()
    with paths[target].open("w", newline="") as stream:
        csv.writer(stream).writerows(content)
    with pytest.raises(ValueError):
        run_conversion(paths)
    assert not paths["output"].exists()


def test_source_patient_alias_is_optional_when_ids_already_agree(tmp_path):
    paths, _, mapping, rows = bridge_fixture(tmp_path)
    for row, mapped in zip(rows, mapping):
        row["patient_id"] = mapped["patient_id"]
        del mapped["source_patient_id"]
    write_csv(paths["csv"], rows, CSV_FIELDS)
    write_csv(paths["map"], mapping, MAP_FIELDS[:-1])
    assert run_conversion(paths)["groups"] == 7


def test_conversion_does_not_replace_existing_artifacts(tmp_path):
    paths, _, _, _ = bridge_fixture(tmp_path)
    run_conversion(paths)
    original = paths["output"].read_bytes()
    with pytest.raises(ValueError, match="not overwritten"):
        run_conversion(paths)
    assert paths["output"].read_bytes() == original


def test_default_decoding_is_single_beam_greedy():
    parser = argparse.ArgumentParser()
    add_decoding_arguments(parser)
    config = config_from_args(parser.parse_args([]))
    assert config == DecodingConfig()
    assert config.generation_kwargs() == {"do_sample": False, "num_beams": 1, "num_return_sequences": 1}


@pytest.mark.parametrize("settings", [
    {"mode": "beam"}, {"temperature": 0}, {"temperature": -1}, {"temperature": float("nan")},
    {"temperature": float("inf")}, {"top_p": 0}, {"top_p": -0.1}, {"top_p": 1.1},
    {"top_p": float("nan")}, {"top_k": -1}, {"top_k": 2.5}, {"top_k": True},
])
def test_invalid_sampling_parameters_are_rejected(settings):
    with pytest.raises(ValueError):
        DecodingConfig(**{"mode": "sample", **settings})


@pytest.mark.parametrize("settings", [{"temperature": 0.8}, {"top_p": 0.9}, {"top_k": 10}])
def test_greedy_does_not_silently_ignore_sampling_options(settings):
    with pytest.raises(ValueError, match="require --decoding sample"):
        DecodingConfig(**settings)


def test_sampling_config_and_input_seed_are_explicit():
    config = DecodingConfig("sample", 0.7, 0.9, 20)
    assert config.generation_kwargs() == {"do_sample": True, "num_beams": 1, "num_return_sequences": 1,
                                         "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    assert config.identity()["sampling_seed_scope"] == "input-key-v1"
    assert input_seed(9, ("case", 2)) == input_seed(9, ["case", 2])
    assert len({input_seed(seed, ("case", index)) for seed in (9, 10) for index in (1, 2)}) == 4


def stub_inference(tmp_path, monkeypatch, module_name, *, generated_output=None, pad_id=None):
    import torch
    from PIL import Image
    entry = importlib.import_module("inference." + module_name)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"Synthetic model stub; not usable weights")
    for phase in ("a", "p", "v"):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{phase}.png")
    rows = [{"metadata": {"logical_key": "case", "slice_index": index, "patient_id": "p1",
                          "class": "HCC", "positive": 1, "analysis_center": "synthetic"},
             "messages": [{"role": "user", "content": [
                 *[{"type": "image", "image": f"{phase}.png"} for phase in ("a", "p", "v")],
                 {"type": "text", "text": "Synthetic prompt"}]}]} for index in range(3)]
    source = tmp_path / "inputs.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    calls = []

    class Processor:
        tokenizer = types.SimpleNamespace(pad_token_id=pad_id)
        def apply_chat_template(self, *args, **kwargs): return "prompt"
        def __call__(self, **kwargs): return {"input_ids": torch.tensor([[0]])}
        def batch_decode(self, generated, **kwargs): return [json.dumps({"tokens": generated[0].tolist()})]

    class Model:
        def to(self, device): return self
        def eval(self): pass
        def generate(self, **kwargs):
            calls.append(kwargs)
            if generated_output is not None:
                return generated_output
            tokens = torch.randint(1, 256, (1, 4)) if kwargs["do_sample"] else torch.full((1, 4), 17)
            scores = tuple(torch.zeros((1, 256)) for _ in range(4))
            return types.SimpleNamespace(sequences=torch.cat([torch.tensor([[0]]), tokens], dim=1),
                                         scores=scores, logits=scores)

    monkeypatch.setattr(entry.AutoProcessor, "from_pretrained", lambda *a, **k: Processor())
    if module_name == "infer_qwen_real961":
        monkeypatch.setattr(entry.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: Model())
        monkeypatch.setitem(sys.modules, "qwen_vl_utils", types.SimpleNamespace(
            process_vision_info=lambda messages: ([object()] * 3, None)))
    else:
        monkeypatch.setattr(entry, "load_model", lambda *a, **k: (Model(), "synthetic"))
        monkeypatch.setattr(entry, "prepare_inputs", lambda *a, **k: {"input_ids": torch.tensor([[0]])})

    def run(output, flags=()):
        monkeypatch.setattr(sys, "argv", ["infer", "--model-dir", str(model), "--json-path", str(source),
            "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu", "--seed", "101", *flags])
        entry.main()
    return entry, calls, run


@pytest.mark.parametrize("module_name", ["infer_qwen_real961", "infer_hf_vlm_revision"])
@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_actual_entries_bind_decoding_and_resume_without_changing_sample_stream(tmp_path, monkeypatch, module_name, mode):
    entry, calls, run = stub_inference(tmp_path, monkeypatch, module_name)
    flags = [] if mode == "greedy" else ["--decoding", "sample", "--temperature", "0.7", "--top-p", "0.9", "--top-k", "20"]
    config = DecodingConfig() if mode == "greedy" else DecodingConfig("sample", 0.7, 0.9, 20)
    full = tmp_path / "full.csv"
    run(full, flags)
    assert len(calls) == 3
    for kwargs in calls:
        assert {key: kwargs[key] for key in config.generation_kwargs()} == config.generation_kwargs()
    state = json.loads(full.with_suffix(".csv.resume.json").read_text())
    assert state["complete"] is True
    assert state["identity"]["settings"]["decoding"] == config.identity()
    assert state["identity"]["settings"]["seed"] == 101
    assert state["identity"]["decoding_sha256"] == hashlib.sha256((ROOT / "inference/decoding.py").read_bytes()).hexdigest()
    assert state["identity"]["entry_sha256"] == hashlib.sha256(Path(entry.__file__).read_bytes()).hexdigest()
    if module_name == "infer_hf_vlm_revision":
        assert state["identity"]["settings"]["limit"] is None
    with full.open(newline="") as stream:
        full_rows = list(csv.DictReader(stream))
    if mode == "sample":
        assert len({row["raw"] for row in full_rows}) == 3

    # Seed a verified partial file with one completed record. The real entry
    # skips it, while our CPU model double consumes torch's actual RNG stream.
    resumed = tmp_path / "resumed.csv"
    guard = ResumeGuard(resumed, state["identity"])
    write_csv(guard.partial, full_rows[:1], entry.FIELDNAMES)
    guard.checkpoint(guard.partial)
    guard.close()
    calls.clear()
    run(resumed, flags)
    assert len(calls) == 2
    assert resumed.read_bytes() == full.read_bytes()

    if mode == "sample":
        monkeypatch.setattr(entry.AutoProcessor, "from_pretrained", lambda *a, **k: pytest.fail("changed decoding must reject before model load"))
        changed = ["--decoding", "sample", "--temperature", "0.8", "--top-p", "0.9", "--top-k", "20"]
        with pytest.raises(ValueError, match="changed"):
            run(resumed, changed)


@pytest.mark.parametrize("module_name", ["infer_qwen_real961", "infer_hf_vlm_revision"])
def test_actual_entries_use_unprocessed_logits_and_include_eos_equal_to_pad(tmp_path, monkeypatch, module_name):
    import math
    import torch
    # Token probabilities from raw logits are 2/6 and 2/4. Processed scores
    # deliberately differ, as with temperature/truncation during sampling.
    raw_logits = (torch.tensor([[0., math.log(2), math.log(3)]]),
                  torch.tensor([[0., 0., math.log(2)]]))
    processed = (torch.tensor([[20., 0., 0.]]), torch.tensor([[20., 0., 0.]]))
    generated = types.SimpleNamespace(sequences=torch.tensor([[0, 1, 2]]), logits=raw_logits, scores=processed)
    _, calls, run = stub_inference(tmp_path, monkeypatch, module_name, generated_output=generated, pad_id=2)
    output = tmp_path / "raw-nll.csv"
    run(output, ["--decoding", "sample", "--temperature", "0.7", "--top-p", "0.9", "--top-k", "20"])
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["avg_nll"]) for row in rows] == pytest.approx([(math.log(3) + math.log(2)) / 2] * 3)
    assert all(call["output_logits"] is True for call in calls)
    state = json.loads(output.with_suffix(".csv.resume.json").read_text())
    assert state["identity"]["settings"]["decoding"]["nll_source"] == "unprocessed_logits_single_sequence_v1"


@pytest.mark.parametrize("module_name", ["infer_qwen_real961", "infer_hf_vlm_revision"])
def test_actual_entries_do_not_fall_back_to_processed_scores_when_logits_missing(tmp_path, monkeypatch, module_name):
    import torch
    generated = types.SimpleNamespace(sequences=torch.tensor([[0, 1]]), scores=(torch.zeros((1, 2)),))
    _, _, run = stub_inference(tmp_path, monkeypatch, module_name, generated_output=generated)
    output = tmp_path / "missing-logits.csv"
    with pytest.raises(ValueError, match="logits"):
        run(output)
    assert not output.exists()


@pytest.mark.parametrize("has_logits", [True, False])
def test_generate_one_helper_requires_raw_logits_for_nll(has_logits):
    import math
    import torch
    from inference.generation import generate_one

    class Processor:
        tokenizer = types.SimpleNamespace(pad_token_id=2)
        def apply_chat_template(self, *args, **kwargs): return "prompt"
        def __call__(self, **kwargs): return {"input_ids": torch.tensor([[0]])}
        def batch_decode(self, *args, **kwargs): return ["synthetic completion"]

    class Model:
        def generate(self, **kwargs):
            assert kwargs["output_logits"] is True
            assert kwargs["do_sample"] is True and kwargs["num_beams"] == kwargs["num_return_sequences"] == 1
            output = types.SimpleNamespace(sequences=torch.tensor([[0, 1, 2]]),
                scores=(torch.tensor([[20., 0., 0.]]), torch.tensor([[20., 0., 0.]])))
            if has_logits:
                output.logits = (torch.tensor([[0., math.log(2), math.log(3)]]), torch.tensor([[0., 0., math.log(2)]]))
            return output

    kwargs = {"model": Model(), "processor": Processor(), "messages": [], "images": [],
              "device": "cpu", "decoding": DecodingConfig("sample", 0.7, 0.9, 20)}
    if has_logits:
        text, nll = generate_one(**kwargs)
        assert text == "synthetic completion"
        assert nll == pytest.approx((math.log(3) + math.log(2)) / 2)
    else:
        with pytest.raises(ValueError, match="unprocessed logits"):
            generate_one(**kwargs)
