"""Synthetic command integration tests; no model or clinical results are used."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import venv

import pytest

from evaluation import postprocess_verl_checkpoint as postprocess


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "legacy_finalizer", ROOT / "statistics" / "finalize_revision_statistics.py"
)
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


class CommandsChecked(Exception):
    """Stop before unrelated checkpoint hashing after checking real commands."""


def write_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def setup_postprocess(tmp_path, monkeypatch, python, evaluator):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({"variant": "full", "seed": 42}))
    metadata = tmp_path / "metadata.csv"
    write_csv(metadata, {
        "split": "test", "logical_key": "synthetic", "statistical_patient_id": "synthetic",
        "slice_index": 0, "images": json.dumps(["a.png", "p.png", "v.png"]),
        "bbox_2d": json.dumps([[100, 100, 200, 200]]), "class": "HCC", "model_label": "HCC",
        "analysis_center": "synthetic", "positive": 1, "lesion_count": 1, "is_duplicate": 0,
    })
    monkeypatch.setattr(postprocess, "discover_latest_checkpoint", lambda _: (1, tmp_path / "checkpoint"))
    monkeypatch.setattr(postprocess, "validate_merged_model", lambda _: None)
    monkeypatch.setattr(postprocess, "wait_for_capacity_gpu", lambda *args: 0)
    monkeypatch.setattr(sys, "argv", [
        "postprocess", "--allow-legacy-protocol", "--run-dir", str(run_dir),
        "--variant", "full", "--seed", "42", "--cohort", "private",
        "--merge-python", str(python), "--eval-python", str(python),
        "--verl-repo", str(tmp_path), "--inference-script", str(tmp_path / "inference.py"),
        "--evaluation-script", str(evaluator), "--test-json", str(tmp_path / "test.json"),
        "--data-root", str(tmp_path), "--metadata", str(metadata),
        "--dataset-manifest", str(tmp_path / "manifest.json"),
        "--dataset-audit", str(tmp_path / "audit.json"), "--expected-patients", "1",
    ])
    return run_dir


def handle_model_stage(command):
    if "--target_dir" in command:
        Path(command[command.index("--target_dir") + 1]).mkdir()
        return True
    if "--model-dir" in command:
        raw = json.dumps({"detections": [
            {"phase": phase, "label": "HCC", "bbox_2d": [100, 100, 200, 200]}
            for phase in ("a", "p", "v")
        ]})
        write_csv(Path(command[command.index("--output") + 1]), {
            "logical_key": "synthetic", "slice_index": 0, "avg_nll": 0, "raw": raw,
        })
        return True
    return False


@pytest.mark.parametrize("name", ["evaluate_real961_predictions.py", "evaluate_sim999_predictions.py"])
def test_bundled_legacy_evaluator_child_completes(tmp_path, monkeypatch, name):
    run_dir = setup_postprocess(tmp_path, monkeypatch, sys.executable, ROOT / "evaluation" / name)

    def run_child(command, *args):
        if handle_model_stage(command):
            return
        completed = subprocess.run(command, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        metrics = json.loads((run_dir / "evaluation" / "metrics.json").read_text())
        assert metrics["classification_overall"]["accuracy"] == 1.0
        assert "LEGACY PROTOCOL" in completed.stderr
        raise CommandsChecked

    monkeypatch.setattr(postprocess, "run_logged", run_child)
    with pytest.raises(CommandsChecked):
        postprocess.main()


@pytest.fixture(scope="module")
def isolated_python(tmp_path_factory):
    root = tmp_path_factory.mktemp("executable-venv")
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    return root / "bin" / "python"


def assert_venv_interpreter(executable, isolated_python):
    prefix = subprocess.check_output([executable, "-c", "import sys; print(sys.prefix)"], text=True).strip()
    assert Path(prefix).resolve() == isolated_python.parent.parent.resolve()


def test_postprocess_preserves_all_venv_children_and_custom_interface(tmp_path, monkeypatch, isolated_python):
    # Even a custom script with a bundled basename must keep its own CLI contract.
    setup_postprocess(tmp_path, monkeypatch, isolated_python, tmp_path / "evaluate_sim999_predictions.py")
    calls = []

    def check_command(command, *args):
        assert_venv_interpreter(command[0], isolated_python)
        calls.append(command)
        if handle_model_stage(command):
            return
        assert "--allow-legacy-protocol" not in command
        raise CommandsChecked

    monkeypatch.setattr(postprocess, "run_logged", check_command)
    with pytest.raises(CommandsChecked):
        postprocess.main()
    assert len(calls) == 3


def test_finalizer_preserves_venv_child(tmp_path, monkeypatch, isolated_python):
    for method in finalizer.METHODS:
        for seed in finalizer.SEEDS:
            for path in (finalizer.patient_predictions(tmp_path, method, seed), finalizer.metrics_path(tmp_path, method, seed)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic placeholder: not read before the first child command")
    monkeypatch.setattr(sys, "argv", [
        "finalize", "--allow-legacy-protocol", "--root", str(tmp_path),
        "--cohort", "private", "--python", str(isolated_python),
    ])

    def check_command(command, *args):
        assert_venv_interpreter(command[0], isolated_python)
        raise CommandsChecked

    monkeypatch.setattr(finalizer, "run", check_command)
    with pytest.raises(CommandsChecked):
        finalizer.main()
