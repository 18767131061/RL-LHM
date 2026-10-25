#!/usr/bin/env python3
"""Optional four-/six-dimensional joint-grid search with five-seed tasks.

This tool creates Cartesian-grid tasks and scores supplied validation outputs.
Use training/rars_oat.py for the separate one-parameter sensitivity protocol.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAIRS = {
    "private": (("HCC", "ICC"), ("HCC", "FNH"), ("HCC", "HA_Hemangioma"), ("ICC", "HA_Hemangioma")),
    "public": (("HCC", "ICC"), ("HCC", "HA_Hemangioma"), ("ICC", "HA_Hemangioma"),
               ("ICC", "BCLM"), ("ICC", "CRLM"), ("BCLM", "CRLM")),
}
LABELS = {"private": ["HCC", "ICC", "FNH", "HA_Hemangioma", "nodule_CYST"],
          "public": ["HCC", "ICC", "HA_Hemangioma", "BCLM", "CRLM"]}
SEEDS = [42, 2026, 3407, 1234, 5678]
VALUES = [i / 10 for i in range(1, 16)]


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(payload):
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def axes(config):
    return config["loco_pairs"] if config.get("protocol") == "loco" else PAIRS[config["cohort"]]


def validate_profile(profile):
    if not isinstance(profile, Mapping):
        raise ValueError("Risk profile must be an object")
    cohort = profile.get("cohort")
    if cohort not in PAIRS:
        raise ValueError("Risk profile requires a known cohort")
    if profile.get("protocol") == "oat":
        # OAT includes the shared other-pair weight as a search axis.
        # Joint-grid/LOCO profiles use the fixed background weight.
        other = profile.get("other_weight")
        if type(other) not in (int, float) or not math.isfinite(other) or other not in VALUES:
            raise ValueError("OAT other_weight must be in {0.1,...,1.5}")
    elif profile.get("other_weight") != 0.4:
        raise ValueError("Joint-grid profile requires known cohort and fixed other_weight=0.4")
    pairs = PAIRS[cohort]
    if profile.get("protocol") == "loco":
        classes = profile.get("active_classes", [])
        pairs = profile.get("loco_pairs", [])
        if not isinstance(classes, list) or not classes or any(not isinstance(c, str) for c in classes) or len(classes) != len(set(classes)) or not set(classes) <= set(LABELS[cohort]) or not pairs:
            raise ValueError("LOCO profile requires its explicit active classes and search pairs")
        if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(classes) for pair in pairs):
            raise ValueError("LOCO search pair is outside its active classes")
        if len({frozenset(pair) for pair in pairs}) != len(pairs):
            raise ValueError("Duplicate unordered LOCO search pair")
    expected = {"--".join(pair) for pair in pairs}
    weights = profile.get("pair_weights", {})
    if not isinstance(weights, Mapping):
        raise ValueError("Profile pair_weights must be an object")
    if set(weights) != expected:
        raise ValueError("Profile must specify each independent unordered pair exactly once")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v not in VALUES for v in weights.values()):
        raise ValueError("Profile weights must be in {0.1,...,1.5}")
    return profile


def candidate(config, index):
    values = config["values"]
    dimensions = len(axes(config))
    count = len(values) ** dimensions
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("Candidate index outside finite grid")
    coordinates = [0] * dimensions
    cursor = index
    for pos in range(dimensions - 1, -1, -1):
        cursor, coordinates[pos] = divmod(cursor, len(values))
    profile = {"cohort": config["cohort"], "other_weight": 0.4,
               "pair_weights": {"--".join(pair): values[value] for pair, value in zip(axes(config), coordinates)}}
    if config.get("protocol") == "loco":
        profile.update(protocol="loco", active_classes=config["classes"], loco_pairs=config["loco_pairs"])
    return validate_profile(profile)


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("cohort") not in PAIRS or config.get("seeds") != SEEDS:
        raise ValueError("Search requires the ordered five-seed set 42,2026,3407,1234,5678")
    values = config.get("values")
    synthetic = config.get("synthetic_test", False)
    if type(synthetic) is not bool:
        raise ValueError("synthetic_test must be a Boolean")
    if not isinstance(values, list) or not values or len(set(values)) != len(values):
        raise ValueError("Grid values must be nonempty and unique")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v not in VALUES for v in values):
        raise ValueError("Unsupported grid values")
    if not synthetic and values != VALUES:
        raise ValueError("Formal search must enumerate all 15 levels; subsets are synthetic_test only")
    if config.get("protocol") not in ("main", "loco"):
        raise ValueError("protocol must be main or loco")
    if config.get("selection_split") != "validation" or config.get("formal_test_used_for_selection") is not False:
        raise ValueError("Only held-out validation can select weights")
    if config.get("protocol") == "main" and config.get("split_id") != {"private": "P1", "public": "U1"}[config["cohort"]]:
        raise ValueError("Main selection must use P1/U1")
    if config.get("protocol") == "loco" and (not config.get("training_center") or not config.get("held_out_center") or config["training_center"] == config["held_out_center"]):
        raise ValueError("LOCO must specify different training and held-out centers")
    for key in ("train_file", "validation_manifest", "test_identity_manifest", "output_root"):
        raw = config.get(key)
        if not isinstance(raw, str) or not raw or "${" in raw:
            raise ValueError(f"Supply explicit {key}")
        config[key] = str((path.parent / raw).resolve())
    models = config.get("seed_model_paths", {})
    if set(models) != {str(seed) for seed in SEEDS}:
        raise ValueError("Supply one SFT checkpoint path for every seed")
    config["seed_model_paths"] = {key: str((path.parent / value).resolve()) for key, value in models.items()}
    if len(set(config["seed_model_paths"].values())) != 5:
        raise ValueError("Each seed requires its own SFT checkpoint path")
    files = config.get("seed_train_files", {})
    if set(files) != {str(seed) for seed in SEEDS}:
        raise ValueError("Supply a training feature cache for each SFT seed")
    config["seed_train_files"] = {key: str((path.parent / value).resolve()) for key, value in files.items()}
    if len(set(config["seed_train_files"].values())) != 5:
        raise ValueError("Seed-specific SFT feature caches require distinct training files")
    if config["train_file"] != config["seed_train_files"][str(SEEDS[0])]:
        raise ValueError("train_file must identify the seed-42 training cache")
    if config.get("preparation_audit"):
        config["preparation_audit"] = str((path.parent / config["preparation_audit"]).resolve())
    candidate(config, 0)  # validate explicit LOCO axes; never infer them from P1/U1
    return config


def training_row_digest(row):
    """Bind each persistent input to its data, allowing seed-specific features."""
    payload = copy.deepcopy(row)
    metadata = [payload.get("extra_info")]
    if isinstance(payload.get("reward_model"), dict):
        metadata.append(payload["reward_model"].get("extra_info"))
    for extra in metadata:
        if isinstance(extra, dict):
            for key in ("visual_z", "visual_z_contract", "visual_z_cache_id"):
                extra.pop(key, None)
    return digest(payload)


def validate_seed_caches(files, models, train_payloads, train_patients, require_contract=False):
    train_uids = set(train_payloads)
    seed_hashes = {}
    feature_dimension = None
    for seed, cache in files.items():
        ids, people, dims, feature_contracts = set(), set(), set(), set()
        with Path(cache).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                extra = row.get("extra_info", {})
                uid = extra.get("sample_uid")
                feature_contracts.add(extra.get("visual_z_contract"))
                vector = extra.get("visual_z")
                if uid in ids or not isinstance(vector, list) or not vector or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or sum(v*v for v in vector) <= 0:
                    raise ValueError("Every seed cache requires unique inputs and finite nonzero visual_z")
                ids.add(uid)
                people.add(extra.get("patient_id"))
                dims.add(len(vector))
                if train_payloads.get(uid) != training_row_digest(row):
                    raise ValueError("Seed cache changes an input's patient, image, target or metadata")
        if ids != train_uids or people != train_patients or len(dims) != 1:
            raise ValueError("Seed caches differ in training population or vector dimensions")
        dimension = next(iter(dims))
        if feature_dimension is not None and dimension != feature_dimension:
            raise ValueError("Seed caches have different visual feature dimensions")
        feature_dimension = dimension
        if require_contract and feature_contracts == {None}:
            raise ValueError("Prepared formal runs require model-bound visual feature cache contracts")
        if feature_contracts != {None}:
            from scripts.extract_qwen_visual_latents import CONTRACT, extract_cache
            if feature_contracts != {CONTRACT}:
                raise ValueError("Mixed or unsupported visual feature contracts")
            sidecar = Path(cache).with_suffix(Path(cache).suffix + ".audit.json")
            audit = json.loads(sidecar.read_text())
            identity = audit["identity"]
            extract_cache(identity["input_path"], cache, models[seed],
                          identity["image_root"], identity["device"], identity["dtype"],
                          validate_only=True)
        seed_hashes[seed] = sha(cache)
    return seed_hashes, feature_dimension


def payloads(path):
    result = {}
    people = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = row["extra_info"]["sample_uid"]
        if uid in result:
            raise ValueError("Duplicate prepared training UID")
        result[uid] = training_row_digest(row)
        people.add(row["extra_info"]["patient_id"])
    return result, people


def bind_preparation(config, train_payloads):
    """Bind generated split/count outputs without asking for copied numeric dictionaries."""
    path = config.get("preparation_audit")
    if not path:
        return None
    audit = json.loads(Path(path).read_text())
    expected_mode = "private" if config["cohort"] == "private" else "5cls"
    if audit.get("data_mode") != expected_mode or audit.get("active_classes") != config.get("classes", LABELS[config["cohort"]]):
        raise ValueError("Preparation cohort or active classes differ from search")
    for output in audit["outputs"].values():
        if sha(output["path"]) != output["sha256"]:
            raise ValueError("Prepared data output changed")
    training = audit["outputs"]["real961_rl_train"]
    if payloads(training["path"])[0] != train_payloads:
        raise ValueError("Search caches do not preserve prepared fit groups and their count metadata")
    if config["protocol"] == "loco":
        loco = audit.get("loco") or {}
        checks = (loco.get("source_center") == config["training_center"],
                  loco.get("held_out_center") == config["held_out_center"],
                  loco.get("split_id") == config["split_id"],
                  loco.get("validation_manifest_sha256") == sha(config["validation_manifest"]),
                  loco.get("reserved_test_sha256") == sha(config["test_identity_manifest"]))
        if not all(checks):
            raise ValueError("LOCO preparation split, validation or reserved-test identity differs from search")
    return sha(path)


def identities(config):
    from evaluation.evaluate_paper_protocol import validate_manifest
    train_patients, train_uids = set(), set()
    train_rows = 0
    train_payloads = {}
    with Path(config["train_file"]).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra_info", {})
            uid, patient = extra.get("sample_uid"), extra.get("patient_id")
            if not isinstance(uid, str) or not uid.strip() or uid in train_uids or not isinstance(patient, str) or not patient.strip():
                raise ValueError("Training requires unique sample_uid and explicit patient_id")
            if config["protocol"] == "loco" and extra.get("center_id") != config["training_center"]:
                raise ValueError("LOCO training row is not from training_center")
            train_patients.add(patient)
            train_uids.add(uid)
            train_payloads[uid] = training_row_digest(row)
            train_rows += 1
    seed_hashes, feature_dimension = validate_seed_caches(config["seed_train_files"], config["seed_model_paths"], train_payloads, train_patients,
        require_contract=bool(config.get("preparation_audit")) and not config.get("synthetic_test", False))
    manifest = json.loads(Path(config["validation_manifest"]).read_text())
    classes, patients, groups = validate_manifest(manifest)
    from evaluation.schema import canonical_label
    if [canonical_label(value) for value in manifest.get("macro_classes", classes)] != classes:
        raise ValueError("Source validation macro classes must equal its active diagnostic classes")
    if manifest.get("split") != "validation" or manifest.get("split_id") != config["split_id"]:
        raise ValueError("Validation manifest split identity does not match search")
    if set(classes) != set(LABELS[config["cohort"]]) and config["protocol"] != "loco":
        raise ValueError("Validation class vocabulary differs from the cohort")
    if config["protocol"] == "loco":
        allowed = config.get("classes")
        if not allowed or classes != allowed or not set(classes) <= set(LABELS[config["cohort"]]):
            raise ValueError("LOCO requires its explicit ordered active class rule")
        if any(row.get("center_id") != config["training_center"] for row in manifest["patients"]):
            raise ValueError("LOCO validation must be inside training_center")
    test = json.loads(Path(config["test_identity_manifest"]).read_text())
    if test.get("split") != "test" or not isinstance(test.get("patients"), list) or not test["patients"]:
        raise ValueError("Provide an identity-only reserved-test manifest (no outcomes required)")
    test_ids = [row.get("patient_id") for row in test["patients"]]
    if any(not isinstance(v, str) or not v.strip() for v in test_ids) or len(test_ids) != len(set(test_ids)):
        raise ValueError("Invalid or duplicate reserved-test patient identities")
    if config["protocol"] == "loco" and any(row.get("center_id") != config["held_out_center"] for row in test["patients"]):
        raise ValueError("LOCO reserved test must be the held-out center")
    validation_uids = {group["group_id"] for group in groups}
    if not train_rows or not groups or train_patients & patients.keys() or train_patients & set(test_ids) or patients.keys() & set(test_ids) or train_uids & validation_uids:
        raise ValueError("Training/validation/test identities overlap or a required split is empty")
    preparation_sha = bind_preparation(config, train_payloads)
    return {"preparation_audit_sha256": preparation_sha, "seed_train_sha256": seed_hashes, "training_payload_sha256": digest(train_payloads),
            "feature_dimension": feature_dimension, "train_sha256": sha(config["train_file"]), "validation_sha256": sha(config["validation_manifest"]),
            "test_identity_sha256": sha(config["test_identity_manifest"]), "train_rows": train_rows,
            "train_patients": len(train_patients), "validation_patients": len(patients),
            "test_patients": len(test_ids), "classes": classes}


class Search:
    def __init__(self, config, database):
        self.config = config
        self.identity = identities(config)
        self.search_hash = digest({"config": config, "identity": self.identity})
        self.n_candidates = len(config["values"]) ** len(axes(config))
        self.n_tasks = self.n_candidates * len(SEEDS)
        self.db = sqlite3.connect(database, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS completed (task INTEGER PRIMARY KEY, candidate INTEGER NOT NULL, seed INTEGER NOT NULL, score REAL NOT NULL, record TEXT NOT NULL, UNIQUE(candidate,seed))")
        found = self.db.execute("SELECT value FROM metadata WHERE key='search_hash'").fetchone()
        if found and found[0] != self.search_hash:
            raise ValueError("Database belongs to a different grid, split, seed set, or file version")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('search_hash',?)", (self.search_hash,))
        self.db.commit()

    def task(self, task_id):
        if type(task_id) is not int or not 0 <= task_id < self.n_tasks:
            raise ValueError("Task index outside grid")
        index, seed_index = divmod(task_id, len(SEEDS))
        seed = SEEDS[seed_index]
        profile = candidate(self.config, index)
        folder = Path(self.config["output_root"]) / self.search_hash[:16] / f"candidate_{index:08d}" / f"seed_{seed}"
        return {"search_hash": self.search_hash, "task_id": task_id, "candidate_index": index, "seed": seed,
                "profile": profile, "profile_digest": digest(profile), "identity": self.identity,
                "model_path": self.config["seed_model_paths"][str(seed)], "train_file": self.config["seed_train_files"][str(seed)], "task_dir": str(folder)}

    def pending(self, shard=0, shards=1, start=0, stop=None):
        if type(shards) is not int or shards < 1 or not 0 <= shard < shards:
            raise ValueError("Require 0 <= shard < shards")
        stop = self.n_tasks if stop is None else stop
        if not 0 <= start <= stop <= self.n_tasks:
            raise ValueError("Invalid task range")
        first = start + ((shard - start) % shards)
        for task_id in range(first, stop, shards):
            if not self.db.execute("SELECT 1 FROM completed WHERE task=?", (task_id,)).fetchone():
                yield self.task(task_id)

    def prepare(self, task_id):
        task = self.task(task_id)
        folder = Path(task["task_dir"])
        profile_path = folder / "risk_profile.json"
        task_path = folder / "task.json"
        if task_path.exists() and json.loads(task_path.read_text()) != task:
            raise ValueError("Task directory contains a conflicting identity")
        write_json(profile_path, task["profile"])
        write_json(task_path, task)
        command = [sys.executable, str(ROOT / "training/run_verl_lhm_variant.py"),
                   "--train-file", task["train_file"], "--model-path", task["model_path"],
                   "--output-dir", str(folder / "training"), "--variant", "full_rl_lhm", "--seed", str(task["seed"]),
                   "--scripts-dir", str(ROOT), "--risk-profile", str(profile_path),
                   "--n-gpus", str(self.config.get("n_gpus", 4))]
        write_json(folder / "train_command.json", command)
        return task, command

    def score(self, task_id, prediction_path, prediction_receipt):
        from evaluation.evaluate_paper_protocol import evaluate
        task = self.task(task_id)
        folder = Path(task["task_dir"])
        if not (folder / "task.json").is_file() or json.loads((folder / "task.json").read_text()) != task:
            raise ValueError("Prepare this exact task before scoring")
        train_dir = folder / "training"
        run_path, done_path = train_dir / "run_manifest.json", train_dir / "training_complete.json"
        run, done = json.loads(run_path.read_text()), json.loads(done_path.read_text())
        checks = (run.get("seed") == task["seed"], done.get("seed") == task["seed"],
                  run.get("variant") == "full_rl_lhm", done.get("variant") == "full_rl_lhm",
                  done.get("status") == "training_complete", done.get("smoke") is False,
                  run.get("configuration_only") is False, run.get("smoke") is False,
                  run.get("train_file_sha256") == self.identity["seed_train_sha256"][str(task["seed"])],
                  run.get("model_path") == task["model_path"],
                  run.get("risk_profile_sha256") == sha(folder / "risk_profile.json"),
                  json.loads((folder / "risk_profile.json").read_text()) == task["profile"],
                  done.get("run_manifest_sha256") == sha(run_path))
        if not all(checks):
            raise ValueError("Training completion, seed, profile, model, or data identity mismatch")
        receipt = json.loads(Path(prediction_receipt).read_text())
        checkpoint = Path(receipt.get("checkpoint", ""))
        try:
            checkpoint.resolve().relative_to(train_dir.resolve())
        except ValueError as error:
            raise ValueError("Validation checkpoint is outside this candidate/seed run") from error
        if not checkpoint.is_dir() or checkpoint.name != done.get("final_checkpoint"):
            raise ValueError("Validation must use the completed final-epoch checkpoint")
        if receipt.get("task_digest") != digest(task) or receipt.get("validation_sha256") != self.identity["validation_sha256"] or receipt.get("predictions_sha256") != sha(prediction_path) or receipt.get("partition") != "validation":
            raise ValueError("Prediction receipt does not match this validation task")
        manifest = json.loads(Path(self.config["validation_manifest"]).read_text())
        result = evaluate(manifest, Path(prediction_path), [0.5])
        score = result["classification"]["macro_f1"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Validation macro-F1 must be finite and in [0,1]")
        record = {"task_digest": digest(task), "identity": self.identity,
                  "evidence": {str(Path(p).resolve()): sha(p) for p in
                               (run_path, done_path, prediction_path, prediction_receipt, folder / "risk_profile.json")},
                  "score": score, "coverage": result["coverage"]}
        previous = self.db.execute("SELECT record FROM completed WHERE task=?", (task_id,)).fetchone()
        if previous:
            if previous[0] == canonical(record):
                return record  # exact replay only; no duplicate contributes twice
            raise ValueError("Conflicting duplicate task result")
        with self.db:
            self.db.execute("INSERT INTO completed VALUES (?,?,?,?,?)", (task_id, task["candidate_index"], task["seed"], score, canonical(record)))
        return record

    def finalize(self, output, refit_config=None):
        """Expose selected models, or prepare a separately identified source-only refit."""
        summary = self.summary()
        best = summary["global_best_in_grid"]
        if best is None:
            raise ValueError("Complete every candidate and seed before final model preparation")
        synthetic = self.config.get("synthetic_test", False)
        result = {"search_hash": self.search_hash, "profile": best["profile"],
                  "selected_candidate": best["candidate_index"], "synthetic_test": synthetic,
                  "models": [], "refit_tasks": []}
        if refit_config is None:
            for index, seed in enumerate(SEEDS):
                task = self.task(best["candidate_index"] * len(SEEDS) + index)
                training = Path(task["task_dir"]) / "training"
                done = json.loads((training / "training_complete.json").read_text())
                checkpoint = training / done["final_checkpoint"]
                if not checkpoint.is_dir():
                    raise ValueError("Selected checkpoint no longer exists")
                result["models"].append({"seed": seed, "checkpoint": str(checkpoint),
                    "train_file": task["train_file"], "train_sha256": self.identity["seed_train_sha256"][str(seed)]})
            result.update(status="synthetic_selected_references" if synthetic else "selected_models_ready",
                          fit_scope="selection_train_only", identity=self.identity)
        else:
            preparation = self.config.get("preparation_audit")
            if not preparation or sha(preparation) != self.identity["preparation_audit_sha256"]:
                raise ValueError("Refit requires the unchanged source preparation audit bound to selection")
            audit = json.loads(Path(preparation).read_text())
            loco = audit.get("loco") or {}
            if self.config["protocol"] != "loco" or loco.get("final_fit_scope") != "train_plus_validation":
                raise ValueError("Refit must be explicitly prepared from source train plus validation")
            combined = audit["outputs"]["real961_rl_refit_train_validation"]
            if sha(combined["path"]) != combined["sha256"]:
                raise ValueError("Prepared combined refit data changed")
            expected_payloads, expected_people = payloads(combined["path"])
            fit_payloads, fit_people = payloads(self.config["train_file"])
            validation = json.loads(Path(self.config["validation_manifest"]).read_text())
            expected_uids = set(fit_payloads) | {row["group_id"] for row in validation["input_groups"]}
            if set(expected_payloads) != expected_uids or expected_people != fit_people | {row["patient_id"] for row in validation["patients"]}:
                raise ValueError("Refit must contain exactly the source fit/validation union")
            for line in Path(combined["path"]).read_text().splitlines():
                if line.strip() and json.loads(line)["extra_info"].get("center_id") != self.config["training_center"]:
                    raise ValueError("Refit contains a non-source center")
            refit_path = Path(refit_config).resolve()
            refit = json.loads(refit_path.read_text())
            if refit.get("fit_scope") != "train_plus_validation":
                raise ValueError("Refit config must explicitly declare train_plus_validation")
            mappings = {}
            for key in ("seed_train_files", "seed_model_paths"):
                values = refit.get(key, {})
                if set(values) != {str(seed) for seed in SEEDS}:
                    raise ValueError("Refit requires a cache and explicitly selected SFT model for every seed")
                mappings[key] = {seed: str((refit_path.parent / path).resolve()) for seed, path in values.items()}
                if len(set(mappings[key].values())) != len(SEEDS):
                    raise ValueError("Refit seed paths must be distinct")
            if any(not Path(path).is_dir() for path in mappings["seed_model_paths"].values()):
                raise ValueError("Refit SFT model paths must exist")
            hashes, dimension = validate_seed_caches(mappings["seed_train_files"], mappings["seed_model_paths"], expected_payloads, expected_people,
                                                      require_contract=not synthetic)
            root = Path(self.config["output_root"]) / self.search_hash[:16] / "final_refit"
            for seed in SEEDS:
                folder = root / f"seed_{seed}"
                profile_path = folder / "risk_profile.json"
                command = [sys.executable, str(ROOT / "training/run_verl_lhm_variant.py"),
                    "--train-file", mappings["seed_train_files"][str(seed)],
                    "--model-path", mappings["seed_model_paths"][str(seed)],
                    "--output-dir", str(folder / "training"), "--variant", "full_rl_lhm",
                    "--seed", str(seed), "--scripts-dir", str(ROOT),
                    "--risk-profile", str(profile_path), "--n-gpus", str(self.config.get("n_gpus", 4))]
                task = {"seed": seed, "fit_scope": "train_plus_validation", "command": command,
                        "train_sha256": hashes[str(seed)], "preparation_sha256": sha(preparation),
                        "synthetic_test": synthetic, "training_executed": False}
                task_path = folder / "task.json"
                if task_path.exists() and json.loads(task_path.read_text()) != task:
                    raise ValueError("Refit output directory belongs to a different model/cache identity")
                write_json(profile_path, best["profile"])
                write_json(task_path, task)
                write_json(folder / "train_command.json", command)
                result["refit_tasks"].append(task)
            result.update(status="synthetic_refit_commands_only" if synthetic else "refit_commands_prepared_not_trained",
                          fit_scope="train_plus_validation", feature_dimension=dimension,
                          class_counts=combined["class_counts"], preparation_sha256=sha(preparation))
        write_json(output, result)
        return result

    def summary(self):
        count, candidates_complete = 0, 0
        current, entries = None, []
        best = None
        def accept(index, rows):
            nonlocal candidates_complete, best
            if [seed for seed, _ in rows] != sorted(SEEDS):
                return
            candidates_complete += 1
            mean = math.fsum(value for _, value in rows) / len(rows)
            # Deterministic tie break: first mixed-radix candidate index.
            if best is None or mean > best["mean_validation_macro_f1"]:
                best = {"candidate_index": index, "mean_validation_macro_f1": mean,
                        "seed_scores": dict(rows), "profile": candidate(self.config, index)}
        for task_id, index, seed, score, raw in self.db.execute("SELECT task,candidate,seed,score,record FROM completed ORDER BY candidate,seed"):
            task = self.task(task_id)
            record = json.loads(raw)
            if task["candidate_index"] != index or task["seed"] != seed or not math.isfinite(score) or not 0 <= score <= 1 or record.get("score") != score or record.get("task_digest") != digest(task) or record.get("identity") != self.identity:
                raise ValueError("Invalid result identity or finite score in completion database")
            for path, expected in record["evidence"].items():
                if sha(path) != expected:
                    raise ValueError("Completed task evidence changed after scoring")
            if current is not None and index != current:
                accept(current, entries)
                entries = []
            current = index
            entries.append((seed, score))
            count += 1
        if current is not None:
            accept(current, entries)
        complete = count == self.n_tasks and candidates_complete == self.n_candidates
        return {"search_hash": self.search_hash, "expected_candidates": self.n_candidates, "expected_tasks": self.n_tasks,
                "completed_tasks": count, "completed_candidates": candidates_complete,
                "exhaustive": complete and not self.config.get("synthetic_test", False),
                "status": "synthetic_complete" if complete and self.config.get("synthetic_test", False) else "exhaustive_complete" if complete else "partial",
                "protocol": self.config["protocol"], "split_id": self.config["split_id"],
                "best_among_completed_candidates": best, "global_best_in_grid": best if complete else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks")
    tasks.add_argument("--shard", type=int, default=0)
    tasks.add_argument("--shards", type=int, default=1)
    tasks.add_argument("--start", type=int, default=0)
    tasks.add_argument("--stop", type=int)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--task", type=int, required=True)
    run = commands.add_parser("run")
    run.add_argument("--task", type=int, required=True)
    score = commands.add_parser("score")
    score.add_argument("--task", type=int, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--receipt", type=Path, required=True)
    commands.add_parser("summary")
    final = commands.add_parser("finalize")
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--refit-config", type=Path)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    search = Search(load_config(args.config), args.database)
    if args.command == "tasks":
        for task in search.pending(args.shard, args.shards, args.start, args.stop):
            print(canonical(task))
    elif args.command in ("prepare", "run"):
        task, command = search.prepare(args.task)
        if args.command == "run":
            if search.config.get("synthetic_test", False):
                raise ValueError("Synthetic grids cannot launch model training")
            train_dir = Path(task["task_dir"]) / "training"
            if (train_dir / "training_complete.json").exists():
                raise ValueError("Training is already complete; score its validation outputs instead")
            if train_dir.exists():
                train_dir.rename(train_dir.with_name(f"training_incomplete_{time.time_ns()}"))
            subprocess.run(command, check=True)
        print(canonical({"task": task, "command": command}))
    elif args.command == "score":
        print(canonical(search.score(args.task, args.predictions, args.receipt)))
    elif args.command == "finalize":
        print(canonical(search.finalize(args.output, args.refit_config)))
    else:
        summary = search.summary()
        if args.command == "freeze":
            if summary["status"] != "exhaustive_complete":
                raise ValueError("Only a complete formal full grid can freeze selected weights")
            write_json(args.output.with_suffix(".profile.json"), summary["global_best_in_grid"]["profile"])
            write_json(args.output, {"status": "frozen_global_best_in_finite_grid", **summary,
                                     "selection_partition": "validation", "profile": summary["global_best_in_grid"]["profile"]})
        print(canonical(summary))

if __name__ == "__main__":
    main()
