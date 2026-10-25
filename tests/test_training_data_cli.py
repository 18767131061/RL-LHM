"""Exercise public CLI contracts using only Python's standard library.

Subprocesses use -S so config generation and conversion cannot rely on installed
torch/VeRL/model packages. All source records and image files are synthetic.
"""

import ast
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANT_FACTORS = {
    "vanilla_grpo": (False, False, False),
    "shae_grpo": (False, True, False),
    "dacvq_vgs_grpo": (False, False, True),
    "shae_dacvq_vgs": (False, True, True),
    "rars_grpo": (True, False, False),
    "rars_shae": (True, True, False),
    "rars_dacvq_vgs": (True, False, True),
    "full_rl_lhm": (True, True, True),
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TrainingDataCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.model = self.work / "empty-model-directory"
        self.model.mkdir()
        self.train_path = self.work / "train.jsonl"
        self.validation_path = self.work / "validation.jsonl"
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
        )
        for phase in ("a", "p", "v"):
            (self.work / f"{phase}.png").write_bytes(png)
        self.train_rows = [
            self.source_row("train1", "HCC"),
            self.source_row("train2", "HCC"),
            self.source_row("train3", "ICC"),
            self.source_row("train4", None),
        ]
        self.validation_rows = [self.source_row("validation1", "FNH")]
        write_jsonl(self.train_path, self.train_rows)
        write_jsonl(self.validation_path, self.validation_rows)

    @staticmethod
    def source_row(uid, label):
        return {
            "metadata": {
                "logical_key": uid,
                "slice_index": 0,
                "patient_id": uid,
                "statistical_patient_id": "stat-" + uid,
                "positive": int(label is not None),
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Diagnose the three phases."}]
                    + [{"type": "image", "image": f"{phase}.png"} for phase in ("a", "p", "v")],
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"detections": [
                            {
                                "phase": phase,
                                "label": label or "no-lesion",
                                "bbox_2d": [0, 0, 1, 1] if label else [-1, -1, -1, -1],
                            }
                            for phase in ("a", "p", "v")
                        ]}),
                    }],
                },
            ],
        }

    def run_cli(self, script, arguments, succeeds=True):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / script), *map(str, arguments)],
            cwd=self.work,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def convert(self, name="prepared", extra=(), with_validation=True, succeeds=True):
        output = self.work / name
        args = [
            "--data-root", self.work,
            "--train-jsonl", self.train_path,
            "--output-dir", output,
        ]
        if with_validation:
            args += ["--validation-jsonl", self.validation_path]
        return output, self.run_cli(
            "preprocess/build_verl_real961_dataset.py", [*args, *extra], succeeds=succeeds
        )

    def launch_config(self, train, variant, extra=(), succeeds=True):
        output = self.work / ("run_" + variant)
        result = self.run_cli("training/run_verl_lhm_variant.py", [
            "--train-file", train,
            "--model-path", self.model,
            "--output-dir", output,
            "--variant", variant,
            "--seed", "2026",
            "--scripts-dir", ROOT,
            "--n-gpus", "2",
            "--config-only",
            *extra,
        ], succeeds=succeeds)
        if not succeeds:
            return result
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(json.loads(result.stdout), manifest)
        self.assertFalse((output / "training_complete.json").exists())
        self.assertEqual(manifest["sample_identity_contract"]["persistent_sample_key"], "extra_info.sample_uid")
        self.assertTrue(any(arg.startswith("++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=")
                            for arg in manifest["command"]))
        return manifest

    def test_default_conversion_preserves_validation_and_train_only_counts(self):
        output, result = self.convert()
        train = read_jsonl(output / "real961_rl_train.jsonl")
        validation = read_jsonl(output / "real961_rl_validation.jsonl")
        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 1)
        self.assertFalse({r["uid"] for r in train} & {r["uid"] for r in validation})
        self.assertFalse((output / "real961_rl_refit_train_validation.jsonl").exists())
        for row in train + validation + read_jsonl(output / "real961_rl_monitor.jsonl"):
            self.assertEqual(row["extra_info"]["class_counts"], {"HCC": 2, "ICC": 1})
            self.assertEqual(row["reward_model"]["extra_info"], row["extra_info"])
        audit = json.loads(result.stdout)
        self.assertEqual(audit, json.loads((output / "audit.json").read_text()))
        self.assertEqual(audit["class_counts_definition"]["unit"], "positive_three_phase_input_group")
        self.assertEqual(audit["sources"]["train"]["sha256"], hashlib.sha256(self.train_path.read_bytes()).hexdigest())
        self.assertEqual(audit["outputs"]["real961_rl_train"]["background_input_groups"], 1)
        self.assertEqual(audit["outputs"]["real961_rl_validation"]["class_counts_population"], "supplied_train_source")

    def test_refit_is_explicit_additional_output_with_separate_counts(self):
        ordinary, _ = self.convert()
        output, result = self.convert("refit", extra=("--refit-train-validation",))
        self.assertEqual(
            (ordinary / "real961_rl_train.jsonl").read_bytes(),
            (output / "real961_rl_train.jsonl").read_bytes(),
        )
        rows = read_jsonl(output / "real961_rl_refit_train_validation.jsonl")
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["extra_info"]["class_counts"], {"HCC": 2, "ICC": 1, "FNH": 1})
        audit = json.loads(result.stdout)
        self.assertTrue(audit["refit_requested"])
        self.assertFalse(audit["validation_held_out_from_refit"])
        self.assertEqual(audit["outputs"]["real961_rl_refit_train_validation"]["class_counts_population"], "supplied_train_and_validation_sources")

    def test_center_metadata_is_preserved_and_conflicts_rejected(self):
        for row in self.train_rows + self.validation_rows:
            row['metadata']['analysis_center'] = 'source'
        write_jsonl(self.train_path, self.train_rows)
        write_jsonl(self.validation_path, self.validation_rows)
        output, _ = self.convert()
        for row in read_jsonl(output / 'real961_rl_train.jsonl'):
            self.assertEqual(row['extra_info']['center_id'], 'source')
            self.assertEqual(row['reward_model']['extra_info']['center_id'], 'source')
        self.train_rows[0]['metadata']['center_id'] = 'different'
        write_jsonl(self.train_path, self.train_rows)
        _, result = self.convert('bad_center', succeeds=False)
        self.assertIn('consistent nonempty', result.stderr)

    def test_public_counts_use_same_positive_group_rule(self):
        self.train_rows = [self.source_row('b1', 'BCLM'), self.source_row('c1', 'CRLM'),
                           self.source_row('b2', 'BCLM'), self.source_row('bg', None)]
        write_jsonl(self.train_path, self.train_rows)
        output, result = self.convert(extra=('--data-mode', '5cls'), with_validation=False)
        for row in read_jsonl(output / 'real961_rl_train.jsonl'):
            self.assertEqual(row['extra_info']['class_counts'], {'BCLM': 2, 'CRLM': 1})
            self.assertEqual(row['extra_info']['data_mode'], '5cls')
        self.assertEqual(json.loads(result.stdout)['active_classes'], ['HCC', 'ICC', 'HA_Hemangioma', 'BCLM', 'CRLM'])
        _, result = self.convert('public3_bad', extra=('--data-mode', '3cls'), with_validation=False, succeeds=False)
        self.assertIn('out of contract', result.stderr)

    def loco_source(self):
        rows = []
        # Reference patient diagnosis differs from lesion/group majority on purpose.
        for label in ('HCC', 'ICC'):
            for index in range(4):
                row = self.source_row(f'{label}{index}', 'HCC')
                row['metadata'].update(analysis_center='source', **{'class': label})
                rows.append(row)
        write_jsonl(self.train_path, rows)
        test = self.work / 'reserved_test.json'
        test.write_text(json.dumps({'split': 'test', 'patients': [{'patient_id': 'heldout', 'center_id': 'target'}]}))
        return rows, test

    def convert_loco(self, name='loco', extra=(), succeeds=True):
        output = self.work / name
        return output, self.run_cli('preprocess/build_verl_real961_dataset.py', [
            '--data-root', self.work, '--loco-source-jsonl', self.train_path,
            '--output-dir', output, '--source-center', 'source', '--held-out-center', 'target',
            '--tuning-fraction', '.25', '--split-seed', '97', '--active-classes', 'HCC', 'ICC',
            '--split-id', 'source_to_target', '--test-identity-manifest', self.work / 'reserved_test.json',
            *extra], succeeds=succeeds)

    def test_loco_partition_uses_patient_truth_and_counts_fit_groups(self):
        self.loco_source()
        output, result = self.convert_loco(extra=('--refit-train-validation',))
        again, _ = self.convert_loco('again', extra=('--refit-train-validation',))
        train = read_jsonl(output / 'real961_rl_train.jsonl')
        val = read_jsonl(output / 'real961_rl_validation.jsonl')
        audit = json.loads(result.stdout)
        self.assertEqual(audit['loco']['train_patients_by_class'], {'HCC': 3, 'ICC': 3})
        self.assertEqual(audit['loco']['validation_patients_by_class'], {'HCC': 1, 'ICC': 1})
        self.assertEqual(len(train), 6)
        self.assertEqual(len(val), 2)
        self.assertFalse({r['extra_info']['patient_id'] for r in train} & {r['extra_info']['patient_id'] for r in val})
        self.assertTrue(all(r['extra_info']['class_counts'] == {'HCC': 6} for r in train + val))
        refit = read_jsonl(output / 'real961_rl_refit_train_validation.jsonl')
        self.assertTrue(all(r['extra_info']['class_counts'] == {'HCC': 8} for r in refit))
        self.assertEqual((output / 'real961_rl_train.jsonl').read_bytes(), (again / 'real961_rl_train.jsonl').read_bytes())
        manifest = json.loads((output / 'validation_manifest.json').read_text())
        self.assertEqual({p['truth'] for p in manifest['patients']}, {'HCC', 'ICC'})
        self.assertEqual({g['group_id'] for g in manifest['input_groups']}, {r['uid'] for r in val})
        self.assertEqual(len(read_jsonl(output / 'validation_source.jsonl')), 2)
        self.assertEqual(audit['loco']['final_fit_scope'], 'train_plus_validation')
        self.assertEqual(audit['sources']['train']['input_groups'], 8)

    def test_loco_rejects_missing_reference_center_and_test_leakage(self):
        for fault in ('missing_truth', 'target_row', 'test_overlap', 'patient_conflict'):
            with self.subTest(fault=fault):
                rows, test = self.loco_source()
                if fault == 'missing_truth': rows[0]['metadata'].pop('class')
                elif fault == 'target_row': rows[0]['metadata']['analysis_center'] = 'target'
                elif fault == 'test_overlap':
                    test.write_text(json.dumps({'split':'test','patients':[{'patient_id':rows[0]['metadata']['statistical_patient_id'],'center_id':'target'}]}))
                elif fault == 'patient_conflict': rows[1]['metadata']['statistical_patient_id'] = rows[0]['metadata']['statistical_patient_id']; rows[1]['metadata']['class'] = 'ICC'
                write_jsonl(self.train_path, rows)
                output, _ = self.convert_loco(fault, succeeds=False)
                self.assertFalse(output.exists())

    def test_monitor_without_validation_is_explicit_training_diagnostic(self):
        output, result = self.convert(with_validation=False)
        self.assertFalse((output / "real961_rl_validation.jsonl").exists())
        audit = json.loads(result.stdout)
        self.assertIsNone(audit["sources"]["validation"])
        self.assertEqual(audit["outputs"]["real961_rl_monitor"]["role"], "training_subset_diagnostic_not_validation")

    def test_refit_requires_validation_source(self):
        output, result = self.convert(with_validation=False, extra=("--refit-train-validation",), succeeds=False)
        self.assertIn("requires --validation-jsonl", result.stderr)
        self.assertFalse(output.exists())

    def test_overlap_rejected_for_each_patient_identity(self):
        for field in ("patient_id", "statistical_patient_id"):
            with self.subTest(field=field):
                validation = self.source_row("other", "FNH")
                validation["metadata"][field] = self.train_rows[0]["metadata"][field]
                write_jsonl(self.validation_path, [validation])
                output, result = self.convert(name=field, succeeds=False)
                self.assertIn("Patient overlap", result.stderr)
                self.assertIn(field, result.stderr)
                self.assertFalse(output.exists())

    def test_duplicate_groups_and_empty_validation_are_rejected(self):
        write_jsonl(self.train_path, [*self.train_rows, self.train_rows[0]])
        output, result = self.convert(name="duplicate", succeeds=False)
        self.assertIn("Duplicate input-group UID", result.stderr)
        self.assertFalse(output.exists())
        write_jsonl(self.train_path, self.train_rows)
        self.validation_path.write_text("")
        output, result = self.convert(name="empty_validation", succeeds=False)
        self.assertIn("validation source is empty", result.stderr)
        self.assertFalse(output.exists())

    def test_all_eight_factorial_variants_through_actual_config_only_cli(self):
        from rl.policy_contract import verify_overrides
        prepared, _ = self.convert()
        seen = set()
        for variant, factors in VARIANT_FACTORS.items():
            with self.subTest(variant=variant):
                rars, shae, curriculum = factors
                manifest = self.launch_config(prepared / "real961_rl_train.jsonl", variant)
                self.assertEqual(manifest["components"], {"reward": "rars" if rars else "vanilla", "shae": shae, "vgs": curriculum})
                self.assertEqual(manifest["runtime_environment"]["RL_LHM_REWARD_MODE"], "rars" if rars else "vanilla")
                command = manifest["command"]
                # Pass the generated argv through the actual prelaunch guard,
                # including non-policy keys that can accidentally be duplicated.
                verify_overrides(command[2:])
                workers = [arg for arg in command if arg.startswith("data.dataloader_num_workers=")]
                self.assertEqual(workers, [f"data.dataloader_num_workers={0 if curriculum else 2}"])
                self.assertIn("algorithm.adv_estimator=" + ("shae" if shae else "grpo"), command)
                self.assertEqual("+data.sampler.use_vgs=True" in command, curriculum)
                self.assertEqual("+algorithm.shae_gate_mid=0.05" in command, shae)
                self.assertEqual(manifest["epochs"], 6)
                self.assertEqual(manifest["max_response_length"], 256)
                for fixed in (
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    "actor_rollout_ref.actor.clip_ratio=0.20",
                    "actor_rollout_ref.actor.kl_loss_coef=0.01",
                    "actor_rollout_ref.rollout.n=2",
                    "trainer.test_freq=-1",
                    "trainer.val_before_train=False",
                ):
                    self.assertIn(fixed, command)
                self.assertEqual(manifest["runtime_environment"]["RL_LHM_RISK_PROFILE"], "")
                self.assertTrue(manifest["configuration_only"])
                self.assertEqual(manifest["policy_objective_verification"]["status"], "runtime_verification_required_before_training")
                self.assertIsNone(manifest["validation_file"])
                self.assertEqual(manifest["validation_usage"]["data_val_files_role"], "training_file_placeholder_not_held_out_validation")
                seen.add(factors)
        self.assertEqual(len(seen), 8)

    def test_launcher_binds_optional_validation_without_enabling_validation(self):
        prepared, _ = self.convert()
        validation = prepared / "real961_rl_validation.jsonl"
        manifest = self.launch_config(prepared / "real961_rl_train.jsonl", "vanilla_grpo", extra=("--validation-file", validation))
        self.assertIn(f"data.val_files={validation}", manifest["command"])
        self.assertEqual(manifest["validation_file_sha256"], hashlib.sha256(validation.read_bytes()).hexdigest())
        self.assertEqual(manifest["validation_usage"]["test_freq"], -1)
        self.assertEqual(manifest["validation_usage"]["data_val_files_role"], "explicit_validation_file_partition_not_checked")

    def test_launcher_rejects_missing_validation(self):
        prepared, _ = self.convert()
        result = self.launch_config(prepared / "real961_rl_train.jsonl", "vanilla_grpo", extra=("--validation-file", self.work / "missing.jsonl"), succeeds=False)
        self.assertIn("Missing validation file", result.stderr)

    def test_launcher_rejects_explicit_training_file_as_validation(self):
        prepared, _ = self.convert()
        train = prepared / "real961_rl_train.jsonl"
        result = self.launch_config(train, "vanilla_grpo", extra=("--validation-file", train), succeeds=False)
        self.assertIn("--validation-file must differ", result.stderr)

    def test_agent_loop_patch_preserves_backup_and_is_idempotent(self):
        source = ROOT / "rl/patch_verl_agent_loop_logprobs.py"
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"OLD", "LEGACY_NEW", "NEW"}
        }
        for label in ("OLD", "LEGACY_NEW"):
            with self.subTest(label=label):
                target = self.work / (label + ".py")
                original = constants[label]
                target.write_text(original)
                record = self.work / (label + ".json")
                args = ["--agent-loop", target, "--audit", record]
                first = self.run_cli("rl/patch_verl_agent_loop_logprobs.py", args)
                self.assertFalse(json.loads(first.stdout)["already_applied"])
                self.assertEqual(target.read_text(), constants["NEW"])
                self.assertEqual(target.with_suffix(".py.pre_rl_lhm").read_text(), original)
                second = self.run_cli("rl/patch_verl_agent_loop_logprobs.py", args)
                result = json.loads(second.stdout)
                self.assertTrue(result["already_applied"])
                self.assertEqual(result["sha256_before"], result["sha256_after"])
                self.assertEqual(target.with_suffix(".py.pre_rl_lhm").read_text(), original)

    def test_agent_loop_patch_rejects_unsupported_source_without_writing(self):
        target = self.work / "unrecognized.py"
        target.write_text("# Synthetic unsupported source\n")
        original = target.read_bytes()
        record = self.work / "unrecognized.json"
        self.run_cli("rl/patch_verl_agent_loop_logprobs.py", [
            "--agent-loop", target, "--audit", record,
        ], succeeds=False)
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse(target.with_suffix(".py.pre_rl_lhm").exists())
        self.assertFalse(record.exists())


if __name__ == "__main__":
    unittest.main()
