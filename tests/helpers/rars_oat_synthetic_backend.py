"""CPU-only synthetic OAT command backend. Never creates experimental results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "infer"))
    parser.add_argument("--synthetic-test", action="store_true", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()
    plan_path = next((parent / "plan.json" for parent in args.output.resolve().parents if (parent / "plan.json").is_file()), None)
    if plan_path is None or json.loads(plan_path.read_text()).get("synthetic_test") is not True:
        raise ValueError("Synthetic backend requires an explicitly synthetic OAT plan")
    if args.mode == "train":
        folder = args.output
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "global_step_1"
        checkpoint.mkdir()
        (checkpoint / "SYNTHETIC_NOT_MODEL_WEIGHTS.txt").write_text(f"synthetic seed={args.seed}; profile={sha(args.profile)}")
        run = {"seed": args.seed, "variant": "full_rl_lhm", "configuration_only": False, "smoke": False,
               "train_file_sha256": sha(args.train), "model_path": str(args.model),
               "risk_profile_sha256": sha(args.profile), "synthetic_test": True,
               "runtime_environment": {"RL_LHM_RISK_PROFILE": str(args.profile)},
               "command": ["SYNTHETIC_CPU_TEST",
                           "++ray_kwargs.ray_init.runtime_env.env_vars.RL_LHM_RISK_PROFILE=" + json.dumps(str(args.profile))]}
        write(folder / "run_manifest.json", run)
        write(folder / "training_complete.json", {"seed": args.seed, "variant": "full_rl_lhm", "smoke": False,
              "status": "training_complete", "run_manifest_sha256": sha(folder / "run_manifest.json"),
              "final_checkpoint": checkpoint.name, "synthetic_test": True})
    else:
        assert (args.checkpoint / "SYNTHETIC_NOT_MODEL_WEIGHTS.txt").is_file()
        manifest = json.loads(args.manifest.read_text())
        assert manifest["split"] == "validation"
        truth = {row["patient_id"]: row["truth"] for row in manifest["patients"]}
        task = json.loads(args.task.read_text())
        profile = json.loads(Path(task["profile_path"]).read_text())
        value = next((weight for weight in [*profile["pair_weights"].values(), profile["other_weight"]] if weight != 1), 1)
        labels = manifest["classes"]
        rows = []
        for index, group in enumerate(manifest["input_groups"]):
            correct = value >= .3 and args.seed != 42
            label = truth[group["patient_id"]] if correct else labels[(index+1) % len(labels)]
            rows.append({"group_id": group["group_id"], "patient_id": group["patient_id"], "partition": "validation",
                         "raw": json.dumps({"detections": [{"phase": ph, "label": label, "bbox_2d": [0, 0, 10, 10]}
                                                           for ph in ("a", "p", "v")]}), "completion_nll": .1})
        if args.csv:
            with args.output.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["logical_key", "slice_index", "patient_id", "avg_nll", "raw"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({"logical_key": row["group_id"], "slice_index": 0, "patient_id": row["patient_id"],
                                     "avg_nll": row["completion_nll"], "raw": row["raw"]})
        else:
            args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
