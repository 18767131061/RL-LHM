import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "statistics"))

import apply_r1_1_bh  # noqa: E402


def paired_payload(cohort, seed, p):
    return {
        "reference": "reference",
        "candidate": "candidate",
        "mcnemar": {"exact_two_sided_p": p},
        "paired_permutation_two_sided_p": {
            "accuracy": p,
            "macro_recall": p + 0.01,
            "macro_f1": p + 0.02,
        },
        "provenance": {
            "cohort": cohort,
            "model_seed": seed,
            "aggregation_rule": "majority_nll_tiebreak",
            "evaluation_version": "v1",
            "test_manifest": {"sha256": "manifest-sha"},
            "reference_prediction": {"sha256": "reference-sha"},
            "candidate_prediction": {"sha256": "candidate-sha"},
        },
    }


def test_r1_1_bh_uses_one_60_hypothesis_family(tmp_path, monkeypatch):
    roots = {"private": tmp_path / "private", "public": tmp_path / "public"}
    for cohort, root in roots.items():
        root.mkdir()
        for seed in apply_r1_1_bh.SEEDS:
            for index, (reference, candidate) in enumerate(apply_r1_1_bh.COMPARISONS):
                path = root / f"seed_{seed}_{reference}_vs_{candidate}.json"
                path.write_text(
                    json.dumps(paired_payload(cohort, seed, 0.01 + index * 0.01)),
                    encoding="utf-8",
                )
    output = tmp_path / "r1_1_bh.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_r1_1_bh.py",
            "--private-paired-root",
            str(roots["private"]),
            "--public-paired-root",
            str(roots["public"]),
            "--output",
            str(output),
        ],
    )
    apply_r1_1_bh.main()
    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 21
    payload = json.loads(
        (roots["private"] / "seed_42_vanilla_grpo_vs_full_rl_lhm.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["bh"]["hypothesis_count"] == 60
    assert payload["bh"]["family_id"].endswith("n60")
    assert set(payload["bh"]["q_values"]) == {"accuracy", "macro_recall", "macro_f1"}
