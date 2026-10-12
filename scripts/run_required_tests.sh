#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
LOG_DIR="${RL_LHM_TEST_LOG_DIR:-$ROOT_DIR/outputs/test_logs}"
PYTHON_BIN="${RL_LHM_PYTHON:-python}"
if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--cpu-only") ]]; then
  printf 'Usage: %s [--cpu-only]\n' "$0" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"
TESTS=(tests/test_iou.py tests/test_hungarian_matching.py tests/test_three_matching_protocols.py tests/test_detection_metrics.py tests/test_classification_metrics.py tests/test_patient_aggregation.py tests/test_four_class_loco.py tests/test_rars_weights.py tests/test_paired_patient_tests.py tests/test_apply_r1_1_bh.py tests/test_data_manifest.py tests/test_full_reward.py tests/test_paper_protocol_cli.py tests/test_reward_schema_contract.py tests/test_training_data_cli.py)
if [[ "${1:-}" != "--cpu-only" ]]; then
  TESTS+=(tests/test_preprocess_and_inference.py tests/test_rl_core.py tests/test_verl_sample_uid.py)
fi
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m pytest "${TESTS[@]}" -q -p no:cacheprovider 2>&1 | tee "$LOG_DIR/required_tests.log"
