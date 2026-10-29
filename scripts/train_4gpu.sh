#!/usr/bin/env bash
set -euo pipefail
stage="${1:?Usage: bash scripts/train_4gpu.sh sft|rl [config.json]}"
case "$stage" in
  sft|rl) ;;
  *) echo "stage must be sft or rl" >&2; exit 2 ;;
esac
config="${2:-configs/${stage}_4xa800.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec torchrun --standalone --nnodes=1 --nproc_per_node=4 -m training.run "$stage" --config "$config"
