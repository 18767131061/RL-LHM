#!/usr/bin/env bash
# Create the official pinned Linux FSDP2/vLLM environment in a separate directory.
set -euo pipefail

if [[ $# != 1 || "$1" != /* ]]; then
    echo "Usage: bash scripts/setup_verl.sh /absolute/path/to/verl-runtime" >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo "The pinned VeRL GPU environment requires Linux and a CUDA 13-compatible driver." >&2
    exit 2
fi
command -v git >/dev/null
command -v uv >/dev/null

runtime_dir="$1"
source_commit="a911bca25d6adb26f5bef2dc57e7ba62e99b7ade"
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e "$runtime_dir" ]]; then
    echo "Choose a new runtime directory; the requested directory already exists." >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$runtime_dir")"
git init "$runtime_dir"
git -C "$runtime_dir" remote add origin https://github.com/verl-project/verl.git
git -C "$runtime_dir" fetch --depth 1 origin "$source_commit"
git -C "$runtime_dir" checkout --detach FETCH_HEAD
test "$(git -C "$runtime_dir" rev-parse HEAD)" = "$source_commit"

(
    cd -- "$runtime_dir"
    uv sync --frozen --all-packages --extra fsdp --extra vllm --python 3.12
)
uv pip install --python "$runtime_dir/.venv/bin/python" -r "$project_dir/requirements-verl.txt"
(
    cd -- "$project_dir"
    "$runtime_dir/.venv/bin/python" -m training.verl_run --config configs/rl_verl_4xa800.json --check-config >/dev/null
)
printf 'VeRL environment: %s/.venv\n' "$runtime_dir"
printf 'Activate: source %q/.venv/bin/activate\n' "$runtime_dir"
