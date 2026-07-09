#!/usr/bin/env bash
set -euo pipefail

host="${1:-gpuserver4090}"
repo_dir="${2:-Bayesian-PhysTwin}"
python_bin="${PYTHON:-python3}"

ssh "$host" "cd '$repo_dir' && $python_bin -m pip install -e '.[dev]' && bash scripts/local_smoke_test.sh"
