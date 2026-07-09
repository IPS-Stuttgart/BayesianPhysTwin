#!/usr/bin/env bash
set -euo pipefail

host="${1:-gpuserver4090}"
repo_dir="${2:-Bayesian-PhysTwin}"
run_dir="${3:-runs/synthetic_v1}"
python_bin="${PYTHON:-python3}"

ssh "$host" "cd '$repo_dir' && mkdir -p '$run_dir' && PYTHONPATH=src $python_bin -m bayesian_phystwin.cli.synthetic_benchmark --seeds 0:20 --conditions clean,iid,correlated --action-modes dynamic,quasi_static --output-json '$run_dir/results.json' --output-csv '$run_dir/aggregate.csv'"
