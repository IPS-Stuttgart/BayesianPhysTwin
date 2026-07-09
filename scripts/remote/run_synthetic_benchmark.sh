#!/usr/bin/env bash
set -euo pipefail

host="${1:-gpuserver4090}"
repo_dir="${2:-Bayesian-PhysTwin}"
run_dir="${3:-runs/synthetic_v3}"
python_bin="${PYTHON:-python3}"

ssh "$host" "cd '$repo_dir' && mkdir -p '$run_dir' && PYTHONPATH=src $python_bin -m bayesian_phystwin.cli.synthetic_benchmark --seeds 1000:1020 --conditions clean,iid,correlated --action-modes dynamic,quasi_static --bias-process-variance 1e-5 --bias-initial-variance 1e-7 --bias-cue-persistence 0.85 --bias-cue-threshold 0.20 --bias-minimum-run-length 5 --output-json '$run_dir/results.json' --output-csv '$run_dir/aggregate.csv' --output-reliability-csv '$run_dir/reliability.csv'"
