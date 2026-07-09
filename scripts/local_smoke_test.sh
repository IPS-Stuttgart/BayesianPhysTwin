#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

python3 -m pytest -s
python3 examples/reliability_weighting_demo.py
python3 -m bayesian_phystwin.cli.residual_replay \
    examples/residuals_demo.csv \
    --summary-json "$tmp_dir/summary.json" \
    --scored-csv "$tmp_dir/scored.csv" \
    >/dev/null
test -s "$tmp_dir/summary.json"
test -s "$tmp_dir/scored.csv"
