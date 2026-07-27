#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

python3 -m pytest -s
python3 examples/reliability_weighting_demo.py
python3 -m bayesian_phystwin.cli.main commands --all --format json \
    >"$tmp_dir/commands.json"
test -s "$tmp_dir/commands.json"
python3 - <<'PY'
from bayesian_phystwin.cli.command_registry import iter_commands
from bayesian_phystwin.cli.main import main

assert main(["--help"]) == 0
assert main(["experiment", "list"]) == 0
for command in iter_commands(statuses={"stable"}):
    assert main([*command.route, "--help"]) == 0, command.route
PY
python3 -m bayesian_phystwin.cli.main residual replay \
    examples/residuals_demo.csv \
    --summary-json "$tmp_dir/summary.json" \
    --scored-csv "$tmp_dir/scored.csv" \
    >/dev/null
test -s "$tmp_dir/summary.json"
test -s "$tmp_dir/scored.csv"
python3 -m bayesian_phystwin.cli.main benchmark synthetic \
    --seeds 0 \
    --conditions correlated \
    --action-modes dynamic \
    --steps 30 \
    --train-steps 20 \
    --stiffness-count 5 \
    --damping-count 5 \
    --control-scale-count 5 \
    --output-json "$tmp_dir/synthetic.json" \
    --output-csv "$tmp_dir/synthetic.csv" \
    --output-reliability-csv "$tmp_dir/synthetic-reliability.csv" \
    >/dev/null
test -s "$tmp_dir/synthetic.json"
test -s "$tmp_dir/synthetic.csv"
test -s "$tmp_dir/synthetic-reliability.csv"
