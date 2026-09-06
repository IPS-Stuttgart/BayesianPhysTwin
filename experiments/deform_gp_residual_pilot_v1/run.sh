#!/usr/bin/env bash
set -euo pipefail
out="$1"
python=/home/florianpfaff/source-only/deform-bayesian-v1/venv/bin/python
test -x "$python"
export PYTHONPATH="$GITHUB_WORKSPACE/src:$GITHUB_WORKSPACE/scripts/remote:$GITHUB_WORKSPACE/experiments/deform_gp_residual_pilot_v1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cache=/home/github-runner/.cache/bpt-gp-residual-pilot-v1
mkdir -p "$cache"
"$python" -c 'import sys, numpy, scipy, torch; print("RUNTIME", sys.version, numpy.__version__, scipy.__version__, torch.__version__)'
"$python" experiments/deform_gp_residual_pilot_v1/export.py \
  --cache "$cache" --partitions validation fit --output "$out/export.json"
phase=$(python3 -c "import json; print(json.load(open('.github/requests/deform-gp-residual-pilot-v1.json'))['phase'])")
if [ "$phase" = gp-comparison ]; then
  "$python" experiments/deform_gp_residual_pilot_v1/compare.py --cache "$cache" --output "$out"
fi
