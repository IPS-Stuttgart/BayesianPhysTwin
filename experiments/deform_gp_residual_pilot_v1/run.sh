#!/usr/bin/env bash
set -euo pipefail
out="$1"
legacy=/home/florianpfaff/source-only/deform-bayesian-v1/venv
export PYTHONPATH="$GITHUB_WORKSPACE/src:$GITHUB_WORKSPACE/scripts/remote:$GITHUB_WORKSPACE/experiments/deform_gp_residual_pilot_v1:$legacy/lib/python3.10/site-packages"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cache=/home/github-runner/.cache/bpt-gp-residual-pilot-v1
mkdir -p "$cache"
python - <<'PY'
import sys
from pathlib import Path
root=Path('/home/florianpfaff/source-only/deform-bayesian-v1/venv')
print('INTERPRETER', sys.executable, sys.version)
print('LEGACY_CONFIG', (root/'pyvenv.cfg').read_text())
for p in (root/'lib').glob('python*'):
    print('LEGACY_PACKAGES', p, (p/'site-packages/numpy').exists(), (p/'site-packages/torch').exists())
import numpy, scipy, torch
print('RUNTIME', numpy.__version__, scipy.__version__, torch.__version__)
PY
python experiments/deform_gp_residual_pilot_v1/export.py \
  --cache "$cache" --partitions validation fit --output "$out/export.json"
phase=$(python -c "import json; print(json.load(open('.github/requests/deform-gp-residual-pilot-v1.json'))['phase'])")
if [ "$phase" = gp-comparison ]; then
  python experiments/deform_gp_residual_pilot_v1/compare.py --cache "$cache" --output "$out"
fi
