#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os
from pathlib import Path
root=Path('/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/train-8cc85de7/training_run')
for name in ['source_manifest.json', 'preflight.json', 'training_validation_result.json']:
    p=root/name
    data=json.loads(p.read_text())
    print('FILE', str(p), 'KEYS', list(data))
    for k,v in data.items():
        if k not in ('validation','checkpoints','files','trajectories','records','split'):
            text=json.dumps(v)
            if len(text)<6000: print(k,text)
    if 'split' in data: print('SPLIT', json.dumps(data['split']))
    for k in ['files','trajectories']:
        if k in data: print('FIRST', k, repr(data[k])[:3000])
for parent in [Path('/home/florianpfaff/source-only'), Path('/home/github-runner'), Path('/opt')]:
    try:
        print('RUNTIME_DIRS', str(parent), sorted(p.name for p in parent.iterdir() if p.is_dir() and any(t in p.name.lower() for t in ['deform','venv','env','conda','python'])))
    except PermissionError:
        print('NOT_LISTABLE', str(parent))
PY
