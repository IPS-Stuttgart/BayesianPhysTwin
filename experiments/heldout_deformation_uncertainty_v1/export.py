"""Export source-only forecasts from the existing source-qualified DEFORM twin."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts' / 'remote'):
    sys.path.insert(0, str(p))
import numpy as np


def identity(path: Path) -> dict:
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from experiments.deform_dlo45_frozen_v1.core import (
        _load_named_from_manifest, _load_protocol, _setup_torch,
        _assert_upstream_and_initialization, posterior_runtime, source_runtime,
        local_runtime,
    )
    from experiments.deform_dlo45_frozen_v1.model import _load_full_model
    from bayesian_phystwin_experiments.deform_dlo_local_residual import predict_deform_local_residual

    parent = Path('/home/github-runner/.cache/workflows/deform-dlo45-time-budget-recovery-v3/runs/33361441865-1')
    data_root = Path('/mnt/seagate10tb/florianpfaff/datasets/deform/data_set').resolve()
    upstream = Path('/home/florianpfaff/source-only/DEFORM-b73b8b8')
    for dlo in ('DLO4', 'DLO5'):
        source_runtime._install_eval_read_guard(data_root / dlo / 'eval')
    manifest_out = {'source_revision': os.environ.get('GITHUB_SHA'), 'official_eval_opened': False, 'dlos': {}}
    for dlo in ('DLO4', 'DLO5'):
        source = parent / (dlo.lower() + '-source')
        seal = json.loads((source / 'method_seal.json').read_text())
        manifest = json.loads((source / 'source_manifest.json').read_text())
        assert seal['target_eval_read'] is False
        assert manifest['dlo'] == dlo
        verified = {}
        for label in ('source_manifest', 'physical_checkpoint', 'full_covariance_model', 'protocol'):
            p = Path(seal[label]['path']).resolve()
            if label != 'protocol' and not p.is_relative_to(source):
                raise ValueError('A source identity points outside its source directory')
            actual = identity(p)
            if actual['sha256'] != seal[label]['sha256']:
                raise ValueError('Source identity mismatch: ' + label)
            verified[label] = actual
        parts = manifest['partitions']
        assert {k: len(parts[k]) for k in ('fit', 'calibration', 'source_test')} == {'fit': 39, 'calibration': 9, 'source_test': 8}
        all_names = parts['fit'] + parts['calibration'] + parts['source_test']
        assert len(set(all_names)) == 56
        protocol = _load_protocol(Path(verified['protocol']['path']))
        device = 'cuda:0'
        _assert_upstream_and_initialization(protocol, upstream, dlo)
        torch = _setup_torch(protocol, device)
        source_runtime._seed_everything(torch, int(protocol['physical_training']['seed']))
        modules = source_runtime._load_upstream(upstream)
        state = dict(torch.load(verified['physical_checkpoint']['path'], map_location='cpu', weights_only=True)['model_state_dict'])
        model = _load_full_model(Path(verified['full_covariance_model']['path']))
        payload = {}
        for partition in ('fit', 'calibration', 'source_test'):
            names = parts[partition]
            trajectories = _load_named_from_manifest(manifest, names, frame_count=500, node_count=12)
            roll = posterior_runtime._evaluate_state(dict(state), trajectories, modules=modules, torch=torch, device=device, dlo_type=dlo, node_count=12)
            physical = np.asarray(roll['predictions'], dtype=np.float64)
            targets = np.asarray(roll['targets'], dtype=np.float64)
            initial, action = local_runtime._causal_inputs(trajectories, names)
            mean = np.asarray(predict_deform_local_residual(model, initial, action, physical, shrinkage=0.25)['predictions'], dtype=np.float64)
            assert mean.shape == targets.shape == (len(names), 498, 12, 3)
            assert np.isfinite(mean).all() and np.isfinite(targets).all()
            payload[partition + '_mean'] = mean
            payload[partition + '_truth'] = targets
            payload[partition + '_initial'] = np.stack([trajectories[n][1] for n in names])
            payload[partition + '_names'] = np.asarray(names)
            print(json.dumps({'dlo': dlo, 'partition': partition, 'count': len(names), 'shape': list(mean.shape), 'status': 'exported'}), flush=True)
        with np.load(source / 'source_predictions.npz', allow_pickle=False) as old:
            assert list(old['names']) == list(payload['source_test_names'])
            parity = float(np.max(np.abs(old['candidate'] - payload['source_test_mean'])))
            if parity > 1e-8:
                raise ValueError(f'Frozen source mean parity failed: {parity}')
        file = args.output / (dlo + '.npz')
        np.savez_compressed(file, **payload)
        manifest_out['dlos'][dlo] = {'identities': verified, 'carrier': identity(file), 'partitions': parts, 'source_mean_parity_max_abs_m': parity}
    (args.output / 'manifest.json').write_text(json.dumps(manifest_out, indent=2))
    print('SOURCE_ONLY_EXPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
