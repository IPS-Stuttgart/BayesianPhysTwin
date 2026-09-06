"""Export DLO2 development-only residual evidence; never reopen frozen targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path('/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/train-8cc85de7/training_run')
UPSTREAM = Path('/home/florianpfaff/source-only/deform-bayesian-v1/DEFORM-b73b8b8')
TRAIN_SHA = '1f8d092bc38b03f6cdd68ef38abcb7d403d914e38ba483698579deaeea8c2572'
MANIFEST_SHA = '7c5501997e6bab7b0537ef9cda932ec19312e40618f03a4fad80ffc1622a6d98'
CHECKPOINT_SHA = 'b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main(output: Path) -> None:
    import run_deform_dlo2_local_residual as v5
    import run_deform_dlo_local_residual as local
    import run_deform_dlo_source as source
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features,
        fit_deform_local_residual,
        load_deform_dlo2_local_residual_protocol,
        predict_deform_local_residual,
    )
    started = time.perf_counter()
    trainpath, manifestpath = ROOT/'training_validation_result.json', ROOT/'source_manifest.json'
    if sha(trainpath) != TRAIN_SHA or sha(manifestpath) != MANIFEST_SHA:
        raise ValueError('Original development metadata hash mismatch')
    protocolpath = Path('configs/sota/deform_dlo2_local_residual_v5.json').resolve()
    protocol = load_deform_dlo2_local_residual_protocol(protocolpath)
    training, manifest, _ = v5._verify_training_result(trainpath, protocol=protocol, protocol_path=protocolpath)
    fitnames, valnames = list(manifest['split']['fit']), list(manifest['split']['validation'])
    forbidden = set(manifest['split']['source_test'])
    names = fitnames + valnames
    if len(fitnames) != 40 or len(valnames) != 8 or len(set(names)) != 48 or set(names) & forbidden:
        raise ValueError('Development partitions differ from the original 40/8 contract')
    allowed = {str((UPSTREAM/'data_set'/'DLO2'/'train'/name).resolve()) for name in names}
    reads: set[str] = set()

    def guard(event: str, args: tuple) -> None:
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if 'eval' in p.parts and 'data_set' in p.parts:
            raise PermissionError('Official evaluation is forbidden in this pilot')
        if p.suffix == '.pkl' and 'data_set' in p.parts:
            if str(p) not in allowed:
                raise PermissionError('Non-development trajectory access denied')
            reads.add(str(p))

    sys.addaudithook(guard)
    source._install_eval_read_guard(UPSTREAM/'data_set'/'DLO2'/'eval')
    source._install_eval_read_guard(UPSTREAM/'data_set'/'DLO1'/'eval')
    source._assert_upstream(UPSTREAM, protocol['upstream']['commit'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    source._seed_everything(torch, 42)
    modules = source._load_upstream(UPSTREAM)
    state, checkpoint = v5._checkpoint_state(training, torch=torch)
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Development checkpoint changed')
    write(output/'access_manifest.json', {
        'stage': 'before-development-read', 'source_test_opened': False,
        'official_eval_read': False, 'fit_names': fitnames, 'validation_names': valnames,
        'training_sha256': TRAIN_SHA, 'manifest_sha256': MANIFEST_SHA,
        'checkpoint_sha256': CHECKPOINT_SHA, 'source_revision': os.environ.get('GITHUB_SHA'),
        'claim_boundary': 'retrospective development diagnostic; not fresh confirmation',
    })
    trajectories = source._load_named_trajectories(manifest, names, frame_count=500, node_count=12)
    rollout = v5._rollout(state, trajectories, modules=modules, torch=torch, device='cuda:0')
    baseline, targets = np.asarray(rollout['predictions']), np.asarray(rollout['targets'])
    initial, action = local._causal_inputs(trajectories, names)
    if len(baseline) != 48 or baseline.shape != targets.shape:
        raise ValueError('Unexpected rollout contract')
    expected = float(training['selected_checkpoint']['validation_l1_m'])
    actual = float(np.mean(np.abs(baseline[40:]-targets[40:])))
    if abs(actual-expected) > 1e-7:
        raise ValueError(f'Physical baseline parity failed: {actual} != {expected}')
    features, frames = build_deform_local_residual_features(initial, action, baseline)
    fingerprints = [hashlib.sha256(a.tobytes()+u.tobytes()).hexdigest() for a,u in zip(initial,action)]
    if set(fingerprints[:40]) & set(fingerprints[40:]):
        raise ValueError('A duplicate causal query crosses fit/development validation')
    groups = sorted(set(fingerprints[:40]))
    tune_groups = set(groups[::5])
    inner_train = np.array([i for i in range(40) if fingerprints[i] not in tune_groups], dtype=int)
    inner_tune = np.array([i for i in range(40) if fingerprints[i] in tune_groups], dtype=int)
    if len(inner_train) < 10 or len(inner_tune) < 2:
        raise ValueError('Insufficient independent causal-query groups')
    predictions = {}
    for label, idx in [('inner',inner_train), ('full',np.arange(40))]:
        fitted = fit_deform_local_residual(initial[idx], action[idx], baseline[idx], targets[idx],
            [names[i] for i in idx], ridge=1.0, variance_floor_m2=1e-6)
        pred = predict_deform_local_residual(fitted, initial, action, baseline, shrinkage=.25)
        predictions['local_'+label] = pred['predictions']
        predictions['local_variance_'+label] = pred['coordinate_variance_m2']
    payload = dict(names=np.array(names), initial=initial, action=action,
        features=features, frames=frames, baseline=baseline, targets=targets,
        inner_train=inner_train, inner_tune=inner_tune, **predictions)
    archive = output/'development_arrays.npz'
    if archive.exists():
        raise FileExistsError(archive)
    np.savez_compressed(archive, **payload)
    report = {
        'schema_version': 1, 'stage': 'development-export-complete',
        'source_revision': os.environ.get('GITHUB_SHA'), 'array_sha256': sha(archive),
        'training_sha256': TRAIN_SHA, 'manifest_sha256': MANIFEST_SHA,
        'checkpoint_sha256': CHECKPOINT_SHA, 'names': names,
        'inner_train_names': [names[i] for i in inner_train],
        'inner_tune_names': [names[i] for i in inner_tune],
        'read_paths': sorted(reads), 'official_eval_read': False, 'source_test_opened': False,
        'physical_validation_l1_mm': actual*1000,
        'local_validation_l1_mm': float(np.mean(np.abs(predictions['local_full'][40:]-targets[40:])))*1000,
        'shapes': {k: list(v.shape) for k,v in payload.items()},
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                    'elapsed_seconds': time.perf_counter()-started},
        'claim_boundary': 'existing development validation only; no independent confirmation',
    }
    write(output/'export_result.json', report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        main(args.output_root)
    except Exception as error:
        write(args.output_root/'technical_failure.json', {'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc(),
            'source_test_opened': False, 'official_eval_read': False})
        raise
