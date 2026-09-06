#!/usr/bin/env python3
"""Exploratory DLO2 development comparison; never opens source-test or eval.

The nonlinear arm is a finite-rank Nystrom Matern-5/2 GP on the remaining
training residual of the existing ridge mean. All kernel hyperparameters and
shrinkages are selected on already-open development validation, not a new test.
This tests mean prediction only, not covariance calibration or graph coupling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist

ROOT = Path('/home/florianpfaff/source-only/deform-dlo2-local-residual-v5')
RIDGES = (0.1, 1.0, 10.0)
SHRINKAGES = (0.0, 0.25, 0.5, 1.0)
LENGTHS = (0.5, 1.0, 2.0)
NOISES = (0.1, 1.0, 10.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def matern(x: np.ndarray, z: np.ndarray, length: float) -> np.ndarray:
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1] or length <= 0:
        raise ValueError('invalid kernel inputs')
    r = np.sqrt(5.0) * cdist(x, z) / (length * np.sqrt(x.shape[1]))
    return (1.0 + r + r * r / 3.0) * np.exp(-r)


def nystrom(x: np.ndarray, q: np.ndarray, z: np.ndarray, length: float):
    kzz = matern(z, z, length)
    lower = np.linalg.cholesky(kzz + 1e-8 * np.eye(len(z)))
    train = solve_triangular(lower, matern(x, z, length).T, lower=True).T
    query = solve_triangular(lower, matern(q, z, length).T, lower=True).T
    return train, query


def gp_mean(phi: np.ndarray, query: np.ndarray, residual: np.ndarray, noise: float):
    if noise <= 0 or not np.isfinite(residual).all():
        raise ValueError('invalid GP likelihood')
    normal = phi.T @ phi + noise * np.eye(phi.shape[1])
    return query @ cho_solve(cho_factor(normal, lower=True), phi.T @ residual)


def self_test() -> dict:
    rng = np.random.default_rng(71)
    x, q = rng.normal(size=(19, 4)), rng.normal(size=(7, 4))
    y = rng.normal(size=(19, 3))
    k = matern(x, x, 1.0)
    assert np.linalg.eigvalsh(k).min() > -1e-10
    p, v = nystrom(x, q, x[:8], 1.0)
    actual = gp_mean(p, v, y, 0.7)
    dual = v @ p.T @ np.linalg.solve(p @ p.T + 0.7 * np.eye(len(p)), y)
    np.testing.assert_allclose(actual, dual, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(gp_mean(p, v, np.zeros_like(y), 0.7), np.zeros((7, 3)))
    # Ordinary linear GP primal/dual identity, independent of robust covariance.
    linear = q @ np.linalg.solve(x.T @ x + 0.7 * np.eye(4), x.T @ y)
    np.testing.assert_allclose(linear, q @ x.T @ np.linalg.solve(x @ x.T + 0.7 * np.eye(19), y), atol=1e-10)
    return {'kernel_psd': True, 'gp_primal_dual': True, 'zero_residual': True,
            'linear_gp_primal_dual': True}


def development_rows(manifest: dict, upstream: Path):
    """Normalize historical metadata without listing or loading data directories."""
    if isinstance(manifest.get('split'), dict) and isinstance(manifest.get('trajectories'), dict):
        fit = list(manifest['split']['fit'])
        val = list(manifest['split']['validation'])
        identities = manifest['trajectories']
        rows = {name: identities[name] for name in fit + val}
    else:
        assignments = manifest.get('assignments', {})
        files = manifest.get('files', [])
        if isinstance(files, dict):
            files = [dict(v, relative_path=k) for k, v in files.items()]
        rows = {}
        fit, val = [], []
        for row in files:
            name = str(row.get('relative_path', row.get('name', row.get('path', ''))))
            split = row.get('split', row.get('partition', assignments.get(name)))
            if split is None:
                for part in ('fit', 'validation'):
                    if name in assignments.get(part, []):
                        split = part
            if split in ('fit', 'validation'):
                rows[name] = row
                (fit if split == 'fit' else val).append(name)
    if len(fit) != 39 or len(val) != 9 or set(fit) & set(val):
        raise ValueError(f'development roster differs: fit={len(fit)}, validation={len(val)}')
    paths = {}
    for name, row in rows.items():
        raw = row.get('path', row.get('absolute_path', row.get('relative_path', name)))
        p = Path(raw)
        if not p.is_absolute():
            candidates = [upstream / folder / p for folder in ('data_set', 'dataset', 'data', '')]
            found = [c for c in candidates if c.is_file()]
            if len(found) != 1:
                raise ValueError(f'cannot uniquely resolve development identity {name}')
            p = found[0]
        p = p.resolve()
        if 'DLO2' not in p.parts or 'train' not in p.parts or 'eval' in p.parts:
            raise PermissionError(f'non-development trajectory: {p}')
        paths[name] = p
    return fit, val, rows, paths


def run(output: Path) -> None:
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    checks = self_test()
    train = ROOT / 'train-8cc85de7/training_run'
    training = json.loads((train / 'training_validation_result.json').read_text())
    manifest_path = train / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    preflight = json.loads((train / 'preflight.json').read_text())
    upstream = ROOT / 'source-8cc85de7/DEFORM'
    fit_names, val_names, identities, paths = development_rows(manifest, upstream)
    allowed = set(paths.values())
    accessed = set()

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.suffix in ('.pkl', '.pickle') and ('data_set' in p.parts or 'dataset' in p.parts or 'DLO2' in p.parts):
            if p not in allowed:
                raise PermissionError(f'blocked non-development data access: {p}')
            accessed.add(str(p))
        if 'eval' in p.parts and ('DLO2' in p.parts or 'DLO1' in p.parts):
            raise PermissionError(f'blocked official evaluation access: {p}')

    sys.addaudithook(audit)
    for name, p in paths.items():
        expected = identities[name].get('sha256')
        if not expected or digest(p) != expected:
            raise ValueError(f'trajectory identity mismatch: {name}')
    import torch
    import run_deform_dlo_source as source
    import run_deform_dlo2_local_residual as v5
    import run_deform_dlo_local_residual as local
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features, fit_deform_local_residual,
        predict_deform_local_residual,
    )
    source._seed_everything(torch, 42)
    modules = source._load_upstream(upstream)
    state, checkpoint = v5._checkpoint_state(training, torch=torch)
    development = source._load_named_trajectories(
        manifest, fit_names + val_names, frame_count=500, node_count=12)
    # Only the explicitly allowlisted development dictionary enters this helper.
    print('DEVELOPMENT_ROLLOUT_START', len(development), flush=True)
    rollout = v5._rollout(state, development, modules=modules, torch=torch, device='cuda:0')
    base = np.asarray(rollout['predictions'], dtype=np.float64)
    targets = np.asarray(rollout['targets'], dtype=np.float64)
    if tuple(rollout.get('names', fit_names + val_names)) != tuple(fit_names + val_names):
        raise ValueError('rollout case order differs')
    fi, fa = local._causal_inputs(development, fit_names)
    vi, va = local._causal_inputs(development, val_names)
    bf, bv, tf, tv = base[:39], base[39:], targets[:39], targets[39:]
    observed_baseline = float(np.mean(np.abs(bv - tv)))
    selected = training.get('selected_checkpoint', {})
    expected_baseline = selected.get('validation_l1_m', training.get('validation_l1_m'))
    if expected_baseline is None or abs(observed_baseline - float(expected_baseline)) > 1e-7:
        raise ValueError(f'baseline reproduction failed: {observed_baseline} vs {expected_baseline}')
    print('BASELINE_REPRODUCED_M', observed_baseline, flush=True)
    rows = []
    def record(name, prediction, parameters):
        if prediction.shape != tv.shape or not np.isfinite(prediction).all():
            raise ValueError('invalid prediction')
        clamps = [0, 1, prediction.shape[2] - 2, prediction.shape[2] - 1]
        np.testing.assert_array_equal(prediction[:, :, clamps], bv[:, :, clamps])
        err = np.mean(np.abs(prediction - tv), axis=(1, 2, 3))
        item = {'method': name, 'parameters': parameters, 'mean_l1_mm': float(err.mean()*1000),
                'per_case_l1_mm': [float(x*1000) for x in err]}
        rows.append(item)
        print('ARM', json.dumps(item, sort_keys=True), flush=True)

    record('baseline', bv, {})
    ridge_one = None
    for ridge in RIDGES:
        model = fit_deform_local_residual(fi, fa, bf, tf, fit_names,
                                         ridge=ridge, variance_floor_m2=1e-10)
        if ridge == 1.0:
            ridge_one = model
        for shrinkage in SHRINKAGES[1:]:
            pred = predict_deform_local_residual(model, vi, va, bv, shrinkage=shrinkage)
            record('ridge', pred['predictions'], {'ridge': ridge, 'shrinkage': shrinkage})
    model = ridge_one
    ffeat, frot = build_deform_local_residual_features(fi, fa, bf)
    vfeat, vrot = build_deform_local_residual_features(vi, va, bv)
    local_target = np.einsum('ntvi,nij->ntvj', tf-bf, frot)[:, :, 2:-2]
    nf, horizon, nodes, dimensions = ffeat.shape
    nv = len(vi)
    ridge_query = np.zeros((nv, horizon, nodes, 3))
    corrections = {(length, noise): np.zeros_like(ridge_query) for length in LENGTHS for noise in NOISES}
    for node in range(nodes):
        x = ((ffeat[:, :, node]-model['feature_location'][node])/model['feature_scale'][node]).reshape(-1, dimensions)
        q = ((vfeat[:, :, node]-model['feature_location'][node])/model['feature_scale'][node]).reshape(-1, dimensions)
        dx = np.column_stack((np.ones(len(x)), x))
        dq = np.column_stack((np.ones(len(q)), q))
        ridge_query[:, :, node] = (dq @ model['coefficients'][node]).reshape(nv, horizon, 3)
        remaining = local_target[:, :, node].reshape(-1, 3) - dx @ model['coefficients'][node]
        rng = np.random.default_rng(7300 + node)
        inducing = x[rng.choice(len(x), size=min(64, len(x)), replace=False)]
        for length in LENGTHS:
            p, v = nystrom(x, q, inducing, length)
            for noise in NOISES:
                corrections[length, noise][:, :, node] = gp_mean(p, v, remaining, noise).reshape(nv, horizon, 3)
        print('GP_NODE_COMPLETE', node, flush=True)
    # Reconstruct the native ridge mean to prove preprocessing/frame parity.
    native = predict_deform_local_residual(model, vi, va, bv, shrinkage=1.0)['predictions']
    reconstructed = bv.copy()
    reconstructed[:, :, 2:-2] += np.einsum('ntvj,nij->ntvi', ridge_query, vrot)
    np.testing.assert_allclose(native, reconstructed, atol=1e-12, rtol=1e-12)
    for (length, noise), correction in corrections.items():
        global_mean = np.einsum('ntvj,nij->ntvi', ridge_query+correction, vrot)
        for shrinkage in SHRINKAGES[1:]:
            prediction = bv.copy()
            prediction[:, :, 2:-2] += shrinkage * global_mean
            record('ridge_plus_matern_gp', prediction, {'ridge': 1.0, 'length': length,
                   'noise_variance': noise, 'inducing_count': 64, 'shrinkage': shrinkage})
    best_ridge = min((r for r in rows if r['method']=='ridge'), key=lambda r:r['mean_l1_mm'])
    best_gp = min((r for r in rows if r['method']=='ridge_plus_matern_gp'), key=lambda r:r['mean_l1_mm'])
    a, b = np.array(best_ridge['per_case_l1_mm']), np.array(best_gp['per_case_l1_mm'])
    result = {'contract': 'deform-gp-residual-development-v1',
              'evidence_status': 'exploratory-already-open-development-validation',
              'fit_trajectories': 39, 'validation_trajectories': 9, 'validation_names': val_names,
              'forecast_horizon': horizon, 'source_test_opened': False, 'official_eval_read': False,
              'fresh_confirmation_claim': False, 'calibration_claim': False,
              'method': 'finite-rank Nystrom Matern GP on remaining ridge training residual',
              'selection_caveat': 'best of 27 GP arms vs 9 ridge arms on the same development validation',
              'checks': dict(checks, native_ridge_mean_parity=True, clamped_nodes_exact=True,
                             baseline_reproduction=True, development_read_allowlist=True),
              'baseline_l1_mm': observed_baseline*1000, 'best_ridge': best_ridge, 'best_gp': best_gp,
              'gp_relative_improvement_over_best_ridge': float(1-b.mean()/a.mean()),
              'gp_wins_over_best_ridge': int((b<a).sum()), 'gp_max_case_ratio': float((b/a).max()),
              'all_arms': rows, 'accessed_trajectory_count': len(accessed),
              'source_identities': {'manifest_sha256': digest(manifest_path), 'checkpoint_sha256': digest(checkpoint),
                                    'training_result_sha256': digest(train/'training_validation_result.json')},
              'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'runtime': {'seconds': time.perf_counter()-start, 'numpy': np.__version__, 'torch': torch.__version__}}
    write(output/'result.json', result)
    print('RESULT_JSON', json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
    elif args.output is None:
        parser.error('--output is required')
    else:
        try:
            run(args.output)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if args.output.is_dir():
                write(args.output/'technical_failure.json', {'status':'technical-failure',
                      'error':str(exc), 'source_test_opened':False, 'official_eval_read':False})
            raise
