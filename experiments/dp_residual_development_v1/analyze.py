"""Fit-only selection and DLO2 retrospective-development scoring for issue #949.

All experts condition on the same baseline-derived field features. No forecast
outcome is supplied to an expert's gate. This pilot models each full free-node
field jointly but has no temporal latent-state model. The outer eight trajectories
are existing development validation, not a fresh test set.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, ndtr
import sklearn
from sklearn.ensemble import RandomForestRegressor

from model import ConditionalMixture, Specification

STRIDE = 25
STRENGTHS = (0.25, 0.5, 1.0)
BANK = [Specification('single', 1)] + [Specification('finite', k) for k in (2, 4, 8)] + [
    Specification('dp', 8, a) for a in (0.1, 1.0, 10.0)
] + [Specification('dp', 16, 1.0)]


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(error: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.einsum('ntvi,nij->ntvj', error[:, :, 2:-2], frames)


def field_to_global(correction: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.einsum('ntvj,nij->ntvi', correction, frames)


def prediction(parent: np.ndarray, correction: np.ndarray, frames: np.ndarray, strength: float) -> np.ndarray:
    n, t, v, _ = parent.shape
    out = parent.copy()
    out[:, :, 2:-2] += strength * field_to_global(correction.reshape(n, t, v-4, 3), frames)
    if not np.array_equal(out[:, :, [0, 1, v-2, v-1]], parent[:, :, [0, 1, v-2, v-1]]):
        raise AssertionError('Clamped nodes changed')
    return out


def point_record(predicted: np.ndarray, target: np.ndarray, names: np.ndarray) -> dict:
    error = predicted - target
    per = np.mean(np.abs(error), axis=(1, 2, 3))*1000
    return {'l1_mm': float(per.mean()), 'rmse_mm': float(np.sqrt(np.mean(error**2))*1000),
            'per_trajectory_l1_mm': dict(zip(names.tolist(), per.tolist())),
            'free_node_l1_mm': float(np.abs(error[:, :, 2:-2]).mean()*1000)}


def training_rows(a: dict, indices: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Collapse duplicate causal-query clusters before any mixture fitting."""
    groups = {}
    for i in indices:
        key = hashlib.sha256(a['initial'][i].tobytes()+a['action'][i].tobytes()).hexdigest()
        groups.setdefault(key, []).append(int(i))
    xx, yy = [], []
    for ids in groups.values():
        i = ids[0]
        if not all(np.array_equal(a['features'][i], a['features'][j]) for j in ids):
            raise ValueError('Identical causal query has conflicting predictor context')
        err = a['targets'][ids].mean(0, keepdims=True) - parent[i:i+1]
        y = canonical(err, a['frames'][i:i+1])[0]
        xx.append(a['features'][i, ::STRIDE].reshape(len(y[::STRIDE]), -1))
        yy.append(y[::STRIDE].reshape(len(y[::STRIDE]), -1))
    return np.concatenate(xx), np.concatenate(yy), len(groups)


def get_x(a: dict, indices: np.ndarray) -> np.ndarray:
    return a['features'][indices].reshape(len(indices)*a['features'].shape[1], -1)


def distribution_record(model: ConditionalMixture, x: np.ndarray, y: np.ndarray,
                        strength: float, shape: tuple[int, ...], hard: bool = False) -> dict:
    w, mu = model.components(x)
    mu = mu * strength
    if hard:
        w = np.eye(w.shape[1])[w.argmax(1)]
    terms = []
    for k, ch in enumerate(model.residual_cholesky):
        diff = y - mu[:, k]
        q = np.einsum('ni,ni->n', diff, cho_solve(ch, diff.T).T)
        terms.append(-0.5*(model.dy*np.log(2*np.pi)+2*np.log(np.diag(ch[0])).sum()+q))
    with np.errstate(divide='ignore'):
        nll = -logsumexp(np.log(w)+np.stack(terms, 1), axis=1)/model.dy
    sd = np.sqrt(np.diagonal(model.conditionals, axis1=1, axis2=2))
    lower0 = (mu-12*sd[None]).min(1)
    upper0 = (mu+12*sd[None]).max(1)
    quantiles = []
    for probability in (0.05, 0.95):
        lo, hi = lower0.copy(), upper0.copy()
        for _ in range(40):
            mid = (lo+hi)/2
            cdf = np.sum(w[:, :, None]*ndtr((mid[:, None]-mu)/sd[None]), axis=1)
            below = cdf < probability
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        quantiles.append((lo+hi)/2)
    lower, upper = quantiles
    covered = (y >= lower) & (y <= upper)
    n, t, _, _ = shape
    return {'joint_field_nll_per_coordinate': float(nll.mean()),
            'per_trajectory_nll': nll.reshape(n, t).mean(1).tolist(),
            'canonical_marginal_coverage90': float(covered.mean()),
            'canonical_marginal_width90_mm': float((upper-lower).mean()*1000),
            'density_units': 'meters; whole free-node field jointly per frame',
            'temporal_joint_density': False,
            'shrinkage_distribution': 'shrink component means; retain component conditional noise covariance'}


def paired(candidate: dict, reference: dict, names: list[str], cluster_keys: list[str]) -> dict:
    d = np.array([candidate['per_trajectory_l1_mm'][k] - reference['per_trajectory_l1_mm'][k] for k in names])
    groups = sorted(set(cluster_keys))
    grouped = np.array([d[np.array(cluster_keys)==k].mean() for k in groups])
    rng = np.random.default_rng(949)
    means = grouped[rng.integers(0, len(grouped), size=(10000, len(grouped)))].mean(1)
    return {'mean_delta_l1_mm_candidate_minus_reference': float(d.mean()),
            'relative_improvement': 1-candidate['l1_mm']/reference['l1_mm'],
            'paired_trajectory_wins': int((d < -1e-10).sum()),
            'ties': int((np.abs(d) <= 1e-10).sum()),
            'trajectory_count': len(d), 'causal_query_cluster_count': len(groups),
            'development_cluster_bootstrap_95_delta_mm': np.quantile(means, [.025, .975]).tolist(),
            'fresh_confirmation': False}


def main(input_path: Path, output: Path) -> None:
    started = time.perf_counter()
    if (output/'result.json').exists():
        raise FileExistsError('Retain results; use a new output directory')
    with np.load(input_path, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    if a['features'].shape[0] != 48 or len(a['names']) != 48 or a['baseline'].shape != a['targets'].shape:
        raise ValueError('Only original DLO2 40-fit / 8-validation export is supported')
    for k, value in a.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Non-finite input {k}')
    inner_train, inner_tune = a['inner_train'], a['inner_tune']
    if set(inner_train) & set(inner_tune) or set(inner_train) | set(inner_tune) != set(range(40)):
        raise ValueError('Inner split crossed fit boundary')
    forbidden_names = {'112.pkl','98.pkl','70.pkl','111.pkl','23.pkl','89.pkl','69.pkl','22.pkl'}
    if set(a['names']) & forbidden_names:
        raise ValueError('Source-test payload is forbidden')
    data_sha = sha(input_path)
    protocol = {'contract': 'dlo2-conditional-dp-mixture-development-v1',
        'issue': 949, 'input_sha256': data_sha, 'source_revision': os.environ.get('GITHUB_SHA'),
        'stride': STRIDE, 'strengths': STRENGTHS, 'bank': [asdict(s) for s in BANK],
        'single_nonlinear_control': {'family': 'RandomForestRegressor', 'depths': [4,8],
            'n_estimators': 64, 'min_samples_leaf': 8, 'seed': 42},
        'selection': 'minimum inner-tune mean trajectory L1 per family; tie by name then strength',
        'outer_validation_names': a['names'][40:].tolist(),
        'data_boundary': 'retrospective development only; baseline checkpoint historically selected on these validation records',
        'query_boundary': 'two observed states, known future clamped inputs, physical rollout only',
        'official_eval_read': False, 'source_test_opened': False,
        'temporal_regimes_implemented': False,
        'normalization': 'fit-only context PCA6 and response scales',
        'epistemic_parameter_integration': False}
    write(output/'protocol.json', protocol)
    xtrain, ytrain, nclusters = training_rows(a, inner_train, a['local_inner'])
    xtune = get_x(a, inner_tune)
    tuning = []
    for spec in BANK:
        model = ConditionalMixture(spec).fit(xtrain, ytrain)
        delta = model.predict(xtune)
        for strength in STRENGTHS:
            pred = prediction(a['local_inner'][inner_tune], delta, a['frames'][inner_tune], strength)
            record = {'name': spec.name, 'family': spec.family, 'strength': strength,
                'specification': asdict(spec), **point_record(pred, a['targets'][inner_tune], a['names'][inner_tune]),
                'fit': model.metadata()}
            tuning.append(record)
        print('INNER', spec.name, min(r['l1_mm'] for r in tuning if r['name']==spec.name), flush=True)
        write(output/'inner_tuning.json', {'training_clusters': nclusters, 'records': tuning})
    for depth in (4, 8):
        rf = RandomForestRegressor(n_estimators=64, max_depth=depth, min_samples_leaf=8,
            random_state=42, n_jobs=1).fit(xtrain, ytrain)
        delta = rf.predict(xtune)
        for strength in STRENGTHS:
            pred = prediction(a['local_inner'][inner_tune], delta, a['frames'][inner_tune], strength)
            tuning.append({'name': f'forest-depth{depth}', 'family': 'forest', 'strength': strength,
                'depth': depth, **point_record(pred, a['targets'][inner_tune], a['names'][inner_tune])})
    selected = {family: min((r for r in tuning if r['family']==family),
        key=lambda r: (r['l1_mm'], r['name'], r['strength'])) for family in ('single','finite','dp','forest')}
    write(output/'inner_tuning.json', {'training_clusters': nclusters, 'records': tuning})
    write(output/'selection_before_outer_scoring.json', {'input_sha256': data_sha, 'selected': selected,
        'outer_outcomes_used_for_selection': False, 'criterion': protocol['selection']})
    ids = np.arange(40,48)
    xtrain, ytrain, nclusters = training_rows(a, np.arange(40), a['local_full'])
    xquery = get_x(a, ids)
    target, parent, frames, names = a['targets'][ids], a['local_full'][ids], a['frames'][ids], a['names'][ids]
    residual = canonical(target-parent, frames).reshape(len(xquery), -1)
    results = {'physical': point_record(a['baseline'][ids], target, names),
               'current_local_residual': point_record(parent, target, names)}
    variance = a['local_variance_full'][ids, :, 2:-2]
    global_error = target[:, :, 2:-2]-parent[:, :, 2:-2]
    nll = 0.5*(np.log(2*np.pi*variance)+global_error**2/variance)
    canonical_var = np.einsum('ntvi,nij->ntvj', variance, frames**2)
    canonical_error = residual.reshape(len(ids), parent.shape[1], parent.shape[2]-4, 3)
    results['current_local_residual']['distribution'] = {
        'joint_field_nll_per_coordinate': float(nll.mean()),
        'canonical_marginal_coverage90': float((np.abs(canonical_error)<=1.6448536269514722*np.sqrt(canonical_var)).mean()),
        'canonical_marginal_width90_mm': float((2*1.6448536269514722*np.sqrt(canonical_var)).mean()*1000),
        'density_units': 'meters; product of existing global marginal Gaussian variances',
        'temporal_joint_density': False}
    predictions = {'physical': a['baseline'][ids], 'current_local_residual': parent}
    for family, choice in selected.items():
        if family == 'forest':
            model = RandomForestRegressor(n_estimators=64, max_depth=choice['depth'], min_samples_leaf=8,
                random_state=42, n_jobs=1).fit(xtrain, ytrain)
        else:
            model = ConditionalMixture(Specification(**choice['specification'])).fit(xtrain, ytrain)
        delta = model.predict(xquery)
        pred = prediction(parent, delta, frames, choice['strength'])
        predictions[family] = pred
        results[family] = {'chosen_name': choice['name'], 'strength': choice['strength'],
            **point_record(pred, target, names)}
        if family != 'forest':
            results[family]['fit'] = model.metadata()
            results[family]['distribution'] = distribution_record(model, xquery, residual, choice['strength'], target.shape)
            hard = prediction(parent, model.predict(xquery, hard=True), frames, choice['strength'])
            predictions[family+'_hard'] = hard
            results[family+'_hard'] = {**point_record(hard, target, names),
                'distribution': distribution_record(model, xquery, residual, choice['strength'], target.shape, hard=True)}
        print('OUTER', family, json.dumps(results[family]), flush=True)
        write(output/'partial_result.json', {'results': results, 'selection': selected})
    keys = [hashlib.sha256(a['initial'][i].tobytes()+a['action'][i].tobytes()).hexdigest() for i in ids]
    comparisons = {}
    for family in ('single','finite','dp','forest'):
        comparisons[family+'_vs_current'] = paired(results[family], results['current_local_residual'], names.tolist(), keys)
    comparisons['dp_vs_finite'] = paired(results['dp'], results['finite'], names.tolist(), keys)
    np.savez_compressed(output/'predictions.npz', names=names, **predictions)
    result = {'contract': protocol['contract'], 'claim_boundary': protocol['data_boundary'],
        'input_sha256': data_sha, 'protocol_sha256': sha(output/'protocol.json'),
        'prediction_sha256': sha(output/'predictions.npz'), 'source_revision': os.environ.get('GITHUB_SHA'),
        'fit_trajectory_count': 40, 'fit_causal_query_clusters': nclusters,
        'development_validation_count': 8, 'results': results, 'comparisons': comparisons,
        'selection': {k: {'name': v['name'], 'strength': v['strength'], 'inner_l1_mm':v['l1_mm']} for k,v in selected.items()},
        'source_test_opened': False, 'official_eval_read': False,
        'limitations': ['retrospective development, not independent confirmation',
            'single optimizer initialization; no concentration posterior',
            'finite-truncated variational conditional mixtures, not sticky HDP dynamics',
            'temporal dependence omitted from training likelihood and prediction distribution',
            'joint spatial free-node field covariance retained',
            'no integrated parameter posterior or independent uncertainty calibration'],
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'elapsed_seconds': time.perf_counter()-started}}
    write(output/'result.json', result)
    print('RESULT', json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        main(args.input, args.output)
    except Exception as error:
        write(args.output/'analysis_failure.json', {'type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(), 'official_eval_read': False, 'source_test_opened': False})
        raise
