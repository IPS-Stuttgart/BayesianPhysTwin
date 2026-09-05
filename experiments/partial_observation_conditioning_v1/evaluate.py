"""Source-only DEFORM sparse-observation mechanism pilot; not official DEFORM inference.

No camera claim, no action acquisition, and no reserved evaluation access. A single
source-fitted linear dynamical surrogate is shared by all observation-update arms.
The linear Gaussian posterior mean also has an equivalent deterministic MAP solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np

D = 24
ARMS = ('full', 'diagonal', 'fixed_gain', 'frozen_cov', 'overwrite', 'graph',
        'ridge_readout', 'model_only')
CONDITIONS = ('center_hidden', 'sparse_two', 'rotating_half', 'gap10', 'all_visible')
CONFIG = {
    'contract': 'partial-observation-conditioning-v1',
    'dlos': ['DLO4', 'DLO5'], 'source_counts': [39, 9, 8],
    'split_domain': 'partial-observation-conditioning-v1',
    'frame_stride': 5, 'score_start_step': 5, 'horizons_steps': [1, 5, 10],
    'noise_std_mm_grid': [0.1, 0.5, 2.0], 'graph_gain_grid': [0.25, 0.5, 1.0],
    'readout_ridge_grid': [0.0001, 0.01, 1.0],
    'dynamics_ridge': 0.001, 'covariance_diagonal_shrinkage': 0.1,
    'covariance_floor_m2': 1e-8, 'maximum_spectral_radius': 0.999,
    'bootstrap_replicates': 10000, 'bootstrap_seed': 61023,
    'primary': 'equal-DLO equal-trajectory equal-condition hidden-node Euclidean RMSE',
    'data_scope': 'official-train-only retrospective intra-object pilot',
    'official_DEFORM_checkpoint_used': False, 'official_eval_access': False,
    'new_hardware': False, 'artificial_measurement_noise': False,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def ridge(x, y, penalty):
    mx, my = x.mean(0), y.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    z = (x - mx) / scale
    w = np.linalg.solve(z.T @ z + penalty * len(x) * np.eye(x.shape[1]), z.T @ (y - my))
    matrix = w.T / scale
    return matrix, my - matrix @ mx


def covariance(errors):
    p = errors.T @ errors / len(errors)
    a = CONFIG['covariance_diagonal_shrinkage']
    return (1-a)*p + a*np.diag(np.diag(p)) + CONFIG['covariance_floor_m2']*np.eye(D)


def split_names(names, dlo):
    if len(names) != 56 or len(set(names)) != 56:
        raise ValueError(f'{dlo}: expected 56 unique source trajectories')
    key = lambda name: hashlib.sha256((CONFIG['split_domain']+'\0'+dlo+'\0'+name).encode()).digest()
    order = sorted(names, key=key)
    return {'fit': order[:39], 'calibration': order[39:48], 'source_test': order[48:]}


def load_train(path):
    path = Path(path)
    if path.parent.name != 'train' or path.suffix != '.pkl':
        raise ValueError('Only explicitly named official train/*.pkl files may be read')
    # The user-authorized, local official dataset contains NumPy pickle arrays.
    with path.open('rb') as stream:
        raw = np.asarray(pickle.load(stream), dtype=np.float64)
    if raw.shape != (500, 3, 12) or not np.isfinite(raw).all():
        raise ValueError(f'Invalid official DEFORM trajectory: {path.name}, {raw.shape}')
    nodes = raw.transpose(0, 2, 1)[::CONFIG['frame_stride']].copy()
    # Orthogonal coordinate change only. No ground-plane clipping of scored truth.
    nodes = nodes[:, :, [0, 2, 1]]
    nodes[:, :, 2] *= -1
    midpoint = (nodes[:, 1] + nodes[:, 10])/2
    boundary = (nodes[:, [0, 1, 10, 11]] - midpoint[:, None]).reshape(len(nodes), 12)
    fractions = np.arange(1, 9)/9
    chord = nodes[:, 1, None] + fractions[None, :, None]*(nodes[:, 10, None]-nodes[:, 1, None])
    residual = (nodes[:, 2:10] - chord).reshape(len(nodes), D)
    return boundary, residual


def fit_model(records):
    b = np.concatenate([r[0] for r in records])
    y = np.concatenate([r[1] for r in records])
    init, init_bias = ridge(b, y, CONFIG['dynamics_ridge'])
    p0 = covariance(y - b @ init.T - init_bias)
    x = np.concatenate([np.c_[r[1][:-1], r[0][:-1], r[0][1:]] for r in records])
    target = np.concatenate([r[1][1:] for r in records])
    m, bias = ridge(x, target, CONFIG['dynamics_ridge'])
    a = m[:, :D].copy()
    rho_before = float(np.max(np.abs(np.linalg.eigvals(a))))
    if rho_before > CONFIG['maximum_spectral_radius']:
        a *= CONFIG['maximum_spectral_radius']/rho_before
    control, bias = ridge(x[:, D:], target-x[:, :D] @ a.T, CONFIG['dynamics_ridge'])
    q = covariance(target - x[:, :D] @ a.T - x[:, D:] @ control.T - bias)
    return {'A': a, 'B': control, 'c': bias, 'init': init, 'init_bias': init_bias,
            'P0': p0, 'Q': q, 'rho_before': np.array(rho_before)}


def masks(condition, length):
    visible = np.zeros((length, 8), dtype=bool)
    if condition in ('center_hidden', 'gap10'):
        visible[:, [0, 1, 6, 7]] = True
        if condition == 'gap10':
            visible[30:40] = False
            visible[65:75] = False
    elif condition == 'sparse_two':
        visible[:, [1, 6]] = True
    elif condition == 'rotating_half':
        for t in range(length):
            visible[t, [0, 1, 6, 7] if (t//10)%2 == 0 else [2, 3, 4, 5]] = True
    elif condition == 'all_visible':
        visible[:] = True
    else:
        raise ValueError(condition)
    return np.repeat(visible, 3, axis=1)


def gain(p, ix, variance):
    if not len(ix):
        return np.empty((D, 0))
    s = p[np.ix_(ix, ix)] + variance*np.eye(len(ix))
    return np.linalg.solve(s, p[ix]).T


def condition_gaussian(mean, p, ix, values, variance):
    k = gain(p, ix, variance)
    mean = mean + k @ (values-mean[ix])
    ih = np.eye(D)
    ih[:, ix] -= k
    posterior = ih @ p @ ih.T + variance*k @ k.T
    return mean, (posterior+posterior.T)/2


def stationary_gain(model, ix, variance):
    p = model['P0'].copy()
    for _ in range(300):
        prior = model['A'] @ p @ model['A'].T + model['Q']
        _, updated = condition_gaussian(np.zeros(D), prior, ix, np.zeros(len(ix)), variance)
        if np.max(np.abs(updated-p)) < 1e-12:
            p = updated
            break
        p = updated
    return gain(model['A'] @ p @ model['A'].T + model['Q'], ix, variance)


def fit_readouts(records, penalty):
    b = np.concatenate([r[0] for r in records])
    y = np.concatenate([r[1] for r in records])
    signatures = set()
    for name in CONDITIONS:
        signatures.update(tuple(np.flatnonzero(row)) for row in masks(name, 100))
    return {ix: ridge(np.c_[b, y[:, ix]], y, penalty) for ix in sorted(signatures)}


def replay(model, boundary, observed, arm, setting, readouts=None, audit=False):
    """Receives only endpoints and masked measurements; no scoring truth argument."""
    length = len(boundary)
    if observed.shape != (length, D) or not np.isfinite(boundary).all():
        raise ValueError('invalid observation contract')
    mean = model['init'] @ boundary[0] + model['init_bias']
    p = model['P0'].copy()
    estimates = np.empty_like(observed)
    variances = np.empty_like(observed)
    futures = {h: np.empty((length-h, D)) for h in CONFIG['horizons_steps']}
    audit_arrays = {k: np.empty_like(observed) for k in ('prior', 'full', 'diagonal', 'scrambled')} if audit else {}
    variance = (setting/1000)**2
    fixed = {}
    for t in range(length):
        if t:
            mean = model['A'] @ mean + model['B'] @ np.r_[boundary[t-1], boundary[t]] + model['c']
            if arm in ('full', 'diagonal'):
                p = model['A'] @ p @ model['A'].T + model['Q']
        ix = np.flatnonzero(np.isfinite(observed[t]))
        values = observed[t, ix]
        if audit:
            audit_arrays['prior'][t] = mean
            audit_arrays['full'][t] = condition_gaussian(mean, p, ix, values, variance)[0]
            audit_arrays['diagonal'][t] = condition_gaussian(mean, np.diag(np.diag(p)), ix, values, variance)[0]
            signs = np.repeat(np.where(np.arange(8)%2 == 0, 1., -1.), 3)
            scrambled = p*signs[:, None]*signs[None, :]
            audit_arrays['scrambled'][t] = condition_gaussian(mean, scrambled, ix, values, variance)[0]
        if arm in ('full', 'diagonal'):
            if arm == 'diagonal':
                p = np.diag(np.diag(p))
            mean, p = condition_gaussian(mean, p, ix, values, variance)
        elif arm in ('fixed_gain', 'frozen_cov'):
            key = tuple(ix)
            if key not in fixed:
                fixed[key] = stationary_gain(model, ix, variance) if arm == 'fixed_gain' else gain(model['P0'], ix, variance)
            mean = mean + fixed[key] @ (values-mean[ix])
        elif arm == 'overwrite':
            mean[ix] = values
        elif arm == 'graph':
            delta = np.zeros((8, 3))
            delta.reshape(-1)[ix] = values-mean[ix]
            visible_nodes = np.unique(ix//3)
            knots = np.r_[0, visible_nodes+1, 9]
            for axis in range(3):
                correction = np.interp(np.arange(1, 9), knots, np.r_[0, delta[visible_nodes, axis], 0])
                mean.reshape(8, 3)[:, axis] += setting*correction
        elif arm == 'ridge_readout':
            matrix, bias = readouts[tuple(ix)]
            mean = matrix @ np.r_[boundary[t], values] + bias
            mean[ix] = values
        elif arm != 'model_only':
            raise ValueError(arm)
        estimates[t] = mean
        variances[t] = np.diag(p)
        prediction = mean.copy()
        for h in range(1, min(max(CONFIG['horizons_steps']), length-1-t)+1):
            prediction = model['A'] @ prediction + model['B'] @ np.r_[boundary[t+h-1], boundary[t+h]] + model['c']
            if h in futures:
                futures[h][t] = prediction
    if not np.isfinite(estimates).all():
        raise FloatingPointError(f'{arm}: nonfinite predictions')
    return estimates, futures, variances, audit_arrays


def rmse_mm(estimate, truth, selection):
    squared = np.sum((estimate-truth).reshape(-1, 8, 3)**2, axis=2)
    pick = selection.reshape(-1, 8, 3)[:, :, 0]
    return None if not np.any(pick) else float(1000*np.sqrt(squared[pick].mean()))


def score(model, record, condition, arm, setting, readouts=None, audit=False):
    boundary, truth = record
    visible = masks(condition, len(truth))
    observed = np.where(visible, truth, np.nan)
    start = CONFIG['score_start_step']
    begin = time.perf_counter()
    estimates, futures, variances, audits = replay(model, boundary, observed, arm, setting, readouts, audit)
    hidden = ~visible
    hidden[:start] = False
    all_nodes = np.ones_like(visible)
    all_nodes[:start] = False
    row = {'hidden_rmse_mm': rmse_mm(estimates, truth, hidden),
           'all_node_rmse_mm': rmse_mm(estimates, truth, all_nodes),
           'seconds': time.perf_counter()-begin,
           'hidden_nodes_scored': int(hidden.sum()//3)}
    for h, predictions in futures.items():
        pick = hidden[:-h] if condition != 'all_visible' else all_nodes[:-h]
        row[f'forecast_{h*CONFIG["frame_stride"]}_frames_rmse_mm'] = rmse_mm(predictions, truth[h:], pick)
    if arm in ('full', 'diagonal') and np.any(hidden):
        row['hidden_coordinate_90_coverage'] = float(np.mean((np.abs(estimates-truth) <= 1.6448536269514722*np.sqrt(np.maximum(variances, 0)))[hidden]))
        row['hidden_coordinate_nanees'] = float(np.mean(((estimates-truth)**2/np.maximum(variances, 1e-30))[hidden]))
        row['hidden_coordinate_90_width_mm'] = float(np.mean((2*1.6448536269514722*np.sqrt(np.maximum(variances, 0)))[hidden])*1000)
    row['same_prior_audit_hidden_rmse_mm'] = {name: rmse_mm(value, truth, hidden) for name, value in audits.items()}
    return row


def calibrate(model, fit, calibration):
    chosen, history, readout_cache = {}, {}, {}
    for arm in ARMS:
        grid = CONFIG['graph_gain_grid'] if arm == 'graph' else CONFIG['readout_ridge_grid'] if arm == 'ridge_readout' else CONFIG['noise_std_mm_grid'] if arm in ('full', 'diagonal', 'fixed_gain', 'frozen_cov') else [1.0]
        losses = []
        for setting in grid:
            readouts = fit_readouts(fit, setting) if arm == 'ridge_readout' else None
            if readouts is not None:
                readout_cache[setting] = readouts
            values = [score(model, rec, con, arm, setting, readouts)['hidden_rmse_mm'] for rec in calibration for con in CONDITIONS[:-1]]
            losses.append(float(np.mean(values)))
        best = int(np.argmin(losses))
        chosen[arm] = grid[best]
        history[arm] = {'grid': grid, 'hidden_rmse_mm': losses, 'selected': grid[best]}
    return chosen, history, readout_cache[chosen['ridge_readout']]


def aggregate(rows):
    output = {}
    for condition in CONDITIONS:
        output[condition] = {}
        for arm in ARMS:
            subset = [r for r in rows if r['condition'] == condition and r['arm'] == arm]
            keys = ('hidden_rmse_mm', 'all_node_rmse_mm', 'forecast_5_frames_rmse_mm', 'forecast_25_frames_rmse_mm', 'forecast_50_frames_rmse_mm')
            output[condition][arm] = {k: None if subset[0][k] is None else float(np.mean([r[k] for r in subset])) for k in keys}
    trajectories = sorted(set((r['dlo'], r['trajectory']) for r in rows))
    vectors = {arm: np.array([np.mean([r['hidden_rmse_mm'] for r in rows if (r['dlo'], r['trajectory']) == case and r['arm'] == arm and r['condition'] != 'all_visible']) for case in trajectories]) for arm in ARMS}
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    ids = np.c_[rng.integers(0, 8, (CONFIG['bootstrap_replicates'], 8)), rng.integers(8, 16, (CONFIG['bootstrap_replicates'], 8))]
    comparisons = {}
    for arm in ARMS[1:]:
        difference = vectors['full']-vectors[arm]
        distribution = difference[ids].mean(1)
        comparisons[arm] = {'full_minus_comparator_mm': float(difference.mean()),
                            'paired_stratified_95_ci_mm': np.quantile(distribution, [.025, .975]).tolist(),
                            'paired_stratified_97p5_ci_mm': np.quantile(distribution, [.0125, .9875]).tolist(),
                            'trajectory_wins': int(np.sum(difference < -1e-10)),
                            'trajectory_ties': int(np.sum(np.abs(difference) <= 1e-10)),
                            'n_trajectories': len(difference),
                            'improvement_percent': float(100*(1-vectors['full'].mean()/vectors[arm].mean())),
                            'per_dlo_difference_mm': {dlo: float(difference[i*8:(i+1)*8].mean()) for i, dlo in enumerate(CONFIG['dlos'])}}
    return output, {arm: float(v.mean()) for arm, v in vectors.items()}, comparisons


def run(dataset_root, output, revision):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', CONFIG)
    models, selections, calibrations, readouts, splits, manifests = {}, {}, {}, {}, {}, {}
    for dlo in CONFIG['dlos']:
        directory = dataset_root/dlo/'train'
        paths = sorted(directory.glob('*.pkl'))
        splits[dlo] = split_names([p.name for p in paths], dlo)
        manifests[dlo] = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths}
        fit = [load_train(directory/name) for name in splits[dlo]['fit']]
        calibration = [load_train(directory/name) for name in splits[dlo]['calibration']]
        print(f'{dlo}: fitting shared surrogate and source-only tuning', flush=True)
        models[dlo] = fit_model(fit)
        selections[dlo], calibrations[dlo], readouts[dlo] = calibrate(models[dlo], fit, calibration)
        np.savez_compressed(output/f'{dlo}_source_model.npz', **models[dlo])
        write_json(output/f'{dlo}_selection.json', calibrations[dlo])
    # No source-test trajectory is deserialized until both model/selection seals exist.
    seal = {'revision': revision, 'protocol_sha256': digest(output/'protocol.json'), 'splits': splits,
            'source_file_manifest': manifests, 'settings': selections,
            'model_sha256': {d: digest(output/f'{d}_source_model.npz') for d in CONFIG['dlos']},
            'source_test_loaded': False, 'official_eval_loaded': False}
    write_json(output/'method_seal.json', seal)
    rows = []
    for dlo in CONFIG['dlos']:
        for name in splits[dlo]['source_test']:
            path = dataset_root/dlo/'train'/name
            if digest(path) != manifests[dlo][name]['sha256']:
                raise ValueError('Source file changed after seal')
            record = load_train(path)
            for condition in CONDITIONS:
                for arm in ARMS:
                    row = score(models[dlo], record, condition, arm, selections[dlo][arm], readouts[dlo], arm == 'full')
                    rows.append(dict(dlo=dlo, trajectory=name, condition=condition, arm=arm, **row))
            print(f'Scored {dlo}/{name}', flush=True)
    aggregates, overall, comparisons = aggregate(rows)
    joint = all(comparisons[arm]['paired_stratified_97p5_ci_mm'][1] < 0 and comparisons[arm]['improvement_percent'] >= 1 for arm in ('diagonal', 'overwrite'))
    best = all(comparisons[arm]['paired_stratified_95_ci_mm'][1] < 0 for arm in ('fixed_gain', 'frozen_cov', 'graph', 'ridge_readout'))
    result = {'contract': CONFIG['contract'], 'status': 'completed-source-only-pilot', 'config': CONFIG,
              'source_revision': revision, 'method_seal_sha256': digest(output/'method_seal.json'),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'runner': os.getenv('RUNNER_NAME', 'local')},
              'accounting': {'fit_trajectories': 78, 'calibration_trajectories': 18, 'source_test_trajectories': 16, 'physical_objects': 2, 'rows': len(rows), 'official_eval_opened': False, 'raw_arrays_uploaded': False},
              'overall_hidden_rmse_mm': overall, 'conditions': aggregates, 'comparisons': comparisons,
              'joint_conditioning_gate_passed': joint, 'strong_controls_all_beaten': best,
              'bayesian_exclusivity_claim': False,
              'limitation': 'Linear surrogate, not official DEFORM checkpoint. Deterministic Gaussian MAP is mean-equivalent. Artificial masks on real mocap; two already-studied objects; no camera or unseen-object claim.'}
    write_json(output/'result.json', result)
    with (output/'cases.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
    lines = ['# DEFORM partial-observation conditioning pilot', '', result['limitation'], '',
             '| Arm | Hidden-node RMSE (mm) |', '|---|---:|']
    lines += [f'| {arm} | {overall[arm]:.6f} |' for arm in ARMS]
    lines += ['', '| Comparator | Full minus comparator (mm) | 95% trajectory CI | Wins/16 |', '|---|---:|---|---:|']
    for arm, row in comparisons.items():
        lo, hi = row['paired_stratified_95_ci_mm']
        lines.append(f"| {arm} | {row['full_minus_comparator_mm']:.6f} | [{lo:.6f}, {hi:.6f}] | {row['trajectory_wins']} |")
    lines += ['', f'Joint-conditioning gate: {joint}. All strong controls beaten: {best}.',
              'Official evaluation split was not opened. Bootstrap resamples whole trajectories within each of two DLOs.',
              'Point gains cannot establish Bayesian exclusivity: this Gaussian posterior mean has a deterministic MAP equivalent.']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    run(args.dataset_root, args.output_dir, args.revision)
