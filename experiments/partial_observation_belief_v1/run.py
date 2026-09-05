"""Retrospective real-trajectory partial-observation mechanism test; NumPy only.

No simulator/checkpoint superiority claim: the shared mean is a source-fitted
ridge correction of endpoint-transported, damped-velocity geometry. Only train/
is read. Recorded boundary motion is known; masks are fixed, not selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARMS = ('prior', 'joint_lowrank', 'diagonal', 'scrambled', 'empirical',
        'interpolation', 'masked_ridge', 'equivalent_map')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True,
                                   allow_nan=False) + '\n')


def masks():
    result = {}
    for k in (1, 2, 4):
        uniform = np.array([3]) if k == 1 else np.rint(np.linspace(0, 7, k)).astype(int)
        start = k // 2
        middle = np.setdiff1d(np.arange(8), np.arange(start, start + 8 - k))
        for name, ids in [('uniform', uniform), ('hidden_middle', middle),
                          ('hidden_right', np.arange(k))]:
            result[f'{name}_{k}'] = ids
    return result


def coords(ids):
    return (np.asarray(ids)[:, None] * 3 + np.arange(3)).ravel()


def read_trajectory(path):
    # Only the user-provided, trusted local DEFORM training pickle files.
    with Path(path).open('rb') as stream:
        a = np.asarray(pickle.load(stream), dtype=np.float64)
    if a.shape != (500, 3, 12) or not np.isfinite(a).all():
        raise ValueError(f'Invalid DEFORM array: {path}, {a.shape}')
    # Preserve measured coordinates: no clamp/clipping of held-out truth.
    return a.transpose(0, 2, 1).copy()


def observation(a, t, p):
    """Prior construction never reads internal nodes after t."""
    times = t + p['gap'] + np.array(p['offsets'])
    prefix = a[t-1:t+1].copy()
    boundary = a[times][:, [0, 1, 10, 11]].copy()
    left = prefix[:, :2].mean(axis=1)
    right = prefix[:, -2:].mean(axis=1)
    w = (np.arange(1, 9) / 9)[None, :, None]
    line = (1-w) * left[:, None] + w * right[:, None]
    shape = prefix[:, 2:10] - line
    future_line = ((1-w) * boundary[:, :2].mean(axis=1)[:, None]
                   + w * boundary[:, 2:].mean(axis=1)[:, None])
    dt = times - t
    factor = .85 * (1 - .85 ** dt) / (1 - .85)
    base = future_line + shape[-1] + factor[:, None, None] * (shape[-1]-shape[-2])
    length = max(float(np.linalg.norm(right[-1]-left[-1])), .01)
    feature = np.concatenate((shape[-1].ravel(), (shape[-1]-shape[-2]).ravel(),
                              (boundary-boundary[:1]).ravel(),
                              (boundary[0]-prefix[-1, [0, 1, 10, 11]]).ravel())) / length
    return feature, base.reshape(-1), times


def rows(paths, p):
    features, targets, groups = [], [], []
    for g, path in enumerate(paths):
        a = read_trajectory(path)
        for t in p['starts']:
            x, base, times = observation(a, t, p)
            features.append(x)
            targets.append(a[times, 2:10].reshape(-1)-base)
            groups.append(g)
    return np.array(features), np.array(targets), np.array(groups)


def fit_ridge(x, y, penalty):
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    z = (x-center)/scale
    ym = y.mean(axis=0)
    b = np.linalg.solve(z.T@z + penalty*np.eye(z.shape[1]), z.T@(y-ym))
    return center, scale, ym, b


def predict(model, x):
    center, scale, ym, b = model
    return (x-center)/scale @ b + ym


def covariance(e, rank=None, shrink=None):
    s = e.T@e/len(e)
    variance = np.maximum(np.diag(s), 1e-10)
    if shrink is not None:
        return (1-shrink)*s + shrink*np.diag(variance) + 1e-10*np.eye(s.shape[0])
    vals, vecs = np.linalg.eigh(s)
    b = vecs[:, -rank:] * np.sqrt(np.maximum(vals[-rank:], 0))
    return b@b.T + np.diag(np.maximum(variance-np.sum(b*b, axis=1), 1e-10))


def gain(c, ids, noise):
    j = coords(ids)
    return np.linalg.solve(c[np.ix_(j, j)]+noise**2*np.eye(len(j)), c[j]).T


def conditional(c, ids, noise):
    k = gain(c, ids, noise)
    v = c-k@c[coords(ids)]
    return k, np.maximum(np.diag((v+v.T)/2), 1e-12)


def map_gain(c, ids, noise):
    # Independent full precision-space solve, not an alias of the gain formula.
    j = coords(ids)
    h = np.eye(len(c))[j]
    precision = np.linalg.solve(c, np.eye(len(c))) + h.T@h/noise**2
    return np.linalg.solve(precision, h.T/noise**2)


def interpolation(e, ids):
    result = np.zeros((len(e), 24))
    for i in range(len(e)):
        for axis in range(3):
            result[i, axis::3] = np.interp(np.arange(2, 10),
                np.r_[1, np.asarray(ids)+2, 10], np.r_[0., e[i, axis::3], 0.])
    return np.tile(result, (1, 3))


def loss(error, groups, ids):
    hidden = coords(np.setdiff1d(np.arange(8), ids))
    per_group = []
    for group in np.unique(groups):
        a = error[groups == group]
        per_group.append([np.sqrt(np.mean(a[:, hidden]**2)),
                          np.sqrt(np.mean(a[:, 48:72]**2))])
    return float(np.mean(per_group))


def candidate_score(c, sigma, e, groups):
    return float(np.mean([loss(e-e[:, coords(ids)]@gain(c, ids, sigma).T,
                                  groups, ids) for ids in masks().values()]))


def fit_models(paths, calibration, p):
    x, y, group = rows(paths, p)
    xc, yc, gc = rows(calibration, p)
    model = fit_ridge(x, y, p['mean_ridge'])
    oof = np.empty_like(y)
    for fold in range(4):
        held = group % 4 == fold
        oof[held] = y[held]-predict(fit_ridge(x[~held], y[~held], p['mean_ridge']), x[held])
    bias = oof.mean(axis=0)
    e = oof-bias
    ec = yc-predict(model, xc)-bias
    rank_candidates = []
    empirical_candidates = []
    for rank in p['ranks']:
        c = covariance(e, rank=rank)
        for sigma in p['noise_m']:
            rank_candidates.append((candidate_score(c, sigma, ec, gc), rank, sigma))
    for shrink in p['shrinkages']:
        c = covariance(e, shrink=shrink)
        for sigma in p['noise_m']:
            empirical_candidates.append((candidate_score(c, sigma, ec, gc), shrink, sigma))
    _, rank, sigma = min(rank_candidates)
    _, shrink, empirical_sigma = min(empirical_candidates)
    joint = covariance(e, rank=rank)
    empirical = covariance(e, shrink=shrink)
    signs = np.random.default_rng(p['seed']).choice([-1., 1.], len(joint))
    scramble = joint * signs[:, None] * signs[None, :]
    covs = {'prior': joint, 'joint_lowrank': joint, 'diagonal': np.diag(np.diag(joint)),
            'scrambled': scramble, 'empirical': empirical, 'equivalent_map': joint}
    matrices, variances, controls, selection = {}, {}, {}, {}
    max_equivalence = 0.
    for name, ids in masks().items():
        j = coords(ids)
        matrices[name], variances[name] = {}, {}
        for arm, c in covs.items():
            sig = empirical_sigma if arm == 'empirical' else sigma
            k, v = conditional(c, ids, sig)
            if arm == 'prior':
                k, v = np.zeros_like(k), np.diag(c)
            if arm == 'equivalent_map':
                km = map_gain(c, ids, sig)
                max_equivalence = max(max_equivalence, float(np.max(np.abs(k-km))))
                k = km
            matrices[name][arm], variances[name][arm] = k, v
        trials = []
        for penalty in p['conditional_ridge']:
            cm = fit_ridge(np.c_[x, e[:, j]], e, penalty)
            pred = predict(cm, np.c_[xc, ec[:, j]])
            trials.append((loss(ec-pred, gc, ids), penalty, cm))
        _, penalty, cm = min(trials, key=lambda row: (row[0], row[1]))
        scale = min(p['interpolation_scales'],
                    key=lambda v: loss(ec-v*interpolation(ec[:, j], ids), gc, ids))
        controls[name] = (cm, scale)
        selection[name] = {'masked_ridge_penalty': penalty, 'interpolation_scale': scale}
    if max_equivalence > 1e-7:
        raise AssertionError(f'Gaussian/MAP mismatch: {max_equivalence}')
    selected = {'joint_rank': rank, 'joint_noise_m': sigma, 'empirical_shrinkage': shrink,
                'empirical_noise_m': empirical_sigma, 'controls': selection,
                'rank_search': rank_candidates, 'empirical_search': empirical_candidates,
                'max_gain_equivalence_error': max_equivalence,
                'max_marginal_mismatch': float(np.max(np.abs(np.diag(joint)-np.diag(scramble))))}
    return (model, bias, matrices, variances, controls), selected


def case_predictions(a, t, name, models, p):
    model, bias, matrices, variances, controls = models
    ids = masks()[name]
    x, base, times = observation(a, t, p)
    prior = base + predict(model, x[None])[0] + bias
    # This is the only post-initialization internal-node observation.
    observed = a[times[0], np.asarray(ids)+2].reshape(-1).copy()
    innovation = observed-prior[coords(ids)]
    cm, scale = controls[name]
    predictions = {arm: prior+k@innovation for arm, k in matrices[name].items()}
    predictions['interpolation'] = prior+scale*interpolation(innovation[None], ids)[0]
    predictions['masked_ridge'] = prior+predict(cm, np.r_[x, innovation][None])[0]
    return predictions, variances[name], times


def ci(values, seed=612, count=4000):
    a = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = np.mean(a[rng.integers(len(a), size=(count, len(a)))], axis=1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def summarize(records):
    summary, paired, settings = {}, {}, {}
    for dlo in ('DLO4', 'DLO5'):
        r = [v for v in records if v['dlo'] == dlo]
        trajectories = sorted({v['trajectory'] for v in r})
        summary[dlo], paired[dlo], settings[dlo] = {}, {}, {}
        vectors = {}
        for arm in ARMS:
            vectors[arm] = {}
            for metric in ('hidden_now_rmse_mm', 'future4_rmse_mm', 'future16_rmse_mm'):
                values = [float(np.mean([v[metric] for v in r
                           if v['arm'] == arm and v['trajectory'] == tr])) for tr in trajectories]
                vectors[arm][metric] = np.array(values)
            summary[dlo][arm] = {k: float(v.mean()) for k, v in vectors[arm].items()}
        for other in ARMS:
            if other == 'joint_lowrank':
                continue
            paired[dlo][other] = {}
            for metric in ('hidden_now_rmse_mm', 'future16_rmse_mm'):
                delta = vectors['joint_lowrank'][metric]-vectors[other][metric]
                paired[dlo][other][metric] = {'mean_difference_mm': float(delta.mean()),
                    'trajectory_bootstrap_95_ci_mm': ci(delta),
                    'wins': int(np.sum(delta < -1e-8)), 'count': len(delta)}
        for name in masks():
            settings[dlo][name] = {}
            for arm in ARMS:
                selected = [v for v in r if v['mask'] == name and v['arm'] == arm]
                settings[dlo][name][arm] = {key: float(np.mean([v[key] for v in selected]))
                    for key in ('hidden_now_rmse_mm', 'future4_rmse_mm', 'future16_rmse_mm')}
    comparisons = ('prior', 'interpolation', 'empirical', 'masked_ridge')
    supported = all(paired[d][arm][metric]['trajectory_bootstrap_95_ci_mm'][1] < 0
                    for d in paired for arm in comparisons
                    for metric in ('hidden_now_rmse_mm', 'future16_rmse_mm'))
    return summary, paired, settings, supported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    save(out/'protocol.json', p)
    manifest, split = {}, {}
    for dlo in ('DLO4', 'DLO5'):
        paths = sorted((args.dataset_root/dlo/'train').glob('*.pkl'))
        if len(paths) != 56:
            raise ValueError(f'{dlo}: expected 56 training files, got {len(paths)}')
        paths.sort(key=lambda path: hashlib.sha256(
            f"{p['split_domain']}:{dlo}:{path.name}".encode()).hexdigest())
        split[dlo] = {'fit': paths[:32], 'calibration': paths[32:44], 'test': paths[44:]}
        manifest[dlo] = {stage: [{'file': str(v), 'sha256': digest(v)} for v in paths]
                         for stage, paths in split[dlo].items()}
    save(out/'manifest.json', manifest)
    predictions, truths, metadata, variances, selections = [], [], [], [], {}
    for dlo, parts in split.items():
        models, selection = fit_models(parts['fit'], parts['calibration'], p)
        selections[dlo] = selection
        save(out/f'{dlo}-selection-before-test.json', selection)
        print(json.dumps({'stage': 'source-calibration-complete', 'dlo': dlo,
                          'rank': selection['joint_rank']}), flush=True)
        for path in parts['test']:
            a = read_trajectory(path)
            for t in p['starts']:
                for name, ids in masks().items():
                    pred, var, times = case_predictions(a, t, name, models, p)
                    # Only the evaluator reads hidden/future outcomes, after predictions.
                    truth = a[times, 2:10].reshape(-1).copy()
                    for arm in ARMS:
                        predictions.append(pred[arm]); truths.append(truth)
                        variances.append(var.get(arm, np.full(72, np.nan)))
                        metadata.append({'dlo': dlo, 'trajectory': path.name,
                                         'time': t, 'mask': name, 'arm': arm})
        print(json.dumps({'stage': 'predictions-complete', 'dlo': dlo}), flush=True)
    pred, truth, var = np.array(predictions), np.array(truths), np.array(variances)
    np.savez_compressed(out/'predictions.npz', prediction=pred, variance=var)
    save(out/'prediction_index.json', metadata)
    save(out/'prediction_seal.json', {'sha256': digest(out/'predictions.npz'),
        'source_revision': os.getenv('GITHUB_SHA', 'local'),
        'protocol_sha256': digest(args.protocol), 'manifest_sha256': digest(out/'manifest.json'),
        'scope': 'metrics not yet calculated; retrospective evaluator loads complete trajectories'})
    records = []
    for i, info in enumerate(metadata):
        ids = masks()[info['mask']]
        hidden = coords(np.setdiff1d(np.arange(8), ids))
        error = pred[i]-truth[i]
        record = dict(info)
        for name, j in [('hidden_now', hidden), ('future4', np.arange(24, 48)),
                        ('future16', np.arange(48, 72))]:
            record[f'{name}_rmse_mm'] = 1000*float(np.sqrt(np.mean(error[j]**2)))
            if np.isfinite(var[i, j]).all():
                record[f'{name}_coverage90'] = float(np.mean(abs(error[j]) <= 1.644853627*np.sqrt(var[i, j])))
                record[f'{name}_interval_width_mm'] = float(3290*np.sqrt(var[i, j]).mean())
                record[f'{name}_marginal_nll'] = float(np.mean(.5*(np.log(2*np.pi*var[i, j])+error[j]**2/var[i, j])))
        records.append(record)
    with (out/'cases.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    summary, paired, settings, supported = summarize(records)
    result = {'status': 'completed-retrospective-source-only-mechanism-test',
        'hypothesis_supported_against_all_controls': supported,
        'baseline': 'source-fitted ridge plus endpoint transport/damped velocity; NOT released DEFORM hybrid',
        'data': 'real DEFORM DLO4/DLO5 training trajectories with imposed fixed missingness',
        'official_eval_files_opened': False, 'new_physical_interactions': False,
        'same_mean_covariance_ablation': True, 'source_revision': os.getenv('GITHUB_SHA', 'local'),
        'protocol_sha256': digest(args.protocol), 'prediction_sha256': digest(out/'predictions.npz'),
        'accounting': {'objects': 2, 'fit_per_object': 32, 'calibration_per_object': 12,
                       'test_per_object': 12, 'windows_per_trajectory': len(p['starts']),
                       'masks': len(masks()), 'arms': len(ARMS), 'rows': len(records)},
        'selected': selections, 'aggregate': summary, 'paired': paired, 'by_mask': settings,
        'inference_boundary': 'paired trajectory bootstrap within each of two objects; no unseen-object claim; all comparisons reported',
        'scope': 'tests spatial-temporal residual conditioning, not native simulator rerun, material inference, real-camera occlusion, or counterfactual benefit'}
    save(out/'result.json', result)
    lines = ['# Partial-observation joint-belief test', '',
             f"**All-control hypothesis supported: {supported}.**", '',
             result['baseline'], '', '| Object | Arm | Hidden now (mm) | Future +16 (mm) |',
             '|---|---|---:|---:|']
    for dlo, arm_results in summary.items():
        for arm, scores in arm_results.items():
            lines.append(f"| {dlo} | {arm} | {scores['hidden_now_rmse_mm']:.4f} | {scores['future16_rmse_mm']:.4f} |")
    lines += ['', 'Imposed missingness on real recorded trajectories. Only train/ opened.',
              'Official evaluation remains closed. Scientific negative results are successful executions.',
              'Covariance and deterministic MAP equivalence is numerical, not a superiority comparison.']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
