#!/usr/bin/env python3
"""Retrospective, fixed-mask conditioning of cached real DEFORM forecasts.

This is a predictive-discrepancy experiment, not a latent physical-state update.
Only source trajectories fit means/covariances and select hyperparameters.
Target prediction accepts current visible coordinates, never hidden outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

HORIZONS = (0, 1, 5, 10, 25)
CUTS = tuple(range(25, 451, 25))
BUDGETS = (1, 2, 4)
D = 24
REGS = (0.0001, 0.01, 0.1, 1.0)
FAMILIES = ('empirical', 'ridge', 'rod_modes', 'rbf', 'translation', 'interpolation')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def masks(budget):
    return [np.arange(start, start + budget) for start in range(9 - budget)]


def coordinates(nodes):
    return (np.asarray(nodes)[:, None] * 3 + np.arange(3)[None, :]).ravel()


def hidden_coordinates(nodes, horizon_index):
    hidden = np.setdiff1d(np.arange(8), nodes)
    return horizon_index * D + coordinates(hidden)


def covariance_model(samples, family, rank=2, length=2.0):
    x = np.asarray(samples, dtype=np.float64).reshape(-1, D * len(HORIZONS))
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite source samples')
    mean = x.mean(axis=0)
    centered = x - mean
    cov = centered.T @ centered / len(x)
    variance = np.maximum(np.diag(cov), 1e-12)
    cov[np.diag_indices_from(cov)] = variance
    if family == 'rod_modes':
        modes = np.sin(np.pi * np.arange(1, 9)[:, None] * np.arange(1, rank + 1)[None, :] / 9)
        q, _ = np.linalg.qr(modes)
        projection = np.kron(np.eye(len(HORIZONS)), np.kron(q @ q.T, np.eye(3)))
        projected = projection @ cov @ projection.T
        scale = np.sqrt(variance / np.maximum(np.diag(projected), 1e-18))
        cov = projected * np.outer(scale, scale)
        # Explicit 5% independent discrepancy, not hidden numerical jitter.
        cov = 0.95 * cov + 0.05 * np.diag(variance)
    elif family == 'rbf':
        node = np.exp(-0.5 * ((np.arange(8)[:, None] - np.arange(8)[None, :]) / length) ** 2)
        # Persistent spatial discrepancy with the same source marginal variances.
        corr = np.kron(np.ones((len(HORIZONS), len(HORIZONS))), np.kron(node, np.eye(3)))
        cov = corr * np.sqrt(np.outer(variance, variance))
        cov = 0.95 * cov + 0.05 * np.diag(variance)
    return {'mean': mean, 'cov': cov, 'samples': centered,
            'noise_scale': float(np.mean(variance[:D]))}


def gains(model, nodes, param, scramble=False):
    obs = coordinates(nodes)
    cov = model['cov']
    if scramble:
        signs = np.tile(np.repeat(np.array([1, -1, 1, -1, -1, 1, -1, 1]), 3), len(HORIZONS))
        cov = cov * np.outer(signs, signs)
    family = param['family']
    noise = param.get('reg', 0.01) * model['noise_scale']
    if family == 'ridge':
        x = model['samples'][:, obs]
        w = np.linalg.solve(x.T @ x / len(x) + noise * np.eye(len(obs)),
                            x.T @ model['samples'] / len(x))
        return w.T, None
    if family == 'translation':
        g = np.zeros((D * len(HORIZONS), len(obs)))
        for node in range(8):
            for h in range(len(HORIZONS)):
                for axis in range(3):
                    g[h * D + node * 3 + axis, axis::3] = 1 / len(nodes)
        return g, None
    if family == 'interpolation':
        # Fixed zero-discrepancy boundary anchors at neighbouring clamped nodes.
        xp = np.r_[-1, nodes, 8]
        weights = np.stack([np.interp(np.arange(8), xp,
                            np.r_[0, np.eye(len(nodes))[:, j], 0])
                            for j in range(len(nodes))], axis=1)
        return np.tile(np.kron(weights, np.eye(3)), (len(HORIZONS), 1)), None
    koo = cov[np.ix_(obs, obs)] + noise * np.eye(len(obs))
    gain = np.linalg.solve(koo, cov[obs, :]).T
    posterior_diag = np.maximum(np.diag(cov) - np.sum(gain * cov[:, obs], axis=1), 0)
    return gain, posterior_diag


def grid(family):
    if family == 'rod_modes':
        return [dict(family=family, rank=k, reg=r) for k in (2, 4) for r in REGS]
    if family == 'rbf':
        return [dict(family=family, length=l, reg=r) for l in (1.0, 2.0, 4.0) for r in REGS]
    if family in ('empirical', 'ridge'):
        return [dict(family=family, reg=r) for r in REGS]
    return [dict(family=family)]


def fit(samples, param):
    return covariance_model(samples, param['family'], param.get('rank', 2), param.get('length', 2))


def select_source(source):
    """LOO across complete source trajectories; never split adjacent windows."""
    selected, records = {}, {}
    for budget in BUDGETS:
        selected[budget], records[budget] = {}, {}
        for family in FAMILIES:
            options = []
            for param in grid(family):
                folds = []
                for held in range(len(source)):
                    model = fit(np.delete(source, held, axis=0), param)
                    val = source[held]
                    losses = []
                    for nodes in masks(budget):
                        gain, _ = gains(model, nodes, param)
                        estimate = model['mean'] + (val[:, coordinates(nodes)] - model['mean'][coordinates(nodes)]) @ gain.T
                        # Equal current-completion and 25-frame forecast weight.
                        idx = np.r_[hidden_coordinates(nodes, 0), hidden_coordinates(nodes, 4)]
                        losses.append(float(np.mean((estimate[:, idx] - val[:, idx]) ** 2)))
                    folds.append(float(np.mean(losses)))
                options.append({'parameters': param, 'cv_mse_m2': float(np.mean(folds)), 'fold_mse_m2': folds})
            best = min(options, key=lambda item: item['cv_mse_m2'])
            selected[budget][family] = best['parameters']
            records[budget][family] = {'selected': best, 'options': options}
            print('SOURCE_SELECTION', budget, family, json.dumps(best), flush=True)
    return selected, records


def read_panel(parent_root, dataset_root, dlo, stage, pins):
    folder = parent_root / (dlo.lower() + '-' + stage)
    archive = folder / ('source_predictions.npz' if stage == 'source' else 'target_predictions.npz')
    manifest_path = folder / ('source_manifest.json' if stage == 'source' else 'eval_manifest.json')
    if digest(archive) != pins['predictions'] or digest(manifest_path) != pins['manifest']:
        raise ValueError('Cached parent artifact identity changed: ' + str(folder))
    manifest = json.loads(manifest_path.read_text())
    with np.load(archive, allow_pickle=False) as z:
        names = list(map(str, z['names']))
        base = np.asarray(z['candidate'], dtype=np.float64)
        hybrid = np.asarray(z['physical'], dtype=np.float64)
    expected_names = manifest['partitions']['source_test'] if stage == 'source' else manifest['ordered_names']
    if names != expected_names or len(names) != (8 if stage == 'source' else 14):
        raise ValueError('Forecast identity order differs')
    if base.shape != (len(names), 498, 12, 3) or hybrid.shape != base.shape or not np.isfinite(base).all() or not np.isfinite(hybrid).all():
        raise ValueError('Invalid forecast shape or nonfinite forecast')
    truth, identities = [], []
    for name in names:
        if Path(name).name != name:
            raise ValueError('Not a basename')
        path = dataset_root / dlo / ('train' if stage == 'source' else 'eval') / name
        identity = manifest['trajectories'][name]
        if path.stat().st_size != identity['size_bytes'] or digest(path) != identity['sha256']:
            raise ValueError('Recorded data identity changed: ' + name)
        # Trusted, checksum-bound official pickle only; never arbitrary uploads.
        with path.open('rb') as f:
            array = np.asarray(pickle.load(f), dtype=np.float32)
        if array.shape != (500, 3, 12) or not np.isfinite(array).all():
            raise ValueError('Invalid recorded trajectory')
        array = array.transpose(0, 2, 1).copy()
        array[:, :, 2] = np.clip(array[:, :, 2], 0.002001, 10000.0)
        truth.append(array[2:].astype(np.float64))
        identities.append({'name': name, 'sha256': identity['sha256']})
    truth = np.stack(truth)
    parity_l1 = float(np.mean(np.abs(truth - hybrid)))
    if 'hybrid_l1_m' in pins and not np.isclose(parity_l1, pins['hybrid_l1_m'], atol=1e-8, rtol=0):
        raise ValueError(f'Original operator/loader parity failed: {parity_l1}')
    residual = truth[:, :, 2:10] - base[:, :, 2:10]
    samples = np.stack([residual[:, np.asarray(CUTS) + h] for h in HORIZONS], axis=2)
    return names, base, hybrid, truth, samples.reshape(len(names), len(CUTS), -1), identities


def predict_readout(base_window, visible_current, nodes, model, param, scramble=False):
    """No hidden state or future outcome is accepted by this prediction API."""
    gain, posterior_diag = gains(model, nodes, param, scramble=scramble)
    visible = np.asarray(visible_current, dtype=np.float64).reshape(-1)
    innovation = visible - base_window[0, nodes].ravel() - model['mean'][coordinates(nodes)]
    correction = model['mean'] + gain @ innovation
    return base_window + correction.reshape(len(HORIZONS), 8, 3), posterior_diag


def evaluate_panel(dlo, names, base, hybrid, truth, source, selected, selection):
    models = {(b, f): fit(source, selected[b][f]) for b in BUDGETS for f in FAMILIES}
    output, covariance_diagnostics, seals = [], [], []
    for budget in BUDGETS:
        best_conventional = min(('ridge', 'rbf', 'translation', 'interpolation'),
                                key=lambda f: selection[budget][f]['selected']['cv_mse_m2'])
        for k, name in enumerate(names):
            accum = {}
            for t in CUTS:
                times = np.asarray(HORIZONS) + t
                bw = base[k, times, 2:10].copy()
                hw = hybrid[k, times, 2:10].copy()
                for nodes in masks(budget):
                    visible = truth[k, t, 2 + nodes].copy()
                    model = models[budget, 'empirical']
                    preds = {'hybrid': hw, 'frozen_candidate': bw,
                             'mean_only': bw + model['mean'].reshape(len(HORIZONS), 8, 3)}
                    variances = {}
                    for family in FAMILIES:
                        estimate, variance = predict_readout(bw, visible, nodes, models[budget, family], selected[budget][family])
                        preds[family] = estimate
                        if variance is not None:
                            variances[family] = variance.reshape(len(HORIZONS), 8, 3)
                    preds['rod_scrambled'], var = predict_readout(bw, visible, nodes, models[budget, 'rod_modes'], selected[budget]['rod_modes'], True)
                    preds['diagonal'] = preds['mean_only'].copy()
                    preds['source_selected_conventional'] = preds[best_conventional]
                    # Hash every prediction before retrieving hidden scoring values.
                    h = hashlib.sha256()
                    for arm in sorted(preds):
                        h.update(arm.encode()); h.update(np.ascontiguousarray(preds[arm]).tobytes())
                    seals.append({'dlo': dlo, 'name': name, 'cut': t + 2, 'budget': budget,
                                  'nodes': (nodes + 2).tolist(), 'sha256': h.hexdigest()})
                    hidden = np.setdiff1d(np.arange(8), nodes)
                    target = truth[k, times, 2:10][:, hidden]
                    for arm, pred in preds.items():
                        errors = pred[:, hidden] - target
                        for j, horizon in enumerate(HORIZONS):
                            key = (arm, horizon)
                            accum.setdefault(key, []).append(float(np.mean(errors[j] ** 2)))
                    for arm, var in variances.items():
                        errors = preds[arm][:, hidden] - target
                        for j in (0, 4):
                            sigma = np.sqrt(np.maximum(var[j, hidden], 1e-18))
                            covariance_diagnostics.append({'dlo': dlo, 'budget': budget, 'arm': arm,
                              'horizon': HORIZONS[j], 'coverage90': float(np.mean(np.abs(errors[j]) <= 1.644853626951472 * sigma))})
            for (arm, horizon), losses in accum.items():
                output.append({'dlo': dlo, 'trajectory': name, 'budget': budget, 'horizon': horizon,
                               'arm': arm, 'coordinate_rmse_mm': 1000 * float(np.sqrt(np.mean(losses))),
                               'window_mask_count': len(losses), 'source_selected_conventional': best_conventional})
    return output, covariance_diagnostics, seals


def summarize(rows, reps=10000):
    arms = sorted(set(r['arm'] for r in rows))
    summary, contrasts = [], []
    rng = np.random.default_rng(20260906)
    for b in BUDGETS:
        for h in HORIZONS:
            vectors = {}
            for arm in arms:
                per_dlo = {}
                for dlo in ('DLO4', 'DLO5'):
                    values = [r['coordinate_rmse_mm'] for r in rows if (r['budget'], r['horizon'], r['arm'], r['dlo']) == (b, h, arm, dlo)]
                    if len(values) != 14:
                        raise ValueError('Incomplete target cohort')
                    per_dlo[dlo] = np.array(values)
                vectors[arm] = per_dlo
                summary.append({'budget': b, 'horizon': h, 'arm': arm,
                                'DLO4': float(per_dlo['DLO4'].mean()), 'DLO5': float(per_dlo['DLO5'].mean()),
                                'equal_dlo_rmse_mm': float(np.mean([v.mean() for v in per_dlo.values()]))})
            for candidate in ('rod_modes', 'empirical'):
                for baseline in ('frozen_candidate', 'mean_only', 'interpolation', 'rbf', 'ridge', 'rod_scrambled', 'source_selected_conventional'):
                    delta = {d: vectors[candidate][d] - vectors[baseline][d] for d in ('DLO4', 'DLO5')}
                    # Conditional-on-these-two-objects uncertainty, not object-population inference.
                    boot = sum(v[rng.integers(0, len(v), size=(reps, len(v)))].mean(axis=1) for v in delta.values()) / 2
                    base_mean = np.mean([v.mean() for v in vectors[baseline].values()])
                    difference = float(np.mean([v.mean() for v in delta.values()]))
                    contrasts.append({'budget': b, 'horizon': h, 'candidate': candidate, 'baseline': baseline,
                      'difference_mm': difference, 'relative_improvement_pct': float(-100 * difference / base_mean),
                      'conditional_trajectory_bootstrap95_mm': np.quantile(boot, [0.025, 0.975]).tolist(),
                      'per_dlo_difference_mm': {d: float(v.mean()) for d, v in delta.items()},
                      'wins': sum(int(np.sum(v < -1e-9)) for v in delta.values()),
                      'ties': sum(int(np.sum(np.abs(v) <= 1e-9)) for v in delta.values()), 'count': 28})
    return summary, contrasts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--parent-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        request = json.loads(args.request.read_text())
        protocol_path = Path(__file__).with_name('protocol.json')
        protocol = json.loads(protocol_path.read_text())
        if request['mode'] != 'evaluate' or digest(protocol_path) != request['protocol_sha256']:
            raise ValueError('Request/protocol binding failed')
        if protocol['horizons'] != list(HORIZONS) or protocol['cutoffs_prediction_index'] != list(CUTS) or protocol['budgets'] != list(BUDGETS):
            raise ValueError('Frozen implementation schedule differs')
        sources, selections, records, source_ids = {}, {}, {}, {}
        for dlo in ('DLO4', 'DLO5'):
            panel = read_panel(args.parent_root, args.dataset_root, dlo, 'source', protocol['pins'][dlo]['source'])
            sources[dlo], source_ids[dlo] = panel[4], panel[5]
            selections[dlo], records[dlo] = select_source(sources[dlo])
        # Both objects' choices frozen before any evaluation data are loaded.
        write_json(args.output / 'source_selection.json', records)
        write_json(args.output / 'model_seal.json', {'selection_sha256': digest(args.output / 'source_selection.json'),
          'protocol_sha256': digest(protocol_path), 'source_identities': source_ids, 'target_loaded': False,
          'source_sha256': digest(Path(__file__)), 'git_sha': os.environ.get('GITHUB_SHA', 'local')})
        rows, diagnostics, seals, target_ids = [], [], [], {}
        for dlo in ('DLO4', 'DLO5'):
            names, base, hybrid, truth, _, identities = read_panel(args.parent_root, args.dataset_root, dlo, 'target', protocol['pins'][dlo]['target'])
            target_ids[dlo] = identities
            result = evaluate_panel(dlo, names, base, hybrid, truth, sources[dlo], selections[dlo], records[dlo])
            rows.extend(result[0]); diagnostics.extend(result[1]); seals.extend(result[2])
        with (args.output / 'per_trajectory.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(args.output / 'prediction_seals.json', seals)
        summary, contrasts = summarize(rows)
        primary = [c for c in contrasts if c['budget'] == 2 and c['horizon'] == 25]
        def won(baseline):
            c = next(c for c in primary if c['candidate'] == 'rod_modes' and c['baseline'] == baseline)
            return c['relative_improvement_pct'] >= 1 and c['conditional_trajectory_bootstrap95_mm'][1] < 0 and all(v < 0 for v in c['per_dlo_difference_mm'].values())
        result = {'contract': protocol['contract'], 'status': 'complete', 'retrospective': True,
          'claim_boundary': protocol['claim_boundary'], 'new_physical_data': False, 'active_sensing': False,
          'physical_backend_retrained': False, 'objects': 2, 'source_trajectories': 16, 'test_trajectories': 28,
          'primary': {'budget': 2, 'horizon': 25, 'structured_vs_no_conditioning_pass': won('mean_only'),
                      'structured_vs_source_selected_conventional_pass': won('source_selected_conventional')},
          'summary': summary, 'contrasts': contrasts, 'target_identities': target_ids,
          'protocol_sha256': digest(protocol_path), 'model_seal_sha256': digest(args.output / 'model_seal.json'),
          'prediction_seals_sha256': digest(args.output / 'prediction_seals.json'),
          'per_trajectory_sha256': digest(args.output / 'per_trajectory.csv'),
          'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                      'seconds': time.monotonic()-started, 'runner': os.environ.get('RUNNER_NAME'),
                      'git_sha': os.environ.get('GITHUB_SHA'), 'run_id': os.environ.get('GITHUB_RUN_ID')}}
        write_json(args.output / 'result.json', result)
        lines = ['# Partial-observation covariance: real DEFORM replay', '', protocol['claim_boundary'], '',
          'Primary: two of eight free nodes observed; hidden-node forecast at +25 frames.',
          'Errors: mean per-trajectory coordinate RMSE in mm; equal weight to each DLO.', '',
          '| Arm | DLO4 | DLO5 | Equal-DLO |', '|---|---:|---:|---:|']
        for r in summary:
            if r['budget']==2 and r['horizon']==25:
                lines.append(f"| {r['arm']} | {r['DLO4']:.4f} | {r['DLO5']:.4f} | {r['equal_dlo_rmse_mm']:.4f} |")
        lines.extend(['', '## Primary contrasts', '', '| Candidate | Comparator | Gain % | Difference mm [95% interval] | Wins |', '|---|---|---:|---|---:|'])
        for c in primary:
            lo,hi=c['conditional_trajectory_bootstrap95_mm']
            lines.append(f"| {c['candidate']} | {c['baseline']} | {c['relative_improvement_pct']:.3f} | {c['difference_mm']:.4f} [{lo:.4f}, {hi:.4f}] | {c['wins']}/28 |")
        lines.extend(['', 'The intervals resample complete trajectories within the two DLOs; they do not establish arbitrary-object generalization.',
          'Empirical Gaussian conditioning and matched direct ridge are algebraically equivalent; parity is intentional.',
          'The rod-mode covariance is a new source-fitted predictive discrepancy model, not a calibrated simulator parameter posterior.',
          'Recorded internal points are artificially masked. Initial full-state and known clamp inputs are inherited identically from cached forecasts.'])
        report='\n'.join(lines)+'\n'
        (args.output / 'report.md').write_text(report)
        print(report, flush=True)
        print('PRIMARY_JSON', json.dumps({'primary':result['primary'], 'contrasts':primary, 'summary':[r for r in summary if r['budget']==2 and r['horizon'] in (0,25)], 'runtime':result['runtime'], 'result_sha256':digest(args.output/'result.json')},sort_keys=True), flush=True)
    except Exception as e:
        write_json(args.output / 'failure.json', {'type': type(e).__name__, 'message': str(e), 'git_sha': os.environ.get('GITHUB_SHA')})
        raise


if __name__ == '__main__':
    main()
