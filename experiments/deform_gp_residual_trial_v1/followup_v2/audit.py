#!/usr/bin/env python3
"""Fit-only covariance calibration and passive-prefix audit of frozen DLO2 GP.

Reads only the two checksum-bound development caches exported by
export_development.py. No simulator calls or raw dataset access here.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gp import GPConfig, ResidualGP, balanced_anchor_indices, matern32, matrix

SEED = 20260909
SHRINKAGE = 0.25
CUT = 50
OBS = np.arange(4, CUT, 5)
GRID = np.linspace(0, 497, 20, dtype=int)
Q_FLOOR = 1e-10  # (0.01 mm)^2 numerical floor, not a fitted noise parameter.


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def fit_fast(x, y, groups, config):
    """Same v1 mean/precision; skip its unused sandwich covariance computation."""
    config.validate()
    x, y = matrix(x, 'features'), matrix(y, 'targets')
    groups = np.asarray(groups)
    if groups.shape != (len(x),) or len(y) != len(x) or len(np.unique(groups)) < 2:
        raise ValueError('Misaligned data or fewer than two trajectory groups')
    loc, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    standard = (x-loc)/scale
    ix = (balanced_anchor_indices(groups, config.max_anchors, config.seed)
          if config.amplitude else np.empty(0, dtype=np.int64))
    anchors = standard[ix]
    chol = (np.linalg.cholesky(matern32(anchors, anchors,
             config.length_multiplier*math.sqrt(x.shape[1]))
             + config.kernel_jitter*np.eye(len(ix)))
            if config.amplitude else np.empty((0, 0)))
    model = ResidualGP(config, loc, scale, anchors, chol, np.empty((0, 0)),
                      np.empty((0, 0)), np.empty(0), np.empty((0, 0, 0)),
                      len(x), len(np.unique(groups)), ix)
    d = model.design(x)
    penalty = config.ridge*np.eye(d.shape[1]); penalty[0, 0] = 0.0
    factor = cho_factor(d.T@d+penalty, lower=True, check_finite=False)
    model.weights = cho_solve(factor, d.T@y, check_finite=False)
    inv = cho_solve(factor, np.eye(d.shape[1]), check_finite=False)
    model.precision_inverse = (inv+inv.T)/2
    model.residual_variance = np.mean((y-d@model.weights)**2, axis=0)
    return model


def fold_indices(n):
    if n != 40:
        raise ValueError('Registered fold assignment requires 40 trajectories')
    p = np.random.default_rng(SEED).permutation(n)
    return [np.sort(p[8*k:8*k+8]) for k in range(5)]


def crossfit(features, target, names, config):
    n, h, nodes, f = features.shape
    if target.shape != (n, h, nodes, 3) or (n, h, nodes) != (40, 498, 8):
        raise ValueError('Unexpected registered fit tensor shape')
    error = np.empty_like(target)
    covariance = np.empty((n, nodes, len(GRID), len(GRID)))
    folds = []
    for k, held in enumerate(fold_indices(n)):
        fit = np.setdiff1d(np.arange(n), held)
        folds.append({'fold': k, 'fit': names[fit].tolist(), 'held': names[held].tolist()})
        for j in range(nodes):
            model = fit_fast(features[fit, :, j].reshape(-1, f),
                             target[fit, :, j].reshape(-1, 3),
                             np.repeat(np.arange(len(fit)), h), config)
            for i in held:
                d = model.design(features[i, :, j])
                error[i, :, j] = target[i, :, j]-SHRINKAGE*d@model.weights
                dg = d[GRID]
                cov = SHRINKAGE**2*dg@model.precision_inverse@dg.T
                covariance[i, j] = (cov+cov.T)/2
        print('CROSSFIT', config.amplitude, k+1, '/ 5', flush=True)
    return error, covariance, folds


def calibrate_covariance(error, covariance):
    """Estimate q_c(a C/mean_diag+b I) using only cross-fit training errors."""
    q = np.maximum(np.mean(error**2, axis=(0, 1)), Q_FLOOR)
    parameters = []
    for j in range(error.shape[2]):
        norm = float(np.diagonal(covariance[:, j], axis1=-2, axis2=-1).mean())
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid covariance normalization')
        c = covariance[:, j]/norm
        lam, u = np.linalg.eigh(c)
        if lam.min() < -1e-7:
            raise ValueError('Covariance not PSD')
        lam = np.maximum(lam, 0.0)
        e = error[:, GRID, j]/np.sqrt(q[j])
        projected = np.einsum('nti,ntc->nic', u, e)
        p2 = np.sum(projected**2, axis=2)
        count = p2.size*3
        def objective(log_ab):
            a, b = np.exp(log_ab)
            v = a*lam+b
            loss = 0.5*np.sum(3*np.log(v)+p2/v)/count
            dv = 0.5*(3/v-p2/v**2)/count
            grad = np.array([np.sum(dv*a*lam), np.sum(dv*b)])
            return float(loss), grad
        opt = minimize(objective, np.log([.5, .5]), jac=True, method='L-BFGS-B',
                       bounds=[(math.log(1e-6), math.log(1e3))]*2,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if not opt.success:
            raise RuntimeError(f'Covariance calibration did not converge: {opt.message}')
        a, b = np.exp(opt.x)
        parameters.append({'a': float(a), 'b': float(b), 'normalizer': norm,
                           'q_m2': q[j].tolist(), 'optimizer_success': bool(opt.success),
                           'objective': float(opt.fun), 'iterations': int(opt.nit)})
    return parameters


def fit_bias(error):
    """Per-node deterministic shrinkage of the ten-frame mean prefix error."""
    prefix = np.mean(error[:, OBS], axis=1)
    numerator = np.sum(prefix*np.mean(error[:, CUT:], axis=1), axis=(0, 2))
    denominator = np.sum(prefix**2, axis=(0, 2))
    return np.clip(numerator/np.maximum(denominator, 1e-30), 0, 1)


def condition(cross, observed_covariance, future_diagonal, innovations):
    """No suffix targets accepted. Return mean increment and future variance."""
    if cross.ndim != 2 or innovations.shape[0] != cross.shape[1]:
        raise ValueError('Misaligned conditioning inputs')
    f = cho_factor(observed_covariance, lower=True, check_finite=False)
    gain = cho_solve(f, cross.T, check_finite=False).T
    delta = gain@innovations
    variance = future_diagonal-np.sum(gain*cross, axis=1)
    if variance.min() < -1e-7*max(1., float(np.max(future_diagonal))):
        raise ValueError('Negative conditional variance')
    return delta, np.maximum(variance, 1e-15)


def full_model_prediction(features, models, params, prefix_target):
    """Adapt fixed means using only the ten supplied local prefix targets."""
    n, h, nodes, _ = features.shape
    if prefix_target.shape != (n, len(OBS), nodes, 3):
        raise ValueError('Exactly ten prefix frames are required, not a full target trajectory')
    mean = np.empty((n, h, nodes, 3)); updated = np.empty_like(mean)
    wrong = np.empty_like(mean); variance = np.empty_like(mean)
    updated_var = np.empty_like(mean); raw_var = np.empty_like(mean)
    diagonally_unchanged = True
    for j, model in enumerate(models):
        p = params[j]; a, b, scale = p['a'], p['b'], p['normalizer']
        q = np.asarray(p['q_m2'])
        ds = [model.design(features[i, :, j]) for i in range(n)]
        means = [SHRINKAGE*d@model.weights for d in ds]
        innovation = [prefix_target[i, :, j]-means[i][OBS] for i in range(n)]
        for i, d in enumerate(ds):
            mean[i, :, j] = means[i]
            projected = SHRINKAGE**2*(d@model.precision_inverse)
            diag = np.sum(projected*d, axis=1)
            cp = projected@d[OBS].T
            cross = a*cp/scale
            obs_cov = cross[OBS]+b*np.eye(len(OBS))
            full_diag = a*diag/scale+b
            delta, cond_diag = condition(cross, obs_cov, full_diag, innovation[i])
            wrong_delta, _ = condition(cross, obs_cov, full_diag, innovation[(i+1)%n])
            unchanged, _ = condition(np.zeros_like(cross), obs_cov, full_diag, innovation[i])
            diagonally_unchanged &= bool(np.array_equal(unchanged, np.zeros_like(unchanged)))
            updated[i, :, j] = means[i]+delta
            wrong[i, :, j] = means[i]+wrong_delta
            variance[i, :, j] = full_diag[:, None]*q
            updated_var[i, :, j] = cond_diag[:, None]*q
            # Diagnostic: uncalibrated noise + shrunken mean-uncertainty variance.
            raw_var[i, :, j] = (1+diag[:, None])*model.residual_variance
    if not diagonally_unchanged:
        raise AssertionError('Zero cross-time covariance changed the mean')
    return mean, updated, wrong, variance, updated_var, raw_var


def world_prediction(baseline, local, frames):
    out = baseline.copy()
    out[:, :, 2:-2] += np.einsum('ntvj,nij->ntvi', local, frames)
    if not np.array_equal(out[:, :, [0, 1, -2, -1]], baseline[:, :, [0, 1, -2, -1]]):
        raise AssertionError('Clamped coordinates changed')
    return out


def probabilistic_scores(error, variance):
    variance = np.broadcast_to(variance, error.shape)
    if not np.isfinite(variance).all() or np.min(variance) <= 0:
        raise ValueError('Invalid predictive variance')
    sd = np.sqrt(variance); z = error/sd
    nll = .5*(np.log(2*np.pi*variance)+z**2)
    crps = sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    axes = tuple(range(1, error.ndim))
    return {'gaussian_nll_nats_m': np.mean(nll, axis=axes),
            'crps_mm': 1000*np.mean(crps, axis=axes),
            'coverage95_percent': 100*np.mean(np.abs(z)<=1.959963984540054, axis=axes),
            'interval95_width_mm': 1000*np.mean(2*1.959963984540054*sd, axis=axes),
            'normalized_squared_error': np.mean(z*z, axis=axes)}


def bootstrap(delta):
    delta = np.asarray(delta)
    r = np.random.default_rng(SEED).choice(delta, size=(5000, len(delta)), replace=True).mean(1)
    return {'mean_difference': float(delta.mean()), 'wins': int(np.sum(delta<0)),
            'n': len(delta), 'descriptive95_interval': np.quantile(r, [.025, .975]).tolist()}


def read_cache(root, manifest, label):
    entries = [x for x in manifest['files'] if x['label']==label]
    if len(entries)!=1 or entries[0]['file'] != f'{label}_queries.npz':
        raise ValueError('Unexpected cache mapping')
    path = root/entries[0]['file']
    if sha256(path) != entries[0]['sha256']:
        raise ValueError('Cache hash mismatch')
    with np.load(path, allow_pickle=False) as f:
        value = {k: f[k] for k in f.files}
    if value['names'].tolist() != entries[0]['names']:
        raise ValueError('Cache identities mismatch')
    for k, a in value.items():
        if a.dtype.kind not in 'US' and not np.isfinite(a).all():
            raise ValueError(f'Nonfinite {label} {k}')
    return value


def main(args):
    start = time.perf_counter(); root = args.cache.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root/'cache_manifest.json').read_text())
    if (manifest['contract'] != 'deform-gp-v2-development-query-cache-v1'
        or manifest['official_eval_read'] is not False or manifest['source_test_opened'] is not False):
        raise ValueError('Unapproved cache provenance')
    fit = read_cache(root, manifest, 'fit')
    xf, yf = fit['features'], fit['local_target']
    # Exact duplicate causal inputs may never cross residual-model folds.
    keys = [hashlib.sha256(b''.join(np.ascontiguousarray(fit[k][i]).tobytes()
                for k in ['initial','action','baseline'])).hexdigest() for i in range(len(xf))]
    if len(set(keys)) != 40:
        raise ValueError('Duplicate fit queries require a separately declared grouped fold assignment')
    configurations = {'ridge': GPConfig(amplitude=0), 'gp': GPConfig()}
    calibrated = {}; fitted = {}; oof = {}; saved_oof = {}
    for name, config in configurations.items():
        error, covariance, folds = crossfit(xf, yf, fit['names'], config)
        params = calibrate_covariance(error, covariance)
        bias = fit_bias(error)
        calibrated[name] = {'config': asdict(config), 'covariance': params,
                            'bias_coefficient': bias.tolist(), 'folds': folds}
        oof[name] = error
        saved_oof[name+'_error'] = error
        saved_oof[name+'_grid_covariance'] = covariance
        fitted[name] = [fit_fast(xf[:, :, j].reshape(-1, xf.shape[-1]),
                       yf[:, :, j].reshape(-1, 3), np.repeat(np.arange(40), 498), config)
                        for j in range(8)]
    write_json(out/'fit_only_calibration.json', calibrated)
    np.savez_compressed(out/'crossfit_diagnostics.npz', **saved_oof)
    write_json(out/'fit_seal.json', {'calibration_sha256': sha256(out/'fit_only_calibration.json'),
        'validation_targets_used_for_parameter_selection': False,
        'audit_sha256': sha256(Path(__file__)), 'seed': SEED, 'observation_indices': OBS.tolist(),
        'suffix_start_index': CUT, 'covariance_grid': GRID.tolist(),
        'cache_manifest_sha256': sha256(root/'cache_manifest.json')})
    # Validation cache is first loaded after all parameter estimation is complete.
    val = read_cache(root, manifest, 'validation')
    if len(val['names']) != 8 or set(val['names']) & set(fit['names']):
        raise ValueError('Invalid validation identities')
    pred_local = {}; var_local = {}; controls = {}
    for name, models in fitted.items():
        params = calibrated[name]['covariance']
        mean, updated, wrong, variance, updated_var, raw_var = full_model_prediction(
            val['features'], models, params, val['local_target'][:, OBS].copy())
        pred_local[name+'_open'] = mean
        pred_local[name+'_conditioned'] = updated
        var_local[name+'_constant'] = np.asarray([p['q_m2'] for p in params])[None, None]
        var_local[name+'_calibrated'] = variance
        var_local[name+'_conditioned'] = updated_var
        var_local[name+'_raw'] = raw_var
        prefix_error = val['local_target'][:, OBS]-mean[:, OBS]
        bias = prefix_error.mean(1)[:, None]
        pred_local[name+'_prefix_bias'] = mean+bias
        pred_local[name+'_fit_shrunk_bias'] = mean+bias*np.asarray(calibrated[name]['bias_coefficient'])[None, None, :, None]
        pred_local[name+'_wrong_prefix_control'] = wrong
        # The prediction function only accepts ten frames; suffix data is not an argument.
        corrupt = val['local_target'].copy(); corrupt[:, CUT:] = 1e6
        again = full_model_prediction(val['features'], models, params, corrupt[:, OBS].copy())
        controls[name+'_suffix_invariance'] = bool(np.array_equal(again[1], updated))
        if not controls[name+'_suffix_invariance']:
            raise AssertionError('Unobserved suffix changed a prediction')
    predictions = {'hybrid': val['baseline']}
    predictions.update({k: world_prediction(val['baseline'], v, val['frames']) for k, v in pred_local.items()})
    # Full-fit means must reproduce the already retained v1 predictions exactly.
    with np.load(root/'validation_predictions.npz', allow_pickle=False) as ref:
        parity = {k: float(np.max(np.abs(predictions[k+'_open']-ref[k]))) for k in ['ridge', 'gp']}
    if max(parity.values()) > 1e-9:
        raise AssertionError(f'V1 mean reproduction failed: {parity}')
    np.savez_compressed(out/'sealed_predictions.npz', names=val['names'], frames=val['frames'],
                        **predictions, **{'local_'+k:v for k,v in pred_local.items()},
                        **{'variance_'+k:v for k,v in var_local.items()})
    write_json(out/'prediction_seal.json', {'sha256': sha256(out/'sealed_predictions.npz'),
        'suffix_targets_used_in_prediction': False, 'parity_m': parity, 'controls': controls})
    # Scoring starts only after the predictions have been saved and hashed.
    errors = {k: 1000*np.mean(np.abs(p[:, CUT:]-val['target'][:, CUT:]), axis=(1, 2, 3))
              for k, p in predictions.items()}
    rows = []
    for i, name in enumerate(val['names']):
        for method, values in errors.items():
            rows.append({'trajectory': name, 'method': method, 'suffix_all_node_l1_mm': float(values[i]),
                'suffix_free_node_l1_mm': float(1000*np.mean(np.abs(
                    predictions[method][i, CUT:, 2:-2]-val['target'][i, CUT:, 2:-2])))})
    with (out/'trajectory_metrics.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    probabilistic = {}; probrows = []
    for name in ['ridge', 'gp']:
        for kind in ['constant', 'calibrated', 'raw', 'conditioned']:
            method = name+'_'+kind
            m = pred_local[name+('_conditioned' if kind=='conditioned' else '_open')]
            e = val['local_target'][:, CUT:]-m[:, CUT:]
            v = var_local[method]
            if v.shape[1] != 1:
                v = v[:, CUT:]
            values = probabilistic_scores(e, v)
            probabilistic[method] = {k: float(a.mean()) for k,a in values.items()}
            for i, t in enumerate(val['names']):
                probrows.append({'trajectory': t, 'method': method, **{k: float(a[i]) for k,a in values.items()}})
    with (out/'probabilistic_metrics.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(probrows[0])); w.writeheader(); w.writerows(probrows)
    comparisons = {f'gp_conditioned_minus_{m}': bootstrap(errors['gp_conditioned']-errors[m])
                  for m in ['gp_open','gp_prefix_bias','gp_fit_shrunk_bias','ridge_conditioned','ridge_open']}
    nll = {}
    for name in ['gp', 'ridge']:
        a = np.asarray([r['gaussian_nll_nats_m'] for r in probrows if r['method']==name+'_calibrated'])
        b = np.asarray([r['gaussian_nll_nats_m'] for r in probrows if r['method']==name+'_constant'])
        nll[name+'_calibrated_minus_same_mean_constant'] = bootstrap(a-b)
    full_errors = {k: float(1000*np.mean(np.abs(p-val['target'])))
                   for k,p in predictions.items() if k in ['hybrid','ridge_open','gp_open']}
    report = {'contract':'deform-gp-uncertainty-prefix-audit-v2', 'status':'completed',
        'historical_development_only':True, 'source_test_opened':False, 'official_eval_read':False,
        'new_robot_experiments':False, 'simulator_retrained':False,
        'validation_used_for_fitting_or_selection':False, 'seed':SEED,
        'fit_count':40, 'validation_count':8, 'physical_object_count':1,
        'observed_prefix_indices_zero_based':OBS.tolist(), 'prefix_frames_observed':len(OBS),
        'suffix_steps':498-CUT, 'suffix_start_forecast_step_one_based':CUT+1,
        'v1_full_forecast_reproduction_l1_mm':full_errors, 'mean_parity_max_abs_m':parity,
        'suffix_mean_l1_mm':{k:float(v.mean()) for k,v in errors.items()},
        'prefix_paired_comparisons_mm':comparisons, 'probabilistic_scores_suffix_free_local_coordinates':probabilistic,
        'same_mean_nll_comparisons':nll, 'controls':controls,
        'diagonal_covariance_mean_unchanged':True, 'clamped_coordinates_exact':True,
        'prediction_sha256':sha256(out/'sealed_predictions.npz'),
        'calibration_sha256':sha256(out/'fit_only_calibration.json'),
        'cache_manifest_sha256':sha256(root/'cache_manifest.json'),
        'audit_sha256':sha256(Path(__file__)), 'elapsed_seconds':time.perf_counter()-start,
        'interpretation':'Empirically calibrated covariance, not a fully Bayesian posterior or new blind test. Cross-fitting holds out only the residual learner; the frozen simulator historically used the fit panel.'}
    write_json(out/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
