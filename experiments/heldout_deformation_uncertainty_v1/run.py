"""Same-mean, source-only held-out deformation-query experiment.

The joint arm is a new covariance-only empirical-Bayes inverse-Wishart layer
around the frozen source twin, not a claim about its original stored covariance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HORIZONS = (49, 99, 249, 497)
METHODS = ('joint_student', 'diagonal_student', 'trajectory_bootstrap',
           'symmetric_bootstrap', 'joint_gaussian')
PRIMARY_CONTROLS = ('diagonal_student', 'trajectory_bootstrap')
SCALES = (0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
THRESHOLDS = np.asarray((0.005, 0.010, 0.020))
PRIOR_WEIGHT = 2.0
FLOOR = 1e-8
QUANTILES = (np.arange(513) + 0.5) / 513
BOOTSTRAP_REPLICATES = 10000


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def query_bank() -> tuple[np.ndarray, list[dict]]:
    weights, metadata = [], []
    def add(family: str, name: str, w: np.ndarray) -> None:
        weights.append(w.reshape(-1))
        metadata.append({'family': family, 'name': name})
    for t in range(4):
        for axis in range(3):
            w = np.zeros((4, 8, 3)); w[t, :, axis] = 1 / 8
            add('centroid', f'centroid_h{HORIZONS[t]+1}_axis{axis}', w)
            w = np.zeros((4, 8, 3)); w[t, :4, axis] = 1 / 4; w[t, 4:, axis] = -1 / 4
            add('relative', f'front_minus_back_h{HORIZONS[t]+1}_axis{axis}', w)
            for node in range(1, 7):
                w = np.zeros((4, 8, 3)); w[t, node-1:node+2, axis] = (1, -2, 1)
                add('bending', f'second_difference_h{HORIZONS[t]+1}_node{node+2}_axis{axis}', w)
    for axis in range(3):
        w = np.zeros((4, 8, 3)); w[0, :, axis] = -1 / 8; w[-1, :, axis] = 1 / 8
        add('temporal', f'late_minus_early_centroid_axis{axis}', w)
    return np.stack(weights), metadata


def extract(positions: np.ndarray, initial: np.ndarray) -> np.ndarray:
    positions, initial = np.asarray(positions), np.asarray(initial)
    assert positions.shape[1:] == (498, 12, 3)
    assert initial.shape == (len(positions), 12, 3)
    return (positions[:, HORIZONS, 2:10] - initial[:, None, 2:10]).reshape(len(positions), -1)


def model(fit_errors: np.ndarray, calibration_errors: np.ndarray) -> dict:
    """Known-zero residual mean; IW prior covariance learned on fit only.

    nu0=d+1+2 and Psi0=2*D. Predictive t degrees of freedom are n+4,
    scale is (Psi0+sum e e^T)/(n+4), covariance denominator n+2.
    Each calibration row is a COMPLETE trajectory, not a frame sample.
    """
    f, e = np.asarray(fit_errors), np.asarray(calibration_errors)
    assert f.ndim == e.ndim == 2 and f.shape[1] == e.shape[1]
    prior_diagonal = np.maximum(np.mean(f*f, axis=0), FLOOR)
    covariance = (e.T @ e + PRIOR_WEIGHT * np.diag(prior_diagonal)) / (len(e) + PRIOR_WEIGHT)
    return {'covariance': covariance, 'df': len(e) + PRIOR_WEIGHT + 2,
            'centered': e - e.mean(axis=0), 'symmetric': np.concatenate((e, -e), axis=0)}


def distribution(m: dict, method: str, q: np.ndarray, scale: float) -> dict:
    if method in ('trajectory_bootstrap', 'symmetric_bootstrap'):
        residuals = m['centered'] if method == 'trajectory_bootstrap' else m['symmetric']
        samples = (residuals @ q.T).T * scale
        assert np.max(np.abs(samples.mean(axis=1))) < 1e-10
        return {'kind': 'empirical', 'samples': samples}
    cov = m['covariance']
    variance = np.sum(q*q * np.diag(cov), axis=1) if method == 'diagonal_student' else np.einsum('qd,de,qe->q', q, cov, q)
    variance = np.maximum(variance, 1e-16) * scale**2
    if method == 'joint_gaussian':
        sd = np.sqrt(variance)
        return {'kind': 'normal', 'sd': sd, 'samples': sd[:, None] * stats.norm.ppf(QUANTILES)}
    df = m['df']
    sd = np.sqrt(variance * (df-2) / df)
    return {'kind': 'student', 'sd': sd, 'df': df, 'samples': sd[:, None] * stats.t.ppf(QUANTILES, df)}


def cdf(d: dict, x: np.ndarray, *, left: bool = False) -> np.ndarray:
    if d['kind'] == 'empirical':
        samples = d['samples'][:, :, None]
        return ((samples < x[:, None, :]) if left else (samples <= x[:, None, :])).mean(axis=1)
    if d['kind'] == 'normal':
        return stats.norm.cdf(x / d['sd'][:, None])
    return stats.t.cdf(x / d['sd'][:, None], d['df'])


def crps(d: dict, error: np.ndarray) -> np.ndarray:
    x = np.sort(d['samples'], axis=1)
    n = x.shape[1]
    pair_half = (x * (2*np.arange(n) - n + 1)[None, :]).sum(axis=1) / n**2
    return np.mean(np.abs(x - error[:, None]), axis=1) - pair_half


def intervals(d: dict) -> tuple[np.ndarray, np.ndarray]:
    if d['kind'] == 'empirical':
        return tuple(np.quantile(d['samples'], (0.05, 0.95), axis=1))
    z = stats.norm.ppf(0.95) if d['kind'] == 'normal' else stats.t.ppf(0.95, d['df'])
    return -z*d['sd'], z*d['sd']


def calibrate(fit_errors: np.ndarray, errors: np.ndarray, q: np.ndarray) -> tuple[dict, dict]:
    """Leave-one-calibration-trajectory-out CRPS, centroid family only."""
    losses = {method: np.zeros(len(SCALES)) for method in METHODS}
    for j in range(len(errors)):
        inner = model(fit_errors, np.delete(errors, j, axis=0))
        err = q @ errors[j]
        for method in METHODS:
            for k, scale in enumerate(SCALES):
                losses[method][k] += float(np.mean(crps(distribution(inner, method, q, scale), err))) / len(errors)
    scales = {method: SCALES[int(np.argmin(losses[method]))] for method in METHODS}
    return scales, {method: losses[method].tolist() for method in METHODS}


def bootstrap_ci(differences: np.ndarray, labels: np.ndarray) -> dict:
    """Stratified trajectory bootstrap, conditional on THESE TWO DLOs."""
    rng = np.random.default_rng(20260906)
    draws = np.zeros(BOOTSTRAP_REPLICATES)
    groups = sorted(set(labels))
    means = {}
    for group in groups:
        values = differences[labels == group]
        means[str(group)] = float(values.mean())
        draws += values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1) / len(groups)
    return {'difference': float(np.mean(list(means.values()))), 'by_dlo': means,
            'ci95': np.quantile(draws, (0.025, 0.975)).tolist(),
            'ci97_5_two_primary_bonferroni': np.quantile(draws, (0.0125, 0.9875)).tolist(),
            'trajectory_wins': int((differences < 0).sum()), 'trajectory_count': len(differences)}


def evaluate(carriers: Path, output: Path, request: dict) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((carriers / 'manifest.json').read_text())
    q, metadata = query_bank()
    calibration_q = q[[m['family'] == 'centroid' for m in metadata]]
    prepared, seal = {}, {'protocol': request, 'query_definitions': metadata, 'query_matrix_sha256': hashlib.sha256(q.tobytes()).hexdigest(), 'dlos': {}}
    for dlo in ('DLO4', 'DLO5'):
        path = carriers / (dlo + '.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['dlos'][dlo]['carrier']['sha256']
        with np.load(path, allow_pickle=False) as a:
            f = extract(a['fit_truth'], a['fit_initial']) - extract(a['fit_mean'], a['fit_initial'])
            e = extract(a['calibration_truth'], a['calibration_initial']) - extract(a['calibration_mean'], a['calibration_initial'])
            assert len(f) == 39 and len(e) == 9
        scales, cv = calibrate(f, e, calibration_q)
        m = model(f, e)
        prepared[dlo] = (m, scales)
        seal['dlos'][dlo] = {'scales': scales, 'source_cv_crps': cv, 'df': m['df'],
                            'model_sha256': hashlib.sha256(m['covariance'].tobytes() + m['centered'].tobytes()).hexdigest()}
    # Complete source fit/calibration seal precedes loading any source-test array.
    seal['sha256'] = digest(seal)
    (output / 'method_seal.json').write_text(json.dumps(seal, indent=2))
    rows, event_rows = [], []
    families = ('centroid', 'relative', 'bending', 'temporal')
    for dlo in ('DLO4', 'DLO5'):
        m, scales = prepared[dlo]
        with np.load(carriers / (dlo + '.npz'), allow_pickle=False) as a:
            means = extract(a['source_test_mean'], a['source_test_initial']) @ q.T
            truths = extract(a['source_test_truth'], a['source_test_initial']) @ q.T
            names = a['source_test_names'].tolist()
        assert len(names) == 8
        for method in METHODS:
            d = distribution(m, method, q, scales[method])
            lower, upper = intervals(d)
            for name, mu, truth in zip(names, means, truths):
                probabilities = 1 - cdf(d, THRESHOLDS[None, :] - mu[:, None]) + cdf(d, -THRESHOLDS[None, :] - mu[:, None], left=True)
                probabilities = np.clip(probabilities, 0, 1)
                labels = (np.abs(truth[:, None]) > THRESHOLDS[None, :]).astype(float)
                error = truth - mu
                scores = crps(d, error)
                logp = np.clip(probabilities, 1e-6, 1-1e-6)
                for family in families:
                    mask = np.asarray([v['family'] == family for v in metadata])
                    rows.append({'dlo': dlo, 'trajectory': name, 'method': method, 'family': family,
                                 'brier': float(np.mean((probabilities[mask] - labels[mask])**2)),
                                 'event_log_loss': float(np.mean(-labels[mask]*np.log(logp[mask]) - (1-labels[mask])*np.log1p(-logp[mask]))),
                                 'crps_m': float(scores[mask].mean()),
                                 'coverage90': float(((error[mask] >= lower[mask]) & (error[mask] <= upper[mask])).mean()),
                                 'width90_m': float((upper[mask] - lower[mask]).mean()),
                                 'mean_abs_error_m': float(np.abs(error[mask]).mean()),
                                 'event_rate': float(labels[mask].mean())})
                for k, definition in enumerate(metadata):
                    for t, threshold in enumerate(THRESHOLDS):
                        event_rows.append({'dlo': dlo, 'trajectory': name, 'method': method,
                                           'query': definition['name'], 'family': definition['family'],
                                           'threshold_m': float(threshold), 'probability': float(probabilities[k,t]), 'event': int(labels[k,t])})
    def dump_csv(path: Path, values: list[dict]) -> None:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    dump_csv(output / 'trajectory_scores.csv', rows)
    dump_csv(output / 'event_probabilities.csv', event_rows)
    summary = {}
    metrics = ('brier', 'event_log_loss', 'crps_m', 'coverage90', 'width90_m', 'mean_abs_error_m', 'event_rate')
    for family in ('heldout_macro',) + families:
        summary[family] = {}
        for method in METHODS:
            chosen = [r for r in rows if r['method'] == method and (r['family'] != 'centroid' if family == 'heldout_macro' else r['family'] == family)]
            summary[family][method] = {metric: float(np.mean([r[metric] for r in chosen])) for metric in metrics}
    cases = sorted({(r['dlo'], r['trajectory']) for r in rows})
    labels = np.asarray([case[0] for case in cases])
    def values(method: str, metric: str, family: str) -> np.ndarray:
        return np.asarray([np.mean([r[metric] for r in rows if (r['dlo'],r['trajectory']) == case and r['method'] == method and (r['family'] != 'centroid' if family == 'heldout_macro' else r['family'] == family)]) for case in cases])
    comparisons = {}
    for method in METHODS[1:]:
        comparisons[method] = {}
        for family in ('heldout_macro',) + families:
            comparisons[method][family] = {metric: bootstrap_ci(values('joint_student',metric,family)-values(method,metric,family), labels) for metric in ('brier','crps_m')}
    gates = {}
    for method in PRIMARY_CONTROLS:
        contrast = comparisons[method]['heldout_macro']['brier']
        gates[method] = contrast['ci97_5_two_primary_bonferroni'][1] < 0 and all(v < 0 for v in contrast['by_dlo'].values())
    parity = max(abs(summary['heldout_macro'][method]['mean_abs_error_m'] - summary['heldout_macro']['joint_student']['mean_abs_error_m']) for method in METHODS)
    assert parity == 0
    result = {'status': 'complete', 'hypothesis_supported': bool(all(gates.values())), 'primary_gates': gates,
              'summary': summary, 'paired_comparisons': comparisons, 'source_scales': {d: prepared[d][1] for d in prepared},
              'same_mean_error_parity': parity, 'dlo_count': 2, 'test_trajectory_count': 16,
              'query_count': len(metadata), 'thresholds_m': THRESHOLDS.tolist(),
              'evidence_class': 'retrospective-source-only-fixed-twin-covariance-layer-pilot',
              'original_bpt_covariance_tested': False, 'new_joint_covariance_layer_tested': True,
              'official_eval_opened': False, 'new_measurements': False,
              'statistical_unit': 'complete trajectory, stratified within two fixed DLOs; not population object inference',
              'method_seal_sha256': seal['sha256'], 'carrier_manifest': source}
    result['result_sha256'] = digest(result)
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    lines = ['# Held-out deformation uncertainty v1', '',
             '**Decision: ' + ('hypothesis supported in this bounded pilot' if result['hypothesis_supported'] else 'hypothesis not established') + '.**', '',
             'Source-only DEFORM DLO4/DLO5; 8 held-out source trajectories each. Frozen source twin means; new covariance-only posterior layer. Calibration uses centroid queries only. Held-out score equally weights relative, bending, and temporal families.', '',
             '| Method | Brier | CRPS (mm) | 90% coverage | Width (mm) |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        v = summary['heldout_macro'][method]
        lines.append(f"| {method} | {v['brier']:.6f} | {1000*v['crps_m']:.4f} | {100*v['coverage90']:.2f}% | {1000*v['width90_m']:.3f} |")
    lines += ['', '## Primary paired Brier contrasts', '']
    for method in PRIMARY_CONTROLS:
        c = comparisons[method]['heldout_macro']['brier']
        lines.append(f"Joint minus {method}: {c['difference']:+.6f}; simultaneous-control interval {c['ci97_5_two_primary_bonferroni']}; by DLO {c['by_dlo']}; gate {gates[method]}.")
    lines += ['', 'All methods share exactly the same point mean. Bootstrap resamples complete residual trajectories, not coordinates. Every arm has its own leave-one-calibration-trajectory-out CRPS scale fit. No test-family outcomes enter calibration.', '',
              'This is a retrospective 16-trajectory pilot conditional on two DLOs, not fresh confirmation, a test of the original stored BPT covariance, general object transfer, calibrated safety, or a universal Bayesian advantage. CRPS for continuous laws uses fixed 513-point quantile quadrature; primary Brier probabilities are analytic (Student/Gaussian) or exact empirical.', '', 'Result SHA256: `' + result['result_sha256'] + '`']
    report = '\n'.join(lines) + '\n'
    (output / 'report.md').write_text(report)
    print(report)
    return result


def self_test() -> None:
    rng = np.random.default_rng(1729)
    q, meta = query_bank()
    assert q.shape == (99, 96)
    assert sum(m['family'] == 'centroid' for m in meta) == 12
    f, e = rng.normal(size=(39,96)), rng.normal(size=(9,96))
    m = model(f,e)
    assert m['df'] == 13
    assert np.linalg.eigvalsh(m['covariance']).min() > 0
    # Direct projection equals covariance pushforward, and diagonal preserves coordinate marginals.
    identity_q = np.eye(96)
    full = distribution(m,'joint_student',identity_q,1)
    diagonal = distribution(m,'diagonal_student',identity_q,1)
    np.testing.assert_allclose(full['sd'],diagonal['sd'],atol=1e-14)
    np.testing.assert_allclose((e @ q.T).T @ (e @ q.T), q @ (e.T @ e) @ q.T, atol=1e-10)
    for method in METHODS:
        d = distribution(m,method,q,1)
        assert np.max(np.abs(d['samples'].mean(axis=1))) < 1e-10
        assert (crps(d,np.zeros(99)) >= 0).all()
        p = 1-cdf(d,np.ones((99,3)))+cdf(d,-np.ones((99,3)))
        assert ((p>=0)&(p<=1)).all()
    d = {'kind':'empirical','samples':np.asarray([[-2.,0.,2.]])}
    assert abs(crps(d,np.array([0.]))[0]-4/9) < 1e-12
    # A common displacement cancels in bending and relative-deformation queries.
    translation = np.ones((4,8,3)).reshape(-1)
    assert np.max(np.abs((q @ translation)[[x['family'] in ('relative','bending','temporal') for x in meta]])) < 1e-12
    print('SELF_TEST_OK: PSD, IW degrees of freedom, marginal parity, joint projection, exact shared means, proper scores, query semantics')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--request',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.request is None or args.output is None:
        parser.error('--request and --output are required')
    request = json.loads(args.request.read_text())
    assert request['mode'] == 'evaluate' and request['official_eval_access'] is False
    script = Path(__file__)
    assert hashlib.sha256(script.read_bytes()).hexdigest() == request['evaluator_sha256']
    protocol = json.loads(script.with_name('protocol.json').read_text())
    assert digest(protocol) == request['protocol_sha256']
    assert protocol['horizon_indices'] == list(HORIZONS)
    assert protocol['methods'] == list(METHODS)
    assert protocol['primary_controls'] == list(PRIMARY_CONTROLS)
    assert protocol['scale_grid'] == list(SCALES)
    assert protocol['thresholds_m'] == THRESHOLDS.tolist()
    carriers = args.output.parent / 'carriers'
    try:
        evaluate(carriers,args.output,protocol)
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'failure.json').write_text(json.dumps({'status':'technical-failure','error':repr(exc)}))
        raise


if __name__ == '__main__':
    main()
