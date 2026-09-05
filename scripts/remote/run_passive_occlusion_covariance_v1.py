"""Passive partial-observation covariance mechanism; NumPy only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import time
import traceback
from pathlib import Path

import numpy as np

OFFSETS = (0, 10, 25, 50)
MASKS = {"25pct": np.array([3, 4]), "50pct": np.arange(2, 6), "75pct": np.arange(1, 7)}
SEEDS = (601, 602, 603, 604)

def anchors(length: int) -> np.ndarray:
    """Prediction-array indices; no anchor-dependent selection."""
    out = np.arange(49, length - max(OFFSETS), 10)
    if len(out) < 10:
        raise ValueError("insufficient forecast length")
    return out

def windows(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4 or array.shape[2:] != (12, 3):
        raise ValueError("expected cases x time x 12 nodes x 3 coordinates")
    return array[:, anchors(array.shape[1])[:, None] + np.array(OFFSETS)[None, :], 2:10, :]

def fit_covariance(errors: np.ndarray, rank: int = 8, shrinkage: float = .1) -> np.ndarray:
    """Second moment about a fixed mean, retaining rank-r coherent errors.

    Source trajectories contribute equal numbers of windows. No point mean is
    refitted. The noncentral moment models error about the frozen predictor.
    """
    x = np.asarray(errors, dtype=np.float64).reshape(-1, 96)
    if not np.all(np.isfinite(x)) or x.shape[0] < 2:
        raise ValueError("invalid source errors")
    moment = x.T @ x / len(x)
    scale = np.sqrt(np.maximum(np.diag(moment), 1e-8))
    correlation = moment / np.outer(scale, scale)
    val, vec = np.linalg.eigh((correlation + correlation.T) / 2)
    u = vec[:, -rank:] * np.sqrt(np.maximum(val[-rank:], 0))[None]
    low = u @ u.T
    diagonal = np.maximum(np.diag(correlation) - np.diag(low), 0)
    correlation = (1 - shrinkage) * (low + np.diag(diagonal)) + shrinkage * np.diag(np.diag(correlation))
    covariance = correlation * np.outer(scale, scale) + np.eye(96) * 1e-8
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ValueError("non-positive covariance")
    return covariance

def covariance_arms(full: np.ndarray) -> dict[str, np.ndarray]:
    arms = {"full": full, "diagonal": np.diag(np.diag(full))}
    for seed in SEEDS:
        sign = np.random.default_rng(seed).choice([-1., 1.], size=len(full))
        arms[f"scrambled_{seed}"] = full * np.outer(sign, sign)
    for cov in arms.values():
        np.testing.assert_allclose(np.diag(cov), np.diag(full), rtol=0, atol=0)
    return arms

def coordinates(nodes: np.ndarray, offsets: tuple[int, ...] = (0,)) -> np.ndarray:
    return np.array([h * 24 + n * 3 + c for h in offsets for n in nodes for c in range(3)])

def conditional(cov: np.ndarray, visible: np.ndarray, noise_variance: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    obs = coordinates(visible)
    gain = np.linalg.solve(cov[np.ix_(obs, obs)] + np.eye(len(obs)) * noise_variance, cov[obs, :]).T
    posterior = cov - gain @ cov[obs, :]
    posterior = (posterior + posterior.T) / 2
    if np.min(np.linalg.eigvalsh(posterior)) < -1e-10:
        raise ValueError("non-positive conditional covariance")
    return gain, posterior

def interpolation_matrix(visible: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(np.arange(8), visible, np.eye(len(visible))[i]) for i in range(len(visible))], axis=1)

def simple_correction(innovation: np.ndarray, visible: np.ndarray, method: str) -> np.ndarray:
    x = innovation.reshape(*innovation.shape[:-1], len(visible), 3)
    if method == "global":
        correction = np.repeat(x.mean(axis=-2, keepdims=True), 8, axis=-2)
    elif method == "interpolation":
        correction = np.einsum('hv,...vc->...hc', interpolation_matrix(visible), x)
    else:
        raise ValueError(method)
    return np.repeat(correction[..., None, :, :], len(OFFSETS), axis=-3)

def fit_simple_gains(errors: np.ndarray) -> dict[str, np.ndarray]:
    gains = {}
    for name, hidden in MASKS.items():
        visible = np.setdiff1d(np.arange(8), hidden)
        innovation = errors[..., 0, visible, :].reshape(*errors.shape[:2], -1)
        for method in ('global', 'interpolation'):
            correction = simple_correction(innovation, visible, method)
            c = correction[..., hidden, :]
            y = errors[..., hidden, :]
            num = np.sum(c * y, axis=(0, 1, 3, 4))
            den = np.sum(c * c, axis=(0, 1, 3, 4))
            gains[f'{name}_{method}'] = np.clip(num / np.maximum(den, 1e-20), 0., 2.)
    return gains

def predict(base: np.ndarray, observations: np.ndarray, visible: np.ndarray,
            covs: dict[str, np.ndarray], gains: dict[str, np.ndarray], mask: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """This API accepts only frozen prior windows and current VISIBLE readouts.

    It cannot access hidden or future ground truth. All arms share the identical
    frozen prior. Every update is reset to that prior to isolate conditioning.
    """
    if base.shape[-3:] != (4, 8, 3):
        raise ValueError('invalid base shape')
    if observations.shape != (*base.shape[:2], len(visible), 3):
        raise ValueError('observations must contain visible nodes only')
    innovation = (observations - base[..., 0, visible, :]).reshape(*base.shape[:2], -1)
    predictions = {'unchanged': base.copy()}
    covariance = {}
    for name, cov in covs.items():
        gain, posterior = conditional(cov, visible)
        correction = (innovation @ gain.T).reshape(base.shape)
        predictions[name] = base + correction
        covariance[name] = posterior
    for method in ('global', 'interpolation'):
        correction = simple_correction(innovation, visible, method)
        predictions[method] = base + correction * gains[f'{mask}_{method}'][None, None, :, None, None]
    if not all(np.isfinite(x).all() for x in predictions.values()):
        raise FloatingPointError('nonfinite forecast')
    return predictions, covariance

def self_test() -> None:
    rng = np.random.default_rng(72)
    source = rng.normal(size=(12, 18, 4, 8, 3))
    # Shared spatial mode ensures the positive control has a real effect.
    source += 4 * rng.normal(size=(12, 18, 1, 1, 3))
    cov = fit_covariance(source)
    covs = covariance_arms(cov)
    gains = fit_simple_gains(source)
    hidden = MASKS['50pct']; visible = np.setdiff1d(np.arange(8), hidden)
    base = np.zeros((3, 11, 4, 8, 3))
    observation = np.ones((3, 11, len(visible), 3))
    result, post = predict(base, observation, visible, covs, gains, '50pct')
    np.testing.assert_array_equal(result['diagonal'][..., hidden, :], base[..., hidden, :])
    assert np.linalg.norm(result['full'][..., hidden, :]) > 1
    assert np.trace(post['full']) <= np.trace(cov)
    for arm in covs.values():
        assert np.min(np.linalg.eigvalsh(arm)) > 0
    np.testing.assert_allclose(np.linalg.eigvalsh(covs['scrambled_601']), np.linalg.eigvalsh(cov))
    # Gaussian conditional mean equals the deterministic joint quadratic MAP.
    gain, posterior = conditional(cov, visible)
    obs = coordinates(visible)
    precision = np.linalg.inv(cov)
    precision[obs, obs] += 1e6
    rhs = np.zeros(96); rhs[obs] = 1e6
    expected = np.linalg.solve(precision, rhs)
    np.testing.assert_allclose(gain @ np.ones(len(obs)), expected, rtol=1e-6, atol=1e-6)
    # Mutating all hidden/future truth has no effect on inference.
    truth = rng.normal(size=base.shape)
    allowed = truth[..., 0, visible, :].copy()
    a, _ = predict(base, allowed, visible, covs, gains, '50pct')
    truth[..., 1:, :, :] = 1e9
    truth[..., 0, hidden, :] = -1e9
    b, _ = predict(base, truth[..., 0, visible, :], visible, covs, gains, '50pct')
    for arm in a:
        np.testing.assert_array_equal(a[arm], b[arm])
    np.testing.assert_allclose(simple_correction(np.ones((2, len(visible)*3)), visible, 'interpolation'), 1)
    assert windows(np.zeros((2, 499, 12, 3))).shape == (2, 40, 4, 8, 3)
    print('SELF_TEST_PASS: marginal parity, PSD, hidden/future leakage, deterministic MAP equivalence, diagonal no-transfer, fixed masks, shapes')


# The driver separates source fitting, masked inference, and hidden-value scoring.

DLOS = ('DLO4', 'DLO5')
CONTRACT = 'passive-occlusion-covariance-v1'
CACHE = '/home/github-runner/.cache/workflows/deform-dlo45-time-budget-recovery-v3/runs/33361441865-1'
DATA = '/mnt/seagate10tb/florianpfaff/datasets/deform/data_set'
PARENTS = {
    'DLO4': {
        'source_predictions.npz': ('b858c2f4c0e107d367857dc2c4e17e753b6853e4593af64c4ce489564ae88fca', 9736605),
        'source_manifest.json': ('1001cdb688072675eafdba34c8dc7f937be2fb7800c65d756e31214bc244fa17', 14511),
        'target_predictions.npz': ('b28f0bde14fdef1bc557cc3a8b7cde2b87832b06ba971aa4d8d82afb2905364b', 18273132),
        'eval_manifest.json': ('10ce91fc82df072cdbf36ce4f4bc1a0b57fc112f8f0b356b3aecafee26306d', 3877),
    },
    'DLO5': {
        'source_predictions.npz': ('a8f8b3253f2588eed3fb0e92a8812ce534dba857c82a3af9d3f0fe0dadab39b6', 9757431),
        'source_manifest.json': ('6bc72a4b3c7dae13c39181d28aebe9228fb1049fbe4bf26871b95fb518997a04', 14523),
        'target_predictions.npz': ('86b5fd6cde97594e96640c98e071862260941bebb21b82d4b65cbd9abdd0f612', 18280280),
        'eval_manifest.json': ('48bd004dbba2161609ada67ec76369a9271fa6899624c1ba1a84b747654e6b7d', 3871),
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n')


def verify(path: Path, expected_hash: str, size: int) -> None:
    if not path.is_file() or path.stat().st_size != size or digest(path) != expected_hash:
        raise RuntimeError(f'File identity mismatch: {path}')


def load_panel(dlo: str, stage: str, cache: Path = Path(CACHE)) -> tuple[list[str], np.ndarray, dict]:
    directory = cache / f'{dlo.lower()}-{stage}'
    name = 'source_predictions.npz' if stage == 'source' else 'target_predictions.npz'
    manifest_name = 'source_manifest.json' if stage == 'source' else 'eval_manifest.json'
    manifest = json.loads((directory / manifest_name).read_text())
    if manifest['dlo'] != dlo:
        raise ValueError('DLO manifest mismatch')
    names = manifest['partitions']['source_test'] if stage == 'source' else manifest['ordered_names']
    with np.load(directory / name, allow_pickle=False) as archive:
        if archive['names'].tolist() != names:
            raise ValueError('Forecast/manifest ordering mismatch')
        predictions = np.asarray(archive['candidate'], dtype=np.float64)
    expected = 8 if stage == 'source' else 14
    if len(names) != expected or len(set(names)) != expected or predictions.shape != (expected, 498, 12, 3):
        raise ValueError(f'Unexpected {dlo}/{stage} roster or prediction shape: {predictions.shape}')
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite parent forecast')
    return names, predictions, manifest


def load_reference(dlo: str, stage: str, name: str, manifest: dict, root: Path = Path(DATA)) -> np.ndarray:
    """Trusted hash-bound pickle; entire record deserialized only by IO/scoring.

    Inference never receives this full array. Coordinates and [2:] alignment
    exactly match the successful retained-parent evaluation.
    """
    partition = 'train' if stage == 'source' else 'eval'
    if Path(name).name != name or not name.endswith('.pkl'):
        raise ValueError('Invalid trajectory name')
    path = root / dlo / partition / name
    identity = manifest['trajectories'][name]
    verify(path, identity['sha256'], identity['size_bytes'])
    with path.open('rb') as stream:
        raw = pickle.load(stream)  # noqa: S301 -- trusted, hash-bound public benchmark
    array = np.asarray(raw, dtype=np.float32)
    if array.shape != (500, 3, 12) or not np.isfinite(array).all():
        raise ValueError('Invalid reference shape or values')
    nodes = array.transpose(0, 2, 1).astype(np.float64, copy=True)
    nodes[:, :, 2] = np.clip(nodes[:, :, 2], 2e-3 + 1e-6, 10000.)
    return nodes[2:]


def score(predictions: dict, posterior: dict, truth: np.ndarray, hidden: np.ndarray,
          dlo: str, names: list[str], mask: str) -> list[dict]:
    records = []
    for arm, prediction in predictions.items():
        error = (prediction - truth)[..., hidden, :]
        for lead_index, lead in enumerate(OFFSETS):
            e = error[:, :, lead_index]
            rmse = np.sqrt(np.mean(np.sum(e * e, axis=-1), axis=(1, 2))) * 1000
            mae = np.mean(np.abs(e), axis=(1, 2, 3)) * 1000
            metrics = {'rmse_mm': rmse, 'coordinate_mae_mm': mae}
            if arm in posterior:
                idx = coordinates(hidden, (lead_index,))
                cov = posterior[arm][np.ix_(idx, idx)]
                variance = np.diag(cov)
                flat = e.reshape(len(names), -1, len(idx))
                sd = np.sqrt(variance)
                metrics['coverage_90'] = np.mean(np.abs(flat) <= 1.6448536269514722 * sd, axis=(1, 2))
                metrics['mean_full_width_mm'] = np.full(len(names), np.mean(2 * 1.6448536269514722 * sd) * 1000)
                sign, logdet = np.linalg.slogdet(cov)
                if sign != 1:
                    raise ValueError('Conditional marginal covariance not positive definite')
                quadratic = np.einsum('nwi,ij,nwj->nw', flat, np.linalg.inv(cov), flat)
                metrics['normalized_nees'] = np.mean(quadratic, axis=1) / len(idx)
                metrics['nll_per_coordinate'] = .5 * (np.log(2 * np.pi) + logdet / len(idx) + metrics['normalized_nees'])
            for i, name in enumerate(names):
                records.append({'dlo': dlo, 'trajectory': name, 'mask': mask, 'lead_frames': lead,
                                'arm': arm, **{k: float(v[i]) for k, v in metrics.items()}})
    # Average losses across fixed scrambled replicates, NEVER average predictions.
    for name in names:
        for lead in OFFSETS:
            selected = [r for r in records if r['trajectory'] == name and r['lead_frames'] == lead and r['arm'].startswith('scrambled_')]
            if len(selected) != len(SEEDS):
                raise ValueError('Missing scramble replicate')
            numeric = [k for k in selected[0] if k not in ('dlo', 'trajectory', 'mask', 'lead_frames', 'arm')]
            records.append({'dlo': dlo, 'trajectory': name, 'mask': mask, 'lead_frames': lead, 'arm': 'scrambled_average_loss',
                            **{k: float(np.mean([r[k] for r in selected])) for k in numeric}})
    return records


def contrast(records: list[dict], reference: str) -> dict:
    rng = np.random.default_rng(20260906)
    bootstrap = np.zeros(20000)
    candidate_values, reference_values, by_dlo = [], [], {}
    for dlo in DLOS:
        cand = {r['trajectory']: r['rmse_mm'] for r in records if r['dlo'] == dlo and r['arm'] == 'full'}
        ref = {r['trajectory']: r['rmse_mm'] for r in records if r['dlo'] == dlo and r['arm'] == reference}
        if len(cand) != 14 or set(cand) != set(ref):
            raise ValueError('Incomplete paired trajectory panel')
        c = np.array([cand[n] for n in sorted(cand)])
        b = np.array([ref[n] for n in sorted(cand)])
        difference = c - b
        bootstrap += .5 * np.mean(difference[rng.integers(0, len(c), size=(len(bootstrap), len(c)))], axis=1)
        by_dlo[dlo] = {'full_mm': float(c.mean()), 'reference_mm': float(b.mean()),
                       'difference_mm': float(difference.mean()), 'wins': int(np.sum(difference < 0)),
                       'ties': int(np.sum(difference == 0)), 'worst_ratio': float(np.max(c / np.maximum(b, 1e-12)))}
        candidate_values.append(c)
        reference_values.append(b)
    cmean = float(np.mean(candidate_values)); bmean = float(np.mean(reference_values))
    ci = np.quantile(bootstrap, [.025, .975]).tolist()
    # Three prespecified primary contrasts, familywise alpha .05 Bonferroni.
    corrected = np.quantile(bootstrap, [.05 / 6, 1 - .05 / 6]).tolist()
    return {'full_mm': cmean, 'reference_mm': bmean, 'difference_mm': cmean-bmean,
            'relative_improvement_pct': 100 * (1-cmean/bmean), 'paired_bootstrap_95_mm': ci,
            'paired_bootstrap_98p333_mm': corrected, 'positive_95': ci[1] < 0,
            'positive_primary_familywise': corrected[1] < 0, 'by_dlo': by_dlo}


def summarize(records: list[dict]) -> dict:
    result = {}
    for mask in MASKS:
        for lead in OFFSETS:
            rows = [r for r in records if r['mask'] == mask and r['lead_frames'] == lead]
            arms = sorted({r['arm'] for r in rows})
            means = {arm: float(np.mean([r['rmse_mm'] for r in rows if r['arm'] == arm])) for arm in arms}
            result[f'{mask}_lead{lead}'] = {'mean_rmse_mm': means, 'contrasts': {
                ref: contrast(rows, ref) for ref in ('diagonal', 'scrambled_average_loss', 'interpolation', 'global')
            }}
    primary = result['50pct_lead0']['contrasts']
    accepted = all(primary[ref]['positive_primary_familywise'] for ref in ('diagonal', 'scrambled_average_loss', 'interpolation'))
    return {'primary_condition': '50pct_lead0', 'primary_three_control_gate_passed': accepted, 'conditions': result}


def markdown(summary: dict) -> str:
    lines = ['# Passive occlusion covariance: retrospective real-data mechanism test', '',
             '28 evaluation trajectories, 14 each from DLO4/DLO5. Entire trajectories are the bootstrap units.',
             'No new physical data, active observation selection, hybrid retraining, or target fitting.',
             '', '| Hidden nodes | Lead (ms) | Unchanged/diagonal | Full covariance | Scrambled mean loss | Interpolation | Global |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mask in MASKS:
        for lead in OFFSETS:
            m = summary['conditions'][f'{mask}_lead{lead}']['mean_rmse_mm']
            lines.append(f"| {mask} | {lead*10} | {m['diagonal']:.4f} | {m['full']:.4f} | {m['scrambled_average_loss']:.4f} | {m['interpolation']:.4f} | {m['global']:.4f} |")
    lines += ['', 'All errors are hidden-node Euclidean 3D RMSE in mm, averaged equally across trajectories/DLOs.', '',
              '## Primary test: 50% hidden, current-time reconstruction', '']
    for ref, c in summary['conditions']['50pct_lead0']['contrasts'].items():
        lines.append(f"- Full versus `{ref}`: {c['relative_improvement_pct']:.2f}% improvement; difference {c['difference_mm']:.4f} mm, paired 95% interval {c['paired_bootstrap_95_mm']}; 98.333% interval {c['paired_bootstrap_98p333_mm']}.")
    lines += ['', f"Three-primary-control gate passed: **{summary['primary_three_control_gate_passed']}**.", '',
              '## Interpretation boundary', '',
              'This is source-fitted empirical joint predictive-error conditioning over fixed, already-open hybrid forecasts. It is not a fresh confirmatory cohort, feedback into a simulator, or proof of universally calibrated physical-state uncertainty. The diagonal arm cannot transfer information to wholly hidden nodes. Interpolation/global controls receive the same readouts and source-fitted gains. A deterministic optimizer using the identical joint Gaussian model reproduces the full conditional mean.',
              'Source residual structure comes from the prior 39-fit hybrid applied to eight source-held trajectories; the evaluation forecasts come from the frozen all-56 retraining. Both are unchanged. Hyperparameters and primary endpoints are fixed before this numerical run.', '']
    return '\n'.join(lines)


def execute(request_path: Path, output: Path) -> dict:
    request = json.loads(request_path.read_text())
    protocol_path = Path('experiments/passive_occlusion_covariance_v1/protocol.json')
    if request.get('mode') != 'evaluate' or request.get('contract') != CONTRACT:
        raise ValueError('Invalid evaluation request')
    if digest(protocol_path) != request['protocol_sha256'] or digest(Path(__file__)) != request['implementation_sha256']:
        raise ValueError('Frozen protocol/implementation mismatch')
    protocol = json.loads(protocol_path.read_text())
    if protocol['contract'] != CONTRACT or protocol['offsets_frames'] != list(OFFSETS):
        raise ValueError('Wrong protocol')
    if output.exists():
        raise FileExistsError('Refusing to overwrite experiment output')
    output.mkdir(parents=True)
    write_json(output / 'request.json', request)
    write_json(output / 'protocol.json', protocol)
    start = time.time()
    # Validate all eight parent files before any outcome deserialization.
    for dlo in DLOS:
        for filename, (sha, size) in PARENTS[dlo].items():
            stage = 'source' if filename.startswith('source_') else 'target'
            verify(Path(CACHE) / f'{dlo.lower()}-{stage}' / filename, sha, size)
    models, source_info = {}, {}
    for dlo in DLOS:
        names, forecast, manifest = load_panel(dlo, 'source')
        reference = np.stack([load_reference(dlo, 'source', n, manifest) for n in names])
        errors = windows(reference) - windows(forecast)
        cov = fit_covariance(errors)
        gains = fit_simple_gains(errors)
        models[dlo] = (covariance_arms(cov), gains)
        model_path = output / f'{dlo}_source_model.npz'
        np.savez_compressed(model_path, full_covariance_m2=cov, **gains)
        source_info[dlo] = {'source_names': names, 'source_trajectories': len(names),
                           'windows_per_trajectory': errors.shape[1], 'source_model_sha256': digest(model_path),
                           'source_forecast_sha256': PARENTS[dlo]['source_predictions.npz'][0],
                           'target_fitted': False}
    write_json(output / 'source_model_seal.json', {'created_unix': time.time(), 'models': source_info,
                                                  'target_observations_read': False})
    prediction_seals, references = {}, {}
    # All arms and both DLOs sealed before hidden-value scoring.
    for dlo in DLOS:
        names, forecast, manifest = load_panel(dlo, 'target')
        base = windows(forecast)
        references[dlo] = (names, manifest)
        covs, gains = models[dlo]
        for mask, hidden in MASKS.items():
            visible = np.setdiff1d(np.arange(8), hidden)
            # IO isolates only allowed readouts; predictor has no full target argument.
            observations = np.stack([load_reference(dlo, 'target', n, manifest)[anchors(498)][:, visible + 2] for n in names])
            predictions, posterior = predict(base, observations, visible, covs, gains, mask)
            np.testing.assert_array_equal(predictions['diagonal'][..., hidden, :], base[..., hidden, :])
            path = output / f'{dlo}_{mask}_predictions.npz'
            np.savez_compressed(path, names=np.array(names), anchors=anchors(498), visible=visible, hidden=hidden,
                                **{f'prediction__{k}': v for k, v in predictions.items()},
                                **{f'posterior__{k}': v for k, v in posterior.items()})
            prediction_seals[f'{dlo}_{mask}'] = {'sha256': digest(path), 'size_bytes': path.stat().st_size}
            del predictions, posterior, observations
    write_json(output / 'prediction_seal.json', {'created_unix': time.time(), 'predictions': prediction_seals,
                                                'hidden_outcomes_used_by_inference': False,
                                                'hidden_outcomes_scored': False})
    records = []
    for dlo in DLOS:
        names, manifest = references[dlo]
        truth = windows(np.stack([load_reference(dlo, 'target', n, manifest) for n in names]))
        for mask, hidden in MASKS.items():
            path = output / f'{dlo}_{mask}_predictions.npz'
            identity = prediction_seals[f'{dlo}_{mask}']
            verify(path, identity['sha256'], identity['size_bytes'])
            with np.load(path, allow_pickle=False) as stored:
                predictions = {k.removeprefix('prediction__'): stored[k] for k in stored.files if k.startswith('prediction__')}
                posterior = {k.removeprefix('posterior__'): stored[k] for k in stored.files if k.startswith('posterior__')}
            records += score(predictions, posterior, truth, hidden, dlo, names, mask)
    summary = summarize(records)
    summary.update({'contract': CONTRACT, 'retrospective': True, 'trajectory_count': 28,
                    'source_trajectory_count': 16, 'source': source_info, 'parent_artifacts': PARENTS,
                    'protocol_sha256': digest(protocol_path), 'implementation_sha256': digest(Path(__file__)),
                    'github_sha': os.environ.get('GITHUB_SHA', ''), 'github_run_id': os.environ.get('GITHUB_RUN_ID', ''),
                    'runner_name': os.environ.get('RUNNER_NAME', ''), 'numpy': np.__version__,
                    'python': platform.python_version(), 'wall_seconds': time.time()-start,
                    'source_seal_sha256': digest(output / 'source_model_seal.json'),
                    'prediction_seal_sha256': digest(output / 'prediction_seal.json')})
    write_json(output / 'summary.json', summary)
    write_json(output / 'trajectory_records.json', {'records': records})
    fields = ['dlo', 'trajectory', 'mask', 'lead_frames', 'arm', 'rmse_mm', 'coordinate_mae_mm',
              'coverage_90', 'mean_full_width_mm', 'normalized_nees', 'nll_per_coordinate']
    with (output / 'trajectory_records.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    text = markdown(summary)
    (output / 'summary.md').write_text(text)
    print(text, flush=True)
    print('RESULT_JSON_PRIMARY=' + json.dumps(summary['conditions']['50pct_lead0'], sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.request is None or args.output is None:
        parser.error('--request and --output required')
    try:
        execute(args.request, args.output)
    except Exception:
        # Technical failure is retained, never interpreted as a negative science result.
        if args.output.is_dir() and not (args.output / 'failure.json').exists():
            write_json(args.output / 'failure.json', {'status': 'technical_failure', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
