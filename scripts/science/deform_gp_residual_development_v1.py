"""Development-only, matched sparse-GP residual trial; never opens evaluation data.

Zero-mean Matern-3/2 FITC Gaussian processes, independently per node and with a
normalized material-graph covariance. GP hyperparameters are fixed before the
historical eight-case development validation is read. All shrinkage arms are
reported, not promoted as independent confirmation. No covariance-calibration
claim is made. The DEFORM baseline is the trained rod-plus-GCN hybrid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist, pdist


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def material_kernel(count: int) -> np.ndarray:
    lap = np.zeros((count, count))
    for i in range(count - 1):
        lap[i, i] += 1
        lap[i + 1, i + 1] += 1
        lap[i, i + 1] -= 1
        lap[i + 1, i] -= 1
    base = np.linalg.solve(np.eye(count) + 2 * lap, np.eye(count))
    cov = base @ base.T
    scale = np.sqrt(np.diag(cov))
    return cov / scale[:, None] / scale[None, :]


def kernel(x, z, nx, nz, length: float, graph=None):
    distance = cdist(x, z) * (np.sqrt(3.0) / length)
    result = (1 + distance) * np.exp(-distance)
    if graph is not None:
        result *= graph[np.asarray(nx)[:, None], np.asarray(nz)[None, :]]
    return result


class SparseGP:
    """Whitened inducing-variable FITC GP, independent output amplitudes."""

    def __init__(self, inducing=128, noise=0.05, seed=7, graph=None):
        self.inducing, self.noise, self.seed = inducing, noise, seed
        self.graph = graph

    def fit(self, x: np.ndarray, y: np.ndarray, nodes: np.ndarray):
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite GP training arrays")
        self.location = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-8)
        xx = (x - self.location) / self.scale
        rng = np.random.default_rng(self.seed)
        pool = rng.choice(len(xx), min(2048, len(xx)), replace=False)
        distances = pdist(xx[pool[:512]])
        positive = distances[distances > 1e-10]
        self.length = float(np.median(positive)) if len(positive) else 1.0
        # Greedy kernel-distance coverage, using only fitting features.
        selected = [int(pool[0])]
        nearest = np.full(len(pool), np.inf)
        for _ in range(min(self.inducing, len(pool)) - 1):
            i = selected[-1]
            sim = kernel(xx[pool], xx[i:i + 1], nodes[pool], nodes[i:i + 1],
                         self.length, self.graph)[:, 0]
            nearest = np.minimum(nearest, np.maximum(2 - 2 * sim, 0))
            nearest[np.isin(pool, selected)] = -1
            selected.append(int(pool[np.argmax(nearest)]))
        self.z, self.znodes = xx[selected], nodes[selected]
        self.lmm = cholesky(kernel(self.z, self.z, self.znodes, self.znodes,
                                  self.length, self.graph)
                           + np.eye(len(selected)) * 1e-7, lower=True)
        self.amplitude = np.maximum(np.sqrt(np.mean(y * y, axis=0)), 1e-8)
        response = y / self.amplitude
        normal, rhs = np.eye(len(selected)), np.zeros((len(selected), y.shape[1]))
        for start in range(0, len(xx), 2048):
            sl = slice(start, start + 2048)
            phi = self._phi(xx[sl], nodes[sl])
            diagonal = self.noise + np.maximum(1 - np.sum(phi * phi, axis=1), 0)
            normal += phi.T @ (phi / diagonal[:, None])
            rhs += phi.T @ (response[sl] / diagonal[:, None])
        self.lnormal = cholesky(normal, lower=True)
        self.beta = cho_solve((self.lnormal, True), rhs)
        return self

    def _phi(self, xx, nodes):
        return solve_triangular(self.lmm,
                                kernel(xx, self.z, nodes, self.znodes,
                                       self.length, self.graph).T,
                                lower=True).T

    def predict(self, x, nodes):
        xx = (x - self.location) / self.scale
        mean, variance = [], []
        for start in range(0, len(xx), 2048):
            sl = slice(start, start + 2048)
            phi = self._phi(xx[sl], nodes[sl])
            solved = solve_triangular(self.lnormal, phi.T, lower=True)
            latent = (np.maximum(1 - np.sum(phi * phi, axis=1), 0)
                      + np.sum(solved * solved, axis=0))
            mean.append((phi @ self.beta) * self.amplitude)
            variance.append((latent[:, None] + self.noise) * self.amplitude ** 2)
        return np.concatenate(mean), np.concatenate(variance)


def self_test():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(20, 2))
    y = np.column_stack((np.sin(x[:, 0]), np.cos(x[:, 1])))
    nodes = np.zeros(len(x), dtype=int)
    gp = SparseGP(inducing=len(x)).fit(x, y, nodes)
    mean, variance = gp.predict(x, nodes)
    xx = (x - gp.location) / gp.scale
    kk = kernel(xx, xx, nodes, nodes, gp.length)
    dense = kk @ np.linalg.solve(kk + .05 * np.eye(len(x)), y)
    assert np.max(np.abs(mean - dense)) < 2e-5
    assert np.isfinite(variance).all() and np.min(variance) > 0
    graph = material_kernel(8)
    assert np.linalg.eigvalsh(graph).min() > 0
    assert np.allclose(np.diag(graph), 1)
    gp2 = SparseGP(inducing=len(x)).fit(x, y, nodes)
    assert np.array_equal(gp2.predict(x, nodes)[0], mean)
    coupled = SparseGP(inducing=15, graph=graph).fit(x, y, np.arange(20) % 8)
    assert np.isfinite(coupled.predict(x, np.arange(20) % 8)[0]).all()
    print('SELF_TEST_PASS dense-GP parity, positive variance, graph PSD, deterministic fit', flush=True)


def locate_upstream() -> Path:
    root = Path('/home/florianpfaff/source-only/deform-bayesian-v1')
    candidates = [root / 'upstream', root / 'upstream' / 'DEFORM', root / 'DEFORM']
    upstream = root / 'upstream'
    if upstream.is_dir():
        candidates += sorted(p for p in upstream.iterdir() if p.is_dir())
    for path in candidates:
        if (path / 'train_DEFORM.py').is_file() and (path / 'data_set/DLO2/train').is_dir():
            return path.resolve()
    raise FileNotFoundError('Approved DEFORM upstream not found: ' + str(candidates))


def metrics(prediction, truth, baseline, ridge, names):
    errors = np.mean(np.abs(prediction - truth), axis=(1, 2, 3)) * 1000
    base = np.mean(np.abs(baseline - truth), axis=(1, 2, 3)) * 1000
    rid = np.mean(np.abs(ridge - truth), axis=(1, 2, 3)) * 1000
    return {
        'mean_coordinate_l1_mm': float(errors.mean()),
        'rmse_mm': float(np.sqrt(np.mean((prediction - truth) ** 2)) * 1000),
        'improvement_over_hybrid_percent': float(100 * (1 - errors.mean() / base.mean())),
        'improvement_over_ridge_percent': float(100 * (1 - errors.mean() / rid.mean())),
        'wins_over_ridge': int(np.sum(errors < rid - 1e-10)),
        'wins_over_hybrid': int(np.sum(errors < base - 1e-10)),
        'horizon_thirds_l1_mm': [float(np.mean(np.abs(prediction[:, s] - truth[:, s])) * 1000)
                                for s in np.array_split(np.arange(truth.shape[1]), 3)],
        'case_l1_mm': dict(zip(names, map(float, errors))),
    }


def run(output: Path):
    import torch
    import run_deform_dlo_action_residual as common
    import run_deform_dlo2_local_residual as runtime
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features, fit_deform_local_residual,
        predict_deform_local_residual,
    )
    output.mkdir(parents=True, exist_ok=True)
    root = Path('/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/train-8cc85de7/training_run')
    training_path, manifest_path = root / 'training_validation_result.json', root / 'source_manifest.json'
    training = json.loads(training_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    selected = training['model_selection']['DLO2']
    checkpoint = root / 'checkpoints' / (selected['selected_checkpoint'] + '.pt')
    if digest(checkpoint) != 'b7ba823f75a7f8afbdf5c28f208624b4f930a798550bc394831a1b54be6b93d554be':
        raise ValueError('Frozen checkpoint hash mismatch')
    partition = training['partitions']['DLO2']
    if len(partition['fit_names']) != 40 or len(partition['validation_names']) != 8:
        raise ValueError('Historical development partition changed')
    if set(partition['fit_names']) & set(partition['validation_names']):
        raise ValueError('Overlapping development partitions')
    protocol = json.loads(Path('configs/sota/deform_dlo2_local_residual_v6.json').read_text())
    protocol['source_protocol']['path'] = protocol['source_protocol']['repository_path']
    upstream = locate_upstream()
    print('UPSTREAM', str(upstream), flush=True)
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '740aafbc714130ecdbc24b1a09c09511b5ecb1ca4':
        raise ValueError('Upstream source revision mismatch: ' + revision)
    common._seed_everything(7)
    _, deform_func, deform_sim = common._install_upstream_imports(upstream)
    provenance = {
        'status': 'historical-development-only', 'official_eval_read': False,
        'source_test_read': False, 'new_robot_data': False,
        'baseline': 'frozen 6400-update DEFORM rod-plus-GCN hybrid',
        'upstream_revision': revision, 'checkpoint_sha256': digest(checkpoint),
        'training_result_sha256': digest(training_path), 'source_manifest_sha256': digest(manifest_path),
        'fit_names': partition['fit_names'], 'validation_names': partition['validation_names'],
        'gp': {'approximation': 'FITC', 'kernel': 'Matern-3/2', 'noise_to_signal_variance': .05,
               'lengthscale': 'median fitting-feature distance', 'node_inducing': 128,
               'coupled_inducing': 256, 'shrinkages': [.25, .5, 1.0], 'seed': 7},
        'selection': 'all predefined arms reported; no independent confirmation',
    }
    (output / 'protocol.json').write_text(json.dumps(provenance, indent=2))
    cache = Path('/home/github-runner/.cache/workflows/deform-gp-residual-development-v1/cache')
    cache.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256((digest(checkpoint) + digest(training_path) + revision).encode()).hexdigest()[:16]
    data = {}
    for prefix in ('fit', 'validation'):
        path = cache / (cache_key + '-' + prefix + '.npz')
        print('ROLLOUT', prefix, 'cached' if path.is_file() else 'compute', flush=True)
        if path.is_file():
            with np.load(path, allow_pickle=False) as stored:
                arrays = tuple(stored[key] for key in ('initial', 'action', 'baseline', 'targets'))
                names = tuple(str(n) for n in stored['names'])
            if names != tuple(partition[prefix + '_names']):
                raise ValueError('Cached partition mismatch')
            data[prefix] = (*arrays, names)
        else:
            data[prefix] = runtime._rollout(prefix=prefix, protocol=protocol,
                training_result=training, source_manifest=manifest, training_root=root,
                upstream_root=upstream, device=torch.device('cuda'),
                deform_sim=deform_sim, deform_func=deform_func)
            initial, action, baseline, targets, names = data[prefix]
            np.savez_compressed(path, initial=initial, action=action, baseline=baseline,
                                targets=targets, names=np.asarray(names))
        print('ROLLOUT_DONE', prefix, data[prefix][2].shape, flush=True)
    fi, fa, fb, fy, fn = data['fit']
    vi, va, vb, vy, vn = data['validation']
    baseline_error = float(np.mean(np.abs(vb - vy)))
    expected = float(selected['selected_validation_l1_m'])
    if abs(baseline_error - expected) > 1e-5:
        raise ValueError(f'Hybrid reproduction mismatch: {baseline_error} vs {expected}')
    ridge_model = fit_deform_local_residual(fi, fa, fb, fy, fn, ridge=1.0, variance_floor_m2=1e-8)
    ridge = predict_deform_local_residual(ridge_model, vi, va, vb, shrinkage=.25)['prediction']
    xfit, fframe = build_deform_local_residual_features(fi, fa, fb)
    xval, vframe = build_deform_local_residual_features(vi, va, vb)
    target = np.einsum('ntvi,nij->ntvj', fy - fb, fframe)[:, :, 2:-2]
    shape, node_count = target.shape, target.shape[2]
    results = {'hybrid': metrics(vb, vy, vb, ridge, vn),
               'ridge_s0.25': metrics(ridge, vy, vb, ridge, vn)}
    np.savez_compressed(output / 'validation_references.npz', hybrid=vb, ridge=ridge,
                        target=vy, names=np.asarray(vn))
    print('BASELINE_RESULTS', json.dumps(results), flush=True)
    for mode in ('independent', 'material_coupled'):
        start = time.monotonic()
        if mode == 'independent':
            means, variances = [], []
            for node in range(node_count):
                xx = xfit[:, :, node].reshape(-1, xfit.shape[-1])
                yy = target[:, :, node].reshape(-1, 3)
                vv = xval[:, :, node].reshape(-1, xval.shape[-1])
                gp = SparseGP(128, seed=7 + node).fit(xx, yy, np.zeros(len(xx), dtype=int))
                m, v = gp.predict(vv, np.zeros(len(vv), dtype=int))
                means.append(m.reshape(*xval.shape[:2], 3))
                variances.append(v.reshape(*xval.shape[:2], 3))
                print('GP_NODE_DONE', node, flush=True)
            mean, variance = np.stack(means, axis=2), np.stack(variances, axis=2)
        else:
            xx, yy = xfit.reshape(-1, xfit.shape[-1]), target.reshape(-1, 3)
            nodes = np.broadcast_to(np.arange(node_count), shape[:-1]).reshape(-1)
            vv = xval.reshape(-1, xval.shape[-1])
            vnodes = np.broadcast_to(np.arange(node_count), xval.shape[:-1]).reshape(-1)
            gp = SparseGP(256, graph=material_kernel(node_count)).fit(xx, yy, nodes)
            mean, variance = gp.predict(vv, vnodes)
            mean = mean.reshape(*xval.shape[:-1], 3)
            variance = variance.reshape(mean.shape)
        global_correction = np.einsum('ntvj,nij->ntvi', mean, vframe)
        np.savez_compressed(output / (mode + '_posterior.npz'),
                            canonical_mean=mean, canonical_variance=variance, frame=vframe)
        for shrinkage in (.25, .5, 1.0):
            prediction = vb.copy()
            prediction[:, :, 2:-2] += shrinkage * global_correction
            assert np.array_equal(prediction[:, :, :2], vb[:, :, :2])
            assert np.array_equal(prediction[:, :, -2:], vb[:, :, -2:])
            label = mode + '_s' + str(shrinkage)
            results[label] = metrics(prediction, vy, vb, ridge, vn)
            results[label]['fit_and_prediction_seconds'] = time.monotonic() - start
            print('GP_RESULT', label, json.dumps(results[label]), flush=True)
        (output / 'results.json').write_text(json.dumps({'protocol': provenance, 'results': results}, indent=2))
    print('FINAL_RESULTS', json.dumps(results), flush=True)
    print('DEVELOPMENT_TRIAL_COMPLETE official_eval_read=false source_test_read=false', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.output:
        run(args.output)
    else:
        parser.error('Supply --self-test or --output')
