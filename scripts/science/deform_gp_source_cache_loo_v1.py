"""Retrospective source-only leave-one-trajectory-out GP comparison.

Uses the existing checksum-bound DLO4/DLO5 source cache, NOT official evaluation
trajectories. All variants and shrinkage factors are fixed in code. No model
selection or independent-confirmation claim is made from this development run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deform_gp_residual_development_v1 import SparseGP, digest, material_kernel

PINS = {
    'DLO4': 'cffcd511d5b4dca35d6955c9630ac75ddb00951304fb6fec3aa43fca72b796cd',
    'DLO5': '73cc8b06b631c252e9d5d5b6ff440975b1d3ad5feee9fed818f8ba295c20b37b',
}
SHRINKAGES = (.25, .5, 1.0)


def ridge_mean(x, y, query):
    """Same standardized per-node ridge mean, lambda=1, unpenalized intercept."""
    location, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    design = np.column_stack((np.ones(len(x)), (x - location) / scale))
    test = np.column_stack((np.ones(len(query)), (query - location) / scale))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    return test @ np.linalg.solve(design.T @ design + penalty, design.T @ y)


def run(cache: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'status': 'retrospective source-only development',
        'design': 'within-object leave-one-complete-trajectory-out; seven train, one predict',
        'objects': ['DLO4', 'DLO5'], 'case_count': 16, 'forecast_frames': 498,
        'source_cache_run': 34052556765, 'source_cache_sha256': PINS,
        'official_evaluation_access': False, 'new_robot_data': False,
        'feature_inputs': 'two initial states, known prescribed boundary motion, frozen hybrid rollout',
        'shrinkages': SHRINKAGES, 'ridge': 1., 'gp_noise_variance_ratio': .05,
        'gp_lengthscale': 'median training-feature distance',
        'gp_inducing_per_node': 128, 'gp_inducing_pooled': 256,
        'variants': ['ridge', 'gp_independent', 'gp_pooled_identity', 'gp_material_coupled'],
        'selection': 'none; report all predefined arms',
        'inference_claim': 'point prediction only, not covariance calibration or Bayesian superiority',
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = {}
    start = time.monotonic()
    for dlo, pin in PINS.items():
        path = cache / (dlo + '.npz')
        if digest(path) != pin:
            raise ValueError('Source cache checksum mismatch: ' + dlo)
        with np.load(path, allow_pickle=False) as z:
            x, frames, truth, baseline, legacy = (z[k] for k in
                ('features', 'frames', 'truth', 'hybrid', 'legacy_candidate'))
            names = tuple(map(str, z['names']))
        assert x.shape == (8, 498, 8, 92)
        target = np.einsum('ntvi,nij->ntvj', truth - baseline, frames)[:, :, 2:-2]
        # Refuse exact query duplicates split between folds.
        signatures = [x[i].tobytes() for i in range(8)]
        assert len(set(signatures)) == 8
        predictions = {'hybrid': baseline.copy(), 'frozen_legacy': legacy.copy()}
        for variant in protocol['variants']:
            for shrink in SHRINKAGES:
                predictions[variant + '_s' + str(shrink)] = baseline.copy()
        for held in range(8):
            train = np.arange(8) != held
            xx, yy, query = x[train], target[train], x[held:held + 1]
            mean = {'ridge': np.zeros((498, 8, 3)), 'gp_independent': np.zeros((498, 8, 3))}
            for node in range(8):
                tx = xx[:, :, node].reshape(-1, 92)
                ty = yy[:, :, node].reshape(-1, 3)
                qx = query[0, :, node]
                mean['ridge'][:, node] = ridge_mean(tx, ty, qx)
                gp = SparseGP(128, seed=7 + node).fit(tx, ty, np.zeros(len(tx), dtype=int))
                mean['gp_independent'][:, node] = gp.predict(qx, np.zeros(498, dtype=int))[0]
            tx, ty = xx.reshape(-1, 92), yy.reshape(-1, 3)
            nx = np.broadcast_to(np.arange(8), xx.shape[:-1]).reshape(-1)
            qx = query.reshape(-1, 92)
            nq = np.broadcast_to(np.arange(8), query.shape[:-1]).reshape(-1)
            for variant, graph in [('gp_pooled_identity', np.eye(8)),
                                   ('gp_material_coupled', material_kernel(8))]:
                gp = SparseGP(256, seed=7, graph=graph).fit(tx, ty, nx)
                mean[variant] = gp.predict(qx, nq)[0].reshape(498, 8, 3)
            for variant, correction in mean.items():
                global_correction = np.einsum('tvj,ij->tvi', correction, frames[held])
                for shrink in SHRINKAGES:
                    predictions[variant + '_s' + str(shrink)][held, :, 2:-2] += shrink * global_correction
            print('FOLD_DONE', dlo, held, names[held], 'elapsed_s', round(time.monotonic() - start, 2), flush=True)
        results[dlo] = {}
        for label, prediction in predictions.items():
            error = np.abs(prediction - truth)
            if label != 'frozen_legacy':
                assert np.array_equal(prediction[:, :, [0, 1, 10, 11]], baseline[:, :, [0, 1, 10, 11]])
            results[dlo][label] = {
                'mean_coordinate_l1_mm': float(error.mean() * 1000),
                'case_l1_mm': dict(zip(names, map(float, error.mean(axis=(1, 2, 3)) * 1000), strict=True)),
                'horizon_thirds_l1_mm': [float(error[:, s].mean() * 1000)
                    for s in np.array_split(np.arange(498), 3)],
            }
        np.savez_compressed(output / (dlo + '_predictions.npz'), truth=truth, names=np.array(names), **predictions)
        print('OBJECT_RESULTS', dlo, json.dumps(results[dlo]), flush=True)
        (output / 'results.json').write_text(json.dumps({'protocol': protocol, 'results': results}, indent=2))
    aggregate = {label: float(np.mean([results[d][label]['mean_coordinate_l1_mm'] for d in PINS]))
                 for label in results['DLO4']}
    (output / 'aggregate.json').write_text(json.dumps(aggregate, indent=2))
    print('AGGREGATE', json.dumps(aggregate), flush=True)
    print('COMPLETE_SECONDS', time.monotonic() - start, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)
