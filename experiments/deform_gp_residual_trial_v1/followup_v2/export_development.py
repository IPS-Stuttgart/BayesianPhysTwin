#!/usr/bin/env python3
"""Replay the unchanged v1 run and retain its fit/validation query tensors.

No source-test or official-evaluation access. The v1 runner retains all checksum,
read-boundary, historical-baseline and linear-limit parity checks. This wrapper
observes its two native rollouts without changing the simulator or predictions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT/'src'), str(ROOT/'scripts/remote')]
import run_development as original
import run_deform_dlo2_local_residual as runtime
import run_deform_dlo_local_residual as local_runtime
from bayesian_phystwin_experiments import deform_dlo_local_residual as legacy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    native = runtime._rollout
    retained = []

    def capture(state, trajectories, **kwargs):
        if len(retained) >= 2:
            raise RuntimeError('Unexpected extra rollout')
        result = native(state, trajectories, **kwargs)
        label, expected = [('fit', 40), ('validation', 8)][len(retained)]
        names = list(result['names'])
        if len(names) != expected:
            raise RuntimeError('Unexpected development cohort size')
        initial, action = local_runtime._causal_inputs(trajectories, names)
        baseline = np.asarray(result['predictions'], dtype=np.float64)
        target = np.asarray(result['targets'], dtype=np.float64)
        features, frames = legacy.build_deform_local_residual_features(initial, action, baseline)
        local_target = np.einsum('ntvi,nij->ntvj', target-baseline, frames)[:, :, 2:-2]
        path = output/f'{label}_queries.npz'
        np.savez_compressed(path, names=np.asarray(names), initial=initial, action=action,
                            baseline=baseline, target=target, frames=frames,
                            features=features, local_target=local_target)
        retained.append({'label': label, 'names': names, 'file': path.name,
                         'sha256': original.sha256(path), 'feature_shape': list(features.shape)})
        print('RETAINED_QUERY_CACHE', label, features.shape, flush=True)
        return result

    runtime._rollout = capture
    try:
        result = original.run(argparse.Namespace(repository=ROOT, output=output,
                              training_result=original.DEFAULT_TRAINING,
                              upstream_root=args.upstream_root, device='cuda:0'))
    finally:
        runtime._rollout = native
    if len(retained) != 2:
        raise RuntimeError('Expected exactly fit and validation query exports')
    original.write_json(output/'cache_manifest.json', {
        'contract': 'deform-gp-v2-development-query-cache-v1',
        'exporter_sha256': original.sha256(Path(__file__)), 'files': retained,
        'source_test_opened': False, 'official_eval_read': False,
        'historical_fit_validation_only': True,
        'parent_report_sha256': original.sha256(output/'report.json'),
        'mean_l1_mm': result['mean_l1_mm'],
        'warning': 'Fit-fold holdouts exclude residual-model fitting, not historical simulator training.'})


if __name__ == '__main__':
    main()
