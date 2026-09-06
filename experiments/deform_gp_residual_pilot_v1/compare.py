"""Matched exploratory GP mean comparison on original DLO2 training splits.

Not a new official test, not a Bayesian-uncertainty or physical-identification
claim. The checkpoint and features are shared; only the residual family changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from gp import SparseGP, self_test

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    _collapse_duplicate_queries,
    build_deform_local_residual_features,
    fit_deform_local_residual,
    predict_deform_local_residual,
)

CONFIG = {
    'contract': 'deform-gp-residual-pilot-v1',
    'primary_metric': 'equal-trajectory-all-node-coordinate-L1-metres',
    'fit_partition': 'original-DLO2-train-fit-40',
    'validation_partition': 'original-DLO2-train-validation-8',
    'held_partition': 'original-DLO2-train-source_test-8',
    'claim_boundary': 'Exploratory reuse of historical source splits, not fresh independent or official confirmation.',
    'lengths': [0.5, 1.0, 2.0],
    'noise_variances': [0.01, 0.1, 1.0],
    'shrinkages': [0.25, 0.5, 1.0],
    'ridge_penalties': [0.1, 1.0, 10.0],
    'material_lengths': [0.25, 1.0],
    'per_node_inducing_count': 128,
    'shared_inducing_count': 256,
    'seed': 20260907,
    'fit_labels_subsampled': False,
    'official_eval_read': False,
    'new_data_collection': False,
    'physical_checkpoint_refit': False,
    'uncertainty_value_evaluated': False,
    'output_coordinates': 'three-independent-GPs-in-action-local-frame',
}


def save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load(cache: Path, partition: str) -> dict:
    with np.load(cache / f'{partition}.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data['names'] = [str(x) for x in data['names']]
    return data


def fields(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, frames = build_deform_local_residual_features(
        data['initial'], data['action'], data['predictions'])
    residual = np.einsum('ntvi,nij->ntvj',
                         data['targets'] - data['predictions'], frames)[:, :, 2:-2]
    return features, frames, residual


def corrected(data: dict, frames: np.ndarray, mean: np.ndarray, shrinkage: float) -> np.ndarray:
    prediction = np.asarray(data['predictions'], dtype=np.float64).copy()
    prediction[:, :, 2:-2] += shrinkage * np.einsum('ntvj,nij->ntvi', mean, frames)
    if not np.array_equal(prediction[:, :, [0, 1, -2, -1]],
                          data['predictions'][:, :, [0, 1, -2, -1]]):
        raise AssertionError('clamped nodes changed')
    return prediction


def metrics(prediction: np.ndarray, data: dict) -> dict:
    error = np.abs(prediction - data['targets'])
    cases = error.mean(axis=(1, 2, 3))
    baseline = np.abs(data['predictions'] - data['targets']).mean(axis=(1, 2, 3))
    return {'mean_l1_mm': float(1000 * cases.mean()),
            'free_node_l1_mm': float(1000 * error[:, :, 2:-2].mean()),
            'case_l1_mm': dict(zip(data['names'], (1000 * cases).tolist())),
            'wins_vs_hybrid': int(np.sum(cases < baseline)),
            'relative_gain_vs_hybrid': float(1 - cases.mean() / baseline.mean()),
            'early_middle_late_l1_mm': [float(1000 * part.mean())
                                       for part in np.array_split(error, 3, axis=1)]}


def causal_keys(data: dict) -> set[str]:
    return {hashlib.sha256(b''.join(np.ascontiguousarray(data[k][i]).tobytes()
                                   for k in ('initial', 'action', 'predictions'))).hexdigest()
            for i in range(len(data['names']))}


class GPFamily:
    def __init__(self, features: np.ndarray, residual: np.ndarray,
                 length: float, arc_length: float | None):
        self.shared = arc_length is not None
        self.keep = np.ones(features.shape[-1], dtype=bool)
        if self.shared:
            self.keep[4:6] = False
        raw = features[..., self.keep]
        axes = (0, 1, 2) if self.shared else (0, 1)
        self.location = raw.mean(axis=axes)
        self.scale = raw.std(axis=axes)
        self.scale = np.where(self.scale > 1e-10, self.scale, 1.0)
        x = self.transform(features)
        if self.shared:
            self.models = [SparseGP.prepare(x.reshape(-1, x.shape[-1]), residual.reshape(-1, 3),
                length=length, arc_length=arc_length, inducing_count=CONFIG['shared_inducing_count'])]
        else:
            self.models = [SparseGP.prepare(x[:, :, node].reshape(-1, x.shape[-1]),
                residual[:, :, node].reshape(-1, 3), length=length,
                inducing_count=CONFIG['per_node_inducing_count']) for node in range(x.shape[2])]

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = (features[..., self.keep] - self.location) / self.scale
        if self.shared:
            arc = np.linspace(-1.0, 1.0, features.shape[2] + 4)[2:-2]
            arc = np.broadcast_to(arc[None, None, :, None], (*features.shape[:3], 1))
            x = np.concatenate((x, arc), axis=-1)
        return x

    def all_means(self, features: np.ndarray, noises: list[float]) -> dict[float, np.ndarray]:
        x = self.transform(features)
        shape = (*x.shape[:3], 3)
        result = {noise: np.empty(shape) for noise in noises}
        inputs = [x.reshape(-1, x.shape[-1])] if self.shared else [
            x[:, :, node].reshape(-1, x.shape[-1]) for node in range(x.shape[2])]
        for node, (model, values) in enumerate(zip(self.models, inputs)):
            weights = {noise: model.coefficients(noise)[0] for noise in noises}
            outputs = {noise: [] for noise in noises}
            for start in range(0, len(values), 2048):
                phi = model.features(values[start:start + 2048])
                for noise in noises:
                    outputs[noise].append((phi @ weights[noise]) * model.scale + model.location)
            for noise in noises:
                predicted = np.concatenate(outputs[noise])
                if self.shared:
                    result[noise] = predicted.reshape(shape)
                else:
                    result[noise][:, :, node] = predicted.reshape(*x.shape[:2], 3)
        return result


def main(cache: Path, output: Path) -> None:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'protocol.json', CONFIG)
    controls = self_test()
    save(output / 'numerical_controls.json', controls)
    fit, validation = load(cache, 'fit'), load(cache, 'validation')
    original_fit_count = len(fit['names'])
    if causal_keys(fit) & causal_keys(validation):
        raise ValueError('fit/validation causal-query overlap')
    initial, action, base, target, groups = _collapse_duplicate_queries(
        fit['initial'], fit['action'], fit['predictions'], fit['targets'], fit['names'])
    fit = {'initial': initial, 'action': action, 'predictions': base, 'targets': target,
           'names': [group[0] for group in groups]}
    xf, rf, yf = fields(fit)
    xv, rv, _ = fields(validation)
    records, selected, retained = [], {}, {}
    best_validation_predictions = {}

    def candidate(family: str, spec: dict, prediction: np.ndarray, model) -> None:
        score = metrics(prediction, validation)
        record = {'family': family, 'spec': spec, 'validation': score}
        records.append(record)
        if family not in selected or score['mean_l1_mm'] < selected[family]['validation']['mean_l1_mm']:
            selected[family] = record
            retained[family] = model
            best_validation_predictions[family] = prediction.copy()

    for ridge in CONFIG['ridge_penalties']:
        model = fit_deform_local_residual(fit['initial'], fit['action'], base, target,
            fit['names'], ridge=ridge, variance_floor_m2=1e-6)
        for shrinkage in CONFIG['shrinkages']:
            prediction = predict_deform_local_residual(model, validation['initial'], validation['action'],
                validation['predictions'], shrinkage=shrinkage)['predictions']
            spec = {'ridge': ridge, 'shrinkage': shrinkage}
            candidate('ridge_selected', spec, prediction, model)
            if ridge == 1.0 and shrinkage == 0.25:
                candidate('ridge_existing', spec, prediction, model)
    for family, material_lengths in [('gp_per_node', [None]),
                                     ('gp_material_shared', CONFIG['material_lengths'])]:
        for length in CONFIG['lengths']:
            for arc_length in material_lengths:
                mark = time.monotonic()
                model = GPFamily(xf, yf, length, arc_length)
                for noise, mean in model.all_means(xv, CONFIG['noise_variances']).items():
                    for shrinkage in CONFIG['shrinkages']:
                        prediction = corrected(validation, rv, mean, shrinkage)
                        candidate(family, {'length': length, 'material_length': arc_length,
                            'noise_variance': noise, 'shrinkage': shrinkage}, prediction, model)
                print('GP_FIT', family, length, arc_length, 'seconds', time.monotonic() - mark,
                      'best_validation_mm', selected[family]['validation']['mean_l1_mm'], flush=True)
                save(output / 'validation_candidates.json', {'records': records})
    best_family = min(('gp_per_node', 'gp_material_shared'),
                      key=lambda k: selected[k]['validation']['mean_l1_mm'])
    seal = {'selected': selected, 'gp_selected_family': best_family,
            'source_test_opened_by_this_comparison': False,
            'protocol_sha256': hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest(),
            'source_revision': os.environ.get('GITHUB_SHA')}
    save(output / 'selection_before_source_test.json', seal)
    print('SELECTION_SEALED', json.dumps(seal), flush=True)
    from export import export
    source_receipt = export(cache, ['source_test'])
    save(output / 'held_export.json', source_receipt)
    held = load(cache, 'source_test')
    if (causal_keys(fit) | causal_keys(validation)) & causal_keys(held):
        raise ValueError('development/source-test causal-query overlap')
    xh, rh, _ = fields(held)
    results = {'hybrid': metrics(held['predictions'], held),
               'persistence': metrics(held['persistence'], held)}
    predictions = {'hybrid': held['predictions']}
    for family, record in selected.items():
        spec, model = record['spec'], retained[family]
        if family.startswith('ridge'):
            prediction = predict_deform_local_residual(model, held['initial'], held['action'],
                held['predictions'], shrinkage=spec['shrinkage'])['predictions']
        else:
            mean = model.all_means(xh, [spec['noise_variance']])[spec['noise_variance']]
            prediction = corrected(held, rh, mean, spec['shrinkage'])
        results[family] = metrics(prediction, held)
        predictions[family] = prediction
    reference = np.array(list(results['ridge_existing']['case_l1_mm'].values()))
    tuned = np.array(list(results['ridge_selected']['case_l1_mm'].values()))
    rng = np.random.default_rng(CONFIG['seed'])
    bootstrap = rng.integers(0, len(reference), size=(10000, len(reference)))
    contrasts = {}
    for family in ('gp_per_node', 'gp_material_shared'):
        values = np.array(list(results[family]['case_l1_mm'].values()))
        difference = values - reference
        contrasts[family] = {'gain_vs_existing_ridge': float(1 - values.mean() / reference.mean()),
            'gain_vs_selected_ridge': float(1 - values.mean() / tuned.mean()),
            'wins_vs_existing_ridge': int(np.sum(values < reference)),
            'wins_vs_selected_ridge': int(np.sum(values < tuned)),
            'paired_mean_difference_mm': float(difference.mean()),
            'descriptive_trajectory_bootstrap_95_interval_mm': np.quantile(
                difference[bootstrap].mean(axis=1), [0.025, 0.975]).tolist()}
    result = {'contract': CONFIG['contract'], 'claim_boundary': CONFIG['claim_boundary'],
        'status': 'completed-exploratory-source-comparison',
        'source_revision': os.environ.get('GITHUB_SHA'), 'workflow_run_id': os.environ.get('GITHUB_RUN_ID'),
        'original_fit_count': original_fit_count, 'fit_causal_query_groups': len(groups),
        'validation_count': len(validation['names']), 'source_test_count': len(held['names']),
        'horizon': int(held['predictions'].shape[1]), 'node_count': int(held['predictions'].shape[2]),
        'gp_selected_family': best_family, 'selected': selected, 'held_results': results,
        'gp_vs_ridge': contrasts, 'numerical_controls': controls,
        'official_eval_read': False, 'new_data_collection': False,
        'uncertainty_value_evaluated': False, 'elapsed_seconds': time.monotonic() - started}
    save(output / 'result.json', result)
    np.savez_compressed(output / 'held_predictions.npz', names=np.asarray(held['names']),
                        targets=held['targets'], **predictions)
    lines = ['# DEFORM GP residual pilot v1', '', CONFIG['claim_boundary'], '',
             '| Method | Source-test coordinate L1 (mm) | Wins vs hybrid |', '|---|---:|---:|']
    for family, score in results.items():
        lines.append(f"| {family} | {score['mean_l1_mm']:.6f} | {score['wins_vs_hybrid']}/8 |")
    lines += ['', f'GP family chosen on validation: **{best_family}**.', '',
              'Checkpoint and original features are matched. Forty fit recordings, eight validation recordings,',
              'eight historical source-test recordings; no official evaluation or new robot experiment.',
              'This pilot tests predictive means, not a distinct Bayesian uncertainty advantage.', '',
              '```json', json.dumps(contrasts, indent=2), '```']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('RESULT_JSON', json.dumps(result, sort_keys=True), flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.cache, args.output)
