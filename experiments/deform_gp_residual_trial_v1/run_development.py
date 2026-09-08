#!/usr/bin/env python3
"""One fixed GP-vs-ridge DLO2 development comparison; never opens source/eval.

Run inside the existing, pinned CUDA/DEFORM environment. This entry point does
not install packages, train the DEFORM checkpoint, discover new datasets, or
promote any manuscript claim. The code below has unit coverage; its native
DEFORM integration is executed only by the explicit development request.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import traceback

import numpy as np
from gp import GPConfig, fit_gp

BASE_REVISION = '9d7383ea56a0a9e3ad6753d1c42fe653cd7e615d'
TRAINING_SHA256 = '1f8d092bc38b03f6cdd68ef38abcb7d403d914e38ba483698579deaeea8c2572'
MANIFEST_SHA256 = '7c5501997e6bab7b0537ef9cda932ec19312e40618f03a4fad80ffc1622a6d98'
CHECKPOINT_SHA256 = 'b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65'
DEFAULT_TRAINING = Path('/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/'
                        'train-8cc85de7/training_run/training_validation_result.json')
CONFIG = GPConfig()
SHRINKAGE = 0.25
VARIANCE_FLOOR = 1e-6
REFERENCE_VALIDATION_BASELINE_M = 0.00817261882312534


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require_identity(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f'pinned identity mismatch or missing file: {path}')


def trajectory_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read the native manifest's list of file identities, without opening data."""
    rows = manifest.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('native manifest must contain a nonempty files list')
    records = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get('name') or not row.get('path'):
            raise ValueError('invalid trajectory file identity')
        if row['name'] in records:
            raise ValueError('duplicate trajectory name in manifest')
        records[row['name']] = row
    return records


def split_names(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    if (manifest.get('dlo_type') != 'DLO2' or manifest.get('partition') != 'train'
            or manifest.get('official_eval_read') is not False):
        raise ValueError('only the existing DLO2 training manifest is allowed')
    splits = manifest['split']
    for key, count in (('fit', 40), ('validation', 8), ('source_test', 8)):
        if len(splits[key]) != count or len(set(splits[key])) != count:
            raise ValueError(f'expected {count} distinct {key} identities')
    sets = [set(splits[k]) for k in ('fit', 'validation', 'source_test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('overlapping historical partitions')
    identities = trajectory_records(manifest)
    paths = [str(Path(identities[name]['path']).resolve())
             for name in sum([list(splits[k]) for k in ('fit','validation','source_test')], [])]
    if len(set(paths)) != len(paths):
        raise ValueError('different trajectory names alias the same path')
    if any('eval' in Path(p).parts for p in paths):
        raise ValueError('a training identity resolves into official eval')
    return list(splits['fit']), list(splits['validation'])


def install_read_guard(manifest: dict[str, Any], allowed_names: list[str], upstream: Path) -> list[str]:
    """Reject source-test reads and any unlisted file under known data roots.

    This is a Python audit guard, not an OS sandbox. The reused trajectory loader
    is Python and is additionally called only with the explicit approved names.
    """
    identities = trajectory_records(manifest)
    all_paths = {Path(item['path']).resolve() for item in identities.values()}
    allowed = {Path(identities[name]['path']).resolve() for name in allowed_names}
    forbidden = all_paths - allowed
    roots = {p.parent for p in all_paths} | {(upstream/'data_set').resolve()}
    opened: list[str] = []

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p in forbidden:
            raise PermissionError(f'source-test access forbidden: {p}')
        in_data = any(p == root or root in p.parents for root in roots)
        if in_data and p not in allowed:
            raise PermissionError(f'unlisted dataset access forbidden: {p}')
        if p in allowed:
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR)):
                raise PermissionError('dataset writes are forbidden')
            opened.append(str(p))
    sys.addaudithook(audit)
    return opened


def correction_to_world(baseline: np.ndarray, residual_local: np.ndarray,
                        frames: np.ndarray, shrinkage: float) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=np.float64)
    residual_local = np.asarray(residual_local, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    if baseline.ndim != 4 or baseline.shape[-1] != 3 or baseline.shape[2] < 5:
        raise ValueError('baseline must be [trajectory, horizon, node, xyz]')
    if residual_local.shape != (*baseline.shape[:2], baseline.shape[2]-4, 3):
        raise ValueError('residual shape mismatch')
    if frames.shape != (baseline.shape[0], 3, 3):
        raise ValueError('frame shape mismatch')
    if not np.isfinite(baseline).all() or not np.isfinite(residual_local).all() or not np.isfinite(frames).all():
        raise ValueError('nonfinite prediction input')
    if not 0 <= shrinkage <= 1:
        raise ValueError('invalid shrinkage')
    result = baseline.copy()
    result[:, :, 2:-2] += shrinkage*np.einsum('ntvj,nij->ntvi', residual_local, frames)
    return result


def query_keys(initial: np.ndarray, action: np.ndarray, baseline: np.ndarray) -> list[str]:
    return [hashlib.sha256(b''.join(np.ascontiguousarray(a[i]).tobytes()
                                  for a in (initial, action, baseline))).hexdigest()
            for i in range(len(initial))]


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository.resolve()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be empty; never overwrite a result')
    if 'source-only' in out.parts or 'data_set' in out.parts:
        raise ValueError('outputs may not be written into retained source/data directories')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    preflight = {'status': 'preflight', 'scope': 'DLO2 historical fit/validation development only',
                 'gp': asdict(CONFIG), 'shrinkage': SHRINKAGE, 'base_revision': BASE_REVISION,
                 'source_test_opened': False, 'official_eval_read': False,
                 'new_robot_experiments': False, 'claim_promotion': False,
                 'runner_sha256': sha256(Path(__file__)),
                 'gp_sha256': sha256(Path(__file__).with_name('gp.py'))}
    write_json(out/'preflight.json', preflight)
    # Enforce unchanged shared implementation; the experiment itself lives elsewhere.
    changed = subprocess.check_output(['git','-C',str(root),'diff',BASE_REVISION,'--',
                                      'src','scripts/remote','configs/sota'], text=True)
    if changed.strip():
        raise RuntimeError('shared source differs from the inspected baseline revision')
    sys.path[:0] = [str(root/'src'), str(root/'scripts/remote')]
    import run_deform_dlo2_local_residual as runtime
    import run_deform_dlo_local_residual as local_runtime
    import run_deform_dlo_source as source_runtime
    from bayesian_phystwin_experiments import deform_dlo_local_residual as legacy

    training_path = args.training_result.resolve()
    require_identity(training_path, TRAINING_SHA256)
    protocol_path = root/'configs/sota/deform_dlo2_local_residual_v5.json'
    protocol = legacy.load_deform_dlo2_local_residual_protocol(protocol_path)
    training, manifest, manifest_path = runtime._verify_training_result(
        training_path, protocol=protocol, protocol_path=protocol_path)
    require_identity(manifest_path, MANIFEST_SHA256)
    fit_names, val_names = split_names(manifest)
    opened = install_read_guard(manifest, fit_names+val_names, args.upstream_root)
    upstream_revision = protocol['upstream']['commit']
    if manifest['upstream']['commit'] != upstream_revision:
        raise ValueError('manifest and protocol disagree on upstream revision')
    source_runtime._assert_upstream(args.upstream_root, upstream_revision)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    source_runtime._seed_everything(torch, 42)
    modules = source_runtime._load_upstream(args.upstream_root)
    # Verify the fixed 6400-update checkpoint *before* torch loads its bytes.
    checkpoint_path = Path(training['selected_checkpoint']['checkpoint']['path']).resolve()
    require_identity(checkpoint_path, CHECKPOINT_SHA256)
    state, loaded_path = runtime._checkpoint_state(training, torch=torch)
    if checkpoint_path != loaded_path:
        raise ValueError('unexpected checkpoint routing')
    preflight.update({'training_result_sha256': sha256(training_path),
                      'manifest_sha256': sha256(manifest_path),
                      'checkpoint_sha256': sha256(checkpoint_path),
                      'fit_names': fit_names, 'validation_names': val_names,
                      'python': sys.version, 'torch': torch.__version__,
                      'cuda': torch.version.cuda, 'device': args.device,
                      'upstream_revision': upstream_revision,
                      'historical_selected_checkpoint_metadata': training['selected_checkpoint']})
    write_json(out/'preflight.json', preflight)

    def load_rollout(names: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        print(f'Loading {len(names)} approved development trajectories', flush=True)
        trajectories = source_runtime._load_named_trajectories(
            manifest, names, frame_count=500, node_count=12)
        print(f'Replaying frozen checkpoint for {len(names)} trajectories', flush=True)
        rollout = runtime._rollout(state, trajectories, modules=modules,
                                   torch=torch, device=args.device)
        if set(rollout['names']) != set(names):
            raise ValueError('rollout identity mismatch')
        initial, action = local_runtime._causal_inputs(trajectories, list(rollout['names']))
        return initial, action, rollout

    fit_initial, fit_action, fit_rollout = load_rollout(fit_names)
    fit_names = list(fit_rollout['names'])
    base_fit = np.asarray(fit_rollout['predictions'], dtype=np.float64)
    target_fit = np.asarray(fit_rollout['targets'], dtype=np.float64)
    ridge_model = legacy.fit_deform_local_residual(
        fit_initial, fit_action, base_fit, target_fit, fit_names,
        ridge=1.0, variance_floor_m2=VARIANCE_FLOOR)
    ci, ca, cb, ct, clusters = legacy._collapse_duplicate_queries(
        fit_initial, fit_action, base_fit, target_fit, fit_names)
    features, frames = legacy.build_deform_local_residual_features(ci, ca, cb)
    residual = np.einsum('ntvi,nij->ntvj', ct-cb, frames)[:, :, 2:-2]
    groups = np.repeat(np.arange(len(ci)), cb.shape[1])
    nonlinear_models, linear_models = [], []
    for node in range(features.shape[2]):
        x = features[:, :, node].reshape(-1, features.shape[3])
        y = residual[:, :, node].reshape(-1, 3)
        linear_models.append(fit_gp(x, y, groups, replace(CONFIG, amplitude=0.0)))
        nonlinear_models.append(fit_gp(x, y, groups, CONFIG))
        print(f'Fitted node {node+1}/{features.shape[2]}', flush=True)
    np.savez_compressed(out/'gp_models.npz', **{
        f'node_{i}_{key}': value for i, model in enumerate(nonlinear_models)
        for key, value in model.arrays().items()})
    write_json(out/'fit_seal.json', {'gp': asdict(CONFIG), 'shrinkage': SHRINKAGE,
                                   'trajectory_clusters': [list(g) for g in clusters],
                                   'model_sha256': sha256(out/'gp_models.npz'),
                                   'validation_used_for_fit_or_selection': False})
    # No GP tuning on the validation panel; one fixed nonlinear candidate.
    vi, va, val = load_rollout(val_names)
    val_names = list(val['names'])
    base = np.asarray(val['predictions'], dtype=np.float64)
    if set(query_keys(ci,ca,cb)) & set(query_keys(vi,va,base)):
        raise ValueError('an exact causal query is shared across fit/validation')
    vf, vr = legacy.build_deform_local_residual_features(vi,va,base)

    def predict(models: list[Any]) -> np.ndarray:
        local = np.stack([model.predict(vf[:,:,i].reshape(-1,vf.shape[3])).reshape(
            *vf.shape[:2],3) for i,model in enumerate(models)],axis=2)
        return correction_to_world(base, local, vr, SHRINKAGE)
    gp_prediction = predict(nonlinear_models)
    linear_prediction = predict(linear_models)
    ridge_prediction = legacy.predict_deform_local_residual(
        ridge_model,vi,va,base,shrinkage=SHRINKAGE)['predictions']
    parity = float(np.max(np.abs(linear_prediction-ridge_prediction)))
    if parity > 1e-9:
        raise AssertionError(f'linear GP/repository ridge mean parity failed: {parity} m')
    if not np.array_equal(gp_prediction[:,:,[0,1,-2,-1]],base[:,:,[0,1,-2,-1]]):
        raise AssertionError('clamped baseline coordinates changed')
    # Seal predictions before scoring or any outcome-dependent decision.
    np.savez_compressed(out/'validation_predictions.npz', names=np.asarray(val_names),
                        baseline=base, ridge=ridge_prediction, gp=gp_prediction)
    target = np.asarray(val['targets'], dtype=np.float64)
    predictions = {'baseline':base,'ridge':ridge_prediction,'gp':gp_prediction}
    errors = {key: np.mean(np.abs(p-target),axis=(1,2,3)) for key,p in predictions.items()}
    if abs(float(errors['baseline'].mean())-REFERENCE_VALIDATION_BASELINE_M)>1e-6:
        raise AssertionError('historical baseline reproduction differs by more than 1 micrometre')
    rows = []
    for i,name in enumerate(val_names):
        rows.append({'trajectory':name, **{f'{k}_l1_mm':float(e[i]*1000) for k,e in errors.items()},
                     'gp_minus_ridge_mm':float((errors['gp'][i]-errors['ridge'][i])*1000)})
    with (out/'trajectory_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    horizon_rows=[]
    for h in range(base.shape[1]):
        horizon_rows.append({'step':h+1,**{f'{k}_l1_mm':float(np.mean(np.abs(p[:,h]-target[:,h]))*1000)
                                        for k,p in predictions.items()}})
    with (out/'horizon_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(horizon_rows[0]));writer.writeheader();writer.writerows(horizon_rows)
    delta = errors['gp']-errors['ridge']
    # Cluster exact duplicate validation queries before descriptive bootstrap.
    keys=query_keys(vi,va,base)
    cluster_delta=np.asarray([np.mean(delta[np.asarray(keys)==key]) for key in sorted(set(keys))])
    rng=np.random.default_rng(20260907)
    draws=rng.choice(cluster_delta,size=(5000,len(cluster_delta)),replace=True).mean(axis=1)*1000
    result={**preflight,'status':'completed-development-comparison',
            'validation_names':val_names,'linear_mean_max_abs_difference_m':parity,
            'mean_l1_mm':{k:float(v.mean()*1000) for k,v in errors.items()},
            'gp_improvement_over_ridge_percent':float(100*(1-errors['gp'].mean()/errors['ridge'].mean())),
            'gp_wins_vs_ridge':int(np.sum(delta<0)), 'trajectory_count':len(delta),
            'validation_causal_query_clusters':len(cluster_delta),
            'descriptive_cluster_bootstrap_gp_minus_ridge_mm':np.quantile(draws,[.025,.975]).tolist(),
            'prediction_sha256':sha256(out/'validation_predictions.npz'),
            'opened_trajectory_paths':sorted(set(opened)),
            'elapsed_seconds':time.perf_counter()-started,
            'gp_parameters_selected_on_validation':False,
            'uncertainty_calibration_tested':False,
            'graph_coupling_tested':False,
            'interpretation':'Historical development comparison, not fresh confirmation or a uniquely Bayesian gain.'}
    write_json(out/'report.json',result)
    text='# DEFORM GP residual development result\n\n'+result['interpretation']+'\n\n'
    text+='| Method | Mean coordinate L1 (mm) |\n|---|---:|\n'
    text+=''.join(f'| {k} | {v:.6f} |\n' for k,v in result['mean_l1_mm'].items())
    text+=f"\nGP wins against ridge: {result['gp_wins_vs_ridge']}/{len(delta)}.\n"
    text+='\nOfficial eval and historical source-test were not opened. No calibration or graph-coupling claim.\n'
    (out/'RESULTS.md').write_text(text)
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--training-result',type=Path,default=DEFAULT_TRAINING)
    parser.add_argument('--upstream-root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        # Do not turn a failed integration into a scientific negative.
        traceback.print_exc()
        print(f'TECHNICAL FAILURE; NO EMPIRICAL CONCLUSION: {type(exc).__name__}: {exc}',file=sys.stderr)
        if args.output.is_dir():
            write_json(args.output/'technical_failure.json', {'type':type(exc).__name__, 'message':str(exc)})
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
