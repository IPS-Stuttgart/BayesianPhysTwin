"""Export only pinned historical DEFORM fit/validation trajectories for a DP pilot.

Never calls the historical source gate. Metadata and checkpoint identities must
match the protocol; no inferred cache equivalence or substituted checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts' / 'remote')]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def candidates(filename: str) -> list[Path]:
    roots = [Path(p) for p in (
        '/home/florianpfaff/source-only/deform-dlo2-local-residual-v5',
        '/home/github-runner/.cache/workflows/deform-dlo2-local-residual-v5',
        '/home/florianpfaff/source-only/deform-dlo2-local-residual-v6',
        '/home/github-runner/.cache/workflows/deform-dlo2-local-residual-v6')]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root):
            p = Path(current)
            dirs[:] = sorted(d for d in dirs if d not in (
                'eval', 'test', '.git', '__pycache__', 'site-packages', 'node_modules')
                and not any(x in d.lower() for x in ('held', 'confirm', 'target')))
            if len(p.relative_to(root).parts) >= 6:
                dirs[:] = []
            if filename in files:
                found.append(p / filename)
    return sorted(set(found))


def verified_path(filename: str, expected: str, audit: dict, out: Path) -> Path:
    records = []
    for path in candidates(filename):
        actual = digest(path)
        records.append(dict(path=str(path), sha256=actual, matches=actual == expected,
                            size=path.stat().st_size))
    audit[filename] = dict(expected_sha256=expected, candidates=records)
    write(out / 'identity_audit.json', audit)
    print('IDENTITY', json.dumps({filename: audit[filename]}), flush=True)
    for record in records:
        if record['matches']:
            return Path(record['path'])
    raise RuntimeError(f'No exact pinned {filename} found; no scientific data opened')


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT/'configs/sota/deform_dlo2_local_residual_v6.json').read_text())
    audit = dict(scope='historical development only', official_eval_read=False,
                 source_test_read=False, baseline_retrained=False)
    matched, failures = {}, []
    for key, name in [('training_result','training_validation_result.json'),
                      ('source_manifest','source_manifest.json'),
                      ('selected_checkpoint','update_6400.pt')]:
        try:
            matched[key] = verified_path(name, protocol[key]['sha256'], audit, out)
        except RuntimeError as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError('; '.join(failures))
    training = json.loads(matched['training_result'].read_text())
    manifest = json.loads(matched['source_manifest'].read_text())
    fit_names = list(manifest['split']['fit'])
    val_names = list(manifest['split']['validation'])
    names = fit_names + val_names
    forbidden = set(manifest['split']['source_test'])
    if len(fit_names) != 40 or len(val_names) != 8 or len(set(names)) != 48:
        raise ValueError('Expected disjoint pinned 40-fit/8-validation split')
    if set(names) & forbidden:
        raise ValueError('Development split overlaps source-test split')
    allowed = {Path(manifest['trajectories'][n]['path']).resolve() for n in names}

    def guard(event, event_args):
        if event != 'open' or not event_args:
            return
        raw = event_args[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(raw)).resolve()
        if p.suffix == '.pkl' and p not in allowed:
            raise PermissionError(f'Pilot forbids non-development pickle read: {p}')
        if 'eval' in p.parts and any(x.startswith('DLO') for x in p.parts):
            raise PermissionError(f'Pilot forbids official evaluation read: {p}')
    sys.addaudithook(guard)
    write(out/'data_boundary.json', dict(fit=fit_names, validation=val_names,
          source_test_read=False, official_eval_read=False, fresh_confirmation=False))
    import numpy as np
    import torch
    import run_deform_dlo_source as source
    import run_deform_dlo2_local_residual as runtime
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features, fit_deform_local_residual,
        predict_deform_local_residual, deform_causal_inputs)
    upstream = args.upstream.resolve()
    audit['upstream'] = source._assert_upstream(upstream, protocol['upstream']['commit'])
    source._seed_everything(torch, 42)
    modules = source._load_upstream(upstream)
    bundle = torch.load(matched['selected_checkpoint'], map_location='cpu', weights_only=True)
    state = bundle.get('model_state_dict')
    if not isinstance(state, dict):
        raise ValueError('Pinned checkpoint has no model_state_dict')
    trajectories = source._load_named_trajectories(manifest, names,
                                                  frame_count=500, node_count=12)
    prediction_parts, target_parts = [], []
    for start in range(0, len(names), 8):
        batch_names = names[start:start+8]
        started = time.monotonic()
        with torch.no_grad():
            result = runtime._rollout(state, {n: trajectories[n] for n in batch_names},
                                     modules=modules, torch=torch, device=args.device)
        if list(result['names']) != batch_names:
            raise ValueError('Rollout reordered trajectories')
        prediction_parts.append(np.asarray(result['predictions']))
        target_parts.append(np.asarray(result['targets']))
        print('ROLLOUT', batch_names, 'seconds', time.monotonic()-started, flush=True)
    baseline, targets = np.concatenate(prediction_parts), np.concatenate(target_parts)
    initial, action = deform_causal_inputs(np.stack([trajectories[n] for n in names]))
    features, frames = build_deform_local_residual_features(initial, action, baseline)
    residual = np.einsum('ntvi,nij->ntvj', targets-baseline, frames)[:, :, 2:-2]
    local = protocol['local_residual']
    ridge = fit_deform_local_residual(initial[:40], action[:40], baseline[:40],
             targets[:40], fit_names, ridge=float(local['fixed_arm']['ridge']),
             variance_floor_m2=float(local['coordinate_variance_floor_m2']))
    current = predict_deform_local_residual(ridge, initial, action, baseline,
                shrinkage=float(local['fixed_arm']['shrinkage']))['predictions']
    hashes = []
    for i in range(len(names)):
        h = hashlib.sha256()
        for array in (initial[i], action[i], baseline[i]):
            h.update(np.ascontiguousarray(array).tobytes())
        hashes.append(h.hexdigest())
    measured = np.mean(np.abs(baseline[40:]-targets[40:]), axis=(1,2,3))
    expected = float(training['selected_checkpoint']['validation_l1_m'])
    audit['baseline_reproduction'] = dict(expected_l1_m=expected,
        measured_l1_m=float(measured.mean()), tolerance_m=1e-7,
        passed=bool(abs(measured.mean()-expected) <= 1e-7))
    audit['shape'] = list(baseline.shape)
    audit['baseline_validation_per_case_mm'] = (1000*measured).tolist()
    audit['current_validation_per_case_mm'] = (1000*np.mean(
        np.abs(current[40:]-targets[40:]),axis=(1,2,3))).tolist()
    write(out/'identity_audit.json', audit)
    print('EXPORT_AUDIT', json.dumps(audit), flush=True)
    if not audit['baseline_reproduction']['passed']:
        raise RuntimeError('Pinned baseline reproduction failed; no DP scoring authorized')
    np.savez_compressed(out/'development.npz', features=features, residual=residual,
        baseline=baseline, targets=targets, frames=frames, current_predictions=current,
        names=np.asarray(names), roles=np.asarray(['fit']*40+['validation']*8),
        query_hashes=np.asarray(hashes))
    print('EXPORTED', str(out/'development.npz'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output/'failure.json', dict(error=type(exc).__name__, message=str(exc),
              traceback=traceback.format_exc(), status='technical failure; no DP result'))
        raise
