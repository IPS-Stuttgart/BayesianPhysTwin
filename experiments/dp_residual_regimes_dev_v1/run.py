"""Grouped development screen; never opens official DEFORM evaluation data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn

from .model import ConditionalResidual, Spec

BLOCK = 6
SEED = 20260907
PINS = {
    "DLO4": ("1001cdb688072675eafdba34c8dc7f937be2fb7800c65d756e31214bc244fa17", "b858c2f4c0e107d367857dc2c4e17e753b6853e4593af64c4ce489564ae88fca", 0.01253017906857404),
    "DLO5": ("6bc72a4b3c7dae13c39181d28aebe9228fb1049fbe4bf26871b95fb518997a04", "a8f8b3253f2588eed3fb0e92a8812ce534dba857c82a3af9d3f0fe0dadab39b6", 0.00918328492025507),
}
FAMILIES = {
    "ridge": [Spec("ridge")],
    "single-gaussian": [Spec("gmm", 1)],
    "finite-gmm": [Spec("gmm", k) for k in (2, 4, 8)],
    "finite-bayes": [Spec("finite-bayes", 8, a) for a in (0.1, 1.0, 10.0)],
    "dp": [Spec("dp", 8, a) for a in (0.1, 1.0, 10.0)],
}
SHRINKAGES = (0.0, 0.25, 0.5, 1.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def direct_source(root: Path, directory: Path) -> None:
    """Reproduce the independently checksum-pinned source-cache export.

    Only the sixteen manifest-named train trajectories are read. Trusted legacy
    pickles are accepted only after their exact bytes match the pinned manifest.
    """
    import pickle
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features,
    )

    data = Path("/mnt/seagate10tb/florianpfaff/datasets/deform/data_set")
    receipt = {"contract": "source-cache-export-v1", "official_eval_access": False,
               "source_revision": "9d7383ea56a0a9e3ad6753d1c42fe653cd7e615d", "objects": {}}
    for dlo, (mh, ph, error) in PINS.items():
        folder = root / (dlo.lower() + "-source")
        mp, pp = folder / "source_manifest.json", folder / "source_predictions.npz"
        if digest(mp) != mh or digest(pp) != ph:
            raise ValueError("original source cache digest differs")
        manifest = json.loads(mp.read_text())
        with np.load(pp, allow_pickle=False) as z:
            names = list(map(str, z["names"]))
            hybrid, candidate = z["physical"].astype(float), z["candidate"].astype(float)
        if names != manifest["partitions"]["source_test"] or len(names) != 8:
            raise ValueError("source-only roster differs")
        records = []
        for name in names:
            if Path(name).name != name:
                raise ValueError("unsafe trajectory name")
            p = (data / dlo / "train" / name).resolve()
            if not p.is_relative_to((data / dlo / "train").resolve()):
                raise ValueError("trajectory escapes source train directory")
            identity = manifest["trajectories"][name]
            payload = p.read_bytes()
            if len(payload) != identity["size_bytes"] or hashlib.sha256(payload).hexdigest() != identity["sha256"]:
                raise ValueError("source trajectory digest differs")
            a = np.asarray(pickle.loads(payload), dtype=np.float32)
            if a.shape != (500, 3, 12) or not np.isfinite(a).all():
                raise ValueError("source trajectory shape or values differ")
            a = a.transpose(0, 2, 1).copy()
            a[:, :, 2] = np.clip(a[:, :, 2], 0.002001, 10000.0)
            records.append(a.astype(float))
        raw = np.stack(records)
        truth, initial, action = raw[:, 2:], raw[:, :2], raw[:, 2:, [0, 1, 10, 11]]
        if not np.isclose(np.mean(np.abs(hybrid - truth)), error, atol=1e-8, rtol=0):
            raise ValueError("source baseline reproduction failed")
        features, frames = build_deform_local_residual_features(initial, action, hybrid)
        # Adversarial suffix mutation cannot alter permitted predictor inputs.
        mutated = raw.copy()
        mutated[:, 2:, 2:-2] += 17.0
        f2, r2 = build_deform_local_residual_features(mutated[:, :2], mutated[:, 2:, [0, 1, 10, 11]], hybrid)
        if not np.array_equal(features, f2) or not np.array_equal(frames, r2):
            raise ValueError("future-free-node leakage detected")
        p = directory / (dlo + ".npz")
        np.savez_compressed(p, names=np.array(names), initial=initial, action=action,
                            hybrid=hybrid, legacy_candidate=candidate, truth=truth,
                            features=features, frames=frames)
        receipt["objects"][dlo] = {"file": p.name, "sha256": digest(p),
            "source_manifest_sha256": mh, "source_prediction_sha256": ph,
            "future_suffix_mutation_invariance": True}
    write(directory / "receipt.json", receipt)


def load(cache: Path) -> dict[str, object]:
    receipt = json.loads((cache / "receipt.json").read_text())
    if receipt["official_eval_access"] is not False:
        raise ValueError("cache is not source-only")
    cases, values, folds = [], [], []
    for dlo, (mh, ph, expected_l1) in PINS.items():
        entry = receipt["objects"][dlo]
        p = cache / entry["file"]
        if digest(p) != entry["sha256"] or entry["source_manifest_sha256"] != mh or entry["source_prediction_sha256"] != ph:
            raise ValueError("export provenance mismatch")
        with np.load(p, allow_pickle=False) as z:
            v = {k: z[k] for k in z.files}
        if v["hybrid"].shape != (8, 498, 12, 3) or v["features"].shape != (8, 498, 8, 92):
            raise ValueError("unexpected cache dimensions")
        if not np.isclose(np.abs(v["hybrid"] - v["truth"]).mean(), expected_l1, atol=1e-8, rtol=0):
            raise ValueError("baseline parity failed")
        if not np.array_equal(v["hybrid"][:, :, [0,1,10,11]], v["legacy_candidate"][:, :, [0,1,10,11]]):
            raise ValueError("legacy clamped baseline parity failed")
        hashes = []
        for j in range(8):
            h = hashlib.sha256()
            for key in ("initial", "action", "hybrid"):
                h.update(np.ascontiguousarray(v[key][j]).tobytes())
            hashes.append(h.hexdigest())
            cases.append({"dlo": dlo, "name": str(v["names"][j]), "query_sha256": hashes[-1]})
        unique = sorted(set(hashes))
        if len(unique) < 4:
            raise ValueError("too few distinct causal queries for four grouped folds")
        assign = {h: j % 4 for j, h in enumerate(unique)}
        folds.extend(assign[h] for h in hashes)
        values.append(v)
    v = {k: np.concatenate([x[k] for x in values]) for k in values[0] if k != "names"}
    n, horizon, nodes, _ = v["truth"].shape
    chunks = horizon // BLOCK
    features = v["features"].reshape(n, chunks, BLOCK, 8, 92)
    # One shared mode for all nodes in a six-frame future segment.
    x = np.concatenate([features[:, :, 0], features.mean(axis=2), features[:, :, -1]], axis=-1).reshape(n * chunks, -1)
    residual = v["truth"][:, :, 2:-2] - v["legacy_candidate"][:, :, 2:-2]
    canonical = np.einsum("ntvi,nij->ntvj", residual, v["frames"])
    y = canonical.reshape(n * chunks, -1)
    baseline = v["legacy_candidate"][:, :, 2:-2].reshape(n * chunks, -1)
    truth = v["truth"][:, :, 2:-2].reshape(n * chunks, -1)
    frames = np.repeat(v["frames"], chunks, axis=0)
    return {"receipt": receipt, "cases": cases, "folds": np.array(folds), "x": x, "y": y,
            "baseline": baseline, "truth": truth, "frames": frames, "arrays": v,
            "trajectory": np.repeat(np.arange(n), chunks), "chunks": chunks}


def evaluate(cache: Path, output: Path) -> dict[str, object]:
    data = load(cache)
    n = len(data["cases"])
    arrays, trajectory = data["arrays"], data["trajectory"]
    names = list(FAMILIES)
    predictions = {k: arrays["legacy_candidate"].copy() for k in names}
    raw_predictions = {k: arrays["legacy_candidate"].copy() for k in names}
    mean_predictions = {k: arrays["legacy_candidate"].copy() for k in names}
    hard_predictions = {k: arrays["legacy_candidate"].copy() for k in names}
    records = []
    protocol = {"contract": "dp-residual-regimes-development-v1", "development_only": True,
        "official_eval_access": False, "block_frames": BLOCK, "input_pca": 6,
        "output_pca": 8, "seed": SEED, "families": {k:[s.__dict__ for s in v] for k,v in FAMILIES.items()},
        "shrinkages": SHRINKAGES, "fold_assignment": data["folds"].tolist(),
        "cases": data["cases"], "fit_only_transforms": True,
        "outer_test_fold": "f", "inner_validation_fold": "(f+1)%4",
        "fit_folds": "remaining two", "all_models_fitted_to_residual_of": "frozen legacy candidate",
        "primary_readout": "world-coordinate marginal mixture median",
        "variance_inference": "plug-in variational mixture; not exact posterior predictive",
        "covariance_or_physical_regime_claim": False,
        "input_receipt": data["receipt"], "source_files": {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}}
    write(output / "protocol.json", protocol)
    for fold in range(4):
        tf = data["folds"][trajectory]
        fit = (tf != fold) & (tf != (fold + 1) % 4)
        val = tf == (fold + 1) % 4
        test = tf == fold
        test_ids = np.flatnonzero(data["folds"] == fold)
        fold_result = {"fold": fold, "fit_trajectories": sorted(set(trajectory[fit].tolist())),
            "validation_trajectories": sorted(set(trajectory[val].tolist())),
            "test_trajectories": test_ids.tolist(), "families": {}}
        for family, specs in FAMILIES.items():
            best = None
            options = []
            for spec in specs:
                start = time.monotonic()
                model = ConditionalResidual(spec, seed=SEED + fold).fit(data["x"][fit], data["y"][fit])
                validation = model.predict(data["x"][val], data["frames"][val])
                losses = [float(np.abs(data["baseline"][val] + s * validation["median"] - data["truth"][val]).mean()) for s in SHRINKAGES]
                # Nonconvergence is retained, but cannot supply a selected correction.
                chosen = int(np.argmin(losses)) if model.metadata["converged"] else 0
                option = {"spec": spec.__dict__, "name": spec.name, "shrinkage": SHRINKAGES[chosen],
                    "validation_l1_m": losses[chosen], "validation_curve_l1_m": losses,
                    "fit_metadata": model.metadata, "elapsed_seconds": time.monotonic() - start}
                options.append(option)
                if best is None or (losses[chosen], SHRINKAGES[chosen]) < (best[0], best[1]):
                    best = (losses[chosen], SHRINKAGES[chosen], model, option)
            _, shrink, selected, selected_info = best
            # Record the decision before accessing test outcomes in this fold.
            seal = {"fold": fold, "family": family, "selected": selected_info, "options": options}
            write(output / f"selection-fold{fold}-{family}.json", seal)
            test_prediction = selected.predict(data["x"][test], data["frames"][test])
            shape = (len(test_ids), 498, 8, 3)
            predictions[family][test_ids, :, 2:-2] += (shrink * test_prediction["median"]).reshape(shape)
            raw_predictions[family][test_ids, :, 2:-2] += test_prediction["median"].reshape(shape)
            mean_predictions[family][test_ids, :, 2:-2] += (shrink * test_prediction["mean"]).reshape(shape)
            hard_predictions[family][test_ids, :, 2:-2] += (shrink * test_prediction["hard_mean"]).reshape(shape)
            fold_result["families"][family] = {**seal, "test_average_max_input_weight": float(test_prediction["weights"].max(axis=1).mean())}
            print(f'FOLD {fold} {family}: selected {selected_info["name"]}, shrinkage={shrink}, active={selected.metadata["active_weight_gt_001"]}', flush=True)
        records.append(fold_result)
    all_predictions = {"hybrid": arrays["hybrid"], "legacy": arrays["legacy_candidate"], **predictions}
    scores = {}
    for name, prediction in all_predictions.items():
        errors = prediction - arrays["truth"]
        l1 = np.abs(errors).mean(axis=(1,2,3))
        scores[name] = {"all_node_l1_mm": float(l1.mean()*1000),
            "free_node_l1_mm": float(np.abs(errors[:,:,2:-2]).mean()*1000),
            "per_trajectory_l1_mm": (l1*1000).tolist(),
            "per_dlo_l1_mm": {d:float(l1[[c['dlo']==d for c in data['cases']]].mean()*1000) for d in PINS}}
        if name in names:
            scores[name]["unshrunk_selected_model_l1_mm"] = float(np.abs(raw_predictions[name]-arrays["truth"]).mean()*1000)
            scores[name]["soft_mean_l1_mm"] = float(np.abs(mean_predictions[name]-arrays["truth"]).mean()*1000)
            scores[name]["hard_mean_l1_mm"] = float(np.abs(hard_predictions[name]-arrays["truth"]).mean()*1000)
    contrasts = {}
    dp = np.array(scores['dp']['per_trajectory_l1_mm'])
    for name in ('legacy','ridge','single-gaussian','finite-gmm','finite-bayes'):
        ref = np.array(scores[name]['per_trajectory_l1_mm'])
        delta = dp-ref
        rng = np.random.default_rng(SEED)
        # Descriptive stratified trajectory bootstrap, not independent-object CI.
        samples = np.column_stack([rng.integers(0,8,(5000,8)),rng.integers(8,16,(5000,8))])
        interval = np.quantile(delta[samples].mean(axis=1),[.025,.975])
        contrasts[name] = {'delta_mm':float(delta.mean()),'relative_improvement_pct':float((1-dp.mean()/ref.mean())*100),
            'wins':int(np.sum(delta < -1e-10)), 'ties':int(np.sum(np.abs(delta)<=1e-10)),
            'descriptive_bootstrap_interval_mm':interval.tolist(), 'n_trajectories':n,
            'bootstrap_limitation':'two objects; overlapping training folds; descriptive only'}
    result = {"contract":protocol["contract"],"status":"completed", "development_only":True,
        "official_eval_access":False,"n_trajectories":n,"object_count":2,
        "n_nonoverlapping_blocks":len(data['x']),"scores":scores,"dp_contrasts":contrasts,
        "folds":records,"versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
        "github_run_id":os.environ.get('GITHUB_RUN_ID'), "protocol_sha256":digest(output/'protocol.json'),
        "interpretation":"Retrospective source-only grouped development; not fresh confirmation, calibrated uncertainty, or physical-state improvement."}
    write(output / "result.json",result)
    rows = ['# DP residual-regime development screen', '',
        'Sixteen already-opened DEFORM source trajectories, two DLOs, four grouped outer folds. No official evaluation trajectories read.', '',
        '| Method | DLO4 L1 (mm) | DLO5 L1 (mm) | Mean L1 (mm) |', '|---|---:|---:|---:|']
    for name,s in scores.items():
        rows.append(f'| {name} | {s["per_dlo_l1_mm"]["DLO4"]:.6f} | {s["per_dlo_l1_mm"]["DLO5"]:.6f} | {s["all_node_l1_mm"]:.6f} |')
    rows += ['', 'Each added model uses the same input-only features, fit-only low-rank transforms, and recorded training outcomes. Its shrinkage is selected on separate validation trajectories, including exact unchanged-candidate fallback at zero.', '',
        'This screen compares low-rank conditional residual readout distributions. It does not implement full switching physical dynamics, demonstrate calibration, or establish a general benefit of Dirichlet processes.', '',
        'The source-export receipt, code digests, fold assignments, all validation choices, convergence records, paired errors, and exact versions are retained in protocol.json and result.json.', '']
    (output/'report.md').write_text('\n'.join(rows))
    print('\n'.join(rows),flush=True)
    print('DP_CONTRASTS '+json.dumps(contrasts,sort_keys=True),flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-root',type=Path)
    source.add_argument('--cache-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'result.json').exists():
        raise ValueError('refusing to overwrite an existing scientific result')
    if args.cache_dir:
        evaluate(args.cache_dir,args.output)
    else:
        with tempfile.TemporaryDirectory(prefix='dp-source-only-') as temporary:
            directory=Path(temporary)
            direct_source(args.source_root,directory)
            evaluate(directory,args.output)


if __name__=='__main__':
    main()
