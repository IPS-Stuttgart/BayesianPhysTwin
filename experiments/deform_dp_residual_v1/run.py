"""Retrospective DP residual-expert screen; no official DEFORM evaluation reads.

Joint variational Gaussian mixtures induce input-conditioned affine residual
experts. Conditional prediction uses plug-in variational moments, not a fully
integrated posterior. The unchanged reference is a ridge surrogate, not a
physical DEFORM checkpoint. All selection precedes held-source scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

import reference as ref

CONFIG = {
    "contract": "deform-dp-residual-screen-v1",
    "reference_blob": "57f9aff414f39f52a4ef346e525f40342554e75e",
    "split_domain": ref.CONFIG["split_domain"],
    "dlos": ["DLO4", "DLO5"], "fit": 32, "validation": 12, "test": 12,
    "origins": [25, 100, 200, 300, 400], "horizons": [5, 20, 50],
    "input_rank": 6, "residual_rank": 6,
    "component_caps": [2, 4, 8], "concentrations": [0.1, 1.0, 10.0],
    "shrinkages": [0.0, 0.25, 0.5, 1.0],
    "rbf_gamma": [0.125, 0.5, 2.0], "rbf_ridge": [0.1, 1.0, 10.0],
    "mixture_mean_precision": 5.0, "mixture_covariance_regularizer": 1e-5,
    "n_init": 3, "max_iter": 1000, "tol": 1e-5,
    "seed": 20260907, "bootstrap_replicates": 10000,
    "feature_std_floor": 1e-6, "pca_singular_floor": 1e-8,
    "official_eval_access": False, "new_acquisition": False,
    "primary": "trajectory-mean coordinate L1; DP must beat ridge, finite mixture and RBF with paired 95% upper bounds <0 and >=1% mean gain over ridge",
    "scope": "Retrospective DLO4/DLO5 source-only surrogate screen, not the physical checkpoint, a sticky HDP, Prob4D integration, or independent confirmation.",
}
ARMS = ("persistence", "boundary_interpolation", "reference_ridge", "single_expert",
        "finite_mixture", "dp_mixture", "rbf_residual")


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def dump(path, obj):
    Path(path).write_text(json.dumps(encode(obj), sort_keys=True, indent=2,
                                   allow_nan=False) + "\n")


def compress_fit(a, rank):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or len(a) < 2:
        raise ValueError("Expected finite sample matrix")
    center = a.mean(0)
    scale = np.maximum(a.std(0), CONFIG["feature_std_floor"])
    z = (a-center)/scale
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    rank = min(rank, int(np.sum(singular > CONFIG["pca_singular_floor"])))
    if rank == 0:
        # Explicit constant-data representation, not a numerical rescue.
        return dict(center=center, scale=scale, basis=np.zeros((a.shape[1], 1)),
                    inverse=np.zeros((1, a.shape[1])))
    sd = singular[:rank]/np.sqrt(len(a)-1)
    return dict(center=center, scale=scale, basis=vt[:rank].T/sd,
                inverse=sd[:, None]*vt[:rank])


def project(model, a):
    return ((np.asarray(a)-model["center"])/model["scale"])@model["basis"]


def decode(model, z):
    return (np.asarray(z)@model["inverse"])*model["scale"]+model["center"]


def fit_mixture(x, y, kind, cap, alpha):
    joint = np.column_stack([x, y])
    d = joint.shape[1]
    prior_kind = "dirichlet_process" if kind == "dp" else "dirichlet_distribution"
    estimator = BayesianGaussianMixture(
        n_components=cap, covariance_type="full", n_init=CONFIG["n_init"],
        max_iter=CONFIG["max_iter"], tol=CONFIG["tol"], random_state=CONFIG["seed"],
        reg_covar=CONFIG["mixture_covariance_regularizer"],
        weight_concentration_prior_type=prior_kind,
        weight_concentration_prior=alpha if kind == "dp" else alpha/cap,
        mean_precision_prior=CONFIG["mixture_mean_precision"],
        mean_prior=np.zeros(d), degrees_of_freedom_prior=d+2,
        covariance_prior=np.eye(d),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(joint)
    if not estimator.converged_:
        raise RuntimeError(f"Mixture did not converge after {estimator.n_iter_} iterations")
    return dict(kind="mixture", weights=estimator.weights_, means=estimator.means_,
                covariances=estimator.covariances_, nx=x.shape[1], cap=cap,
                prior=prior_kind, concentration=alpha, iterations=estimator.n_iter_,
                training_bound=float(estimator.lower_bound_),
                warnings=[str(w.message) for w in caught],
                effective_components=int(np.sum(estimator.weights_ > .01)))


def conditional(model, x):
    """Only input covariates enter gates: no future residual is an argument."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if not np.isfinite(x).all() or x.shape[1] != model["nx"]:
        raise ValueError("Invalid conditioning inputs")
    nx = model["nx"]
    logs, means = [], []
    for weight, mu, cov in zip(model["weights"], model["means"],
                               model["covariances"], strict=True):
        xx, xy = cov[:nx, :nx], cov[:nx, nx:]
        chol = np.linalg.cholesky(xx)
        delta = x-mu[:nx]
        standardized = np.linalg.solve(chol, delta.T).T
        log_density = -.5*(nx*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()
                            +np.sum(standardized**2, axis=1))
        logs.append(np.log(weight)+log_density)
        means.append(mu[nx:]+delta@np.linalg.solve(xx, xy))
    log_weights = np.stack(logs, axis=1)
    weights = np.exp(log_weights-logsumexp(log_weights, axis=1)[:, None])
    prediction = np.einsum("nk,nkd->nd", weights, np.stack(means, axis=1))
    return prediction, weights


def kernel(x, z, gamma):
    return np.exp(-gamma*np.sum((x[:, None]-z[None, :])**2, axis=2)/x.shape[1])


def correction(model, x):
    if model["kind"] == "mixture":
        return conditional(model, x)[0]
    if model["kind"] == "zero":
        return np.zeros((len(x), model["ny"]))
    return kernel(x, model["x"], model["gamma"])@model["coef"]


def select_family(x, y, xv, validation_error, output_transform, family):
    """Exactly nine structures each for finite, DP and RBF; common shrinkage grid."""
    records, selected, best = [], None, np.inf
    if family == "single_expert":
        settings = [(1, 1.)]
    elif family == "rbf_residual":
        settings = [(g, a) for g in CONFIG["rbf_gamma"] for a in CONFIG["rbf_ridge"]]
    else:
        settings = [(k, a) for k in CONFIG["component_caps"] for a in CONFIG["concentrations"]]
    for first, second in settings:
        try:
            if family == "rbf_residual":
                model = dict(kind="rbf", x=x, gamma=first,
                             coef=np.linalg.solve(kernel(x, x, first)+second*np.eye(len(x)), y))
            else:
                model = fit_mixture(x, y, "dp" if family == "dp_mixture" else "finite",
                                    first, second)
            raw = decode(output_transform, correction(model, xv))
            for shrinkage in CONFIG["shrinkages"]:
                loss = float(np.mean(np.abs(validation_error-shrinkage*raw)))
                records.append(dict(setting=[first, second], shrinkage=shrinkage,
                                    validation_l1_m=loss, status="ok"))
                if loss < best:
                    best = loss
                    selected = dict(model=model, shrinkage=shrinkage,
                                    setting=[first, second], validation_l1_m=loss)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as error:
            records.append(dict(setting=[first, second], status="failed", reason=str(error)))
    if selected is None:
        selected = dict(model=dict(kind="zero", ny=y.shape[1]), shrinkage=0.,
                        setting=None, validation_l1_m=float(np.mean(np.abs(validation_error))))
    return selected, records


def predict_family(selected, transform, x, baseline):
    if selected["shrinkage"] == 0.:
        return np.asarray(baseline).copy()  # Byte-exact fallback, no model evaluation.
    return baseline+selected["shrinkage"]*decode(transform, correction(selected["model"], x))


def assemble(arrays, horizon, models, outcomes):
    features, baseline, truth, simple, current = [], [], [], [], []
    for a in arrays:
        for origin in CONFIG["origins"]:
            f, base, now = ref.inputs(a, origin, horizon)
            model = models[origin]
            features.append(f)
            baseline.append(base+ref.transform(model, f)[0]@model["w"])
            simple.append(base)
            current.append(now)
            if outcomes:
                truth.append(a[origin+horizon, 2:10].ravel())
    return dict(features=np.array(features), baseline=np.array(baseline),
                simple=np.array(simple), current=np.array(current),
                truth=np.array(truth) if outcomes else None)


def bootstrap_delta(candidate, comparator):
    delta = np.asarray(candidate)-np.asarray(comparator)
    if delta.shape != (24,):
        raise ValueError("Expected twelve complete trajectories per fixed object")
    rng = np.random.default_rng(CONFIG["seed"])
    indices = np.column_stack([rng.integers(i, i+12, (CONFIG["bootstrap_replicates"], 12))
                               for i in (0, 12)])
    return dict(difference_mm=float(delta.mean()*1000),
                ci95_mm=(1000*np.quantile(delta[indices].mean(1), [.025, .975])).tolist(),
                wins=int(np.sum(delta < 0)), ties=int(np.sum(delta == 0)))


def run(root, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    dump(out/"protocol.json", CONFIG)
    code_hashes = {p.name: ref.digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    dump(out/"code_identity.json", code_hashes)
    manifest, loaded, paths_by_dlo = {}, {}, {}
    for dlo in CONFIG["dlos"]:
        fit, val, test = ref.split_paths(root, dlo)
        manifest[dlo] = {part: [dict(name=p.name, sha256=ref.digest(p)) for p in paths]
                         for part, paths in (("fit", fit), ("calibration", val), ("source_test", test))}
        hashes = [v["sha256"] for rows in manifest[dlo].values() for v in rows]
        if len(set(hashes)) != 56:
            raise ValueError("Duplicate file contents within source partitions")
        loaded[dlo] = ([ref.load(p) for p in fit], [ref.load(p) for p in val])
        paths_by_dlo[dlo] = test
    dump(out/"input_manifest.json", manifest)
    if ref.digest(out/"input_manifest.json") != "9a1ef8d257f4150df842feecc2d45b579eb49a1276e0b023d670221084c0646f":
        raise ValueError("Source bytes or splits differ from the previous run")
    all_models, validation_records = {}, {}
    for dlo in CONFIG["dlos"]:
        fit, val = loaded[dlo]
        for horizon in CONFIG["horizons"]:
            reference_models = {}
            for origin in CONFIG["origins"]:
                ins = [ref.inputs(a, origin, horizon) for a in fit]
                f = np.stack([r[0] for r in ins])
                residual = np.stack([a[origin+horizon, 2:10].ravel()-r[1]
                                     for a, r in zip(fit, ins, strict=True)])
                reference_models[origin] = ref.fit_model(f, residual)
            train = assemble(fit, horizon, reference_models, True)
            validation = assemble(val, horizon, reference_models, True)
            xp = compress_fit(train["features"], CONFIG["input_rank"])
            yp = compress_fit(train["truth"]-train["baseline"], CONFIG["residual_rank"])
            x, y = project(xp, train["features"]), project(yp, train["truth"]-train["baseline"])
            xv = project(xp, validation["features"])
            chosen, records = {}, {}
            for family in ARMS[3:]:
                chosen[family], records[family] = select_family(
                    x, y, xv, validation["truth"]-validation["baseline"], yp, family)
            key = f"{dlo}-h{horizon}"
            all_models[key] = dict(reference=reference_models, xp=xp, yp=yp, selected=chosen)
            validation_records[key] = records
            print(key, {k: (v["setting"], v["shrinkage"], round(v["validation_l1_m"]*1000, 4))
                        for k, v in chosen.items()}, flush=True)
    dump(out/"frozen_models.json", all_models)
    dump(out/"validation_records.json", validation_records)
    predictions, scoring = {}, {}
    # Full pickles contain future bytes; only allowed context reaches inference.
    for dlo, paths in paths_by_dlo.items():
        arrays = [ref.load(p) for p in paths]
        for horizon in CONFIG["horizons"]:
            key = f"{dlo}-h{horizon}"
            m = all_models[key]
            data = assemble(arrays, horizon, m["reference"], False)
            x = project(m["xp"], data["features"])
            predictions[key] = {"persistence": data["current"],
                                "boundary_interpolation": data["simple"],
                                "reference_ridge": data["baseline"]}
            for family in ARMS[3:]:
                predictions[key][family] = predict_family(m["selected"][family], m["yp"], x, data["baseline"])
    dump(out/"predictions.json", predictions)
    dump(out/"prediction_seal.json", {"code_sha256": code_hashes,
         **{p: ref.digest(out/p) for p in ("protocol.json", "input_manifest.json", "frozen_models.json", "predictions.json")},
         "future_internal_values_passed_to_inference": False, "official_eval_files_opened": 0})
    print("Predictions sealed. Held-source scoring starts now.", flush=True)
    losses = {a: [] for a in ARMS}
    squared = {a: [] for a in ARMS}
    rows = []
    for dlo, paths in paths_by_dlo.items():
        truth = {h: np.stack([ref.load(p)[o+h, 2:10].ravel() for p in paths for o in CONFIG["origins"]])
                 for h in CONFIG["horizons"]}
        for arm in ARMS:
            error = np.stack([truth[h]-predictions[f"{dlo}-h{h}"][arm] for h in CONFIG["horizons"]])
            e = error.reshape(3, 12, 5, 24).transpose(1, 0, 2, 3)
            l1 = np.abs(e).mean(axis=(1, 2, 3))
            mse = (e**2).mean(axis=(1, 2, 3))
            losses[arm].extend(l1.tolist())
            squared[arm].extend(mse.tolist())
            for i, p in enumerate(paths):
                rows.append(dict(dlo=dlo, trajectory=p.name, arm=arm,
                                 l1_mm=1000*l1[i], rmse_mm=1000*np.sqrt(mse[i])))
        scoring[dlo] = {arm: float(np.mean(losses[arm][-12:])*1000) for arm in ARMS}
    result = dict(config=CONFIG, per_object_l1_mm=scoring,
                  aggregate={a: dict(l1_mm=1000*np.mean(losses[a]),
                                     rmse_mm=1000*np.sqrt(np.mean(squared[a]))) for a in ARMS},
                  dp_minus_comparator={a: bootstrap_delta(losses["dp_mixture"], losses[a]) for a in ARMS if a != "dp_mixture"},
                  accounting=dict(objects=2, fit_recordings=64, validation_recordings=24,
                                  held_source_recordings=24, forecast_contexts=360, official_eval_files=0),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, sklearn=sklearn.__version__, runner=os.environ.get("RUNNER_NAME", "local-session")))
    gain = 1-result["aggregate"]["dp_mixture"]["l1_mm"]/result["aggregate"]["reference_ridge"]["l1_mm"]
    result["relative_gain_over_ridge"] = gain
    result["primary_supported"] = bool(gain >= .01 and all(
        result["dp_minus_comparator"][a]["ci95_mm"][1] < 0
        for a in ("reference_ridge", "finite_mixture", "rbf_residual")))
    dump(out/"trajectory_scores.json", rows)
    dump(out/"result.json", result)
    text = ["# DP residual screen: real-data result", "", CONFIG["scope"], "",
            f"Primary criterion supported: {result['primary_supported']}", "",
            "| Method | Coordinate L1 (mm) | RMSE (mm) |", "|---|---:|---:|"]
    text += [f"| {a} | {v['l1_mm']:.6f} | {v['rmse_mm']:.6f} |" for a, v in result["aggregate"].items()]
    text += ["", "DP-minus-comparator complete-trajectory bootstrap:", "",
             json.dumps(encode(result["dp_minus_comparator"]), indent=2)]
    (out/"SUMMARY.md").write_text("\n".join(text)+"\n")
    print((out/"SUMMARY.md").read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_root, args.output)
