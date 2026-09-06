"""Passive real-data test of conditional posterior-predictive query uncertainty.

Standalone NumPy/SciPy capsule. It does not run DEFORM's physical simulator.
Only explicitly named DLO4/DLO5/train/*.pkl files may be read. Source-test
recordings are historically exposed: this is a retrospective hypothesis test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.special import betaln, logsumexp
from scipy.stats import norm, t as student

CONFIG = {
    "contract": "deform-conditional-query-posterior-v1",
    "split_domain": "conditional-query-posterior-v1-20260906",
    "dlos": ["DLO4", "DLO5"],
    "fit": 32, "calibration": 12, "test": 12,
    "source_sizes": [8, 16, 32], "primary_source_size": 32,
    "origins": [25, 100, 200, 300, 400], "horizons": [5, 20, 50],
    "ridge_grid": [1.0, 10.0, 100.0],
    "variance_temperature_grid": [0.25, 0.5, 1., 2., 4., 8., 16., 32., 64.],
    "bootstrap_replicates": 10000, "seed": 20260906,
    "absolute_query_error_threshold_m": 0.01,
    "official_eval_access": False, "new_acquisition": False,
    "claim_boundary": "Retrospective source-held real-trajectory surrogate study; not a DEFORM simulator rerun, unseen-object validation, or full-twin superiority.",
}
ARMS = ("posterior_student", "plugin_gaussian", "gaussian_posterior_covariance", "same_covariance_gaussian",
        "global_shrinkage", "global_residual_bootstrap", "local_residual_bootstrap")


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_paths(root, dlo):
    directory = Path(root) / dlo / "train"
    paths = sorted(directory.glob("*.pkl"))
    if len(paths) != 56 or any(not p.is_file() for p in paths):
        raise ValueError(f"Expected exactly 56 files in {directory}, found {len(paths)}")
    paths.sort(key=lambda p: hashlib.sha256(
        (CONFIG["split_domain"] + "/" + dlo + "/" + p.name).encode()).digest())
    return paths[:32], paths[32:44], paths[44:]


def load(path):
    # The user's verified public dataset uses trusted numerical pickles.
    if path.parent.name != "train" or path.parent.parent.name not in CONFIG["dlos"]:
        raise ValueError("Access outside the registered source train partition")
    with path.open("rb") as f:
        x = np.asarray(pickle.load(f), dtype=float)
    if x.shape != (500, 3, 12) or not np.isfinite(x).all():
        raise ValueError(f"Invalid trajectory {path}: {x.shape}")
    return x.transpose(0, 2, 1).copy()  # Raw metre coordinates; no truth clipping.


def context(prefix, endpoints, horizon):
    """Accepts NO future internal node array. Endpoints are logged action inputs."""
    now, prev = prefix[-1], prefix[-3]
    left = endpoints[:2].mean(0) - now[:2].mean(0)
    right = endpoints[2:].mean(0) - now[-2:].mean(0)
    alpha = np.arange(2, 10)[:, None] / 11.0
    base = now[2:10] + (1-alpha)*left + alpha*right
    shape = now[2:10] - now[[0, 11]].mean(0)
    velocity = (now[2:10] - prev[2:10]) / 2
    feature = np.r_[shape.ravel(), velocity.ravel()*horizon, left, right]
    return feature, base.ravel(), now[2:10].ravel()


def inputs(x, origin, horizon):
    return context(x[origin-2:origin+1], x[origin+horizon, [0, 1, 10, 11]], horizon)


def queries():
    dev = [np.ones(8)/8, np.r_[np.ones(4)/4, -np.ones(4)/4]]
    held = [np.r_[.5, .5, np.zeros(6)], np.r_[np.zeros(3), .5, .5, np.zeros(3)],
            np.r_[np.zeros(6), .5, .5], np.array([.25, .25, -.25, -.25, -.25, -.25, .25, .25])]
    def build(weights):
        return np.stack([np.outer(w, np.eye(3)[axis]).ravel() for w in weights for axis in range(3)])
    return build(dev), build(held)


def design_fit(f):
    center, scale = f.mean(0), f.std(0)
    scale = np.maximum(scale, 1e-6)
    z = (f-center)/scale
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    rank = min(8, len(f)//2, int(np.sum(singular > 1e-8)))
    basis = vt[:rank].T * (np.sqrt(max(len(f)-1, 1))/np.maximum(singular[:rank], 1e-8))
    x = np.column_stack([np.ones(len(f)), z@basis])
    return x, center, scale, basis


def ridge_fit(x, y, ridge):
    precision = x.T@x + ridge*np.eye(x.shape[1])
    v = np.linalg.inv(precision)
    w = v@x.T@y
    hat = np.einsum("ni,ij,nj->n", x, v, x)
    loo = (y-x@w) / np.maximum(1-hat, 1e-8)[:, None]
    return w, v, loo


def fit_model(f, y):
    x, center, feature_scale, basis = design_fit(f)
    # All tuning here is source-fit-only, grouped by complete trajectory.
    ridge = min(CONFIG["ridge_grid"], key=lambda a: np.mean(ridge_fit(x, y, a)[2]**2))
    _, _, first_loo = ridge_fit(x, y, ridge)
    # Empirical-Bayes heteroscedastic scale; fixed before posterior integration.
    g = x[:, :min(x.shape[1], 4)]
    log_power = np.log(np.maximum(np.mean(first_loo**2, axis=1), 1e-10))
    beta = np.linalg.solve(g.T@g + 10*np.eye(g.shape[1]), g.T@log_power)
    log_center = float(np.mean(g@beta))
    s2 = np.exp(np.clip(g@beta-log_center, -3, 3))
    xw, yw = x/np.sqrt(s2[:, None]), y/np.sqrt(s2[:, None])
    w, v, loo_w = ridge_fit(xw, yw, ridge)
    loo = loo_w*np.sqrt(s2[:, None])
    d = y.shape[1]
    prior_variance = max(float(np.mean(first_loo**2)), 1e-8)
    # W|Sigma ~ MN(0, I/ridge, Sigma); Sigma ~ IW((d+2), prior_variance I).
    psi = prior_variance*np.eye(d) + yw.T@yw - w.T@(xw.T@xw + ridge*np.eye(x.shape[1]))@w
    psi = (psi+psi.T)/2
    np.linalg.cholesky(psi)
    df = float(len(f)+3)  # nu_n - output_dimension + 1
    empirical = loo.T@loo/len(loo)
    empirical = .5*empirical + .5*np.diag(np.diag(empirical)) + 1e-10*np.eye(d)
    return dict(w=w, v=v, psi=psi, df=df, center=center, feature_scale=feature_scale,
                basis=basis, beta=beta, log_center=log_center, x=x,
                normalized_residuals=loo/np.sqrt(s2[:, None]), raw_residuals=loo,
                empirical=empirical, ridge=float(ridge))


def transform(model, f):
    x = np.r_[1., ((f-model["center"])/model["feature_scale"])@model["basis"]]
    s2 = float(np.exp(np.clip(x[:len(model["beta"])]@model["beta"]-model["log_center"], -3, 3)))
    return x, s2


def make_distribution(model, f, q, arm, temperature=1., k=8, bandwidth=.5):
    x, s2 = transform(model, f)
    h = float(x@model["v"]@x)
    projected = np.einsum("qi,ij,qj->q", q, model["psi"], q)
    if arm == "posterior_student":
        return dict(kind="student", scale=np.sqrt(temperature*(s2+h)*projected/model["df"]), df=model["df"])
    if arm in ("plugin_gaussian", "same_covariance_gaussian", "gaussian_posterior_covariance"):
        multiplier = s2 + (0. if arm == "plugin_gaussian" else h)
        return dict(kind="normal", scale=np.sqrt(temperature*multiplier*projected/(model["df"]-2)))
    if arm == "global_shrinkage":
        return dict(kind="normal", scale=np.sqrt(temperature*np.einsum("qi,ij,qj->q", q, model["empirical"], q)))
    if arm == "global_residual_bootstrap":
        residual = model["raw_residuals"]
    elif arm == "local_residual_bootstrap":
        order = np.argsort(np.sum((model["x"]-x)**2, axis=1), kind="stable")[:min(k, len(model["x"]))]
        residual = np.sqrt(s2)*model["normalized_residuals"][order]
    else:
        raise ValueError(arm)
    # Symmetric whole-vector resampling preserves the exact shared predictive mean.
    locations = np.concatenate([residual@q.T, -residual@q.T], axis=0).T
    diagonal = np.maximum(np.mean(residual**2, axis=0), 1e-10)
    kernel = bandwidth*np.sqrt((q*q)@diagonal)
    return dict(kind="mixture", locations=np.sqrt(temperature)*locations,
                scale=np.sqrt(temperature)*np.maximum(kernel, 1e-6))


def nll(dist, error):
    if dist["kind"] == "student":
        return -student.logpdf(error/dist["scale"], dist["df"]) + np.log(dist["scale"])
    if dist["kind"] == "normal":
        return .5*(error/dist["scale"])**2 + np.log(dist["scale"]) + .5*np.log(2*np.pi)
    z = (error[:, None]-dist["locations"])/dist["scale"][:, None]
    return -logsumexp(norm.logpdf(z)-np.log(dist["scale"][:, None]), axis=1) + np.log(z.shape[1])


def cdf(dist, value):
    value = np.broadcast_to(value, dist["scale"].shape)
    if dist["kind"] == "student":
        return student.cdf(value/dist["scale"], dist["df"])
    if dist["kind"] == "normal":
        return norm.cdf(value/dist["scale"])
    return np.mean(norm.cdf((value[:, None]-dist["locations"])/dist["scale"][:, None]), axis=1)


def abs_normal(delta, sigma):
    z = delta/sigma
    return 2*sigma*norm.pdf(z) + delta*(2*norm.cdf(z)-1)


def crps(dist, error):
    s = dist["scale"]
    if dist["kind"] == "student":
        df, z = dist["df"], error/s
        constant = 2*np.sqrt(df)/(df-1)*np.exp(betaln(.5, df-.5)-2*betaln(.5, df/2))
        return s*(z*(2*student.cdf(z, df)-1) + 2*student.pdf(z, df)*(df+z*z)/(df-1)-constant)
    if dist["kind"] == "normal":
        return abs_normal(error, s) - s/np.sqrt(np.pi)
    loc = dist["locations"]
    first = abs_normal(error[:, None]-loc, s[:, None]).mean(1)
    pair = loc[:, :, None]-loc[:, None, :]
    return first-.5*abs_normal(pair, np.sqrt(2)*s[:, None, None]).mean((1, 2))


def scores(dist, error):
    if dist["kind"] == "mixture":
        lo = np.zeros_like(error)
        hi = np.max(np.abs(dist["locations"]), axis=1) + 8*dist["scale"]
        for _ in range(45):
            mid = (lo+hi)/2
            below = cdf(dist, mid) < .95
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        half_width = (lo+hi)/2
    else:
        factor = student.ppf(.95, dist["df"]) if dist["kind"] == "student" else norm.ppf(.95)
        half_width = factor*dist["scale"]
    threshold = CONFIG["absolute_query_error_threshold_m"]
    probability = 1-cdf(dist, threshold)+cdf(dist, -threshold)
    return dict(nll=nll(dist, error), crps_m=crps(dist, error),
                coverage90=(np.abs(error) <= half_width).astype(float),
                width90_m=2*half_width, brier=(probability-(np.abs(error)>threshold))**2)


def calibrate(model, features, errors, q):
    settings = {}
    for arm in ARMS:
        if arm == "same_covariance_gaussian":
            settings[arm] = dict(settings["posterior_student"])
            continue
        candidates = [(temp, k, bw) for temp in CONFIG["variance_temperature_grid"]
                      for k in ([8, 16] if arm == "local_residual_bootstrap" else [32])
                      for bw in ([.25, .5, 1.] if "bootstrap" in arm else [.5])]
        def objective(setting):
            temp, k, bw = setting
            return np.mean([np.mean(nll(make_distribution(model, f, q, arm, temp, k, bw), e@q.T))
                            for f, e in zip(features, errors, strict=True)])
        winner = min(candidates, key=objective)
        settings[arm] = dict(temperature=float(winner[0]), k=winner[1], bandwidth=winner[2])
    return settings


def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, str)):
        return obj
    return {k: serialize(v) for k, v in obj.items()}


def deserialize_distribution(obj):
    return {k: np.asarray(v) if k in ("scale", "locations") else v for k, v in obj.items()}


def aggregate(rows):
    metrics = ("nll", "crps_m", "coverage90", "width90_m", "brier")
    summaries, comparisons = {}, {}
    for size in CONFIG["source_sizes"]:
        chosen = [r for r in rows if r["source_size"] == size]
        case_keys = sorted({(r["dlo"], r["trajectory"]) for r in chosen})
        blocks = {arm: np.array([[np.mean([r[m] for r in chosen if r["arm"] == arm and
                    (r["dlo"], r["trajectory"]) == key]) for m in metrics] for key in case_keys]) for arm in ARMS}
        summaries[str(size)] = {arm: dict(zip(metrics, value.mean(0).tolist(), strict=True)) for arm, value in blocks.items()}
        rng = np.random.default_rng(CONFIG["seed"])
        selected = np.column_stack([rng.choice(np.flatnonzero(np.array([k[0] for k in case_keys]) == dlo),
                         size=(CONFIG["bootstrap_replicates"], 12), replace=True) for dlo in CONFIG["dlos"]])
        contrasts = {}
        for arm in ARMS[1:]:
            delta = blocks[ARMS[0]]-blocks[arm]
            interval = np.quantile(delta[selected].mean(1), [.025, .975], axis=0)
            contrasts[arm] = {m: {"difference": float(delta[:, j].mean()),
                "trajectory_bootstrap_95_ci": interval[:, j].tolist(),
                "trajectory_wins": int(np.sum(delta[:, j] < 0))} for j, m in enumerate(metrics)}
        comparisons[str(size)] = contrasts
    primary = comparisons[str(CONFIG["primary_source_size"])]
    supported = all(primary[arm][m]["trajectory_bootstrap_95_ci"][1] < 0
                    for arm in ("plugin_gaussian", "global_shrinkage", "global_residual_bootstrap", "local_residual_bootstrap")
                    for m in ("nll", "crps_m"))
    return dict(aggregate=summaries, posterior_minus_comparator=comparisons,
                primary_hypothesis_supported=supported,
                decision="supported-on-this-retrospective-panel" if supported else "not-established-on-this-panel",
                statistical_unit="complete trajectory; bootstrap stratified within two fixed DLOs; not an object-population CI")


def run(root, output, revision):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    dump(out/"protocol.json", CONFIG)
    devq, heldq = queries()
    manifest, sources, held_paths = {}, {}, {}
    for dlo in CONFIG["dlos"]:
        fit, cal, test = split_paths(root, dlo)
        manifest[dlo] = {part: [{"name": p.name, "sha256": digest(p)} for p in paths]
                         for part, paths in (("fit", fit), ("calibration", cal), ("source_test", test))}
        sources[dlo] = ([load(p) for p in fit], [load(p) for p in cal])
        held_paths[dlo] = test
    dump(out/"input_manifest.json", manifest)
    models = []
    for dlo in CONFIG["dlos"]:
        fit, cal = sources[dlo]
        for size in CONFIG["source_sizes"]:
            for origin in CONFIG["origins"]:
                for horizon in CONFIG["horizons"]:
                    fin = [inputs(a, origin, horizon) for a in fit[:size]]
                    features = np.stack([a[0] for a in fin])
                    residuals = np.stack([a[origin+horizon, 2:10].ravel()-b[1] for a, b in zip(fit[:size], fin, strict=True)])
                    model = fit_model(features, residuals)
                    cin = [inputs(a, origin, horizon) for a in cal]
                    cerr = np.stack([a[origin+horizon, 2:10].ravel()-b[1]-transform(model, b[0])[0]@model["w"]
                                     for a, b in zip(cal, cin, strict=True)])
                    settings = calibrate(model, [a[0] for a in cin], cerr, devq)
                    models.append(dict(dlo=dlo, source_size=size, origin=origin, horizon=horizon,
                                       model=model, settings=settings))
        print(f"Source fitting and calibration complete: {dlo}", flush=True)
    dump(out/"frozen_models.json", {str(i): serialize(m) for i, m in enumerate(models)})
    predictions = []
    # No future internal outcomes are passed to prediction. All models are frozen.
    for dlo, paths in held_paths.items():
        for path in paths:
            trajectory = load(path)
            for model_id, entry in enumerate(models):
                if entry["dlo"] != dlo:
                    continue
                f, base, current = inputs(trajectory, entry["origin"], entry["horizon"])
                model = entry["model"]
                mean = base + transform(model, f)[0]@model["w"]
                predictions.append(dict(dlo=dlo, trajectory=path.name, model_id=model_id,
                    source_size=entry["source_size"], origin=entry["origin"], horizon=entry["horizon"],
                    mean=mean.tolist(), current=current.tolist(), distributions={
                        arm: serialize(make_distribution(model, f, heldq, arm, **entry["settings"][arm])) for arm in ARMS}))
    dump(out/"sealed_predictions.json", {str(i): p for i, p in enumerate(predictions)})
    seal = {"source_revision": revision, "protocol_sha256": digest(out/"protocol.json"),
            "models_sha256": digest(out/"frozen_models.json"), "predictions_sha256": digest(out/"sealed_predictions.json"),
            "input_manifest_sha256": digest(out/"input_manifest.json"), "future_internal_values_used_in_prediction": False}
    dump(out/"prediction_seal.json", seal)
    print("All target predictions sealed; scoring starts now.", flush=True)
    rows, point = [], []
    for dlo, paths in held_paths.items():
        for path in paths:
            trajectory = load(path)
            for p in predictions:
                if (p["dlo"], p["trajectory"]) != (dlo, path.name):
                    continue
                error = trajectory[p["origin"]+p["horizon"], 2:10].ravel()-np.array(p["mean"])
                point.append(float(np.mean(np.abs(error))))
                for arm in ARMS:
                    dist = deserialize_distribution(p["distributions"][arm])
                    score = scores(dist, error@heldq.T)
                    rows.append({**{k: p[k] for k in ("dlo", "trajectory", "source_size", "origin", "horizon")},
                                 "arm": arm, **{m: float(v.mean()) for m, v in score.items()}})
    result = {**aggregate(rows), "config": CONFIG, "seal": seal,
              "accounting": {"physical_objects": 2, "fit_trajectories_per_object": 32,
                  "calibration_trajectories_per_object": 12, "source_test_trajectories": 24,
                  "models": len(models), "prediction_contexts": len(predictions), "held_query_count": len(heldq),
                  "max_predictive_mean_difference": 0., "official_eval_files_opened": 0,
                  "raw_truth_clipping": False},
              "common_mean_coordinate_l1_m_descriptive": float(np.mean(point)),
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                              "runner": os.environ.get("RUNNER_NAME", "local")}}
    dump(out/"result.json", result)
    with (out/"trajectory_panel_scores.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
    table = ["# Conditional query posterior: real-data result", "", result["decision"], "",
             CONFIG["claim_boundary"], "", "Primary source size: 32. Each value equally weights the 24 complete source-test trajectories.", "",
             "| Method | NLL | CRPS (mm) | 90% coverage | Width (mm) | Brier |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, v in result["aggregate"]["32"].items():
        table.append(f"| {arm} | {v['nll']:.6f} | {1000*v['crps_m']:.4f} | {100*v['coverage90']:.2f}% | {1000*v['width90_m']:.3f} | {v['brier']:.6f} |")
    table += ["", "Posterior-minus-comparator trajectory-bootstrap 95% intervals:", ""]
    for arm, contrasts in result["posterior_minus_comparator"]["32"].items():
        table.append(f"{arm}: NLL {contrasts['nll']}; CRPS {contrasts['crps_m']}")
    table += ["", "No new physical actions, no active sensing, no official evaluation files, and no model/threshold changes after source-test scoring.",
              "Covariance hyperparameters, heteroscedastic scale, and temperatures are empirical-Bayes/source fitted; this is not fully Bayesian hyperparameter integration.",
              "The common mean is an action-conditioned ridge surrogate, not an official DEFORM simulator checkpoint."]
    (out/"SUMMARY.md").write_text("\n".join(table)+"\n")
    print(json.dumps({"decision": result["decision"], "primary": result["aggregate"]["32"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    run(args.dataset_root, args.output, args.revision)
