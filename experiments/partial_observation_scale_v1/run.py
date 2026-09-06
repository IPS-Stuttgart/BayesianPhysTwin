"""Source-locked passive visible-to-hidden uncertainty-scale experiment.

All arms share the exact empirical conditional mean. The new hypothesis concerns
observation-dependent posterior uncertainty, not another mean-accuracy contest.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import traceback
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

ANCHORS = np.arange(25, 451, 25)
HORIZON = 30
DLOS = ("DLO4", "DLO5")
MASKS = {"spread25": [0, 7], "spread50": [0, 2, 5, 7],
         "left25": [0, 1], "right25": [6, 7],
         "left50": [0, 1, 2, 3], "right50": [4, 5, 6, 7]}
ARMS = ("static_gaussian", "bayesian_scale_t", "moment_matched_gaussian",
        "calibrated_moment_gaussian", "static_student_t",
        "direct_hetero_gaussian", "direct_hetero_student_t")
DF_GRID = (3., 5., 10., 30., 100.)
BETA_GRID = (0., .25, .5, .75, 1.)
CONTRACT = "partial-observation-scale-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def load_bundle(scratch: Path, dlo: str, phase: str, output: Path):
    path = scratch / f"{dlo}_{phase}.npz"
    receipt = read(output / f"{phase}_inputs.json")[dlo]
    if sha(path) != receipt["archive_sha256"]:
        raise ValueError("Parent prepared array digest mismatch")
    with np.load(path, allow_pickle=False) as data:
        pred = np.asarray(data["prediction"], dtype=float)
        truth = np.asarray(data["truth"], dtype=float)
        names = list(data["names"].astype(str))
    expected = 56 if phase == "source" else 14
    if pred.shape != (expected, 498, 12, 3) or truth.shape != pred.shape:
        raise ValueError("Unexpected frozen DEFORM array shape")
    if len(set(names)) != expected or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Invalid parent arrays or names")
    residual = (truth[:, :, 2:10] - pred[:, :, 2:10]).reshape(expected, 498, 24)
    return np.concatenate((residual[:, ANCHORS], residual[:, ANCHORS + HORIZON]), axis=-1), names


def split_names(names: list[str]) -> dict[str, list[int]]:
    order = sorted(range(len(names)), key=lambda i: hashlib.sha256(
        (CONTRACT + "/source/" + names[i]).encode()).hexdigest())
    if len(order) != 56 or len(set(names)) != 56:
        raise ValueError("Exactly 56 unique source trajectories required")
    return {"fit": order[:32], "select": order[32:44], "calibrate": order[44:]}


def indices(nodes):
    obs = np.array([3 * n + c for n in nodes for c in range(3)], dtype=int)
    hidden = np.array([24 + 3 * n + c for n in range(8) if n not in nodes for c in range(3)])
    return obs, hidden


def moment_fit(e):
    flat = np.asarray(e, dtype=float).reshape(-1, 48)
    mu = flat.mean(axis=0)
    z = flat - mu
    sample = z.T @ z / (len(z) - 1) + np.eye(48) * 1e-10
    return mu, sample


def maps(cov, nodes, nugget):
    obs, hidden = indices(nodes)
    coo = cov[np.ix_(obs, obs)] + np.eye(len(obs)) * nugget
    cho = cov[np.ix_(hidden, obs)]
    gain = np.linalg.solve(coo, cho.T).T
    schur = cov[np.ix_(hidden, hidden)] - gain @ cho.T
    schur = (schur + schur.T) / 2
    np.linalg.cholesky(schur)
    return gain, coo, schur


def visible_prediction(observed, mu, cov, nodes, nugget):
    """Only observed residual coordinates enter this inference function."""
    obs, hidden = indices(nodes)
    observed = np.asarray(observed, dtype=float)
    if observed.shape[-1] != len(obs) or not np.isfinite(observed).all():
        raise ValueError("Invalid observed residual vector")
    gain, coo, schur = maps(cov, nodes, nugget)
    innovation = observed - mu[obs]
    conditional_mean = mu[hidden] + innovation @ gain.T
    white = np.linalg.solve(np.linalg.cholesky(coo), innovation.reshape(-1, len(obs)).T).T
    q = np.sum(white ** 2, axis=1).reshape(observed.shape[:-1])
    return conditional_mean, q, schur


def packets(e, mu, cov, nugget):
    result = {}
    for name, nodes in MASKS.items():
        obs, hidden = indices(nodes)
        mean, q, schur = visible_prediction(e[..., obs], mu, cov, nodes, nugget)
        error = e[..., hidden] - mean
        dimension = len(hidden)
        chol = np.linalg.cholesky(schur)
        white = np.linalg.solve(chol, error.reshape(-1, dimension).T).T
        result[name] = {"error": error, "q": q,
                        "mahal": np.sum(white ** 2, axis=1).reshape(error.shape[:-1]),
                        "logdet": float(2 * np.log(np.diag(chol)).sum()),
                        "variance": np.diag(schur), "observed_dimension": len(obs),
                        "dimension": dimension}
    return result


def law(q, m, config, bayes=None):
    """Return df and Student scale multiplier; df=None denotes Gaussian covariance.

lambda ~ InvGamma(nu/2, scale*(nu-2)/2), and [O,H]|lambda is
Gaussian with covariance lambda*C. Conditional mean is fixed. Conditioning on
O yields df=nu+m and scale=(scale*(nu-2)+Mahalanobis(O))/(nu+m).
    """
    family = config["family"]
    if family == "by_dimension":
        return law(q, m, config["configs"][str(m)], bayes)
    scale = float(config.get("scale", 1.))
    if not scale > 0:
        raise ValueError("Scale must be positive")
    if family in ("bayes", "moment"):
        b = config if family == "bayes" else bayes
        if b is None or b["nu"] <= 2:
            raise ValueError("Finite-variance prior required")
        df = b["nu"] + m
        covariance_multiplier = (b["scale"] * (b["nu"] - 2) + q) / (df - 2)
        if family == "moment":
            return None, scale * covariance_multiplier
        return df, covariance_multiplier * (df - 2) / df
    beta = float(config.get("beta", 0.))
    factor = scale * ((1 - beta) + beta * q / m)
    factor = np.maximum(factor, 1e-12)
    nu = config.get("nu")
    if nu is None:
        return None, factor
    if nu <= 2:
        raise ValueError("Finite variance required")
    return float(nu), factor * (nu - 2) / nu


def log_score(packet, df, multiplier):
    k = packet["dimension"]
    u = packet["mahal"] / multiplier
    ld = packet["logdet"] + k * np.log(multiplier)
    if df is None:
        return .5 * (k * np.log(2 * np.pi) + ld + u) / k
    return (special.gammaln(df / 2) - special.gammaln((df + k) / 2)
            + .5 * (k * np.log(df * np.pi) + ld)
            + .5 * (df + k) * np.log1p(u / df)) / k


def crps_values(error, scale, df):
    z = error / scale
    if df is None:
        return scale * (z * (2 * special.ndtr(z) - 1)
                        + 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    pdf = stats.t.pdf(z, df)
    constant = 2 * np.sqrt(df) / (df - 1) * np.exp(
        special.betaln(.5, df - .5) - 2 * special.betaln(.5, df / 2))
    return scale * (z * (2 * stats.t.cdf(z, df) - 1)
                    + 2 * pdf * (df + z ** 2) / (df - 1) - constant)


def mean_nll(data, config, bayes=None):
    return float(np.mean([np.mean(log_score(p, *law(p["q"], p["observed_dimension"], config, bayes)))
                          for p in data.values()]))


def fit_scale(data, config, bayes=None):
    def objective(logscale):
        return mean_nll(data, {**config, "scale": float(np.exp(logscale))}, bayes)
    result = optimize.minimize_scalar(objective, bounds=(-8., 8.), method="bounded",
                                      options={"xatol": 1e-7, "maxiter": 100})
    if not result.success or not math.isfinite(result.fun):
        raise ValueError("Source-only scale optimization failed")
    return {**config, "scale": float(np.exp(result.x))}, float(result.fun)


def choose(data, grid):
    results = [fit_scale(data, config) for config in grid]
    best = min(range(len(results)), key=lambda i: results[i][1])
    return results[best][0], [{"config": cfg, "selection_nll": score} for cfg, score in results]


def flexible_direct(selected, calibration, student):
    """Likelihood-fit variance regression, separately for each observation count.

Continuous beta and an expanded df grid include the Bayesian conditional family
at each count. This is a strong conventional control, not a fixed-variance foil.
    """
    configs, details = {}, {}
    for m in (6, 12):
        dev = {k: p for k, p in selected.items() if p["observed_dimension"] == m}
        cal = {k: p for k, p in calibration.items() if p["observed_dimension"] == m}
        dfs = sorted(set(DF_GRID) | {nu + m for nu in DF_GRID}) if student else [None]
        candidates = []
        for nu in dfs:
            def objective(beta):
                return fit_scale(dev, {"family": "direct", "beta": beta, "nu": nu})[1]
            optimum = optimize.minimize_scalar(objective, bounds=(0., 1.), method="bounded",
                                                 options={"xatol": 1e-5, "maxiter": 60})
            if not optimum.success:
                raise ValueError("Direct heteroscedastic optimization failed")
            for beta in (0., float(optimum.x), 1.):
                config, score = fit_scale(dev, {"family": "direct", "beta": beta, "nu": nu})
                candidates.append((config, score))
        winner = min(candidates, key=lambda item: item[1])[0]
        configs[str(m)], _ = fit_scale(cal, winner)
        details[str(m)] = [{"config": c, "selection_nll": v} for c, v in candidates]
    return {"family": "by_dimension", "configs": configs}, details


def fit(scratch, output, protocol_path):
    metadata = {"contract": CONTRACT, "objects": {}}
    for dlo in DLOS:
        e, names = load_bundle(scratch, dlo, "source", output)
        split = split_names(names)
        mu, sample = moment_fit(e[split["fit"]])
        trials = []
        for shrink, nugget in itertools.product((0., .1, .3, .6), (1e-8, 1e-6, 1e-4)):
            cov = (1 - shrink) * sample + shrink * np.diag(np.diag(sample))
            p = packets(e[split["select"]], mu, cov, nugget)
            mse = float(np.mean([np.mean(v["error"] ** 2) for v in p.values()]))
            trials.append({"shrinkage": shrink, "nugget": nugget, "selection_mse": mse})
        base = min(trials, key=lambda x: x["selection_mse"])
        cov = (1 - base["shrinkage"]) * sample + base["shrinkage"] * np.diag(np.diag(sample))
        selected_data = packets(e[split["select"]], mu, cov, base["nugget"])
        calibration = packets(e[split["calibrate"]], mu, cov, base["nugget"])
        configs, grid_scores = {}, {}
        configs["static_gaussian"], _ = fit_scale(calibration, {"family": "direct", "beta": 0., "nu": None})
        grids = {
            "bayesian_scale_t": [{"family": "bayes", "nu": nu} for nu in DF_GRID],
            "static_student_t": [{"family": "direct", "beta": 0., "nu": nu} for nu in DF_GRID]}
        for arm, grid in grids.items():
            selected, grid_scores[arm] = choose(selected_data, grid)
            configs[arm], _ = fit_scale(calibration, selected)
        for arm, student in (("direct_hetero_gaussian", False), ("direct_hetero_student_t", True)):
            configs[arm], grid_scores[arm] = flexible_direct(selected_data, calibration, student)
        configs["moment_matched_gaussian"] = {"family": "moment", "scale": 1.}
        configs["calibrated_moment_gaussian"], _ = fit_scale(calibration,
            {"family": "moment"}, configs["bayesian_scale_t"])
        model_path = output / f"{dlo}_scale_model.npz"
        np.savez_compressed(model_path, mean=mu, covariance=cov)
        metadata["objects"][dlo] = {
            "partitions": {key: [names[i] for i in value] for key, value in split.items()},
            "base": base, "base_trials": trials, "configs": configs, "grid_scores": grid_scores,
            "calibration_scores_descriptive": {arm: mean_nll(calibration, config, configs["bayesian_scale_t"])
                                               for arm, config in configs.items()},
            "model_sha256": sha(model_path)}
        print("SOURCE_FROZEN", dlo, json.dumps({"base": base, "configs": configs}), flush=True)
    write(output / "scale_selection.json", metadata)
    write(output / "source_seal.json", {"contract": CONTRACT,
        "implementation_sha256": sha(Path(__file__)), "protocol_sha256": sha(protocol_path),
        "selection_sha256": sha(output / "scale_selection.json"),
        "source_inputs_sha256": sha(output / "source_inputs.json"),
        "model_sha256": {d: sha(output / f"{d}_scale_model.npz") for d in DLOS},
        "new_target_outcomes_read": False, "backbone_trained_on_all_source": True,
        "evidence_class": "retrospective-fixed-two-object-scale-update"})


def metric_rows(dlo, names, data, configs):
    rows = []
    for mask, p in data.items():
        error = p["error"]
        common_rmse = np.sqrt(np.mean(np.sum(error.reshape(*error.shape[:2], -1, 3) ** 2, axis=-1), axis=(1, 2))) * 1000
        for arm in ARMS:
            df, multiplier = law(p["q"], p["observed_dimension"], configs[arm], configs["bayesian_scale_t"])
            sd = np.sqrt(multiplier[..., None] * p["variance"])
            joint_nll = log_score(p, df, multiplier).mean(1)
            crps = crps_values(error, sd, df).mean(axis=(1, 2)) * 1000
            quantile = stats.norm.ppf(.95) if df is None else stats.t.ppf(.95, df)
            halfwidth = sd * quantile
            coverage = (np.abs(error) <= halfwidth).mean(axis=(1, 2))
            width = (2 * halfwidth).mean(axis=(1, 2)) * 1000
            if df is None:
                prob = 2 * special.ndtr(-.02 / sd)
                nn = (p["mahal"] / multiplier / p["dimension"]).mean(1)
            else:
                prob = 2 * stats.t.sf(.02 / sd, df)
                nn = (p["mahal"] / (multiplier * df / (df - 2)) / p["dimension"]).mean(1)
            brier = ((prob - (np.abs(error) > .02)) ** 2).mean(axis=(1, 2))
            for i, name in enumerate(names):
                rows.append({"dlo": dlo, "trajectory": name, "mask": mask, "arm": arm,
                             "nll": float(joint_nll[i]), "crps_mm": float(crps[i]),
                             "coverage90": float(coverage[i]), "width90_mm": float(width[i]),
                             "brier20mm": float(brier[i]), "nanees": float(nn[i]),
                             "rmse_mm": float(common_rmse[i])})
    return rows


def aggregate(rows):
    # Equal masks, trajectories, then fixed objects. No coordinate/window pseudoreplication.
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for metric in ("nll", "crps_mm", "coverage90", "width90_mm", "brier20mm", "nanees", "rmse_mm"):
            object_means = []
            for dlo in DLOS:
                cases = sorted({r["trajectory"] for r in rows if r["dlo"] == dlo})
                object_means.append(np.mean([np.mean([r[metric] for r in rows if
                    r["arm"] == arm and r["dlo"] == dlo and r["trajectory"] == case]) for case in cases]))
            out[arm][metric] = float(np.mean(object_means))
    return out


def paired(rows, comparator, metric):
    differences = []
    by_object = {}
    for dlo in DLOS:
        cases = sorted({r["trajectory"] for r in rows if r["dlo"] == dlo})
        d = []
        for case in cases:
            def value(arm):
                a = [r[metric] for r in rows if r["arm"] == arm and r["dlo"] == dlo and r["trajectory"] == case]
                if len(a) != len(MASKS):
                    raise ValueError("Incomplete method/mask/case records")
                return np.mean(a)
            d.append(value("bayesian_scale_t") - value(comparator))
        a = np.asarray(d)
        differences.append(a)
        by_object[dlo] = {"difference": float(a.mean()), "wins": int((a < 0).sum()), "count": len(a)}
    rng = np.random.default_rng(20260906)
    boot = np.mean([d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1) for d in differences], axis=0)
    return {"difference": float(np.mean([d.mean() for d in differences])),
            "interval95": np.quantile(boot, [.025, .975]).tolist(), "objects": by_object}


def evaluate(scratch, output, protocol_path):
    seal = read(output / "source_seal.json")
    if seal["implementation_sha256"] != sha(Path(__file__)) or seal["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Code/protocol changed after source sealing")
    if seal["selection_sha256"] != sha(output / "scale_selection.json"):
        raise ValueError("Source selections changed")
    selections = read(output / "scale_selection.json")
    rows, artifacts = [], {}
    for dlo in DLOS:
        model_path = output / f"{dlo}_scale_model.npz"
        if sha(model_path) != seal["model_sha256"][dlo]:
            raise ValueError("Source model changed")
        e, names = load_bundle(scratch, dlo, "target", output)
        record = selections["objects"][dlo]
        all_source = sum(record["partitions"].values(), [])
        if set(names) & set(all_source):
            raise ValueError("Source/evaluation trajectory identity overlap")
        with np.load(model_path, allow_pickle=False) as a:
            mu, cov = a["mean"], a["covariance"]
        data = packets(e, mu, cov, record["base"]["nugget"])
        # Retain auditable scalar sufficient statistics for independent re-scoring.
        payload = {"names": np.asarray(names)}
        for mask, p in data.items():
            for key, value in p.items():
                payload[mask + "__" + key] = np.asarray(value)
        path = output / f"{dlo}_scoring_inputs.npz"
        np.savez_compressed(path, **payload)
        artifacts[dlo] = sha(path)
        rows.extend(metric_rows(dlo, names, data, record["configs"]))
    expected = 2 * 14 * len(MASKS) * len(ARMS)
    if len(rows) != expected:
        raise ValueError("Incomplete evaluation")
    means = aggregate(rows)
    if max(abs(v["rmse_mm"] - means["bayesian_scale_t"]["rmse_mm"]) for v in means.values()) > 1e-12:
        raise ValueError("Matched-mean error invariant failed")
    comparisons = {a: {m: paired(rows, a, m) for m in ("nll", "crps_mm", "brier20mm")}
                   for a in ARMS if a != "bayesian_scale_t"}
    controls = ("static_gaussian", "calibrated_moment_gaussian", "static_student_t",
                "direct_hetero_gaussian", "direct_hetero_student_t")
    def passes(arm):
        c = comparisons[arm]
        return (c["nll"]["difference"] <= -.01 and c["nll"]["interval95"][1] < 0
                and all(v["difference"] < 0 for v in c["nll"]["objects"].values())
                and c["crps_mm"]["interval95"][1] < .1)
    decision = {"scale_update_value_vs_static_t": passes("static_student_t"),
                "integration_value_vs_exact_moment_gaussian": passes("moment_matched_gaussian"),
                "stronger_model_value": all(passes(a) for a in controls)}
    result = {"contract": CONTRACT, "status": "complete", "decision": decision,
              "means": means, "comparisons": comparisons, "record_count": len(rows),
              "source_seal_sha256": sha(output / "source_seal.json"), "scoring_inputs_sha256": artifacts,
              "execution_commit": os.environ.get("GITHUB_SHA", "local-synthetic-only"),
              "workflow_run": os.environ.get("GITHUB_RUN_ID", "local"),
              "limitations": ["retrospective same two objects", "coordinate withholding, not camera occlusion",
                              "known recorded future boundaries", "backbone trained on all source records",
                              "32 fit + 12 hyperparameter selection + 12 scale calibration per object",
                              "scale-mixture predictive discrepancy, not physical parameter posterior",
                              "same exact mean; no point-accuracy claim", "framewise latent scales; temporal independence is not validated"]}
    write(output / "result.json", result)
    write(output / "per_case.json", {"records": rows})
    report = ["# Visible-to-hidden uncertainty update: completed retrospective test", "",
              "All arms have exactly identical conditional point predictions.", "",
              "| Method | Joint NLL/coordinate | CRPS (mm) | 90% coverage | Width (mm) | Brier, error >20mm |",
              "|---|---:|---:|---:|---:|---:|"]
    for a, v in means.items():
        report.append(f"| {a} | {v['nll']:.6f} | {v['crps_mm']:.4f} | {100*v['coverage90']:.2f}% | {v['width90_mm']:.3f} | {v['brier20mm']:.6f} |")
    report += ["", "## Paired NLL differences: Bayesian minus comparator", ""]
    for a, c in comparisons.items():
        v = c["nll"]
        report.append(f"- {a}: {v['difference']:+.6f} [{v['interval95'][0]:+.6f}, {v['interval95'][1]:+.6f}]; per-object {v['objects']}")
    report += ["", "## Prespecified decisions", "", "```json", json.dumps(decision, indent=2), "```", "",
               "Negative scientific decisions are valid completed results, not technical failures.", "", "## Boundaries", ""]
    report += ["- " + s for s in result["limitations"]]
    text = "\n".join(report) + "\n"
    (output / "report.md").write_text(text)
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "score"))
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    if protocol["contract"] != CONTRACT or protocol["horizon"] != HORIZON or protocol["masks"] != MASKS:
        raise ValueError("Protocol differs from fixed implementation")
    if protocol["source_counts"] != [32, 12, 12] or protocol["df_grid"] != list(DF_GRID):
        raise ValueError("Source split or degrees-of-freedom grid changed")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        (fit if args.phase == "fit" else evaluate)(args.scratch, args.output, args.protocol)
    except Exception as exc:
        write(args.output / f"failure_{args.phase}.json", {"type": type(exc).__name__, "error": str(exc),
                                                         "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
