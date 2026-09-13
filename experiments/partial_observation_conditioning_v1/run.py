"""Retrospective, source-tuned partial-observation conditioning on real DEFORM data.

No action selection, new physical data, target tuning, or silent cohort replacement.
The structured model is a new empirical-Bayes Gaussian residual belief around the
frozen DEFORM hybrid, not a posterior over physical material parameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import pickle
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

ANCHORS = np.arange(25, 451, 25)
HORIZONS = (0, 10, 30)
MASKS = {
    "spread25": [0, 7], "spread50": [0, 2, 5, 7],
    "left25": [0, 1], "right25": [6, 7],
    "left50": [0, 1, 2, 3], "right50": [4, 5, 6, 7],
}
ARMS = ("physical", "matched_prior", "structured", "diagonal", "scrambled",
        "empirical", "ridge", "translation", "interpolation", "frozen_interpolation",
        "map_equivalent")
DLOS = ("DLO4", "DLO5")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked(path, digest):
    path = Path(path)
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"Missing or changed frozen input: {path}")
    return path


def load_numeric_trajectory(path, digest):
    # Only explicitly hashed, official numeric trajectory pickles are admitted.
    checked(path, digest)
    with Path(path).open("rb") as f:
        x = np.asarray(pickle.load(f), dtype=np.float32)
    if x.shape != (500, 3, 12) or not np.isfinite(x).all():
        raise ValueError(f"Invalid official trajectory: {path}")
    x = x.transpose(0, 2, 1).copy()
    x[:, :, 2] = np.clip(x[:, :, 2], 2e-3 + 1e-6, 10000.0)
    return x


def prepare(request, scratch, output, phase):
    root = Path(request["parent_root"])
    upstream = Path(request["upstream_root"])
    if phase == "target" and not (output / "source_seal.json").is_file():
        raise RuntimeError("Source fitting must be sealed before target access")
    receipts = {}
    for dlo in DLOS:
        pins = request["pins"][dlo]
        folder = root / (dlo.lower() + "-target")
        if phase == "source":
            manifest_path = root / (dlo.lower() + "-source") / "source_manifest.json"
        else:
            manifest_path = folder / "eval_manifest.json"
        manifest = json.loads(checked(manifest_path, pins[phase + "_manifest"]).read_text())
        names = manifest["ordered_names"]
        expected_count = 56 if phase == "source" else 14
        if len(names) != expected_count or len(set(names)) != expected_count:
            raise ValueError("Frozen roster count or uniqueness changed")
        partition = "train" if phase == "source" else "eval"
        trajectories = {}
        for name in names:
            if Path(name).name != name:
                raise ValueError("Unexpected trajectory name")
            p = Path(request["dataset_root"]) / dlo / partition / name
            trajectories[name] = load_numeric_trajectory(p, manifest["trajectories"][name]["sha256"])
        if phase == "source":
            import run_deform_dlo_source as source_runtime
            import run_deform_dlo_longrun_posterior as posterior_runtime
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Frozen DEFORM replay requires CUDA")
            source_runtime._assert_upstream(upstream, request["upstream_commit"])
            source_runtime._seed_everything(torch, 42)
            modules = source_runtime._load_upstream(upstream)
            checkpoint = checked(folder / "alltrain/physical_update_6400.pt", pins["checkpoint"])
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
            rollout = posterior_runtime._evaluate_state(
                dict(state), trajectories, modules=modules, torch=torch,
                device=os.environ.get("PARTIAL_DEVICE", "cuda:0"), dlo_type=dlo, node_count=12)
            prediction = np.asarray(rollout["predictions"])
            truth = np.asarray(rollout["targets"])
            if list(rollout["names"]) != names:
                raise RuntimeError("Source replay order changed")
            checkpoint_digest = sha(checkpoint)
        else:
            archive = checked(folder / "target_predictions.npz", pins["predictions"])
            with np.load(archive, allow_pickle=False) as a:
                if list(a["names"].astype(str)) != names:
                    raise RuntimeError("Cached prediction roster differs")
                prediction = a["physical"].copy()
            truth = np.stack([trajectories[name][2:] for name in names])
            checkpoint_digest = pins["checkpoint"]
            original_l1 = float(np.abs(prediction - truth).mean() * 1000)
            if abs(original_l1 - pins["physical_l1_mm"]) > 0.001:
                raise RuntimeError(f"Original DEFORM target parity failed: {dlo} {original_l1}")
        if prediction.shape != (expected_count, 498, 12, 3) or prediction.shape != truth.shape:
            raise RuntimeError("Unexpected rollout dimensions")
        if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
            raise RuntimeError("Nonfinite frozen rollout")
        destination = scratch / f"{dlo}_{phase}.npz"
        np.savez_compressed(destination, names=np.asarray(names), prediction=prediction, truth=truth)
        receipts[dlo] = {"archive_sha256": sha(destination), "manifest_sha256": sha(manifest_path),
                         "checkpoint_sha256": checkpoint_digest, "trajectories": names}
        print("PREPARED", phase, dlo, prediction.shape, flush=True)
    write(output / f"{phase}_inputs.json", receipts)


def load_bundle(scratch, dlo, phase):
    with np.load(scratch / f"{dlo}_{phase}.npz", allow_pickle=False) as a:
        return a["prediction"].astype(float), a["truth"].astype(float), list(a["names"].astype(str))


def windows(prediction, truth, horizon):
    p = prediction[:, :, 2:10].reshape(len(prediction), 498, 24)
    r = (truth[:, :, 2:10] - prediction[:, :, 2:10]).reshape(len(prediction), 498, 24)
    e = r[:, ANCHORS] if horizon == 0 else np.concatenate((r[:, ANCHORS], r[:, ANCHORS + horizon]), axis=-1)
    return e, p[:, ANCHORS], p[:, ANCHORS + horizon]


def indices(nodes, horizon):
    obs = np.array([3 * n + c for n in nodes for c in range(3)], dtype=int)
    hidden_nodes = [n for n in range(8) if n not in nodes]
    hidden = np.array([3 * n + c for n in hidden_nodes for c in range(3)], dtype=int)
    return obs, hidden + (24 if horizon else 0), hidden_nodes


def moments(e):
    flat = e.reshape(-1, e.shape[-1])
    mean = flat.mean(0)
    d = flat - mean
    cov = d.T @ d / (len(d) - 1) + np.eye(d.shape[-1]) * 1e-10
    return mean, cov


def covariance(sample, horizon, params):
    alpha, bend, tau, _ = params
    if bend == 0:
        target = np.diag(np.diag(sample))
    else:
        second = np.diff(np.eye(8), n=2, axis=0)
        spatial = np.linalg.inv(np.eye(8) + bend * second.T @ second)
        sd = np.sqrt(np.diag(spatial))
        spatial = spatial / sd[:, None] / sd[None, :]
        corr = np.kron(spatial, np.eye(3))
        if horizon:
            rho = np.exp(-horizon / tau)
            corr = np.block([[corr, rho * corr], [rho * corr, corr]])
        scale = np.sqrt(np.diag(sample))
        target = corr * scale[:, None] * scale[None, :]
    c = (1 - alpha) * sample + alpha * target
    return (c + c.T) / 2


def gaussian_map(cov, obs, hidden, noise):
    innovation = cov[np.ix_(obs, obs)] + noise * np.eye(len(obs))
    gain = np.linalg.solve(innovation, cov[np.ix_(obs, hidden)]).T
    posterior = cov[np.ix_(hidden, hidden)] - gain @ cov[np.ix_(obs, hidden)]
    return gain, (posterior + posterior.T) / 2


def information_map(cov, obs, hidden, noise):
    precision = np.linalg.inv(cov)
    precision[obs, obs] += 1 / noise
    rhs = np.zeros((len(cov), len(obs)))
    rhs[obs, np.arange(len(obs))] = 1 / noise
    return np.linalg.solve(precision, rhs)[hidden]


def interpolation_map(nodes, hidden_nodes):
    order = np.argsort(nodes)
    weights = np.column_stack([
        np.interp(hidden_nodes, np.asarray(nodes)[order], np.eye(len(nodes))[order, j])
        for j in range(len(nodes))])
    return np.kron(weights, np.eye(3))


def mean_loss(e, mean, cov, horizon, noise):
    z = e - mean
    losses = []
    for nodes in MASKS.values():
        obs, hidden, _ = indices(nodes, horizon)
        gain, _ = gaussian_map(cov, obs, hidden, noise)
        err = z[:, :, hidden] - z[:, :, obs] @ gain.T
        losses.append(float(np.mean(err ** 2)))
    return float(np.mean(losses))


def simple_error(e, p0, ph, mean, horizon, nodes, arm, value):
    z = e - mean
    obs, hidden, hn = indices(nodes, horizon)
    if arm == "translation":
        gain = np.kron(np.ones((len(hn), len(nodes))) / len(nodes), np.eye(3))
    else:
        gain = interpolation_map(nodes, hn)
    correction = value * (z[:, :, obs] @ gain.T)
    if arm == "frozen_interpolation":
        hi = hidden - (24 if horizon else 0)
        correction = correction + (p0 + mean[:24] - ph - mean[-24:])[:, :, hi]
    return z[:, :, hidden] - correction


def fit_all(scratch, output, request_path):
    selection = {}
    for dlo in DLOS:
        pred, truth, names = load_bundle(scratch, dlo, "source")
        order = sorted(range(56), key=lambda i: hashlib.sha256(("partial-v1/" + names[i]).encode()).hexdigest())
        fit_ids, val_ids = order[:40], order[40:]
        selection[dlo] = {"fit_names": [names[i] for i in fit_ids], "validation_names": [names[i] for i in val_ids], "horizons": {}}
        arrays = {}
        for horizon in HORIZONS:
            e, p0, ph = windows(pred, truth, horizon)
            mu, s = moments(e[fit_ids])
            selected = {}
            grid_struct = list(itertools.product((.25, .5, .75, 1.), (1., 10.), (50.,) if horizon == 0 else (10., 50., 200.), (1e-8, 1e-6, 1e-4)))
            grid_emp = [(a, 0., 50., n) for a in (0., .1, .3, .6, 1.) for n in (1e-8, 1e-6, 1e-4)]
            for arm, grid in (("structured", grid_struct), ("empirical", grid_emp)):
                scores = [mean_loss(e[val_ids], mu, covariance(s, horizon, q), horizon, q[3]) for q in grid]
                best = int(np.argmin(scores))
                selected[arm] = {"parameters": list(grid[best]), "validation_mse_m2": scores[best], "all_scores": scores}
            ridge_grid = (1e-8, 1e-6, 1e-4, 1e-2)
            ridge_scores = [mean_loss(e[val_ids], mu, s, horizon, n) for n in ridge_grid]
            best = int(np.argmin(ridge_scores))
            selected["ridge"] = {"noise": ridge_grid[best], "validation_mse_m2": ridge_scores[best]}
            for arm in ("translation", "interpolation", "frozen_interpolation"):
                gains = (0., .25, .5, .75, 1.)
                scores = [float(np.mean([np.mean(simple_error(e[val_ids], p0[val_ids], ph[val_ids], mu, horizon, nodes, arm, g) ** 2) for nodes in MASKS.values()])) for g in gains]
                best = int(np.argmin(scores))
                selected[arm] = {"gain": gains[best], "validation_mse_m2": scores[best]}
            final_mu, final_s = moments(e)
            arrays[f"mean_{horizon}"] = final_mu
            arrays[f"sample_{horizon}"] = final_s
            for arm in ("structured", "empirical"):
                arrays[f"{arm}_{horizon}"] = covariance(final_s, horizon, selected[arm]["parameters"])
            selection[dlo]["horizons"][str(horizon)] = selected
            print("SOURCE_SELECTED", dlo, horizon, {a: {k:v for k,v in q.items() if k != "all_scores"} for a,q in selected.items()}, flush=True)
        np.savez_compressed(output / f"{dlo}_model.npz", **arrays)
    write(output / "source_selection.json", selection)
    write(output / "source_seal.json", {
        "request_sha256": sha(request_path), "implementation_sha256": sha(__file__),
        "selection_sha256": sha(output / "source_selection.json"),
        "models": {d: sha(output / f"{d}_model.npz") for d in DLOS},
        "source_inputs_sha256": sha(output / "source_inputs.json"),
        "target_outcomes_read_in_this_experiment": False,
        "backbone_pretrained_on_all_source_records": True,
        "evidence_class": "retrospective-new-operator-on-previously-opened-targets"})


def metrics(error, posterior=None):
    ncase, nt, dimension = error.shape
    xyz = error.reshape(ncase, nt, dimension // 3, 3)
    shape = xyz - xyz.mean(axis=2, keepdims=True)
    result = {"rmse_mm": np.sqrt(np.mean(np.sum(xyz ** 2, axis=-1), axis=(1, 2))) * 1000,
              "l1_mm": np.mean(np.abs(error), axis=(1, 2)) * 1000,
              "translation_free_rmse_mm": np.sqrt(np.mean(np.sum(shape ** 2, axis=-1), axis=(1, 2))) * 1000}
    if posterior is not None:
        chol = np.linalg.cholesky(posterior)
        whitened = np.linalg.solve(chol, error.reshape(-1, dimension).T).T.reshape(error.shape)
        nees = np.sum(whitened ** 2, axis=-1) / dimension
        logdet = 2 * np.log(np.diag(chol)).sum()
        sd = np.sqrt(np.diag(posterior))
        result.update({"joint_nll_per_coordinate": .5 * (dimension * np.log(2*np.pi) + logdet + dimension * nees.mean(1)) / dimension,
                       "normalized_joint_nees": nees.mean(1),
                       "coverage90": np.mean(np.abs(error) <= 1.6448536269514722 * sd, axis=(1, 2)),
                       "width90_mm": np.full(ncase, 2 * 1.6448536269514722 * sd.mean() * 1000)})
    return result


def score_all(scratch, output, request_path):
    seal = json.loads((output / "source_seal.json").read_text())
    checked(request_path, seal["request_sha256"])
    checked(__file__, seal["implementation_sha256"])
    checked(output / "source_selection.json", seal["selection_sha256"])
    selection = json.loads((output / "source_selection.json").read_text())
    records = []
    max_parity = 0.
    prediction_hash = hashlib.sha256()
    for dlo in DLOS:
        checked(output / f"{dlo}_model.npz", seal["models"][dlo])
        pred, truth, names = load_bundle(scratch, dlo, "target")
        with np.load(output / f"{dlo}_model.npz", allow_pickle=False) as model:
            for horizon in HORIZONS:
                e, p0, ph = windows(pred, truth, horizon)
                mu = model[f"mean_{horizon}"]
                z = e - mu
                struct = model[f"structured_{horizon}"]
                pars = selection[dlo]["horizons"][str(horizon)]
                signs = np.random.default_rng(20260906).choice([-1., 1.], len(mu))
                covs = {"structured": struct, "diagonal": np.diag(np.diag(struct)),
                        "scrambled": struct * signs[:, None] * signs[None, :],
                        "empirical": model[f"empirical_{horizon}"]}
                if not all(np.allclose(np.diag(c), np.diag(struct), atol=1e-12, rtol=0) for c in covs.values()):
                    raise RuntimeError("Pre-conditioning marginal parity failed")
                for mask, nodes in MASKS.items():
                    obs, hidden, _ = indices(nodes, horizon)
                    observed = z[:, :, obs].copy()
                    for arm in ARMS:
                        post = None
                        if arm == "physical":
                            error = e[:, :, hidden]
                        elif arm == "matched_prior":
                            error = z[:, :, hidden]
                            post = struct[np.ix_(hidden, hidden)]
                        elif arm in covs or arm == "map_equivalent":
                            source_arm = "structured" if arm == "map_equivalent" else arm
                            noise = pars["empirical" if arm == "empirical" else "structured"]["parameters"][3]
                            c = covs[source_arm]
                            gain, post = gaussian_map(c, obs, hidden, noise)
                            if arm == "map_equivalent":
                                equivalent = information_map(c, obs, hidden, noise)
                                parity = float(np.max(np.abs(observed @ (equivalent - gain).T)))
                                max_parity = max(max_parity, parity)
                                if parity > 1e-8:
                                    raise RuntimeError(f"Gaussian/MAP parity failure: {parity}")
                                gain = equivalent
                            correction = observed @ gain.T
                            prediction_hash.update(np.ascontiguousarray(correction).tobytes())
                            error = z[:, :, hidden] - correction
                        elif arm == "ridge":
                            gain, _ = gaussian_map(model[f"sample_{horizon}"], obs, hidden, pars[arm]["noise"])
                            error = z[:, :, hidden] - observed @ gain.T
                        else:
                            error = simple_error(e, p0, ph, mu, horizon, nodes, arm, pars[arm]["gain"])
                        mm = metrics(error, post)
                        for i, name in enumerate(names):
                            records.append({"dlo": dlo, "trajectory": name, "horizon": horizon, "mask": mask,
                                            "arm": arm, "anchors": len(ANCHORS), "hidden_nodes": 8-len(nodes),
                                            **{key: float(val[i]) for key,val in mm.items()}})
    write(output / "case_metrics.json", records)
    keys = sorted(set().union(*(r.keys() for r in records)))
    with (output / "case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for horizon in HORIZONS:
        summary[str(horizon)] = {}
        for arm in ARMS:
            subset = [r for r in records if r["horizon"] == horizon and r["arm"] == arm]
            summary[str(horizon)][arm] = {key: float(np.mean([r[key] for r in subset])) for key in subset[0] if key not in ("dlo","trajectory","horizon","mask","arm","anchors","hidden_nodes")}
    comparisons = {}
    for baseline in ("diagonal", "empirical", "ridge", "translation", "interpolation", "frozen_interpolation"):
        per_dlo = {}
        arrays = []
        for dlo in DLOS:
            names = sorted({r["trajectory"] for r in records if r["dlo"] == dlo})
            values = {}
            for arm in ("structured", baseline):
                values[arm] = np.array([np.mean([r["rmse_mm"] for r in records if r["dlo"]==dlo and r["trajectory"]==name and r["horizon"]==30 and r["arm"]==arm]) for name in names])
            delta = values["structured"] - values[baseline]
            arrays.append(delta)
            per_dlo[dlo] = {"difference_mm": float(delta.mean()), "wins": int(np.sum(delta < -1e-9)),
                            "ties": int(np.sum(np.abs(delta)<=1e-9)), "n": len(delta),
                            "improvement_percent": float(100*(1-values["structured"].mean()/values[baseline].mean()))}
        rng = np.random.default_rng(20260906)
        boot = np.mean([d[rng.integers(0,len(d),(10000,len(d)))].mean(1) for d in arrays], axis=0)
        comparisons[baseline] = {"difference_mm": float(np.mean([d.mean() for d in arrays])),
                                 "stratified_trajectory_bootstrap95_mm": np.quantile(boot,[.025,.975]).tolist(),
                                 "per_dlo": per_dlo}
    controls = ("empirical", "ridge", "translation", "interpolation", "frozen_interpolation")
    strong = all(comparisons[a]["stratified_trajectory_bootstrap95_mm"][1] < 0 and all(d["improvement_percent"] >= 1 for d in comparisons[a]["per_dlo"].values()) for a in controls)
    result = {"status":"completed", "evidence_class":"retrospective-previously-opened-DLO4-DLO5",
              "primary":"hidden 3D point RMSE, horizon 30, equal trajectory and mask weighting",
              "physical_objects":2, "evaluation_trajectories":28, "source_trajectories":112,
              "summary":summary, "primary_comparisons":comparisons,
              "structured_model_beats_all_non_equivalent_controls":strong,
              "gaussian_map_max_abs_difference_m":max_parity,
              "conditional_corrections_sha256":prediction_hash.hexdigest(),
              "case_metrics_sha256":sha(output / "case_metrics.json"),
              "limitations":["Historical targets already opened; no fresh confirmation.",
                 "Two fixed physical objects; bootstrap conditions on these objects.",
                 "Future measured boundary trajectories are inputs to the frozen DEFORM predictor.",
                 "Artificial coordinate withholding, not real camera occlusion.",
                 "Source records also trained the frozen backbone; no independent calibration claim.",
                 "Gaussian residual posterior, not an identified physical-parameter posterior.",
                 "Equivalent deterministic Gaussian MAP must tie in point predictions."]}
    write(output / "result.json", result)
    lines = ["# Partial-observation conditioning: completed retrospective DEFORM test", "",
             "Two objects, 28 complete evaluation trajectories; 56 source trajectories per object.",
             "Fixed sparse masks; only withheld internal nodes are scored. Units: millimetres.", "",
             "| Method | Reconstruction RMSE | +10 frames RMSE | +30 frames RMSE | +30 translation-free RMSE |",
             "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        vals = [summary[str(h)][arm]["rmse_mm"] for h in HORIZONS]
        lines.append(f"| {arm} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {summary['30'][arm]['translation_free_rmse_mm']:.4f} |")
    lines += ["", "## Primary paired comparisons (structured minus comparator)", ""]
    for arm, c in comparisons.items():
        lo,hi = c["stratified_trajectory_bootstrap95_mm"]
        lines.append(f"- {arm}: {c['difference_mm']:+.4f} mm, 95% trajectory-bootstrap interval [{lo:+.4f}, {hi:+.4f}]; per-object results: {c['per_dlo']}")
    lines += ["", f"Structured-model advantage over all non-equivalent controls: **{strong}**.",
              f"Maximum Bayesian/information-form MAP prediction discrepancy: {max_parity:.3g} m.",
              "", "## Uncertainty at +30 frames", "",
              "| Method | 90% coverage | Full interval width (mm) | Joint NLL / coordinate | Normalized joint NEES |", "|---|---:|---:|---:|---:|"]
    for arm in ("structured","diagonal","scrambled","empirical"):
        r=summary["30"][arm]
        lines.append(f"| {arm} | {100*r['coverage90']:.2f}% | {r['width90_mm']:.3f} | {r['joint_nll_per_coordinate']:.4f} | {r['normalized_joint_nees']:.4f} |")
    lines += ["", "## Boundaries", ""] + ["- " + x for x in result["limitations"]]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("source","fit","target","score"))
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.scratch.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    request=json.loads(args.request.read_text())
    if request["contract"]!="partial-observation-conditioning-v1" or request["horizons"]!=list(HORIZONS) or request["masks"]!=MASKS:
        raise ValueError("Protocol and executable disagree")
    if os.environ.get("GITHUB_RUN_ATTEMPT","1")!="1":
        raise RuntimeError("Use a new explicitly versioned request, not an Actions rerun")
    try:
        if args.phase in ("source","target"):
            prepare(request,args.scratch,args.output,args.phase)
        elif args.phase=="fit":
            fit_all(args.scratch,args.output,args.request)
        else:
            score_all(args.scratch,args.output,args.request)
        write(args.output / f"{args.phase}_execution.json", {"status":"success","github_sha":os.environ.get("GITHUB_SHA"),"run_id":os.environ.get("GITHUB_RUN_ID"),"python":platform.python_version(),"numpy":np.__version__})
    except Exception as exc:
        write(args.output / "failure.json", {"phase":args.phase,"exception":repr(exc),"traceback":traceback.format_exc(),"scientific_result_available":False})
        raise


if __name__=="__main__":
    main()
