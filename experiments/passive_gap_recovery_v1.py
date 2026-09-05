"""Passive gaps on recorded cloth motion; source-only fitting, no robot actions.

This research instrument uses the repository's limited spring-mesh pilot, not
released PhysTwin or an RGB-D provider. The likelihood is Gaussian; its filtering
mean equals same-model sequential MAP. Missingness is imposed on real measured
trajectories, with driven corners continuously observed and never scored.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

PROTOCOL = {
    "study_id": "passive-gap-recovery-real-v1",
    "dataset_record": "14644526",
    "source": "all 32 already-open free-hanging shake recordings",
    "target": "all 32 already-open free-hanging twist recordings",
    "excluded": "all 56 collision recordings; no numerical access",
    "evidence_class": "retrospective real-trajectory mechanism test with imposed missing observations",
    "information": "causal all-marker observations; current driven corners always observed; no future free-marker information",
    "physical_prior": "existing limited equal-mass spring-mesh pilot; source-selected stiffness/damping per specimen",
    "sample_stride": 4,
    "rate_hz": 30,
    "initialization_seconds": 1.0,
    "forecast_seconds": 5.0,
    "gap_onsets_after_cutoff_frames": [15, 60, 105],
    "gap_lengths_frames": [3, 9, 18],
    "recovery_frames": 9,
    "source_objective": "equal recording and condition mean prior RMSE; clean plus three gap/recovery conditions",
    "bayes_rho": [0.8, 0.95, 1.0],
    "bayes_q_over_r": [0.0001, 0.01, 1.0, 100.0],
    "measurement_variance_m2": 0.000001,
    "deterministic_alpha": [0.25, 0.5, 0.8, 1.0],
    "deterministic_beta": [0.0, 0.1, 0.3, 0.6, 1.0],
    "deterministic_rho": [0.8, 0.95, 1.0],
    "primary": "gap-and-recovery prior RMSE vs source-selected deterministic method; equal specimen, recording, gap-length weights",
    "secondary": "gap and recovery separately, clean cost, stationary-gain and covariance-reset ablations, same-prior restart diagnostic",
    "primary_gate": "paired specimen bootstrap 95% interval below zero, at least 5% relative RMSE reduction, clean-cost upper interval <= 0.5 mm",
    "bootstrap_repetitions": 10000,
    "bootstrap_seed": 1464452606,
    "uncertainty_metrics": "descriptive nominal-90% coordinate coverage, width and Gaussian NLL; no calibration claim",
    "raw_data_upload": False,
    "hyperparameters_changed_after_target": False,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def save(path, value):
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype="<f8")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def patterns(n, cutoff):
    """Time-only masks, fixed without geometry or outcome information."""
    result = {}
    for length in [0] + PROTOCOL["gap_lengths_frames"]:
        observed = np.ones(n, dtype=bool)
        gap = np.zeros(n, dtype=bool)
        recovery = np.zeros(n, dtype=bool)
        for offset in PROTOCOL["gap_onsets_after_cutoff_frames"]:
            start = cutoff + offset
            stop = start + length
            if stop + PROTOCOL["recovery_frames"] > n:
                raise ValueError("Recording does not cover the complete fixed gap roster")
            if length:
                observed[start:stop] = False
                gap[start:stop] = True
                recovery[stop:stop + PROTOCOL["recovery_frames"]] = True
        score = np.arange(n) > cutoff if length == 0 else gap | recovery
        result["clean" if length == 0 else f"gap{length}"] = (observed, score, gap, recovery)
    return result


def stationary(rho, ratio):
    f = np.array([[1.0, rho], [0.0, rho]])
    q = ratio * np.array([[0.25, 0.5], [0.5, 1.0]])
    p = np.eye(2)
    for _ in range(2000):
        prior = f @ p @ f.T + q
        k = prior[:, 0] / (prior[0, 0] + 1.0)
        updated = prior - np.outer(k, prior[0])
        if np.max(abs(updated - p)) < 1e-12:
            p = updated
            break
        p = updated
    return f, q, p, k


def infer(observed, baseline, config, *, details=False):
    """Only masked observations enter inference. Output is BEFORE each update.

    Each coordinate has a 2D discrepancy/increment state. Covariances are in
    measurement-variance units, avoiding numerical scale differences. The MAP
    route uses an information-form update independent of the Kalman correction.
    """
    observed = np.asarray(observed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if observed.shape != baseline.shape or observed.ndim != 3:
        raise ValueError("Expected matching T x marker x coordinate arrays")
    if not np.isfinite(baseline).all() or not np.isfinite(observed[0]).all():
        raise ValueError("Finite baseline and initial observations are required")
    mode = config["mode"]
    n, nodes, dims = observed.shape
    r = observed - baseline
    result = np.empty_like(observed)
    var = np.empty_like(observed)
    m = np.zeros((nodes, dims, 2))
    m[..., 0] = r[0]
    rho = config.get("rho", 1.0)
    f, q, pss, kss = stationary(rho, config.get("ratio", 1.0))
    p = np.broadcast_to(pss, (nodes, dims, 2, 2)).copy()
    missing = np.zeros((nodes, dims), dtype=bool)
    restarts = []
    result[0] = observed[0]
    var[0] = PROTOCOL["measurement_variance_m2"]
    for t in range(1, n):
        m = m @ f.T
        p = f @ p @ f.T + q
        valid = np.isfinite(r[t])
        returning = valid & missing
        if mode == "reset_covariance" and np.any(returning):
            p[returning] = f @ pss @ f.T + q
        result[t] = baseline[t] + m[..., 0]
        var[t] = (p[..., 0, 0] + 1.0) * PROTOCOL["measurement_variance_m2"]
        error = np.where(valid, r[t] - m[..., 0], 0.0)
        if mode in ("bayes", "map", "reset_covariance"):
            k = p[..., :, 0] / (p[..., 0, 0, None] + 1.0)
        elif mode == "steady_gain":
            k = np.broadcast_to(kss, m.shape)
        elif mode == "alpha_beta":
            k = np.broadcast_to(np.array([config["alpha"], config["beta"]]), m.shape)
        elif mode == "physics_only":
            k = np.zeros_like(m)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if details and mode == "bayes" and np.any(returning) and t + 1 < n:
            full = (m + k * error[..., None]) @ f.T
            reset = (m + kss * error[..., None]) @ f.T
            restarts.append((t + 1, returning.copy(), baseline[t + 1] + full[..., 0], baseline[t + 1] + reset[..., 0]))
        if mode == "map":
            # Gaussian mean/mode identity, implemented by information addition.
            precision = np.linalg.inv(p)
            rhs = np.einsum("...ij,...j->...i", precision, m)
            precision[..., 0, 0] += valid
            rhs[..., 0] += np.where(valid, r[t], 0.0)
            m = np.linalg.solve(precision, rhs[..., None])[..., 0]
            p = np.linalg.inv(precision)
        else:
            m += k * error[..., None]
            if mode in ("bayes", "reset_covariance"):
                # Joseph form preserves PSD through long missing intervals.
                a = np.broadcast_to(np.eye(2), p.shape).copy()
                a[..., :, 0] -= k * valid[..., None]
                p = a @ p @ np.swapaxes(a, -1, -2) + np.einsum("...i,...j->...ij", k, k) * valid[..., None, None]
        missing = ~valid
    if mode == "physics_only":
        result = baseline.copy()
    if not np.isfinite(result).all() or not np.isfinite(var).all():
        raise FloatingPointError("Nonfinite forecast; never silently drop a case")
    return result, var, restarts


def rmse(pred, truth, times):
    valid = times[:, None] & np.isfinite(truth).all(axis=-1)
    if not np.any(valid):
        raise ValueError("No scored markers")
    return float(1000 * np.sqrt(np.mean(np.sum((pred[valid] - truth[valid]) ** 2, axis=-1))))


def score(pred, variance, truth, times):
    valid = times[:, None] & np.isfinite(truth).all(axis=-1)
    error = pred[valid] - truth[valid]
    v = variance[valid]
    return {
        "rmse_mm": rmse(pred, truth, times),
        "p95_marker_error_mm": float(1000 * np.quantile(np.linalg.norm(error, axis=-1), .95)),
        "coverage90": float(np.mean(abs(error) <= 1.6448536269514722 * np.sqrt(v))),
        "full_width90_mm": float(1000 * np.mean(2 * 1.6448536269514722 * np.sqrt(v))),
        "coordinate_nll": float(np.mean(.5 * (np.log(2 * np.pi * v) + error ** 2 / v))),
        "marker_samples": int(valid.sum()),
    }


def candidates():
    ab = [dict(mode="alpha_beta", alpha=a, beta=b, rho=r) for a, b, r in itertools.product(PROTOCOL["deterministic_alpha"], PROTOCOL["deterministic_beta"], PROTOCOL["deterministic_rho"])]
    bayes = [dict(mode="bayes", rho=r, ratio=q) for r, q in itertools.product(PROTOCOL["bayes_rho"], PROTOCOL["bayes_q_over_r"])]
    ema = [c for c in ab if c["beta"] == 0.0 and c["rho"] == 1.0]
    return ab, bayes, ema


def source_loss(records, bases, config, raw=False):
    losses = []
    for rec, base in zip(records, bases):
        if raw:
            base = np.zeros_like(base)
        for obs_mask, scoring, _, _ in patterns(len(base), rec["cutoff"]).values():
            y = np.where(obs_mask[:, None, None], rec["truth"], np.nan)
            pred, _, _ = infer(y, base, config)
            losses.append(rmse(pred, rec["truth"], scoring))
    return float(np.mean(losses))


def fit(specimen, source):
    # Equal contribution from each source recording; no target is passed here.
    bank_losses = []
    for k in range(source[0]["bank"].shape[0]):
        bank_losses.append(np.mean([rmse(r["bank"][k], r["truth"], np.arange(len(r["truth"])) > r["cutoff"]) for r in source]))
    chosen = int(np.argmin(bank_losses))
    bases = [r["bank"][chosen] for r in source]
    ab, bs, ema = candidates()
    selections = {}
    losses = {}
    for name, grid, raw in [("alpha_beta", ab, False), ("raw_alpha_beta", ab, True), ("exponential", ema, False), ("bayes", bs, False)]:
        values = [source_loss(source, bases, c, raw) for c in grid]
        i = int(np.argmin(values))
        selections[name] = grid[i]
        losses[name] = values[i]
    selections["last_residual"] = dict(mode="alpha_beta", alpha=1.0, beta=0.0, rho=1.0)
    selections["raw_persistence"] = selections["last_residual"]
    selections["raw_constant_velocity"] = dict(mode="alpha_beta", alpha=1.0, beta=1.0, rho=1.0)
    selections["physics_only"] = dict(mode="physics_only")
    for name in ["last_residual", "raw_persistence", "raw_constant_velocity", "physics_only"]:
        losses[name] = source_loss(source, bases, selections[name], name.startswith("raw_"))
    selections["steady_gain"] = {**selections["bayes"], "mode": "steady_gain"}
    selections["reset_covariance"] = {**selections["bayes"], "mode": "reset_covariance"}
    selections["same_model_map"] = {**selections["bayes"], "mode": "map"}
    losses["steady_gain"] = source_loss(source, bases, selections["steady_gain"])
    deterministic = [n for n in losses if n != "bayes"]
    reference = min(deterministic, key=lambda n: (losses[n], n))
    return {"specimen": specimen, "physical_bank_index": chosen, "physical_source_rmse_mm": [float(x) for x in bank_losses], "configs": selections, "source_rmse_mm": losses, "source_selected_reference": reference}


def read_records(root, output):
    # The existing audit hashes protected files, but does not numerically parse them.
    from experiments.tracking_cloth_deformation_v1.data import audit_dataset, input_view, infer_source_scale, read_prefix, scoring_view
    from experiments.tracking_cloth_deformation_v1.model import parameter_bank, rollout
    old_path = Path(__file__).parent / "tracking_cloth_deformation_v1" / "protocol.json"
    old = json.loads(old_path.read_text())
    cases, inventory = audit_dataset(root, old)
    save(output / "inventory.json", inventory)
    sources = [c for c in cases if c.motion == "shake"]
    targets = [c for c in cases if c.motion == "twist"]
    scale = {}
    for case in sources:
        _, prefix = read_prefix(case, old["prefix_seconds"])
        s = infer_source_scale(case, prefix)
        if case.specimen in scale and scale[case.specimen] != s:
            raise ValueError("Source coordinate scales disagree")
        scale[case.specimen] = s

    def load(case, bank_indices=None):
        inputs = input_view(case, old, scale[case.specimen])
        free = np.ones(len(inputs.order), dtype=bool)
        free[inputs.corners] = False
        # Recorded boundary is an always-available current-time input, not future
        # free-state data. Spring integration at t never depends on boundary t+1.
        pars = parameter_bank(old)
        indices = list(range(len(pars))) if bank_indices is None else bank_indices
        bank = np.stack([rollout(inputs, pars[k], old, False)[:, free] for k in indices])
        truth = scoring_view(case, inputs)[:, free]
        if not np.isfinite(truth[0]).all():
            raise ValueError("Missing initialization")
        return {"name": case.path.name, "specimen": case.specimen, "material": case.material, "motion": case.motion, "cutoff": inputs.cutoff, "truth": truth, "bank": bank, "times": inputs.times}
    return sources, targets, load


def interval(differences, seed=1464452606):
    x = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(PROTOCOL["bootstrap_repetitions"], len(x)))
    return [float(v) for v in np.quantile(x[draws].mean(axis=1), [.025, .975])]


def summarize(rows, fits, parity, restart):
    specimens = sorted(fits)
    methods = list(next(iter(fits.values()))["configs"])
    def group(method, conditions, phase="combined"):
        result = []
        for specimen in specimens:
            name = fits[specimen]["source_selected_reference"] if method == "source_selected" else method
            values = [r["rmse_mm"] for r in rows if r["specimen"] == specimen and r["method"] == name and r["condition"] in conditions and r["phase"] == phase]
            if not values:
                raise ValueError("Missing specimen in aggregate")
            result.append(float(np.mean(values)))
        return np.array(result)
    gap_conditions = [f"gap{x}" for x in PROTOCOL["gap_lengths_frames"]]
    table = {}
    for method in methods + ["source_selected"]:
        table[method] = {"gap_recovery_rmse_mm": float(group(method, gap_conditions).mean()), "gap_rmse_mm": float(group(method, gap_conditions, "gap").mean()), "recovery_rmse_mm": float(group(method, gap_conditions, "recovery").mean()), "clean_rmse_mm": float(group(method, ["clean"]).mean())}
    b = group("bayes", gap_conditions)
    contrasts = {}
    for method in methods + ["source_selected"]:
        reference = group(method, gap_conditions)
        d = b - reference
        # Specimen bootstrap is primary; material-cluster sensitivity acknowledges
        # that A2/A3 specimens of a fabric type are not broad material diversity.
        material_d = [float(np.mean([d[i] for i, s in enumerate(specimens) if s.startswith(m + "_")])) for m in ("cotton", "denim", "polyester", "wool")]
        contrasts[method] = {"difference_mm": float(d.mean()), "specimen_bootstrap95_mm": interval(d), "material_bootstrap95_mm": interval(material_d), "relative_improvement_pct": float(100 * (reference.mean() - b.mean()) / reference.mean()), "specimen_wins": int(np.sum(d < -1e-9)), "specimen_ties": int(np.sum(abs(d) <= 1e-9)), "specimen_differences_mm": dict(zip(specimens, d.tolist()))}
    clean_delta = group("bayes", ["clean"]) - group("source_selected", ["clean"])
    primary = contrasts["source_selected"]
    gate = primary["specimen_bootstrap95_mm"][1] < 0 and primary["relative_improvement_pct"] >= 5 and interval(clean_delta)[1] <= .5
    return {"protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL), "specimens": len(specimens), "target_recordings": len({r["recording"] for r in rows}), "table": table, "contrasts_bayes_minus_reference": contrasts, "clean_cost_mm": float(clean_delta.mean()), "clean_cost_bootstrap95_mm": interval(clean_delta), "primary_gate_passed": bool(gate), "same_model_map_max_abs_mean_difference_m": parity, "same_prior_recovery_diagnostics": restart, "interpretation": "Only recursive covariance versus fixed-gain/point controls on masked real motion. Gaussian MAP parity prevents a claim of unique posterior-mean superiority over equivalent MAP. Not a PhysTwin benchmark, RGB-D test, calibrated-uncertainty claim, independent fresh-object confirmation or robot-action experiment."}


def report(result):
    lines = ["# Passive gap-and-recovery: real cloth trajectories", "", f"Primary gate passed: **{result['primary_gate_passed']}**.", "", f"{result['target_recordings']} twist recordings; {result['specimens']} fabric/size specimens. Source: 32 shake recordings. All were already opened by earlier studies; this is a new retrospective mechanism question.", "", "| Method | Gap + recovery RMSE (mm) | Gap RMSE (mm) | Recovery RMSE (mm) | Clean RMSE (mm) |", "|---|---:|---:|---:|---:|"]
    for name, v in result["table"].items():
        lines.append(f"| {name} | {v['gap_recovery_rmse_mm']:.4f} | {v['gap_rmse_mm']:.4f} | {v['recovery_rmse_mm']:.4f} | {v['clean_rmse_mm']:.4f} |")
    lines += ["", "## Paired Bayesian-minus-reference contrasts", ""]
    for name in ["source_selected", "last_residual", "alpha_beta", "raw_alpha_beta", "steady_gain", "reset_covariance", "same_model_map"]:
        v = result["contrasts_bayes_minus_reference"][name]
        lines.append(f"- {name}: {v['difference_mm']:.4f} mm; specimen bootstrap 95% {v['specimen_bootstrap95_mm']}; improvement {v['relative_improvement_pct']:.3f}%; wins {v['specimen_wins']}/8.")
    lines += ["", f"Clean cost: {result['clean_cost_mm']:.4f} mm; 95% interval {result['clean_cost_bootstrap95_mm']}.", f"Gaussian mean/MAP maximum difference: {result['same_model_map_max_abs_mean_difference_m']:.3g} m.", "", result["interpretation"], "", "Driven corners are continuously observed conditioning inputs and excluded from scores. Native motion-capture values are the reference; no synthetic motion, noise, robot probing, target tuning, or collision-recording numerical access. All scores are pre-update. Uncertainty scores are descriptive and appear only for the Bayesian arm in the detailed CSV."]
    return "\n".join(lines) + "\n"


def run(root, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", PROTOCOL)
    save(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "runner": os.environ.get("RUNNER_NAME"), "run_id": os.environ.get("GITHUB_RUN_ID")})
    sources, targets, load = read_records(root, output)
    groups = {}
    for case in sources:
        groups.setdefault(case.specimen, []).append(case)
    def source_job(item):
        specimen, cases = item
        records = [load(c) for c in cases]
        value = fit(specimen, records)
        print("SOURCE_FIT", specimen, value["source_selected_reference"], flush=True)
        return specimen, value
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fits = dict(pool.map(source_job, sorted(groups.items())))
    save(output / "source_fit.json", fits)
    fit_digest = digest(fits)
    print("SOURCE_FIT_SEALED", fit_digest, flush=True)
    pending = []
    seals = []
    parity = 0.0
    restart = []
    for case in targets:
        fitted = fits[case.specimen]
        rec = load(case, [fitted["physical_bank_index"]])
        base = rec["bank"][0]
        for condition, (mask, scoring, gap, recovery) in patterns(len(base), rec["cutoff"]).items():
            observed = np.where(mask[:, None, None], rec["truth"], np.nan)
            predictions = {}
            for name, config in fitted["configs"].items():
                b = np.zeros_like(base) if name.startswith("raw_") else base
                predictions[name] = infer(observed, b, config, details=True)
            parity = max(parity, float(np.max(abs(predictions["bayes"][0] - predictions["same_model_map"][0]))))
            if parity > 1e-8:
                raise AssertionError("Gaussian mean/MAP parity violated")
            seals.append({"recording": rec["name"], "condition": condition, "observed_sha256": array_hash(observed), "predictions": {name: array_hash(p[0], p[1]) for name, p in predictions.items()}})
            pending.append((rec, condition, scoring, gap, recovery, predictions))
        print("PREDICTED", rec["name"], flush=True)
    save(output / "prediction_seal.json", {"source_fit_sha256": fit_digest, "protocol_sha256": digest(PROTOCOL), "prediction_digest": digest(seals), "records": seals, "scoring_started": False})
    print("ALL_TARGET_PREDICTIONS_SEALED", digest(seals), flush=True)
    rows = []
    for rec, condition, scoring, gap, recovery, predictions in pending:
        for name, (pred, variance, _) in predictions.items():
            phases = [("combined", scoring)] if condition == "clean" else [("combined", scoring), ("gap", gap), ("recovery", recovery)]
            for phase, time_mask in phases:
                values = score(pred, variance, rec["truth"], time_mask)
                if name != "bayes":
                    # Covariances of heuristic arms are not meaningful forecasts.
                    values = {k: v for k, v in values.items() if k not in ("coverage90", "full_width90_mm", "coordinate_nll")}
                rows.append({"recording": rec["name"], "specimen": rec["specimen"], "condition": condition, "phase": phase, "method": name, **values})
        for t, valid, full, reset in predictions["bayes"][2]:
            valid &= np.isfinite(rec["truth"][t])
            if np.any(valid):
                restart.append({"recording": rec["name"], "specimen": rec["specimen"], "condition": condition, "frame": t, "adaptive_next_coordinate_rmse_mm": float(1000 * np.sqrt(np.mean((full[valid] - rec["truth"][t][valid]) ** 2))), "fixed_next_coordinate_rmse_mm": float(1000 * np.sqrt(np.mean((reset[valid] - rec["truth"][t][valid]) ** 2)))})
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    result = summarize(rows, fits, parity, restart)
    result["source_fit_sha256"] = fit_digest
    result["prediction_seal_sha256"] = digest(seals)
    save(output / "result.json", result)
    text = report(result)
    (output / "report.md").write_text(text)
    print(text, flush=True)
    save(output / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})


def self_test():
    rng = np.random.default_rng(17)
    y = rng.normal(size=(200, 3, 3)).cumsum(axis=0) * .002
    base = np.zeros_like(y)
    masked = y.copy()
    masked[60:78] = np.nan
    masked[100:110, 0] = np.nan
    c = dict(mode="bayes", rho=.95, ratio=.01)
    p, v, _ = infer(masked, base, c)
    m, _, _ = infer(masked, base, {**c, "mode": "map"})
    assert np.max(abs(p - m)) < 1e-9
    changed = masked.copy()
    changed[120:] += 100
    p2, _, _ = infer(changed, base, c)
    np.testing.assert_array_equal(p[:121], p2[:121])
    # At the return frame predictions cannot depend on the incoming value.
    changed = masked.copy()
    changed[78] += 100
    p2, _, _ = infer(changed, base, c)
    np.testing.assert_array_equal(p[:79], p2[:79])
    assert not np.array_equal(p[79], p2[79])
    assert np.min(v) > 0 and v[77].mean() > v[59].mean()
    # Last residual really is the last available observation for zero baseline.
    last, _, _ = infer(masked, base, dict(mode="alpha_beta", alpha=1., beta=0., rho=1.))
    np.testing.assert_allclose(last[61:79], np.broadcast_to(y[59], last[61:79].shape))
    reset, _, _ = infer(masked, base, {**c, "mode": "reset_covariance"})
    np.testing.assert_array_equal(reset[:79], p[:79])
    assert np.max(abs(reset[79] - p[79])) > 0
    pats = patterns(181, 30)
    for length in (3, 9, 18):
        seen, scored, gap, recovery = pats[f"gap{length}"]
        assert gap.sum() == 3 * length and recovery.sum() == 27
        assert np.array_equal(~seen, gap) and not np.any(gap & recovery)
        assert np.array_equal(scored, gap | recovery)
    assert interval(np.ones(8)) == [1.0, 1.0]
    print("SELF_TEST_PASS: MAP parity, causal suffix, pre-update timing, gap covariance growth, exact last-residual, reset ablation, fixed masks, bootstrap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.dataset_root is None or args.output is None:
            parser.error("--dataset-root and --output are required")
        start = time.monotonic()
        run(args.dataset_root, args.output, args.workers)
        print(f"Elapsed seconds: {time.monotonic() - start:.2f}")
