#!/usr/bin/env python3
"""Retrospective real-data audit; frozen means, source-only covariance controls.

No robot actions, new holdouts, model selection, or source/target replacements.
The historical v6 reproduction is unchanged; capture only observes its outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FRACTIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
ARMS = ("full_v6", "diagonal_v6", "scrambled_v6",
        "empirical_rank_matched", "empirical_query_gaussian",
        "diagonal_query_recalibrated")
SEED = 260906
REPETITIONS = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def empirical_model(errors, marginal, rank):
    """PCA residual covariance, fixed 10% diagonal shrinkage, matched rank/marginals.

    Uses the same centered source residuals as the original covariance estimator.
    No target arrays or fitted hyperparameters enter this function.
    """
    require(errors.ndim == 2 and len(errors) > 1, "invalid source residual matrix")
    require(np.isfinite(errors).all(), "nonfinite source residual")
    require(marginal.shape == (errors.shape[1],), "marginal shape mismatch")
    require(np.all(marginal > 0), "nonpositive reference marginals")
    centered = errors - errors.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    r = min(int(rank), len(singular))
    factor = (vt[:r].T * singular[:r]) / math.sqrt(len(centered))
    factor *= math.sqrt(0.9)
    empirical_diag = np.mean(centered * centered, axis=0)
    diagonal = np.maximum(empirical_diag - np.sum(factor * factor, axis=1), 1e-12)
    scale = np.sqrt(marginal / (diagonal + np.sum(factor * factor, axis=1)))
    return SimpleNamespace(diagonal=diagonal * scale**2, factor=factor * scale[:, None], multiplier=1.0)


def selected_metrics(probabilities, labels):
    """Equal counts per object, pooled query/windows; stable index breaks ties."""
    p = np.asarray(probabilities).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    require(p.shape == y.shape and len(p) > 0, "empty or mismatched selection arrays")
    require(np.isfinite(p).all() and np.isfinite(y).all(), "nonfinite selection input")
    order = np.lexsort((np.arange(len(p)), p))
    output = {}
    for fraction in FRACTIONS:
        count = max(1, min(len(p), int(math.floor(fraction * len(p)))))
        chosen = order[:count]
        harm = float(y[chosen].mean())
        output[str(fraction)] = {"count": count, "acceptance": count / len(p),
                                 "accepted_event_rate": harm,
                                 "decision_loss": (float(y[chosen].sum()) + 0.1 * (len(p) - count)) / len(p)}
    return output


def object_audit(v6, v3, row, capture, source_truth, target_truth, out):
    object_id = row["object_id"]
    x = np.asarray(capture.source_residuals, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    errors = np.asarray(capture.target_errors, dtype=float)
    bank = v6.query_bank(target_truth.shape[1])
    names = list(bank)
    w = np.column_stack([bank[name][0] for name in names])
    source_q = x @ w
    source_truth_q = source_truth @ w
    target_error_q = errors @ w
    target_truth_q = target_truth @ w
    target_mean_q = target_truth_q - target_error_q
    models = v6.covariance_arms(v3.base, capture.covariance,
        seed=v6.stable_seed(260901, object_id, "scrambled-factor"))
    original = list(models.values())
    marginal = v6.marginal_variance(original[0])
    empirical = empirical_model(x, marginal, capture.covariance.factor.shape[1])
    parity = float(np.max(np.abs(v6.marginal_variance(empirical) - marginal)))
    require(parity < 1e-10, "empirical marginal parity failed")
    predictions = {a: [] for a in ARMS}
    metrics = {a: [] for a in ARMS}
    labels_list, variances, thresholds, radii, raw_list = [], [], [], [], []
    # All covariance fits above depend on source residuals only.
    for j, name in enumerate(names):
        weight, event = bank[name]
        raw = {a: v6.covariance_query_variance(model, weight) for a, model in models.items()}
        calibration = v6.source_query_calibration(x, source_truth, weight, raw,
            event=event, probability=0.9, event_quantile=0.9)
        empirical_var = v6.covariance_query_variance(empirical, weight)
        mse = max(float(np.mean(source_q[:, j]**2)), 1e-12)
        scale = calibration["shared_variance_scale"]
        vs = [raw[a] * scale for a in models] + [empirical_var * scale, mse,
            raw["diagonal_marginal_matched"] * (mse / raw["diagonal_marginal_matched"])]
        threshold = calibration["event_threshold"]
        radius = calibration["shared_radius_multiplier"]
        truth = target_truth_q[:, j]
        labels = (truth > threshold) if event == "upper" else (np.abs(truth) > threshold)
        y = labels.astype(float)
        labels_list.append(y)
        variances.append(vs)
        thresholds.append(threshold)
        radii.append(radius)
        raw_list.append(list(raw.values()) + [empirical_var])
        for arm, variance in zip(ARMS, vs, strict=True):
            p = np.clip(v6.event_probability(target_mean_q[:, j], variance, threshold, event), 1e-9, 1 - 1e-9)
            predictions[arm].append(p)
            sq = target_error_q[:, j]**2 / variance
            flag = p > 0.1
            metrics[arm].append({
                "brier": float(np.mean((p-y)**2)),
                "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
                "query_nll": float(np.mean(0.5*(math.log(2*math.pi*variance)+sq))),
                "coverage90": float(np.mean(np.abs(target_error_q[:, j]) <= radius*math.sqrt(variance))),
                "width90": float(2*radius*math.sqrt(variance)),
                "nanees": float(np.mean(sq)),
                "decision_loss": float(np.mean(np.where(flag, 0.1, y))),
                "acceptance": float(np.mean(~flag)),
            })
    labels = np.column_stack(labels_list)
    predictions = {a: np.column_stack(v) for a, v in predictions.items()}
    scalar_fit = {a: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]} for a, ms in metrics.items()}
    matched = {a: selected_metrics(predictions[a], labels) for a in ARMS}
    probability_parity = float(np.max(np.abs(predictions["full_v6"] - predictions["empirical_query_gaussian"])))
    variance_array = np.asarray(variances)
    # Save sufficient statistics and exact per-window decisions, not raw datasets.
    np.savez_compressed(out / "predictions" / f"{object_id}.npz",
        query_names=np.asarray(names), source_query_errors=source_q,
        source_query_truth=source_truth_q, target_query_mean=target_mean_q,
        target_query_error=target_error_q, labels=labels, thresholds=np.asarray(thresholds),
        radii=np.asarray(radii), variances=variance_array, raw_variances=np.asarray(raw_list),
        arm_names=np.asarray(ARMS), **{f"prob_{a}": p for a, p in predictions.items()})
    return {"object_id": object_id, "window_count": len(target_truth), "query_count": len(bank),
            "source_episode_ids": row["source_episode_ids"], "target_episode_id": row["target_episode_id"],
            "source_residual_sha256": digest(x), "predictive_mean_sha256": v6.array_digest(target_truth-errors),
            "empirical_rank": empirical.factor.shape[1], "reference_rank": capture.covariance.factor.shape[1],
            "coordinate_marginal_max_abs": parity, "empirical_probability_max_abs": probability_parity,
            "empirical_variance_max_abs": float(np.max(np.abs(variance_array[:, 0]-variance_array[:, 4]))),
            "metrics": scalar_fit, "matched": matched}


def paired(values):
    vector = np.asarray(values, dtype=float)
    # Ignore only numerical roundoff for the scientific win-count, not raw differences.
    rng = np.random.default_rng(SEED)
    samples = vector[rng.integers(0, len(vector), size=(REPETITIONS, len(vector)))].mean(axis=1)
    return {"mean_difference": float(vector.mean()),
            "bootstrap95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "wins_ties_losses": [int((vector < -1e-12).sum()), int((abs(vector) <= 1e-12).sum()), int((vector > 1e-12).sum())]}


def summarize(rows):
    metrics = {a: {k: float(np.mean([r["metrics"][a][k] for r in rows])) for k in rows[0]["metrics"][a]} for a in ARMS}
    contrasts = {}
    for arm in ARMS[1:]:
        contrasts[arm] = {k: paired([r["metrics"]["full_v6"][k]-r["metrics"][arm][k] for r in rows]) for k in ("brier", "decision_loss", "query_nll")}
        contrasts[arm]["matched"] = {str(f): paired([r["matched"]["full_v6"][str(f)]["accepted_event_rate"]-r["matched"][arm][str(f)]["accepted_event_rate"] for r in rows]) for f in FRACTIONS}
    matched = {a: {str(f): {k: float(np.mean([r["matched"][a][str(f)][k] for r in rows])) for k in ("acceptance", "accepted_event_rate", "decision_loss")} for f in FRACTIONS} for a in ARMS}
    return metrics, contrasts, matched


def self_test():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 8)); marginal = np.linspace(0.1, 1.0, 8)
    model = empirical_model(x, marginal, 3)
    cov = np.diag(model.diagonal) + model.factor @ model.factor.T
    require(np.allclose(np.diag(cov), marginal), "test: diagonal mismatch")
    require(np.linalg.eigvalsh(cov).min() > 0, "test: not PSD")
    q = (x-x.mean(0)) @ np.ones(8)
    mse = np.mean(q*q)
    for raw in (0.01, 1.0, 20.0):
        require(np.isclose(raw*(mse/raw), mse), "test: independent calibration identity")
    p = np.linspace(0, 1, 100); y = p > 0.8
    a = selected_metrics(p, y); b = selected_metrics(p**2, y)
    require(a == b, "test: monotone rank invariance")
    require(selected_metrics(p, y) == a, "test: nondeterministic ties")
    for f in FRACTIONS:
        require(a[str(f)]["count"] == int(100*f), "test: unmatched count")
    # Pairing must bootstrap objects and recognize an exact null.
    require(paired([0.0, 0.0, 0.0])["bootstrap95"] == [0.0, 0.0], "test: null interval")
    print("SELF_TEST_PASS: PSD, rank, marginals, calibration identity, matched counts, ranking, object bootstrap", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    require(args.output is not None, "--output required")
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True); (out/"predictions").mkdir(exist_ok=True)
    recovery = load(root/"recovery/scripts/science/run_deform360_dependence_query_v6_bound_carrier_recovery_v1.py", "audit_recovery")
    original_load = recovery.load_module
    rows = []
    def observing_load(path, name):
        module = original_load(path, name)
        if name == "deform360_dependence_query_v6_original":
            original_capture = module.evaluate_object_with_capture
            def observing_capture(*a, **kw):
                result = original_capture(*a, **kw)
                row, capture, st, tt = result
                audited = object_audit(module, a[0], row, capture, st, tt, out)
                rows.append(audited)
                dump(out/"partial.json", {"status": "incomplete", "objects": rows})
                return result
            module.evaluate_object_with_capture = observing_capture
        return module
    recovery.load_module = observing_load
    _, reproduction = recovery.run(
        base_runner_path=root/"v6/scripts/science/run_deform360_dependence_query_v6.py",
        protocol_path=root/"v6/protocols/deform360_dependence_query_v6.json",
        parent_protocol_path=root/"parent/protocols/deform360_untouched_confirmation_v5.json",
        parent_result_path=root/"parent-evidence/result.json",
        readiness_path=root/"parent-evidence/bound-readiness.json",
        data_root=Path("/mnt/seagate10tb/florianpfaff/datasets/deform360"),
        parent_control_root=root/"parent", frozen_root=root/"frozen")
    require(len(rows) == 92, "incomplete 92-object roster")
    expected = json.loads((root/"v6-evidence/result.json").read_text())
    prior = {r["object_id"]: r for r in expected["objects"]}
    require(set(prior) == {r["object_id"] for r in rows}, "object roster changed")
    for r in rows:
        p = prior[r["object_id"]]
        require(r["predictive_mean_sha256"] == p["predictive_mean_sha256"], "mean prediction changed")
        for new, old in (("brier", "event_brier"), ("decision_loss", "decision_loss"), ("query_nll", "query_nll")):
            require(np.isclose(r["metrics"]["full_v6"][new], p["arm_summary"]["full_low_rank"][old], atol=1e-11, rtol=1e-11), "parent score failed to reproduce")
        for f in FRACTIONS:
            require(len({r["matched"][a][str(f)]["count"] for a in ARMS}) == 1, "counts not matched")
    require(all(r["parent_point_result_exact"] for r in reproduction["objects"]), "point-model parity failed")
    metrics, contrasts, matched = summarize(rows)
    null = max(r["empirical_probability_max_abs"] for r in rows) < 1e-10
    result = {"schema": "deform360-empirical-null-audit-v1", "status": "complete",
        "run_id": os.environ.get("GITHUB_RUN_ID"), "commit": os.environ.get("GITHUB_SHA"),
        "retrospective": True, "fresh_confirmation": False,
        "source_only_covariance_fit": True, "point_mean_exactly_reproduced": True,
        "object_count": len(rows), "query_count": 5, "objects": rows,
        "metrics": metrics, "contrasts": contrasts, "matched": matched,
        "empirical_query_probability_equivalent": null,
        "max_probability_difference": max(r["empirical_probability_max_abs"] for r in rows),
        "max_variance_difference": max(r["empirical_variance_max_abs"] for r in rows),
        "max_coordinate_marginal_difference": max(r["coordinate_marginal_max_abs"] for r in rows),
        "decision": "no_unique_bayesian_value_for_these_scalar_queries" if null else "inspect_paired_controls",
        "boundary": "Scalar query empirical baseline need not preserve original coordinate marginals. The rank-matched covariance comparator does. No calibration, whole-trajectory, robot safety or fresh-generalization claim."}
    dump(out/"result.json", result)
    dump(out/"historical-reproduction.json", reproduction)
    lines = ["# Deform360 empirical-null and matched-acceptance audit", "",
        f"Decision: **{result['decision']}**.",
        "92 previously evaluated real objects; 5 frozen tactile queries; exact historical means.",
        "All new covariance controls fit only source residuals. Retrospective, not fresh confirmation.", "",
        "| Arm | Brier | Fixed-cost loss | Acceptance | NLL | 90% coverage |",
        "|---|---:|---:|---:|---:|---:|"]
    for a, m in metrics.items():
        lines.append(f"| {a} | {m['brier']:.9f} | {m['decision_loss']:.9f} | {m['acceptance']:.3%} | {m['query_nll']:.6f} | {m['coverage90']:.3%} |")
    lines += ["", "## Matched 40% acceptance", "", "| Comparator | Full minus comparator event risk | 95% object-bootstrap interval |", "|---|---:|---:|---|"]
    for a, c in contrasts.items():
        v=c["matched"]["0.4"]
        lines.append(f"| {a} | {v['mean_difference']:.9f} | {v['bootstrap95']} |")
    lines += ["", f"Maximum full/empirical-query probability difference: {result['max_probability_difference']:.3g}.",
        "The full v6 per-query scale cancels its raw projected covariance: raw_variance * (source_MSE/raw_variance) = source_MSE.",
        "Therefore this study cannot establish an exclusive Bayesian advantage over a same-mean source empirical Gaussian for the registered scalar queries.",
        "This does not refute the value of joint covariance for joint events or unseen queries. Such outcomes are not tested here.", ""]
    (out/"report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
