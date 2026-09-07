#!/usr/bin/env python3
"""Nested historical-development replay of the already-defined DP pilot.

No source-test or official-evaluation recordings are in the input artifact.
Tune on 32/8 inside the original 40 fit trajectories, refit on 40, and score
all eight historical validation trajectories after saving predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(REPO / "src")]
import run_dp_residual_dlo1_pilot as m  # noqa: E402 -- source-tree bootstrap

from bayesian_phystwin_experiments.deform_dlo_local_residual import (  # noqa: E402 -- source-tree bootstrap
    build_deform_local_residual_features,
    fit_deform_local_residual,
    predict_deform_local_residual,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    preparation = json.loads(args.preparation.read_text())
    if (
        m.sha(args.bundle)
        != "70894781f89351bb25fc3b53f7b1453d9e6b36aeebc03caa7cb341ac4023ad3d"
    ):
        raise ValueError("Unexpected development bundle")
    if (
        preparation["checkpoint_sha256"]
        != "ea2b9e83d09a05bf94eae25aa1dafb449868d4a31961b06ec7b4420216d726e0"
    ):
        raise ValueError("Unexpected hybrid checkpoint")
    if any(
        preparation[k]
        for k in ("source_test_read", "official_eval_read", "other_dlos_read")
    ):
        raise ValueError("Input preparation exceeded the source-development boundary")
    archive = np.load(args.bundle, allow_pickle=False)
    names = archive["names"].tolist()
    split = archive["split"].tolist()
    if (
        len(names) != 48
        or len(set(names)) != 48
        or split != ["fit"] * 40 + ["validation"] * 8
    ):
        raise ValueError("Unexpected trajectory identities or split")
    permutation = np.random.default_rng(m.SEED).permutation(40)
    train_idx = permutation[:32]
    tune_idx = permutation[32:]
    test_idx = np.arange(40, 48)
    protocol = dict(
        contract="dp-residual-cached-nested-development-v1",
        bundle_sha256=m.sha(args.bundle),
        preparation=preparation,
        inner_fit_names=[names[i] for i in train_idx],
        inner_validation_names=[names[i] for i in tune_idx],
        refit_names=names[:40],
        outer_historical_validation_names=names[40:],
        seed=m.SEED,
        ridges=m.RIDGES,
        shrinkages=m.SHRINKAGES,
        finite_k=[2, 3, 4, 6],
        dp_cap=6,
        dp_concentrations=[0.1, 1.0, 10.0],
        context_pca=4,
        residual_pca=3,
        nonlinear_features=64,
        nonlinear_bandwidths=[1.0, 3.0],
        source_test_read=False,
        official_eval_read=False,
        other_dlos_read=False,
        prior_scope="two-stage-summary-mixture-not-joint-regression-posterior",
        fresh_confirmatory_evidence=False,
        git_sha=os.environ.get("GITHUB_SHA"),
        python=platform.python_version(),
        numpy=np.__version__,
    )
    m.write_json(out / "protocol.json", protocol)
    initial = archive["initial"]
    action = archive["action"]
    baseline = np.asarray(archive["baseline"], dtype=np.float64)
    features, frames = build_deform_local_residual_features(initial, action, baseline)
    # Only the original fit labels enter model fitting and selection.
    fit_target = np.asarray(archive["targets"][:40], dtype=np.float64)
    residual = np.einsum(
        "ntvi,nij->ntvj", (fit_target - baseline[:40])[:, :, 2:-2], frames[:40]
    )
    ft = features[train_idx]
    fr = residual[train_idx]
    fv = features[tune_idx]
    vr = frames[tune_idx]
    table = []
    selection = {}

    def consider(family, design, stats, gate, details):
        tw = np.ones((32, 1)) if gate is None else gate.train_weights
        vw = np.ones((8, 1)) if gate is None else gate.weights(fv)
        for ridge in m.RIDGES:
            fits, support = m.fit_experts(stats, tw, ridge)
            corrections, _ = m.component_predictions(design, fits, fv, vr)
            for shrink in m.SHRINKAGES:
                prediction = m.point_prediction(
                    baseline[tune_idx], corrections, vw, shrink
                )
                metric = m.point_metrics(prediction, fit_target[tune_idx])
                spec = {
                    **details,
                    "family": family,
                    "ridge": ridge,
                    "shrinkage": shrink,
                }
                table.append(dict(spec=spec, **metric))
                if (
                    family not in selection
                    or metric["mean_l1_mm"]
                    < selection[family]["validation"]["mean_l1_mm"] - 1e-12
                ):
                    selection[family] = dict(spec=spec, validation=metric)
        print(
            "INNER_VALIDATED",
            family,
            details,
            selection[family]["validation"]["mean_l1_mm"],
            flush=True,
        )

    design = m.Design.fit(ft)
    stats = m.sufficient_statistics(design, ft, fr)
    consider("single_tuned", design, stats, None, {})
    for k in (2, 3, 4, 6):
        gate = m.fit_gate(ft, fr, "finite", k, 1.0 / k)
        consider(
            "finite_tuned", design, stats, gate, dict(cap=k, concentration=1.0 / k)
        )
    for alpha in (0.1, 1.0, 10.0):
        gate = m.fit_gate(ft, fr, "dp", 6, alpha)
        consider("dp_tuned", design, stats, gate, dict(cap=6, concentration=alpha))
    del stats
    for bandwidth in (1.0, 3.0):
        design = m.Design.fit(ft, bandwidth)
        stats = m.sufficient_statistics(design, ft, fr)
        consider("nonlinear_tuned", design, stats, None, dict(bandwidth=bandwidth))
        del stats
    selection["fixed_current"] = dict(
        spec=dict(family="fixed_current", ridge=1.0, shrinkage=0.5), validation=None
    )
    m.write_json(out / "inner_candidates.json", table)
    m.write_json(out / "selection.json", selection)
    print("INNER_SELECTION_SEALED", json.dumps(selection), flush=True)
    design = m.Design.fit(features[:40])
    stats = m.sufficient_statistics(design, features[:40], residual)
    forecasts = {}
    records = {}
    saved = {}
    refit_metadata = {}
    for family, row in selection.items():
        spec = row["spec"]
        dg = design
        st = stats
        gate = None
        if family == "nonlinear_tuned":
            dg = m.Design.fit(features[:40], spec["bandwidth"])
            st = m.sufficient_statistics(dg, features[:40], residual)
        if family in ("dp_tuned", "finite_tuned"):
            gate = m.fit_gate(
                features[:40],
                residual,
                "dp" if family == "dp_tuned" else "finite",
                spec["cap"],
                spec["concentration"],
            )
        tw = np.ones((40, 1)) if gate is None else gate.train_weights
        weights = np.ones((8, 1)) if gate is None else gate.weights(features[test_idx])
        fits, support = m.fit_experts(st, tw, spec["ridge"])
        correction, variance = m.component_predictions(
            dg, fits, features[test_idx], frames[test_idx], True
        )
        predicted = m.point_prediction(
            baseline[test_idx], correction, weights, spec["shrinkage"]
        )
        if family == "fixed_current":
            original = fit_deform_local_residual(
                initial[:40],
                action[:40],
                baseline[:40],
                fit_target,
                names[:40],
                ridge=1.0,
                variance_floor_m2=m.FLOOR,
            )
            check = predict_deform_local_residual(
                original,
                initial[test_idx],
                action[test_idx],
                baseline[test_idx],
                shrinkage=0.5,
            )
            np.testing.assert_allclose(
                predicted, check["predictions"], rtol=1e-7, atol=1e-9
            )
            np.testing.assert_allclose(
                variance[0] + (0.5 * correction[0]) ** 2 + m.FLOOR,
                check["coordinate_variance_m2"][:, :, 2:-2],
                rtol=1e-6,
                atol=1e-10,
            )
            print("PRODUCTION_REFERENCE_MEAN_AND_VARIANCE_REPRODUCED", flush=True)
        assert np.array_equal(
            predicted[:, :, [0, 1, -2, -1]], baseline[test_idx][:, :, [0, 1, -2, -1]]
        )
        m.save_fitted(out / f"{family}_fitted.npz", dg, fits, gate, support)
        refit_metadata[family] = dict(
            support=support,
            gate=None if gate is None else gate.information,
            model_sha256=m.sha(out / f"{family}_fitted.npz"),
        )
        forecasts[family] = (
            predicted,
            correction,
            variance,
            weights,
            spec["shrinkage"],
        )
        saved[family + "_prediction"] = predicted
        saved[family + "_weights"] = weights
        if family == "dp_tuned":
            hard = np.eye(weights.shape[1])[np.argmax(weights, axis=1)]
            saved["dp_hard_assignment_prediction"] = m.point_prediction(
                baseline[test_idx], correction, hard, spec["shrinkage"]
            )
        print("REFIT_AND_PREDICTED", family, flush=True)
    np.savez_compressed(
        out / "frozen_predictions.npz", names=np.asarray(names[40:]), **saved
    )
    m.write_json(
        out / "prediction_barrier.json",
        dict(
            selection_sha256=m.sha(out / "selection.json"),
            predictions_sha256=m.sha(out / "frozen_predictions.npz"),
            protocol_sha256=m.sha(out / "protocol.json"),
            refit_metadata=refit_metadata,
            outer_labels_used_for_model_selection=False,
            source_test_read=False,
            official_eval_read=False,
        ),
    )
    target = np.asarray(archive["targets"][40:], dtype=np.float64)
    records["baseline"] = m.point_metrics(baseline[test_idx], target)
    if (
        abs(
            records["baseline"]["mean_l1_mm"] - preparation["validation_baseline_l1_mm"]
        )
        > 1e-4
    ):
        raise ValueError("Cached baseline metric failed reproduction")
    for family, (predicted, c, v, w, s) in forecasts.items():
        distribution, _, _ = m.distribution_metrics(
            baseline[test_idx], target, c, v, w, s
        )
        records[family] = dict(
            **m.point_metrics(predicted, target),
            **distribution,
            spec=selection[family]["spec"],
        )
    records["dp_hard_assignment"] = m.point_metrics(
        saved["dp_hard_assignment_prediction"], target
    )
    comparisons = {
        reference: m.paired_comparison(records["dp_tuned"], records[reference])
        for reference in (
            "baseline",
            "fixed_current",
            "single_tuned",
            "finite_tuned",
            "nonlinear_tuned",
        )
    }
    result = dict(
        contract="dp-residual-cached-nested-development-result-v1",
        results=records,
        dp_comparisons=comparisons,
        selection=selection,
        refit_metadata=refit_metadata,
        outer_names=names[40:],
        outer_trajectory_count=8,
        inner_fit_count=32,
        inner_validation_count=8,
        refit_count=40,
        source_test_read=False,
        official_eval_read=False,
        other_dlos_read=False,
        fresh_confirmatory_evidence=False,
        prediction_barrier_sha256=m.sha(out / "prediction_barrier.json"),
        elapsed_seconds=time.time() - started,
        limitations=[
            "Historical DLO1 validation influenced earlier checkpoint/method development",
            "Single 32/8 inner split; eight outer trajectories; exploratory only",
            "Two-stage summary DP, not full DP regression or sticky state inference",
            "Marginal coordinate uncertainty, not joint trajectory calibration",
        ],
    )
    m.write_json(out / "result.json", result)
    text = [
        "# Cached DLO1 nested-development DP residual pilot",
        "",
        "32 inner-fit / 8 inner-validation; refit on 40; score eight historical validation recordings.",
        "No source-test, official-evaluation, or other-DLO recording is in this input artifact.",
        "",
        "| Method | L1 (mm) | Late L1 (mm) | 90% coverage | Mean interval width (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, row in records.items():
        coverage = (
            f"{100 * row['coordinate_90_coverage']:.2f}%"
            if "coordinate_90_coverage" in row
            else "—"
        )
        width = (
            f"{row['mean_90_interval_width_mm']:.3f}"
            if "mean_90_interval_width_mm" in row
            else "—"
        )
        text.append(
            f"| {family} | {row['mean_l1_mm']:.6f} | {row['last_quarter_l1_mm']:.6f} | {coverage} | {width} |"
        )
    text += [
        "",
        "## DP paired comparisons",
        "",
        json.dumps(comparisons, indent=2),
        "",
        "This is retrospective exploratory evidence, not a new independent benchmark claim.",
    ]
    (out / "report.md").write_text("\n".join(text) + "\n")
    print("CACHED_PILOT_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
