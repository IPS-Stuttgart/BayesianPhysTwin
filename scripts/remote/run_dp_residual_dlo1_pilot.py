#!/usr/bin/env python3
"""Retrospective DLO1-only DP residual pilot; never an official evaluation.

Two-stage variational DP on joint trajectory summaries, followed by soft,
trajectory-weighted ridge experts. Prediction gates marginalize out the
residual summary. No target-conditioned test assignment, sticky dynamics,
full state inference, or external observation-covariance claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, ndtr
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

SEED = 20260907
RIDGES = (0.01, 1.0, 100.0)
SHRINKAGES = (0.0, 0.125, 0.25, 0.5, 1.0)
FLOOR = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


@dataclass
class Projection:
    location: np.ndarray
    scale: np.ndarray
    pca: PCA

    @classmethod
    def fit(cls, x, dimensions, standardize=True):
        location = x.mean(0)
        scale = x.std(0) if standardize else np.full(x.shape[1], x.std())
        scale = np.where(scale > 1e-8, scale, 1.0)
        pca = PCA(n_components=dimensions, whiten=True, svd_solver="full")
        pca.fit((x - location) / scale)
        if np.min(pca.explained_variance_) < 1e-12:
            raise ValueError("Degenerate trajectory-summary projection")
        return cls(location, scale, pca)

    def transform(self, x):
        return self.pca.transform((x - self.location) / self.scale)


def context_summary(features):
    # All entries are functions of the permitted predictor inputs.
    n, t, _, _ = features.shape
    indices = np.linspace(0, t - 1, 8, dtype=int)
    return np.concatenate(
        (
            features[:, indices].reshape(n, -1),
            features.mean(1).reshape(n, -1),
            features.std(1).reshape(n, -1),
        ),
        axis=1,
    )


def residual_summary(residual):
    n, t, _, _ = residual.shape
    return residual[:, np.linspace(0, t - 1, 8, dtype=int)].reshape(n, -1)


@dataclass
class Gate:
    context: Projection
    residual: Projection
    mixture: BayesianGaussianMixture
    train_weights: np.ndarray
    information: dict

    def weights(self, features):
        x = self.context.transform(context_summary(features))
        d = x.shape[1]
        logweights = []
        for weight, mean, covariance in zip(
            self.mixture.weights_,
            self.mixture.means_,
            self.mixture.covariances_,
            strict=False,
        ):
            # Marginal p(context | component): no query residual is supplied.
            logweights.append(
                np.log(max(weight, 1e-300))
                + multivariate_normal.logpdf(x, mean=mean[:d], cov=covariance[:d, :d])
            )
        logweights = np.stack(logweights, axis=1)
        return np.exp(logweights - logsumexp(logweights, axis=1, keepdims=True))


def fit_gate(features, residual, family, k, concentration):
    cx = Projection.fit(context_summary(features), 4)
    ry = Projection.fit(residual_summary(residual), 3, standardize=False)
    joint = np.concatenate(
        (
            cx.transform(context_summary(features)),
            ry.transform(residual_summary(residual)),
        ),
        axis=1,
    )
    prior = "dirichlet_process" if family == "dp" else "dirichlet_distribution"
    model = BayesianGaussianMixture(
        n_components=k,
        covariance_type="full",
        weight_concentration_prior_type=prior,
        weight_concentration_prior=concentration,
        mean_precision_prior=1.0,
        covariance_prior=np.eye(joint.shape[1]),
        reg_covar=1e-5,
        n_init=5,
        max_iter=1000,
        tol=1e-5,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(joint)
    info = dict(
        family=family,
        cap=k,
        concentration=concentration,
        converged=bool(model.converged_),
        iterations=int(model.n_iter_),
        component_weights=model.weights_.tolist(),
        active_components_above_0p05=int(np.sum(model.weights_ > 0.05)),
        warnings=[str(w.message) for w in caught],
    )
    if not model.converged_:
        raise RuntimeError(f"Gate did not converge: {info}")
    return Gate(cx, ry, model, model.predict_proba(joint), info)


@dataclass
class Design:
    location: np.ndarray
    scale: np.ndarray
    omega: np.ndarray | None = None
    phase: np.ndarray | None = None

    @classmethod
    def fit(cls, features, bandwidth=None):
        location = features.mean(axis=(0, 1))
        scale = features.std(axis=(0, 1))
        scale = np.where(scale > 1e-10, scale, 1.0)
        if bandwidth is None:
            return cls(location, scale)
        rng = np.random.default_rng(SEED)
        omega = rng.normal(size=(features.shape[-1], 64))
        omega /= np.sqrt(features.shape[-1]) * bandwidth
        return cls(location, scale, omega, rng.uniform(0, 2 * np.pi, 64))

    def node(self, features, node):
        x = (features[:, :, node] - self.location[node]) / self.scale[node]
        columns = [np.ones((*x.shape[:2], 1)), x]
        if self.omega is not None:
            columns.append(np.sqrt(2 / 64) * np.cos(x @ self.omega + self.phase))
        return np.concatenate(columns, axis=-1)


def sufficient_statistics(design, features, residual):
    out = []
    for node in range(features.shape[2]):
        x = design.node(features, node)
        y = residual[:, :, node]
        out.append((x.swapaxes(1, 2) @ x, x.swapaxes(1, 2) @ y, y, x))
    return out


def fit_experts(stats, weights, ridge):
    n, k = weights.shape
    fits, support = [], []
    for component in range(k):
        w = weights[:, component].copy()
        mass = float(w.sum())
        ess = float(mass**2 / max(np.sum(w * w), 1e-300))
        fallback = mass < 2.0 or ess < 2.0
        if fallback:
            w = np.ones(n)
        support.append(
            dict(
                expected_trajectories=mass,
                effective_trajectories=ess,
                global_expert_fallback=fallback,
            )
        )
        coefficients, covariances, variances = [], [], []
        effective = w.sum() ** 2 / (w @ w)
        for gram, cross, y, x in stats:
            penalty = np.eye(gram.shape[-1]) * ridge
            penalty[0, 0] = 0
            normal = np.einsum("n,nde->de", w, gram) + penalty
            factor = cho_factor(normal, lower=True, check_finite=True)
            beta = cho_solve(factor, np.einsum("n,ndc->dc", w, cross))
            bread = cho_solve(factor, np.eye(normal.shape[0]))
            errors = y - x @ beta
            scores = (cross - gram @ beta) * w[:, None, None]
            cov = []
            for c in range(3):
                transformed = scores[:, :, c] @ bread
                cov.append(transformed.T @ transformed * effective / (effective - 1))
            coefficients.append(beta)
            covariances.append(np.stack(cov))
            variances.append(
                np.einsum("n,ntc->c", w, errors**2) / (w.sum() * y.shape[1])
            )
        fits.append(
            (np.stack(coefficients), np.stack(covariances), np.stack(variances))
        )
    return fits, support


def component_predictions(design, fits, features, frames, uncertainty=False):
    means, variances = [], []
    for beta, covariance, noise in fits:
        mu, va = [], []
        for node in range(features.shape[2]):
            x = design.node(features, node)
            mu.append(x @ beta[node])
            if uncertainty:
                epistemic = np.stack(
                    [np.sum((x @ covariance[node, c]) * x, axis=-1) for c in range(3)],
                    axis=-1,
                )
                va.append(np.maximum(epistemic, 0) + noise[node])
        canonical = np.stack(mu, axis=2)
        means.append(np.einsum("ntvj,nij->ntvi", canonical, frames))
        if uncertainty:
            variances.append(
                np.einsum("ntvj,nij->ntvi", np.stack(va, axis=2), frames**2)
            )
    return np.stack(means), np.stack(variances) if uncertainty else None


def point_prediction(baseline, corrections, weights, shrinkage):
    out = np.asarray(baseline, dtype=np.float64).copy()
    out[:, :, 2:-2] += shrinkage * np.einsum("nk,kntvc->ntvc", weights, corrections)
    return out


def point_metrics(prediction, target):
    absolute = np.abs(prediction - target)
    return dict(
        mean_l1_mm=float(absolute.mean() * 1000),
        trajectory_l1_mm=(absolute.mean((1, 2, 3)) * 1000).tolist(),
        last_quarter_l1_mm=float(
            absolute[:, -max(1, absolute.shape[1] // 4) :].mean() * 1000
        ),
        rmse_mm=float(np.sqrt(np.mean((prediction - target) ** 2)) * 1000),
    )


def distribution_metrics(baseline, target, corrections, variances, weights, shrinkage):
    means = baseline[None, :, :, 2:-2] + shrinkage * corrections
    variance = variances + ((1 - shrinkage) * corrections) ** 2 + FLOOR
    y = target[None, :, :, 2:-2]
    logw = np.log(np.maximum(weights.T, 1e-300))[:, :, None, None, None]
    logpdf = -0.5 * (np.log(2 * np.pi * variance) + (y - means) ** 2 / variance)
    nll = -logsumexp(logw + logpdf, axis=0)
    w = weights.T[:, :, None, None, None]
    mean = np.sum(w * means, axis=0)
    within = np.sum(w * variance, axis=0)
    between = np.sum(w * (means - mean[None]) ** 2, axis=0)
    lo0 = np.min(means - 12 * np.sqrt(variance), axis=0)
    hi0 = np.max(means + 12 * np.sqrt(variance), axis=0)
    quantiles = []
    for p in (0.05, 0.95):
        lo, hi = lo0.copy(), hi0.copy()
        for _ in range(35):
            mid = (lo + hi) / 2
            cdf = np.sum(w * ndtr((mid[None] - means) / np.sqrt(variance)), axis=0)
            lo = np.where(cdf < p, mid, lo)
            hi = np.where(cdf >= p, mid, hi)
        quantiles.append((lo + hi) / 2)
    lower, upper = quantiles
    covered = (target[:, :, 2:-2] >= lower) & (target[:, :, 2:-2] <= upper)
    return (
        dict(
            marginal_gaussian_mixture_nll_per_coordinate_m=float(nll.mean()),
            coordinate_90_coverage=float(covered.mean()),
            mean_90_interval_width_mm=float((upper - lower).mean() * 1000),
            coordinate_nees=float(np.mean((y[0] - mean) ** 2 / (within + between))),
            mean_between_component_variance_mm2=float(between.mean() * 1e6),
        ),
        mean,
        within + between,
    )


def paired_comparison(candidate, reference):
    c = np.asarray(candidate["trajectory_l1_mm"])
    r = np.asarray(reference["trajectory_l1_mm"])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(c), size=(10000, len(c)))
    improvement = 100 * (1 - c[idx].mean(1) / r[idx].mean(1))
    return dict(
        relative_improvement_percent=float(100 * (1 - c.mean() / r.mean())),
        wins=int(np.sum(c < r - 1e-9)),
        ties=int(np.sum(np.abs(c - r) <= 1e-9)),
        n_trajectories=len(c),
        paired_trajectory_bootstrap_95_percent=np.quantile(
            improvement, [0.025, 0.975]
        ).tolist(),
    )


def save_fitted(path, design, fits, gate, support):
    arrays = dict(location=design.location, scale=design.scale)
    if design.omega is not None:
        arrays.update(omega=design.omega, phase=design.phase)
    for k, (beta, covariance, variance) in enumerate(fits):
        arrays.update(
            {
                f"beta_{k}": beta,
                f"covariance_{k}": covariance,
                f"variance_{k}": variance,
            }
        )
    if gate is not None:
        for name, projection in [
            ("context", gate.context),
            ("residual", gate.residual),
        ]:
            arrays.update(
                {
                    f"{name}_location": projection.location,
                    f"{name}_scale": projection.scale,
                    f"{name}_pca_mean": projection.pca.mean_,
                    f"{name}_pca_components": projection.pca.components_,
                    f"{name}_pca_variance": projection.pca.explained_variance_,
                }
            )
        arrays.update(
            gate_weights=gate.mixture.weights_,
            gate_means=gate.mixture.means_,
            gate_covariances=gate.mixture.covariances_,
            train_responsibilities=gate.train_weights,
        )
    arrays["support_json"] = np.asarray(json.dumps(support))
    np.savez_compressed(path, **arrays)


def self_test():
    rng = np.random.default_rng(71)
    f = rng.normal(size=(12, 10, 2, 5))
    r = rng.normal(size=(12, 10, 2, 3)) * 0.001
    design = Design.fit(f)
    stats = sufficient_statistics(design, f, r)
    fits, _ = fit_experts(stats, np.ones((12, 1)), 1.0)
    frames = np.tile(np.eye(3), (12, 1, 1))
    correction, variance = component_predictions(design, fits, f, frames, True)
    x = design.node(f, 0).reshape(-1, 6)
    penalty = np.eye(6)
    penalty[0, 0] = 0
    expected = np.linalg.solve(x.T @ x + penalty, x.T @ r[:, :, 0].reshape(-1, 3))
    np.testing.assert_allclose(fits[0][0][0], expected, atol=1e-11)
    baseline = np.zeros((12, 10, 6, 3))
    pred = point_prediction(baseline, correction, np.ones((12, 1)), 0.5)
    assert np.array_equal(pred[:, :, [0, 1, -2, -1]], baseline[:, :, [0, 1, -2, -1]])
    assert np.all(variance >= 0)
    for family in ("dp", "finite"):
        gate = fit_gate(f, r, family, 3, 1.0)
        weight = gate.weights(f)
        np.testing.assert_allclose(weight.sum(1), 1.0, atol=1e-12)
        assert weight.shape == (12, 3)
        # There is intentionally no residual/target argument to query weights.
        np.testing.assert_array_equal(weight, gate.weights(f))
    print(
        "SELF_TEST_PASS: weighted ridge equivalence, finite covariance, "
        "exact clamps, normalized DP/finite marginal gates",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_root is None:
        parser.error("--output-root is required")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts/remote"))
    sys.path.insert(0, str(repo / "src"))
    import run_deform_dlo_action_residual as common
    import run_deform_dlo_longrun_posterior as posterior
    import run_deform_dlo_source as source
    import scipy
    import sklearn
    import torch

    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features,
        deform_causal_inputs,
        fit_deform_local_residual,
        predict_deform_local_residual,
    )

    protocol_path = repo / "configs/sota/deform_dlo_local_residual_v4.json"
    protocol = json.loads(protocol_path.read_text())
    manifest_path = repo / protocol["source_manifest"]["repository_path"]
    longrun_path = Path(protocol["longrun_result"]["path"])
    common._verify_identity(
        manifest_path, protocol["source_manifest"], label="source manifest"
    )
    common._verify_identity(
        longrun_path, protocol["longrun_result"], label="frozen baseline result"
    )
    manifest = json.loads(manifest_path.read_text())
    longrun = json.loads(longrun_path.read_text())
    assert manifest["dlo_type"] == "DLO1" and manifest["partition"] == "train"
    all_names = sum(
        (manifest["split"][s] for s in ("fit", "validation", "source_test")), []
    )
    if len(set(all_names)) != len(all_names):
        raise ValueError("Trajectory split overlap")
    for name in all_names:
        p = Path(manifest["trajectories"][name]["path"]).resolve()
        if p.parent.name != "train" or p.parent.parent.name != "DLO1":
            raise ValueError("Only DLO1/train is permitted")
    upstream = Path(manifest["trajectories"][all_names[0]]["path"]).parents[3]
    source._assert_upstream(upstream, longrun["upstream"]["commit"])
    for forbidden in (
        upstream / "data_set/DLO1/eval",
        *(upstream / f"data_set/DLO{i}" for i in range(2, 6)),
    ):
        source._install_eval_read_guard(forbidden)
    longrun_protocol = json.loads(
        (repo / protocol["longrun_protocol"]["repository_path"]).read_text()
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = longrun_protocol["training"][
        "cublas_workspace_config"
    ]
    source._seed_everything(torch, 42)
    torch.set_num_threads(2)
    modules = source._load_upstream(upstream)
    state = posterior._checkpoint_states(longrun, {6400}, torch=torch)[6400]
    if (
        longrun["selected_checkpoint"]["checkpoint"]["sha256"]
        != protocol["baseline"]["checkpoint_sha256"]
    ):
        raise ValueError("Unexpected baseline checkpoint")
    design_spec = dict(
        contract="dp-residual-dlo1-retrospective-pilot-v1",
        observation_contract="two-states-known-clamped-action-baseline-only",
        protocol_sha256=sha(protocol_path),
        source_manifest_sha256=sha(manifest_path),
        baseline_checkpoint_sha256=protocol["baseline"]["checkpoint_sha256"],
        splits=manifest["split"],
        ridges=RIDGES,
        shrinkages=SHRINKAGES,
        finite_k=[2, 3, 4, 6],
        dp_cap=6,
        dp_concentrations=[0.1, 1.0, 10.0],
        context_pca=4,
        residual_pca=3,
        seed=SEED,
        nonlinear_rff_features=64,
        nonlinear_bandwidths=[1.0, 3.0],
        prior_scope="two-stage-summary-mixture-not-joint-regression-posterior",
        fresh_confirmatory_evidence=False,
        official_eval_read=False,
        other_dlo_read=False,
        source_test_read=False,
        git_sha=os.environ.get("GITHUB_SHA"),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        sklearn=sklearn.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    write_json(output / "protocol.json", design_spec)

    def load_split(names):
        trajectories = source._load_named_trajectories(
            manifest, names, frame_count=500, node_count=13
        )
        rollout = common._rollout(
            state, trajectories, modules=modules, torch=torch, device="cuda:0"
        )
        if list(rollout["names"]) != list(names):
            raise ValueError("Trajectory ordering changed")
        initial, action = deform_causal_inputs(
            np.stack([trajectories[name] for name in names])
        )
        baseline = np.asarray(rollout["predictions"], dtype=np.float64)
        target = np.asarray(rollout["targets"], dtype=np.float64)
        features, frames = build_deform_local_residual_features(
            initial, action, baseline
        )
        residual = np.einsum("ntvi,nij->ntvj", (target - baseline)[:, :, 2:-2], frames)
        return dict(
            names=names,
            initial=initial,
            action=action,
            baseline=baseline,
            target=target,
            features=features,
            frames=frames,
            residual=residual,
        )

    fit = load_split(manifest["split"]["fit"])
    print("FIT_ROLLOUT_READY", fit["baseline"].shape, flush=True)
    val = load_split(manifest["split"]["validation"])
    common._require_baseline_reproduction(
        float(np.abs(val["baseline"] - val["target"]).mean()),
        expected=protocol["baseline"]["validation_l1_m"],
        tolerance=1e-7,
        stage="validation",
    )
    print("VALIDATION_BASELINE_REPRODUCED", flush=True)
    original = fit_deform_local_residual(
        fit["initial"],
        fit["action"],
        fit["baseline"],
        fit["target"],
        fit["names"],
        ridge=1.0,
        variance_floor_m2=FLOOR,
    )
    reference = predict_deform_local_residual(
        original, val["initial"], val["action"], val["baseline"], shrinkage=0.5
    )
    design = Design.fit(fit["features"])
    stats = sufficient_statistics(design, fit["features"], fit["residual"])
    ref_fits, ref_support = fit_experts(stats, np.ones((len(fit["names"]), 1)), 1.0)
    ref_c, ref_v = component_predictions(
        design, ref_fits, val["features"], val["frames"], True
    )
    replicated = point_prediction(
        val["baseline"], ref_c, np.ones((len(val["names"]), 1)), 0.5
    )
    np.testing.assert_allclose(
        replicated, reference["predictions"], rtol=1e-7, atol=1e-9
    )
    np.testing.assert_allclose(
        ref_v[0] + (0.5 * ref_c[0]) ** 2 + FLOOR,
        reference["coordinate_variance_m2"][:, :, 2:-2],
        rtol=1e-6,
        atol=1e-10,
    )
    print("PRODUCTION_RESIDUAL_MEAN_AND_VARIANCE_REPRODUCED", flush=True)

    finalists = {
        "fixed_current": dict(
            design=design,
            fits=ref_fits,
            gate=None,
            support=ref_support,
            spec=dict(family="fixed_current", ridge=1.0, shrinkage=0.5),
            validation=point_metrics(replicated, val["target"]),
        )
    }
    table = []

    def consider(family, dg, statistics, gate, details):
        train_w = (
            np.ones((len(fit["names"]), 1)) if gate is None else gate.train_weights
        )
        val_w = (
            np.ones((len(val["names"]), 1))
            if gate is None
            else gate.weights(val["features"])
        )
        for ridge in RIDGES:
            models, support = fit_experts(statistics, train_w, ridge)
            corrections, _ = component_predictions(
                dg, models, val["features"], val["frames"]
            )
            for shrink in SHRINKAGES:
                predicted = point_prediction(
                    val["baseline"], corrections, val_w, shrink
                )
                metric = point_metrics(predicted, val["target"])
                spec = {
                    **details,
                    "family": family,
                    "ridge": ridge,
                    "shrinkage": shrink,
                }
                table.append(dict(spec=spec, **metric))
                if (
                    family not in finalists
                    or metric["mean_l1_mm"]
                    < finalists[family]["validation"]["mean_l1_mm"] - 1e-12
                ):
                    finalists[family] = dict(
                        design=dg,
                        fits=models,
                        gate=gate,
                        support=support,
                        spec=spec,
                        validation=metric,
                    )
        print(
            "VALIDATED",
            family,
            details,
            finalists[family]["validation"]["mean_l1_mm"],
            flush=True,
        )

    consider("single_tuned", design, stats, None, {})
    for k in (2, 3, 4, 6):
        gate = fit_gate(fit["features"], fit["residual"], "finite", k, 1.0 / k)
        consider("finite_tuned", design, stats, gate, gate.information)
    for alpha in (0.1, 1.0, 10.0):
        gate = fit_gate(fit["features"], fit["residual"], "dp", 6, alpha)
        consider("dp_tuned", design, stats, gate, gate.information)
    del stats
    for bandwidth in (1.0, 3.0):
        nonlinear = Design.fit(fit["features"], bandwidth)
        nonlinear_stats = sufficient_statistics(
            nonlinear, fit["features"], fit["residual"]
        )
        consider(
            "nonlinear_tuned",
            nonlinear,
            nonlinear_stats,
            None,
            dict(bandwidth=bandwidth),
        )
        del nonlinear_stats

    selection = {}
    for family, record in finalists.items():
        path = output / f"{family}_fitted.npz"
        save_fitted(
            path, record["design"], record["fits"], record["gate"], record["support"]
        )
        selection[family] = dict(
            spec=record["spec"],
            validation=record["validation"],
            model_sha256=sha(path),
            support=record["support"],
        )
    write_json(output / "validation_candidates.json", table)
    write_json(output / "selection.json", selection)
    write_json(
        output / "source_opening.json",
        dict(
            selection_sha256=sha(output / "selection.json"),
            validation_candidates_sha256=sha(output / "validation_candidates.json"),
            protocol_sha256=sha(output / "protocol.json"),
            source_test_read=False,
            official_eval_read=False,
            other_dlo_read=False,
        ),
    )
    print("SOURCE_SELECTION_SEALED", json.dumps(selection), flush=True)
    test = load_split(manifest["split"]["source_test"])
    common._require_baseline_reproduction(
        float(np.abs(test["baseline"] - test["target"]).mean()),
        expected=protocol["baseline"]["source_test_l1_m"],
        tolerance=1e-7,
        stage="source-test",
    )
    results = {"baseline": point_metrics(test["baseline"], test["target"])}
    arrays = dict(
        names=np.asarray(test["names"]),
        baseline=test["baseline"],
        target=test["target"],
    )
    for family, record in finalists.items():
        gate = record["gate"]
        weights = (
            np.ones((len(test["names"]), 1))
            if gate is None
            else gate.weights(test["features"])
        )
        corrections, variances = component_predictions(
            record["design"], record["fits"], test["features"], test["frames"], True
        )
        shrink = record["spec"]["shrinkage"]
        predicted = point_prediction(test["baseline"], corrections, weights, shrink)
        distribution, _, variance = distribution_metrics(
            test["baseline"], test["target"], corrections, variances, weights, shrink
        )
        results[family] = dict(
            **point_metrics(predicted, test["target"]),
            **distribution,
            spec=record["spec"],
            test_weights=weights.tolist(),
        )
        arrays[family + "_prediction"] = predicted
        arrays[family + "_coordinate_variance"] = variance
        assert np.array_equal(
            predicted[:, :, [0, 1, -2, -1]], test["baseline"][:, :, [0, 1, -2, -1]]
        )
        print("TEST_RESULT", family, json.dumps(results[family]), flush=True)
        if family == "dp_tuned":
            hard = np.eye(weights.shape[1])[np.argmax(weights, axis=1)]
            hp = point_prediction(test["baseline"], corrections, hard, shrink)
            results["dp_hard_assignment"] = point_metrics(hp, test["target"])
    comparisons = {
        reference: paired_comparison(results["dp_tuned"], results[reference])
        for reference in (
            "baseline",
            "fixed_current",
            "single_tuned",
            "finite_tuned",
            "nonlinear_tuned",
        )
    }
    np.savez_compressed(output / "source_predictions.npz", **arrays)
    result = dict(
        contract="dp-residual-dlo1-retrospective-pilot-result-v1",
        results=results,
        dp_comparisons=comparisons,
        selection=selection,
        source_names=test["names"],
        source_test_read=True,
        official_eval_read=False,
        other_dlo_read=False,
        source_selection_sha256=sha(output / "selection.json"),
        prediction_sha256=sha(output / "source_predictions.npz"),
        independent_trajectory_units=len(test["names"]),
        fresh_confirmatory_evidence=False,
        elapsed_seconds=time.time() - start,
        limitations=[
            "DLO1 source split has been used in earlier method development",
            "Two-stage summary DP with plug-in marginal gating; not full joint regression inference",
            "No external observation-covariance model or sticky temporal transitions",
            "Marginal coordinate scores do not establish joint trajectory calibration",
        ],
    )
    write_json(output / "result.json", result)
    print("FINAL_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
