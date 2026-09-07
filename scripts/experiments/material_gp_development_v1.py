#!/usr/bin/env python3
"""Development-only matched DLO2 residual comparison. No new benchmark claim."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

SEED = 20260907
SHRINKAGES = (0.0, 0.125, 0.25, 0.5, 1.0)
GP_GRID = tuple(itertools.product((0.5, 1.5, 4.0), (0.05, 0.25, 1.0)))
TIME_SAMPLES = 24
TOTAL_INDUCING = 768


def kernel(x, z, length, material_length=None):
    """PSD linear + Matern-3/2 kernel; last column is material coordinate."""
    if length <= 0 or (material_length is not None and material_length <= 0):
        raise ValueError("Kernel length scales must be positive")
    a, b = x[:, :-1], z[:, :-1]
    dim = a.shape[1]
    dot = a @ b.T / dim
    d2 = np.maximum(
        np.sum(a * a, 1)[:, None] / dim + np.sum(b * b, 1)[None, :] / dim - 2 * dot, 0
    )
    distance = np.sqrt(3 * d2) / length
    cov = 0.25 * (1 + dot) + (1 + distance) * np.exp(-distance)
    if material_length is not None:
        ds = np.sqrt(3) * np.abs(x[:, -1, None] - z[None, :, -1]) / material_length
        cov *= (1 + ds) * np.exp(-ds)
    return cov


def kernel_diag(x):
    return 1.25 + 0.25 * np.mean(x[:, :-1] ** 2, axis=1)


class SparseGP:
    """Whitened FITC posterior with fixed hyperparameters and training-only inducing points."""

    def __init__(self, length=1.5, noise=0.25, material_length=None, inducing=96):
        if noise <= 0 or inducing < 1:
            raise ValueError("Positive observation noise and inducing budget required")
        self.length, self.noise = float(length), float(noise)
        self.material_length, self.inducing = material_length, int(inducing)

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or not len(x):
            raise ValueError("Invalid GP training shapes")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite GP training arrays")
        indices = np.random.default_rng(SEED).choice(
            len(x), min(self.inducing, len(x)), replace=False
        )
        self.z = x[indices].copy()
        km = kernel(self.z, self.z, self.length, self.material_length)
        self.lm = np.linalg.cholesky(km + np.eye(len(km)) * 1e-8)
        phi = solve_triangular(
            self.lm, kernel(x, self.z, self.length, self.material_length).T, lower=True
        ).T
        remainder = np.maximum(kernel_diag(x) - np.sum(phi * phi, axis=1), 0)
        d = self.noise + remainder
        a = np.eye(len(km)) + phi.T @ (phi / d[:, None])
        self.la = np.linalg.cholesky(a)
        self.weights = cho_solve((self.la, True), phi.T @ (y / d[:, None]))
        return self

    def predict(self, x, variance=False):
        means, variances = [], []
        for offset in range(0, len(x), 1024):
            batch = x[offset : offset + 1024]
            phi = solve_triangular(
                self.lm,
                kernel(batch, self.z, self.length, self.material_length).T,
                lower=True,
            ).T
            means.append(phi @ self.weights)
            if variance:
                p = solve_triangular(self.la, phi.T, lower=True)
                latent = (
                    kernel_diag(batch)
                    - np.sum(phi * phi, axis=1)
                    + np.sum(p * p, axis=0)
                )
                if np.min(latent) < -1e-6:
                    raise FloatingPointError("Negative conditional GP variance")
                variances.append(np.maximum(latent, 0) + self.noise)
        mean = np.concatenate(means)
        return (mean, np.concatenate(variances)) if variance else mean


class ResidualGP:
    """Identical pooled input normalization and total inducing budget for both GP arms."""

    def __init__(self, shared, length, noise):
        self.shared, self.length, self.noise = shared, length, noise

    def _inputs(self, features):
        n, t, nodes, _ = features.shape
        standardized = (features - self.location) / self.scale
        material = np.broadcast_to(
            np.linspace(0, 1, nodes)[None, None, :, None], (n, t, nodes, 1)
        )
        return np.concatenate((standardized, material), axis=-1)

    def fit(self, features, residual):
        self.location = np.mean(features, axis=(0, 1, 2))
        self.scale = np.std(features, axis=(0, 1, 2))
        self.scale = np.where(self.scale > 1e-10, self.scale, 1)
        times = np.unique(
            np.linspace(0, features.shape[1] - 1, TIME_SAMPLES).round().astype(int)
        )
        x = self._inputs(features[:, times])
        y = residual[:, times]
        self.models, self.y_scales = [], []
        groups = [None] if self.shared else list(range(features.shape[2]))
        for node in groups:
            xx = (
                x.reshape(-1, x.shape[-1])
                if node is None
                else x[:, :, node].reshape(-1, x.shape[-1])
            )
            yy = y.reshape(-1, 3) if node is None else y[:, :, node].reshape(-1, 3)
            sy = np.maximum(np.sqrt(np.mean(yy * yy, axis=0)), 1e-4)
            gp = SparseGP(
                self.length,
                self.noise,
                0.4 if self.shared else None,
                TOTAL_INDUCING if self.shared else TOTAL_INDUCING // features.shape[2],
            )
            self.models.append(gp.fit(xx, yy / sy))
            self.y_scales.append(sy)
        return self

    def predict(self, features, variance=False):
        x = self._inputs(features)
        n, t, nodes, _ = x.shape
        means = np.empty((n, t, nodes, 3))
        variances = np.empty_like(means) if variance else None
        for j, (model, sy) in enumerate(zip(self.models, self.y_scales, strict=False)):
            xx = (
                x.reshape(-1, x.shape[-1])
                if self.shared
                else x[:, :, j].reshape(-1, x.shape[-1])
            )
            result = model.predict(xx, variance=variance)
            if variance:
                mu, var = result
                vv = var[:, None] * sy**2
            else:
                mu = result
            if self.shared:
                means[:] = (mu * sy).reshape(n, t, nodes, 3)
                if variance:
                    variances[:] = vv.reshape(n, t, nodes, 3)
            else:
                means[:, :, j] = (mu * sy).reshape(n, t, 3)
                if variance:
                    variances[:, :, j] = vv.reshape(n, t, 3)
        return (means, variances) if variance else means


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_key(initial, action, baseline):
    digest = hashlib.sha256()
    for value in (initial, action, baseline):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def group_rows(data, indices):
    groups = {}
    for i in indices:
        key = query_key(data["initial"][i], data["action"][i], data["baseline"][i])
        groups.setdefault(key, []).append(int(i))
    result = {
        k: np.stack([data[k][rows[0]] for rows in groups.values()])
        for k in ("initial", "action", "baseline")
    }
    result["target"] = np.stack(
        [data["target"][rows].mean(axis=0) for rows in groups.values()]
    )
    result["names"] = [
        "|".join(str(data["names"][i]) for i in rows) for rows in groups.values()
    ]
    result["keys"] = list(groups)
    result["cluster_sizes"] = [len(rows) for rows in groups.values()]
    return result


def subset(grouped, indices):
    return {
        k: value[indices]
        for k, value in grouped.items()
        if isinstance(value, np.ndarray)
    }


def rotate(mean, frames):
    return np.einsum("ntvj,nij->ntvi", mean, frames)


def candidate(baseline, correction, shrinkage):
    result = baseline.copy()
    result[:, :, 2:-2] += shrinkage * correction
    return result


def gp_output(model, x, frames, baseline, shrinkage):
    mean, var = model.predict(x, variance=True)
    prediction = candidate(baseline, rotate(mean, frames), shrinkage)
    marginal = shrinkage**2 * np.einsum("ntvj,nij->ntvi", var, frames**2) + 1e-6
    return prediction, marginal


def metrics(prediction, target, baseline, names, variance=None):
    error = prediction - target
    case = np.mean(np.abs(error), axis=(1, 2, 3)) * 1000
    base = np.mean(np.abs(baseline - target), axis=(1, 2, 3)) * 1000
    result = dict(
        l1_mm=float(case.mean()),
        rmse_mm=float(np.sqrt(np.mean(error**2)) * 1000),
        wins_vs_hybrid=int(np.sum(case < base - 1e-9)),
        worst_ratio_vs_hybrid=float(np.max(case / base)),
        per_trajectory_l1_mm=dict(zip(names, case.tolist(), strict=False)),
    )
    cuts = np.array_split(np.arange(error.shape[1]), 3)
    result["early_middle_late_l1_mm"] = [
        float(np.mean(np.abs(error[:, cut])) * 1000) for cut in cuts
    ]
    if variance is not None:
        e = error[:, :, 2:-2]
        v = np.maximum(variance, 1e-12)
        result.update(
            coordinate_coverage90=float(
                np.mean(np.abs(e) <= 1.6448536269514722 * np.sqrt(v))
            ),
            normalized_coordinate_nees=float(np.mean(e * e / v)),
            mean_full_width90_mm=float(
                2 * 1.6448536269514722 * np.mean(np.sqrt(v)) * 1000
            ),
            coordinate_gaussian_nll=float(
                np.mean(0.5 * (np.log(2 * np.pi * v) + e * e / v))
            ),
        )
    return result


def paired_difference(left, right):
    delta = np.asarray(left) - np.asarray(right)
    rng = np.random.default_rng(SEED)
    boots = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return dict(
        mean_delta_mm=float(delta.mean()),
        bootstrap95_mm=np.quantile(boots, [0.025, 0.975]).tolist(),
        wins=int(np.sum(delta < -1e-9)),
        ties=int(np.sum(np.abs(delta) <= 1e-9)),
        n=len(delta),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features,
        fit_deform_local_residual,
        predict_deform_local_residual,
    )

    manifest_path = args.data.with_name("export.json")
    export = json.loads(manifest_path.read_text())
    if (
        export["npz_sha256"] != checksum(args.data)
        or export["official_eval_read"]
        or export["source_test_read"]
    ):
        raise ValueError("Development export identity/boundary failed")
    with np.load(args.data, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    fit_count = int(data["fit_count"][0])
    if fit_count != 40 or len(data["names"]) != 48:
        raise ValueError("Expected original 40/8 development split")
    train = group_rows(data, range(fit_count))
    validation = group_rows(data, range(fit_count, len(data["names"])))
    if set(train["keys"]) & set(validation["keys"]):
        raise ValueError(
            "Exact causal-query overlap between fit and development-validation"
        )
    fit_x, fit_frames = build_deform_local_residual_features(
        train["initial"], train["action"], train["baseline"]
    )
    val_x, val_frames = build_deform_local_residual_features(
        validation["initial"], validation["action"], validation["baseline"]
    )
    fit_y = np.einsum(
        "ntvi,nij->ntvj", (train["target"] - train["baseline"])[:, :, 2:-2], fit_frames
    )
    order = np.random.default_rng(SEED).permutation(len(fit_x))
    inner_count = max(4, len(order) // 4)
    tune, learn = order[:inner_count], order[inner_count:]
    learn_data, tune_data = subset(train, learn), subset(train, tune)
    tune_frames = fit_frames[tune]
    selection = dict(
        scope="development-only; reused historical validation, not new confirmation",
        seed=SEED,
        original_fit_count=40,
        original_validation_count=8,
        fit_clusters=len(fit_x),
        validation_clusters=len(val_x),
        fit_cluster_sizes=train["cluster_sizes"],
        validation_cluster_sizes=validation["cluster_sizes"],
        inner_train_names=[train["names"][i] for i in learn],
        inner_tune_names=[train["names"][i] for i in tune],
        time_samples=TIME_SAMPLES,
        total_inducing_budget=TOTAL_INDUCING,
        gp_grid=GP_GRID,
        shrinkages=SHRINKAGES,
        all_candidates=[],
        selected={},
    )

    def ridge_fit(grouped, names, ridge):
        return fit_deform_local_residual(
            grouped["initial"],
            grouped["action"],
            grouped["baseline"],
            grouped["target"],
            names,
            ridge=ridge,
            variance_floor_m2=1e-6,
        )

    def ridge_output(model, grouped, shrinkage):
        p = predict_deform_local_residual(
            model,
            grouped["initial"],
            grouped["action"],
            grouped["baseline"],
            shrinkage=shrinkage,
        )
        return p["predictions"], p["coordinate_variance_m2"][:, :, 2:-2]

    names_learn = [train["names"][i] for i in learn]
    ridge_candidates = []
    for ridge in (0.1, 1.0, 10.0):
        model = ridge_fit(learn_data, names_learn, ridge)
        for shrinkage in SHRINKAGES:
            pred, _ = ridge_output(model, tune_data, shrinkage)
            loss = float(np.mean(np.abs(pred - tune_data["target"])) * 1000)
            ridge_candidates.append(
                dict(family="ridge", ridge=ridge, shrinkage=shrinkage, inner_l1_mm=loss)
            )
    selection["all_candidates"].extend(ridge_candidates)
    selection["selected"]["ridge_tuned"] = min(
        ridge_candidates, key=lambda r: r["inner_l1_mm"]
    )
    selection["selected"]["ridge_existing"] = dict(
        family="ridge", ridge=1.0, shrinkage=0.25
    )
    for shared in (False, True):
        family = "gp_material" if shared else "gp_independent"
        bank = []
        for length, noise in GP_GRID:
            model = ResidualGP(shared, length, noise).fit(fit_x[learn], fit_y[learn])
            correction = rotate(model.predict(fit_x[tune]), tune_frames)
            for shrinkage in SHRINKAGES:
                pred = candidate(tune_data["baseline"], correction, shrinkage)
                loss = float(np.mean(np.abs(pred - tune_data["target"])) * 1000)
                bank.append(
                    dict(
                        family=family,
                        length=length,
                        noise=noise,
                        shrinkage=shrinkage,
                        inner_l1_mm=loss,
                    )
                )
            print(
                "TUNING",
                family,
                length,
                noise,
                "best-so-far",
                min(b["inner_l1_mm"] for b in bank),
                flush=True,
            )
        selection["all_candidates"].extend(bank)
        selection["selected"][family] = min(bank, key=lambda r: r["inner_l1_mm"])
    (out / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    print(
        "FROZEN_SELECTION",
        json.dumps(selection["selected"], sort_keys=True),
        flush=True,
    )
    predictions = {"hybrid": validation["baseline"].copy()}
    variances, calibrations = {}, {}
    for name, config in selection["selected"].items():
        shrinkage = config["shrinkage"]
        if config["family"] == "ridge":
            inner_model = ridge_fit(learn_data, names_learn, config["ridge"])
            inner_pred, inner_var = ridge_output(inner_model, tune_data, shrinkage)
            full_model = ridge_fit(train, train["names"], config["ridge"])
            pred, var = ridge_output(full_model, validation, shrinkage)
        else:
            shared = name == "gp_material"
            inner_model = ResidualGP(shared, config["length"], config["noise"]).fit(
                fit_x[learn], fit_y[learn]
            )
            inner_pred, inner_var = gp_output(
                inner_model, fit_x[tune], tune_frames, tune_data["baseline"], shrinkage
            )
            full_model = ResidualGP(shared, config["length"], config["noise"]).fit(
                fit_x, fit_y
            )
            pred, var = gp_output(
                full_model, val_x, val_frames, validation["baseline"], shrinkage
            )
        if not np.array_equal(
            pred[:, :, [0, 1, -2, -1]], validation["baseline"][:, :, [0, 1, -2, -1]]
        ):
            raise AssertionError("Clamped boundary was changed")
        scale = float(
            np.mean(
                (inner_pred - tune_data["target"])[:, :, 2:-2] ** 2
                / np.maximum(inner_var, 1e-12)
            )
        )
        calibrations[name] = max(scale, 1e-6)
        predictions[name], variances[name] = pred, var
        print(
            "FITTED",
            name,
            "inner-only covariance scale",
            calibrations[name],
            flush=True,
        )
    np.savez_compressed(
        out / "predictions.npz",
        **predictions,
        **{k + "_variance": v for k, v in variances.items()},
    )
    seal = dict(
        selection_sha256=checksum(out / "selection.json"),
        predictions_sha256=checksum(out / "predictions.npz"),
        input_sha256=checksum(args.data),
        calibration_scales=calibrations,
        validation_targets_used_for_residual_selection_or_fit=False,
        baseline_previously_selected_on_validation=True,
    )
    (out / "prediction_seal.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n"
    )
    records = {}
    for name, pred in predictions.items():
        var = variances.get(name)
        records[name] = metrics(
            pred, validation["target"], validation["baseline"], validation["names"], var
        )
        if var is not None:
            records[name]["inner_scaled_uncertainty"] = metrics(
                pred,
                validation["target"],
                validation["baseline"],
                validation["names"],
                var * calibrations[name],
            )
    pairs = {}
    for gp in ("gp_independent", "gp_material"):
        for base in ("hybrid", "ridge_existing", "ridge_tuned"):
            pairs[gp + "-minus-" + base] = paired_difference(
                list(records[gp]["per_trajectory_l1_mm"].values()),
                list(records[base]["per_trajectory_l1_mm"].values()),
            )
    result = dict(
        schema="material-gp-development-result-v1",
        scope=selection["scope"],
        implementation_sha=os.environ.get("GITHUB_SHA", "local"),
        script_sha256=checksum(Path(__file__)),
        run_id=os.environ.get("GITHUB_RUN_ID"),
        export=export,
        selected=selection["selected"],
        fit_clusters=len(fit_x),
        validation_clusters=len(val_x),
        results=records,
        paired=pairs,
        calibration_scales=calibrations,
        official_eval_read=False,
        source_test_read=False,
        elapsed_seconds=time.perf_counter() - started,
        limitations=[
            "historical validation already used for baseline selection",
            "GP likelihood treats thinned within-trajectory errors as independent; uncertainty is diagnostic",
            "shrinkage and scalar scaling are operational adjustments, not exact calibrated physical posteriors",
            "FITC approximation; three output axes conditionally independent in the canonical frame",
            "single DLO development panel, not a benchmark or cross-object result",
        ],
    )
    (out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
