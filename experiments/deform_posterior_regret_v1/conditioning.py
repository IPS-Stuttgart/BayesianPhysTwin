"""Full temporal Schur complement of the frozen finite-rank GP."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .gp import GPConfig, fit_gp
from .risk import TemporalBlocks

SHRINKAGE = 0.25
GRID = np.linspace(0, 497, 20, dtype=int)
Q_FLOOR = 1e-10  # Frozen follow-up's (0.01 mm)^2 floor.


def prefix_indices(endpoint):
    if endpoint not in (50, 150, 250):
        raise ValueError("unregistered prefix endpoint")
    return np.arange(endpoint - 46, endpoint, 5)


def fit_models(features, targets, names):
    n, h, nodes, width = features.shape
    if targets.shape != (n, h, nodes, 3):
        raise ValueError("training shapes differ")
    return [
        fit_gp(
            features[:, :, j].reshape(-1, width),
            targets[:, :, j].reshape(-1, 3),
            np.repeat(np.arange(len(names)), h),
            GPConfig(),
        )
        for j in range(nodes)
    ]


def predict_open(models, features):
    return np.stack(
        [SHRINKAGE * m.predict(features[:, j]) for j, m in enumerate(models)], axis=1
    )


def covariance_shapes(models, features, indices, robust=False):
    result = []
    for j, model in enumerate(models):
        d = model.design(features[indices, j])
        if robust:
            cov = SHRINKAGE**2 * np.einsum(
                "tp,cpq,sq->cts", d, model.cluster_covariance, d
            )
        else:
            shape = SHRINKAGE**2 * d @ model.precision_inverse @ d.T
            cov = np.broadcast_to(shape, (3, len(d), len(d))).copy()
        result.append((cov + cov.swapaxes(-1, -2)) / 2)
    return np.stack(result)


def calibrate(error, shapes):
    """Follow-up v2 covariance family, fitted to recording-OOF errors only.

    shapes: [recording,node,coordinate,20,20]. No eigenvalue clipping.
    Positive diagonal noise makes the covariance strictly positive definite.
    """
    q = np.maximum(np.mean(error**2, axis=(0, 1)), Q_FLOOR)
    parameters = []
    for node in range(error.shape[2]):
        norm = float(np.diagonal(shapes[:, node], axis1=-2, axis2=-1).mean())
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("invalid covariance normalization")
        shape = shapes[:, node] / norm
        standardized = (
            error[:, GRID, node].transpose(0, 2, 1) / np.sqrt(q[node])[None, :, None]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(shape)
        if eigenvalues.min() < -1e-7:
            raise ValueError("covariance shape is not PSD")
        projected = np.einsum("ncti,nct->nci", eigenvectors, standardized)
        squared = projected**2

        def objective(log_ab, eigenvalues=eigenvalues, squared=squared):
            a, b = np.exp(log_ab)
            variance = a * eigenvalues + b
            if np.any(variance <= 0):
                raise ValueError("nonpositive calibrated spectrum")
            loss = 0.5 * np.mean(np.log(variance) + squared / variance)
            dv = 0.5 * (1 / variance - squared / variance**2) / squared.size
            gradient = np.array([np.sum(dv * a * eigenvalues), np.sum(dv * b)])
            return float(loss), gradient

        opt = minimize(
            objective,
            np.log([0.5, 0.5]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(np.log(1e-6), np.log(1e3))] * 2,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not opt.success:
            raise RuntimeError(f"covariance calibration failed: {opt.message}")
        a, b = np.exp(opt.x)
        parameters.append(
            {
                "a": float(a),
                "b": float(b),
                "normalizer": norm,
                "q_m2": q[node].tolist(),
                "objective": float(opt.fun),
            }
        )
    return parameters


def condition_block(covariance, innovation, observed_count):
    pp = covariance[:observed_count, :observed_count]
    fp = covariance[observed_count:, :observed_count]
    factor = cho_factor(pp, lower=True)
    gain = cho_solve(factor, fp.T).T
    conditional = covariance[observed_count:, observed_count:] - gain @ fp.T
    conditional = (conditional + conditional.T) / 2
    np.linalg.cholesky(conditional)
    return gain @ innovation, conditional


def forecast_pair(models, parameters, features, prefix_target, endpoint, horizon):
    """Only the ten supplied past free-node observations are accepted."""
    obs = prefix_indices(endpoint)
    future = np.arange(endpoint, endpoint + horizon)
    if future[-1] >= len(features) or prefix_target.shape != (10, len(models), 3):
        raise ValueError("prefix or horizon contract differs")
    indices = np.r_[obs, future]
    open_all = predict_open(models, features)
    b = open_all[future].copy()
    c = b.copy()
    innovation = prefix_target - open_all[obs]
    shape = covariance_shapes(models, features, indices)
    robust_shape = covariance_shapes(models, features, indices, robust=True)
    calibrated = np.empty((len(models), 3, horizon, horizon))
    raw = np.empty_like(calibrated)
    for node, (p, model) in enumerate(zip(parameters, models, strict=True)):
        for coordinate in range(3):
            k = p["q_m2"][coordinate] * (
                p["a"] * shape[node, coordinate] / p["normalizer"]
                + p["b"] * np.eye(len(indices))
            )
            delta, conditional = condition_block(k, innovation[:, node, coordinate], 10)
            c[:, node, coordinate] += delta
            calibrated[node, coordinate] = conditional
            k_raw = model.residual_variance[coordinate] * (
                shape[node, coordinate] + np.eye(len(indices))
            )
            _, raw[node, coordinate] = condition_block(
                k_raw, innovation[:, node, coordinate], 10
            )
    return (
        b,
        c,
        innovation,
        TemporalBlocks(calibrated),
        TemporalBlocks(raw),
        robust_shape,
    )


def sandwich_operator(shapes, parameters, innovation):
    horizon = shapes.shape[-1] - 10
    blocks = np.empty((*shapes.shape[:2], horizon, horizon))
    for node, p in enumerate(parameters):
        for coordinate in range(3):
            k = p["q_m2"][coordinate] * (
                p["a"] * shapes[node, coordinate] / p["normalizer"]
                + p["b"] * np.eye(horizon + 10)
            )
            _, blocks[node, coordinate] = condition_block(
                k, innovation[:, node, coordinate], 10
            )
    return TemporalBlocks(blocks)


def world_forecast(baseline, free_local, frame):
    result = baseline.copy()
    result[:, 2:-2] += np.einsum("tnj,ij->tni", free_local, frame)
    if not np.array_equal(result[:, [0, 1, -2, -1]], baseline[:, [0, 1, -2, -1]]):
        raise AssertionError("clamped-node parity failed")
    return result
