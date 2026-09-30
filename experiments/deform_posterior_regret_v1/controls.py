"""Equally informed, trajectory-fitted empirical admission controls.

No control constructs a different point forecast. Entire recording residuals,
not independently resampled coordinates or frames, form bootstrap draws.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit

RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)
LENGTHS = (0.5, 1.0, 2.0, 4.0)
NEIGHBORS = (4, 8, 16, 32)
INDEPENDENT = (
    "expected_regret",
    "constant_diagonal",
    "sandwich",
    "trajectory_bootstrap",
    "empirical_regret",
    "ridge_risk",
    "kernel_risk",
)


def context_features(baseline, candidate, innovation, boundary, endpoint, horizon):
    """Only prefix, identical forecasts and known clamped-node future inputs."""
    d = candidate - baseline
    return np.r_[
        innovation.ravel() * 1000,
        d.mean(0).ravel() * 1000,
        np.sqrt(np.mean(d**2, axis=0)).ravel() * 1000,
        baseline.mean(0).ravel() * 1000,
        np.sqrt(np.mean(baseline**2, axis=0)).ravel() * 1000,
        (boundary[-1] - boundary[0]).ravel() * 1000,
        np.sqrt(np.mean(np.diff(boundary, axis=0) ** 2, axis=0)).ravel() * 1000,
        endpoint / 498,
        horizon / 498,
    ]


def standardize(train, query):
    location, scale = train.mean(0), train.std(0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return (train - location) / scale, (query - location) / scale


def regression(train, labels, query, kind, setting):
    x, z = standardize(train, query)
    if kind == "ridge_risk":
        d, q = np.c_[np.ones(len(x)), x], np.c_[np.ones(len(z)), z]
        penalty = setting * np.eye(d.shape[1])
        penalty[0, 0] = 0
        weights = solve(d.T @ d + penalty, d.T @ labels, assume_a="pos")
        result = q @ weights
    else:
        distances = cdist(x, x, "sqeuclidean") / x.shape[1]
        kernel = np.exp(-distances / (2 * setting**2))
        # Frozen regularization one; bandwidth alone is source-selected.
        weights = solve(kernel + np.eye(len(x)), labels, assume_a="pos")
        result = (
            np.exp(-cdist(z, x, "sqeuclidean") / (2 * setting**2 * x.shape[1]))
            @ weights
        )
    return result


def regret_predictions(training, queries, kind, setting):
    """Predict actual regret; use recording-OOF regression errors for its CDF."""
    x = np.asarray([r["features"] for r in training])
    z = np.asarray([r["features"] for r in queries])
    regrets = np.asarray([r["regret_mm2"] for r in training])
    recordings = np.asarray([r["recording"] for r in training])
    unique = sorted(set(recordings))
    residual = np.empty(len(training))
    for held in np.array_split(np.asarray(unique), 4):
        mask = np.isin(recordings, held)
        residual[mask] = regrets[mask] - regression(
            x[~mask], regrets[~mask], x[mask], kind, setting
        )
    predicted = regression(x, regrets, z, kind, setting)
    # Each training recording contributes the same number of contexts.
    return (np.sum(predicted[:, None] + residual[None] > 1, axis=1) + 0.5) / (
        len(residual) + 1
    )


def empirical_predictions(training, queries, kind, neighbors):
    x = np.asarray([r["features"] for r in training])
    z = np.asarray([r["features"] for r in queries])
    x, z = standardize(x, z)
    distance = cdist(z, x, "sqeuclidean") / x.shape[1]
    probabilities = []
    for row, query in enumerate(queries):
        eligible = [
            i
            for i, r in enumerate(training)
            if r["endpoint"] == query["endpoint"] and r["horizon"] == query["horizon"]
        ]
        order = sorted(
            eligible, key=lambda i: (float(distance[row, i]), training[i]["key"])
        )
        chosen = order[:neighbors]
        if not chosen:
            raise ValueError("no trajectory-aligned empirical support")
        if kind == "empirical_regret":
            regrets = np.asarray([training[i]["regret_mm2"] for i in chosen])
        else:
            # Each draw transports ONE whole source recording's future residual.
            # Its ten-frame prefix selects the neighbors; no query suffix is read.
            d = query["c"] - query["b"]
            regrets = np.asarray(
                [
                    1e6 * np.mean(d**2 - 2 * d * training[i]["open_error"])
                    for i in chosen
                ]
            )
        probabilities.append((np.sum(regrets > 1.0) + 0.5) / (len(chosen) + 1))
    return np.asarray(probabilities)


def fit_probability_calibration(probabilities, outcomes):
    """Source-only positive-slope logistic calibration; raw scores are retained."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid raw probability")
    # Declared continuity correction is for probability calibration, not PSD repair.
    p = (p * len(p) + 0.5) / (len(p) + 1)
    logit = np.log(p / (1 - p))

    def objective(theta):
        slope = np.exp(theta[0])
        eta = slope * logit + theta[1]
        loss = np.mean(np.logaddexp(0, eta) - y * eta) + 0.01 * np.sum(theta**2)
        residual = expit(eta) - y
        gradient = (
            np.array([np.mean(residual * slope * logit), residual.mean()])
            + 0.02 * theta
        )
        return loss, gradient

    result = minimize(
        objective, np.zeros(2), jac=True, method="L-BFGS-B", bounds=[(-5, 5), (-10, 10)]
    )
    if not result.success:
        raise RuntimeError("probability calibration failed")
    return {
        "log_slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "training_context_count": len(p),
    }


def apply_probability_calibration(probabilities, fit):
    n = fit["training_context_count"]
    p = (np.asarray(probabilities) * n + 0.5) / (n + 1)
    return expit(np.exp(fit["log_slope"]) * np.log(p / (1 - p)) + fit["intercept"])


def predict_empirical(training, queries, settings):
    result = {}
    for kind in ("ridge_risk", "kernel_risk"):
        result[kind] = regret_predictions(training, queries, kind, settings[kind])
    for kind in ("trajectory_bootstrap", "empirical_regret"):
        result[kind] = empirical_predictions(training, queries, kind, settings[kind])
    return result


def choose_settings(panels):
    """Panels are independently refitted inner train/held recordings."""
    chosen, scores = {}, {}
    for kind, settings in (
        ("ridge_risk", RIDGES),
        ("kernel_risk", LENGTHS),
        ("trajectory_bootstrap", NEIGHBORS),
        ("empirical_regret", NEIGHBORS),
    ):
        losses = []
        for setting in settings:
            values = []
            for train, held in panels:
                truth = np.asarray([r["regret_mm2"] > 1 for r in held])
                if kind in ("ridge_risk", "kernel_risk"):
                    p = regret_predictions(train, held, kind, setting)
                else:
                    p = empirical_predictions(train, held, kind, setting)
                # Three cuts per recording and each horizon are balanced.
                values.append(float(np.mean((p - truth) ** 2)))
            losses.append(float(np.mean(values)))
        best = min(range(len(settings)), key=lambda i: (losses[i], settings[i]))
        chosen[kind] = settings[best]
        scores[kind] = {str(s): v for s, v in zip(settings, losses, strict=True)}
    return chosen, scores
