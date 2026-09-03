"""Same-query-marginal dependence controls for genuinely multivariate events."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtri
from scipy.stats import qmc

ARMS = (
    "structured_gaussian",
    "independent",
    "scrambled",
    "query_gaussian",
    "empirical_copula",
)
EVENTS = (
    "high_load_and_imbalance",
    "high_load_or_imbalance",
    "horizontal_and_vertical",
    "any_spatial_imbalance",
    "two_spatial_imbalances",
)


def query_bank(dimension: int) -> np.ndarray:
    if dimension < 192 or dimension % 96:
        raise ValueError("expected at least two pooled 6x16 tactile sensors")
    sensors = dimension // 96
    weights = np.zeros((5, sensors, 6, 16))
    weights[0] = 1 / dimension
    weights[1, 0] = 1 / 96
    weights[1, -1] = -1 / 96
    weights[2, :, :, :8] = -2 / dimension
    weights[2, :, :, 8:] = 2 / dimension
    weights[3, :, :3, :] = -2 / dimension
    weights[3, :, 3:, :] = 2 / dimension
    center = np.zeros((sensors, 6, 16), dtype=bool)
    center[:, 2:4, 4:12] = True
    weights[4, center] = 1 / int(center.sum())
    weights[4, ~center] = -1 / int((~center).sum())
    return weights.reshape(5, dimension)


def event_values(queries: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    queries = np.asarray(queries, dtype=float)
    if queries.shape[-1] != 5 or np.shape(thresholds) != (5,):
        raise ValueError("five queries and five thresholds required")
    values = np.abs(queries).copy()
    values[..., 0] = queries[..., 0]
    high = values > thresholds
    return np.stack(
        (
            high[..., 0] & high[..., 1],
            high[..., 0] | high[..., 1],
            high[..., 2] & high[..., 3],
            np.any(high[..., 2:], axis=-1),
            np.sum(high[..., 2:], axis=-1) >= 2,
        ),
        axis=-1,
    )


def source_thresholds(truth: np.ndarray, quantile: float) -> np.ndarray:
    values = np.abs(truth).copy()
    values[:, 0] = truth[:, 0]
    return np.quantile(values, quantile, axis=0)


def correlation(covariance: np.ndarray, ridge: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (5, 5) or not np.isfinite(covariance).all():
        raise ValueError("invalid five-query covariance")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    result = covariance / np.outer(scale, scale)
    np.fill_diagonal(result, 1.0)
    result = (1 - ridge) * result + ridge * np.eye(5)
    np.linalg.cholesky(result)
    return result


def rank_couple(marginals: np.ndarray, template: np.ndarray) -> np.ndarray:
    if marginals.shape != template.shape or not np.isfinite(template).all():
        raise ValueError("coupling shapes or values invalid")
    ordered = np.argsort(template, axis=0, kind="stable")
    result = np.empty_like(marginals)
    np.put_along_axis(result, ordered, marginals, axis=0)
    return result


def coupled_draws(
    errors: np.ndarray,
    projected_covariance: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if errors.ndim != 2 or errors.shape[1] != 5 or len(errors) < 8:
        raise ValueError("insufficient finite five-query residuals")
    if not np.isfinite(errors).all():
        raise ValueError("nonfinite source residuals")
    centered = errors - errors.mean(axis=0)
    power = int(protocol["sobol_power"])
    count = 2**power
    marginal = np.quantile(centered, (np.arange(count) + 0.5) / count, axis=0)
    marginal -= marginal.mean(axis=0)
    ridge = float(protocol["correlation_identity_ridge"])
    full = correlation(projected_covariance, ridge)
    direct = correlation(np.cov(centered, rowvar=False), ridge)
    permutation = [2, 4, 1, 0, 3]
    signs = np.array([1, -1, 1, -1, 1])
    scrambled = full[np.ix_(permutation, permutation)] * np.outer(signs, signs)
    factors = {
        "structured_gaussian": np.linalg.cholesky(full),
        "independent": np.eye(5),
        "scrambled": np.linalg.cholesky(scrambled),
        "query_gaussian": np.linalg.cholesky(direct),
    }
    draws: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    for repeat in range(int(protocol["integration_replicates"])):
        seed = int(protocol["random_seed"]) + repeat
        uniform = qmc.Sobol(5, scramble=True, seed=seed).random_base2(power)
        normal = ndtri(np.clip(uniform, 1e-12, 1 - 1e-12))
        for arm, factor in factors.items():
            draws[arm].append(rank_couple(marginal, normal @ factor.T))
        rng = np.random.default_rng(seed)
        indices = np.floor((np.arange(count) + 0.5) * len(errors) / count).astype(int)
        rng.shuffle(indices)
        # Shared rows retain the empirical copula; random ordering breaks only ties.
        draws["empirical_copula"].append(rank_couple(marginal, centered[indices]))
    stacked = {arm: np.stack(value) for arm, value in draws.items()}
    parity = max(
        float(np.max(np.abs(np.sort(value, axis=1) - marginal)))
        for value in stacked.values()
    )
    mean_error = max(
        float(np.max(np.abs(value.mean(axis=1)))) for value in stacked.values()
    )
    return stacked, {
        "sorted_query_marginal_max_error": parity,
        "shared_point_mean_max_error": mean_error,
    }


def event_predictions(
    mean: np.ndarray,
    thresholds: np.ndarray,
    draws: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    pairs = list(combinations(range(5), 2))
    result: dict[str, np.ndarray] = {}
    for arm, samples in draws.items():
        repeats, count, _ = samples.shape
        probabilities = np.empty((repeats, len(mean), len(EVENTS)))
        variograms = np.empty((len(mean), len(pairs)))
        for index, point in enumerate(mean):
            values = point + samples
            probabilities[:, index] = event_values(values, thresholds).mean(axis=1)
            for pair_index, (left, right) in enumerate(pairs):
                variograms[index, pair_index] = np.mean(
                    np.sqrt(np.abs(values[..., left] - values[..., right]))
                )
        result[f"p_{arm}"] = probabilities.mean(axis=0)
        result[f"integration_sd_{arm}"] = probabilities.std(axis=0)
        result[f"variogram_{arm}"] = variograms
    return result


def fit_direct_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> dict[str, np.ndarray]:
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    design = np.column_stack((np.ones(len(features)), (features - center) / scale))
    coefficients = np.zeros((len(EVENTS), design.shape[1]))
    for index in range(len(EVENTS)):
        y = labels[:, index].astype(float)
        if np.all(y == y[0]):
            p = (y.sum() + 1) / (len(y) + 2)
            coefficients[index, 0] = np.log(p / (1 - p))
            continue

        def objective(beta: np.ndarray, y: np.ndarray = y) -> tuple[float, np.ndarray]:
            logits = design @ beta
            penalty = beta.copy()
            penalty[0] = 0
            loss = np.sum(np.logaddexp(0, logits) - y * logits)
            gradient = design.T @ (expit(logits) - y) + l2 * penalty
            return float(loss + l2 * np.dot(penalty, penalty) / 2), gradient

        fitted = minimize(
            objective,
            coefficients[index],
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7},
        )
        if not fitted.success:
            raise ValueError(f"direct logistic did not converge: {fitted.message}")
        coefficients[index] = fitted.x
    return {"center": center, "scale": scale, "coefficients": coefficients}


def direct_predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    design = np.column_stack(
        (np.ones(len(features)), (features - model["center"]) / model["scale"])
    )
    return expit(design @ model["coefficients"].T)


def score_predictions(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    thresholds: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    labels = event_values(truth, thresholds).astype(float)
    fallback = float(protocol["loss"]["fallback"])
    clipping = float(protocol["probability_log_clip"])
    metrics: dict[str, Any] = {}
    for key, p in predictions.items():
        if not key.startswith("p_"):
            continue
        if (
            p.shape != labels.shape
            or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))
        ):
            raise ValueError("invalid sealed event probabilities")
        clipped = np.clip(p, clipping, 1 - clipping)
        act = p < fallback
        loss = np.where(act, labels, fallback)
        metrics[key[2:]] = {
            "brier": float(np.mean((p - labels) ** 2)),
            "log_loss": float(
                np.mean(-labels * np.log(clipped) - (1 - labels) * np.log1p(-clipped))
            ),
            "decision_loss": float(loss.mean()),
            "execute_fraction": float(act.mean()),
            "brier_by_event": np.mean((p - labels) ** 2, axis=0).tolist(),
            "loss_by_event": loss.mean(axis=0).tolist(),
        }
        if f"variogram_{key[2:]}" in predictions:
            observed = np.column_stack(
                [
                    np.sqrt(np.abs(truth[:, left] - truth[:, right]))
                    for left, right in combinations(range(5), 2)
                ]
            )
            metrics[key[2:]]["variogram_score"] = float(
                np.mean((observed - predictions[f"variogram_{key[2:]}"]) ** 2)
            )
    metrics["always_fallback"] = {"decision_loss": fallback, "execute_fraction": 0.0}
    metrics["always_execute"] = {
        "decision_loss": float(labels.mean()),
        "execute_fraction": 1.0,
    }
    full_act = predictions["p_structured_gaussian"] < fallback
    direct_act = np.zeros_like(full_act)
    for event in range(len(EVENTS)):
        order = np.argsort(predictions["p_direct_logistic"][:, event], kind="stable")
        direct_act[order[: int(full_act[:, event].sum())], event] = True
    rates = full_act.mean(axis=0)
    metrics["matched_activity_direct"] = {
        "decision_loss": float(np.where(direct_act, labels, fallback).mean()),
        "execute_fraction": float(direct_act.mean()),
    }
    metrics["matched_activity_random_expected"] = {
        "decision_loss": float(
            np.mean(rates * labels.mean(axis=0) + (1 - rates) * fallback)
        ),
        "execute_fraction": float(rates.mean()),
    }
    return {"metrics": metrics, "event_frequencies": labels.mean(axis=0).tolist()}


def aggregate(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"object_count": 0, "superiority_gate": False, "decision_gate": False}
    methods = rows[0]["metrics"]
    means = {
        method: {
            name: float(np.mean([row["metrics"][method][name] for row in rows]))
            for name, value in methods[method].items()
            if isinstance(value, (int, float))
        }
        for method in methods
    }
    rng = np.random.default_rng(int(protocol["random_seed"]))
    indices = rng.integers(
        0, len(rows), (int(protocol["bootstrap_repetitions"]), len(rows))
    )
    comparators = protocol["primary_comparators"]
    adjusted_tail = 0.05 / (2 * len(comparators))
    contrasts = {}
    for comparator in comparators:
        delta = np.array(
            [
                row["metrics"]["structured_gaussian"]["brier"]
                - row["metrics"][comparator]["brier"]
                for row in rows
            ]
        )
        bootstrap = delta[indices].mean(axis=1)
        contrasts[comparator] = {
            "full_minus_comparator": float(delta.mean()),
            "object_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "adjusted_interval": np.quantile(
                bootstrap, [adjusted_tail, 1 - adjusted_tail]
            ).tolist(),
            "wins": int(np.sum(delta < 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta > 0)),
        }
    enough = len(rows) >= int(protocol["minimum_objects_for_aggregate_claim"])
    full = means["structured_gaussian"]
    return {
        "object_count": len(rows),
        "methods": means,
        "brier_contrasts": contrasts,
        "superiority_gate": bool(
            enough
            and all(item["adjusted_interval"][1] < 0 for item in contrasts.values())
        ),
        "decision_gate": bool(
            enough
            and full["execute_fraction"] >= 0.2
            and full["decision_loss"] < means["always_fallback"]["decision_loss"]
            and full["decision_loss"] < means["always_execute"]["decision_loss"]
        ),
        "fresh_confirmation": False,
        "uniquely_bayesian_advantage": False,
    }
