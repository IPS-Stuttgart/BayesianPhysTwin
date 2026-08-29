"""Small NumPy spring-mesh pilot and source-only generalized Bayes.

This is an explicitly limited equal-marker-mass spring model, not PhysTwin,
clothilde-sim, a FEM reproduction, or a newly validated material estimator.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import Inputs

ARMS = (
    "persistence", "nominal_physics", "last_residual", "nominal_state_injection",
    "map_physics", "bayesian_physics", "guarded_bayesian_physics",
)


def parameter_bank(protocol: dict[str, Any]) -> list[tuple[float, float]]:
    return list(itertools.product(protocol["stiffness_per_mass"], protocol["damping_per_mass"]))


def edges(markers: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = (5, 4) if markers == 20 else (4, 3)
    links, weights = [], []
    for r in range(rows):
        for c in range(cols):
            for dr, dc, weight in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.5),
                                    (1, -1, 0.5), (0, 2, 0.1), (2, 0, 0.1)):
                if 0 <= r + dr < rows and 0 <= c + dc < cols:
                    links.append((r * cols + c, (r + dr) * cols + c + dc))
                    weights.append(weight)
    return np.asarray(links, dtype=int), np.asarray(weights)


def velocity(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
    t = times - times.mean()
    return np.einsum("t,tnd->nd", t, positions) / np.dot(t, t)


def rollout(inputs: Inputs, parameters: tuple[float, float], protocol: dict[str, Any],
            inject: bool) -> np.ndarray:
    """Symplectic spring rollout with recorded Dirichlet corner input.

    Rest edge lengths use the initial frame, not the forecast outcomes. A state
    injection uses the final permitted prefix position and backward-only velocity.
    """
    links, relative_k = edges(len(inputs.order))
    left, right = links.T
    rest = np.linalg.norm(inputs.prefix[0, right] - inputs.prefix[0, left], axis=1)
    if np.min(rest) <= 1e-6:
        raise ValueError("Degenerate initial spring")
    k, damping = parameters
    start = inputs.cutoff if inject else 0
    x = inputs.prefix[start].copy()
    if inject:
        v = velocity(inputs.times[start - 4:start + 1], inputs.prefix[start - 4:start + 1])
    else:
        v = velocity(inputs.times[:5], inputs.prefix[:5])
    result = np.empty((len(inputs.times), len(inputs.order), 3))
    result[:start + 1] = inputs.prefix[:start + 1]
    origin = inputs.prefix[0].mean(axis=0)
    substeps = protocol["integration_substeps"]
    for t in range(start + 1, len(inputs.times)):
        full_dt = inputs.times[t] - inputs.times[t - 1]
        dt = full_dt / substeps
        boundary_v = (inputs.boundary[t] - inputs.boundary[t - 1]) / full_dt
        for sub in range(1, substeps + 1):
            delta = x[right] - x[left]
            lengths = np.linalg.norm(delta, axis=1)
            force = (k * relative_k * (lengths - rest) / np.maximum(lengths, 1e-9))[:, None] * delta
            acceleration = -damping * v
            acceleration[:, 2] -= protocol["gravity_m_s2"]
            np.add.at(acceleration, left, force)
            np.add.at(acceleration, right, -force)
            v += dt * acceleration
            x += dt * v
            fraction = sub / substeps
            x[inputs.corners] = ((1 - fraction) * inputs.boundary[t - 1]
                                 + fraction * inputs.boundary[t])
            v[inputs.corners] = boundary_v
        if not np.isfinite(x).all() or np.max(np.linalg.norm(x - origin, axis=1)) > 10:
            raise ValueError("Numerically invalid rollout; no silent case deletion")
        result[t] = x
    return result


@dataclass(frozen=True)
class Predictions:
    inputs: Inputs
    nominal: np.ndarray
    bank: np.ndarray


def predict(inputs: Inputs, protocol: dict[str, Any]) -> Predictions:
    nominal = rollout(inputs, tuple(protocol["nominal_parameters"]), protocol, False)
    bank = np.stack([rollout(inputs, parameters, protocol, True)
                     for parameters in parameter_bank(protocol)])
    return Predictions(inputs, nominal, bank)


def masks(inputs: Inputs, truth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(truth).all(axis=2)
    valid[:inputs.cutoff + 1] = False
    valid[:, inputs.corners] = False
    if not np.any(valid):
        raise ValueError("No evaluable free-marker forecast samples")
    return valid


def squared_error(mean: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    if not np.isfinite(mean).all():
        raise ValueError("Nonfinite prediction")
    return float(np.mean(np.sum((mean[valid] - truth[valid]) ** 2, axis=1)))


def source_weights(losses: np.ndarray, floor_m: float) -> np.ndarray:
    """Gibbs update: one normalized trajectory loss per source recording.

    The source-derived temperature is fixed before target use. Marker/time rows
    are not counted as independent likelihood groups or inferential replicates.
    """
    temperature = max(float(np.median(np.min(losses, axis=1))), floor_m ** 2)
    logits = -np.sum(losses, axis=0) / (2 * temperature)
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


def means(predictions: Predictions, weights: np.ndarray, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    inputs = predictions.inputs
    residual = inputs.prefix[-1] - predictions.nominal[inputs.cutoff]
    nominal_index = parameter_bank(protocol).index(tuple(protocol["nominal_parameters"]))
    result = {
        "persistence": np.broadcast_to(inputs.prefix[-1], predictions.nominal.shape).copy(),
        "nominal_physics": predictions.nominal,
        "last_residual": predictions.nominal + residual,
        "nominal_state_injection": predictions.bank[nominal_index],
        "map_physics": predictions.bank[int(np.argmax(weights))],
        "bayesian_physics": np.einsum("k,ktnd->tnd", weights, predictions.bank),
    }
    # The boundary is a conditioning input, never a scored output.
    for name in ("persistence", "last_residual"):
        result[name][:, inputs.corners] = inputs.boundary
    return result


def horizon_bins(inputs: Inputs) -> np.ndarray:
    duration = inputs.times[-1] - inputs.times[inputs.cutoff]
    return np.clip(((inputs.times - inputs.times[inputs.cutoff]) / duration * 3).astype(int), 0, 2)


def fit_specimen(records: list[tuple[Predictions, np.ndarray]], protocol: dict[str, Any]) -> dict[str, Any]:
    """Leave one speed/grasp source recording out; never access target outcomes."""
    if len(records) != 4:
        raise ValueError("Exactly four shaking source conditions are required per specimen")
    loss = np.asarray([[squared_error(p, truth, masks(pred.inputs, truth)) for p in pred.bank]
                       for pred, truth in records])
    oof = []
    residual_squares = {arm: [[], [], []] for arm in ARMS[:-1]}
    for held, (prediction, truth) in enumerate(records):
        w = source_weights(np.delete(loss, held, axis=0), protocol["measurement_floor_m"])
        arm_means = means(prediction, w, protocol)
        valid = masks(prediction.inputs, truth)
        bins = horizon_bins(prediction.inputs)
        oof.append({arm: np.sqrt(squared_error(mean, truth, valid))
                    for arm, mean in arm_means.items()})
        ensemble_var = np.einsum("k,ktnd->tnd", w,
                                 (prediction.bank - arm_means["bayesian_physics"]) ** 2)
        for arm, mean in arm_means.items():
            for b in range(3):
                select = valid & (bins[:, None] == b)
                if not np.any(select):
                    raise ValueError("Empty source calibration horizon bin")
                error2 = (mean[select] - truth[select]) ** 2
                if arm == "bayesian_physics":
                    error2 = error2 - ensemble_var[select]
                # Equal recording contribution to each calibration bin.
                residual_squares[arm][b].append(float(np.mean(error2)))
    baseline = np.asarray([row["nominal_physics"] for row in oof])
    residual = np.asarray([row["last_residual"] for row in oof])
    candidate = np.asarray([row["bayesian_physics"] for row in oof])
    reference = min(float(baseline.mean()), float(residual.mean()))
    accepted = bool(np.all(candidate <= baseline) and
                    candidate.mean() < (1 - protocol["guard_minimum_relative_gain"]) * reference)
    noise = {arm: [max(float(np.mean(values)), protocol["measurement_floor_m"] ** 2)
                    for values in bins] for arm, bins in residual_squares.items()}
    return {
        "source_posterior_weights": source_weights(loss, protocol["measurement_floor_m"]).tolist(),
        "guard_accepts": accepted,
        "guard_basis": "OOF mean beats nominal and last_residual by frozen margin; no OOF nominal regression",
        "oof_record_rmse_m": oof,
        "source_residual_variance_m2": noise,
    }


def complete_beliefs(prediction: Predictions, fit: dict[str, Any], protocol: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    weights = np.asarray(fit["source_posterior_weights"])
    arm_means = means(prediction, weights, protocol)
    bins = horizon_bins(prediction.inputs)
    beliefs = {}
    for arm, mean in arm_means.items():
        variance = np.broadcast_to(np.asarray(fit["source_residual_variance_m2"][arm])[bins, None, None], mean.shape).copy()
        if arm == "bayesian_physics":
            variance += np.einsum("k,ktnd->tnd", weights, (prediction.bank - mean) ** 2)
        beliefs[arm] = (mean, variance)
    # Preserve both mean and covariance together; do not reconstruct on rejection.
    chosen = "bayesian_physics" if fit["guard_accepts"] else "nominal_physics"
    beliefs["guarded_bayesian_physics"] = beliefs[chosen]
    return beliefs


def score(mean: np.ndarray, variance: np.ndarray, truth: np.ndarray, inputs: Inputs) -> dict[str, float | int]:
    valid = masks(inputs, truth)
    e = mean[valid] - truth[valid]
    var = variance[valid]
    if not np.isfinite(var).all() or np.any(var <= 0):
        raise ValueError("Invalid predictive variance")
    return {
        "rmse_mm": 1000 * float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
        "mean_marker_error_mm": 1000 * float(np.mean(np.linalg.norm(e, axis=1))),
        "coordinate_nll": float(np.mean(0.5 * (np.log(2 * np.pi * var) + e ** 2 / var))),
        "coordinate_90_coverage": float(np.mean(abs(e) <= 1.6448536269514722 * np.sqrt(var))),
        "mean_full_90_width_mm": 1000 * float(np.mean(2 * 1.6448536269514722 * np.sqrt(var))),
        "free_marker_samples": int(valid.sum()),
        "missing_free_marker_samples": int((len(inputs.times) - inputs.cutoff - 1) *
                                            (len(inputs.order) - 2) - valid.sum()),
    }
