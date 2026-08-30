"""Belief construction and sealing helpers for the active-probe cloth pilot.

The helpers in this module are deliberately outcome-blind.  They turn frozen
finite-model weights into complete mean/variance trajectories and expose the
small set of deterministic operations needed by the registered retrospective
probe study.  They do not select from or inspect held-out twisting outcomes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .active_probe import POLICIES, normalize_weights
from .data import Inputs
from .model import Predictions, horizon_bins


def _finite_array(value: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _residual_variance(value: object, *, name: str) -> np.ndarray:
    variance = _finite_array(value, name=name, ndim=1)
    if variance.shape != (3,) or np.any(variance <= 0.0):
        raise ValueError(f"{name} must contain three positive horizon variances")
    return variance


def _variance_field(
    inputs: Inputs, shape: tuple[int, ...], residual_variance: object
) -> np.ndarray:
    residual = _residual_variance(
        residual_variance, name="residual_variance"
    )
    bins = horizon_bins(inputs)
    variance = np.broadcast_to(residual[bins, None, None], shape).copy()
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise ValueError("predictive variance must be finite and positive")
    return variance


def active_mask(inputs: Inputs) -> np.ndarray:
    """Return the registered post-prefix, non-driven-marker mask."""
    times = np.asarray(inputs.times)
    order = np.asarray(inputs.order)
    corners = np.asarray(inputs.corners, dtype=int)
    if times.ndim != 1 or order.ndim != 1:
        raise ValueError("times and marker order must be one-dimensional")
    if not 0 <= int(inputs.cutoff) < len(times):
        raise ValueError("cutoff is outside the trajectory")
    if corners.shape != (2,) or np.any(corners < 0) or np.any(corners >= len(order)):
        raise ValueError("exactly two valid driven-corner indices are required")
    mask = np.ones((len(times), len(order)), dtype=bool)
    mask[: int(inputs.cutoff) + 1] = False
    mask[:, corners] = False
    if not np.any(mask):
        raise ValueError("active mask contains no scored samples")
    return mask


def array_digest(value: object) -> str:
    """Hash array values together with their canonical shape and dtype."""
    array = np.ascontiguousarray(np.asarray(value))
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def belief_digest(mean: object, variance: object) -> str:
    """Bind a complete mean and variance trajectory into one content ID."""
    payload = json.dumps(
        {"mean": array_digest(mean), "variance": array_digest(variance)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def weighted_belief(
    prediction: Predictions, weights: object, residual_variance: object
) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match one finite-model trajectory mixture."""
    bank = _finite_array(prediction.bank, name="prediction.bank", ndim=4)
    if bank.shape[0] < 2 or bank.shape[-1] != 3:
        raise ValueError("prediction bank must have shape (models, time, markers, 3)")
    probabilities = normalize_weights(weights)
    if probabilities.shape != (bank.shape[0],):
        raise ValueError("weights must match the prediction-bank model count")
    mean = np.einsum("k,ktnd->tnd", probabilities, bank)
    variance = np.einsum(
        "k,ktnd->tnd", probabilities, (bank - mean[None, ...]) ** 2
    )
    variance += _variance_field(prediction.inputs, mean.shape, residual_variance)
    if not np.all(np.isfinite(mean)) or np.any(variance <= 0.0):
        raise ValueError("invalid moment-matched belief")
    return mean, variance


def loss_vector(prediction: Predictions, truth: object) -> np.ndarray:
    """Return one normalized trajectory loss for every model member."""
    bank = _finite_array(prediction.bank, name="prediction.bank", ndim=4)
    observed = np.asarray(truth, dtype=np.float64)
    if observed.shape != bank.shape[1:]:
        raise ValueError("truth must match one prediction-bank trajectory")
    valid = active_mask(prediction.inputs) & np.isfinite(observed).all(axis=2)
    if not np.any(valid):
        raise ValueError("truth contains no evaluable active samples")
    if not np.all(np.isfinite(bank[:, valid, :])):
        raise ValueError("prediction bank is nonfinite on evaluable samples")
    error = bank[:, valid, :] - observed[valid][None, ...]
    losses = np.mean(np.sum(error * error, axis=-1), axis=1)
    losses = np.maximum(losses, 0.0)
    losses.setflags(write=False)
    return losses


def posterior_temperature(losses: object, measurement_floor_m: float) -> float:
    """Freeze the generalized-Bayes temperature from source records only."""
    matrix = _finite_array(losses, name="losses", ndim=2)
    if matrix.shape[0] < 1 or matrix.shape[1] < 2 or np.any(matrix < 0.0):
        raise ValueError("losses must be nonnegative with shape (records, models>=2)")
    floor = float(measurement_floor_m)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("measurement_floor_m must be finite and positive")
    return max(float(np.median(np.min(matrix, axis=1))), floor * floor)


def _last_residual_mean(
    prediction: Predictions, prefix_last: object, boundary: object
) -> np.ndarray:
    prefix = _finite_array(prefix_last, name="prefix_last", ndim=2)
    if prefix.shape != prediction.nominal.shape[1:]:
        raise ValueError("prefix_last must match the marker-state shape")
    driven = _finite_array(boundary, name="boundary", ndim=3)
    if driven.shape != (prediction.nominal.shape[0], 2, 3):
        raise ValueError("boundary must have shape (time, 2, 3)")
    mean = _finite_array(prediction.nominal, name="prediction.nominal", ndim=3)
    residual = prefix - mean[int(prediction.inputs.cutoff)]
    mean += residual[None, ...]
    mean[:, np.asarray(prediction.inputs.corners, dtype=int)] = driven
    return mean


def calibrated_residuals(
    records: Sequence[tuple[Predictions, object]],
    weights: object,
    protocol: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Estimate three equal-record residual-variance bins from source data."""
    if not records:
        raise ValueError("at least one source record is required")
    floor = float(protocol["measurement_floor_m"])
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("measurement_floor_m must be finite and positive")
    floor2 = floor * floor
    pooled: dict[str, list[list[float]]] = {
        "bayesian": [[], [], []],
        "nominal_physics": [[], [], []],
        "last_residual": [[], [], []],
    }
    for prediction, raw_truth in records:
        truth = np.asarray(raw_truth, dtype=np.float64)
        if truth.shape != prediction.nominal.shape:
            raise ValueError("source truth must match the prediction trajectory")
        valid = active_mask(prediction.inputs) & np.isfinite(truth).all(axis=2)
        if not np.any(valid):
            raise ValueError("source record contains no evaluable samples")
        probabilities = normalize_weights(weights)
        if probabilities.shape != (prediction.bank.shape[0],):
            raise ValueError("weights must match every source prediction bank")
        bayesian_mean = np.einsum("k,ktnd->tnd", probabilities, prediction.bank)
        ensemble_variance = np.einsum(
            "k,ktnd->tnd",
            probabilities,
            (prediction.bank - bayesian_mean[None, ...]) ** 2,
        )
        last_residual = _last_residual_mean(
            prediction,
            prediction.inputs.prefix[-1],
            prediction.inputs.boundary,
        )
        means = {
            "bayesian": bayesian_mean,
            "nominal_physics": prediction.nominal,
            "last_residual": last_residual,
        }
        bins = horizon_bins(prediction.inputs)
        for name, mean in means.items():
            error2 = (mean - truth) ** 2
            if name == "bayesian":
                error2 = error2 - ensemble_variance
            for bin_index in range(3):
                selected = valid & (bins[:, None] == bin_index)
                if not np.any(selected):
                    raise ValueError("empty source calibration horizon bin")
                pooled[name][bin_index].append(
                    max(float(np.mean(error2[selected])), floor2)
                )
    return {
        name: [max(float(np.mean(values)), floor2) for values in horizon_values]
        for name, horizon_values in pooled.items()
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the active-probe roster and its information boundary."""
    conditions = tuple(str(value) for value in protocol["probe_conditions"])
    fixed_order = tuple(str(value) for value in protocol["fixed_probe_order"])
    policies = tuple(str(value) for value in protocol["probe_policies"])
    budgets = tuple(protocol["probe_budgets"])
    if len(conditions) != 4 or len(set(conditions)) != len(conditions):
        raise ValueError("probe_conditions must contain four unique actions")
    if fixed_order != conditions:
        raise ValueError("fixed_probe_order must equal the registered action roster")
    if policies != POLICIES:
        raise ValueError("probe_policies must match the implemented policy roster")
    if budgets != (0, 1, 2, 4):
        raise ValueError("probe_budgets must be exactly [0, 1, 2, 4]")
    if protocol["primary_budget"] not in budgets:
        raise ValueError("primary_budget must be registered")
    if protocol["held_material_candidate_inputs_used_for_selection"] is not False:
        raise ValueError("held-material candidate inputs may not drive selection")
    if protocol["held_material_twist_inputs_used_for_selection"] is not False:
        raise ValueError("held-material twist inputs may not drive selection")
    if protocol["paper_claim_authorized"] is not False:
        raise ValueError("the public-data pilot may not self-authorize a paper claim")


def arm_specs(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the complete 18-arm registered comparison roster."""
    validate_protocol(protocol)
    controls = ("nominal_physics", "last_residual")
    single_probes = tuple(
        f"single_probe_{condition}" for condition in protocol["probe_conditions"]
    )
    policies = tuple(
        f"{policy}_k{budget}"
        for policy in protocol["probe_policies"]
        for budget in protocol["probe_budgets"]
    )
    result = controls + single_probes + policies
    if len(result) != 18 or len(set(result)) != len(result):
        raise ValueError("active-probe arm roster is not complete and unique")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def build_belief_arms(
    prediction: Predictions,
    *,
    prefix_last: object,
    boundary: object,
    fold: Mapping[str, Any],
    specimen: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Construct every registered arm without reading a target outcome."""
    validate_protocol(protocol)
    residuals = _mapping(fold["source_residual_variance_m2"], name="residuals")
    nominal_mean = _finite_array(
        prediction.nominal, name="prediction.nominal", ndim=3
    )
    if nominal_mean.shape != prediction.bank.shape[1:]:
        raise ValueError("nominal trajectory and model bank disagree")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "nominal_physics": (
            nominal_mean,
            _variance_field(
                prediction.inputs,
                nominal_mean.shape,
                residuals["nominal_physics"],
            ),
        )
    }
    last_mean = _last_residual_mean(prediction, prefix_last, boundary)
    result["last_residual"] = (
        last_mean,
        _variance_field(
            prediction.inputs, last_mean.shape, residuals["last_residual"]
        ),
    )
    bayesian_residual = residuals["bayesian"]
    single = _mapping(specimen["single_probe_weights"], name="single_probe_weights")
    for condition in protocol["probe_conditions"]:
        if condition not in single:
            raise ValueError(f"missing single-probe weights for {condition}")
        result[f"single_probe_{condition}"] = weighted_belief(
            prediction, single[condition], bayesian_residual
        )
    states = _mapping(specimen["policy_states"], name="policy_states")
    for policy in protocol["probe_policies"]:
        policy_states = _mapping(states[policy], name=f"policy_states[{policy}]")
        for budget in protocol["probe_budgets"]:
            state = _mapping(
                policy_states[str(budget)],
                name=f"policy_states[{policy}][{budget}]",
            )
            selected = tuple(str(value) for value in state["selected_actions"])
            if len(selected) != int(budget) or len(set(selected)) != len(selected):
                raise ValueError("selected action roster does not match its budget")
            if not set(selected).issubset(protocol["probe_conditions"]):
                raise ValueError("policy state contains an unregistered action")
            result[f"{policy}_k{budget}"] = weighted_belief(
                prediction, state["weights"], bayesian_residual
            )
    expected = set(arm_specs(protocol))
    if set(result) != expected:
        raise ValueError("constructed belief roster differs from the protocol")
    for mean, variance in result.values():
        if mean.shape != nominal_mean.shape or variance.shape != nominal_mean.shape:
            raise ValueError("belief trajectory shape mismatch")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("belief trajectory contains nonfinite values")
        if np.any(variance <= 0.0):
            raise ValueError("belief variance must be positive")
    return result


__all__ = [
    "active_mask",
    "arm_specs",
    "array_digest",
    "belief_digest",
    "build_belief_arms",
    "calibrated_residuals",
    "loss_vector",
    "posterior_temperature",
    "validate_protocol",
    "weighted_belief",
]
