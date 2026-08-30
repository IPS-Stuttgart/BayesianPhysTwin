"""Outcome-blind finite-model probe selection for the tracking-cloth pilot.

The selection rules consume only source-frozen model-disagreement matrices and
current discrete-model weights.  Outcomes are requested from the supplied
mapping only after an action has been selected.  This is a retrospective replay
of already recorded actions, not an online robot controller or a safety policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

POLICIES = ("fixed_order", "parameter_information", "task_directed")


def _finite_array(value: object, *, name: str, ndim: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def normalize_weights(value: object) -> FloatArray:
    """Return a finite, strictly positive normalized model-weight vector."""
    weights = _finite_array(value, name="weights", ndim=1)
    if weights.size < 2 or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("weights must be a nonnegative vector with positive mass")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have finite positive mass")
    weights /= total
    tiny = np.finfo(np.float64).tiny
    weights = np.maximum(weights, tiny)
    weights /= weights.sum()
    weights.setflags(write=False)
    return weights


def _temperature(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return result


def update_weights(weights: object, loss: object, temperature: float) -> FloatArray:
    """Apply one Gibbs update from one normalized trajectory loss vector."""
    prior = normalize_weights(weights)
    losses = _finite_array(loss, name="loss", ndim=1)
    if losses.shape != prior.shape or np.any(losses < 0.0):
        raise ValueError("loss must be nonnegative and match weights")
    temperature = _temperature(temperature)
    logits = np.log(prior) - losses / (2.0 * temperature)
    logits -= float(np.max(logits))
    return normalize_weights(np.exp(logits))


def weights_from_records(losses: object, temperature: float) -> FloatArray:
    """Fit one equal-record generalized-Bayes prior over model members."""
    matrix = _finite_array(losses, name="losses", ndim=2)
    if matrix.shape[0] < 1 or matrix.shape[1] < 2 or np.any(matrix < 0.0):
        raise ValueError(
            "losses must have shape (records, models), with nonnegative values"
        )
    temperature = _temperature(temperature)
    return update_weights(np.ones(matrix.shape[1]), matrix.sum(axis=0), temperature)


def pairwise_trajectory_mse(bank: object, valid: object) -> FloatArray:
    """Pairwise mean squared Euclidean trajectory disagreement.

    ``bank`` has shape ``(models, time, points, 3)`` and ``valid`` has shape
    ``(time, points)``.  The metric matches the point-wise squared-error scale
    used by the spring-pilot likelihood.
    """
    values = _finite_array(bank, name="bank", ndim=4)
    if values.shape[0] < 2 or values.shape[-1] != 3:
        raise ValueError("bank must have shape (models>=2, time, points, 3)")
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != values.shape[1:3] or not np.any(mask):
        raise ValueError("valid must select at least one time-point entry")
    selected = values[:, mask, :]
    delta = selected[:, None, :, :] - selected[None, :, :, :]
    distance = np.mean(np.sum(delta * delta, axis=-1), axis=-1)
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    if np.any(distance < -1e-15):
        raise ValueError("pairwise disagreement must be nonnegative")
    distance = np.maximum(distance, 0.0)
    distance.setflags(write=False)
    return distance


def validate_distance_matrix(value: object, *, models: int, name: str) -> FloatArray:
    matrix = _finite_array(value, name=name, ndim=2)
    if matrix.shape != (models, models):
        raise ValueError(f"{name} must have shape ({models}, {models})")
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    if not np.allclose(matrix, matrix.T, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12 * scale, rtol=0.0):
        raise ValueError(f"{name} must have a zero diagonal")
    if np.any(matrix < -1e-12 * scale):
        raise ValueError(f"{name} must be nonnegative")
    matrix = np.maximum(0.5 * (matrix + matrix.T), 0.0)
    matrix.setflags(write=False)
    return matrix


def entropy(weights: object) -> float:
    probabilities = normalize_weights(weights)
    return float(-np.sum(probabilities * np.log(probabilities)))


def model_spread(weights: object, distance: object) -> float:
    """Expected squared pairwise disagreement divided by two."""
    probabilities = normalize_weights(weights)
    matrix = validate_distance_matrix(
        distance, models=probabilities.size, name="distance"
    )
    return float(0.5 * probabilities @ matrix @ probabilities)


def pseudo_posteriors(
    weights: object, probe_distance: object, temperature: float
) -> FloatArray:
    """Posterior for each model-member pseudo-outcome under source noise.

    Row ``j`` is the posterior obtained if member ``j`` generated the noiseless
    probe mean.  The source-frozen temperature softens pairwise separation.
    """
    probabilities = normalize_weights(weights)
    distance = validate_distance_matrix(
        probe_distance, models=probabilities.size, name="probe_distance"
    )
    temperature = _temperature(temperature)
    rows = np.stack(
        [
            update_weights(probabilities, distance[index], temperature)
            for index in range(probabilities.size)
        ]
    )
    rows.setflags(write=False)
    return rows


def parameter_information_utility(
    weights: object, probe_distance: object, temperature: float
) -> float:
    """Expected entropy reduction over discrete physical-model members."""
    probabilities = normalize_weights(weights)
    posteriors = pseudo_posteriors(probabilities, probe_distance, temperature)
    expected = sum(
        float(probabilities[index]) * entropy(posteriors[index])
        for index in range(probabilities.size)
    )
    return float(max(entropy(probabilities) - expected, 0.0))


def task_variance_reduction_utility(
    weights: object,
    probe_distance: object,
    target_distance: object,
    temperature: float,
) -> float:
    """Expected fractional contraction of held-out-task model spread."""
    probabilities = normalize_weights(weights)
    target = validate_distance_matrix(
        target_distance, models=probabilities.size, name="target_distance"
    )
    current = model_spread(probabilities, target)
    if current <= np.finfo(np.float64).eps:
        return 0.0
    posteriors = pseudo_posteriors(probabilities, probe_distance, temperature)
    expected = sum(
        float(probabilities[index]) * model_spread(posteriors[index], target)
        for index in range(probabilities.size)
    )
    reduction = (current - expected) / current
    return float(np.clip(reduction, 0.0, 1.0))


def select_action(
    *,
    policy: str,
    weights: object,
    remaining_actions: Sequence[str],
    probe_distances: Mapping[str, object],
    target_distance: object,
    temperature: float,
    fixed_order: Sequence[str],
) -> tuple[str, dict[str, float | None]]:
    """Select one action without consuming an action outcome."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    remaining = tuple(str(action) for action in remaining_actions)
    if not remaining or len(remaining) != len(set(remaining)):
        raise ValueError("remaining_actions must be nonempty and unique")
    if set(probe_distances) != set(fixed_order):
        raise ValueError(
            "probe_distances and fixed_order must define the same action roster"
        )
    if not set(remaining).issubset(probe_distances):
        raise ValueError("remaining action is absent from probe_distances")
    probabilities = normalize_weights(weights)
    target = validate_distance_matrix(
        target_distance, models=probabilities.size, name="target_distance"
    )
    utilities: dict[str, float | None] = {}
    if policy == "fixed_order":
        order = tuple(str(action) for action in fixed_order)
        if len(order) != len(set(order)) or set(order) != set(probe_distances):
            raise ValueError("fixed_order must contain every action exactly once")
        selected = next(action for action in order if action in remaining)
        utilities.update({action: None for action in remaining})
        return selected, utilities
    for action in sorted(remaining):
        probe = validate_distance_matrix(
            probe_distances[action],
            models=probabilities.size,
            name=f"probe_distances[{action}]",
        )
        if policy == "parameter_information":
            utility = parameter_information_utility(probabilities, probe, temperature)
        else:
            utility = task_variance_reduction_utility(
                probabilities, probe, target, temperature
            )
        utilities[action] = utility
    maximum = max(float(value) for value in utilities.values() if value is not None)
    tolerance = 1e-14 * max(1.0, abs(maximum))
    selected = min(
        action
        for action, value in utilities.items()
        if value is not None and maximum - float(value) <= tolerance
    )
    return selected, utilities


@dataclass(frozen=True)
class PolicyState:
    budget: int
    selected_actions: tuple[str, ...]
    weights: FloatArray
    steps: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.budget, bool)
            or int(self.budget) != self.budget
            or self.budget < 0
        ):
            raise ValueError("budget must be a nonnegative integer")
        weights = normalize_weights(self.weights)
        actions = tuple(str(action) for action in self.selected_actions)
        if len(actions) != len(set(actions)) or len(actions) != self.budget:
            raise ValueError("selected_actions must be unique and match budget")
        object.__setattr__(self, "budget", int(self.budget))
        object.__setattr__(self, "selected_actions", actions)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "steps", tuple(dict(step) for step in self.steps))


def simulate_policy(
    *,
    policy: str,
    initial_weights: object,
    probe_distances: Mapping[str, object],
    target_distance: object,
    observed_losses: Mapping[str, object],
    temperature: float,
    fixed_order: Sequence[str],
    budgets: Sequence[int],
) -> dict[int, PolicyState]:
    """Replay a sequential policy and consume only selected probe outcomes."""
    requested = tuple(int(value) for value in budgets)
    action_count = len(probe_distances)
    if (
        not requested
        or tuple(sorted(set(requested))) != requested
        or requested[0] != 0
        or requested[-1] > action_count
    ):
        raise ValueError("budgets must be sorted unique values starting at zero")
    if set(observed_losses) != set(probe_distances):
        raise ValueError("observed_losses must expose the complete registered roster")
    weights = normalize_weights(initial_weights)
    states: dict[int, PolicyState] = {0: PolicyState(0, (), weights, ())}
    selected: list[str] = []
    steps: list[dict[str, Any]] = []
    remaining = set(probe_distances)
    for step_index in range(1, requested[-1] + 1):
        action, utilities = select_action(
            policy=policy,
            weights=weights,
            remaining_actions=tuple(sorted(remaining)),
            probe_distances=probe_distances,
            target_distance=target_distance,
            temperature=temperature,
            fixed_order=fixed_order,
        )
        entropy_before = entropy(weights)
        target_before = model_spread(weights, target_distance)
        # This is the only point at which the selected action's outcome is read.
        selected_loss = observed_losses[action]
        weights = update_weights(weights, selected_loss, temperature)
        selected.append(action)
        remaining.remove(action)
        steps.append(
            {
                "step": step_index,
                "selected_action": action,
                "utilities": utilities,
                "entropy_before": entropy_before,
                "entropy_after": entropy(weights),
                "target_model_spread_before": target_before,
                "target_model_spread_after": model_spread(weights, target_distance),
            }
        )
        if step_index in requested:
            states[step_index] = PolicyState(
                step_index, tuple(selected), weights, tuple(steps)
            )
    return states


__all__ = [
    "POLICIES",
    "PolicyState",
    "entropy",
    "model_spread",
    "normalize_weights",
    "pairwise_trajectory_mse",
    "parameter_information_utility",
    "pseudo_posteriors",
    "select_action",
    "simulate_policy",
    "task_variance_reduction_utility",
    "update_weights",
    "validate_distance_matrix",
    "weights_from_records",
]
