"""Transparent kinematic and contact-aware cloth predictors."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import InputView

KINEMATIC_ARMS = ("persistence", "constant_velocity", "last_residual")
PHYSICS_ARM = "bayesian_contact_physics"
AUXILIARY_ARMS = ("nominal_contact_physics", "map_contact_physics")
ALL_ARMS = KINEMATIC_ARMS + (PHYSICS_ARM,) + AUXILIARY_ARMS


@dataclass(frozen=True)
class PhysicsFit:
    parameters: tuple[tuple[float, float, float], ...]
    weights: np.ndarray
    losses_m2: np.ndarray
    temperature_m2: float

    def record(self) -> dict[str, Any]:
        return {
            "parameters": [list(item) for item in self.parameters],
            "weights": self.weights.tolist(),
            "losses_m2": self.losses_m2.tolist(),
            "temperature_m2": self.temperature_m2,
        }

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> PhysicsFit:
        parameters = tuple(tuple(float(x) for x in row) for row in value["parameters"])
        weights = np.asarray(value["weights"], dtype=float)
        losses = np.asarray(value["losses_m2"], dtype=float)
        temperature = float(value["temperature_m2"])
        if len(parameters) != len(weights) or weights.shape != losses.shape:
            raise ValueError("physics fit dimensions changed")
        valid_weights = (
            np.isfinite(weights).all()
            and not np.any(weights < 0)
            and np.isclose(weights.sum(), 1)
        )
        if not valid_weights:
            raise ValueError("invalid physics weights")
        return cls(parameters, weights, losses, temperature)


def parameter_bank(protocol: dict[str, Any]) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in itertools.product(
            protocol["stiffness_per_mass"],
            protocol["damping_per_mass"],
            protocol["self_collision_stiffness_per_mass"],
        )
    )


def grid_edges() -> tuple[np.ndarray, np.ndarray]:
    links: list[tuple[int, int]] = []
    weights: list[float] = []
    rows, columns = 5, 4
    for row in range(rows):
        for column in range(columns):
            for delta_row, delta_column, weight in (
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 0.5),
                (1, -1, 0.5),
                (0, 2, 0.1),
                (2, 0, 0.1),
            ):
                other_row = row + delta_row
                other_column = column + delta_column
                if 0 <= other_row < rows and 0 <= other_column < columns:
                    links.append(
                        (row * columns + column, other_row * columns + other_column)
                    )
                    weights.append(weight)
    return np.asarray(links, dtype=int), np.asarray(weights, dtype=float)


def self_collision_pairs() -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for left in range(20):
        row_left, column_left = divmod(left, 4)
        for right in range(left + 1, 20):
            row_right, column_right = divmod(right, 4)
            # Adjacent and next-nearest structural neighbours should be governed
            # by springs rather than the coarse collision repulsion.
            if abs(row_left - row_right) <= 2 and abs(column_left - column_right) <= 2:
                continue
            pairs.append((left, right))
    return np.asarray(pairs, dtype=int)


def _linear_velocity(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        raise ValueError("velocity fit needs at least two frames")
    centered = times - times.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        raise ValueError("degenerate velocity timestamps")
    return np.einsum("t,tnd->nd", centered, positions) / denominator


def _window(times: np.ndarray, seconds: float) -> slice:
    threshold = times[-1] - seconds
    first = int(np.searchsorted(times, threshold, side="left"))
    first = min(first, len(times) - 2)
    return slice(first, len(times))


def kinematic_predictions(
    inputs: InputView, protocol: dict[str, Any]
) -> dict[str, np.ndarray]:
    shape = (len(inputs.times), 20, 3)
    predictions = {name: np.empty(shape, dtype=float) for name in KINEMATIC_ARMS}
    for prediction in predictions.values():
        prediction[: inputs.cutoff + 1] = inputs.cloth_prefix

    start = inputs.cloth_prefix[-1]
    prefix_times = inputs.times[: inputs.cutoff + 1]
    short_slice = _window(
        prefix_times, float(protocol["short_velocity_window_seconds"])
    )
    long_slice = _window(prefix_times, float(protocol["long_velocity_window_seconds"]))
    short_velocity = _linear_velocity(
        prefix_times[short_slice], inputs.cloth_prefix[short_slice]
    )
    long_velocity = _linear_velocity(
        prefix_times[long_slice], inputs.cloth_prefix[long_slice]
    )
    residual_velocity = short_velocity - long_velocity
    decay = float(protocol["residual_velocity_decay_seconds"])

    for index in range(inputs.cutoff + 1, len(inputs.times)):
        horizon = float(inputs.times[index] - inputs.times[inputs.cutoff])
        predictions["persistence"][index] = start
        predictions["constant_velocity"][index] = start + horizon * short_velocity
        transient = decay * (1.0 - np.exp(-horizon / decay))
        predictions["last_residual"][index] = (
            start + horizon * long_velocity + transient * residual_velocity
        )
    return predictions


def rod_forecast(inputs: InputView, protocol: dict[str, Any]) -> np.ndarray:
    """Hold the recorded static rod pose fixed after the causal prefix.

    The dataset's self-collision protocol uses a static metallic rod.  A
    robust prefix median suppresses sub-millimetre marker jitter without
    accessing any future rod coordinate.
    """

    del protocol
    rod = inputs.rod_prefix.copy()
    reference = np.median(rod, axis=0)
    output = np.empty((len(inputs.times), 2, 3), dtype=float)
    output[: inputs.cutoff + 1] = rod
    output[inputs.cutoff + 1 :] = reference
    return output


def _closest_on_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    start, end = segments
    axis = end - start
    denominator = float(np.dot(axis, axis))
    if denominator <= 1e-10:
        raise ValueError("degenerate predicted rod segment")
    fraction = ((points - start) @ axis) / denominator
    fraction = np.clip(fraction, 0.0, 1.0)
    return start + fraction[:, None] * axis


def contact_rollout(
    inputs: InputView,
    parameters: tuple[float, float, float],
    protocol: dict[str, Any],
) -> np.ndarray:
    """Run the reduced contact-aware spring model from the causal prefix end."""

    stiffness, damping, collision_stiffness = parameters
    links, relative_stiffness = grid_edges()
    left, right = links.T
    initial = inputs.cloth_prefix[0]
    rest_lengths = np.linalg.norm(initial[right] - initial[left], axis=1)
    if np.min(rest_lengths) <= 1e-6:
        raise ValueError("degenerate cloth spring")
    pairs = self_collision_pairs()
    pair_left, pair_right = pairs.T
    predicted_rod = rod_forecast(inputs, protocol)
    output = np.empty((len(inputs.times), 20, 3), dtype=float)
    output[: inputs.cutoff + 1] = inputs.cloth_prefix
    x = inputs.cloth_prefix[-1].copy()
    prefix_times = inputs.times[: inputs.cutoff + 1]
    velocity_slice = _window(
        prefix_times, float(protocol["short_velocity_window_seconds"])
    )
    velocity = _linear_velocity(
        prefix_times[velocity_slice], inputs.cloth_prefix[velocity_slice]
    )
    substeps = int(protocol["integration_substeps"])
    gravity = float(protocol["gravity_m_s2"])
    contact_radius = float(protocol["contact_radius_m"])
    collision_distance = float(protocol["self_collision_distance_m"])
    friction_rate = float(protocol["rod_friction_rate"])
    origin = initial.mean(axis=0)

    for index in range(inputs.cutoff + 1, len(inputs.times)):
        full_dt = float(inputs.times[index] - inputs.times[index - 1])
        dt = full_dt / substeps
        rod_previous = predicted_rod[index - 1]
        rod_current = predicted_rod[index]
        for substep in range(1, substeps + 1):
            fraction = substep / substeps
            rod = (1.0 - fraction) * rod_previous + fraction * rod_current
            delta = x[right] - x[left]
            length = np.linalg.norm(delta, axis=1)
            force = (
                stiffness
                * relative_stiffness
                * (length - rest_lengths)
                / np.maximum(length, 1e-9)
            )[:, None] * delta
            acceleration = -damping * velocity
            acceleration[:, 2] -= gravity
            np.add.at(acceleration, left, force)
            np.add.at(acceleration, right, -force)

            if collision_stiffness > 0 and len(pairs):
                pair_delta = x[pair_right] - x[pair_left]
                pair_length = np.linalg.norm(pair_delta, axis=1)
                active = pair_length < collision_distance
                if np.any(active):
                    normal = pair_delta[active] / np.maximum(
                        pair_length[active, None], 1e-9
                    )
                    repulsion = (
                        collision_stiffness
                        * (collision_distance - pair_length[active])[:, None]
                        * normal
                    )
                    np.add.at(acceleration, pair_left[active], -repulsion)
                    np.add.at(acceleration, pair_right[active], repulsion)

            velocity += dt * acceleration
            x += dt * velocity

            closest = _closest_on_segments(x, rod)
            separation = x - closest
            distance = np.linalg.norm(separation, axis=1)
            active = distance < contact_radius
            if np.any(active):
                normal = separation[active] / np.maximum(distance[active, None], 1e-9)
                x[active] += (contact_radius - distance[active])[:, None] * normal
                normal_velocity = np.sum(velocity[active] * normal, axis=1)
                inward = normal_velocity < 0
                active_indices = np.flatnonzero(active)
                if np.any(inward):
                    velocity[active_indices[inward]] -= (
                        normal_velocity[inward, None] * normal[inward]
                    )
                projected_normal = (
                    np.sum(velocity[active] * normal, axis=1)[:, None] * normal
                )
                tangential = velocity[active] - projected_normal
                velocity[active] = (
                    projected_normal + np.exp(-friction_rate * dt) * tangential
                )

        if not np.isfinite(x).all() or not np.isfinite(velocity).all():
            raise ValueError("nonfinite contact rollout")
        if np.max(np.linalg.norm(x - origin, axis=1)) > 5.0:
            raise ValueError("contact rollout escaped the registered domain")
        output[index] = x
    return output


def valid_future(inputs: InputView, truth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(truth).all(axis=2)
    valid[: inputs.cutoff + 1] = False
    if not np.any(valid):
        raise ValueError("no valid future cloth outcomes")
    return valid


def trajectory_mse(
    prediction: np.ndarray, truth: np.ndarray, inputs: InputView
) -> float:
    valid = valid_future(inputs, truth)
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite prediction")
    return float(np.mean(np.sum((prediction[valid] - truth[valid]) ** 2, axis=1)))


def fit_physics(
    inputs: InputView,
    truth: np.ndarray,
    protocol: dict[str, Any],
) -> PhysicsFit:
    parameters = parameter_bank(protocol)
    predictions = [contact_rollout(inputs, item, protocol) for item in parameters]
    losses = np.asarray(
        [trajectory_mse(prediction, truth, inputs) for prediction in predictions],
        dtype=float,
    )
    temperature = max(
        float(np.min(losses)),
        float(protocol["measurement_floor_m"]) ** 2,
    )
    logits = -losses / (2.0 * temperature)
    weights = np.exp(logits - np.max(logits))
    weights /= weights.sum()
    return PhysicsFit(parameters, weights, losses, temperature)


def all_predictions(
    inputs: InputView,
    fit: PhysicsFit,
    protocol: dict[str, Any],
) -> dict[str, np.ndarray]:
    predictions = kinematic_predictions(inputs, protocol)
    bank = np.stack(
        [contact_rollout(inputs, item, protocol) for item in fit.parameters], axis=0
    )
    nominal = tuple(float(value) for value in protocol["nominal_parameters"])
    try:
        nominal_index = fit.parameters.index(nominal)
    except ValueError as error:
        raise ValueError(
            "nominal contact parameters are absent from the bank"
        ) from error
    predictions["nominal_contact_physics"] = bank[nominal_index]
    predictions["map_contact_physics"] = bank[int(np.argmax(fit.weights))]
    predictions[PHYSICS_ARM] = np.einsum("k,ktnd->tnd", fit.weights, bank)
    return predictions
