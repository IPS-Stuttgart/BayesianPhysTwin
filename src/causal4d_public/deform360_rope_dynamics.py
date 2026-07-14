"""Shared acceleration-normalized rope dynamics for the Deform360 pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


PARAMETER_NAMES = (
    "spring_acceleration_per_m_s2",
    "edge_damping_per_s",
    "bending_acceleration_per_m_s2",
    "bending_damping_per_s",
    "contact_acceleration_per_m_s2",
    "contact_damping_per_s",
    "drag_per_s",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class SharedRopeDynamicsParameters:
    """Mass-normalized coefficients identifiable from kinematic trajectories."""

    spring_acceleration_per_m_s2: float
    edge_damping_per_s: float
    bending_acceleration_per_m_s2: float
    bending_damping_per_s: float
    contact_acceleration_per_m_s2: float
    contact_damping_per_s: float
    drag_per_s: float

    def __post_init__(self) -> None:
        values = self.as_array()
        _require(np.all(np.isfinite(values)), "rope dynamics parameters are non-finite")
        _require(np.all(values >= 0.0), "rope dynamics parameters must be nonnegative")

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.spring_acceleration_per_m_s2,
                self.edge_damping_per_s,
                self.bending_acceleration_per_m_s2,
                self.bending_damping_per_s,
                self.contact_acceleration_per_m_s2,
                self.contact_damping_per_s,
                self.drag_per_s,
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_array(cls, values: np.ndarray) -> SharedRopeDynamicsParameters:
        array = np.asarray(values, dtype=np.float64)
        _require(array.shape == (len(PARAMETER_NAMES),), "invalid parameter vector")
        return cls(*map(float, array))


@dataclass(frozen=True)
class RopeDynamicsObservation:
    """One ordered source trajectory with registered kinematic contacts."""

    episode_id: str
    positions_m: np.ndarray
    controller_positions_m: np.ndarray
    contact_active: np.ndarray
    contact_node_indices: tuple[int, ...]
    contact_offsets_m: np.ndarray
    dt_seconds: float

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float64)
        controllers = np.asarray(self.controller_positions_m, dtype=np.float64)
        active = np.asarray(self.contact_active, dtype=bool)
        offsets = np.asarray(self.contact_offsets_m, dtype=np.float64)
        _require(bool(self.episode_id), "episode id must be nonempty")
        _require(
            positions.ndim == 3 and positions.shape[2] == 3 and len(positions) >= 4,
            "positions must have shape (T,N,3) with at least four frames",
        )
        _require(
            controllers.ndim == 3
            and controllers.shape[0] == positions.shape[0]
            and controllers.shape[2] == 3,
            "controller positions must have shape (T,C,3)",
        )
        _require(
            active.shape == controllers.shape[:2],
            "contact activity must have shape (T,C)",
        )
        _require(
            len(self.contact_node_indices) == controllers.shape[1],
            "one contact node is required per controller",
        )
        _require(
            offsets.shape == (controllers.shape[1], 3),
            "contact offsets must have shape (C,3)",
        )
        _require(
            all(0 <= node < positions.shape[1] for node in self.contact_node_indices),
            "contact node index is outside the rope graph",
        )
        _require(
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(controllers))
            and np.all(np.isfinite(offsets)),
            "observation trajectories contain non-finite values",
        )
        _require(self.dt_seconds > 0.0, "frame interval must be positive")
        positions = positions.copy()
        controllers = controllers.copy()
        active = active.copy()
        offsets = offsets.copy()
        positions.setflags(write=False)
        controllers.setflags(write=False)
        active.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "controller_positions_m", controllers)
        object.__setattr__(self, "contact_active", active)
        object.__setattr__(self, "contact_offsets_m", offsets)


@dataclass(frozen=True)
class RopeDynamicsFit:
    parameters: SharedRopeDynamicsParameters
    parameter_covariance: np.ndarray
    residual_acceleration_rmse_m_s2: float
    design_row_count: int
    episode_diagnostics: tuple[dict[str, Any], ...]
    ridge_strength: float

    def __post_init__(self) -> None:
        covariance = np.asarray(self.parameter_covariance, dtype=np.float64)
        _require(
            covariance.shape == (len(PARAMETER_NAMES), len(PARAMETER_NAMES)),
            "invalid rope parameter covariance shape",
        )
        _require(np.all(np.isfinite(covariance)), "parameter covariance is non-finite")
        covariance = covariance.copy()
        covariance.setflags(write=False)
        object.__setattr__(self, "parameter_covariance", covariance)


def _chain_force_bases(
    positions: np.ndarray,
    velocities: np.ndarray,
    rest_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = len(positions)
    _require(rest_lengths.shape == (node_count - 1,), "rest-length count mismatch")
    spring = np.zeros_like(positions)
    damping = np.zeros_like(positions)
    difference = positions[1:] - positions[:-1]
    length = np.linalg.norm(difference, axis=1)
    direction = difference / np.maximum(length[:, None], 1e-8)
    edge_force = (length - rest_lengths)[:, None] * direction
    relative_velocity = velocities[1:] - velocities[:-1]
    spring[:-1] += edge_force
    spring[1:] -= edge_force
    damping[:-1] += relative_velocity
    damping[1:] -= relative_velocity
    return spring, damping


def _contact_force_bases(
    positions: np.ndarray,
    velocities: np.ndarray,
    controller_positions: np.ndarray,
    controller_velocities: np.ndarray,
    contact_active: np.ndarray,
    contact_node_indices: tuple[int, ...],
    contact_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spring = np.zeros_like(positions)
    damping = np.zeros_like(positions)
    for controller, node in enumerate(contact_node_indices):
        if contact_active[controller]:
            spring[node] += (
                controller_positions[controller]
                + contact_offsets[controller]
                - positions[node]
            )
            damping[node] += controller_velocities[controller] - velocities[node]
    return spring, damping


def _bending_force_bases(
    positions: np.ndarray, velocities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    node_count = len(positions)
    second_difference = np.zeros((node_count - 2, node_count), dtype=np.float64)
    for row in range(node_count - 2):
        second_difference[row, row : row + 3] = (1.0, -2.0, 1.0)
    biharmonic = second_difference.T @ second_difference
    return -biharmonic @ positions, -biharmonic @ velocities


def rope_acceleration_bases(
    positions_m: np.ndarray,
    velocities_m_s: np.ndarray,
    controller_positions_m: np.ndarray,
    controller_velocities_m_s: np.ndarray,
    contact_active: np.ndarray,
    contact_node_indices: tuple[int, ...],
    contact_offsets_m: np.ndarray,
    rest_lengths_m: np.ndarray,
) -> np.ndarray:
    """Return per-parameter acceleration bases with shape ``(7,N,3)``."""

    positions = np.asarray(positions_m, dtype=np.float64)
    velocities = np.asarray(velocities_m_s, dtype=np.float64)
    controllers = np.asarray(controller_positions_m, dtype=np.float64)
    controller_velocity = np.asarray(controller_velocities_m_s, dtype=np.float64)
    active = np.asarray(contact_active, dtype=bool)
    offsets = np.asarray(contact_offsets_m, dtype=np.float64)
    _require(
        positions.ndim == 2 and positions.shape[1] == 3,
        "positions must have shape (N,3)",
    )
    _require(velocities.shape == positions.shape, "velocity shape mismatch")
    _require(
        controllers.shape == controller_velocity.shape
        and controllers.ndim == 2
        and controllers.shape[1] == 3,
        "controller state must have shape (C,3)",
    )
    _require(active.shape == (len(controllers),), "contact activity shape mismatch")
    _require(
        len(contact_node_indices) == len(controllers),
        "contact node/controller count mismatch",
    )
    _require(offsets.shape == controllers.shape, "contact offset shape mismatch")
    spring, edge_damping = _chain_force_bases(
        positions, velocities, np.asarray(rest_lengths_m, dtype=np.float64)
    )
    bending, bending_damping = _bending_force_bases(positions, velocities)
    contact, contact_damping = _contact_force_bases(
        positions,
        velocities,
        controllers,
        controller_velocity,
        active,
        contact_node_indices,
        offsets,
    )
    return np.stack(
        (
            spring,
            edge_damping,
            bending,
            bending_damping,
            contact,
            contact_damping,
            -velocities,
        )
    )


def _observation_design(
    observation: RopeDynamicsObservation,
    rest_lengths_m: np.ndarray,
    gravity_m_s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    positions = observation.positions_m
    controllers = observation.controller_positions_m
    dt = observation.dt_seconds
    velocities = np.gradient(positions, dt, axis=0, edge_order=2)
    controller_velocities = np.gradient(controllers, dt, axis=0, edge_order=2)
    accelerations = np.gradient(velocities, dt, axis=0, edge_order=2)
    design = []
    target = []
    for frame in range(1, len(positions) - 1):
        bases = rope_acceleration_bases(
            positions[frame],
            velocities[frame],
            controllers[frame],
            controller_velocities[frame],
            observation.contact_active[frame],
            observation.contact_node_indices,
            observation.contact_offsets_m,
            rest_lengths_m,
        )
        design.append(np.moveaxis(bases, 0, -1).reshape(-1, len(PARAMETER_NAMES)))
        target.append((accelerations[frame] - gravity_m_s2).reshape(-1))
    matrix = np.concatenate(design, axis=0)
    values = np.concatenate(target, axis=0)
    return (
        matrix,
        values,
        {
            "episode_id": observation.episode_id,
            "frame_count": len(positions),
            "node_count": positions.shape[1],
            "controller_count": controllers.shape[1],
            "design_row_count": len(values),
            "active_contact_fraction": float(np.mean(observation.contact_active)),
        },
    )


def fit_shared_rope_dynamics(
    observations: Sequence[RopeDynamicsObservation],
    rest_lengths_m: np.ndarray,
    *,
    gravity_m_s2: np.ndarray = np.asarray((0.0, 0.0, -9.81)),
    ridge_strength: float = 1e-4,
) -> RopeDynamicsFit:
    """Fit one nonnegative shared parameter vector from source episodes only."""

    _require(bool(observations), "at least one source observation is required")
    _require(ridge_strength >= 0.0, "ridge strength must be nonnegative")
    rest = np.asarray(rest_lengths_m, dtype=np.float64)
    gravity = np.asarray(gravity_m_s2, dtype=np.float64)
    _require(rest.ndim == 1 and np.all(rest > 0.0), "rest lengths must be positive")
    _require(gravity.shape == (3,) and np.all(np.isfinite(gravity)), "invalid gravity")
    node_counts = {observation.positions_m.shape[1] for observation in observations}
    _require(node_counts == {len(rest) + 1}, "source rope node counts disagree")
    designs = []
    targets = []
    diagnostics = []
    for observation in observations:
        matrix, values, summary = _observation_design(observation, rest, gravity)
        designs.append(matrix)
        targets.append(values)
        diagnostics.append(summary)
    design = np.concatenate(designs, axis=0)
    target = np.concatenate(targets, axis=0)
    scale = np.sqrt(np.mean(design**2, axis=0))
    _require(
        np.all(scale > 1e-12),
        "source panel does not excite every rope dynamics parameter",
    )
    standardized = design / scale
    augmented_design = np.vstack(
        (standardized, np.sqrt(ridge_strength) * np.eye(len(PARAMETER_NAMES)))
    )
    augmented_target = np.concatenate((target, np.zeros(len(PARAMETER_NAMES))))
    try:
        from scipy.optimize import lsq_linear
    except ImportError as error:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("SciPy is required for shared rope fitting") from error
    result = lsq_linear(augmented_design, augmented_target, bounds=(0.0, np.inf))
    _require(result.success, f"shared rope fit failed: {result.message}")
    parameters = result.x / scale
    residual = target - design @ parameters
    degrees = max(len(target) - len(PARAMETER_NAMES), 1)
    variance = float(residual @ residual / degrees)
    standardized_covariance = variance * np.linalg.pinv(
        standardized.T @ standardized + ridge_strength * np.eye(len(PARAMETER_NAMES))
    )
    covariance = standardized_covariance / scale[:, None] / scale[None, :]
    offset = 0
    for summary, values in zip(diagnostics, targets, strict=True):
        count = len(values)
        episode_residual = residual[offset : offset + count]
        summary["residual_acceleration_rmse_m_s2"] = float(
            np.sqrt(np.mean(episode_residual**2))
        )
        offset += count
    return RopeDynamicsFit(
        parameters=SharedRopeDynamicsParameters.from_array(parameters),
        parameter_covariance=covariance,
        residual_acceleration_rmse_m_s2=float(np.sqrt(np.mean(residual**2))),
        design_row_count=len(target),
        episode_diagnostics=tuple(diagnostics),
        ridge_strength=float(ridge_strength),
    )


def _project_chain_rest_lengths(
    positions: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    iterations: int,
    fixed_nodes: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64).copy()
    _require(iterations >= 0, "constraint iterations must be nonnegative")
    fixed = (
        np.zeros(len(values), dtype=bool)
        if fixed_nodes is None
        else np.asarray(fixed_nodes, dtype=bool)
    )
    _require(fixed.shape == (len(values),), "fixed-node mask shape mismatch")
    if iterations == 0:
        return values
    for _ in range(iterations):
        for edges in (range(len(rest_lengths)), range(len(rest_lengths) - 1, -1, -1)):
            for edge in edges:
                difference = values[edge + 1] - values[edge]
                length = float(np.linalg.norm(difference))
                if length <= 1e-12:
                    continue
                left_weight = float(not fixed[edge])
                right_weight = float(not fixed[edge + 1])
                weight = left_weight + right_weight
                if weight == 0.0:
                    continue
                correction = (length - rest_lengths[edge]) * difference / length
                values[edge] += left_weight / weight * correction
                values[edge + 1] -= right_weight / weight * correction
    return values


def rollout_rope_dynamics(
    initial_positions_m: np.ndarray,
    initial_velocities_m_s: np.ndarray,
    controller_positions_m: np.ndarray,
    contact_active: np.ndarray,
    contact_node_indices: tuple[int, ...],
    contact_offsets_m: np.ndarray,
    rest_lengths_m: np.ndarray,
    parameters: SharedRopeDynamicsParameters,
    *,
    dt_seconds: float,
    gravity_m_s2: np.ndarray = np.asarray((0.0, 0.0, -9.81)),
    substeps: int = 4,
    constraint_iterations: int = 0,
    kinematic_contacts: bool = False,
) -> np.ndarray:
    """Roll out an ordered rope graph under a fixed realized-contact schedule.

    ``kinematic_contacts`` models an active prehensile grasp as a positional
    constraint. The default retains the fitted spring-contact dynamics.
    """

    positions = np.asarray(initial_positions_m, dtype=np.float64).copy()
    velocities = np.asarray(initial_velocities_m_s, dtype=np.float64).copy()
    controllers = np.asarray(controller_positions_m, dtype=np.float64)
    active = np.asarray(contact_active, dtype=bool)
    offsets = np.asarray(contact_offsets_m, dtype=np.float64)
    rest = np.asarray(rest_lengths_m, dtype=np.float64)
    gravity = np.asarray(gravity_m_s2, dtype=np.float64)
    _require(
        positions.ndim == 2 and positions.shape[1] == 3,
        "initial positions must have shape (N,3)",
    )
    _require(velocities.shape == positions.shape, "initial velocity shape mismatch")
    _require(
        controllers.ndim == 3 and controllers.shape[2] == 3 and len(controllers) >= 2,
        "controller positions must have shape (T,C,3)",
    )
    _require(active.shape == controllers.shape[:2], "contact schedule shape mismatch")
    _require(
        len(contact_node_indices) == controllers.shape[1],
        "contact node/controller count mismatch",
    )
    _require(
        offsets.shape == (controllers.shape[1], 3),
        "contact offsets must have shape (C,3)",
    )
    _require(rest.shape == (len(positions) - 1,), "rest-length count mismatch")
    _require(dt_seconds > 0.0 and substeps >= 1, "invalid rollout time step")
    _require(constraint_iterations >= 0, "constraint iterations must be nonnegative")
    _require(gravity.shape == (3,), "gravity must have shape (3,)")
    controller_velocities = np.gradient(controllers, dt_seconds, axis=0, edge_order=1)
    trajectory = np.empty((len(controllers), len(positions), 3), dtype=np.float64)
    trajectory[0] = positions
    parameter_values = parameters.as_array()
    step = dt_seconds / substeps
    for frame in range(1, len(controllers)):
        for substep in range(substeps):
            previous_positions = positions.copy()
            fraction = (substep + 1) / substeps
            controller = (1.0 - fraction) * controllers[
                frame - 1
            ] + fraction * controllers[frame]
            controller_velocity = (1.0 - fraction) * controller_velocities[
                frame - 1
            ] + fraction * controller_velocities[frame]
            bases = rope_acceleration_bases(
                positions,
                velocities,
                controller,
                controller_velocity,
                active[frame],
                contact_node_indices,
                offsets,
                rest,
            )
            acceleration = gravity + np.tensordot(parameter_values, bases, axes=(0, 0))
            velocities += step * acceleration
            positions += step * velocities
            fixed_nodes = np.zeros(len(positions), dtype=bool)
            if kinematic_contacts:
                for contact_index, node_index in enumerate(contact_node_indices):
                    if active[frame, contact_index]:
                        positions[node_index] = (
                            controller[contact_index] + offsets[contact_index]
                        )
                        fixed_nodes[node_index] = True
            if constraint_iterations:
                positions = _project_chain_rest_lengths(
                    positions,
                    rest,
                    iterations=constraint_iterations,
                    fixed_nodes=fixed_nodes,
                )
                velocities = (positions - previous_positions) / step
        trajectory[frame] = positions
    _require(np.all(np.isfinite(trajectory)), "rope rollout produced non-finite states")
    return trajectory


__all__ = [
    "PARAMETER_NAMES",
    "RopeDynamicsFit",
    "RopeDynamicsObservation",
    "SharedRopeDynamicsParameters",
    "fit_shared_rope_dynamics",
    "rollout_rope_dynamics",
    "rope_acceleration_bases",
]
