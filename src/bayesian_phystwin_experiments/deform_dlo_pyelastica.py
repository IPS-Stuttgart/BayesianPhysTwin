"""Pinned PyElastica portability adapter for the DEFORM DLO protocol."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

Array = np.ndarray[Any, Any]
DEFORM_FRAME_DT_S = 0.01
DEFORM_CLAMPED_NODES = (0, 1, 10, 11)


@dataclass(frozen=True)
class PyElasticaParameters:
    """One member of the preregistered finite portability bank."""

    youngs_modulus_pa: float
    density_kg_m3: float
    damping_constant: float
    integration_substeps: int

    def to_record(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def deform_pyelastica_parameter_bank(
    protocol: Mapping[str, object],
) -> tuple[PyElasticaParameters, ...]:
    """Expand the exact 3x2x2x3 bank in stable nested-loop order."""

    backend = protocol.get("backend_portability")
    if not isinstance(backend, Mapping):
        raise ValueError("DEFORM backend portability contract is missing")
    raw_bank = backend.get("parameter_bank")
    if not isinstance(raw_bank, Mapping):
        raise ValueError("DEFORM backend parameter bank is missing")
    youngs = tuple(
        float(value) for value in cast(list[Any], raw_bank.get("youngs_modulus_pa"))
    )
    densities = tuple(
        float(value) for value in cast(list[Any], raw_bank.get("density_kg_m3"))
    )
    damping = tuple(
        float(value) for value in cast(list[Any], raw_bank.get("damping_constant"))
    )
    substeps = tuple(
        int(value) for value in cast(list[Any], raw_bank.get("integration_substeps"))
    )
    if (
        youngs != (1e5, 1e6, 1e7)
        or densities != (900.0, 1200.0)
        or damping != (0.1, 1.0)
        or substeps != (2, 4, 8)
    ):
        raise ValueError("DEFORM backend parameter bank differs")
    return tuple(
        PyElasticaParameters(modulus, density, damper, substep)
        for modulus in youngs
        for density in densities
        for damper in damping
        for substep in substeps
    )


def deform_pyelastica_directors(positions: Array) -> Array:
    """Construct a continuous orthonormal material frame along one open rod."""

    nodes = np.asarray(positions, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 4:
        raise ValueError("PyElastica rod positions must have shape (V, 3)")
    edges = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if not np.isfinite(nodes).all() or np.any(lengths <= 1e-8):
        raise ValueError("PyElastica rod geometry is degenerate")
    tangents = edges / lengths[:, None]
    directors = np.empty((3, 3, edges.shape[0]), dtype=np.float64)
    previous: Array | None = None
    axes = np.eye(3, dtype=np.float64)
    for index, tangent in enumerate(tangents):
        if previous is None:
            axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
            normal = axis - np.dot(axis, tangent) * tangent
        else:
            normal = previous - np.dot(previous, tangent) * tangent
            if np.linalg.norm(normal) <= 1e-8:
                axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
                normal = axis - np.dot(axis, tangent) * tangent
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal)
        directors[0, :, index] = normal
        directors[1, :, index] = binormal
        directors[2, :, index] = tangent
        previous = normal
    return cast(Array, np.asarray(directors, dtype=np.float64))


def deform_pyelastica_kinematic_sample(
    endpoint_series: Array,
    time_s: float,
    *,
    frame_dt_s: float = DEFORM_FRAME_DT_S,
) -> tuple[Array, Array]:
    """Linearly interpolate registered clamp positions and their causal rate."""

    series = np.asarray(endpoint_series, dtype=np.float64)
    if (
        series.ndim != 3
        or series.shape[0] < 2
        or series.shape[2] != 3
        or not np.isfinite(series).all()
        or not math.isfinite(time_s)
        or not math.isfinite(frame_dt_s)
        or frame_dt_s <= 0.0
    ):
        raise ValueError("PyElastica endpoint series is invalid")
    coordinate = float(np.clip(time_s / frame_dt_s, 0.0, series.shape[0] - 1))
    lower = min(int(math.floor(coordinate)), series.shape[0] - 2)
    alpha = coordinate - lower
    position = (1.0 - alpha) * series[lower] + alpha * series[lower + 1]
    velocity = (series[lower + 1] - series[lower]) / frame_dt_s
    return position, velocity


def simulate_deform_pyelastica(
    trajectory: Array,
    parameters: PyElasticaParameters,
    *,
    elastica: Any,
    frame_dt_s: float = DEFORM_FRAME_DT_S,
    poisson_ratio: float = 0.45,
    radius_to_mean_edge_ratio: float = 0.05,
) -> Array:
    """Roll one two-frame-conditioned trajectory through PyElastica."""

    observed = np.asarray(trajectory, dtype=np.float64)
    if (
        observed.ndim != 3
        or observed.shape[0] < 3
        or observed.shape[1:] != (12, 3)
        or not np.isfinite(observed).all()
    ):
        raise ValueError("PyElastica DEFORM trajectory must have shape (T, 12, 3)")
    if (
        parameters.youngs_modulus_pa <= 0.0
        or parameters.density_kg_m3 <= 0.0
        or parameters.damping_constant < 0.0
        or parameters.integration_substeps < 1
        or not 0.0 < poisson_ratio < 0.5
        or radius_to_mean_edge_ratio <= 0.0
    ):
        raise ValueError("PyElastica simulation parameters are invalid")

    current = observed[1]
    edges = np.diff(current, axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    base_length = float(np.sum(edge_lengths))
    base_radius = float(np.mean(edge_lengths) * radius_to_mean_edge_ratio)
    directors = deform_pyelastica_directors(current)
    direction = current[-1] - current[0]
    if np.linalg.norm(direction) <= 1e-8:
        direction = directors[2, :, 0]
    direction = direction / np.linalg.norm(direction)
    normal = directors[0, :, 0]
    shear_modulus = parameters.youngs_modulus_pa / (2.0 * (1.0 + poisson_ratio))

    class Simulator(
        elastica.BaseSystemCollection,  # type: ignore[misc]
        elastica.Constraints,  # type: ignore[misc]
        elastica.Forcing,  # type: ignore[misc]
        elastica.Damping,  # type: ignore[misc]
    ):
        pass

    endpoint_series = observed[1:, DEFORM_CLAMPED_NODES]

    class RegisteredTrajectoryConstraint(elastica.ConstraintBase):  # type: ignore[misc]
        def __init__(self, *args: Any, endpoint_series: Array, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.endpoint_series = np.asarray(endpoint_series, dtype=np.float64)

        def constrain_values(self, system: Any, time: np.float64) -> None:
            positions, _ = deform_pyelastica_kinematic_sample(
                self.endpoint_series, float(time), frame_dt_s=frame_dt_s
            )
            system.position_collection[:, self._constrained_position_idx] = positions.T

        def constrain_rates(self, system: Any, time: np.float64) -> None:
            _, velocities = deform_pyelastica_kinematic_sample(
                self.endpoint_series, float(time), frame_dt_s=frame_dt_s
            )
            system.velocity_collection[:, self._constrained_position_idx] = velocities.T

    simulator = Simulator()
    rod = elastica.CosseratRod.straight_rod(
        n_elements=11,
        start=current[0].copy(),
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=parameters.density_kg_m3,
        youngs_modulus=parameters.youngs_modulus_pa,
        shear_modulus=shear_modulus,
        position=current.T.copy(),
        directors=directors,
    )
    rod.velocity_collection[...] = ((observed[1] - observed[0]) / frame_dt_s).T
    simulator.append(rod)
    simulator.constrain(rod).using(
        RegisteredTrajectoryConstraint,
        endpoint_series=endpoint_series,
        constrained_position_idx=DEFORM_CLAMPED_NODES,
    )
    step_dt = frame_dt_s / parameters.integration_substeps
    simulator.add_forcing_to(rod).using(
        elastica.GravityForces,
        acc_gravity=np.asarray((0.0, 0.0, -9.81), dtype=np.float64),
    )
    simulator.dampen(rod).using(
        elastica.AnalyticalLinearDamper,
        damping_constant=parameters.damping_constant,
        time_step=step_dt,
    )
    simulator.finalize()
    stepper = elastica.PositionVerlet()
    time = np.float64(0.0)
    predictions = np.empty((observed.shape[0] - 2, 12, 3), dtype=np.float64)
    for frame in range(predictions.shape[0]):
        for _ in range(parameters.integration_substeps):
            time = stepper.step(simulator, time, np.float64(step_dt))
        predictions[frame] = rod.position_collection.T
    predictions[:, DEFORM_CLAMPED_NODES] = observed[2:, DEFORM_CLAMPED_NODES]
    if not np.isfinite(predictions).all():
        raise RuntimeError("PyElastica produced non-finite DLO predictions")
    return cast(Array, np.asarray(predictions, dtype=np.float64))
