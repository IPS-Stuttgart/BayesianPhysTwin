"""Dependency-free replay adapters for common material simulators.

The portable material-trajectory producer intentionally does not depend on heavy
simulator packages. This module provides small structural adapters for native
SOFA ``MechanicalObject``, MuJoCo Flex, PositionBasedDynamics particle state,
Genesis MPM entity state, and Warp FEM displacement fields while keeping those
dependencies on the caller side. Importing this module therefore never imports
SOFA, MuJoCo, pyPBD, Genesis, or Warp.

These adapters establish state-access compatibility only. They do not qualify a
backend runtime, its physical fidelity, uncertainty calibration, or downstream
Prob4D/Causal4D value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


class _SofaPositionValueV1(Protocol):
    value: object


class _SofaMechanicalObjectV1(Protocol):
    position: _SofaPositionValueV1


class _PBDParticleDataV1(Protocol):
    def getNumberOfParticles(self) -> object: ...

    def getVertices(self) -> object: ...


class _PBDSimulationModelV1(Protocol):
    def getParticles(self) -> _PBDParticleDataV1: ...


class _WarpDiscreteFieldV1(Protocol):
    degree: object
    dof_values: object


class _GenesisMPMEntityV1(Protocol):
    def get_state(self) -> object: ...


class _GenesisMPMStateV1(Protocol):
    pos: object
    active: object


def _no_op() -> None:
    return None


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _integer_vector(owner: object, name: str) -> npt.NDArray[np.int64]:
    value = getattr(owner, name, None)
    if value is None:
        raise TypeError(f"engine object must expose {name}")
    array = np.asarray(value)
    _require(array.ndim == 1, f"{name} must be a one-dimensional array")
    _require(
        np.issubdtype(array.dtype, np.integer),
        f"{name} must contain integer indices",
    )
    return np.ascontiguousarray(array, dtype=np.int64)


def _host_value(value: object) -> object:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        return to_numpy()
    return value


def _material_positions(value: object, *, label: str) -> FloatArray:
    positions = np.ascontiguousarray(np.asarray(_host_value(value))).copy()
    _require(
        positions.ndim == 2 and positions.shape[0] >= 1 and positions.shape[1] == 3,
        f"{label} positions must have shape (N,3)",
    )
    _require(
        np.issubdtype(positions.dtype, np.floating),
        f"{label} positions must be floating point",
    )
    _require(
        np.all(np.isfinite(positions)),
        f"{label} positions contain non-finite values",
    )
    return cast(FloatArray, positions)


@dataclass(slots=True)
class SofaMechanicalObjectReplayV1:
    """Adapt one SOFA Vec3 ``MechanicalObject`` to material replay v1.

    ``animate_callback`` is normally ``Sofa.Simulation.animate``. The adapter
    passes ``root_node`` and the registered output ``time_step_s`` to it. The
    mechanical state is read from the public ``MechanicalObject.position.value``
    surface and must already use metres in the canonical coordinate frame.
    """

    mechanical_object: object
    root_node: object
    animate_callback: Callable[[object, float], object]
    time_step_s: float
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None

    def __post_init__(self) -> None:
        if not callable(self.animate_callback):
            raise TypeError("animate_callback must be callable")
        if not callable(self.synchronize_callback):
            raise TypeError("synchronize_callback must be callable")
        position = getattr(self.mechanical_object, "position", None)
        if position is None or not hasattr(position, "value"):
            raise TypeError(
                "mechanical_object must expose the SOFA position.value surface"
            )
        self.time_step_s = _positive_float(self.time_step_s, name="time_step_s")

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        mechanical_object = cast(_SofaMechanicalObjectV1, self.mechanical_object)
        return _material_positions(
            mechanical_object.position.value,
            label="SOFA MechanicalObject",
        )

    def step(self) -> object:
        return self.animate_callback(self.root_node, self.time_step_s)


@dataclass(slots=True)
class MuJoCoFlexReplayV1:
    """Adapt one MuJoCo Flex to the fixed-material replay protocol.

    MuJoCo stores all flex vertices in ``data.flexvert_xpos`` and exposes the
    per-flex slices through ``model.flex_vertadr`` and ``model.flex_vertnum``.
    ``step_callback`` is normally ``mujoco.mj_step``. The selected flex vertex
    order is preserved exactly and must already be expressed in metres in the
    canonical coordinate frame.
    """

    model: object
    data: object
    flex_id: int
    step_callback: Callable[[object, object], object]
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None
    _vertex_start: int = field(init=False, repr=False)
    _vertex_stop: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.step_callback):
            raise TypeError("step_callback must be callable")
        if not callable(self.synchronize_callback):
            raise TypeError("synchronize_callback must be callable")
        flex_id = _nonnegative_integer(self.flex_id, name="flex_id")
        addresses = _integer_vector(self.model, "flex_vertadr")
        counts = _integer_vector(self.model, "flex_vertnum")
        _require(
            addresses.shape == counts.shape,
            "MuJoCo flex vertex address/count arrays must have the same shape",
        )
        _require(
            flex_id < len(addresses),
            "flex_id exceeds the MuJoCo flex roster",
        )
        start = int(addresses[flex_id])
        count = int(counts[flex_id])
        _require(start >= 0, "MuJoCo flex vertex address must be nonnegative")
        _require(count >= 1, "MuJoCo flex must contain at least one vertex")
        self.flex_id = flex_id
        self._vertex_start = start
        self._vertex_stop = start + count

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        value = getattr(self.data, "flexvert_xpos", None)
        if value is None:
            raise TypeError("MuJoCo data must expose flexvert_xpos")
        positions = _material_positions(value, label="MuJoCo flexvert_xpos")
        _require(
            self._vertex_stop <= len(positions),
            "MuJoCo flex vertex slice exceeds data.flexvert_xpos",
        )
        return cast(
            FloatArray,
            np.ascontiguousarray(
                positions[self._vertex_start : self._vertex_stop]
            ).copy(),
        )

    def step(self) -> object:
        return self.step_callback(self.model, self.data)


@dataclass(slots=True)
class PositionBasedDynamicsReplayV1:
    """Adapt pyPBD ``SimulationModel`` particle state to material replay v1.

    The pinned pyPBD API exposes one global ``ParticleData`` object through
    ``SimulationModel.getParticles()`` and its persistent particle rows through
    ``ParticleData.getVertices()``. ``time_step.step(simulation_model)`` advances
    the registered PBD/XPBD solver by one output step.
    """

    simulation_model: object
    time_step: object
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.simulation_model, "getParticles", None)):
            raise TypeError("simulation_model must expose getParticles()")
        if not callable(getattr(self.time_step, "step", None)):
            raise TypeError("time_step must expose step(simulation_model)")
        if not callable(self.synchronize_callback):
            raise TypeError("synchronize_callback must be callable")

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        model = cast(_PBDSimulationModelV1, self.simulation_model)
        particles = model.getParticles()
        get_particle_count = getattr(particles, "getNumberOfParticles", None)
        if not callable(get_particle_count):
            raise TypeError("pyPBD ParticleData must expose getNumberOfParticles()")
        particle_count = _nonnegative_integer(
            get_particle_count(),
            name="pyPBD particle count",
        )
        _require(
            particle_count >= 1,
            "pyPBD ParticleData must contain at least one particle",
        )
        get_vertices = getattr(particles, "getVertices", None)
        if not callable(get_vertices):
            raise TypeError("pyPBD ParticleData must expose getVertices()")
        positions = _material_positions(
            get_vertices(),
            label="PositionBasedDynamics ParticleData",
        )
        _require(
            len(positions) == particle_count,
            "pyPBD particle count does not match getVertices()",
        )
        return positions

    def step(self) -> object:
        time_step = cast(Any, self.time_step)
        return time_step.step(self.simulation_model)


@dataclass(slots=True)
class GenesisMPMEntityReplayV1:
    """Adapt one Genesis ``MPMEntity`` to fixed-material replay v1.

    Genesis exposes entity-local particle state through ``entity.get_state()``.
    Its ``pos`` and ``active`` arrays have shapes ``(B,N,3)`` and ``(B,N)``.
    This adapter selects one registered environment, freezes its active-particle
    roster at construction, and rejects later insertion, deletion, or activity
    changes rather than silently changing material identity.
    """

    entity: object
    step_callback: Callable[[], object]
    environment_index: int = 0
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None
    _particle_count: int = field(init=False, repr=False)
    _active_mask: npt.NDArray[np.bool_] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.entity, "get_state", None)):
            raise TypeError("Genesis MPM entity must expose get_state()")
        if not callable(self.step_callback):
            raise TypeError("step_callback must be callable")
        if not callable(self.synchronize_callback):
            raise TypeError("synchronize_callback must be callable")
        self.environment_index = _nonnegative_integer(
            self.environment_index,
            name="environment_index",
        )
        positions, active = self._read_entity_state()
        self._particle_count = positions.shape[0]
        self._active_mask = np.ascontiguousarray(active, dtype=np.bool_)
        _require(
            np.any(self._active_mask),
            "Genesis MPM environment must contain at least one active particle",
        )

    def _read_entity_state(
        self,
    ) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        entity = cast(_GenesisMPMEntityV1, self.entity)
        state = entity.get_state()
        if state is None:
            raise ValueError("Genesis MPM entity returned no active state")
        if not hasattr(state, "pos") or not hasattr(state, "active"):
            raise TypeError("Genesis MPM state must expose pos and active arrays")
        genesis_state = cast(_GenesisMPMStateV1, state)

        positions = np.ascontiguousarray(
            np.asarray(_host_value(genesis_state.pos))
        ).copy()
        active = np.ascontiguousarray(
            np.asarray(_host_value(genesis_state.active))
        ).copy()
        _require(
            positions.ndim == 3
            and positions.shape[0] >= 1
            and positions.shape[1] >= 1
            and positions.shape[2] == 3,
            "Genesis MPM positions must have shape (B,N,3)",
        )
        _require(
            np.issubdtype(positions.dtype, np.floating),
            "Genesis MPM positions must be floating point",
        )
        _require(
            active.ndim == 2 and active.shape == positions.shape[:2],
            "Genesis MPM active mask must have shape (B,N)",
        )
        _require(
            np.issubdtype(active.dtype, np.bool_)
            or np.issubdtype(active.dtype, np.integer),
            "Genesis MPM active mask must be boolean or integer",
        )
        if np.issubdtype(active.dtype, np.integer):
            _require(
                np.all((active == 0) | (active == 1)),
                "Genesis MPM integer active mask must contain only zero or one",
            )
        _require(
            self.environment_index < positions.shape[0],
            "environment_index exceeds the Genesis MPM batch",
        )

        selected_positions = np.ascontiguousarray(
            positions[self.environment_index]
        ).copy()
        selected_active = np.ascontiguousarray(
            active[self.environment_index], dtype=np.bool_
        )
        _require(
            np.all(np.isfinite(selected_positions[selected_active])),
            "Genesis MPM active positions contain non-finite values",
        )
        return cast(FloatArray, selected_positions), selected_active

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        positions, active = self._read_entity_state()
        _require(
            len(positions) == self._particle_count,
            "Genesis MPM particle count changed during replay",
        )
        _require(
            np.array_equal(active, self._active_mask),
            "Genesis MPM active-particle roster changed during replay",
        )
        return cast(
            FloatArray,
            np.ascontiguousarray(positions[self._active_mask]).copy(),
        )

    def step(self) -> object:
        return self.step_callback()


@dataclass(slots=True)
class WarpFEMDisplacementReplayV1:
    """Adapt a degree-1 Warp FEM displacement field to material replay v1.

    ``displacement_field`` is a Warp FEM ``DiscreteField`` whose ``dof_values``
    are three-dimensional nodal displacements. ``reference_positions_m`` is the
    fixed node roster in the exact same partition/order. The adapter adds the
    displacement DOFs to that frozen reference and returns absolute positions.
    """

    displacement_field: object
    reference_positions_m: object
    step_callback: Callable[[], object]
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None
    _reference_positions_m: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.step_callback):
            raise TypeError("step_callback must be callable")
        if not callable(self.synchronize_callback):
            raise TypeError("synchronize_callback must be callable")

        degree = getattr(self.displacement_field, "degree", None)
        if isinstance(degree, (bool, np.bool_)) or not isinstance(
            degree, (int, np.integer)
        ):
            raise TypeError("Warp FEM displacement field must expose integer degree")
        _require(
            int(degree) == 1,
            "Warp FEM displacement replay requires a degree-1 nodal field",
        )

        dof_values = getattr(self.displacement_field, "dof_values", None)
        if dof_values is None or not callable(getattr(dof_values, "numpy", None)):
            raise TypeError(
                "Warp FEM displacement_field.dof_values must expose Warp array numpy()"
            )

        self._reference_positions_m = _material_positions(
            self.reference_positions_m,
            label="Warp FEM reference",
        )

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        field_value = cast(_WarpDiscreteFieldV1, self.displacement_field)
        dof_values = getattr(field_value, "dof_values", None)
        if dof_values is None or not callable(getattr(dof_values, "numpy", None)):
            raise TypeError(
                "Warp FEM displacement_field.dof_values must expose Warp array numpy()"
            )
        displacements = _material_positions(
            dof_values,
            label="Warp FEM displacement",
        )
        _require(
            displacements.shape == self._reference_positions_m.shape,
            "Warp FEM displacement and reference node rosters must match",
        )
        positions = np.ascontiguousarray(self._reference_positions_m + displacements)
        _require(
            np.all(np.isfinite(positions)),
            "Warp FEM absolute positions contain non-finite values",
        )
        return cast(FloatArray, positions)

    def step(self) -> object:
        return self.step_callback()


__all__ = [
    "GenesisMPMEntityReplayV1",
    "MuJoCoFlexReplayV1",
    "PositionBasedDynamicsReplayV1",
    "SofaMechanicalObjectReplayV1",
    "WarpFEMDisplacementReplayV1",
]
