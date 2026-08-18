"""Dependency-free replay adapters for common material simulators.

The portable material-trajectory producer intentionally does not depend on heavy
simulator packages. This module provides small structural adapters for native
SOFA ``MechanicalObject``, MuJoCo Flex, and PositionBasedDynamics particle state
while keeping those dependencies on the caller side. Importing this module
therefore never imports SOFA, MuJoCo, or pyPBD.

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


def _material_positions(value: object, *, label: str) -> FloatArray:
    positions = np.ascontiguousarray(np.asarray(value)).copy()
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
            raise TypeError(
                "pyPBD ParticleData must expose getNumberOfParticles()"
            )
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


__all__ = [
    "MuJoCoFlexReplayV1",
    "PositionBasedDynamicsReplayV1",
    "SofaMechanicalObjectReplayV1",
]
