"""Dependency-free PhysX deformable-volume replay adapter.

PhysX keeps the authoritative deformable-volume simulation-node positions in a
GPU ``PxVec4`` buffer exposed by ``PxDeformableVolume::getSimPositionInvMassBufferD``.
The producer that owns PhysX and CUDA must synchronize the engine and copy that
buffer to host memory. This module only validates the copied simulation-mesh
state and adapts it to the portable ``MaterialTrajectoryReplayV1`` protocol.

Using the collision or render mesh would change material identity and is
therefore intentionally unsupported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _host_array(value: object) -> npt.NDArray[Any]:
    """Copy a common host-visible array facade into contiguous NumPy memory."""

    current = value
    block_until_ready = getattr(current, "block_until_ready", None)
    if callable(block_until_ready):
        synchronized = block_until_ready()
        if synchronized is not None:
            current = synchronized
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    to_numpy = getattr(current, "numpy", None)
    if callable(to_numpy):
        current = to_numpy()
    return np.ascontiguousarray(np.asarray(current)).copy()


@dataclass(slots=True)
class PhysXDeformableVolumeReplayV1:
    """Adapt one PhysX deformable-volume simulation mesh to material replay v1.

    ``read_sim_position_inv_mass_callback`` must return the host-visible copy of
    ``PxDeformableVolume::getSimPositionInvMassBufferD()`` in simulation-mesh
    vertex order. PhysX stores one ``PxVec4`` per simulation vertex: XYZ position
    followed by inverse mass. The callback therefore must return floating
    ``(simulation_vertex_count, 4)`` data.

    ``synchronize_callback`` is intentionally mandatory. The owning PhysX/CUDA
    producer must ensure simulation tasks have finished and that the device-to-
    host copy is complete before the read callback is evaluated. This adapter
    does not import PhysX, allocate CUDA memory, or infer synchronization.
    """

    simulation_vertex_count: int
    read_sim_position_inv_mass_callback: Callable[[], object]
    advance_callback: Callable[[], object]
    synchronize_callback: Callable[[], object]
    context: object | None = None

    def __post_init__(self) -> None:
        self.simulation_vertex_count = _positive_integer(
            self.simulation_vertex_count,
            name="simulation_vertex_count",
        )
        for name in (
            "read_sim_position_inv_mass_callback",
            "advance_callback",
            "synchronize_callback",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        state = _host_array(self.read_sim_position_inv_mass_callback())
        expected_shape = (self.simulation_vertex_count, 4)
        _require(
            state.shape == expected_shape,
            "PhysX simulation position/inverse-mass buffer must have shape "
            f"{expected_shape}",
        )
        _require(
            np.issubdtype(state.dtype, np.floating),
            "PhysX simulation position/inverse-mass buffer must be floating point",
        )
        _require(
            np.all(np.isfinite(state)),
            "PhysX simulation position/inverse-mass buffer contains non-finite values",
        )
        _require(
            np.all(state[:, 3] >= 0.0),
            "PhysX simulation inverse masses must be nonnegative",
        )
        return cast(FloatArray, np.ascontiguousarray(state[:, :3]).copy())

    def step(self) -> object:
        return self.advance_callback()


__all__ = ["PhysXDeformableVolumeReplayV1"]
