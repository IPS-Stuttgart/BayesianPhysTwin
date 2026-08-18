"""Dependency-free Genesis MPM replay for the material-trajectory producer.

Genesis remains an optional producer-side dependency.  Callers pass an already
built scene and one MPM entity; this module uses only the public
``get_particles_pos()`` and ``scene.step()`` surfaces and therefore never imports
Genesis itself.

The adapter establishes state-access compatibility only.  It does not qualify a
Genesis runtime, its physical fidelity, calibration, transfer, or downstream
Prob4D/Causal4D value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


def _no_op() -> None:
    return None


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _environment_index(value: object) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError("env_index must be a nonnegative integer or None")
    return int(value)


def _host_copy(value: object) -> npt.NDArray[Any]:
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
class GenesisMPMEntityReplayV1:
    """Adapt one built Genesis MPM entity to material replay v1.

    ``entity.get_particles_pos()`` is treated as the persistent material-particle
    roster.  For batched scenes, ``env_index`` must select exactly one environment
    unless the engine already returns a singleton batch.  The selected particle
    order must remain fixed for the complete replay.
    """

    scene: object
    entity: object
    env_index: int | None = None
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.scene, "step", None)):
            raise TypeError("scene must expose step()")
        if not callable(getattr(self.entity, "get_particles_pos", None)):
            raise TypeError("entity must expose get_particles_pos()")
        if not callable(self.synchronize_callback):
            raise TypeError("synchronize_callback must be callable")
        self.env_index = _environment_index(self.env_index)
        if self.context is None:
            self.context = self.entity

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        get_positions = cast(Any, self.entity).get_particles_pos
        if self.env_index is None:
            raw = get_positions()
        else:
            raw = get_positions(envs_idx=[self.env_index])
        positions = _host_copy(raw)
        if positions.ndim == 3:
            _require(
                positions.shape[0] == 1,
                "Genesis batched particle positions require an exact env_index",
            )
            positions = np.ascontiguousarray(positions[0]).copy()
        _require(
            positions.ndim == 2
            and positions.shape[0] >= 1
            and positions.shape[1] == 3,
            "Genesis MPM particle positions must have shape (N,3)",
        )
        _require(
            np.issubdtype(positions.dtype, np.floating),
            "Genesis MPM particle positions must be floating point",
        )
        _require(
            np.all(np.isfinite(positions)),
            "Genesis MPM particle positions contain non-finite values",
        )
        particle_count = getattr(self.entity, "n_particles", None)
        if particle_count is not None:
            if (
                isinstance(particle_count, (bool, np.bool_))
                or not isinstance(particle_count, (int, np.integer))
            ):
                raise TypeError("Genesis entity n_particles must be an integer")
            _require(
                int(particle_count) == len(positions),
                "Genesis entity particle count does not match get_particles_pos()",
            )
        return cast(FloatArray, positions)

    def step(self) -> object:
        return cast(Any, self.scene).step()


__all__ = ["GenesisMPMEntityReplayV1"]
