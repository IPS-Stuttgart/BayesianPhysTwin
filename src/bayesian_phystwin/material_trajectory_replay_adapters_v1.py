"""Dependency-free replay adapters for external material simulators.

The adapters in this module expose engine-native state through the structural
``MaterialTrajectoryReplayV1`` protocol without importing the optional engine
packages into BayesianPhysTwin.
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


def _to_numpy(value: object) -> npt.NDArray[Any]:
    """Synchronize common array facades and copy them into host NumPy memory."""

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
class DolfinxDisplacementReplayV1:
    """Adapt globally ordered DOLFINx displacements to material positions.

    The caller must gather or otherwise register one stable global geometry-node
    order before constructing the adapter. Rank-local rows, ghost degrees of
    freedom, and repartition-dependent orderings are deliberately not accepted as
    material identity by this class.
    """

    reference_positions_m: object
    displacement_callback: Callable[[], object]
    step_callback: Callable[[], object]
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None

    def __post_init__(self) -> None:
        for name in (
            "displacement_callback",
            "step_callback",
            "synchronize_callback",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        reference = _to_numpy(self.reference_positions_m)
        _require(
            reference.ndim == 2
            and reference.shape[0] >= 1
            and reference.shape[1] == 3,
            "reference_positions_m must have shape (S,3)",
        )
        _require(
            np.issubdtype(reference.dtype, np.floating),
            "reference_positions_m must be floating point",
        )
        _require(
            np.all(np.isfinite(reference)),
            "reference_positions_m contains non-finite values",
        )
        self.reference_positions_m = reference

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        reference = cast(npt.NDArray[Any], self.reference_positions_m)
        displacement = _to_numpy(self.displacement_callback())
        _require(
            displacement.shape == reference.shape,
            "DOLFINx displacement shape differs from the registered reference shape",
        )
        _require(
            displacement.dtype == reference.dtype,
            "DOLFINx displacement dtype differs from the registered reference dtype",
        )
        _require(
            np.all(np.isfinite(displacement)),
            "DOLFINx displacement contains non-finite values",
        )
        return cast(FloatArray, np.ascontiguousarray(reference + displacement))

    def step(self) -> object:
        return self.step_callback()


@dataclass(slots=True)
class PyElasticaRodReplayV1:
    """Adapt one PyElastica rod's persistent node order to material rows."""

    rod: object
    step_callback: Callable[[], object]
    synchronize_callback: Callable[[], object] = _no_op
    context: object | None = None

    def __post_init__(self) -> None:
        if not hasattr(self.rod, "position_collection"):
            raise TypeError("rod must expose position_collection")
        for name in ("step_callback", "synchronize_callback"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def synchronize(self) -> object:
        return self.synchronize_callback()

    def get_material_positions_m(self) -> FloatArray:
        native = _to_numpy(getattr(self.rod, "position_collection"))
        _require(
            native.ndim == 2 and native.shape[0] == 3 and native.shape[1] >= 2,
            "PyElastica position_collection must have shape (3,N) with N >= 2",
        )
        _require(
            np.issubdtype(native.dtype, np.floating),
            "PyElastica position_collection must be floating point",
        )
        _require(
            np.all(np.isfinite(native)),
            "PyElastica position_collection contains non-finite values",
        )
        return cast(FloatArray, np.ascontiguousarray(native.T))

    def step(self) -> object:
        return self.step_callback()


__all__ = [
    "DolfinxDisplacementReplayV1",
    "PyElasticaRodReplayV1",
]
