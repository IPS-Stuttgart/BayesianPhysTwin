"""Immutable data types for the Causal4D provider v2 boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


def _immutable_array(value: object, *, dtype: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhysTwinSpringGraphConfig:
    """Radius-neighbourhood settings stored in released PhysTwin parameters."""

    object_radius: float
    object_max_neighbours: int
    controller_radius: float
    controller_max_neighbours: int

    def __post_init__(self) -> None:
        radii = (float(self.object_radius), float(self.controller_radius))
        if any(not np.isfinite(value) or value <= 0.0 for value in radii):
            raise ValueError("spring radii must be positive and finite")
        if self.object_max_neighbours < 1 or self.controller_max_neighbours < 1:
            raise ValueError("spring neighbour limits must be positive")


@dataclass(frozen=True)
class PhysTwinSpringGraph:
    """Immutable provider-owned spring graph with explicit physical units."""

    vertices: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    masses: np.ndarray
    num_object_springs: int
    num_object_points: int | None = None

    def __post_init__(self) -> None:
        vertices = _immutable_array(self.vertices, dtype=np.float32, name="vertices")
        springs = _immutable_array(self.springs, dtype=np.int32, name="springs")
        rest_lengths = _immutable_array(
            self.rest_lengths, dtype=np.float32, name="rest_lengths"
        )
        masses = _immutable_array(self.masses, dtype=np.float32, name="masses")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError("vertices must have shape (V, 3) with V > 0")
        if springs.ndim != 2 or springs.shape[1] != 2:
            raise ValueError("springs must have shape (S, 2)")
        if rest_lengths.shape != (len(springs),):
            raise ValueError("rest_lengths must identify every spring")
        if masses.shape != (len(vertices),):
            raise ValueError("masses must identify every vertex")
        if len(springs) and (np.any(springs < 0) or np.any(springs >= len(vertices))):
            raise ValueError("spring endpoint exceeds the vertex array")
        if np.any(rest_lengths < 0.0) or np.any(masses <= 0.0):
            raise ValueError("rest lengths must be nonnegative and masses positive")
        object_springs = int(self.num_object_springs)
        if not 0 <= object_springs <= len(springs):
            raise ValueError("num_object_springs must lie in [0, S]")
        object_points = (
            None if self.num_object_points is None else int(self.num_object_points)
        )
        if object_points is not None and not 1 <= object_points <= len(vertices):
            raise ValueError("num_object_points must lie in [1, V]")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "springs", springs)
        object.__setattr__(self, "rest_lengths", rest_lengths)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "num_object_springs", object_springs)
        object.__setattr__(self, "num_object_points", object_points)

    @property
    def vertices_m(self) -> np.ndarray:
        return self.vertices

    @property
    def rest_lengths_m(self) -> np.ndarray:
        return self.rest_lengths

    @property
    def masses_kg(self) -> np.ndarray:
        return self.masses


@dataclass(frozen=True)
class PhysTwinControllerLayout:
    """Released hand count and deterministic controller-point partition."""

    hand_count: int
    group_ids: np.ndarray

    def __post_init__(self) -> None:
        groups = _immutable_array(self.group_ids, dtype=np.int32, name="group_ids")
        hand_count = int(self.hand_count)
        if groups.ndim != 1 or len(groups) == 0 or np.any(groups < 0):
            raise ValueError("group_ids must be a nonempty nonnegative vector")
        if hand_count < 1 or set(groups.tolist()) != set(range(hand_count)):
            raise ValueError("group_ids must contain every contiguous hand identifier")
        object.__setattr__(self, "hand_count", hand_count)
        object.__setattr__(self, "group_ids", groups)


@dataclass(frozen=True)
class PhysTwinCase:
    """Schema-validated released PhysTwin case owned by the provider boundary."""

    case_name: str
    object_points_m: np.ndarray
    object_visibilities: np.ndarray
    object_motions_valid: np.ndarray
    controller_points_m: np.ndarray
    surface_points_m: np.ndarray
    interior_points_m: np.ndarray
    graph_config: PhysTwinSpringGraphConfig
    baseline_trajectory_m: np.ndarray | None = None
    _provider_data: Mapping[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )
    _provider_optimal: Mapping[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.case_name:
            raise ValueError("case_name must be nonempty")
        object_points = _immutable_array(
            self.object_points_m, dtype=np.float32, name="object_points_m"
        )
        visible = _immutable_array(
            self.object_visibilities, dtype=bool, name="object_visibilities"
        )
        motion_valid = _immutable_array(
            self.object_motions_valid, dtype=bool, name="object_motions_valid"
        )
        controls = _immutable_array(
            self.controller_points_m, dtype=np.float32, name="controller_points_m"
        )
        surface = _immutable_array(
            self.surface_points_m, dtype=np.float32, name="surface_points_m"
        )
        interior = _immutable_array(
            self.interior_points_m, dtype=np.float32, name="interior_points_m"
        )
        if object_points.ndim != 3 or object_points.shape[2] != 3:
            raise ValueError("object_points_m must have shape (T, N, 3)")
        frame_count, object_count, _ = object_points.shape
        if visible.shape != (frame_count, object_count):
            raise ValueError("object_visibilities must have shape (T, N)")
        if motion_valid.shape != (frame_count, object_count):
            raise ValueError("object_motions_valid must have shape (T, N)")
        if (
            controls.ndim != 3
            or controls.shape[0] != frame_count
            or controls.shape[2] != 3
        ):
            raise ValueError("controller_points_m must have shape (T, C, 3)")
        for name, values in (
            ("surface_points_m", surface),
            ("interior_points_m", interior),
        ):
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError(f"{name} must have shape (K, 3)")
        baseline = None
        if self.baseline_trajectory_m is not None:
            baseline = _immutable_array(
                self.baseline_trajectory_m,
                dtype=np.float32,
                name="baseline_trajectory_m",
            )
            expected = (frame_count, object_count + len(surface) + len(interior), 3)
            if baseline.shape != expected:
                raise ValueError(
                    "baseline_trajectory_m must have shape "
                    f"{expected}, got {baseline.shape}"
                )
        object.__setattr__(self, "object_points_m", object_points)
        object.__setattr__(self, "object_visibilities", visible)
        object.__setattr__(self, "object_motions_valid", motion_valid)
        object.__setattr__(self, "controller_points_m", controls)
        object.__setattr__(self, "surface_points_m", surface)
        object.__setattr__(self, "interior_points_m", interior)
        object.__setattr__(self, "baseline_trajectory_m", baseline)
        object.__setattr__(
            self, "_provider_data", MappingProxyType(dict(self._provider_data))
        )
        object.__setattr__(
            self, "_provider_optimal", MappingProxyType(dict(self._provider_optimal))
        )

    @property
    def frame_count(self) -> int:
        return int(self.object_points_m.shape[0])

    @property
    def original_count(self) -> int:
        return int(self.object_points_m.shape[1])

    @property
    def structure_points_m(self) -> np.ndarray:
        values = np.concatenate(
            (self.object_points_m[0], self.surface_points_m, self.interior_points_m),
            axis=0,
        ).astype(np.float32, copy=False)
        values.setflags(write=False)
        return values

    @property
    def num_surface_points(self) -> int:
        return self.original_count + len(self.surface_points_m)
