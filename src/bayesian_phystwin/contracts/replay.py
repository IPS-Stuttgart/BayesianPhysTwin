"""Typed, immutable contracts for PhysTwin replay providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np


def _identifier(value: str, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be a nonempty identifier")
    return result


def _finite_float_array(
    values: np.ndarray,
    *,
    name: str,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    array = np.array(values, dtype=np.float32, copy=True, order="C")
    if array.ndim != ndim or (
        trailing_shape and array.shape[-len(trailing_shape) :] != trailing_shape
    ):
        suffix = "" if not trailing_shape else f" ending in {trailing_shape}"
        raise ValueError(f"{name} must be a {ndim}-dimensional array{suffix}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _replay_controls(values: np.ndarray) -> np.ndarray:
    controls = _finite_float_array(
        values,
        name="controller_points_m",
        ndim=3,
        trailing_shape=(3,),
    )
    if controls.shape[0] < 1 or controls.shape[1] < 1:
        raise ValueError("controller_points_m must have shape (T>=1, C>=1, 3)")
    return controls


def _group_log_scales(values: np.ndarray) -> np.ndarray:
    scales = _finite_float_array(values, name="group_log_scales", ndim=1)
    if not len(scales):
        raise ValueError("group_log_scales must be nonempty")
    return scales


@dataclass(frozen=True, slots=True)
class InitialReplayRequestV1:
    """Fully specified replay from the released simulator initial state."""

    request_id: str
    simulator_configuration_id: str
    initial_state_id: str
    group_log_scales: np.ndarray
    controller_points_m: np.ndarray
    frame_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _identifier(self.request_id, name="request_id"))
        object.__setattr__(
            self,
            "simulator_configuration_id",
            _identifier(
                self.simulator_configuration_id,
                name="simulator_configuration_id",
            ),
        )
        object.__setattr__(
            self,
            "initial_state_id",
            _identifier(self.initial_state_id, name="initial_state_id"),
        )
        object.__setattr__(self, "group_log_scales", _group_log_scales(self.group_log_scales))
        controls = _replay_controls(self.controller_points_m)
        object.__setattr__(self, "controller_points_m", controls)
        frame_count = int(self.frame_count)
        if frame_count < 1 or frame_count > len(controls):
            raise ValueError(
                "frame_count must be positive and covered by controller_points_m"
            )
        object.__setattr__(self, "frame_count", frame_count)


@dataclass(frozen=True, slots=True)
class RestartReplayRequestV1:
    """Fully specified replay from an explicit position/velocity endpoint state."""

    request_id: str
    simulator_configuration_id: str
    initial_state_id: str
    group_log_scales: np.ndarray
    controller_points_m: np.ndarray
    position_m: np.ndarray
    velocity_mps: np.ndarray
    start_frame: int
    stop_frame: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _identifier(self.request_id, name="request_id"))
        object.__setattr__(
            self,
            "simulator_configuration_id",
            _identifier(
                self.simulator_configuration_id,
                name="simulator_configuration_id",
            ),
        )
        object.__setattr__(
            self,
            "initial_state_id",
            _identifier(self.initial_state_id, name="initial_state_id"),
        )
        object.__setattr__(self, "group_log_scales", _group_log_scales(self.group_log_scales))
        controls = _replay_controls(self.controller_points_m)
        object.__setattr__(self, "controller_points_m", controls)
        position = _finite_float_array(
            self.position_m,
            name="position_m",
            ndim=2,
            trailing_shape=(3,),
        )
        velocity = _finite_float_array(
            self.velocity_mps,
            name="velocity_mps",
            ndim=2,
            trailing_shape=(3,),
        )
        if position.shape != velocity.shape or position.shape[0] < 1:
            raise ValueError(
                "position_m and velocity_mps must have matching shape (N>=1, 3)"
            )
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_mps", velocity)
        start_frame = int(self.start_frame)
        stop_frame = int(self.stop_frame)
        if not 0 <= start_frame < stop_frame <= len(controls):
            raise ValueError(
                "restart interval must be nonempty and covered by controller_points_m"
            )
        object.__setattr__(self, "start_frame", start_frame)
        object.__setattr__(self, "stop_frame", stop_frame)


ReplayRequestV1: TypeAlias = InitialReplayRequestV1 | RestartReplayRequestV1


@dataclass(frozen=True, slots=True)
class ReplayTrajectoryV1:
    """Immutable trajectory and provenance returned by provider API v2."""

    positions_m: np.ndarray
    velocities_mps: np.ndarray
    frame_ids: np.ndarray
    dt_s: float
    request_id: str
    simulator_configuration_id: str
    initial_state_id: str

    def __post_init__(self) -> None:
        positions = _finite_float_array(
            self.positions_m,
            name="positions_m",
            ndim=3,
            trailing_shape=(3,),
        )
        velocities = _finite_float_array(
            self.velocities_mps,
            name="velocities_mps",
            ndim=3,
            trailing_shape=(3,),
        )
        if (
            positions.shape != velocities.shape
            or positions.shape[0] < 1
            or positions.shape[1] < 1
        ):
            raise ValueError(
                "positions_m and velocities_mps must have matching "
                "shape (T>=1, N>=1, 3)"
            )
        frame_ids = np.array(self.frame_ids, dtype=np.int64, copy=True, order="C")
        if frame_ids.shape != (len(positions),):
            raise ValueError("frame_ids must identify every trajectory frame")
        if np.any(frame_ids < 0):
            raise ValueError("frame_ids must be nonnegative")
        if len(frame_ids) > 1 and np.any(np.diff(frame_ids) <= 0):
            raise ValueError("frame_ids must be strictly increasing")
        frame_ids.setflags(write=False)
        dt_s = float(self.dt_s)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "dt_s", dt_s)
        object.__setattr__(self, "request_id", _identifier(self.request_id, name="request_id"))
        object.__setattr__(
            self,
            "simulator_configuration_id",
            _identifier(
                self.simulator_configuration_id,
                name="simulator_configuration_id",
            ),
        )
        object.__setattr__(
            self,
            "initial_state_id",
            _identifier(self.initial_state_id, name="initial_state_id"),
        )


@runtime_checkable
class PhysTwinReplayProviderV1(Protocol):
    """Legacy mutable replay protocol retained for existing Causal4D consumers."""

    @property
    def device(self) -> str:
        """Torch device used by the provider."""

        ...

    def set_group_log_scales(self, values: np.ndarray) -> None:
        """Set grouped spring log-scales for subsequent replays."""

        ...

    def set_controller_points(self, values: np.ndarray) -> None:
        """Set the controller trajectory for subsequent replays."""

        ...

    def replay_initial(self, *, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
        """Replay from the released initial state."""

        ...

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        """Replay from an explicit endpoint state and return positions."""

        ...

    def close(self) -> None:
        """Release simulator resources."""

        ...


@runtime_checkable
class PhysTwinReplayProviderV2(Protocol):
    """Stateless public replay protocol using explicit typed requests."""

    @property
    def device(self) -> str:
        """Torch device used by the provider."""

        ...

    @property
    def frame_dt_s(self) -> float:
        """Physical interval represented by one returned trajectory frame."""

        ...

    @property
    def simulator_configuration_id(self) -> str:
        """Identifier for the fixed simulator configuration owned by this provider."""

        ...

    @property
    def released_initial_state_id(self) -> str:
        """Identifier for the released state used by initial-state requests."""

        ...

    def replay(self, request: ReplayRequestV1) -> ReplayTrajectoryV1:
        """Execute one fully specified replay request."""

        ...

    def close(self) -> None:
        """Release simulator resources."""

        ...


__all__ = [
    "InitialReplayRequestV1",
    "PhysTwinReplayProviderV1",
    "PhysTwinReplayProviderV2",
    "ReplayRequestV1",
    "ReplayTrajectoryV1",
    "RestartReplayRequestV1",
]
