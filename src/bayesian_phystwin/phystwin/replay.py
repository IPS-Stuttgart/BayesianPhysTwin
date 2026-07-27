"""Owned PhysTwin replay implementation behind the versioned provider contracts."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.replay import (
    InitialReplayRequestV1,
    ReplayRequestV1,
    ReplayTrajectoryV1,
    RestartReplayRequestV1,
)


def _state_numpy(state: Any, wp: Any) -> tuple[np.ndarray, np.ndarray]:
    """Copy one Warp simulator state into host NumPy arrays."""

    position = wp.to_torch(state.wp_x).detach().cpu().numpy().copy()
    velocity = wp.to_torch(state.wp_v).detach().cpu().numpy().copy()
    return position, velocity


def _rollout_initial_trajectory(
    simulator: Any,
    wp: Any,
    *,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the released initial state and return position and velocity histories."""

    frame_count = int(frame_count)
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    simulator.set_init_state(
        simulator.wp_init_vertices,
        simulator.wp_init_velocities,
    )
    wp.synchronize()
    positions = []
    velocities = []
    position, velocity = _state_numpy(simulator.wp_states[0], wp)
    positions.append(position)
    velocities.append(velocity)
    for frame in range(1, frame_count):
        simulator.set_controller_target(frame, pure_inference=True)
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        wp.capture_launch(simulator.forward_graph)
        wp.synchronize()
        position, velocity = _state_numpy(simulator.wp_states[-1], wp)
        positions.append(position)
        velocities.append(velocity)
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
        )
    return np.stack(positions), np.stack(velocities)


def _rollout_restart_trajectory(
    simulator: Any,
    torch: Any,
    wp: Any,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay an explicit endpoint state and return future positions and velocities."""

    position = np.asarray(position_m, dtype=np.float32)
    velocity = np.asarray(velocity_mps, dtype=np.float32)
    if position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape:
        raise ValueError("restart position and velocity must have shape (N, 3)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("restart state must be finite")
    start_frame = int(start_frame)
    stop_frame = int(stop_frame)
    if not 0 <= start_frame < stop_frame:
        raise ValueError("restart frame interval must be nonempty")

    position_tensor = torch.as_tensor(
        position,
        dtype=torch.float32,
        device=device,
    ).contiguous()
    velocity_tensor = torch.as_tensor(
        velocity,
        dtype=torch.float32,
        device=device,
    ).contiguous()
    position_wp = wp.from_torch(position_tensor, dtype=wp.vec3, requires_grad=False)
    velocity_wp = wp.from_torch(velocity_tensor, dtype=wp.vec3, requires_grad=False)
    simulator.set_init_state(position_wp, velocity_wp)
    wp.synchronize()
    positions = []
    velocities = []
    for frame in range(start_frame, stop_frame):
        simulator.set_controller_target(frame, pure_inference=True)
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        wp.capture_launch(simulator.forward_graph)
        wp.synchronize()
        next_position, next_velocity = _state_numpy(simulator.wp_states[-1], wp)
        positions.append(next_position)
        velocities.append(next_velocity)
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
        )
    return np.stack(positions), np.stack(velocities)


class OfficialPhysTwinReplayProviderV2:
    """Official Warp adapter implementing the explicit provider API v2 contract."""

    def __init__(
        self,
        simulator: Any,
        torch: Any,
        wp: Any,
        *,
        device: str,
        frame_dt_s: float,
        simulator_configuration_id: str,
        released_initial_state_id: str,
    ) -> None:
        frame_dt_s = float(frame_dt_s)
        if not np.isfinite(frame_dt_s) or frame_dt_s <= 0.0:
            raise ValueError("frame_dt_s must be positive and finite")
        configuration_id = str(simulator_configuration_id).strip()
        initial_state_id = str(released_initial_state_id).strip()
        if not configuration_id or not initial_state_id:
            raise ValueError("provider identifiers must be nonempty")
        device_name = str(device).strip()
        if not device_name:
            raise ValueError("device must be a nonempty identifier")
        self._simulator = simulator
        self._torch = torch
        self._wp = wp
        self._device = device_name
        self._frame_dt_s = frame_dt_s
        self._simulator_configuration_id = configuration_id
        self._released_initial_state_id = initial_state_id
        self._closed = False

    @property
    def device(self) -> str:
        return self._device

    @property
    def frame_dt_s(self) -> float:
        return self._frame_dt_s

    @property
    def simulator_configuration_id(self) -> str:
        return self._simulator_configuration_id

    @property
    def released_initial_state_id(self) -> str:
        return self._released_initial_state_id

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PhysTwin replay provider is closed")

    def _validate_group_log_scales(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        target = self._simulator.group_log_scale_tensor
        expected = tuple(int(value) for value in target.shape)
        if array.shape != expected:
            raise ValueError(f"group log-scales must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("group log-scales must be finite")
        return array

    def _validate_controller_points(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        current = self._simulator.controller_points
        expected = tuple(int(value) for value in current.shape)
        if array.shape != expected:
            raise ValueError(f"controller points must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("controller points must be finite")
        return array

    def _apply_group_log_scales(self, values: np.ndarray) -> None:
        target = self._simulator.group_log_scale_tensor
        with self._torch.no_grad():
            target.copy_(
                self._torch.as_tensor(
                    values,
                    dtype=self._torch.float32,
                    device=self._device,
                )
            )
        self._wp.synchronize()

    def _apply_controller_points(self, values: np.ndarray) -> None:
        self._simulator.controller_points = self._torch.as_tensor(
            values,
            dtype=self._torch.float32,
            device=self._device,
        ).contiguous()
        self._wp.synchronize()

    def replay(self, request: ReplayRequestV1) -> ReplayTrajectoryV1:
        """Execute a request whose mutable simulator inputs are all explicit."""

        self._require_open()
        if request.simulator_configuration_id != self._simulator_configuration_id:
            raise ValueError(
                "replay request simulator_configuration_id does not match provider"
            )
        if (
            isinstance(request, InitialReplayRequestV1)
            and request.initial_state_id != self._released_initial_state_id
        ):
            raise ValueError(
                "initial replay request does not identify the released state"
            )

        group_log_scales = self._validate_group_log_scales(request.group_log_scales)
        controller_points = self._validate_controller_points(
            request.controller_points_m
        )
        self._apply_group_log_scales(group_log_scales)
        self._apply_controller_points(controller_points)
        frame_ids: np.ndarray
        if isinstance(request, InitialReplayRequestV1):
            positions, velocities = _rollout_initial_trajectory(
                self._simulator,
                self._wp,
                frame_count=request.frame_count,
            )
            frame_ids = np.arange(request.frame_count, dtype=np.int64)
        elif isinstance(request, RestartReplayRequestV1):
            positions, velocities = _rollout_restart_trajectory(
                self._simulator,
                self._torch,
                self._wp,
                request.position_m,
                request.velocity_mps,
                start_frame=request.start_frame,
                stop_frame=request.stop_frame,
                device=self._device,
            )
            frame_ids = np.arange(
                request.start_frame,
                request.stop_frame,
                dtype=np.int64,
            )
        else:  # pragma: no cover - defensive against untyped callers
            raise TypeError("unsupported replay request type")
        return ReplayTrajectoryV1(
            positions_m=positions,
            velocities_mps=velocities,
            frame_ids=frame_ids,
            dt_s=self._frame_dt_s,
            request_id=request.request_id,
            simulator_configuration_id=request.simulator_configuration_id,
            initial_state_id=request.initial_state_id,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._simulator = None
        gc.collect()
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "empty_cache"):
            cuda.empty_cache()

    def __enter__(self) -> OfficialPhysTwinReplayProviderV2:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _initialize_official_simulator(*args: Any, **kwargs: Any) -> Any:
    """Isolated compatibility seam for the released simulator constructor."""

    from bayesian_phystwin.phystwin_state_injection import _initialize_simulator

    return _initialize_simulator(*args, **kwargs)


def create_official_replay_provider_v2(
    official_repo: str | Path,
    data: Mapping[str, object],
    optimal: Mapping[str, object],
    checkpoint_path: str | Path,
    graph: Any,
    *,
    num_surface_points: int,
    original_count: int,
    dt: float,
    num_substeps: int,
    self_collision: bool,
    simulator_configuration_id: str,
    released_initial_state_id: str,
    deterministic_spring_forces: bool = False,
    spring_parameterization: str = "dense",
    device: str,
) -> OfficialPhysTwinReplayProviderV2:
    """Construct an official v2 provider without exposing simulator internals."""

    dt = float(dt)
    num_substeps = int(num_substeps)
    if not np.isfinite(dt) or dt <= 0.0 or num_substeps < 1:
        raise ValueError("dt and num_substeps must be positive")
    configuration_id = str(simulator_configuration_id).strip()
    initial_state_id = str(released_initial_state_id).strip()
    device_name = str(device).strip()
    if not configuration_id or not initial_state_id or not device_name:
        raise ValueError("provider identifiers and device must be nonempty")
    simulator, torch, wp, _ = _initialize_official_simulator(
        official_repo,
        dict(data),
        dict(optimal),
        checkpoint_path,
        graph,
        num_surface_points=num_surface_points,
        original_count=original_count,
        dt=dt,
        num_substeps=num_substeps,
        self_collision=self_collision,
        deterministic_spring_forces=deterministic_spring_forces,
        spring_parameterization=spring_parameterization,
        device=device_name,
    )
    return OfficialPhysTwinReplayProviderV2(
        simulator,
        torch,
        wp,
        device=device_name,
        frame_dt_s=dt * num_substeps,
        simulator_configuration_id=configuration_id,
        released_initial_state_id=initial_state_id,
    )


__all__ = [
    "OfficialPhysTwinReplayProviderV2",
    "create_official_replay_provider_v2",
]
