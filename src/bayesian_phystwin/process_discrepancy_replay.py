"""Opt-in PhysTwin replay support for latent process-force schedules."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@runtime_checkable
class ProcessForceReplayProvider(Protocol):
    """Replay boundary required for opt-in time-varying nodal forces."""

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        """Run the unchanged baseline restart path."""
        ...

    def replay_restart_with_force_schedule(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        force_schedule_n: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        """Run a restart with one nodal force field per output frame."""
        ...

    def clear_external_forces(self) -> None:
        """Disable and zero any previously configured force input."""
        ...


def replay_with_process_force_schedule(
    provider: ProcessForceReplayProvider,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    force_schedule_n: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
) -> np.ndarray:
    """Dispatch exact zeros through the provider's unchanged baseline path."""

    schedule = np.asarray(force_schedule_n, dtype=np.float64)
    position = np.asarray(position_m)
    velocity = np.asarray(velocity_mps)
    _require(0 <= start_frame < stop_frame, "restart interval must be nonempty")
    _require(
        position.ndim == 2
        and position.shape[1] == 3
        and velocity.shape == position.shape,
        "restart position and velocity must have shape (node, 3)",
    )
    _require(
        schedule.shape == (stop_frame - start_frame, *position.shape),
        "force_schedule_n must have shape (stop-start, node, 3)",
    )
    _require(
        np.all(np.isfinite(position))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(schedule)),
        "restart state and force_schedule_n must be finite",
    )
    if not np.any(schedule != 0.0):
        provider.clear_external_forces()
        return provider.replay_restart(
            position_m,
            velocity_mps,
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
    return provider.replay_restart_with_force_schedule(
        position_m,
        velocity_mps,
        schedule,
        start_frame=start_frame,
        stop_frame=stop_frame,
    )


class OfficialProcessDiscrepancyReplayAdapter:
    """Opt-in force-schedule adapter for ``OfficialPhysTwinReplayProvider``.

    The adapter is intentionally not added to the stable Causal4D provider-v1
    protocol.  It uses the official provider's owned simulator boundary and the
    already captured opt-in external-force tensors.  Zero schedules always call
    the existing ``replay_restart`` implementation.
    """

    def __init__(self, official_provider: Any) -> None:
        required = ("_simulator", "_torch", "_wp", "_device", "_require_open")
        _require(
            all(hasattr(official_provider, name) for name in required),
            "official_provider does not expose the owned PhysTwin runtime",
        )
        self._provider = official_provider

    def clear_external_forces(self) -> None:
        self._provider._require_open()
        simulator = self._provider._simulator
        _require(
            hasattr(simulator, "clear_external_forces"),
            "simulator does not support opt-in external forces",
        )
        simulator.clear_external_forces()
        self._provider._wp.synchronize()

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        self.clear_external_forces()
        return np.asarray(
            self._provider.replay_restart(
                position_m,
                velocity_mps,
                start_frame=start_frame,
                stop_frame=stop_frame,
            )
        )

    def replay_restart_with_force_schedule(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        force_schedule_n: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        self._provider._require_open()
        schedule_input = np.asarray(force_schedule_n, dtype=np.float64)
        schedule = np.asarray(schedule_input, dtype=np.float32)
        position = np.asarray(position_m, dtype=np.float32)
        velocity = np.asarray(velocity_mps, dtype=np.float32)
        _require(0 <= start_frame < stop_frame, "restart interval must be nonempty")
        _require(
            position.ndim == 2
            and position.shape[1] == 3
            and velocity.shape == position.shape,
            "restart state must have shape (node, 3)",
        )
        _require(
            schedule.shape == (stop_frame - start_frame, *position.shape),
            "force schedule must match restart frames and nodes",
        )
        _require(
            np.all(np.isfinite(position))
            and np.all(np.isfinite(velocity))
            and np.all(np.isfinite(schedule)),
            "restart state and force schedule must be finite",
        )
        if not np.any(schedule_input != 0.0):
            return self.replay_restart(
                position,
                velocity,
                start_frame=start_frame,
                stop_frame=stop_frame,
            )
        _require(
            not np.any((schedule_input != 0.0) & (schedule == 0.0)),
            "nonzero force schedule underflows the float32 simulator input",
        )

        simulator = self._provider._simulator
        torch = self._provider._torch
        wp = self._provider._wp
        _require(
            hasattr(simulator, "set_external_forces")
            and hasattr(simulator, "clear_external_forces"),
            "simulator does not support opt-in external forces",
        )
        simulator.clear_external_forces()
        wp.synchronize()
        position_tensor = torch.as_tensor(
            position,
            dtype=torch.float32,
            device=self._provider._device,
        ).contiguous()
        velocity_tensor = torch.as_tensor(
            velocity,
            dtype=torch.float32,
            device=self._provider._device,
        ).contiguous()
        position_wp = wp.from_torch(
            position_tensor,
            dtype=wp.vec3,
            requires_grad=False,
        )
        velocity_wp = wp.from_torch(
            velocity_tensor,
            dtype=wp.vec3,
            requires_grad=False,
        )
        simulator.set_init_state(position_wp, velocity_wp)
        wp.synchronize()
        future = []
        try:
            for offset, frame in enumerate(range(start_frame, stop_frame)):
                simulator.set_external_forces(schedule[offset])
                wp.synchronize()
                simulator.set_controller_target(frame, pure_inference=True)
                if simulator.object_collision_flag:
                    simulator.update_collision_graph()
                wp.capture_launch(simulator.forward_graph)
                wp.synchronize()
                next_position = (
                    wp.to_torch(simulator.wp_states[-1].wp_x)
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
                future.append(next_position)
                simulator.set_init_state(
                    simulator.wp_states[-1].wp_x,
                    simulator.wp_states[-1].wp_v,
                )
        finally:
            simulator.clear_external_forces()
            wp.synchronize()
        return np.stack(future)
