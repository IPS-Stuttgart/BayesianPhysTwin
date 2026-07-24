"""Frame-wise equivariant-force injection into the pinned PhysTwin Warp path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .phystwin_discrepancy_localization import _rollout_state_segment
from .phystwin_equivariant_force import maximum_node_force_sim
from .phystwin_state_injection import _state_numpy


@dataclass(frozen=True)
class ForceRolloutDiagnostics:
    """Target-free diagnostics recorded for one force-conditioned rollout."""

    admission_weight: float
    force_unit_contract: str
    force_scale_sim: float
    maximum_force_sim: float
    mean_force_sim: float
    active_force_frames: int
    frame_count: int


def controller_attachment_matrix(
    springs: np.ndarray,
    *,
    num_object_nodes: int,
    num_control_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map control points to attached object nodes from the PhysTwin graph."""

    edges = np.asarray(springs, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("springs must have shape (S, 2)")
    if num_object_nodes < 1 or num_control_nodes < 1:
        raise ValueError("object and control node counts must be positive")
    total = num_object_nodes + num_control_nodes
    if np.any(edges < 0) or np.any(edges >= total):
        raise ValueError("spring endpoint is outside the combined graph")
    matrix = np.zeros((num_object_nodes, num_control_nodes), dtype=float)
    for first, second in edges:
        first_object = first < num_object_nodes
        second_object = second < num_object_nodes
        if first_object == second_object:
            continue
        object_node = int(first if first_object else second)
        control_node = int(second if first_object else first) - num_object_nodes
        matrix[object_node, control_node] += 1.0
    support = np.sum(matrix, axis=1)
    selected = support > 0.0
    matrix[selected] /= support[selected, None]
    support = selected.astype(float)
    matrix.setflags(write=False)
    support.setflags(write=False)
    return matrix, support


def controller_conditioning_fields(
    positions_m: np.ndarray,
    controller_previous_m: np.ndarray,
    controller_target_m: np.ndarray,
    attachment_matrix: np.ndarray,
    *,
    frame_dt_s: float,
    support_prior: np.ndarray,
    activity_speed_mps: float,
) -> dict[str, np.ndarray | float]:
    """Build residual-independent action/contact cues for one rollout frame."""

    positions = np.asarray(positions_m, dtype=float)
    previous = np.asarray(controller_previous_m, dtype=float)
    target = np.asarray(controller_target_m, dtype=float)
    attachment = np.asarray(attachment_matrix, dtype=float)
    prior = np.asarray(support_prior, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (N, 3)")
    if previous.shape != target.shape or previous.ndim != 2 or previous.shape[1] != 3:
        raise ValueError("controller arrays must share shape (C, 3)")
    if attachment.shape != (len(positions), len(target)):
        raise ValueError("attachment_matrix must have shape (N, C)")
    if prior.shape != (len(positions),) or np.any((prior < 0.0) | (prior > 1.0)):
        raise ValueError("support_prior must be an N-vector in [0, 1]")
    if frame_dt_s <= 0.0 or activity_speed_mps <= 0.0:
        raise ValueError("timing and activity scales must be positive")
    if not all(
        np.all(np.isfinite(values))
        for values in (positions, previous, target, attachment, prior)
    ):
        raise ValueError("conditioning inputs must be finite")

    action_support = (np.sum(np.abs(attachment), axis=1) > 0.0).astype(float)
    mapped_target = attachment @ target
    controller_velocity = (target - previous) / frame_dt_s
    mapped_velocity = attachment @ controller_velocity
    control_displacement = (
        mapped_target - positions
    ) * action_support[:, None]
    control_velocity = mapped_velocity * action_support[:, None]
    external_support = np.maximum(action_support, prior)
    activity = float(
        np.clip(
            np.max(np.linalg.norm(controller_velocity, axis=1), initial=0.0)
            / activity_speed_mps,
            0.0,
            1.0,
        )
    )
    return {
        "control_displacement_m": control_displacement,
        "control_velocity_mps": control_velocity,
        "action_support": action_support,
        "external_support": external_support,
        "action_activity": activity,
    }


def predict_equivariant_force(
    model: Any,
    torch: Any,
    *,
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    rest_positions_m: np.ndarray,
    object_edges: np.ndarray,
    rest_lengths_m: np.ndarray,
    conditioning: dict[str, np.ndarray | float],
    gravity_mps2: np.ndarray,
    force_scale_sim: float,
    regime_probabilities: np.ndarray,
    latent: np.ndarray,
    admission_weight: float,
    device: str,
) -> np.ndarray:
    """Evaluate one bounded simulator-force field without residual cues."""

    if not 0.0 <= admission_weight <= 1.0:
        raise ValueError("admission_weight must lie in [0, 1]")
    if force_scale_sim <= 0.0 or not np.isfinite(force_scale_sim):
        raise ValueError("force_scale_sim must be positive and finite")
    if admission_weight == 0.0:
        return np.zeros_like(np.asarray(positions_m, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        force = model(
            positions_m=torch.as_tensor(
                positions_m, dtype=torch.float32, device=device
            ),
            velocities_mps=torch.as_tensor(
                velocities_mps, dtype=torch.float32, device=device
            ),
            rest_positions_m=torch.as_tensor(
                rest_positions_m, dtype=torch.float32, device=device
            ),
            edges=torch.as_tensor(object_edges, dtype=torch.long, device=device),
            rest_lengths_m=torch.as_tensor(
                rest_lengths_m, dtype=torch.float32, device=device
            ),
            control_displacement_m=torch.as_tensor(
                conditioning["control_displacement_m"],
                dtype=torch.float32,
                device=device,
            ),
            control_velocity_mps=torch.as_tensor(
                conditioning["control_velocity_mps"],
                dtype=torch.float32,
                device=device,
            ),
            action_support=torch.as_tensor(
                conditioning["action_support"],
                dtype=torch.float32,
                device=device,
            ),
            external_support=torch.as_tensor(
                conditioning["external_support"],
                dtype=torch.float32,
                device=device,
            ),
            gravity_mps2=torch.as_tensor(
                gravity_mps2, dtype=torch.float32, device=device
            ),
            force_scale_sim=force_scale_sim,
            action_activity=float(conditioning["action_activity"]),
            regime_probabilities=torch.as_tensor(
                regime_probabilities, dtype=torch.float32, device=device
            ),
            latent=torch.as_tensor(latent, dtype=torch.float32, device=device),
            admission_weight=admission_weight,
        )
    result = np.ascontiguousarray(force.detach().cpu().numpy(), dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("equivariant force policy produced non-finite values")
    return result


def predict_equivariant_force_ensemble(
    models: tuple[Any, ...] | list[Any],
    torch: Any,
    *,
    latents: tuple[np.ndarray, ...] | list[np.ndarray],
    positions_m: np.ndarray,
    velocities_mps: np.ndarray,
    rest_positions_m: np.ndarray,
    object_edges: np.ndarray,
    rest_lengths_m: np.ndarray,
    conditioning: dict[str, np.ndarray | float],
    gravity_mps2: np.ndarray,
    force_scale_sim: float,
    regime_probabilities: np.ndarray,
    admission_weight: float,
    device: str,
) -> np.ndarray:
    """Average paired seed-model force fields before one Warp step."""

    members = tuple(models)
    latent_values = tuple(latents)
    if not members or len(members) != len(latent_values):
        raise ValueError("models and latents must be nonempty paired sequences")
    if admission_weight == 0.0:
        return np.zeros_like(np.asarray(positions_m, dtype=np.float32))
    predictions = [
        predict_equivariant_force(
            model,
            torch,
            positions_m=positions_m,
            velocities_mps=velocities_mps,
            rest_positions_m=rest_positions_m,
            object_edges=object_edges,
            rest_lengths_m=rest_lengths_m,
            conditioning=conditioning,
            gravity_mps2=gravity_mps2,
            force_scale_sim=force_scale_sim,
            regime_probabilities=regime_probabilities,
            latent=latent,
            admission_weight=admission_weight,
            device=device,
        )
        for model, latent in zip(members, latent_values, strict=True)
    ]
    result = np.mean(
        np.stack(predictions).astype(np.float64),
        axis=0,
    ).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("equivariant-force ensemble produced non-finite values")
    return np.ascontiguousarray(result)


def _rollout_equivariant_force_policy_segment(
    simulator: Any,
    torch: Any,
    wp: Any,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    controller_points_m: np.ndarray,
    attachment_matrix: np.ndarray,
    support_prior: np.ndarray,
    regime_probabilities: np.ndarray,
    force_scale_sim: float,
    frame_dt_s: float,
    activity_speed_mps: float,
    admission_weight: float,
    device: str,
    force_predictor: Callable[
        [np.ndarray, np.ndarray, dict[str, np.ndarray | float], np.ndarray],
        np.ndarray,
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ForceRolloutDiagnostics]:
    """Run one force policy through the shared official-Warp stepping path."""

    controllers = np.asarray(controller_points_m, dtype=float)
    regimes = np.asarray(regime_probabilities, dtype=float)
    if not 0 <= start_frame < stop_frame <= len(controllers):
        raise ValueError("rollout interval is outside the controller trajectory")
    if regimes.shape[0] < stop_frame:
        raise ValueError("regime probabilities do not cover the rollout")
    if force_scale_sim <= 0.0 or not np.isfinite(force_scale_sim):
        raise ValueError("force_scale_sim must be positive and finite")
    if admission_weight == 0.0:
        simulator.clear_external_forces()
        positions, velocities = _rollout_state_segment(
            simulator,
            torch,
            wp,
            position_m,
            velocity_mps,
            start_frame=start_frame,
            stop_frame=stop_frame,
            device=device,
        )
        force_history = np.zeros(
            (stop_frame - start_frame, len(position_m), 3), dtype=np.float32
        )
        return (
            positions,
            velocities,
            force_history,
            ForceRolloutDiagnostics(
                admission_weight=0.0,
                force_unit_contract=(
                    "warp_simulator_generalized_force_not_newtons"
                ),
                force_scale_sim=float(force_scale_sim),
                maximum_force_sim=0.0,
                mean_force_sim=0.0,
                active_force_frames=0,
                frame_count=stop_frame - start_frame,
            ),
        )

    position_tensor = torch.as_tensor(
        np.array(position_m, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    ).contiguous()
    velocity_tensor = torch.as_tensor(
        np.array(velocity_mps, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    ).contiguous()
    simulator.set_init_state(
        wp.from_torch(position_tensor, dtype=wp.vec3, requires_grad=False),
        wp.from_torch(velocity_tensor, dtype=wp.vec3, requires_grad=False),
    )
    wp.synchronize()
    positions = [np.asarray(position_m, dtype=np.float32).copy()]
    velocities = [np.asarray(velocity_mps, dtype=np.float32).copy()]
    force_history = []
    current_position = positions[0]
    current_velocity = velocities[0]
    try:
        for frame in range(start_frame, stop_frame):
            previous_frame = max(frame - 1, 0)
            conditioning = controller_conditioning_fields(
                current_position,
                controllers[previous_frame],
                controllers[frame],
                attachment_matrix,
                frame_dt_s=frame_dt_s,
                support_prior=support_prior,
                activity_speed_mps=activity_speed_mps,
            )
            force = force_predictor(
                current_position,
                current_velocity,
                conditioning,
                regimes[frame],
            )
            simulator.set_external_forces(
                torch.as_tensor(force, dtype=torch.float32, device=device)
            )
            simulator.set_controller_target(frame, pure_inference=True)
            if simulator.object_collision_flag:
                simulator.update_collision_graph()
            wp.capture_launch(simulator.forward_graph)
            wp.synchronize()
            current_position, current_velocity = _state_numpy(
                simulator.wp_states[-1], wp
            )
            if not np.all(np.isfinite(current_position)) or not np.all(
                np.isfinite(current_velocity)
            ):
                raise RuntimeError(f"non-finite Warp state at frame {frame}")
            positions.append(current_position)
            velocities.append(current_velocity)
            force_history.append(force)
            simulator.set_init_state(
                simulator.wp_states[-1].wp_x,
                simulator.wp_states[-1].wp_v,
            )
    finally:
        simulator.clear_external_forces()
        wp.synchronize()

    forces = np.stack(force_history)
    norms = np.linalg.norm(forces, axis=-1)
    diagnostics = ForceRolloutDiagnostics(
        admission_weight=float(admission_weight),
        force_unit_contract="warp_simulator_generalized_force_not_newtons",
        force_scale_sim=float(force_scale_sim),
        maximum_force_sim=maximum_node_force_sim(forces),
        mean_force_sim=float(np.mean(norms)),
        active_force_frames=int(np.sum(np.max(norms, axis=1) > 0.0)),
        frame_count=len(forces),
    )
    return np.stack(positions), np.stack(velocities), forces, diagnostics


def rollout_equivariant_force_segment(
    simulator: Any,
    torch: Any,
    wp: Any,
    model: Any,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    rest_positions_m: np.ndarray,
    object_edges: np.ndarray,
    rest_lengths_m: np.ndarray,
    controller_points_m: np.ndarray,
    attachment_matrix: np.ndarray,
    support_prior: np.ndarray,
    regime_probabilities: np.ndarray,
    latent: np.ndarray,
    gravity_mps2: np.ndarray,
    force_scale_sim: float,
    frame_dt_s: float,
    activity_speed_mps: float,
    admission_weight: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ForceRolloutDiagnostics]:
    """Run official captured Warp steps with one force model."""

    def force_predictor(position, velocity, conditioning, regime):
        return predict_equivariant_force(
            model,
            torch,
            positions_m=position,
            velocities_mps=velocity,
            rest_positions_m=rest_positions_m,
            object_edges=object_edges,
            rest_lengths_m=rest_lengths_m,
            conditioning=conditioning,
            gravity_mps2=gravity_mps2,
            force_scale_sim=force_scale_sim,
            regime_probabilities=regime,
            latent=latent,
            admission_weight=admission_weight,
            device=device,
        )

    return _rollout_equivariant_force_policy_segment(
        simulator,
        torch,
        wp,
        position_m,
        velocity_mps,
        start_frame=start_frame,
        stop_frame=stop_frame,
        controller_points_m=controller_points_m,
        attachment_matrix=attachment_matrix,
        support_prior=support_prior,
        regime_probabilities=regime_probabilities,
        force_scale_sim=force_scale_sim,
        frame_dt_s=frame_dt_s,
        activity_speed_mps=activity_speed_mps,
        admission_weight=admission_weight,
        device=device,
        force_predictor=force_predictor,
    )


def rollout_equivariant_force_ensemble_segment(
    simulator: Any,
    torch: Any,
    wp: Any,
    models: tuple[Any, ...] | list[Any],
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    rest_positions_m: np.ndarray,
    object_edges: np.ndarray,
    rest_lengths_m: np.ndarray,
    controller_points_m: np.ndarray,
    attachment_matrix: np.ndarray,
    support_prior: np.ndarray,
    regime_probabilities: np.ndarray,
    latents: tuple[np.ndarray, ...] | list[np.ndarray],
    gravity_mps2: np.ndarray,
    force_scale_sim: float,
    frame_dt_s: float,
    activity_speed_mps: float,
    admission_weight: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ForceRolloutDiagnostics]:
    """Run the frozen per-frame arithmetic seed ensemble through Warp."""

    def force_predictor(position, velocity, conditioning, regime):
        return predict_equivariant_force_ensemble(
            models,
            torch,
            latents=latents,
            positions_m=position,
            velocities_mps=velocity,
            rest_positions_m=rest_positions_m,
            object_edges=object_edges,
            rest_lengths_m=rest_lengths_m,
            conditioning=conditioning,
            gravity_mps2=gravity_mps2,
            force_scale_sim=force_scale_sim,
            regime_probabilities=regime,
            admission_weight=admission_weight,
            device=device,
        )

    return _rollout_equivariant_force_policy_segment(
        simulator,
        torch,
        wp,
        position_m,
        velocity_mps,
        start_frame=start_frame,
        stop_frame=stop_frame,
        controller_points_m=controller_points_m,
        attachment_matrix=attachment_matrix,
        support_prior=support_prior,
        regime_probabilities=regime_probabilities,
        force_scale_sim=force_scale_sim,
        frame_dt_s=frame_dt_s,
        activity_speed_mps=activity_speed_mps,
        admission_weight=admission_weight,
        device=device,
        force_predictor=force_predictor,
    )
