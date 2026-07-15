"""Generic official-Warp backend for the locked Deform360 replication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from .deform360_phystwin_feasibility import (
    WarpRopeCandidate,
    WarpRopeFeasibilityConfig,
    deform360_xyz_to_warp_xzy,
)
from .deform360_replication_graph import Deform360SparseGraph


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Deform360WarpForecastCase:
    """One prefix-anchored sparse-graph forecast with known controller motion."""

    episode_id: str
    graph: Deform360SparseGraph
    controller_positions_m: np.ndarray
    contact_active: np.ndarray
    contact_node_indices: tuple[int, ...]
    contact_rest_lengths_m: np.ndarray
    dt_seconds: float
    initial_velocities_m_s: np.ndarray | None = None
    object_rest_lengths_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        controllers = np.asarray(self.controller_positions_m, dtype=np.float64)
        active = np.asarray(self.contact_active, dtype=bool)
        rest = np.asarray(self.contact_rest_lengths_m, dtype=np.float64)
        object_edges = np.asarray(self.graph.spring_edges, dtype=np.int32)
        node_count = len(self.graph.positions_m)
        _require(
            controllers.ndim == 3
            and controllers.shape[2] == 3
            and controllers.shape[0] >= 2,
            "controller positions must have shape (T,C,3)",
        )
        _require(active.shape == controllers.shape[:2], "contact state shape differs")
        _require(
            len(self.contact_node_indices) == controllers.shape[1],
            "contact-node count differs from controller count",
        )
        _require(rest.shape == (controllers.shape[1],), "contact-rest count differs")
        _require(
            all(0 <= int(node) < node_count for node in self.contact_node_indices),
            "contact node is outside the object graph",
        )
        _require(
            np.all(np.isfinite(controllers))
            and np.all(np.isfinite(rest))
            and np.all(rest > 0.0),
            "controller trajectory and contact rest lengths must be finite",
        )
        _require(self.dt_seconds > 0.0, "forecast interval must be positive")
        if self.initial_velocities_m_s is None:
            velocities = np.zeros_like(self.graph.positions_m)
        else:
            velocities = np.asarray(self.initial_velocities_m_s, dtype=np.float64)
            _require(
                velocities.shape == self.graph.positions_m.shape
                and np.all(np.isfinite(velocities)),
                "initial velocities differ from the graph",
            )
        if self.object_rest_lengths_m is None:
            object_rest = np.linalg.norm(
                self.graph.positions_m[object_edges[:, 1]]
                - self.graph.positions_m[object_edges[:, 0]],
                axis=1,
            )
        else:
            object_rest = np.asarray(self.object_rest_lengths_m, dtype=np.float64)
            _require(
                object_rest.shape == (len(object_edges),)
                and np.all(np.isfinite(object_rest))
                and np.all(object_rest > 1e-6),
                "object rest lengths differ from the graph",
            )
        for name, values in (
            ("controller_positions_m", controllers),
            ("contact_active", active),
            ("contact_rest_lengths_m", rest),
            ("initial_velocities_m_s", velocities),
            ("object_rest_lengths_m", object_rest),
        ):
            copied = values.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def symmetric_chamfer_distance_m(
    reference_points_m: np.ndarray, prediction_points_m: np.ndarray
) -> float:
    """Symmetric mean point-set Chamfer distance in metres."""

    reference = np.asarray(reference_points_m, dtype=np.float64)
    prediction = np.asarray(prediction_points_m, dtype=np.float64)
    _require(
        reference.ndim == prediction.ndim == 2
        and reference.shape[1] == prediction.shape[1] == 3,
        "Chamfer inputs must have shape (N,3) and (M,3)",
    )
    _require(len(reference) and len(prediction), "Chamfer point sets must be nonempty")
    _require(
        np.all(np.isfinite(reference)) and np.all(np.isfinite(prediction)),
        "Chamfer point sets must be finite",
    )
    difference = reference[:, None, :] - prediction[None, :, :]
    distances = np.linalg.norm(difference, axis=2)
    return 0.5 * (
        float(np.mean(np.min(distances, axis=1)))
        + float(np.mean(np.min(distances, axis=0)))
    )


def sparse_trajectory_chamfer_m(
    reference_point_sets_m: Sequence[np.ndarray], prediction_m: np.ndarray
) -> dict[str, object]:
    """Score a fixed-node prediction against variable-size visual hulls."""

    prediction = np.asarray(prediction_m, dtype=np.float64)
    _require(
        prediction.ndim == 3 and prediction.shape[2] == 3,
        "prediction must have shape (T,N,3)",
    )
    _require(
        len(reference_point_sets_m) == len(prediction),
        "reference and prediction frame counts differ",
    )
    if not np.all(np.isfinite(prediction)):
        return {
            "mean_m": float("inf"),
            "late_mean_m": float("inf"),
            "per_frame_m": [float("inf")] * len(prediction),
        }
    per_frame = np.asarray(
        [
            symmetric_chamfer_distance_m(reference, forecast)
            for reference, forecast in zip(
                reference_point_sets_m, prediction, strict=True
            )
        ],
        dtype=np.float64,
    )
    late_start = max(0, (2 * len(per_frame)) // 3)
    return {
        "mean_m": float(np.mean(per_frame)),
        "late_mean_m": float(np.mean(per_frame[late_start:])),
        "per_frame_m": per_frame.tolist(),
    }


def sparse_graph_strain_summary(
    graph: Deform360SparseGraph,
    prediction_m: np.ndarray,
    *,
    rest_lengths_m: np.ndarray | None = None,
    spring_family: int | None = None,
) -> dict[str, float]:
    """Return absolute relative object-spring strain summaries."""

    prediction = np.asarray(prediction_m, dtype=np.float64)
    if not np.all(np.isfinite(prediction)):
        return {"p95": float("inf"), "p99": float("inf"), "maximum": float("inf")}
    edges = graph.spring_edges
    if rest_lengths_m is None:
        rest = np.linalg.norm(
            graph.positions_m[edges[:, 1]] - graph.positions_m[edges[:, 0]], axis=1
        )
    else:
        rest = np.asarray(rest_lengths_m, dtype=np.float64)
        _require(
            rest.shape == (len(edges),)
            and np.all(np.isfinite(rest))
            and np.all(rest > 1e-6),
            "strain rest lengths differ from the graph",
        )
    if spring_family is not None:
        _require(spring_family in (0, 1), "strain spring family is invalid")
        selected = graph.spring_families == spring_family
        _require(np.any(selected), "strain spring family is empty")
        edges = edges[selected]
        rest = rest[selected]
    lengths = np.linalg.norm(
        prediction[:, edges[:, 1]] - prediction[:, edges[:, 0]], axis=2
    )
    relative = np.abs(lengths / rest[None] - 1.0)
    return {
        "p95": float(np.quantile(relative, 0.95)),
        "p99": float(np.quantile(relative, 0.99)),
        "maximum": float(np.max(relative)),
    }


def warp_xzy_to_deform360_xyz(
    values: np.ndarray,
    *,
    initial_support_height_m: float,
    clearance_m: float,
) -> np.ndarray:
    """Invert :func:`deform360_xyz_to_warp_xzy`."""

    points = np.asarray(values, dtype=np.float64)
    _require(points.shape[-1] == 3, "coordinates must end in dimension three")
    restored = np.empty_like(points)
    restored[..., 0] = points[..., 0]
    restored[..., 1] = (
        points[..., 2] + float(initial_support_height_m) - float(clearance_m)
    )
    restored[..., 2] = points[..., 1]
    return restored


class OfficialWarpSparseGraphRunner:
    """Run an arbitrary sparse object graph through the pinned PhysTwin backend."""

    def __init__(
        self,
        official_repo: str | Path,
        case: Deform360WarpForecastCase,
        config: WarpRopeFeasibilityConfig,
        *,
        device: str = "cuda:0",
    ) -> None:
        try:
            import torch
            import warp as wp
        except ImportError as error:  # pragma: no cover - GPU integration
            raise RuntimeError(
                "official-Warp replication requires torch and warp"
            ) from error
        from bayesian_phystwin._phystwin_warp_backend import (
            load_official_spring_mass_module,
            make_reliability_simulator_class,
        )
        from bayesian_phystwin.phystwin_refit import build_phystwin_track_objective

        self.torch = torch
        self.wp = wp
        self.case = case
        self.config = config
        self.device = device
        wp.set_device(device)
        self.node_count = len(case.graph.positions_m)
        self.support_height_m = float(np.min(case.graph.positions_m[:, 1]))
        self.initial_positions = deform360_xyz_to_warp_xzy(
            case.graph.positions_m,
            initial_support_height_m=self.support_height_m,
            clearance_m=config.initial_ground_clearance_m,
        ).astype(np.float32)
        self.controllers = deform360_xyz_to_warp_xzy(
            case.controller_positions_m,
            initial_support_height_m=self.support_height_m,
            clearance_m=config.initial_ground_clearance_m,
        ).astype(np.float32)
        warp_velocities = np.empty_like(case.initial_velocities_m_s)
        warp_velocities[..., 0] = case.initial_velocities_m_s[..., 0]
        warp_velocities[..., 1] = case.initial_velocities_m_s[..., 2]
        warp_velocities[..., 2] = case.initial_velocities_m_s[..., 1]
        self.initial_velocities = warp_velocities.astype(np.float32)

        stretch_edges = case.graph.spring_edges[case.graph.spring_families == 0]
        bend_edges = case.graph.spring_edges[case.graph.spring_families == 1]
        object_edges = np.concatenate((stretch_edges, bend_edges), axis=0)
        control_edges = np.asarray(
            [
                (self.node_count + controller, int(node))
                for controller, node in enumerate(case.contact_node_indices)
            ],
            dtype=np.int32,
        )
        self.springs = np.concatenate((object_edges, control_edges), axis=0)
        self.stretch_count = len(stretch_edges)
        self.bend_count = len(bend_edges)
        self.num_object_springs = len(object_edges)
        object_rest = case.object_rest_lengths_m.astype(np.float32)
        self.rest_lengths = np.concatenate(
            (object_rest, case.contact_rest_lengths_m.astype(np.float32))
        ).astype(np.float32)
        vertices = np.concatenate((self.initial_positions, self.controllers[0]), axis=0)
        masses = np.concatenate(
            (
                case.graph.masses.astype(np.float32),
                np.ones(len(control_edges), np.float32),
            )
        )

        runtime_config = SimpleNamespace(
            device=device,
            use_graph=True,
            data_type="real",
            collision_learn=False,
            chamfer_weight=0.0,
            track_weight=0.0,
            acc_weight=0.0,
        )
        official = load_official_spring_mass_module(
            Path(official_repo).resolve(), runtime_config=runtime_config
        )
        simulator_class = make_reliability_simulator_class(official)
        frame_count = len(self.controllers)
        visible = np.ones((frame_count, self.node_count), dtype=bool)
        motion_valid = np.ones((frame_count - 1, self.node_count), dtype=bool)
        objective = build_phystwin_track_objective(
            visible, motion_valid, variant="hard"
        )

        def tensor(values: np.ndarray, dtype):
            return torch.as_tensor(values, dtype=dtype, device=device).contiguous()

        dummy_gt = np.repeat(self.initial_positions[None], frame_count, axis=0)
        self.simulator = simulator_class(
            tensor(vertices, torch.float32),
            tensor(self.springs, torch.int32),
            tensor(self.rest_lengths, torch.float32),
            tensor(masses, torch.float32),
            dt=case.dt_seconds / config.substeps,
            num_substeps=config.substeps,
            spring_Y=1000.0,
            collide_elas=config.ground_elasticity,
            collide_fric=0.3,
            dashpot_damping=config.dashpot_damping,
            drag_damping=config.drag_damping,
            collide_object_elas=0.0,
            collide_object_fric=0.0,
            collision_dist=0.01,
            num_object_points=self.node_count,
            num_surface_points=self.node_count,
            num_original_points=self.node_count,
            controller_points=tensor(self.controllers, torch.float32),
            reverse_z=False,
            spring_Y_min=0.0,
            spring_Y_max=1e5,
            gt_object_points=tensor(dummy_gt, torch.float32),
            gt_object_visibilities=tensor(visible.astype(np.int32), torch.int32),
            gt_object_motions_valid=tensor(motion_valid.astype(np.int32), torch.int32),
            self_collision=False,
            disable_backward=True,
            objective=objective,
            observation_variance=1e-4,
            outlier_variance_multiplier=100.0,
            spring_parameterization="dense",
            num_object_springs=self.num_object_springs,
            deterministic_spring_forces=True,
        )
        self.initial_tensor = tensor(self.initial_positions, torch.float32)
        self.velocity_tensor = tensor(self.initial_velocities, torch.float32)
        self.wp_initial = wp.from_torch(
            self.initial_tensor, dtype=wp.vec3, requires_grad=False
        )
        self.wp_velocity = wp.from_torch(
            self.velocity_tensor, dtype=wp.vec3, requires_grad=False
        )
        wp.synchronize()

    def _spring_log_y(self, candidate: WarpRopeCandidate, active: tuple[bool, ...]):
        values = np.empty(len(self.springs), dtype=np.float32)
        values[: self.stretch_count] = candidate.stretch_spring_y
        values[self.stretch_count : self.num_object_springs] = candidate.bend_spring_y
        for controller, enabled in enumerate(active):
            values[self.num_object_springs + controller] = (
                candidate.controller_spring_y
                if enabled
                else self.config.inactive_controller_spring_y
            )
        return self.torch.log(
            self.torch.as_tensor(values, dtype=self.torch.float32, device=self.device)
        ).contiguous()

    def rollout(
        self,
        candidate: WarpRopeCandidate,
        *,
        contact_active: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a deterministic ``(T,N,3)`` prediction in Deform360 xyz."""

        torch = self.torch
        wp = self.wp
        active_values = (
            self.case.contact_active
            if contact_active is None
            else np.asarray(contact_active, dtype=bool)
        )
        _require(
            active_values.shape == self.case.contact_active.shape,
            "rollout contact state shape differs",
        )
        friction = torch.as_tensor(
            [candidate.ground_friction], dtype=torch.float32, device=self.device
        ).contiguous()
        elasticity = torch.as_tensor(
            [self.config.ground_elasticity],
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        self.simulator.set_collide(elasticity, friction)
        self.simulator.set_init_state(
            self.wp_initial, self.wp_velocity, pure_inference=True
        )
        wp.synchronize()
        trajectory = [self.initial_positions.astype(np.float64)]
        previous_active: tuple[bool, ...] | None = None
        for frame in range(1, len(self.controllers)):
            active = tuple(map(bool, active_values[frame]))
            if active != previous_active:
                self.simulator.set_reference_spring_y(
                    self._spring_log_y(candidate, active)
                )
                previous_active = active
            self.simulator.set_controller_target(frame, pure_inference=True)
            wp.capture_launch(self.simulator.forward_graph)
            wp.synchronize()
            position = (
                wp.to_torch(self.simulator.wp_states[-1].wp_x)
                .detach()
                .cpu()
                .numpy()
                .copy()[: self.node_count]
            )
            trajectory.append(position.astype(np.float64))
            if not np.all(np.isfinite(position)):
                trajectory.extend(
                    [
                        np.full_like(self.initial_positions, np.nan, dtype=np.float64)
                        for _ in range(len(self.controllers) - len(trajectory))
                    ]
                )
                break
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
                pure_inference=True,
            )
        warp_prediction = np.stack(trajectory)
        return warp_xzy_to_deform360_xyz(
            warp_prediction,
            initial_support_height_m=self.support_height_m,
            clearance_m=self.config.initial_ground_clearance_m,
        )


__all__ = [
    "Deform360WarpForecastCase",
    "OfficialWarpSparseGraphRunner",
    "sparse_graph_strain_summary",
    "sparse_trajectory_chamfer_m",
    "symmetric_chamfer_distance_m",
    "warp_xzy_to_deform360_xyz",
]
