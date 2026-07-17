"""Source-only training data for the Deform360 Bayesian residual track."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .deform360_independent_source import EXPECTED_INDEPENDENT_SOURCE_EPISODES


ControllerSurfaceProvider = Callable[[float, np.ndarray], np.ndarray]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _deterministic_farthest_points(
    points: np.ndarray, count: int
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 3, "points must be (N,3)")
    _require(1 <= count <= len(values), "invalid farthest-point count")
    center = np.mean(values, axis=0)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmax(np.sum(np.square(values - center), axis=1)))
    minimum_squared_distance = np.sum(
        np.square(values - values[selected[0]]), axis=1
    )
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum_squared_distance))
        candidate_distance = np.sum(
            np.square(values - values[selected[index]]), axis=1
        )
        minimum_squared_distance = np.minimum(
            minimum_squared_distance, candidate_distance
        )
    return selected


def knn_edge_index(points: np.ndarray, neighbor_count: int = 12) -> np.ndarray:
    """Build deterministic directed KNN edges for one material point set."""

    values = np.asarray(points, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 3, "points must be (N,3)")
    _require(1 <= neighbor_count < len(values), "invalid neighbor count")
    squared_distance = np.sum(
        np.square(values[:, None] - values[None]), axis=-1
    )
    np.fill_diagonal(squared_distance, np.inf)
    neighbors = np.argpartition(
        squared_distance, kth=neighbor_count - 1, axis=1
    )[:, :neighbor_count]
    row = np.arange(len(values))[:, None]
    order = np.argsort(squared_distance[row, neighbors], axis=1)
    neighbors = np.take_along_axis(neighbors, order, axis=1)
    target = np.repeat(np.arange(len(values)), neighbor_count)
    source = neighbors.reshape(-1)
    return np.stack((source, target)).astype(np.int64, copy=False)


def spatial_cluster_ids(points: np.ndarray, voxel_size_m: float = 0.03) -> np.ndarray:
    """Group nearby points so dense correlated samples have bounded evidence."""

    values = np.asarray(points, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 3, "points must be (N,3)")
    _require(np.isfinite(voxel_size_m) and voxel_size_m > 0.0, "invalid voxel size")
    coordinate = np.floor((values - np.min(values, axis=0)) / voxel_size_m).astype(
        np.int64
    )
    _, inverse = np.unique(coordinate, axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def contact_probabilities_from_state(
    positions_m: np.ndarray,
    controller_positions_m: np.ndarray,
    closure_probability: np.ndarray,
    *,
    proximity_scale_m: float = 0.03,
    relative_to_nearest: bool = True,
) -> np.ndarray:
    """Construct a causal soft contact proposal from state and known actuation."""

    positions = np.asarray(positions_m, dtype=np.float64)
    controllers = np.asarray(controller_positions_m, dtype=np.float64)
    closure = np.asarray(closure_probability, dtype=np.float64)
    _require(positions.ndim == 2 and positions.shape[1] == 3, "positions must be (N,3)")
    _require(
        controllers.ndim == 2 and controllers.shape[1] == 3,
        "controllers must be (C,3)",
    )
    _require(closure.shape == (len(controllers),), "closure shape differs")
    _require(
        np.all((closure >= 0.0) & (closure <= 1.0)), "closure is outside [0,1]"
    )
    _require(
        np.isfinite(proximity_scale_m) and proximity_scale_m > 0.0,
        "invalid proximity scale",
    )
    squared_distance = np.sum(
        np.square(positions[:, None] - controllers[None]), axis=-1
    )
    if relative_to_nearest:
        distance = np.sqrt(squared_distance)
        distance = distance - np.min(distance, axis=0, keepdims=True)
        squared_distance = np.square(distance)
    proximity = np.exp(-0.5 * squared_distance / proximity_scale_m**2)
    return proximity * closure[None]


def _controller_surface_trajectories(
    transforms: np.ndarray,
    openings: np.ndarray,
    provider: ControllerSurfaceProvider,
    points_per_gripper: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample fixed material locations on each released gripper surface."""

    _require(points_per_gripper >= 1, "controller surface count must be positive")
    frame_count, gripper_count = transforms.shape[:2]
    surface_blocks: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    for gripper_index in range(gripper_count):
        first_surface = np.asarray(
            provider(float(openings[0, gripper_index]), transforms[0, gripper_index]),
            dtype=np.float64,
        )
        _require(
            first_surface.ndim == 2
            and first_surface.shape[1] == 3
            and len(first_surface) >= points_per_gripper
            and np.all(np.isfinite(first_surface)),
            "controller surface provider returned invalid points",
        )
        selected = _deterministic_farthest_points(first_surface, points_per_gripper)
        frames = []
        for frame_index in range(frame_count):
            surface = np.asarray(
                provider(
                    float(openings[frame_index, gripper_index]),
                    transforms[frame_index, gripper_index],
                ),
                dtype=np.float64,
            )
            _require(
                surface.shape == first_surface.shape and np.all(np.isfinite(surface)),
                "controller surface identities changed across frames",
            )
            frames.append(surface[selected])
        surface_blocks.append(np.stack(frames, axis=0))
        group_ids.append(
            np.full(points_per_gripper, gripper_index, dtype=np.int64)
        )
    return np.concatenate(surface_blocks, axis=1), np.concatenate(group_ids)


@dataclass(frozen=True)
class Deform360ResidualSourceEpisode:
    object_id: str
    episode_id: int
    positions_m: np.ndarray
    observed_velocities_mps: np.ndarray
    physics_positions_m: np.ndarray
    physics_prior_kind: str
    prior_reliability: np.ndarray
    controller_positions_m: np.ndarray
    controller_velocities_mps: np.ndarray
    closure_probability: np.ndarray
    controller_group_ids: np.ndarray
    controller_geometry: str
    edge_index: np.ndarray
    cluster_ids: np.ndarray
    frame_interval_s: float
    physics_response_scale: float = 0.0
    physics_reference_response_scale: float = 0.9

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float32)
        velocities = np.asarray(self.observed_velocities_mps, dtype=np.float32)
        physics_positions = np.asarray(self.physics_positions_m, dtype=np.float32)
        reliability = np.asarray(self.prior_reliability, dtype=np.float32)
        controllers = np.asarray(self.controller_positions_m, dtype=np.float32)
        controller_velocities = np.asarray(
            self.controller_velocities_mps, dtype=np.float32
        )
        closure = np.asarray(self.closure_probability, dtype=np.float32)
        controller_groups = np.asarray(self.controller_group_ids, dtype=np.int64)
        edges = np.asarray(self.edge_index, dtype=np.int64)
        clusters = np.asarray(self.cluster_ids, dtype=np.int64)
        _require(
            positions.ndim == 3 and positions.shape[-1] == 3 and len(positions) >= 3,
            "source positions must be (T,N,3)",
        )
        _require(velocities.shape == positions.shape, "source velocities differ")
        _require(
            physics_positions.shape == positions.shape,
            "source physical prediction differs",
        )
        _require(
            self.physics_prior_kind
            in {
                "persistence",
                "sealed_graph_action_support",
                "trusted_sealed_graph_action_support",
            },
            "unsupported physical prior",
        )
        _require(
            np.array_equal(physics_positions[0], positions[0]),
            "physical prediction is not frame-zero anchored",
        )
        _require(reliability.shape == positions.shape[:2], "source reliability differs")
        _require(
            controllers.ndim == 3
            and controllers.shape[0] == len(positions)
            and controllers.shape[-1] == 3,
            "source controllers differ",
        )
        _require(controller_velocities.shape == controllers.shape, "controller velocity differs")
        _require(closure.shape == controllers.shape[:2], "source closure differs")
        _require(
            controller_groups.shape == (controllers.shape[1],),
            "controller group ids differ",
        )
        _require(
            self.controller_geometry in {"end_effector_origins", "gripper_surface"},
            "unsupported controller geometry",
        )
        _require(edges.ndim == 2 and edges.shape[0] == 2, "source edges differ")
        _require(clusters.shape == (positions.shape[1],), "source clusters differ")
        _require(
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
            and np.all(np.isfinite(physics_positions))
            and np.all(np.isfinite(reliability))
            and np.all(np.isfinite(controllers))
            and np.all(np.isfinite(controller_velocities))
            and np.all(np.isfinite(closure)),
            "source episode contains non-finite values",
        )
        _require(
            np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "source reliability is outside [0,1]",
        )
        _require(
            np.all((closure >= 0.0) & (closure <= 1.0)),
            "source closure is outside [0,1]",
        )
        _require(
            np.isfinite(self.frame_interval_s) and self.frame_interval_s > 0.0,
            "invalid frame interval",
        )
        _require(
            np.isfinite(self.physics_response_scale)
            and self.physics_response_scale >= 0.0,
            "invalid physical response scale",
        )
        _require(
            np.isfinite(self.physics_reference_response_scale)
            and self.physics_reference_response_scale > 0.0,
            "invalid reference physical response scale",
        )
        for name, value in (
            ("positions_m", positions),
            ("observed_velocities_mps", velocities),
            ("physics_positions_m", physics_positions),
            ("prior_reliability", reliability),
            ("controller_positions_m", controllers),
            ("controller_velocities_mps", controller_velocities),
            ("closure_probability", closure),
            ("controller_group_ids", controller_groups),
            ("edge_index", edges),
            ("cluster_ids", clusters),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    @property
    def episode_key(self) -> str:
        return f"{self.object_id}/{self.episode_id}"

    def contact_probabilities(self, frame_index: int, positions_m: np.ndarray) -> np.ndarray:
        _require(0 <= frame_index < len(self.positions_m), "frame index is out of range")
        return contact_probabilities_from_state(
            positions_m,
            self.controller_positions_m[frame_index],
            self.closure_probability[frame_index],
            relative_to_nearest=self.controller_geometry == "end_effector_origins",
        )


def _frame_rate_hz(episode_dir: Path) -> float:
    metadata_path = episode_dir / "pcd_clean" / "pcd_clean.meta.json"
    if not metadata_path.is_file():
        return 30.0
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return float(payload.get("parameters", {}).get("frame_rate_hz", 30.0))


def load_deform360_residual_source_episode(
    episode_dir: str | Path,
    *,
    object_id: str,
    episode_id: int,
    maximum_node_count: int = 384,
    neighbor_count: int = 12,
    controller_surface_provider: ControllerSurfaceProvider | None = None,
    controller_points_per_gripper: int = 32,
    physics_prediction_path: str | Path | None = None,
    physics_response_scale: float | None = None,
    physics_reference_response_scale: float = 0.9,
) -> Deform360ResidualSourceEpisode:
    """Load only an already-open episode from the registered source panel."""

    allowed = EXPECTED_INDEPENDENT_SOURCE_EPISODES.get(str(object_id), ())
    _require(
        int(episode_id) in allowed,
        "episode is outside the already-open Bayesian-residual source panel",
    )
    directory = Path(episode_dir).resolve()
    frame_paths = sorted(
        (directory / "pcd_clean").glob("*.npz"), key=lambda path: int(path.stem)
    )
    _require(len(frame_paths) >= 3, "source episode has too few point-cloud frames")
    frame_payloads = []
    for frame_path in frame_paths:
        with np.load(frame_path, allow_pickle=False) as stored:
            frame_payloads.append(
                {
                    "pts": np.asarray(stored["pts"], dtype=np.float32),
                    "vels": np.asarray(stored["vels"], dtype=np.float32),
                    "visibility": np.asarray(
                        stored["visibility_matrix"], dtype=np.float32
                    ),
                }
            )
    point_count = len(frame_payloads[0]["pts"])
    _require(
        all(
            frame["pts"].shape == (point_count, 3)
            and frame["vels"].shape == (point_count, 3)
            and frame["visibility"].shape[0] == point_count
            for frame in frame_payloads
        ),
        "material point identities are inconsistent across frames",
    )
    selected_count = min(int(maximum_node_count), point_count)
    _require(selected_count >= 2, "source episode has too few material points")
    selected = _deterministic_farthest_points(
        frame_payloads[0]["pts"], selected_count
    )
    positions = np.stack([frame["pts"][selected] for frame in frame_payloads])
    velocities = np.stack([frame["vels"][selected] for frame in frame_payloads])
    reliability = np.stack(
        [np.mean(frame["visibility"][selected], axis=1) for frame in frame_payloads]
    )
    if physics_prediction_path is None:
        _require(
            physics_response_scale is None,
            "physical response scale requires a sealed physical prediction",
        )
        physics_positions = np.repeat(positions[0:1], len(positions), axis=0)
        physics_prior_kind = "persistence"
        applied_response_scale = 0.0
    else:
        prediction_path = Path(physics_prediction_path).resolve()
        _require(prediction_path.is_file(), "sealed physical prediction is missing")
        with np.load(prediction_path, allow_pickle=False) as prediction:
            _require(
                "prediction_m" in prediction.files
                and "frame_zero_points_m" in prediction.files,
                "sealed physical prediction has an unsupported schema",
            )
            full_physics = np.asarray(prediction["prediction_m"], dtype=np.float32)
            frame_zero = np.asarray(
                prediction["frame_zero_points_m"], dtype=np.float32
            )
        _require(
            full_physics.shape == (len(positions), point_count, 3)
            and frame_zero.shape == (point_count, 3),
            "sealed physical prediction shape differs from observations",
        )
        _require(
            np.array_equal(frame_zero, frame_payloads[0]["pts"]),
            "sealed physical prediction point identities differ",
        )
        selected_physics = full_physics[:, selected]
        if physics_response_scale is None:
            physics_positions = selected_physics
            physics_prior_kind = "sealed_graph_action_support"
            applied_response_scale = float(physics_reference_response_scale)
        else:
            applied_response_scale = float(physics_response_scale)
            _require(
                np.isfinite(applied_response_scale)
                and applied_response_scale >= 0.0,
                "physical response scale must be finite and nonnegative",
            )
            _require(
                np.isfinite(physics_reference_response_scale)
                and physics_reference_response_scale > 0.0,
                "reference physical response scale must be positive",
            )
            persistence = np.repeat(positions[0:1], len(positions), axis=0)
            if applied_response_scale == 0.0:
                physics_positions = persistence.copy()
            else:
                response = (selected_physics - persistence) / float(
                    physics_reference_response_scale
                )
                physics_positions = persistence + applied_response_scale * response
                physics_positions[0] = positions[0]
            physics_prior_kind = "trusted_sealed_graph_action_support"

    robot_path = directory / "robot" / "robot.npz"
    _require(robot_path.is_file(), "source robot trajectory is missing")
    with np.load(robot_path, allow_pickle=False) as robot:
        transforms = np.asarray(robot["T_worlds"], dtype=np.float64)
        openings = np.asarray(robot["openings"], dtype=np.float64)
    if transforms.ndim == 3:
        transforms = transforms[:, None]
    if openings.ndim == 1:
        openings = openings[:, None]
    frame_count = len(positions)
    _require(
        transforms.ndim == 4
        and transforms.shape[1] == openings.shape[1]
        and len(transforms) >= frame_count
        and len(openings) >= frame_count,
        "source robot trajectory does not cover point-cloud frames",
    )
    transforms = transforms[:frame_count]
    opening_values = openings[:frame_count]
    if controller_surface_provider is None:
        controller_positions = transforms[:, :, :3, 3]
        controller_groups = np.arange(transforms.shape[1], dtype=np.int64)
        controller_geometry = "end_effector_origins"
    else:
        controller_positions, controller_groups = _controller_surface_trajectories(
            transforms,
            opening_values,
            controller_surface_provider,
            int(controller_points_per_gripper),
        )
        controller_geometry = "gripper_surface"
    frame_interval = 1.0 / _frame_rate_hz(directory)
    controller_velocities = np.zeros_like(controller_positions)
    controller_velocities[1:] = np.diff(controller_positions, axis=0) / frame_interval
    controller_velocities[0] = controller_velocities[1]
    low = np.quantile(opening_values, 0.10, axis=0)
    high = np.quantile(opening_values, 0.90, axis=0)
    opening_scale = np.maximum(high - low, 1.0e-6)
    gripper_closure = np.clip((high - opening_values) / opening_scale, 0.0, 1.0)
    closure = gripper_closure[:, controller_groups]
    edge_index = knn_edge_index(
        positions[0], neighbor_count=min(int(neighbor_count), selected_count - 1)
    )
    clusters = spatial_cluster_ids(positions[0])
    return Deform360ResidualSourceEpisode(
        object_id=str(object_id),
        episode_id=int(episode_id),
        positions_m=positions,
        observed_velocities_mps=velocities,
        physics_positions_m=physics_positions,
        physics_prior_kind=physics_prior_kind,
        prior_reliability=reliability,
        controller_positions_m=controller_positions,
        controller_velocities_mps=controller_velocities,
        closure_probability=closure,
        controller_group_ids=controller_groups,
        controller_geometry=controller_geometry,
        edge_index=edge_index,
        cluster_ids=clusters,
        frame_interval_s=frame_interval,
        physics_response_scale=applied_response_scale,
        physics_reference_response_scale=float(physics_reference_response_scale),
    )


__all__ = [
    "Deform360ResidualSourceEpisode",
    "ControllerSurfaceProvider",
    "contact_probabilities_from_state",
    "knn_edge_index",
    "load_deform360_residual_source_episode",
    "spatial_cluster_ids",
]
