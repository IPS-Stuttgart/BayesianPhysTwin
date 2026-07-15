"""Development-only official-PhysTwin Warp backend for the PokeFlex pilot."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d_public.deform360_replication import PINNED_OFFICIAL_PHYSTWIN_COMMIT
from causal4d_public.pokeflex import PINNED_POKEFLEX_COMMIT
from causal4d_public.pokeflex_source_qa import validate_source_qa_artifact


POKEFLEX_WARP_POLICY_SCHEMA_VERSION = 1
POKEFLEX_WARP_ARTIFACT_SCHEMA_VERSION = 1
POKEFLEX_WARP_POLICY_ID = "causal4d-pokeflex-source-warp-v1"
OFFICIAL_SIMULATOR_RELATIVE_PATH = (
    Path("qqtt") / "model" / "diff_simulator" / "spring_mass_warp.py"
)
PINNED_OFFICIAL_SIMULATOR_SHA256 = (
    "7deab9a25f4b8b8772f7df45c35571caf3767d014dd353cad151fe8eddceca1c"
)
CANONICAL_POKEFLEX_WARP_POLICY_SHA256 = (
    "bf534da2543116f486472581e842786f55b1fa538b0f06d5cb9b5d98c6904c26"
)
_MESH_PATTERN = re.compile(r"mesh-f(\d{5})\.obj$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PokeFlexWarpSourceConfig:
    policy_id: str = POKEFLEX_WARP_POLICY_ID
    upstream_commit: str = PINNED_POKEFLEX_COMMIT
    official_phystwin_commit: str = PINNED_OFFICIAL_PHYSTWIN_COMMIT
    official_simulator_sha256: str = PINNED_OFFICIAL_SIMULATOR_SHA256
    expected_source_qa_result_sha256: str | None = None
    prefix_frame_count: int = 6
    frame_dt_seconds: float = 1.0 / 30.0
    graph_node_count: int = 128
    graph_knn: int = 8
    controller_patch_size: int = 4
    evaluation_surface_point_count: int = 512
    force_axis_index: int = 1
    force_threshold_n: float = 3.0
    substeps: int = 64
    initial_ground_clearance_m: float = 0.001
    object_spring_y_grid: tuple[float, ...] = (
        100.0,
        300.0,
        1000.0,
        3000.0,
        10000.0,
    )
    controller_spring_y_grid: tuple[float, ...] = (
        100.0,
        300.0,
        1000.0,
        3000.0,
        10000.0,
    )
    ground_friction_grid: tuple[float, ...] = (0.0, 0.3)
    dashpot_damping: float = 0.0
    drag_damping: float = 3.0
    ground_elasticity: float = 0.0
    inactive_controller_spring_y: float = 1e-12
    maximum_repeat_rollout_rmse_m: float = 1e-4
    maximum_p99_relative_edge_strain: float = 0.5
    minimum_leave_one_out_persistence_win_fraction: float = 0.6
    minimum_pooled_vs_single_source_win_fraction: float = 0.6

    def __post_init__(self) -> None:
        _require(self.policy_id == POKEFLEX_WARP_POLICY_ID, "Warp policy id changed")
        _require(
            self.upstream_commit == PINNED_POKEFLEX_COMMIT,
            "PokeFlex upstream commit changed",
        )
        _require(
            self.official_phystwin_commit == PINNED_OFFICIAL_PHYSTWIN_COMMIT,
            "official PhysTwin commit changed",
        )
        _require(
            self.official_simulator_sha256 == PINNED_OFFICIAL_SIMULATOR_SHA256,
            "official simulator hash changed",
        )
        _require(self.prefix_frame_count >= 2, "prefix is too short")
        _require(self.frame_dt_seconds > 0.0, "frame interval must be positive")
        _require(self.graph_node_count >= 16, "surface graph is too small")
        _require(2 <= self.graph_knn < self.graph_node_count, "graph kNN is invalid")
        _require(
            1 <= self.controller_patch_size <= self.graph_node_count,
            "controller patch is invalid",
        )
        _require(
            self.evaluation_surface_point_count >= self.graph_node_count,
            "evaluation surface is smaller than the graph",
        )
        _require(self.force_axis_index in {0, 1, 2}, "force axis is invalid")
        _require(self.force_threshold_n > 0.0, "force threshold must be positive")
        _require(self.substeps >= 1, "substeps must be positive")
        for name in (
            "object_spring_y_grid",
            "controller_spring_y_grid",
            "ground_friction_grid",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            _require(len(values) >= 1, f"{name} is empty")
            _require(
                np.all(np.isfinite(values)) and np.all(values >= 0.0),
                f"{name} contains invalid values",
            )
        _require(
            self.inactive_controller_spring_y > 0.0,
            "inactive controller stiffness must be positive",
        )
        for value in (
            self.minimum_leave_one_out_persistence_win_fraction,
            self.minimum_pooled_vs_single_source_win_fraction,
        ):
            _require(0.0 <= value <= 1.0, "win-fraction gate is invalid")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PokeFlexWarpSourceConfig:
        fields = cls.__dataclass_fields__
        unknown = set(value) - set(fields)
        _require(not unknown, f"unknown Warp policy fields: {sorted(unknown)}")
        payload = dict(value)
        for key in (
            "object_spring_y_grid",
            "controller_spring_y_grid",
            "ground_friction_grid",
        ):
            if key in payload:
                payload[key] = tuple(map(float, payload[key]))
        return cls(**payload)


def warp_policy_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def validate_warp_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == POKEFLEX_WARP_POLICY_SCHEMA_VERSION,
        "unsupported PokeFlex Warp policy schema",
    )
    _require(
        payload.get("artifact_kind") == "PublicPokeFlexWarpSourcePolicy",
        "unexpected PokeFlex Warp policy kind",
    )
    observed = warp_policy_sha256(payload)
    _require(payload.get("config_sha256") == observed, "Warp policy checksum mismatch")
    if CANONICAL_POKEFLEX_WARP_POLICY_SHA256:
        _require(
            observed == CANONICAL_POKEFLEX_WARP_POLICY_SHA256,
            "Warp policy differs from the canonical lock",
        )
    boundary = payload.get("information_boundary")
    _require(
        boundary
        == {
            "development_takes_only": True,
            "calibration_take_data_allowed": False,
            "target_take_data_allowed": False,
            "source_prediction_metrics_allowed": True,
            "material_identity_metrics_allowed": False,
        },
        "Warp policy information boundary changed",
    )
    config = PokeFlexWarpSourceConfig.from_mapping(payload["config"])
    return {"passed": True, "config_sha256": observed, "config": config}


def load_warp_policy(path: str | Path) -> PokeFlexWarpSourceConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_warp_policy(payload)["config"]


@dataclass(frozen=True)
class PokeFlexWarpCandidate:
    object_spring_y: float
    controller_spring_y: float
    ground_friction: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PokeFlexWarpCase:
    take_id: str
    frame_ids: tuple[int, ...]
    rest_positions_m: np.ndarray
    initial_positions_m: np.ndarray
    initial_velocities_mps: np.ndarray
    target_surfaces_m: tuple[np.ndarray, ...]
    controller_positions_m: np.ndarray
    contact_active: np.ndarray
    springs: np.ndarray
    rest_lengths_m: np.ndarray
    object_spring_count: int
    controller_node_indices: tuple[int, ...]
    dt_seconds: float
    graph_connected: bool
    input_summary: Mapping[str, Any]


def warp_candidates(
    config: PokeFlexWarpSourceConfig,
) -> tuple[PokeFlexWarpCandidate, ...]:
    return tuple(
        PokeFlexWarpCandidate(
            object_spring_y=float(object_y),
            controller_spring_y=float(controller_y),
            ground_friction=float(friction),
        )
        for object_y, controller_y, friction in product(
            config.object_spring_y_grid,
            config.controller_spring_y_grid,
            config.ground_friction_grid,
        )
    )


def _mesh_frame(path: Path) -> int | None:
    match = _MESH_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:4])))
    result = np.asarray(vertices, dtype=np.float64)
    _require(
        result.ndim == 2 and result.shape[1] == 3 and len(result),
        f"mesh contains no vertices: {path.name}",
    )
    _require(np.all(np.isfinite(result)), f"mesh is non-finite: {path.name}")
    return result / 1000.0


def _scipy_tree(points: np.ndarray):
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - integration dependency
        raise RuntimeError("PokeFlex Warp preparation requires scipy") from error
    return cKDTree(points)


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    _require(len(points) >= count >= 1, "farthest-point sample size is invalid")
    center = points.mean(axis=0)
    first = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_squared = np.sum((points - points[first]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum_squared))
        squared = np.sum((points - points[selected[index]]) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, squared)
    return selected


def _evaluation_sample(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[indices]


def _track_surface_nodes(nodes: np.ndarray, surface: np.ndarray) -> np.ndarray:
    neighbor_count = min(8, len(surface))
    _, candidates = _scipy_tree(surface).query(nodes, k=neighbor_count)
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    used: set[int] = set()
    selected = np.empty(len(nodes), dtype=np.int64)
    for node, row in enumerate(candidates):
        choice = next(
            (int(value) for value in row if int(value) not in used), int(row[0])
        )
        selected[node] = choice
        used.add(choice)
    return surface[selected]


def _knn_edges(points: np.ndarray, neighbors: int) -> np.ndarray:
    _, indices = _scipy_tree(points).query(points, k=neighbors + 1)
    edges = {
        tuple(sorted((source, int(target))))
        for source, row in enumerate(np.asarray(indices)[:, 1:])
        for target in row
        if source != int(target)
    }
    _require(bool(edges), "surface graph has no edges")
    return np.asarray(sorted(edges), dtype=np.int32)


def _graph_connected(node_count: int, edges: np.ndarray) -> bool:
    adjacency = [[] for _ in range(node_count)]
    for first, second in edges:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == node_count


def build_pokeflex_warp_case(
    take_root: str | Path,
    config: PokeFlexWarpSourceConfig | None = None,
) -> PokeFlexWarpCase:
    """Build a take-specific sparse graph from source geometry and measured action."""

    cfg = config or PokeFlexWarpSourceConfig()
    root = Path(take_root).resolve()
    robot_path = root / "robot_data.json"
    robot_payload = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(record["frame"]): record for record in robot_payload}
    mesh_paths = sorted(
        (path for path in (root / "meshes").glob("mesh-f*.obj") if _mesh_frame(path)),
        key=lambda path: int(_mesh_frame(path) or -1),
    )
    frames = tuple(int(_mesh_frame(path) or -1) for path in mesh_paths)
    _require(frames == tuple(sorted(robot_by_frame)), "source mesh/robot frames differ")
    _require(len(frames) > cfg.prefix_frame_count, "source take has no forecast future")

    prefix_surfaces = [_obj_vertices(path) for path in mesh_paths[: cfg.prefix_frame_count]]
    rest_surface = prefix_surfaces[0]
    node_indices = _farthest_point_indices(rest_surface, cfg.graph_node_count)
    rest_positions = rest_surface[node_indices]
    tracked = rest_positions.copy()
    for surface in prefix_surfaces[1:]:
        tracked = _track_surface_nodes(tracked, surface)
    initial_positions = tracked
    initial_velocities = np.zeros_like(initial_positions)
    object_edges = _knn_edges(rest_positions, cfg.graph_knn)
    connected = _graph_connected(cfg.graph_node_count, object_edges)
    _require(connected, "take-specific surface graph is disconnected")

    floor_z = float(np.min(rest_surface[:, 2]) - cfg.initial_ground_clearance_m)
    rest_positions = rest_positions.copy()
    initial_positions = initial_positions.copy()
    rest_positions[:, 2] -= floor_z
    initial_positions[:, 2] -= floor_z

    forecast_mesh_paths = mesh_paths[cfg.prefix_frame_count - 1 :]
    target_surfaces = []
    for index, path in enumerate(forecast_mesh_paths):
        surface = prefix_surfaces[-1] if index == 0 else _obj_vertices(path)
        sampled = _evaluation_sample(surface, cfg.evaluation_surface_point_count).copy()
        sampled[:, 2] -= floor_z
        target_surfaces.append(sampled)

    forecast_frames = frames[cfg.prefix_frame_count - 1 :]
    tool_positions = []
    active = []
    for frame in forecast_frames:
        record = robot_by_frame[frame]
        transform = np.asarray(record["T_WT"], dtype=np.float64)
        _require(transform.shape == (4, 4), "tool transform has the wrong shape")
        tool = transform[:3, 3].copy()
        tool[2] -= floor_z
        tool_positions.append(tool)
        force = np.asarray(record["forces"], dtype=np.float64)
        active.append(bool(force[cfg.force_axis_index] > cfg.force_threshold_n))
    tool_array = np.asarray(tool_positions, dtype=np.float64)
    _, patch = _scipy_tree(initial_positions).query(
        tool_array[0], k=cfg.controller_patch_size
    )
    patch = tuple(map(int, np.atleast_1d(patch)))
    controller_positions = np.repeat(
        tool_array[:, None, :], cfg.controller_patch_size, axis=1
    )
    contact_active = np.repeat(
        np.asarray(active, dtype=bool)[:, None], cfg.controller_patch_size, axis=1
    )
    controller_edges = np.asarray(
        [
            (cfg.graph_node_count + controller, node)
            for controller, node in enumerate(patch)
        ],
        dtype=np.int32,
    )
    springs = np.concatenate((object_edges, controller_edges), axis=0)
    object_rest = np.linalg.norm(
        rest_positions[object_edges[:, 1]] - rest_positions[object_edges[:, 0]],
        axis=1,
    )
    controller_rest = np.maximum(
        np.linalg.norm(initial_positions[np.asarray(patch)] - tool_array[0], axis=1),
        1e-3,
    )
    rest_lengths = np.concatenate((object_rest, controller_rest)).astype(np.float32)
    _require(np.all(rest_lengths > 0.0), "surface graph has degenerate springs")
    return PokeFlexWarpCase(
        take_id=root.name,
        frame_ids=forecast_frames,
        rest_positions_m=rest_positions.astype(np.float32),
        initial_positions_m=initial_positions.astype(np.float32),
        initial_velocities_mps=initial_velocities.astype(np.float32),
        target_surfaces_m=tuple(value.astype(np.float64) for value in target_surfaces),
        controller_positions_m=controller_positions.astype(np.float32),
        contact_active=contact_active,
        springs=springs,
        rest_lengths_m=rest_lengths,
        object_spring_count=len(object_edges),
        controller_node_indices=patch,
        dt_seconds=cfg.frame_dt_seconds,
        graph_connected=connected,
        input_summary={
            "robot_sha256": _sha256_file(robot_path),
            "first_mesh_sha256": _sha256_file(mesh_paths[0]),
            "frame_count": len(frames),
            "forecast_frame_count": len(forecast_frames) - 1,
            "ground_shift_z_m": floor_z,
            "material_identity_used": False,
            "initial_velocity_policy": "zero",
            "prefix_surface_tracking": "causal nearest unique candidate",
        },
    )


def geometry_chamfer_m(prediction: np.ndarray, target: np.ndarray) -> float:
    first = _scipy_tree(target).query(prediction, k=1)[0]
    second = _scipy_tree(prediction).query(target, k=1)[0]
    return float(0.5 * (np.mean(first) + np.mean(second)))


def score_geometry_rollout(
    case: PokeFlexWarpCase, trajectory: np.ndarray
) -> dict[str, float]:
    _require(
        trajectory.shape == (len(case.target_surfaces_m), len(case.initial_positions_m), 3),
        "trajectory shape does not match the PokeFlex case",
    )
    values = np.asarray(
        [
            geometry_chamfer_m(prediction, target)
            for prediction, target in zip(
                trajectory[1:], case.target_surfaces_m[1:], strict=True
            )
        ],
        dtype=np.float64,
    )
    return {
        "future_chamfer_m": float(np.mean(values)),
        "endpoint_chamfer_m": float(values[-1]),
    }


class _OfficialWarpSurfaceRunner:
    def __init__(
        self,
        official_repo: Path,
        case: PokeFlexWarpCase,
        config: PokeFlexWarpSourceConfig,
        *,
        device: str,
    ) -> None:
        try:
            import torch
            import warp as wp
        except ImportError as error:  # pragma: no cover - GPU integration
            raise RuntimeError("official PokeFlex-Warp backend requires torch and warp") from error
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
            official_repo, runtime_config=runtime_config
        )
        simulator_class = make_reliability_simulator_class(official)
        frame_count = len(case.target_surfaces_m)
        node_count = len(case.initial_positions_m)
        visible = np.ones((frame_count, node_count), dtype=bool)
        motion_valid = np.ones((frame_count - 1, node_count), dtype=bool)
        objective = build_phystwin_track_objective(visible, motion_valid, variant="hard")

        def tensor(values: np.ndarray, dtype):
            return torch.as_tensor(values, dtype=dtype, device=device).contiguous()

        vertices = np.concatenate(
            (case.initial_positions_m, case.controller_positions_m[0]), axis=0
        )
        masses = np.ones(len(vertices), dtype=np.float32)
        dummy_gt = np.repeat(case.initial_positions_m[None, :, :], frame_count, axis=0)
        self.simulator = simulator_class(
            tensor(vertices, torch.float32),
            tensor(case.springs, torch.int32),
            tensor(case.rest_lengths_m, torch.float32),
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
            num_object_points=node_count,
            num_surface_points=node_count,
            num_original_points=node_count,
            controller_points=tensor(case.controller_positions_m, torch.float32),
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
            num_object_springs=case.object_spring_count,
            deterministic_spring_forces=True,
        )
        self.initial_tensor = tensor(case.initial_positions_m, torch.float32)
        self.velocity_tensor = tensor(case.initial_velocities_mps, torch.float32)
        self.wp_initial = wp.from_torch(
            self.initial_tensor, dtype=wp.vec3, requires_grad=False
        )
        self.wp_velocity = wp.from_torch(
            self.velocity_tensor, dtype=wp.vec3, requires_grad=False
        )
        wp.synchronize()

    def _spring_log_y(
        self, candidate: PokeFlexWarpCandidate, active: Sequence[bool]
    ):
        values = np.empty(len(self.case.springs), dtype=np.float32)
        values[: self.case.object_spring_count] = candidate.object_spring_y
        for controller, enabled in enumerate(active):
            values[self.case.object_spring_count + controller] = (
                candidate.controller_spring_y
                if enabled
                else self.config.inactive_controller_spring_y
            )
        return self.torch.log(
            self.torch.as_tensor(
                values, dtype=self.torch.float32, device=self.device
            )
        ).contiguous()

    def rollout(self, candidate: PokeFlexWarpCandidate) -> tuple[np.ndarray, float]:
        torch = self.torch
        wp = self.wp
        friction = torch.as_tensor(
            [candidate.ground_friction], dtype=torch.float32, device=self.device
        ).contiguous()
        elasticity = torch.as_tensor(
            [self.config.ground_elasticity], dtype=torch.float32, device=self.device
        ).contiguous()
        self.simulator.set_collide(elasticity, friction)
        self.simulator.set_init_state(
            self.wp_initial, self.wp_velocity, pure_inference=True
        )
        wp.synchronize()
        trajectory = [self.case.initial_positions_m.astype(np.float64)]
        strains = []
        previous_active: tuple[bool, ...] | None = None
        object_edges = self.case.springs[: self.case.object_spring_count]
        object_rest = self.case.rest_lengths_m[: self.case.object_spring_count]
        for frame in range(1, len(self.case.target_surfaces_m)):
            active = tuple(map(bool, self.case.contact_active[frame]))
            if active != previous_active:
                self.simulator.set_reference_spring_y(
                    self._spring_log_y(candidate, active)
                )
                previous_active = active
            self.simulator.set_controller_target(frame, pure_inference=True)
            wp.capture_launch(self.simulator.forward_graph)
            wp.synchronize()
            positions = (
                wp.to_torch(self.simulator.wp_states[-1].wp_x)
                .detach()
                .cpu()
                .numpy()
                .copy()
                [: len(self.case.initial_positions_m)]
                .astype(np.float64)
            )
            trajectory.append(positions)
            if not np.all(np.isfinite(positions)):
                missing = len(self.case.target_surfaces_m) - len(trajectory)
                trajectory.extend(
                    [np.full_like(positions, np.nan) for _ in range(missing)]
                )
                return np.stack(trajectory), float("inf")
            lengths = np.linalg.norm(
                positions[object_edges[:, 1]] - positions[object_edges[:, 0]], axis=1
            )
            strains.extend(np.abs(lengths / object_rest - 1.0).tolist())
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
                pure_inference=True,
            )
        p99 = float(np.percentile(strains, 99.0)) if strains else 0.0
        return np.stack(trajectory), p99


def summarize_pooling_controls(
    candidates: Sequence[PokeFlexWarpCandidate],
    take_ids: Sequence[str],
    scores: np.ndarray,
    persistence_scores: np.ndarray,
) -> dict[str, Any]:
    _require(
        scores.shape == (len(candidates), len(take_ids)),
        "candidate score matrix has the wrong shape",
    )
    _require(
        persistence_scores.shape == (len(take_ids),),
        "persistence score vector has the wrong shape",
    )
    valid = np.all(np.isfinite(scores), axis=1)
    _require(np.any(valid), "every Warp candidate is non-finite")
    valid_indices = np.flatnonzero(valid)

    def select(columns: Sequence[int]) -> int:
        return int(
            min(
                valid_indices,
                key=lambda index: (
                    float(np.mean(scores[index, list(columns)])), int(index)
                ),
            )
        )

    pooled_index = select(range(len(take_ids)))
    folds = []
    for held, take_id in enumerate(take_ids):
        training = [index for index in range(len(take_ids)) if index != held]
        pooled_loo_index = select(training)
        single_scores = []
        single_indices = []
        for source in training:
            candidate_index = select([source])
            single_indices.append(candidate_index)
            single_scores.append(float(scores[candidate_index, held]))
        held_score = float(scores[pooled_loo_index, held])
        persistence = float(persistence_scores[held])
        single_median = float(np.median(single_scores))
        folds.append(
            {
                "held_out_take_id": take_id,
                "pooled_leave_one_out_candidate_index": pooled_loo_index,
                "pooled_leave_one_out_chamfer_m": held_score,
                "persistence_chamfer_m": persistence,
                "pooled_better_than_persistence": held_score < persistence,
                "single_source_candidate_indices": single_indices,
                "single_source_held_out_chamfer_m": single_scores,
                "single_source_median_chamfer_m": single_median,
                "pooled_better_than_single_source_median": held_score < single_median,
            }
        )
    return {
        "valid_candidate_count": int(np.sum(valid)),
        "candidate_count": len(candidates),
        "pooled_candidate_index": pooled_index,
        "pooled_candidate": candidates[pooled_index].as_dict(),
        "pooled_source_mean_chamfer_m": float(np.mean(scores[pooled_index])),
        "persistence_source_mean_chamfer_m": float(np.mean(persistence_scores)),
        "leave_one_out": folds,
        "leave_one_out_persistence_win_fraction": float(
            np.mean([value["pooled_better_than_persistence"] for value in folds])
        ),
        "pooled_vs_single_source_win_fraction": float(
            np.mean(
                [value["pooled_better_than_single_source_median"] for value in folds]
            )
        ),
    }


def warp_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _json_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in np.asarray(values, dtype=np.float64)
    ]


def run_pokeflex_warp_source_backend(
    dataset_root: str | Path,
    source_qa: Mapping[str, Any],
    official_repo: str | Path,
    config: PokeFlexWarpSourceConfig | None = None,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Fit only development-take simulator candidates and run pooled controls."""

    cfg = config or PokeFlexWarpSourceConfig()
    validate_source_qa_artifact(source_qa)
    if cfg.expected_source_qa_result_sha256 is not None:
        _require(
            source_qa["result_sha256"] == cfg.expected_source_qa_result_sha256,
            "Warp backend received an unexpected source-QA artifact",
        )
    boundary = source_qa["information_boundary"]
    take_ids = tuple(map(str, boundary["opened_take_ids"]))
    _require(bool(take_ids), "source QA contains no development takes")
    object_id = str(source_qa["object_id"])
    root = Path(dataset_root).resolve()
    official = Path(official_repo).resolve()
    _require(official.is_dir(), "official PhysTwin repository is missing")
    head = subprocess.run(
        ["git", "-C", str(official), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(head == cfg.official_phystwin_commit, "official PhysTwin commit changed")
    simulator_source = official / OFFICIAL_SIMULATOR_RELATIVE_PATH
    _require(
        _sha256_file(simulator_source) == cfg.official_simulator_sha256,
        "official simulator source changed",
    )
    cases = [build_pokeflex_warp_case(root / object_id / take_id, cfg) for take_id in take_ids]
    candidates = warp_candidates(cfg)
    scores = np.full((len(candidates), len(cases)), np.inf, dtype=np.float64)
    endpoints = np.full_like(scores, np.inf)
    strains = np.full_like(scores, np.inf)
    persistence_scores = np.empty(len(cases), dtype=np.float64)
    for case_index, case in enumerate(cases):
        persistence = np.repeat(
            case.initial_positions_m[None, :, :], len(case.target_surfaces_m), axis=0
        )
        persistence_scores[case_index] = score_geometry_rollout(
            case, persistence
        )["future_chamfer_m"]
        runner = _OfficialWarpSurfaceRunner(official, case, cfg, device=device)
        for candidate_index, candidate in enumerate(candidates):
            trajectory, p99_strain = runner.rollout(candidate)
            if np.all(np.isfinite(trajectory)):
                score = score_geometry_rollout(case, trajectory)
                scores[candidate_index, case_index] = score["future_chamfer_m"]
                endpoints[candidate_index, case_index] = score["endpoint_chamfer_m"]
                strains[candidate_index, case_index] = p99_strain
    pooling = summarize_pooling_controls(candidates, take_ids, scores, persistence_scores)
    selected_index = int(pooling["pooled_candidate_index"])
    selected = candidates[selected_index]
    repeat_records = []
    for case in cases:
        runner = _OfficialWarpSurfaceRunner(official, case, cfg, device=device)
        first, _ = runner.rollout(selected)
        second, _ = runner.rollout(selected)
        repeat_rmse = float(np.sqrt(np.mean((first - second) ** 2)))
        repeat_records.append({"take_id": case.take_id, "repeat_rmse_m": repeat_rmse})
    maximum_repeat = max(value["repeat_rmse_m"] for value in repeat_records)
    selected_strain = float(np.max(strains[selected_index]))
    gates = {
        "repeat_determinism_ready": maximum_repeat <= cfg.maximum_repeat_rollout_rmse_m,
        "strain_plausibility_ready": selected_strain
        <= cfg.maximum_p99_relative_edge_strain,
        "leave_one_out_beats_persistence": pooling[
            "leave_one_out_persistence_win_fraction"
        ]
        >= cfg.minimum_leave_one_out_persistence_win_fraction,
        "pooling_beats_single_source": pooling[
            "pooled_vs_single_source_win_fraction"
        ]
        >= cfg.minimum_pooled_vs_single_source_win_fraction,
    }
    result: dict[str, Any] = {
        "schema_version": POKEFLEX_WARP_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "PublicPokeFlexWarpSourceBackend",
        "policy_id": cfg.policy_id,
        "source_qa_result_sha256": source_qa["result_sha256"],
        "official_phystwin": {
            "commit": head,
            "simulator_sha256": _sha256_file(simulator_source),
            "device": device,
        },
        "information_boundary": {
            "opened_take_ids": list(take_ids),
            "development_data_only": True,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "material_identity_metrics_computed": False,
            "source_prediction_metrics_computed": True,
            "raw_data_embedded": False,
        },
        "backend": {
            "graph_policy": "take-specific prefix-tracked surface kNN graph",
            "shared_quantities": [
                "object_spring_y",
                "controller_spring_y",
                "ground_friction",
            ],
            "parameter_interpretation": (
                "official-simulator configuration parameters, not material constants"
            ),
            "frame_dt_assumption_seconds": cfg.frame_dt_seconds,
            "case_summaries": [
                {
                    "take_id": case.take_id,
                    "graph_node_count": len(case.initial_positions_m),
                    "object_spring_count": case.object_spring_count,
                    "controller_patch_size": len(case.controller_node_indices),
                    "forecast_frame_count": len(case.target_surfaces_m) - 1,
                    "graph_connected": case.graph_connected,
                    "input_summary": dict(case.input_summary),
                }
                for case in cases
            ],
        },
        "candidate_grid": [value.as_dict() for value in candidates],
        "future_chamfer_m": _json_matrix(scores),
        "endpoint_chamfer_m": _json_matrix(endpoints),
        "p99_relative_edge_strain": _json_matrix(strains),
        "persistence_chamfer_m": persistence_scores.tolist(),
        "pooling_controls": pooling,
        "determinism": {
            "maximum_repeat_rmse_m": maximum_repeat,
            "takes": repeat_records,
        },
        "selected_p99_relative_edge_strain": selected_strain,
        "gates": gates,
        "source_backend_admitted": all(gates.values()),
        "claim_boundary": (
            "Development-only source admission for a sparse take-specific graph in "
            "the official PhysTwin Warp simulator. This is not the full PhysTwin "
            "reconstruction pipeline, a material-parameter claim, or target evidence."
        ),
    }
    result["result_sha256"] = warp_artifact_sha256(result)
    return result


def validate_warp_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == POKEFLEX_WARP_ARTIFACT_SCHEMA_VERSION,
        "unsupported PokeFlex Warp artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "PublicPokeFlexWarpSourceBackend",
        "unexpected PokeFlex Warp artifact kind",
    )
    _require(
        payload.get("result_sha256") == warp_artifact_sha256(payload),
        "PokeFlex Warp artifact checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(boundary.get("development_data_only") is True, "source boundary changed")
    _require(boundary.get("calibration_take_data_read") is False, "calibration opened")
    _require(boundary.get("target_take_data_read") is False, "target opened")
    _require(
        boundary.get("material_identity_metrics_computed") is False,
        "material identity was assumed",
    )
    return {
        "passed": True,
        "source_backend_admitted": bool(payload["source_backend_admitted"]),
        "result_sha256": payload["result_sha256"],
    }


def write_warp_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
