"""Target-free camera and material-query scheduling for Deform360.

The selector reads only a physical rollout, frame-zero geometry, calibration,
and image dimensions. It never reads an observed object trajectory or target
metric. Query identities are chosen greedily to cover graph-state modes while
remaining physically active, spatially separated, and multiview visible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .observation_belief import array_sha256

PROTOCOL_ID = "deform360-dynamic-tapnextpp-provider-v1"
UPDATE_FRAMES = (19, 38, 57)
QUERY_BIRTH_FRAMES = ((0, 6, 12), (20, 26, 32), (39, 45, 51))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DynamicQueryConfig:
    """Frozen source-independent query-scheduling choices."""

    selected_camera_count: int = 8
    minimum_eligible_camera_count: int = 8
    coverage_weight: float = 1.0
    angular_diversity_weight: float = 0.25
    queries_per_birth: int = 8
    graph_basis_rank: int = 8
    minimum_predicted_motion_m: float = 0.005
    minimum_predicted_visible_views: int = 3
    minimum_spatial_separation_m: float = 0.015
    information_ridge: float = 1e-6
    maximum_motion_weight: float = 4.0

    def __post_init__(self) -> None:
        _require(self.selected_camera_count >= 3, "too few selected cameras")
        _require(
            self.minimum_eligible_camera_count >= self.selected_camera_count,
            "eligible-camera gate precedes selected-camera count",
        )
        _require(self.coverage_weight > 0.0, "coverage weight must be positive")
        _require(
            self.angular_diversity_weight >= 0.0,
            "angular-diversity weight must be nonnegative",
        )
        _require(self.queries_per_birth >= 1, "query count must be positive")
        _require(self.graph_basis_rank >= 1, "graph rank must be positive")
        _require(
            self.minimum_predicted_motion_m > 0.0,
            "motion threshold must be positive",
        )
        _require(
            self.minimum_predicted_visible_views >= 3,
            "claim-bearing queries require at least three views",
        )
        _require(
            self.minimum_predicted_visible_views <= self.selected_camera_count,
            "visibility gate exceeds selected camera count",
        )
        _require(
            self.minimum_spatial_separation_m >= 0.0,
            "spatial separation must be nonnegative",
        )
        _require(self.information_ridge > 0.0, "information ridge must be positive")
        _require(
            self.maximum_motion_weight >= 1.0,
            "maximum motion weight must be at least one",
        )


@dataclass(frozen=True)
class CameraPanel:
    """Deterministically selected camera panel."""

    camera_indices: np.ndarray
    camera_names: tuple[str, ...]
    frame_zero_coverage: np.ndarray
    selection_scores: np.ndarray

    def __post_init__(self) -> None:
        indices = _readonly(self.camera_indices, dtype=np.int64)
        coverage = _readonly(self.frame_zero_coverage, dtype=np.float64)
        scores = _readonly(self.selection_scores, dtype=np.float64)
        count = len(indices)
        _require(count > 0, "camera panel is empty")
        _require(len(self.camera_names) == count, "camera names differ from panel")
        _require(coverage.shape == (count,), "camera coverage shape changed")
        _require(scores.shape == (count,), "camera score shape changed")
        _require(len(set(map(int, indices))) == count, "camera panel repeats a camera")
        _require(np.all(np.isfinite(coverage)), "camera coverage is not finite")
        _require(np.all((coverage >= 0.0) & (coverage <= 1.0)), "invalid coverage")
        _require(np.all(np.isfinite(scores)), "camera scores are not finite")
        object.__setattr__(self, "camera_indices", indices)
        object.__setattr__(self, "frame_zero_coverage", coverage)
        object.__setattr__(self, "selection_scores", scores)


@dataclass(frozen=True)
class DynamicQuerySchedule:
    """Immutable query births for all three Deform360 updates."""

    update_frames: np.ndarray
    birth_frames: np.ndarray
    entity_ids: np.ndarray
    predicted_motion_m: np.ndarray
    predicted_visible_views: np.ndarray
    information_gain: np.ndarray
    config: DynamicQueryConfig
    camera_panel: CameraPanel
    physical_prefix_sha256: str
    graph_basis_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        update = _readonly(self.update_frames, dtype=np.int64)
        birth = _readonly(self.birth_frames, dtype=np.int64)
        entities = _readonly(self.entity_ids, dtype=np.int64)
        motion = _readonly(self.predicted_motion_m, dtype=np.float64)
        visible = _readonly(self.predicted_visible_views, dtype=np.int64)
        gain = _readonly(self.information_gain, dtype=np.float64)
        row_count = len(entities)
        for name, values in (
            ("update_frames", update),
            ("birth_frames", birth),
            ("predicted_motion_m", motion),
            ("predicted_visible_views", visible),
            ("information_gain", gain),
        ):
            _require(values.shape == (row_count,), f"{name} shape changed")
        _require(row_count > 0, "query schedule is empty")
        _require(
            isinstance(self.config, DynamicQueryConfig),
            "query schedule config is invalid",
        )
        _require(
            len(set(map(int, entities))) == row_count,
            "query identities are reused across birth waves",
        )
        _require(np.all(birth <= update), "query birth follows its update")
        _require(np.all(motion > 0.0), "query motion is not positive")
        _require(np.all(visible >= 3), "query lacks three-view support")
        _require(
            np.all(np.isfinite(gain) & (gain > 0.0)),
            "invalid information gain",
        )
        for name, digest in (
            ("physical_prefix_sha256", self.physical_prefix_sha256),
            ("graph_basis_sha256", self.graph_basis_sha256),
            ("artifact_sha256", self.artifact_sha256),
        ):
            _require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"{name} is not a SHA-256 digest",
            )
        object.__setattr__(self, "update_frames", update)
        object.__setattr__(self, "birth_frames", birth)
        object.__setattr__(self, "entity_ids", entities)
        object.__setattr__(self, "predicted_motion_m", motion)
        object.__setattr__(self, "predicted_visible_views", visible)
        object.__setattr__(self, "information_gain", gain)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360DynamicTAPNextPPQuerySchedule",
            "protocol_id": PROTOCOL_ID,
            "config": asdict(self.config),
            "update_frames": self.update_frames.tolist(),
            "birth_frames": self.birth_frames.tolist(),
            "entity_ids": self.entity_ids.tolist(),
            "predicted_motion_m": self.predicted_motion_m.tolist(),
            "predicted_visible_views": self.predicted_visible_views.tolist(),
            "information_gain": self.information_gain.tolist(),
            "camera_panel": {
                "camera_indices": self.camera_panel.camera_indices.tolist(),
                "camera_names": list(self.camera_panel.camera_names),
                "frame_zero_coverage": self.camera_panel.frame_zero_coverage.tolist(),
                "selection_scores": self.camera_panel.selection_scores.tolist(),
            },
            "physical_prefix_sha256": self.physical_prefix_sha256,
            "graph_basis_sha256": self.graph_basis_sha256,
            "information_boundary": {
                "maximum_physical_frame_read": max(UPDATE_FRAMES),
                "observed_object_trajectory_read": False,
                "target_metric_read": False,
                "future_frame_after_update_used_for_that_update": False,
            },
        }


def projection_matrices(
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Return calibrated world-to-pixel projection matrices."""

    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    _require(
        matrices.ndim == 3 and matrices.shape[1:] == (3, 3),
        "intrinsics must have shape (C, 3, 3)",
    )
    _require(
        poses.shape == (len(matrices), 4, 4),
        "camera_to_world must have shape (C, 4, 4)",
    )
    _require(
        np.all(np.isfinite(matrices)) and np.all(np.isfinite(poses)),
        "camera calibration is not finite",
    )
    return np.asarray(
        [
            intrinsic @ np.linalg.inv(pose)[:3]
            for intrinsic, pose in zip(matrices, poses, strict=True)
        ],
        dtype=np.float64,
    )


def project_visibility(
    points_world_m: np.ndarray,
    projections: np.ndarray,
    image_shapes_hw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points and return pixels, projective depth, and image visibility."""

    points = np.asarray(points_world_m, dtype=np.float64)
    matrices = np.asarray(projections, dtype=np.float64)
    shapes = np.asarray(image_shapes_hw, dtype=np.int64)
    _require(
        points.ndim == 2 and points.shape[1] == 3,
        "points must have shape (N, 3)",
    )
    _require(
        matrices.ndim == 3 and matrices.shape[1:] == (3, 4),
        "projections must have shape (C, 3, 4)",
    )
    _require(shapes.shape == (len(matrices), 2), "image shapes must be (C, 2)")
    homogeneous_points = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    projected = np.einsum("cij,nj->cni", matrices, homogeneous_points)
    depth = projected[..., 2]
    pixels = np.full(projected.shape[:2] + (2,), np.nan, dtype=np.float64)
    positive = np.isfinite(depth) & (depth > 1e-9)
    np.divide(
        projected[..., :2],
        depth[..., None],
        out=pixels,
        where=positive[..., None],
    )
    height = shapes[:, 0, None]
    width = shapes[:, 1, None]
    visible = (
        positive
        & np.all(np.isfinite(pixels), axis=-1)
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] <= width - 1)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] <= height - 1)
    )
    return pixels, depth, visible


def select_camera_panel(
    frame_zero_points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shapes_hw: np.ndarray,
    camera_names: Sequence[str],
    *,
    config: DynamicQueryConfig | None = None,
) -> CameraPanel:
    """Select cameras by frame-zero coverage and angular diversity."""

    cfg = config or DynamicQueryConfig()
    points = np.asarray(frame_zero_points_world_m, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    names = tuple(map(str, camera_names))
    _require(len(names) == len(poses), "camera names differ from calibration")
    _require(len(set(names)) == len(names), "camera names are not unique")
    projections = projection_matrices(intrinsics, poses)
    _, _, visible = project_visibility(points, projections, image_shapes_hw)
    coverage = np.mean(visible, axis=1)
    eligible = np.flatnonzero(coverage > 0.0)
    _require(
        len(eligible) >= cfg.minimum_eligible_camera_count,
        "too few frame-zero eligible cameras",
    )

    centroid = np.mean(points, axis=0)
    directions = poses[:, :3, 3] - centroid[None]
    norms = np.linalg.norm(directions, axis=1)
    _require(np.all(norms[eligible] > 1e-9), "camera center meets object centroid")
    directions = directions / np.maximum(norms[:, None], 1e-12)

    selected: list[int] = []
    selected_scores: list[float] = []
    remaining = set(map(int, eligible))
    while len(selected) < cfg.selected_camera_count:
        candidates: list[tuple[float, str, int]] = []
        for camera in remaining:
            if selected:
                dots = np.clip(
                    directions[selected] @ directions[camera],
                    -1.0,
                    1.0,
                )
                diversity = float(np.min(np.arccos(dots)) / np.pi)
            else:
                diversity = 0.0
            score = (
                cfg.coverage_weight * float(coverage[camera])
                + cfg.angular_diversity_weight * diversity
            )
            candidates.append((-score, names[camera], camera))
        _, _, choice = min(candidates)
        if selected:
            dots = np.clip(directions[selected] @ directions[choice], -1.0, 1.0)
            diversity = float(np.min(np.arccos(dots)) / np.pi)
        else:
            diversity = 0.0
        selected_scores.append(
            cfg.coverage_weight * float(coverage[choice])
            + cfg.angular_diversity_weight * diversity
        )
        selected.append(choice)
        remaining.remove(choice)

    return CameraPanel(
        camera_indices=np.asarray(selected, dtype=np.int64),
        camera_names=tuple(names[index] for index in selected),
        frame_zero_coverage=coverage[selected],
        selection_scores=np.asarray(selected_scores, dtype=np.float64),
    )


def _validated_graph_basis(
    graph_basis: np.ndarray,
    *,
    node_count: int,
    rank: int,
) -> np.ndarray:
    basis = np.asarray(graph_basis, dtype=np.float64)
    if basis.ndim == 2 and basis.shape[0] == 3 * node_count:
        basis = basis.reshape(node_count, 3, basis.shape[1])
    _require(
        basis.ndim == 3
        and basis.shape[:2] == (node_count, 3)
        and basis.shape[2] >= rank,
        "graph basis must have shape (N, 3, R>=rank)",
    )
    basis = basis[:, :, :rank]
    _require(np.all(np.isfinite(basis)), "graph basis is not finite")
    return basis


def _select_one_birth(
    physical_positions_m: np.ndarray,
    graph_basis: np.ndarray,
    projections: np.ndarray,
    image_shapes_hw: np.ndarray,
    selected_cameras: np.ndarray,
    candidate_entity_ids: np.ndarray,
    already_selected: set[int],
    *,
    birth_frame: int,
    update_frame: int,
    config: DynamicQueryConfig,
) -> list[tuple[int, float, int, float]]:
    birth_positions = physical_positions_m[birth_frame]
    update_positions = physical_positions_m[update_frame]
    _, _, visible_birth = project_visibility(
        birth_positions,
        projections[selected_cameras],
        image_shapes_hw[selected_cameras],
    )
    _, _, visible_update = project_visibility(
        update_positions,
        projections[selected_cameras],
        image_shapes_hw[selected_cameras],
    )
    visible_count = np.sum(visible_birth & visible_update, axis=0)
    motion = np.linalg.norm(update_positions - birth_positions, axis=1)

    eligible = [
        int(entity)
        for entity in candidate_entity_ids
        if int(entity) not in already_selected
        and motion[int(entity)] >= config.minimum_predicted_motion_m
        and visible_count[int(entity)] >= config.minimum_predicted_visible_views
    ]
    _require(
        len(eligible) >= config.queries_per_birth,
        "too few active multiview-visible query candidates",
    )

    information = np.eye(config.graph_basis_rank, dtype=np.float64)
    information *= config.information_ridge
    chosen: list[tuple[int, float, int, float]] = []
    while len(chosen) < config.queries_per_birth:
        candidates: list[tuple[float, int, float]] = []
        current_logdet = float(np.linalg.slogdet(information)[1])
        chosen_ids = [item[0] for item in chosen]
        for entity in eligible:
            if entity in chosen_ids:
                continue
            if chosen_ids and config.minimum_spatial_separation_m > 0.0:
                distance = np.linalg.norm(
                    birth_positions[chosen_ids] - birth_positions[entity],
                    axis=1,
                )
                if np.any(distance < config.minimum_spatial_separation_m):
                    continue
            relative_motion = motion[entity] / config.minimum_predicted_motion_m
            motion_weight = float(
                np.clip(relative_motion, 1.0, config.maximum_motion_weight)
            )
            jacobian = graph_basis[entity]
            candidate_information = (
                information + motion_weight * (jacobian.T @ jacobian)
            )
            sign, candidate_logdet = np.linalg.slogdet(candidate_information)
            if sign <= 0.0 or not np.isfinite(candidate_logdet):
                continue
            gain = float(candidate_logdet - current_logdet)
            if gain > 0.0:
                candidates.append((-gain, entity, gain))
        _require(
            bool(candidates),
            "spatial or information constraints prevent a complete query birth",
        )
        _, entity, gain = min(candidates, key=lambda item: (item[0], item[1]))
        relative_motion = motion[entity] / config.minimum_predicted_motion_m
        motion_weight = float(
            np.clip(relative_motion, 1.0, config.maximum_motion_weight)
        )
        jacobian = graph_basis[entity]
        information += motion_weight * (jacobian.T @ jacobian)
        chosen.append(
            (
                entity,
                float(motion[entity]),
                int(visible_count[entity]),
                gain,
            )
        )
    return chosen


def build_dynamic_query_schedule(
    physical_positions_m: np.ndarray,
    graph_basis: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shapes_hw: np.ndarray,
    camera_names: Sequence[str],
    *,
    candidate_entity_ids: np.ndarray | None = None,
    update_frames: tuple[int, ...] = UPDATE_FRAMES,
    query_birth_frames: tuple[tuple[int, ...], ...] = QUERY_BIRTH_FRAMES,
    config: DynamicQueryConfig | None = None,
) -> DynamicQuerySchedule:
    """Build all query births without observed object outcomes."""

    cfg = config or DynamicQueryConfig()
    positions = np.asarray(physical_positions_m, dtype=np.float64)
    _require(
        positions.ndim == 3 and positions.shape[2] == 3,
        "physical positions must have shape (T, N, 3)",
    )
    _require(np.all(np.isfinite(positions)), "physical positions are not finite")
    _require(update_frames == UPDATE_FRAMES, "Deform360 update frames changed")
    _require(
        query_birth_frames == QUERY_BIRTH_FRAMES,
        "dynamic query birth frames changed",
    )
    _require(
        len(positions) > max(update_frames),
        "physical rollout does not reach the final causal update",
    )
    _require(
        len(update_frames) == len(query_birth_frames),
        "birth schedule differs from update schedule",
    )
    node_count = positions.shape[1]
    basis = _validated_graph_basis(
        graph_basis,
        node_count=node_count,
        rank=cfg.graph_basis_rank,
    )
    candidates = (
        np.arange(node_count, dtype=np.int64)
        if candidate_entity_ids is None
        else np.asarray(candidate_entity_ids, dtype=np.int64)
    )
    _require(
        candidates.ndim == 1
        and len(candidates) > 0
        and np.all((candidates >= 0) & (candidates < node_count))
        and len(set(map(int, candidates))) == len(candidates),
        "candidate entity IDs are invalid",
    )
    candidates = np.sort(candidates)
    panel = select_camera_panel(
        positions[0],
        intrinsics,
        camera_to_world,
        image_shapes_hw,
        camera_names,
        config=cfg,
    )
    projections = projection_matrices(intrinsics, camera_to_world)

    rows: list[tuple[int, int, int, float, int, float]] = []
    already_selected: set[int] = set()
    for update, births in zip(update_frames, query_birth_frames, strict=True):
        _require(
            tuple(sorted(births)) == births and births[-1] <= update,
            "birth frames must be ordered before their update",
        )
        for birth in births:
            chosen = _select_one_birth(
                positions,
                basis,
                projections,
                np.asarray(image_shapes_hw, dtype=np.int64),
                panel.camera_indices,
                candidates,
                already_selected,
                birth_frame=birth,
                update_frame=update,
                config=cfg,
            )
            for entity, motion, visible, gain in chosen:
                already_selected.add(entity)
                rows.append((update, birth, entity, motion, visible, gain))

    physical_prefix = positions[: max(update_frames) + 1]
    descriptor: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicTAPNextPPQuerySchedule",
        "protocol_id": PROTOCOL_ID,
        "config": asdict(cfg),
        "update_frames": [row[0] for row in rows],
        "birth_frames": [row[1] for row in rows],
        "entity_ids": [row[2] for row in rows],
        "predicted_motion_m": [row[3] for row in rows],
        "predicted_visible_views": [row[4] for row in rows],
        "information_gain": [row[5] for row in rows],
        "camera_panel": {
            "camera_indices": panel.camera_indices.tolist(),
            "camera_names": list(panel.camera_names),
            "frame_zero_coverage": panel.frame_zero_coverage.tolist(),
            "selection_scores": panel.selection_scores.tolist(),
        },
        "physical_prefix_sha256": array_sha256(physical_prefix),
        "graph_basis_sha256": array_sha256(basis),
        "information_boundary": {
            "maximum_physical_frame_read": max(update_frames),
            "observed_object_trajectory_read": False,
            "target_metric_read": False,
            "future_frame_after_update_used_for_that_update": False,
        },
    }
    artifact_sha256 = _canonical_sha256(descriptor, digest_key="artifact_sha256")
    schedule = DynamicQuerySchedule(
        update_frames=np.asarray([row[0] for row in rows], dtype=np.int64),
        birth_frames=np.asarray([row[1] for row in rows], dtype=np.int64),
        entity_ids=np.asarray([row[2] for row in rows], dtype=np.int64),
        predicted_motion_m=np.asarray([row[3] for row in rows], dtype=np.float64),
        predicted_visible_views=np.asarray([row[4] for row in rows], dtype=np.int64),
        information_gain=np.asarray([row[5] for row in rows], dtype=np.float64),
        config=cfg,
        camera_panel=panel,
        physical_prefix_sha256=descriptor["physical_prefix_sha256"],
        graph_basis_sha256=descriptor["graph_basis_sha256"],
        artifact_sha256=artifact_sha256,
    )
    _require(
        _canonical_sha256(schedule.descriptor(), digest_key="artifact_sha256")
        == schedule.artifact_sha256,
        "query schedule descriptor changed after construction",
    )
    return schedule


__all__ = [
    "PROTOCOL_ID",
    "QUERY_BIRTH_FRAMES",
    "UPDATE_FRAMES",
    "CameraPanel",
    "DynamicQueryConfig",
    "DynamicQuerySchedule",
    "build_dynamic_query_schedule",
    "project_visibility",
    "projection_matrices",
    "select_camera_panel",
]
