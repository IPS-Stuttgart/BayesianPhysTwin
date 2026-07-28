"""Target-free Deform360 schedule with active and sentinel graph identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform360_dynamic_query import (
    UPDATE_FRAMES,
    CameraPanel,
    DynamicQueryConfig,
    project_visibility,
    projection_matrices,
    select_camera_panel,
)
from .observation_belief import array_sha256
from .phystwin_active_queries import PhysicsGuidedQueryConfig
from .phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
    MotionStratifiedQueryConfig,
    plan_motion_stratified_queries,
)

PROTOCOL_ID = "deform360-dynamic-tapnextpp-sentinel-v5-source-development"
SHORT_HORIZON_PROTOCOL_ID = (
    "deform360-dynamic-tapnextpp-short-sentinel-v6-source-development"
)
PREFIX_SUPPORT_PROTOCOL_ID = (
    "deform360-dynamic-tapnextpp-prefix-support-sentinel-v7-source-development"
)
PREFIX_END_FRAME = max(UPDATE_FRAMES)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Deform360SentinelQueryConfig:
    """Source-frozen query and visibility settings for one endpoint update."""

    selected_camera_count: int = 8
    minimum_eligible_camera_count: int = 8
    coverage_weight: float = 1.0
    angular_diversity_weight: float = 0.25
    total_query_count: int = 12
    sentinel_query_count: int = 3
    active_minimum_motion_m: float = 0.002
    sentinel_maximum_motion_m: float = 0.0005
    minimum_camera_support: int = 3
    graph_basis_rank: int = 8
    query_birth_frame: int = 0
    query_update_frame: int = PREFIX_END_FRAME
    protocol_id: str = PROTOCOL_ID

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
        _require(
            self.total_query_count >= 3,
            "total query count must support active and sentinel roles",
        )
        _require(
            1 <= self.sentinel_query_count < self.total_query_count,
            "sentinel count must reserve a strict budget subset",
        )
        _require(
            np.isfinite(self.active_minimum_motion_m)
            and self.active_minimum_motion_m > 0.0,
            "active motion threshold must be positive",
        )
        _require(
            np.isfinite(self.sentinel_maximum_motion_m)
            and 0.0 <= self.sentinel_maximum_motion_m
            < self.active_minimum_motion_m,
            "sentinel motion threshold must be separated from active motion",
        )
        _require(
            3 <= self.minimum_camera_support <= self.selected_camera_count,
            "sentinel schedule requires three-view support",
        )
        _require(self.graph_basis_rank >= 1, "graph basis rank must be positive")
        _require(
            0 <= self.query_birth_frame < self.query_update_frame,
            "query birth must precede its update",
        )
        _require(
            self.query_update_frame == PREFIX_END_FRAME,
            "sentinel update frame changed",
        )
        _require(
            self.protocol_id
            in {
                PROTOCOL_ID,
                SHORT_HORIZON_PROTOCOL_ID,
                PREFIX_SUPPORT_PROTOCOL_ID,
            },
            "sentinel protocol ID is not registered",
        )

    @property
    def active_query_count(self) -> int:
        return self.total_query_count - self.sentinel_query_count


@dataclass(frozen=True)
class Deform360SentinelQuerySchedule:
    """Immutable frame-zero query schedule with explicit nuisance roles."""

    update_frames: np.ndarray
    birth_frames: np.ndarray
    entity_ids: np.ndarray
    query_roles: np.ndarray
    predicted_motion_m: np.ndarray
    predicted_visible_views: np.ndarray
    information_gain: np.ndarray
    config: Deform360SentinelQueryConfig
    camera_panel: CameraPanel
    physical_prefix_sha256: str
    graph_basis_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        update = _readonly(self.update_frames, dtype=np.int64)
        birth = _readonly(self.birth_frames, dtype=np.int64)
        entities = _readonly(self.entity_ids, dtype=np.int64)
        roles = _readonly(self.query_roles, dtype="<U8")
        motion = _readonly(self.predicted_motion_m, dtype=np.float64)
        visible = _readonly(self.predicted_visible_views, dtype=np.int64)
        gain = _readonly(self.information_gain, dtype=np.float64)
        row_count = len(entities)
        for name, values in (
            ("update_frames", update),
            ("birth_frames", birth),
            ("query_roles", roles),
            ("predicted_motion_m", motion),
            ("predicted_visible_views", visible),
            ("information_gain", gain),
        ):
            _require(values.shape == (row_count,), f"{name} shape changed")
        _require(
            row_count == self.config.total_query_count,
            "schedule does not meet its fixed query budget",
        )
        _require(
            len(np.unique(entities)) == row_count,
            "query identities repeat",
        )
        _require(
            np.all(birth == self.config.query_birth_frame)
            and np.all(update == self.config.query_update_frame),
            "sentinel schedule differs from its declared branch interval",
        )
        _require(
            set(map(str, roles)) == {ACTIVE_QUERY_ROLE, SENTINEL_QUERY_ROLE},
            "query roles are incomplete or invalid",
        )
        active = roles == ACTIVE_QUERY_ROLE
        sentinel = roles == SENTINEL_QUERY_ROLE
        _require(
            int(np.sum(active)) == self.config.active_query_count
            and int(np.sum(sentinel)) == self.config.sentinel_query_count,
            "query role counts differ from the fixed budget",
        )
        _require(
            np.all(motion[active] >= self.config.active_minimum_motion_m)
            and np.all(
                motion[sentinel] <= self.config.sentinel_maximum_motion_m
            ),
            "query motion differs from its declared role",
        )
        _require(
            np.all(visible >= self.config.minimum_camera_support),
            "query lacks three-view geometric support",
        )
        _require(
            np.all(np.isfinite(gain) & (gain >= 0.0)),
            "query information gain is invalid",
        )
        for name, digest in (
            ("physical_prefix_sha256", self.physical_prefix_sha256),
            ("graph_basis_sha256", self.graph_basis_sha256),
            ("artifact_sha256", self.artifact_sha256),
        ):
            _require(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"{name} is not a SHA-256 digest",
            )
        object.__setattr__(self, "update_frames", update)
        object.__setattr__(self, "birth_frames", birth)
        object.__setattr__(self, "entity_ids", entities)
        object.__setattr__(self, "query_roles", roles)
        object.__setattr__(self, "predicted_motion_m", motion)
        object.__setattr__(self, "predicted_visible_views", visible)
        object.__setattr__(self, "information_gain", gain)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360SentinelQuerySchedule",
            "protocol_id": self.config.protocol_id,
            "config": asdict(self.config),
            "update_frames": self.update_frames.tolist(),
            "birth_frames": self.birth_frames.tolist(),
            "entity_ids": self.entity_ids.tolist(),
            "query_roles": self.query_roles.tolist(),
            "predicted_motion_m": self.predicted_motion_m.tolist(),
            "predicted_visible_views": self.predicted_visible_views.tolist(),
            "information_gain": self.information_gain.tolist(),
            "camera_panel": {
                "camera_indices": self.camera_panel.camera_indices.tolist(),
                "camera_names": list(self.camera_panel.camera_names),
                "frame_zero_coverage": (
                    self.camera_panel.frame_zero_coverage.tolist()
                ),
                "selection_scores": self.camera_panel.selection_scores.tolist(),
            },
            "physical_prefix_sha256": self.physical_prefix_sha256,
            "graph_basis_sha256": self.graph_basis_sha256,
            "information_boundary": {
                "physical_frame_interval_read": [
                    self.config.query_birth_frame,
                    self.config.query_update_frame,
                ],
                "maximum_physical_frame_read": self.config.query_update_frame,
                "maximum_observed_tracker_frame_used_for_planning": None,
                "observed_object_trajectory_read": False,
                "target_metric_read": False,
                "future_frame_after_update_used_for_that_update": False,
            },
        }


def _node_mode_basis(
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
        "graph_basis must have shape (N, 3, R>=rank)",
    )
    basis = basis[:, :, :rank]
    _require(np.all(np.isfinite(basis)), "graph basis is not finite")
    node_basis = np.linalg.norm(basis, axis=1)
    nonempty = np.linalg.norm(node_basis, axis=0) > 0.0
    _require(np.any(nonempty), "graph basis has no nonempty node mode")
    return node_basis[:, nonempty]


def build_deform360_sentinel_query_schedule(
    physical_positions_m: np.ndarray,
    graph_basis: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shapes_hw: np.ndarray,
    camera_names: Sequence[str],
    *,
    candidate_entity_ids: np.ndarray | None = None,
    config: Deform360SentinelQueryConfig | None = None,
) -> Deform360SentinelQuerySchedule:
    """Build a target-free endpoint schedule from the physical prefix only."""

    cfg = config or Deform360SentinelQueryConfig()
    positions = np.asarray(physical_positions_m, dtype=np.float64)
    _require(
        positions.ndim == 3
        and positions.shape[2] == 3
        and len(positions) > PREFIX_END_FRAME,
        "physical positions must reach the prefix endpoint",
    )
    prefix = positions[
        cfg.query_birth_frame : cfg.query_update_frame + 1
    ]
    _require(np.all(np.isfinite(prefix)), "physical prefix is not finite")
    node_count = positions.shape[1]
    mode_basis = _node_mode_basis(
        graph_basis,
        node_count=node_count,
        rank=cfg.graph_basis_rank,
    )
    camera_config = DynamicQueryConfig(
        selected_camera_count=cfg.selected_camera_count,
        minimum_eligible_camera_count=cfg.minimum_eligible_camera_count,
        coverage_weight=cfg.coverage_weight,
        angular_diversity_weight=cfg.angular_diversity_weight,
    )
    panel = select_camera_panel(
        prefix[0],
        intrinsics,
        camera_to_world,
        image_shapes_hw,
        camera_names,
        config=camera_config,
    )
    projections = projection_matrices(intrinsics, camera_to_world)
    selected_projections = projections[panel.camera_indices]
    selected_shapes = np.asarray(image_shapes_hw, dtype=np.int64)[
        panel.camera_indices
    ]
    projected_pixels: list[np.ndarray] = []
    predicted_support: list[np.ndarray] = []
    for frame in prefix:
        pixels, _, visible = project_visibility(
            frame,
            selected_projections,
            selected_shapes,
        )
        projected_pixels.append(pixels)
        predicted_support.append(visible.astype(np.float64))
    pixels_ctn = np.transpose(np.asarray(projected_pixels), (1, 0, 2, 3))
    support_ctn = np.transpose(np.asarray(predicted_support), (1, 0, 2))
    active_config = PhysicsGuidedQueryConfig(
        query_count=cfg.active_query_count,
        maximum_reseeds=0,
        minimum_motion_m=cfg.active_minimum_motion_m,
        minimum_camera_support=cfg.minimum_camera_support,
        support_probability_threshold=0.5,
        contact_exclusion_fraction=0.0,
    )
    stratified_config = MotionStratifiedQueryConfig(
        total_query_count=cfg.total_query_count,
        sentinel_query_count=cfg.sentinel_query_count,
        sentinel_maximum_motion_m=cfg.sentinel_maximum_motion_m,
        sentinel_maximum_reseeds=0,
    )
    plan = plan_motion_stratified_queries(
        prefix,
        pixels_ctn,
        support_ctn,
        mode_basis=mode_basis,
        candidate_ids=candidate_entity_ids,
        active_config=active_config,
        config=stratified_config,
    )
    _require(
        plan.initial_budget_met,
        "motion-stratified frame-zero query budget is incomplete",
    )
    entity_ids = np.concatenate(
        (plan.active.node_ids, plan.sentinel.node_ids)
    )
    roles = np.concatenate(
        (
            np.full(len(plan.active.node_ids), ACTIVE_QUERY_ROLE, dtype="<U8"),
            np.full(
                len(plan.sentinel.node_ids),
                SENTINEL_QUERY_ROLE,
                dtype="<U8",
            ),
        )
    )
    visible_views = np.concatenate(
        (
            np.sum(plan.active.camera_mask, axis=1),
            np.sum(plan.sentinel.camera_mask, axis=1),
        )
    )
    information_gain = np.concatenate(
        (
            plan.active.mode_information_gain,
            plan.sentinel.mode_information_gain,
        )
    )
    motion = np.max(
        np.linalg.norm(prefix - prefix[0][None], axis=2),
        axis=0,
    )[entity_ids]
    descriptor: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SentinelQuerySchedule",
        "protocol_id": cfg.protocol_id,
        "config": asdict(cfg),
        "update_frames": [cfg.query_update_frame] * len(entity_ids),
        "birth_frames": [cfg.query_birth_frame] * len(entity_ids),
        "entity_ids": entity_ids.tolist(),
        "query_roles": roles.tolist(),
        "predicted_motion_m": motion.tolist(),
        "predicted_visible_views": visible_views.tolist(),
        "information_gain": information_gain.tolist(),
        "camera_panel": {
            "camera_indices": panel.camera_indices.tolist(),
            "camera_names": list(panel.camera_names),
            "frame_zero_coverage": panel.frame_zero_coverage.tolist(),
            "selection_scores": panel.selection_scores.tolist(),
        },
        "physical_prefix_sha256": array_sha256(prefix),
        "graph_basis_sha256": array_sha256(mode_basis),
        "information_boundary": {
            "physical_frame_interval_read": [
                cfg.query_birth_frame,
                cfg.query_update_frame,
            ],
            "maximum_physical_frame_read": cfg.query_update_frame,
            "maximum_observed_tracker_frame_used_for_planning": None,
            "observed_object_trajectory_read": False,
            "target_metric_read": False,
            "future_frame_after_update_used_for_that_update": False,
        },
    }
    artifact_sha256 = _canonical_sha256(
        descriptor,
        digest_key="artifact_sha256",
    )
    schedule = Deform360SentinelQuerySchedule(
        update_frames=np.full(
            len(entity_ids),
            cfg.query_update_frame,
            dtype=np.int64,
        ),
        birth_frames=np.full(
            len(entity_ids),
            cfg.query_birth_frame,
            dtype=np.int64,
        ),
        entity_ids=entity_ids,
        query_roles=roles,
        predicted_motion_m=motion,
        predicted_visible_views=visible_views,
        information_gain=information_gain,
        config=cfg,
        camera_panel=panel,
        physical_prefix_sha256=descriptor["physical_prefix_sha256"],
        graph_basis_sha256=descriptor["graph_basis_sha256"],
        artifact_sha256=artifact_sha256,
    )
    _require(
        _canonical_sha256(
            schedule.descriptor(),
            digest_key="artifact_sha256",
        )
        == schedule.artifact_sha256,
        "sentinel schedule descriptor changed after construction",
    )
    return schedule


__all__ = [
    "PREFIX_END_FRAME",
    "PREFIX_SUPPORT_PROTOCOL_ID",
    "PROTOCOL_ID",
    "SHORT_HORIZON_PROTOCOL_ID",
    "Deform360SentinelQueryConfig",
    "Deform360SentinelQuerySchedule",
    "build_deform360_sentinel_query_schedule",
]
