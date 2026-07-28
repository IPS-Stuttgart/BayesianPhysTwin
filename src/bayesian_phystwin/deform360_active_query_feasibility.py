"""Target-free feasibility audit for physics-guided Deform360 queries.

The audit asks a deliberately narrow question before any tracker is run:
can a frozen action-conditioned physical rollout supply a complete budget of
moving graph identities with independently depth-supported frame-zero query
locations?  It reads calibration plus frame-zero depth and masks only.  Future
object observations, tracker outputs, state updates, and evaluation targets are
not accepted by the API.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_dynamic_query import (
    CameraPanel,
    DynamicQueryConfig,
    project_visibility,
    projection_matrices,
    select_camera_panel,
)
from .observation_belief import array_sha256, file_sha256
from .phystwin_active_queries import (
    PhysicsGuidedQueryConfig,
    PhysicsGuidedQueryPlan,
    plan_physics_guided_queries,
)
from .tapnextpp_birth_association import (
    SET_VALUED_COVARIANCE_ASSOCIATION,
    BirthAssociationConfig,
    propose_birth_query_pixels,
)

PROTOCOL_ID = "deform360-active-query-feasibility-v10-source"
ARTIFACT_KIND = "Deform360ActiveQueryFeasibilityAudit"
ARCHIVE_FILENAME = "active_query_feasibility.npz"
REPORT_FILENAME = "active_query_feasibility.json"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
class ActiveQueryFeasibilityConfig:
    """Frozen target-free choices for the source feasibility audit."""

    selected_camera_count: int = 4
    minimum_eligible_camera_count: int = 4
    coverage_weight: float = 1.0
    angular_diversity_weight: float = 0.25
    query_count: int = 8
    maximum_reseeds: int = 0
    minimum_motion_m: float = 0.002
    minimum_camera_support: int = 2
    support_probability_threshold: float = 0.5
    graph_basis_rank: int = 8
    association_search_radius_px: int = 12
    association_depth_scale_m: float = 0.03
    association_minimum_candidate_count: int = 1

    def __post_init__(self) -> None:
        _require(
            self.selected_camera_count >= 3,
            "selected camera count must support panel diversity",
        )
        _require(
            self.minimum_eligible_camera_count >= self.selected_camera_count,
            "eligible camera gate precedes panel selection",
        )
        _require(
            self.minimum_camera_support >= 2
            and self.minimum_camera_support <= self.selected_camera_count,
            "camera support gate is invalid",
        )
        _require(self.query_count >= 1, "query count must be positive")
        _require(
            self.maximum_reseeds == 0,
            "feasibility audit must stop before tracker-driven reseeding",
        )
        _require(
            np.isfinite(self.minimum_motion_m)
            and self.minimum_motion_m > 0.0,
            "motion gate must be finite and positive",
        )
        _require(
            np.isfinite(self.support_probability_threshold)
            and 0.0 < self.support_probability_threshold <= 1.0,
            "support probability threshold must lie in (0, 1]",
        )
        _require(self.graph_basis_rank >= 1, "graph rank must be positive")
        BirthAssociationConfig(
            search_radius_px=self.association_search_radius_px,
            depth_scale_m=self.association_depth_scale_m,
            minimum_candidate_count=self.association_minimum_candidate_count,
            association_mode=SET_VALUED_COVARIANCE_ASSOCIATION,
        )


def readout_modes_to_node_basis(
    graph_basis: np.ndarray,
    *,
    node_count: int,
    rank: int,
) -> np.ndarray:
    """Convert vector readout modes into deterministic node information rows."""

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
    result = np.ascontiguousarray(node_basis[:, nonempty])
    result.setflags(write=False)
    return result


def _plan_arrays(plan: PhysicsGuidedQueryPlan) -> dict[str, np.ndarray]:
    return {
        "plan_node_ids": plan.node_ids,
        "plan_seed_frames": plan.seed_frames,
        "plan_replaces_node_ids": plan.replaces_node_ids,
        "plan_camera_mask": plan.camera_mask,
        "plan_seed_pixels_xy": plan.seed_pixels_xy,
        "plan_motion_score": plan.motion_score,
        "plan_visibility_score": plan.visibility_score,
        "plan_mode_information_gain": plan.mode_information_gain,
        "plan_spatial_diversity_score": plan.spatial_diversity_score,
        "plan_contact_distance_score": plan.contact_distance_score,
        "plan_total_score": plan.total_score,
    }


def _frame_zero_plan(
    plan: PhysicsGuidedQueryPlan,
) -> PhysicsGuidedQueryPlan:
    """Discard later vacancy fills from this initial-budget-only audit."""

    keep = plan.seed_frames == 0
    return PhysicsGuidedQueryPlan(
        node_ids=plan.node_ids[keep],
        seed_frames=plan.seed_frames[keep],
        replaces_node_ids=plan.replaces_node_ids[keep],
        camera_mask=plan.camera_mask[keep],
        seed_pixels_xy=plan.seed_pixels_xy[keep],
        motion_score=plan.motion_score[keep],
        visibility_score=plan.visibility_score[keep],
        mode_information_gain=plan.mode_information_gain[keep],
        spatial_diversity_score=plan.spatial_diversity_score[keep],
        contact_distance_score=plan.contact_distance_score[keep],
        total_score=plan.total_score[keep],
        requested_active_queries=plan.requested_active_queries,
        minimum_camera_support=plan.minimum_camera_support,
        prefix_frame_count=plan.prefix_frame_count,
    )


@dataclass(frozen=True)
class ActiveQueryFeasibilityAudit:
    """Immutable target-free query-budget evidence for one source case."""

    config: ActiveQueryFeasibilityConfig
    camera_panel: CameraPanel
    plan: PhysicsGuidedQueryPlan
    candidate_entity_ids: np.ndarray
    candidate_support_count: np.ndarray
    association_query_points_xy: np.ndarray
    association_valid: np.ndarray
    association_probability: np.ndarray
    association_entropy: np.ndarray
    association_candidate_count: np.ndarray
    association_covariance_px2: np.ndarray
    input_array_sha256: Mapping[str, str]
    artifact_sha256: str

    def __post_init__(self) -> None:
        candidates = np.asarray(self.candidate_entity_ids, dtype=np.int64).copy()
        support = np.asarray(self.candidate_support_count, dtype=np.int64).copy()
        query = np.asarray(
            self.association_query_points_xy,
            dtype=np.float64,
        ).copy()
        valid = np.asarray(self.association_valid, dtype=bool).copy()
        probability = np.asarray(
            self.association_probability,
            dtype=np.float64,
        ).copy()
        entropy = np.asarray(self.association_entropy, dtype=np.float64).copy()
        count = np.asarray(
            self.association_candidate_count,
            dtype=np.int64,
        ).copy()
        covariance = np.asarray(
            self.association_covariance_px2,
            dtype=np.float64,
        ).copy()
        camera_count = self.config.selected_camera_count
        _require(
            self.camera_panel.camera_indices.shape == (camera_count,),
            "camera panel count changed",
        )
        _require(candidates.ndim == 1, "candidate IDs must be a vector")
        _require(
            support.shape == candidates.shape
            and np.all(support >= self.config.minimum_camera_support),
            "candidate support counts are invalid",
        )
        _require(
            len(np.unique(candidates)) == len(candidates)
            and np.all(candidates >= 0),
            "candidate IDs are invalid",
        )
        _require(
            query.ndim == 3
            and query.shape[0] == camera_count
            and query.shape[2] == 2,
            "association query array must have shape (C, N, 2)",
        )
        node_count = query.shape[1]
        _require(
            valid.shape == probability.shape == entropy.shape == count.shape
            == (camera_count, node_count),
            "association arrays have inconsistent shapes",
        )
        _require(
            covariance.shape == (camera_count, node_count, 2, 2),
            "association covariance shape changed",
        )
        _require(
            np.all(np.isfinite(probability))
            and np.all((probability >= 0.0) & (probability <= 1.0)),
            "association probabilities are invalid",
        )
        _require(
            np.all(np.isfinite(entropy))
            and np.all((entropy >= 0.0) & (entropy <= 1.0)),
            "association entropies are invalid",
        )
        _require(np.all(count >= 0), "association candidate counts are invalid")
        _require(
            set(map(int, self.plan.node_ids)).issubset(set(map(int, candidates))),
            "query plan uses an unadmitted association candidate",
        )
        _require(
            self.plan.requested_active_queries == self.config.query_count
            and self.plan.minimum_camera_support
            == self.config.minimum_camera_support
            and self.plan.reseed_count == 0,
            "query plan differs from the feasibility protocol",
        )
        _require(
            all(_valid_digest(value) for value in self.input_array_sha256.values()),
            "input array digest is invalid",
        )
        _require(_valid_digest(self.artifact_sha256), "artifact digest is invalid")
        for values in (
            candidates,
            support,
            query,
            valid,
            probability,
            entropy,
            count,
            covariance,
        ):
            values.setflags(write=False)
        object.__setattr__(self, "candidate_entity_ids", candidates)
        object.__setattr__(self, "candidate_support_count", support)
        object.__setattr__(self, "association_query_points_xy", query)
        object.__setattr__(self, "association_valid", valid)
        object.__setattr__(self, "association_probability", probability)
        object.__setattr__(self, "association_entropy", entropy)
        object.__setattr__(self, "association_candidate_count", count)
        object.__setattr__(self, "association_covariance_px2", covariance)
        object.__setattr__(
            self,
            "input_array_sha256",
            dict(sorted(self.input_array_sha256.items())),
        )

    @property
    def admitted(self) -> bool:
        return self.plan.initial_budget_met

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "candidate_entity_ids": self.candidate_entity_ids,
            "candidate_support_count": self.candidate_support_count,
            "association_query_points_xy": self.association_query_points_xy,
            "association_valid": self.association_valid,
            "association_probability": self.association_probability,
            "association_entropy": self.association_entropy,
            "association_candidate_count": self.association_candidate_count,
            "association_covariance_px2": self.association_covariance_px2,
            **_plan_arrays(self.plan),
        }

    def descriptor(self) -> dict[str, Any]:
        arrays = self.arrays()
        return {
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "protocol_id": PROTOCOL_ID,
            "config": asdict(self.config),
            "admitted": self.admitted,
            "candidate_entity_count": len(self.candidate_entity_ids),
            "selected_entity_ids": self.plan.node_ids.tolist(),
            "selected_camera_support_count": np.sum(
                self.plan.camera_mask,
                axis=1,
            ).tolist(),
            "initial_query_count": self.plan.initial_query_count,
            "requested_query_count": self.plan.requested_active_queries,
            "camera_panel": {
                "camera_indices": self.camera_panel.camera_indices.tolist(),
                "camera_names": list(self.camera_panel.camera_names),
                "frame_zero_coverage": (
                    self.camera_panel.frame_zero_coverage.tolist()
                ),
                "selection_scores": self.camera_panel.selection_scores.tolist(),
            },
            "input_array_sha256": dict(self.input_array_sha256),
            "output_array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(arrays.items())
            },
            "information_boundary": {
                "object_observation_frames_used": [0],
                "known_future_robot_action_read": True,
                "maximum_physical_frame_read": self.plan.prefix_frame_count - 1,
                "future_object_rgb_read": False,
                "future_object_depth_or_mask_read": False,
                "tracker_output_read": False,
                "candidate_state_update_constructed": False,
                "future_identity_or_metric_read": False,
                "held_v8_artifact_or_process_access": False,
                "v1_sealed_target_access": False,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def build_active_query_feasibility_audit(
    physical_rollout_m: np.ndarray,
    graph_basis: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shapes_hw: np.ndarray,
    camera_names: Sequence[str],
    frame_zero_depths_m: np.ndarray,
    frame_zero_object_masks: np.ndarray,
    *,
    config: ActiveQueryFeasibilityConfig | None = None,
) -> ActiveQueryFeasibilityAudit:
    """Build a query-budget audit from predictions and frame-zero geometry."""

    cfg = config or ActiveQueryFeasibilityConfig()
    rollout = np.asarray(physical_rollout_m, dtype=np.float64)
    basis = np.asarray(graph_basis, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    shapes = np.asarray(image_shapes_hw, dtype=np.int64)
    depth = np.asarray(frame_zero_depths_m, dtype=np.float64)
    masks = np.asarray(frame_zero_object_masks, dtype=bool)
    _require(
        rollout.ndim == 3
        and rollout.shape[2] == 3
        and rollout.shape[0] >= 2
        and np.all(np.isfinite(rollout)),
        "physical rollout must have shape (T>=2, N, 3)",
    )
    camera_count = len(matrices)
    _require(
        matrices.shape == (camera_count, 3, 3)
        and poses.shape == (camera_count, 4, 4)
        and shapes.shape == (camera_count, 2),
        "camera calibration arrays have inconsistent shapes",
    )
    _require(
        depth.ndim == 3
        and depth.shape[0] == camera_count
        and masks.shape == depth.shape,
        "frame-zero depth and masks must have shape (C, H, W)",
    )
    _require(
        np.all(np.isfinite(depth)) and np.all(depth >= 0.0),
        "frame-zero depth is invalid",
    )
    _require(
        np.all(shapes == np.asarray(depth.shape[1:], dtype=np.int64)[None]),
        "frame-zero stream shape differs from calibration evidence",
    )
    node_count = rollout.shape[1]
    node_basis = readout_modes_to_node_basis(
        basis,
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
        rollout[0],
        matrices,
        poses,
        shapes,
        camera_names,
        config=camera_config,
    )
    projections = projection_matrices(matrices, poses)
    selected = panel.camera_indices
    selected_projections = projections[selected]
    selected_shapes = shapes[selected]

    frame_pixels: list[np.ndarray] = []
    frame_support: list[np.ndarray] = []
    for positions in rollout:
        pixels, _, visible = project_visibility(
            positions,
            selected_projections,
            selected_shapes,
        )
        frame_pixels.append(pixels)
        frame_support.append(visible.astype(np.float64))
    pixels_ctn = np.transpose(np.asarray(frame_pixels), (1, 0, 2, 3))
    support_ctn = np.transpose(np.asarray(frame_support), (1, 0, 2))

    association = propose_birth_query_pixels(
        rollout[0],
        selected_projections,
        poses[selected],
        depth[selected],
        masks[selected],
        config=BirthAssociationConfig(
            search_radius_px=cfg.association_search_radius_px,
            depth_scale_m=cfg.association_depth_scale_m,
            minimum_candidate_count=cfg.association_minimum_candidate_count,
            association_mode=SET_VALUED_COVARIANCE_ASSOCIATION,
        ),
    )
    association_support = (
        association["valid"]
        & (
            association["association_probability"]
            >= cfg.support_probability_threshold
        )
    )
    support_count = np.sum(association_support, axis=0)
    candidate_ids = np.flatnonzero(
        support_count >= cfg.minimum_camera_support
    ).astype(np.int64)

    pixels_ctn[:, 0] = np.where(
        association_support[..., None],
        association["query_points_xy"],
        np.nan,
    )
    support_ctn[:, 0] = np.where(
        association_support,
        association["association_probability"],
        0.0,
    )
    planner_config = PhysicsGuidedQueryConfig(
        query_count=cfg.query_count,
        maximum_reseeds=cfg.maximum_reseeds,
        minimum_motion_m=cfg.minimum_motion_m,
        minimum_camera_support=cfg.minimum_camera_support,
        support_probability_threshold=cfg.support_probability_threshold,
        contact_exclusion_fraction=0.0,
    )
    if len(candidate_ids):
        complete_plan = plan_physics_guided_queries(
            rollout,
            pixels_ctn,
            support_ctn,
            mode_basis=node_basis,
            candidate_ids=candidate_ids,
            config=planner_config,
        )
    else:
        complete_plan = plan_physics_guided_queries(
            rollout,
            pixels_ctn,
            support_ctn,
            mode_basis=node_basis,
            config=planner_config,
        )
    plan = _frame_zero_plan(complete_plan)
    input_hashes = {
        "physical_rollout_m": array_sha256(rollout),
        "graph_basis": array_sha256(basis),
        "intrinsics": array_sha256(matrices),
        "camera_to_world": array_sha256(poses),
        "image_shapes_hw": array_sha256(shapes),
        "frame_zero_depths_m": array_sha256(depth),
        "frame_zero_object_masks": array_sha256(masks),
    }
    provisional = ActiveQueryFeasibilityAudit(
        config=cfg,
        camera_panel=panel,
        plan=plan,
        candidate_entity_ids=candidate_ids,
        candidate_support_count=support_count[candidate_ids],
        association_query_points_xy=association["query_points_xy"],
        association_valid=association["valid"],
        association_probability=association["association_probability"],
        association_entropy=association["association_entropy"],
        association_candidate_count=association["candidate_count"],
        association_covariance_px2=association[
            "candidate_pixel_covariance_px2"
        ],
        input_array_sha256=input_hashes,
        artifact_sha256="0" * 64,
    )
    artifact_sha256 = _canonical_sha256(
        provisional.descriptor(),
        digest_key="artifact_sha256",
    )
    result = ActiveQueryFeasibilityAudit(
        config=provisional.config,
        camera_panel=provisional.camera_panel,
        plan=provisional.plan,
        candidate_entity_ids=provisional.candidate_entity_ids,
        candidate_support_count=provisional.candidate_support_count,
        association_query_points_xy=provisional.association_query_points_xy,
        association_valid=provisional.association_valid,
        association_probability=provisional.association_probability,
        association_entropy=provisional.association_entropy,
        association_candidate_count=provisional.association_candidate_count,
        association_covariance_px2=provisional.association_covariance_px2,
        input_array_sha256=provisional.input_array_sha256,
        artifact_sha256=artifact_sha256,
    )
    _require(
        _canonical_sha256(
            result.descriptor(),
            digest_key="artifact_sha256",
        )
        == result.artifact_sha256,
        "feasibility descriptor changed after construction",
    )
    return result


def write_active_query_feasibility_artifacts(
    output_dir: str | Path,
    audit: ActiveQueryFeasibilityAudit,
    *,
    case_id: str,
    repository_revision: str,
    protocol_path: str | Path,
    physical_manifest_path: str | Path,
    physical_archive_path: str | Path,
    camera_certificate_sha256: str,
) -> dict[str, Any]:
    """Seal one source feasibility result and its exact numeric arrays."""

    _require(case_id, "case ID is empty")
    _require(
        len(repository_revision) == 40
        and all(
            character in "0123456789abcdef"
            for character in repository_revision
        ),
        "repository revision is invalid",
    )
    _require(
        _valid_digest(camera_certificate_sha256),
        "camera certificate digest is invalid",
    )
    output = Path(output_dir).resolve()
    _require(not output.exists(), "feasibility output directory already exists")
    output.mkdir(parents=True)
    arrays = audit.arrays()
    archive_path = output / ARCHIVE_FILENAME
    temporary = output / (ARCHIVE_FILENAME + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **{
            name: np.ascontiguousarray(np.asarray(values))
            for name, values in arrays.items()
        },
    )
    temporary.replace(archive_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case": case_id,
        "status": "admitted" if audit.admitted else "abstained",
        "repository_revision": repository_revision,
        "audit": audit.descriptor(),
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "physical_manifest": file_sha256(physical_manifest_path),
            "physical_archive": file_sha256(physical_archive_path),
            "camera_certificate": camera_certificate_sha256,
        },
        "archive": {
            "filename": ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(arrays.items())
            },
        },
        "information_boundary": audit.descriptor()["information_boundary"],
    }
    report["result_sha256"] = _canonical_sha256(
        report,
        digest_key="result_sha256",
    )
    (output / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_active_query_feasibility_artifacts(output)
    return report


def validate_active_query_feasibility_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate a sealed feasibility report without reading any outcome."""

    output = Path(output_dir).resolve()
    report = json.loads(
        (output / REPORT_FILENAME).read_text(encoding="utf-8")
    )
    _require(
        report.get("artifact_kind") == ARTIFACT_KIND
        and report.get("protocol_id") == PROTOCOL_ID,
        "feasibility report belongs to another protocol",
    )
    _require(
        report.get("status") in {"admitted", "abstained"},
        "feasibility status is invalid",
    )
    _require(
        report.get("result_sha256")
        == _canonical_sha256(report, digest_key="result_sha256"),
        "feasibility report checksum changed",
    )
    audit = report.get("audit", {})
    _require(
        audit.get("artifact_sha256")
        == _canonical_sha256(audit, digest_key="artifact_sha256"),
        "feasibility audit checksum changed",
    )
    boundary = report.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_depth_or_mask_read") is False
        and boundary.get("tracker_output_read") is False
        and boundary.get("candidate_state_update_constructed") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False
        and boundary.get("v1_sealed_target_access") is False,
        "feasibility report crossed its information boundary",
    )
    archive = output / ARCHIVE_FILENAME
    _require(
        report.get("archive", {}).get("file_sha256") == file_sha256(archive),
        "feasibility archive checksum changed",
    )
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    expected = {
        name: array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    _require(
        expected == report["archive"]["array_sha256"]
        == audit["output_array_sha256"],
        "feasibility array checksum changed",
    )
    _require(
        (report["status"] == "admitted") is bool(audit["admitted"]),
        "feasibility status differs from audit decision",
    )
    return report, arrays


__all__ = [
    "ARCHIVE_FILENAME",
    "ARTIFACT_KIND",
    "PROTOCOL_ID",
    "REPORT_FILENAME",
    "ActiveQueryFeasibilityAudit",
    "ActiveQueryFeasibilityConfig",
    "build_active_query_feasibility_audit",
    "readout_modes_to_node_basis",
    "validate_active_query_feasibility_artifacts",
    "write_active_query_feasibility_artifacts",
]
