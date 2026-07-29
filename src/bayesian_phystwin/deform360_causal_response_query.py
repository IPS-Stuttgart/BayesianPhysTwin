"""Frame-zero query selection for the V12 causal-response depth update."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_active_query_feasibility import readout_modes_to_node_basis
from .deform360_dynamic_query import projection_matrices
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

CONTRACT = "deform360-causal-response-query-v12"
QUERY_ARTIFACT_KIND = "Deform360CausalResponseQueryFeasibilityV12"
QUERY_ARCHIVE_FILENAME = "causal_response_query_v12.npz"
QUERY_REPORT_FILENAME = "causal_response_query_v12.json"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-query-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _report_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-query-feasibility-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CausalResponseQueryConfig:
    """Frozen target-free query choices for V12."""

    prefix_frame_count: int = 58
    query_count: int = 16
    graph_basis_rank: int = 8
    action_support_threshold: float = 0.10
    minimum_camera_support_per_panel: int = 3
    association_support_probability: float = 0.50
    association_search_radius_px: int = 12
    association_depth_scale_m: float = 0.03
    association_minimum_candidate_count: int = 1
    visibility_weight: float = 0.50
    mode_information_weight: float = 1.00
    spatial_diversity_weight: float = 1.00

    def __post_init__(self) -> None:
        _require(self.prefix_frame_count >= 2, "prefix is too short")
        _require(self.query_count >= 1, "query count must be positive")
        _require(self.graph_basis_rank >= 1, "graph rank must be positive")
        _require(
            self.minimum_camera_support_per_panel >= 3,
            "each panel must retain at least three cameras",
        )
        _require(
            np.isfinite(self.action_support_threshold)
            and 0.0 < self.action_support_threshold <= 1.0,
            "action support threshold must lie in (0, 1]",
        )
        _require(
            np.isfinite(self.association_support_probability)
            and 0.0 < self.association_support_probability <= 1.0,
            "association support threshold must lie in (0, 1]",
        )
        _require(
            self.association_search_radius_px >= 1
            and self.association_minimum_candidate_count >= 1
            and np.isfinite(self.association_depth_scale_m)
            and self.association_depth_scale_m > 0.0,
            "association settings are invalid",
        )
        weights = (
            self.visibility_weight,
            self.mode_information_weight,
            self.spatial_diversity_weight,
        )
        _require(
            all(np.isfinite(value) and value >= 0.0 for value in weights)
            and any(value > 0.0 for value in weights),
            "query score weights are invalid",
        )


@dataclass(frozen=True)
class CausalResponseQuerySchedule:
    """Immutable graph identities and disjoint camera-panel provenance."""

    config: CausalResponseQueryConfig
    camera_ids: tuple[str, ...]
    proposal_camera_indices: np.ndarray
    validation_camera_indices: np.ndarray
    entity_ids: np.ndarray
    query_points_world_m: np.ndarray
    association_query_points_xy: np.ndarray
    association_valid: np.ndarray
    association_probability: np.ndarray
    association_entropy: np.ndarray
    association_candidate_count: np.ndarray
    association_covariance_px2: np.ndarray
    selected_action_support: np.ndarray
    selected_total_score: np.ndarray
    eligible_entity_count: int
    input_array_sha256: dict[str, str]
    artifact_sha256: str

    def __post_init__(self) -> None:
        proposal = _readonly(self.proposal_camera_indices, dtype=np.int64)
        validation = _readonly(
            self.validation_camera_indices,
            dtype=np.int64,
        )
        entities = _readonly(self.entity_ids, dtype=np.int64)
        world = _readonly(self.query_points_world_m, dtype=np.float64)
        pixels = _readonly(
            self.association_query_points_xy,
            dtype=np.float64,
        )
        valid = _readonly(self.association_valid, dtype=bool)
        probability = _readonly(
            self.association_probability,
            dtype=np.float64,
        )
        entropy = _readonly(self.association_entropy, dtype=np.float64)
        count = _readonly(
            self.association_candidate_count,
            dtype=np.int64,
        )
        covariance = _readonly(
            self.association_covariance_px2,
            dtype=np.float64,
        )
        action_support = _readonly(
            self.selected_action_support,
            dtype=np.float64,
        )
        total_score = _readonly(
            self.selected_total_score,
            dtype=np.float64,
        )
        camera_count = len(self.camera_ids)
        query_count = len(entities)
        _require(
            camera_count >= 2 * self.config.minimum_camera_support_per_panel
            and len(set(self.camera_ids)) == camera_count,
            "camera identifiers are invalid",
        )
        _require(
            proposal.ndim == validation.ndim == 1
            and len(proposal) >= self.config.minimum_camera_support_per_panel
            and len(validation) >= self.config.minimum_camera_support_per_panel
            and not np.intersect1d(proposal, validation).size
            and np.array_equal(
                np.sort(np.concatenate((proposal, validation))),
                np.arange(camera_count),
            ),
            "camera panels must be disjoint and cover the full panel",
        )
        _require(
            entities.ndim == 1
            and len(np.unique(entities)) == query_count
            and np.all(entities >= 0),
            "query identities are invalid",
        )
        _require(
            world.shape == (query_count, 3)
            and pixels.shape == (camera_count, query_count, 2)
            and valid.shape
            == probability.shape
            == entropy.shape
            == count.shape
            == (camera_count, query_count)
            and covariance.shape == (camera_count, query_count, 2, 2)
            and action_support.shape == total_score.shape == (query_count,),
            "query arrays changed shape",
        )
        _require(
            np.all(np.isfinite(world))
            and np.all(np.isfinite(probability))
            and np.all((probability >= 0.0) & (probability <= 1.0))
            and np.all(action_support >= self.config.action_support_threshold)
            and np.all(action_support <= 1.0),
            "query geometry or probabilities are invalid",
        )
        supported = valid & (probability >= self.config.association_support_probability)
        for panel in (proposal, validation):
            _require(
                np.all(
                    np.sum(supported[panel], axis=0)
                    >= self.config.minimum_camera_support_per_panel
                ),
                "selected query lacks independent panel support",
            )
        _require(
            self.eligible_entity_count >= query_count,
            "eligible count is smaller than the selected budget",
        )
        for digest in (*self.input_array_sha256.values(), self.artifact_sha256):
            _require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                "query digest is invalid",
            )
        object.__setattr__(self, "proposal_camera_indices", proposal)
        object.__setattr__(self, "validation_camera_indices", validation)
        object.__setattr__(self, "entity_ids", entities)
        object.__setattr__(self, "query_points_world_m", world)
        object.__setattr__(self, "association_query_points_xy", pixels)
        object.__setattr__(self, "association_valid", valid)
        object.__setattr__(self, "association_probability", probability)
        object.__setattr__(self, "association_entropy", entropy)
        object.__setattr__(self, "association_candidate_count", count)
        object.__setattr__(self, "association_covariance_px2", covariance)
        object.__setattr__(self, "selected_action_support", action_support)
        object.__setattr__(self, "selected_total_score", total_score)
        object.__setattr__(
            self,
            "input_array_sha256",
            dict(sorted(self.input_array_sha256.items())),
        )

    @property
    def admitted(self) -> bool:
        return len(self.entity_ids) == self.config.query_count

    @property
    def proposal_camera_ids(self) -> tuple[str, ...]:
        return tuple(self.camera_ids[index] for index in self.proposal_camera_indices)

    @property
    def validation_camera_ids(self) -> tuple[str, ...]:
        return tuple(self.camera_ids[index] for index in self.validation_camera_indices)

    def descriptor(self) -> dict[str, Any]:
        arrays = self.arrays()
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalResponseQuerySchedule",
            "contract": CONTRACT,
            "config": asdict(self.config),
            "admitted": self.admitted,
            "camera_ids": list(self.camera_ids),
            "proposal_camera_indices": self.proposal_camera_indices.tolist(),
            "validation_camera_indices": self.validation_camera_indices.tolist(),
            "proposal_camera_ids": list(self.proposal_camera_ids),
            "validation_camera_ids": list(self.validation_camera_ids),
            "selected_entity_ids": self.entity_ids.tolist(),
            "selected_action_support": self.selected_action_support.tolist(),
            "selected_total_score": self.selected_total_score.tolist(),
            "eligible_entity_count": self.eligible_entity_count,
            "input_array_sha256": dict(self.input_array_sha256),
            "output_array_sha256": {
                name: array_sha256(values) for name, values in sorted(arrays.items())
            },
            "information_boundary": {
                "object_observation_frames_used_for_selection": [0],
                "physical_frames_used_for_selection": [0],
                "known_action_support_used": True,
                "action_support_weighted_graph_modes_used": True,
                "predicted_displacement_used": False,
                "tracker_output_read": False,
                "state_innovation_read": False,
                "future_object_observation_read": False,
                "future_identity_or_metric_read": False,
                "proposal_and_validation_panels_disjoint": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "entity_ids": self.entity_ids,
            "query_points_world_m": self.query_points_world_m,
            "association_query_points_xy": self.association_query_points_xy,
            "association_valid": self.association_valid,
            "association_probability": self.association_probability,
            "association_entropy": self.association_entropy,
            "association_candidate_count": self.association_candidate_count,
            "association_covariance_px2": self.association_covariance_px2,
            "selected_action_support": self.selected_action_support,
            "selected_total_score": self.selected_total_score,
        }


def _empty_plan(
    *,
    camera_count: int,
    config: CausalResponseQueryConfig,
) -> PhysicsGuidedQueryPlan:
    return PhysicsGuidedQueryPlan(
        node_ids=np.empty(0, dtype=np.int64),
        seed_frames=np.empty(0, dtype=np.int64),
        replaces_node_ids=np.empty(0, dtype=np.int64),
        camera_mask=np.empty((0, camera_count), dtype=bool),
        seed_pixels_xy=np.empty((0, camera_count, 2), dtype=np.float64),
        motion_score=np.empty(0),
        visibility_score=np.empty(0),
        mode_information_gain=np.empty(0),
        spatial_diversity_score=np.empty(0),
        contact_distance_score=np.empty(0),
        total_score=np.empty(0),
        requested_active_queries=config.query_count,
        minimum_camera_support=(2 * config.minimum_camera_support_per_panel),
        prefix_frame_count=config.prefix_frame_count,
    )


def build_causal_response_query_schedule(
    physical_frame_zero_m: np.ndarray,
    graph_basis: np.ndarray,
    action_support: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    frame_zero_depths_m: np.ndarray,
    frame_zero_object_masks: np.ndarray,
    *,
    camera_ids: tuple[str, ...],
    proposal_camera_indices: np.ndarray,
    validation_camera_indices: np.ndarray,
    config: CausalResponseQueryConfig | None = None,
) -> CausalResponseQuerySchedule:
    """Select sparse identities with independent frame-zero panel support."""

    cfg = config or CausalResponseQueryConfig()
    frame_zero = np.asarray(physical_frame_zero_m, dtype=np.float64)
    basis = np.asarray(graph_basis, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    depths = np.asarray(frame_zero_depths_m, dtype=np.float64)
    masks = np.asarray(frame_zero_object_masks, dtype=bool)
    proposal_panel = np.asarray(proposal_camera_indices, dtype=np.int64)
    validation_panel = np.asarray(validation_camera_indices, dtype=np.int64)
    camera_count = len(camera_ids)
    _require(
        frame_zero.ndim == 2
        and frame_zero.shape[1] == 3
        and np.all(np.isfinite(frame_zero)),
        "physical frame zero must have shape (N, 3)",
    )
    node_count = len(frame_zero)
    _require(
        support.shape == (node_count,)
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "action support must have shape (N,) in [0, 1]",
    )
    _require(
        matrices.shape == (camera_count, 3, 3)
        and poses.shape == (camera_count, 4, 4)
        and depths.ndim == 3
        and depths.shape[0] == camera_count
        and masks.shape == depths.shape
        and np.all(np.isfinite(depths))
        and np.all(depths >= 0.0),
        "frame-zero camera inputs are invalid",
    )
    _require(
        proposal_panel.ndim == validation_panel.ndim == 1
        and len(proposal_panel) >= cfg.minimum_camera_support_per_panel
        and len(validation_panel) >= cfg.minimum_camera_support_per_panel
        and not np.intersect1d(proposal_panel, validation_panel).size
        and np.array_equal(
            np.sort(np.concatenate((proposal_panel, validation_panel))),
            np.arange(camera_count),
        ),
        "camera panels must be disjoint and cover all cameras",
    )
    association = propose_birth_query_pixels(
        frame_zero,
        projection_matrices(matrices, poses),
        poses,
        depths,
        masks,
        config=BirthAssociationConfig(
            search_radius_px=cfg.association_search_radius_px,
            depth_scale_m=cfg.association_depth_scale_m,
            minimum_candidate_count=cfg.association_minimum_candidate_count,
            association_mode=SET_VALUED_COVARIANCE_ASSOCIATION,
        ),
    )
    association_supported = np.asarray(association["valid"], dtype=bool) & (
        np.asarray(association["association_probability"], dtype=np.float64)
        >= cfg.association_support_probability
    )
    eligible = np.flatnonzero(
        (support >= cfg.action_support_threshold)
        & (
            np.sum(association_supported[proposal_panel], axis=0)
            >= cfg.minimum_camera_support_per_panel
        )
        & (
            np.sum(association_supported[validation_panel], axis=0)
            >= cfg.minimum_camera_support_per_panel
        )
    ).astype(np.int64)
    if len(eligible):
        constant_rollout = np.repeat(
            frame_zero[None],
            cfg.prefix_frame_count,
            axis=0,
        )
        pixels = np.repeat(
            np.asarray(association["query_points_xy"])[:, None],
            cfg.prefix_frame_count,
            axis=1,
        )
        probabilities = np.repeat(
            np.where(
                association_supported,
                association["association_probability"],
                0.0,
            )[:, None],
            cfg.prefix_frame_count,
            axis=1,
        )
        node_basis = readout_modes_to_node_basis(
            basis,
            node_count=node_count,
            rank=cfg.graph_basis_rank,
        )
        node_basis = node_basis * np.sqrt(support[:, None])
        plan = plan_physics_guided_queries(
            constant_rollout,
            pixels,
            probabilities,
            mode_basis=node_basis,
            candidate_ids=eligible,
            config=PhysicsGuidedQueryConfig(
                query_count=cfg.query_count,
                maximum_reseeds=0,
                minimum_motion_m=0.0,
                minimum_camera_support=(2 * cfg.minimum_camera_support_per_panel),
                support_probability_threshold=(cfg.association_support_probability),
                contact_exclusion_fraction=0.0,
                motion_weight=0.0,
                visibility_weight=cfg.visibility_weight,
                mode_information_weight=cfg.mode_information_weight,
                spatial_diversity_weight=cfg.spatial_diversity_weight,
                contact_distance_weight=0.0,
            ),
        )
    else:
        plan = _empty_plan(camera_count=camera_count, config=cfg)
    _require(
        np.all(plan.seed_frames == 0) and plan.reseed_count == 0,
        "V12 permits frame-zero queries only",
    )
    selected = plan.node_ids
    input_hashes = {
        "physical_frame_zero_m": array_sha256(frame_zero),
        "graph_basis": array_sha256(basis),
        "action_support": array_sha256(support),
        "intrinsics": array_sha256(matrices),
        "camera_to_world": array_sha256(poses),
        "frame_zero_depths_m": array_sha256(depths),
        "frame_zero_object_masks": array_sha256(masks),
    }
    provisional = CausalResponseQuerySchedule(
        config=cfg,
        camera_ids=tuple(map(str, camera_ids)),
        proposal_camera_indices=proposal_panel,
        validation_camera_indices=validation_panel,
        entity_ids=selected,
        query_points_world_m=frame_zero[selected],
        association_query_points_xy=np.asarray(
            association["query_points_xy"],
        )[:, selected],
        association_valid=np.asarray(association["valid"])[:, selected],
        association_probability=np.asarray(
            association["association_probability"],
        )[:, selected],
        association_entropy=np.asarray(
            association["association_entropy"],
        )[:, selected],
        association_candidate_count=np.asarray(
            association["candidate_count"],
        )[:, selected],
        association_covariance_px2=np.asarray(
            association["candidate_pixel_covariance_px2"],
        )[:, selected],
        selected_action_support=support[selected],
        selected_total_score=plan.total_score,
        eligible_entity_count=len(eligible),
        input_array_sha256=input_hashes,
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor())
    result = CausalResponseQuerySchedule(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )
    _require(
        _canonical_sha256(result.descriptor()) == result.artifact_sha256,
        "query descriptor changed after construction",
    )
    return result


def write_causal_response_query_artifacts(
    output_dir: str | Path,
    schedule: CausalResponseQuerySchedule,
    *,
    case_id: str,
    repository_revision: str,
    protocol_path: str | Path,
    physical_manifest_path: str | Path,
    physical_archive_path: str | Path,
    camera_certificate_sha256: str,
) -> dict[str, Any]:
    """Seal one target-free V12 frame-zero query disposition."""

    _require(bool(case_id), "case ID is empty")
    _require(
        len(repository_revision) == 40
        and all(character in "0123456789abcdef" for character in repository_revision),
        "repository revision is invalid",
    )
    _require(
        len(camera_certificate_sha256) == 64
        and all(
            character in "0123456789abcdef" for character in camera_certificate_sha256
        ),
        "camera certificate digest is invalid",
    )
    output = Path(output_dir).resolve()
    _require(not output.exists(), "query output directory already exists")
    output.mkdir(parents=True)
    arrays = schedule.arrays()
    archive_path = output / QUERY_ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        **{
            name: np.ascontiguousarray(np.asarray(values))
            for name, values in arrays.items()
        },
    )
    boundary = {
        **schedule.descriptor()["information_boundary"],
        "identity_target_read": False,
        "state_update_constructed": False,
        "future_metric_read": False,
        "held_v8_artifact_or_process_access": False,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": QUERY_ARTIFACT_KIND,
        "contract": CONTRACT,
        "case": case_id,
        "status": "admitted" if schedule.admitted else "abstained",
        "repository_revision": repository_revision,
        "schedule": schedule.descriptor(),
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "physical_manifest": file_sha256(physical_manifest_path),
            "physical_archive": file_sha256(physical_archive_path),
            "camera_certificate": camera_certificate_sha256,
        },
        "archive": {
            "filename": QUERY_ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(values) for name, values in sorted(arrays.items())
            },
        },
        "information_boundary": boundary,
    }
    report["result_sha256"] = _report_sha256(report)
    (output / QUERY_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_causal_response_query_artifacts(output)
    return report


def validate_causal_response_query_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate one immutable V12 query-feasibility disposition."""

    output = Path(output_dir).resolve()
    report = json.loads((output / QUERY_REPORT_FILENAME).read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == QUERY_ARTIFACT_KIND
        and report.get("contract") == CONTRACT
        and report.get("status") in {"admitted", "abstained"}
        and report.get("result_sha256") == _report_sha256(report),
        "query report is invalid",
    )
    schedule = report["schedule"]
    _require(
        schedule.get("artifact_sha256") == _canonical_sha256(schedule),
        "query schedule checksum changed",
    )
    boundary = report["information_boundary"]
    _require(
        boundary.get("object_observation_frames_used_for_selection") == [0]
        and boundary.get("physical_frames_used_for_selection") == [0]
        and boundary.get("predicted_displacement_used") is False
        and boundary.get("tracker_output_read") is False
        and boundary.get("state_innovation_read") is False
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("identity_target_read") is False
        and boundary.get("state_update_constructed") is False
        and boundary.get("future_metric_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "query report crossed its information boundary",
    )
    archive_path = output / QUERY_ARCHIVE_FILENAME
    _require(
        report["archive"]["filename"] == QUERY_ARCHIVE_FILENAME
        and report["archive"]["file_sha256"] == file_sha256(archive_path),
        "query archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    observed = {name: array_sha256(values) for name, values in sorted(arrays.items())}
    _require(
        observed
        == report["archive"]["array_sha256"]
        == schedule["output_array_sha256"],
        "query array checksum changed",
    )
    _require(
        report["status"] == ("admitted" if schedule["admitted"] else "abstained"),
        "query status differs from its schedule",
    )
    return report, arrays


__all__ = [
    "CONTRACT",
    "QUERY_ARCHIVE_FILENAME",
    "QUERY_ARTIFACT_KIND",
    "QUERY_REPORT_FILENAME",
    "CausalResponseQueryConfig",
    "CausalResponseQuerySchedule",
    "build_causal_response_query_schedule",
    "validate_causal_response_query_artifacts",
    "write_causal_response_query_artifacts",
]
