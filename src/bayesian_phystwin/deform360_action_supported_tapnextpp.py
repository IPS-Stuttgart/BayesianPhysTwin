"""Action-supported sparse TAPNext++ provider for opened Deform360 sources.

The query selector is deliberately target-free. It starts from the sealed V10
frame-zero association carrier, requires graph action support, and selects a
small graph-informative set without using predicted displacement. Tracker
outputs are sealed separately before any released identity target is scored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_active_query_feasibility import (
    REPORT_FILENAME as V10_REPORT_FILENAME,
)
from .deform360_active_query_feasibility import (
    readout_modes_to_node_basis,
)
from .deform360_dynamic_query import CameraPanel
from .observation_belief import array_sha256, file_sha256
from .phystwin_active_queries import (
    PhysicsGuidedQueryConfig,
    PhysicsGuidedQueryPlan,
    plan_physics_guided_queries,
)
from .tapnextpp_dynamic_multiview import (
    DynamicMultiviewResult,
    dynamic_multiview_result_sha256,
)
from .tapnextpp_dynamic_runtime import (
    DynamicBirthAssociations,
    DynamicTAPNextPPRuntimeResult,
)

PROTOCOL_ID = "deform360-action-supported-tapnextpp-v11-source"
QUERY_ARTIFACT_KIND = "Deform360ActionSupportedTAPNextPPQuerySchedule"
PROVIDER_ARTIFACT_KIND = "Deform360ActionSupportedTAPNextPPProviderPrediction"
QUERY_ARCHIVE_FILENAME = "action_supported_queries.npz"
QUERY_REPORT_FILENAME = "action_supported_queries.json"
PROVIDER_ARCHIVE_FILENAME = "tapnextpp_provider_prediction.npz"
PROVIDER_REPORT_FILENAME = "tapnextpp_provider_prediction.json"


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


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    result.setflags(write=False)
    return result


def _frame_zero_plan(
    plan: PhysicsGuidedQueryPlan,
) -> PhysicsGuidedQueryPlan:
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


def _plan_arrays(plan: PhysicsGuidedQueryPlan) -> dict[str, np.ndarray]:
    return {
        "entity_ids": plan.node_ids,
        "birth_frames": plan.seed_frames,
        "replaces_entity_ids": plan.replaces_node_ids,
        "query_camera_mask": plan.camera_mask,
        "query_pixels_xy": plan.seed_pixels_xy,
        "motion_score": plan.motion_score,
        "visibility_score": plan.visibility_score,
        "mode_information_gain": plan.mode_information_gain,
        "spatial_diversity_score": plan.spatial_diversity_score,
        "contact_distance_score": plan.contact_distance_score,
        "total_score": plan.total_score,
    }


@dataclass(frozen=True)
class ActionSupportedQueryConfig:
    """Frozen V11 target-free query choices."""

    prefix_frame_count: int = 58
    update_frame: int = 57
    query_count: int = 8
    graph_basis_rank: int = 8
    action_support_threshold: float = 0.1
    minimum_camera_support: int = 2
    support_probability_threshold: float = 0.5
    visibility_weight: float = 0.5
    mode_information_weight: float = 1.0
    spatial_diversity_weight: float = 1.0

    def __post_init__(self) -> None:
        _require(
            self.prefix_frame_count >= 2
            and self.update_frame == self.prefix_frame_count - 1,
            "V11 update must end the causal prefix",
        )
        _require(self.query_count >= 1, "query count must be positive")
        _require(self.graph_basis_rank >= 1, "graph rank must be positive")
        _require(
            np.isfinite(self.action_support_threshold)
            and 0.0 < self.action_support_threshold <= 1.0,
            "action support threshold must lie in (0, 1]",
        )
        _require(
            self.minimum_camera_support >= 2,
            "query association requires multiview support",
        )
        _require(
            np.isfinite(self.support_probability_threshold)
            and 0.0 < self.support_probability_threshold <= 1.0,
            "association probability threshold must lie in (0, 1]",
        )
        for name in (
            "visibility_weight",
            "mode_information_weight",
            "spatial_diversity_weight",
        ):
            _require(
                np.isfinite(getattr(self, name)) and getattr(self, name) >= 0.0,
                f"{name} must be finite and nonnegative",
            )
        _require(
            self.visibility_weight
            + self.mode_information_weight
            + self.spatial_diversity_weight
            > 0.0,
            "one target-free query score is required",
        )


@dataclass(frozen=True)
class ActionSupportedQuerySchedule:
    """Sealed sparse material-query schedule and frame-zero associations."""

    config: ActionSupportedQueryConfig
    camera_panel: CameraPanel
    plan: PhysicsGuidedQueryPlan
    query_points_world_m: np.ndarray
    association_query_points_xy: np.ndarray
    association_valid: np.ndarray
    association_probability: np.ndarray
    association_entropy: np.ndarray
    association_candidate_count: np.ndarray
    association_covariance_px2: np.ndarray
    selected_action_support: np.ndarray
    eligible_entity_count: int
    v10_result_sha256: str
    input_array_sha256: Mapping[str, str]
    artifact_sha256: str

    def __post_init__(self) -> None:
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
        count = _readonly(self.association_candidate_count, dtype=np.int64)
        covariance = _readonly(
            self.association_covariance_px2,
            dtype=np.float64,
        )
        support = _readonly(self.selected_action_support, dtype=np.float64)
        query_count = len(self.plan.node_ids)
        camera_count = len(self.camera_panel.camera_indices)
        _require(
            world.shape == (query_count, 3),
            "world query points must have shape (Q, 3)",
        )
        _require(
            pixels.shape == (camera_count, query_count, 2)
            and valid.shape == probability.shape == entropy.shape
            == count.shape
            == (camera_count, query_count),
            "association arrays changed shape",
        )
        _require(
            covariance.shape == (camera_count, query_count, 2, 2),
            "association covariance changed shape",
        )
        _require(
            support.shape == (query_count,)
            and np.all(support >= self.config.action_support_threshold)
            and np.all(support <= 1.0),
            "selected action support is invalid",
        )
        _require(
            np.array_equal(self.plan.seed_frames, np.zeros(query_count, dtype=int))
            and self.plan.reseed_count == 0,
            "V11 permits frame-zero queries only",
        )
        _require(
            np.all(
                np.sum(
                    valid
                    & (
                        probability
                        >= self.config.support_probability_threshold
                    ),
                    axis=0,
                )
                >= self.config.minimum_camera_support
            ),
            "selected query lacks frame-zero association support",
        )
        _require(
            self.eligible_entity_count >= query_count,
            "eligible entity count is smaller than the selected budget",
        )
        _require(_valid_digest(self.v10_result_sha256), "V10 digest is invalid")
        _require(
            all(_valid_digest(value) for value in self.input_array_sha256.values()),
            "query input digest is invalid",
        )
        _require(_valid_digest(self.artifact_sha256), "query digest is invalid")
        object.__setattr__(self, "query_points_world_m", world)
        object.__setattr__(self, "association_query_points_xy", pixels)
        object.__setattr__(self, "association_valid", valid)
        object.__setattr__(self, "association_probability", probability)
        object.__setattr__(self, "association_entropy", entropy)
        object.__setattr__(self, "association_candidate_count", count)
        object.__setattr__(self, "association_covariance_px2", covariance)
        object.__setattr__(self, "selected_action_support", support)
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
            **_plan_arrays(self.plan),
            "query_points_world_m": self.query_points_world_m,
            "association_query_points_xy": (
                self.association_query_points_xy
            ),
            "association_valid": self.association_valid,
            "association_probability": self.association_probability,
            "association_entropy": self.association_entropy,
            "association_candidate_count": self.association_candidate_count,
            "association_covariance_px2": self.association_covariance_px2,
            "selected_action_support": self.selected_action_support,
            "update_frames": np.full(
                len(self.plan.node_ids),
                self.config.update_frame,
                dtype=np.int64,
            ),
        }

    def descriptor(self) -> dict[str, Any]:
        arrays = self.arrays()
        return {
            "schema_version": 1,
            "artifact_kind": QUERY_ARTIFACT_KIND,
            "protocol_id": PROTOCOL_ID,
            "config": asdict(self.config),
            "admitted": self.admitted,
            "eligible_entity_count": self.eligible_entity_count,
            "selected_entity_ids": self.plan.node_ids.tolist(),
            "selected_action_support": self.selected_action_support.tolist(),
            "selected_camera_support_count": np.sum(
                self.plan.camera_mask,
                axis=1,
            ).tolist(),
            "initial_query_count": self.plan.initial_query_count,
            "camera_panel": {
                "camera_indices": self.camera_panel.camera_indices.tolist(),
                "camera_names": list(self.camera_panel.camera_names),
                "frame_zero_coverage": (
                    self.camera_panel.frame_zero_coverage.tolist()
                ),
                "selection_scores": self.camera_panel.selection_scores.tolist(),
            },
            "v10_result_sha256": self.v10_result_sha256,
            "input_array_sha256": dict(self.input_array_sha256),
            "output_array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(arrays.items())
            },
            "information_boundary": {
                "object_observation_frames_used_for_selection": [0],
                "physical_frames_used_for_selection": [0],
                "action_support_used": True,
                "action_support_weighted_mode_basis": True,
                "predicted_displacement_used": False,
                "tracker_output_read": False,
                "identity_target_read": False,
                "state_update_constructed": False,
                "held_v8_artifact_or_process_access": False,
                "v1_sealed_target_access": False,
            },
            "artifact_sha256": self.artifact_sha256,
        }

    def birth_associations(self) -> DynamicBirthAssociations:
        return DynamicBirthAssociations(
            query_points_world_m=self.query_points_world_m,
            query_points_xy=self.association_query_points_xy,
            valid=self.association_valid,
            association_probability=self.association_probability,
            association_entropy=self.association_entropy,
            candidate_pixel_covariance_px2=(
                self.association_covariance_px2
            ),
            candidate_count=self.association_candidate_count,
            camera_indices=self.camera_panel.camera_indices,
            camera_names=self.camera_panel.camera_names,
        )


def build_action_supported_query_schedule(
    v10_report: Mapping[str, Any],
    v10_arrays: Mapping[str, np.ndarray],
    physical_frame_zero_m: np.ndarray,
    graph_basis: np.ndarray,
    action_support: np.ndarray,
    *,
    config: ActionSupportedQueryConfig | None = None,
) -> ActionSupportedQuerySchedule:
    """Select graph-informative queries without predicted displacement."""

    cfg = config or ActionSupportedQueryConfig()
    report = dict(v10_report)
    arrays = {name: np.asarray(value) for name, value in v10_arrays.items()}
    _require(
        report.get("protocol_id")
        == "deform360-active-query-feasibility-v10-source"
        and report.get("result_sha256")
        == _canonical_sha256(report, digest_key="result_sha256"),
        "input is not a sealed V10 report",
    )
    audit = report["audit"]
    panel_payload = audit["camera_panel"]
    panel = CameraPanel(
        camera_indices=np.asarray(
            panel_payload["camera_indices"],
            dtype=np.int64,
        ),
        camera_names=tuple(map(str, panel_payload["camera_names"])),
        frame_zero_coverage=np.asarray(
            panel_payload["frame_zero_coverage"],
            dtype=np.float64,
        ),
        selection_scores=np.asarray(
            panel_payload["selection_scores"],
            dtype=np.float64,
        ),
    )
    frame_zero = np.asarray(physical_frame_zero_m, dtype=np.float64)
    basis = np.asarray(graph_basis, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
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
    candidates = np.asarray(
        arrays["candidate_entity_ids"],
        dtype=np.int64,
    )
    _require(
        candidates.ndim == 1
        and np.all((candidates >= 0) & (candidates < node_count)),
        "V10 candidate identities are invalid",
    )
    eligible = candidates[support[candidates] >= cfg.action_support_threshold]
    _require(len(eligible) > 0, "action support admits no V10 candidate")
    association_pixels = np.asarray(
        arrays["association_query_points_xy"],
        dtype=np.float64,
    )
    association_valid = np.asarray(arrays["association_valid"], dtype=bool)
    association_probability = np.asarray(
        arrays["association_probability"],
        dtype=np.float64,
    )
    _require(
        association_pixels.shape
        == (len(panel.camera_indices), node_count, 2)
        and association_valid.shape == association_probability.shape
        == association_pixels.shape[:2],
        "V10 association arrays changed shape",
    )
    association_support = (
        association_valid
        & (
            association_probability
            >= cfg.support_probability_threshold
        )
    )
    eligible = eligible[
        np.sum(association_support[:, eligible], axis=0)
        >= cfg.minimum_camera_support
    ]
    _require(len(eligible) > 0, "no action-supported multiview candidate")
    constant_rollout = np.repeat(
        frame_zero[None],
        cfg.prefix_frame_count,
        axis=0,
    )
    pixels_ctn = np.repeat(
        association_pixels[:, None],
        cfg.prefix_frame_count,
        axis=1,
    )
    support_ctn = np.repeat(
        np.where(
            association_support,
            association_probability,
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
    complete_plan = plan_physics_guided_queries(
        constant_rollout,
        pixels_ctn,
        support_ctn,
        mode_basis=node_basis,
        candidate_ids=eligible,
        config=PhysicsGuidedQueryConfig(
            query_count=cfg.query_count,
            maximum_reseeds=0,
            minimum_motion_m=0.0,
            minimum_camera_support=cfg.minimum_camera_support,
            support_probability_threshold=(
                cfg.support_probability_threshold
            ),
            contact_exclusion_fraction=0.0,
            motion_weight=0.0,
            visibility_weight=cfg.visibility_weight,
            mode_information_weight=cfg.mode_information_weight,
            spatial_diversity_weight=cfg.spatial_diversity_weight,
            contact_distance_weight=0.0,
        ),
    )
    plan = _frame_zero_plan(complete_plan)
    selected = plan.node_ids
    input_hashes = {
        "physical_frame_zero_m": array_sha256(frame_zero),
        "graph_basis": array_sha256(basis),
        "action_support": array_sha256(support),
        "v10_archive": report["archive"]["file_sha256"],
    }
    provisional = ActionSupportedQuerySchedule(
        config=cfg,
        camera_panel=panel,
        plan=plan,
        query_points_world_m=frame_zero[selected],
        association_query_points_xy=association_pixels[:, selected],
        association_valid=association_valid[:, selected],
        association_probability=association_probability[:, selected],
        association_entropy=np.asarray(
            arrays["association_entropy"],
            dtype=np.float64,
        )[:, selected],
        association_candidate_count=np.asarray(
            arrays["association_candidate_count"],
            dtype=np.int64,
        )[:, selected],
        association_covariance_px2=np.asarray(
            arrays["association_covariance_px2"],
            dtype=np.float64,
        )[:, selected],
        selected_action_support=support[selected],
        eligible_entity_count=len(eligible),
        v10_result_sha256=report["result_sha256"],
        input_array_sha256=input_hashes,
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(
        provisional.descriptor(),
        digest_key="artifact_sha256",
    )
    result = ActionSupportedQuerySchedule(
        **{
            **provisional.__dict__,
            "artifact_sha256": digest,
        }
    )
    _require(
        _canonical_sha256(
            result.descriptor(),
            digest_key="artifact_sha256",
        )
        == result.artifact_sha256,
        "query descriptor changed after construction",
    )
    return result


def write_action_supported_query_artifacts(
    output_dir: str | Path,
    schedule: ActionSupportedQuerySchedule,
    *,
    case_id: str,
    repository_revision: str,
    protocol_path: str | Path,
    v10_output_dir: str | Path,
    physical_manifest_path: str | Path,
    physical_archive_path: str | Path,
) -> dict[str, Any]:
    """Seal one V11 target-free query schedule."""

    _require(case_id, "case ID is empty")
    _require(
        len(repository_revision) == 40,
        "repository revision is invalid",
    )
    output = Path(output_dir).resolve()
    _require(not output.exists(), "query output directory already exists")
    output.mkdir(parents=True)
    archive_path = output / QUERY_ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        **{
            name: np.ascontiguousarray(np.asarray(values))
            for name, values in schedule.arrays().items()
        },
    )
    report = {
        "schema_version": 1,
        "artifact_kind": QUERY_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case": case_id,
        "status": "admitted" if schedule.admitted else "abstained",
        "repository_revision": repository_revision,
        "schedule": schedule.descriptor(),
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "v10_report": file_sha256(
                Path(v10_output_dir) / V10_REPORT_FILENAME
            ),
            "physical_manifest": file_sha256(physical_manifest_path),
            "physical_archive": file_sha256(physical_archive_path),
        },
        "archive": {
            "filename": QUERY_ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(schedule.arrays().items())
            },
        },
        "information_boundary": schedule.descriptor()[
            "information_boundary"
        ],
    }
    report["result_sha256"] = _canonical_sha256(
        report,
        digest_key="result_sha256",
    )
    (output / QUERY_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_action_supported_query_artifacts(output)
    return report


def validate_action_supported_query_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    output = Path(output_dir).resolve()
    report = json.loads(
        (output / QUERY_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    _require(
        report.get("artifact_kind") == QUERY_ARTIFACT_KIND
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("status") in {"admitted", "abstained"}
        and report.get("result_sha256")
        == _canonical_sha256(report, digest_key="result_sha256"),
        "query report is invalid",
    )
    schedule = report["schedule"]
    _require(
        schedule.get("artifact_sha256")
        == _canonical_sha256(schedule, digest_key="artifact_sha256"),
        "query schedule checksum changed",
    )
    boundary = report["information_boundary"]
    _require(
        boundary.get("object_observation_frames_used_for_selection") == [0]
        and boundary.get("physical_frames_used_for_selection") == [0]
        and boundary.get("predicted_displacement_used") is False
        and boundary.get("tracker_output_read") is False
        and boundary.get("identity_target_read") is False
        and boundary.get("state_update_constructed") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "query report crossed its information boundary",
    )
    archive_path = output / QUERY_ARCHIVE_FILENAME
    _require(
        report["archive"]["file_sha256"] == file_sha256(archive_path),
        "query archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    observed = {
        name: array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    _require(
        observed == report["archive"]["array_sha256"]
        == schedule["output_array_sha256"],
        "query array checksum changed",
    )
    return report, arrays


def query_schedule_from_artifacts(
    output_dir: str | Path,
) -> ActionSupportedQuerySchedule:
    report, arrays = validate_action_supported_query_artifacts(output_dir)
    descriptor = report["schedule"]
    panel = descriptor["camera_panel"]
    plan = PhysicsGuidedQueryPlan(
        node_ids=arrays["entity_ids"],
        seed_frames=arrays["birth_frames"],
        replaces_node_ids=arrays["replaces_entity_ids"],
        camera_mask=arrays["query_camera_mask"],
        seed_pixels_xy=arrays["query_pixels_xy"],
        motion_score=arrays["motion_score"],
        visibility_score=arrays["visibility_score"],
        mode_information_gain=arrays["mode_information_gain"],
        spatial_diversity_score=arrays["spatial_diversity_score"],
        contact_distance_score=arrays["contact_distance_score"],
        total_score=arrays["total_score"],
        requested_active_queries=descriptor["config"]["query_count"],
        minimum_camera_support=descriptor["config"][
            "minimum_camera_support"
        ],
        prefix_frame_count=descriptor["config"]["prefix_frame_count"],
    )
    return ActionSupportedQuerySchedule(
        config=ActionSupportedQueryConfig(**descriptor["config"]),
        camera_panel=CameraPanel(
            camera_indices=np.asarray(panel["camera_indices"]),
            camera_names=tuple(panel["camera_names"]),
            frame_zero_coverage=np.asarray(panel["frame_zero_coverage"]),
            selection_scores=np.asarray(panel["selection_scores"]),
        ),
        plan=plan,
        query_points_world_m=arrays["query_points_world_m"],
        association_query_points_xy=arrays[
            "association_query_points_xy"
        ],
        association_valid=arrays["association_valid"],
        association_probability=arrays["association_probability"],
        association_entropy=arrays["association_entropy"],
        association_candidate_count=arrays[
            "association_candidate_count"
        ],
        association_covariance_px2=arrays[
            "association_covariance_px2"
        ],
        selected_action_support=arrays["selected_action_support"],
        eligible_entity_count=descriptor["eligible_entity_count"],
        v10_result_sha256=descriptor["v10_result_sha256"],
        input_array_sha256=descriptor["input_array_sha256"],
        artifact_sha256=descriptor["artifact_sha256"],
    )


def provider_arrays(
    schedule: ActionSupportedQuerySchedule,
    runtime: DynamicTAPNextPPRuntimeResult,
    provider: DynamicMultiviewResult,
) -> dict[str, np.ndarray]:
    """Return the complete sealed numeric provider carrier."""

    query_count = len(schedule.plan.node_ids)
    _require(
        runtime.tracks_xy.shape[2] == query_count
        and provider.trajectory_world_m.shape[1] == query_count,
        "provider query count differs from the schedule",
    )
    return {
        **schedule.arrays(),
        "tracks_xy": runtime.tracks_xy,
        "tracker_visibility_probability": (
            runtime.visibility_probability
        ),
        "tracker_active": runtime.active,
        "trajectory_world_m": provider.trajectory_world_m,
        "proposal_available": provider.proposal_available,
        "accepted_support": provider.accepted_support,
        "prior_reliability": provider.prior_reliability,
        "fused_association_probability": (
            provider.association_probability
        ),
        "local_covariance_m2": provider.local_covariance_m2,
        "naive_independent_covariance_m2": (
            provider.naive_independent_covariance_m2
        ),
        "assignment_mixture_spread_m2": (
            provider.assignment_mixture_spread_m2
        ),
        "independent_support_count": (
            provider.independent_support_count
        ),
        "raw_support_count": provider.raw_support_count,
        "reprojection_rmse_px": provider.reprojection_rmse_px,
        "depth_residual_rmse_m": provider.depth_residual_rmse_m,
        "inlier_camera_mask": provider.inlier_camera_mask,
        "camera_cluster_ids": provider.camera_cluster_ids,
    }


def write_action_supported_provider_artifacts(
    output_dir: str | Path,
    schedule: ActionSupportedQuerySchedule,
    runtime: DynamicTAPNextPPRuntimeResult,
    provider: DynamicMultiviewResult,
    *,
    case_id: str,
    repository_revision: str,
    protocol_path: str | Path,
    query_output_dir: str | Path,
    runtime_provenance: Mapping[str, Any],
    causal_input_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal tracker and metric-lifting outputs before target scoring."""

    output = Path(output_dir).resolve()
    _require(not output.exists(), "provider output directory already exists")
    output.mkdir(parents=True)
    arrays = provider_arrays(schedule, runtime, provider)
    archive_path = output / PROVIDER_ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        **{
            name: np.ascontiguousarray(np.asarray(values))
            for name, values in arrays.items()
        },
    )
    report = {
        "schema_version": 1,
        "artifact_kind": PROVIDER_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case": case_id,
        "status": "prediction_sealed_before_identity_scoring",
        "repository_revision": repository_revision,
        "query_artifact_sha256": schedule.artifact_sha256,
        "provider_result_sha256": dynamic_multiview_result_sha256(provider),
        "runtime": {
            **dict(runtime_provenance),
            "rollout_count": runtime.rollout_count,
            "model_frame_count": runtime.model_frame_count,
            "elapsed_seconds": runtime.elapsed_seconds,
        },
        "causal_input_sha256": dict(causal_input_sha256),
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "query_report": file_sha256(
                Path(query_output_dir) / QUERY_REPORT_FILENAME
            ),
            "query_archive": file_sha256(
                Path(query_output_dir) / QUERY_ARCHIVE_FILENAME
            ),
        },
        "archive": {
            "filename": PROVIDER_ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(arrays.items())
            },
        },
        "information_boundary": {
            "object_rgb_depth_mask_frames_used": [
                0,
                schedule.config.update_frame,
            ],
            "frame_range_semantics": "inclusive",
            "identity_target_read": False,
            "state_update_constructed": False,
            "future_frame_after_update_read": False,
            "held_v8_artifact_or_process_access": False,
            "v1_sealed_target_access": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(
        report,
        digest_key="result_sha256",
    )
    (output / PROVIDER_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_action_supported_provider_artifacts(output)
    return report


def validate_action_supported_provider_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    output = Path(output_dir).resolve()
    report = json.loads(
        (output / PROVIDER_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    _require(
        report.get("artifact_kind") == PROVIDER_ARTIFACT_KIND
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("status")
        == "prediction_sealed_before_identity_scoring"
        and report.get("result_sha256")
        == _canonical_sha256(report, digest_key="result_sha256"),
        "provider report is invalid",
    )
    boundary = report["information_boundary"]
    _require(
        boundary.get("identity_target_read") is False
        and boundary.get("state_update_constructed") is False
        and boundary.get("future_frame_after_update_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "provider report crossed its information boundary",
    )
    archive_path = output / PROVIDER_ARCHIVE_FILENAME
    _require(
        report["archive"]["file_sha256"] == file_sha256(archive_path),
        "provider archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    observed = {
        name: array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    _require(
        observed == report["archive"]["array_sha256"],
        "provider array checksum changed",
    )
    _require(
        arrays["tracks_xy"].shape[1] == 58
        and arrays["trajectory_world_m"].shape[:2]
        == arrays["accepted_support"].shape
        and arrays["trajectory_world_m"].shape[2] == 3
        and len(arrays["entity_ids"]) == 8,
        "provider carrier shape changed",
    )
    return report, arrays


__all__ = [
    "PROTOCOL_ID",
    "PROVIDER_ARCHIVE_FILENAME",
    "PROVIDER_REPORT_FILENAME",
    "QUERY_ARCHIVE_FILENAME",
    "QUERY_REPORT_FILENAME",
    "ActionSupportedQueryConfig",
    "ActionSupportedQuerySchedule",
    "build_action_supported_query_schedule",
    "provider_arrays",
    "query_schedule_from_artifacts",
    "validate_action_supported_provider_artifacts",
    "validate_action_supported_query_artifacts",
    "write_action_supported_provider_artifacts",
    "write_action_supported_query_artifacts",
]
