"""Disjoint-panel TAPNext++ provider for the V13 causal-response carrier.

This module consumes an already sealed V13 frame-zero query schedule and a
causal RGB-D prefix. It never consumes a PhysTwin innovation or an identity
target. The proposal panel supplies the candidate metric observation; the
validation panel can only corroborate or reject it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_adaptive_query import (
    ARTIFACT_KIND as QUERY_ARTIFACT_KIND,
)
from .deform360_causal_response_adaptive_query import (
    REPORT_FILENAME as QUERY_REPORT_FILENAME,
)
from .observation_belief import array_sha256, file_sha256
from .tapnextpp_dynamic_multiview import (
    DynamicMultiviewResult,
    dynamic_multiview_result_sha256,
)
from .tapnextpp_dynamic_runtime import (
    DynamicBirthAssociations,
    DynamicTAPNextPPRuntimeResult,
)

CONTRACT = "deform360-causal-response-tracker-v13"
PROTOCOL_ID = "deform360-causal-response-tracker-v13-source"
PROVIDER_ARTIFACT_KIND = "Deform360CausalResponseTrackerPredictionV13"
PROVIDER_ARCHIVE_FILENAME = "causal_response_tracker_v13.npz"
PROVIDER_REPORT_FILENAME = "causal_response_tracker_v13.json"


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
    key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(key, None)
    return hashlib.sha256(
        b"deform360-causal-response-tracker-v13\0"
        + json.dumps(
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


@dataclass(frozen=True)
class CrossPanelProviderConfig:
    """Frozen target-free corroboration rule."""

    maximum_disagreement_m: float = 0.015
    disagreement_scale_m: float = 0.005

    def __post_init__(self) -> None:
        _require(
            np.isfinite(self.maximum_disagreement_m)
            and self.maximum_disagreement_m > 0.0,
            "cross-panel disagreement threshold must be positive",
        )
        _require(
            np.isfinite(self.disagreement_scale_m)
            and 0.0 < self.disagreement_scale_m <= self.maximum_disagreement_m,
            "cross-panel disagreement scale is invalid",
        )


@dataclass(frozen=True)
class CausalResponseTrackerPrediction:
    """A proposal observation admitted only by an independent camera panel."""

    trajectory_world_m: np.ndarray
    accepted_support: np.ndarray
    prior_reliability: np.ndarray
    local_covariance_m2: np.ndarray
    panel_disagreement_m: np.ndarray
    proposal_accepted_support: np.ndarray
    validation_accepted_support: np.ndarray
    proposal_result_sha256: str
    validation_result_sha256: str
    config: CrossPanelProviderConfig

    def __post_init__(self) -> None:
        trajectory = _readonly(self.trajectory_world_m, dtype=np.float64)
        accepted = _readonly(self.accepted_support, dtype=bool)
        reliability = _readonly(self.prior_reliability, dtype=np.float64)
        covariance = _readonly(self.local_covariance_m2, dtype=np.float64)
        disagreement = _readonly(
            self.panel_disagreement_m,
            dtype=np.float64,
        )
        proposal = _readonly(self.proposal_accepted_support, dtype=bool)
        validation = _readonly(
            self.validation_accepted_support,
            dtype=bool,
        )
        _require(
            trajectory.ndim == 3
            and trajectory.shape[2] == 3
            and accepted.shape
            == reliability.shape
            == disagreement.shape
            == proposal.shape
            == validation.shape
            == trajectory.shape[:2]
            and covariance.shape == (*trajectory.shape[:2], 3, 3),
            "cross-panel provider arrays changed shape",
        )
        _require(
            np.all(np.isfinite(trajectory))
            and np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0))
            and np.all(np.isfinite(covariance))
            and np.all(np.isfinite(disagreement))
            and np.all(disagreement >= 0.0),
            "cross-panel provider arrays are invalid",
        )
        _require(
            np.all(accepted <= (proposal & validation))
            and np.all(reliability[~accepted] == 0.0)
            and np.all(
                disagreement[accepted]
                <= self.config.maximum_disagreement_m
            ),
            "cross-panel admission rule changed",
        )
        for digest in (
            self.proposal_result_sha256,
            self.validation_result_sha256,
        ):
            _require(_valid_digest(digest), "panel result digest is invalid")
        object.__setattr__(self, "trajectory_world_m", trajectory)
        object.__setattr__(self, "accepted_support", accepted)
        object.__setattr__(self, "prior_reliability", reliability)
        object.__setattr__(self, "local_covariance_m2", covariance)
        object.__setattr__(self, "panel_disagreement_m", disagreement)
        object.__setattr__(
            self,
            "proposal_accepted_support",
            proposal,
        )
        object.__setattr__(
            self,
            "validation_accepted_support",
            validation,
        )


def birth_associations_from_adaptive_query(
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> DynamicBirthAssociations:
    """Convert one admitted V13 carrier without recomputing associations."""

    payload = dict(report)
    descriptor = payload.get("schedule", {})
    query = descriptor.get("query_schedule", {})
    _require(
        payload.get("artifact_kind") == QUERY_ARTIFACT_KIND
        and payload.get("status") == "admitted"
        and descriptor.get("admitted") is True
        and query.get("admitted") is True,
        "adaptive query is not an admitted V13 carrier",
    )
    camera_names = tuple(map(str, query["camera_ids"]))
    camera_indices = np.asarray(
        arrays["selected_complete_camera_indices"],
        dtype=np.int64,
    )
    return DynamicBirthAssociations(
        query_points_world_m=arrays["query_points_world_m"],
        query_points_xy=arrays["association_query_points_xy"],
        valid=arrays["association_valid"],
        association_probability=arrays["association_probability"],
        association_entropy=arrays["association_entropy"],
        candidate_pixel_covariance_px2=arrays[
            "association_covariance_px2"
        ],
        candidate_count=arrays["association_candidate_count"],
        camera_indices=camera_indices,
        camera_names=camera_names,
    )


def corroborate_disjoint_panels(
    proposal: DynamicMultiviewResult,
    validation: DynamicMultiviewResult,
    *,
    config: CrossPanelProviderConfig | None = None,
) -> CausalResponseTrackerPrediction:
    """Use one panel for estimation and the other only for corroboration."""

    cfg = config or CrossPanelProviderConfig()
    _require(
        proposal.trajectory_world_m.shape
        == validation.trajectory_world_m.shape
        and proposal.accepted_support.shape
        == validation.accepted_support.shape,
        "proposal and validation panel shapes differ",
    )
    delta = (
        proposal.trajectory_world_m - validation.trajectory_world_m
    )
    disagreement = np.linalg.norm(delta, axis=-1)
    common = proposal.accepted_support & validation.accepted_support
    accepted = common & (disagreement <= cfg.maximum_disagreement_m)
    agreement_weight = np.exp(
        -0.5 * np.square(disagreement / cfg.disagreement_scale_m)
    )
    reliability = (
        np.minimum(
            proposal.prior_reliability,
            validation.prior_reliability,
        )
        * agreement_weight
    )
    reliability = np.where(accepted, reliability, 0.0)

    # Cross-panel disagreement is an observation-only uncertainty term. It
    # cannot make the proposal covariance more confident.
    disagreement_covariance = np.einsum(
        "...i,...j->...ij",
        delta,
        delta,
    )
    covariance = (
        proposal.local_covariance_m2 + disagreement_covariance
    )
    return CausalResponseTrackerPrediction(
        trajectory_world_m=proposal.trajectory_world_m,
        accepted_support=accepted,
        prior_reliability=reliability,
        local_covariance_m2=covariance,
        panel_disagreement_m=disagreement,
        proposal_accepted_support=proposal.accepted_support,
        validation_accepted_support=validation.accepted_support,
        proposal_result_sha256=dynamic_multiview_result_sha256(proposal),
        validation_result_sha256=dynamic_multiview_result_sha256(
            validation
        ),
        config=cfg,
    )


def _panel_arrays(
    prefix: str,
    result: DynamicMultiviewResult,
) -> dict[str, np.ndarray]:
    names = (
        "trajectory_world_m",
        "proposal_available",
        "accepted_support",
        "prior_reliability",
        "association_probability",
        "local_covariance_m2",
        "naive_independent_covariance_m2",
        "assignment_mixture_spread_m2",
        "independent_support_count",
        "raw_support_count",
        "reprojection_rmse_px",
        "depth_residual_rmse_m",
        "inlier_camera_mask",
        "camera_cluster_ids",
    )
    return {
        f"{prefix}_{name}": np.asarray(getattr(result, name))
        for name in names
    }


def tracker_prediction_arrays(
    query_arrays: Mapping[str, np.ndarray],
    runtime: DynamicTAPNextPPRuntimeResult,
    proposal: DynamicMultiviewResult,
    validation: DynamicMultiviewResult,
    prediction: CausalResponseTrackerPrediction,
    *,
    update_frame: int,
) -> dict[str, np.ndarray]:
    """Return every carrier needed for later source scoring or auditing."""

    entity_ids = np.asarray(query_arrays["entity_ids"], dtype=np.int64)
    query_count = len(entity_ids)
    return {
        "entity_ids": entity_ids,
        "birth_frames": np.zeros(query_count, dtype=np.int64),
        "update_frames": np.full(
            query_count,
            update_frame,
            dtype=np.int64,
        ),
        "query_points_world_m": np.asarray(
            query_arrays["query_points_world_m"]
        ),
        "tracks_xy": runtime.tracks_xy,
        "visibility_probability": runtime.visibility_probability,
        "active": runtime.active,
        "trajectory_world_m": prediction.trajectory_world_m,
        "accepted_support": prediction.accepted_support,
        "prior_reliability": prediction.prior_reliability,
        "local_covariance_m2": prediction.local_covariance_m2,
        "panel_disagreement_m": prediction.panel_disagreement_m,
        "proposal_accepted_support": (
            prediction.proposal_accepted_support
        ),
        "validation_accepted_support": (
            prediction.validation_accepted_support
        ),
        **_panel_arrays("proposal", proposal),
        **_panel_arrays("validation", validation),
    }


def write_causal_response_tracker_artifacts(
    output_dir: str | Path,
    query_report: Mapping[str, Any],
    query_arrays: Mapping[str, np.ndarray],
    runtime: DynamicTAPNextPPRuntimeResult,
    proposal: DynamicMultiviewResult,
    validation: DynamicMultiviewResult,
    prediction: CausalResponseTrackerPrediction,
    *,
    case_id: str,
    repository_revision: str,
    protocol_path: str | Path,
    query_output_dir: str | Path,
    runtime_provenance: Mapping[str, Any],
    causal_input_sha256: Mapping[str, Any],
    update_frame: int,
) -> dict[str, Any]:
    """Seal all prefix predictions before any identity target is read."""

    _require(bool(case_id), "case ID is empty")
    _require(
        len(repository_revision) == 40
        and all(
            character in "0123456789abcdef"
            for character in repository_revision
        ),
        "repository revision is invalid",
    )
    _require(
        query_report.get("case") == case_id
        and query_report.get("status") == "admitted",
        "query report differs from the provider case",
    )
    output = Path(output_dir).resolve()
    _require(
        not output.exists(),
        "tracker provider output directory already exists",
    )
    output.mkdir(parents=True)
    arrays = tracker_prediction_arrays(
        query_arrays,
        runtime,
        proposal,
        validation,
        prediction,
        update_frame=update_frame,
    )
    archive_path = output / PROVIDER_ARCHIVE_FILENAME
    np.savez_compressed(
        archive_path,
        **{
            name: np.ascontiguousarray(np.asarray(values))
            for name, values in arrays.items()
        },
    )
    query_root = Path(query_output_dir)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PROVIDER_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "contract": CONTRACT,
        "case": case_id,
        "status": "prediction_sealed_before_identity_scoring",
        "repository_revision": repository_revision,
        "query_result_sha256": query_report["result_sha256"],
        "proposal_result_sha256": prediction.proposal_result_sha256,
        "validation_result_sha256": prediction.validation_result_sha256,
        "cross_panel_config": asdict(prediction.config),
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
                query_root / QUERY_REPORT_FILENAME
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
        "accepted_endpoint_count": int(
            np.sum(prediction.accepted_support[update_frame])
        ),
        "information_boundary": {
            "object_rgb_depth_mask_frames_used": [0, update_frame],
            "frame_range_semantics": "inclusive",
            "physical_innovation_used_for_prior_reliability": False,
            "identity_target_read": False,
            "tactile_event_read": False,
            "state_or_readout_update_constructed": False,
            "future_frame_after_prefix_read": False,
            "future_prediction_metric_read": False,
            "held_v8_artifact_or_process_access": False,
            "v1_sealed_target_access": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(
        report,
        key="result_sha256",
    )
    (output / PROVIDER_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_causal_response_tracker_artifacts(output)
    return report


def validate_causal_response_tracker_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate one immutable V13 tracker prediction."""

    output = Path(output_dir).resolve()
    report = json.loads(
        (output / PROVIDER_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    _require(
        report.get("artifact_kind") == PROVIDER_ARTIFACT_KIND
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("contract") == CONTRACT
        and report.get("status")
        == "prediction_sealed_before_identity_scoring"
        and report.get("result_sha256")
        == _canonical_sha256(report, key="result_sha256"),
        "tracker provider report is invalid",
    )
    boundary = report["information_boundary"]
    required_false = (
        "physical_innovation_used_for_prior_reliability",
        "identity_target_read",
        "tactile_event_read",
        "state_or_readout_update_constructed",
        "future_frame_after_prefix_read",
        "future_prediction_metric_read",
        "held_v8_artifact_or_process_access",
        "v1_sealed_target_access",
    )
    _require(
        all(boundary.get(name) is False for name in required_false),
        "tracker provider crossed its information boundary",
    )
    archive_path = output / PROVIDER_ARCHIVE_FILENAME
    _require(
        report["archive"]["filename"] == PROVIDER_ARCHIVE_FILENAME
        and report["archive"]["file_sha256"] == file_sha256(archive_path),
        "tracker provider archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    observed = {
        name: array_sha256(values)
        for name, values in sorted(arrays.items())
    }
    _require(
        observed == report["archive"]["array_sha256"],
        "tracker provider array checksum changed",
    )
    frame_count, query_count = arrays["accepted_support"].shape
    _require(
        frame_count == 58
        and query_count == 16
        and arrays["tracks_xy"].shape == (8, 58, 16, 2)
        and arrays["trajectory_world_m"].shape == (58, 16, 3)
        and arrays["local_covariance_m2"].shape
        == (58, 16, 3, 3)
        and arrays["entity_ids"].shape == (16,)
        and np.array_equal(
            arrays["birth_frames"],
            np.zeros(16, dtype=np.int64),
        )
        and np.array_equal(
            arrays["update_frames"],
            np.full(16, 57, dtype=np.int64),
        ),
        "tracker provider carrier shape changed",
    )
    return report, arrays


__all__ = [
    "CONTRACT",
    "PROTOCOL_ID",
    "PROVIDER_ARCHIVE_FILENAME",
    "PROVIDER_ARTIFACT_KIND",
    "PROVIDER_REPORT_FILENAME",
    "CausalResponseTrackerPrediction",
    "CrossPanelProviderConfig",
    "birth_associations_from_adaptive_query",
    "corroborate_disjoint_panels",
    "tracker_prediction_arrays",
    "validate_causal_response_tracker_artifacts",
    "write_causal_response_tracker_artifacts",
]
