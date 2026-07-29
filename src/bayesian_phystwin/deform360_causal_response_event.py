"""Earliest causal-response scan and sealed V12 candidate construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform360_causal_response_admission import (
    CausalResponseAdmission,
    CausalResponseAdmissionConfig,
    direct_depth_observation_sha256,
    evaluate_causal_response_admission,
)
from .deform360_causal_response_query import CausalResponseQuerySchedule
from .deform360_causal_response_update import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    CausalResponseMeasurementConfig,
    build_causal_response_measurements,
    predict_causal_response_candidate,
)
from .deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
    DirectDepthEndpointObservations,
    build_direct_depth_observations_for_entities,
)
from .observation_belief import array_sha256
from .phystwin_online_belief import RecursiveRbfBeliefConfig

CONTRACT = "deform360-causal-response-event-v12"
PHYSICAL_BACKBONE = "physical"
PERSISTENCE_BACKBONE = "persistence"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-event-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CausalResponseEventConfig:
    """Frozen sequential stopping rule inside the allowed prefix."""

    endpoint_lag_frames: int = 6
    first_candidate_update_frame: int = 8
    last_candidate_update_frame: int = 57
    baseline_selection_minimum_relative_margin: float = 0.05

    def __post_init__(self) -> None:
        _require(self.endpoint_lag_frames >= 1, "endpoint lag must be positive")
        _require(
            self.first_candidate_update_frame >= self.endpoint_lag_frames
            and self.last_candidate_update_frame >= self.first_candidate_update_frame,
            "candidate frame range is invalid",
        )
        _require(
            np.isfinite(self.baseline_selection_minimum_relative_margin)
            and 0.0 <= self.baseline_selection_minimum_relative_margin <= 1.0,
            "baseline selection margin must lie in [0, 1]",
        )


@dataclass(frozen=True)
class CausalResponseBaselineSelection:
    """Proposal-panel comparison between physical motion and persistence."""

    selected_backbone: str
    reason: str
    supported_count: int
    physical_pairwise_loss_m: float | None
    persistence_pairwise_loss_m: float | None
    relative_physical_improvement: float | None

    def __post_init__(self) -> None:
        _require(
            self.selected_backbone in {PHYSICAL_BACKBONE, PERSISTENCE_BACKBONE},
            "selected backbone is invalid",
        )
        _require(bool(self.reason.strip()), "selection reason is empty")
        _require(self.supported_count >= 0, "selection support is negative")
        for value in (
            self.physical_pairwise_loss_m,
            self.persistence_pairwise_loss_m,
        ):
            _require(
                value is None or (np.isfinite(value) and value >= 0.0),
                "selection loss is invalid",
            )
        _require(
            self.relative_physical_improvement is None
            or np.isfinite(self.relative_physical_improvement),
            "relative physical improvement is invalid",
        )

    def descriptor(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalResponseEventScan:
    """Immutable earliest-passing admission sequence and selected evidence."""

    case_id: str
    config: CausalResponseEventConfig
    query_artifact_sha256: str
    attempts: tuple[CausalResponseAdmission, ...]
    baseline_selections: tuple[CausalResponseBaselineSelection, ...]
    selected_attempt_index: int | None
    selected_proposal: DirectDepthEndpointObservations | None
    selected_validation: DirectDepthEndpointObservations | None
    physical_prefix_sha256: str
    physical_candidate_prefix_sha256: str
    persistence_candidate_prefix_sha256: str
    tactile_prefix_sha256: str
    actuator_prefix_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.case_id.strip()), "case ID is empty")
        for digest in (
            self.query_artifact_sha256,
            self.physical_prefix_sha256,
            self.physical_candidate_prefix_sha256,
            self.persistence_candidate_prefix_sha256,
            self.tactile_prefix_sha256,
            self.actuator_prefix_sha256,
            self.artifact_sha256,
        ):
            _require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                "event digest is invalid",
            )
        frames = [attempt.update_frame for attempt in self.attempts]
        _require(
            len(self.baseline_selections) == len(self.attempts)
            and frames == sorted(set(frames))
            and all(
                frame >= self.config.first_candidate_update_frame
                and frame <= self.config.last_candidate_update_frame
                for frame in frames
            ),
            "event attempts are not an increasing causal scan",
        )
        admitted_indices = [
            index for index, attempt in enumerate(self.attempts) if attempt.admitted
        ]
        if self.selected_attempt_index is None:
            _require(
                not admitted_indices
                and self.selected_proposal is None
                and self.selected_validation is None,
                "unselected scan retains admitted evidence",
            )
        else:
            selected = self.selected_attempt_index
            _require(
                admitted_indices == [selected]
                and selected == len(self.attempts) - 1
                and self.selected_proposal is not None
                and self.selected_validation is not None,
                "selected event is not the first admitted attempt",
            )
            attempt = self.attempts[selected]
            _require(
                np.array_equal(
                    self.selected_proposal.endpoint_frames,
                    [attempt.birth_frame, attempt.update_frame],
                )
                and np.array_equal(
                    self.selected_validation.endpoint_frames,
                    [attempt.birth_frame, attempt.update_frame],
                ),
                "selected endpoint observations changed after the scan",
            )
            _require(
                direct_depth_observation_sha256(self.selected_proposal)
                == attempt.proposal_observation_sha256
                and direct_depth_observation_sha256(self.selected_validation)
                == attempt.validation_observation_sha256,
                "selected endpoint evidence differs from the admission",
            )
            _require(
                attempt.physical_prefix_sha256 == self.physical_prefix_sha256
                and attempt.action_conditioning_prefix_sha256
                == self.physical_candidate_prefix_sha256,
                "selected baseline or action-conditioning prefix changed",
            )

    @property
    def admitted(self) -> bool:
        return self.selected_attempt_index is not None

    @property
    def selected_admission(self) -> CausalResponseAdmission | None:
        if self.selected_attempt_index is None:
            return None
        return self.attempts[self.selected_attempt_index]

    @property
    def selected_backbone(self) -> str:
        if not self.baseline_selections:
            return PERSISTENCE_BACKBONE
        return self.baseline_selections[-1].selected_backbone

    @property
    def maximum_observation_frame(self) -> int:
        if not self.attempts:
            return 0
        return self.attempts[-1].update_frame

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalResponseEventScan",
            "contract": CONTRACT,
            "case_id": self.case_id,
            "config": asdict(self.config),
            "query_artifact_sha256": self.query_artifact_sha256,
            "admitted": self.admitted,
            "attempt_count": len(self.attempts),
            "attempt_artifact_sha256": [
                attempt.artifact_sha256 for attempt in self.attempts
            ],
            "attempt_update_frames": [
                attempt.update_frame for attempt in self.attempts
            ],
            "attempt_reasons": [attempt.reason for attempt in self.attempts],
            "baseline_selections": [
                selection.descriptor() for selection in self.baseline_selections
            ],
            "selected_backbone": self.selected_backbone,
            "selected_attempt_index": self.selected_attempt_index,
            "selected_admission_sha256": (
                None
                if self.selected_admission is None
                else self.selected_admission.artifact_sha256
            ),
            "physical_prefix_sha256": self.physical_prefix_sha256,
            "physical_candidate_prefix_sha256": (self.physical_candidate_prefix_sha256),
            "persistence_candidate_prefix_sha256": (
                self.persistence_candidate_prefix_sha256
            ),
            "tactile_prefix_sha256": self.tactile_prefix_sha256,
            "actuator_prefix_sha256": self.actuator_prefix_sha256,
            "information_boundary": {
                "maximum_object_observation_frame": self.maximum_observation_frame,
                "sequential_stopping_rule": "earliest admitted response",
                "future_object_observation_read": False,
                "future_identity_or_metric_read": False,
                "known_action_support_used": True,
                "prefix_tactile_used": True,
                "measured_prefix_actuator_used": True,
                "proposal_panel_selects_backbone": True,
                "validation_panel_selects_backbone": False,
                "persistence_is_default_backbone": True,
                "physical_action_conditioning_separate_from_baseline": True,
                "validation_panel_forms_update": False,
                "query_abstention_is_exact_baseline_fallback": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def _actuator_displacement_m(
    positions_m: np.ndarray,
    birth_frame: int,
    update_frame: int,
) -> float:
    segment = positions_m[birth_frame : update_frame + 1]
    displacement = segment - segment[0]
    return float(np.max(np.linalg.norm(displacement, axis=2)))


def _proposal_pairwise_loss_m(
    observations: DirectDepthEndpointObservations,
    backbone_m: np.ndarray,
    entity_ids: np.ndarray,
    local: np.ndarray,
    point_weight: np.ndarray,
) -> float:
    entities = entity_ids[local]
    observed = observations.point_world_m[:, local]
    predicted = backbone_m[observations.endpoint_frames][:, entities]
    pair_i, pair_j = np.triu_indices(len(local), k=1)
    observed_distance = np.linalg.norm(
        observed[:, pair_i] - observed[:, pair_j],
        axis=2,
    )
    predicted_distance = np.linalg.norm(
        predicted[:, pair_i] - predicted[:, pair_j],
        axis=2,
    )
    pair_weight = np.sqrt(point_weight[pair_i] * point_weight[pair_j])
    _require(np.any(pair_weight > 0.0), "selector has no positive pair weight")
    pair_weight /= np.sum(pair_weight)
    denominator = float(np.sum(pair_weight[None] * np.square(predicted_distance)))
    scale = (
        1.0
        if denominator <= 1e-12
        else float(
            np.sum(pair_weight[None] * observed_distance * predicted_distance)
            / denominator
        )
    )
    residual_change = (
        observed_distance[1]
        - observed_distance[0]
        - scale * (predicted_distance[1] - predicted_distance[0])
    )
    return float(np.sqrt(np.sum(pair_weight * np.square(residual_change))))


def _select_backbone(
    physical_observations: DirectDepthEndpointObservations,
    persistence_observations: DirectDepthEndpointObservations,
    physical_prediction_m: np.ndarray,
    persistence_prediction_m: np.ndarray,
    entity_ids: np.ndarray,
    config: CausalResponseEventConfig,
) -> CausalResponseBaselineSelection:
    common_support = np.all(physical_observations.accepted_support, axis=0) & np.all(
        persistence_observations.accepted_support, axis=0
    )
    local = np.flatnonzero(common_support)
    if len(local) < 3:
        return CausalResponseBaselineSelection(
            selected_backbone=PERSISTENCE_BACKBONE,
            reason="insufficient-selector-support-persistence-default",
            supported_count=len(local),
            physical_pairwise_loss_m=None,
            persistence_pairwise_loss_m=None,
            relative_physical_improvement=None,
        )
    physical_association = np.sqrt(
        np.prod(physical_observations.association_probability[:, local], axis=0)
    )
    persistence_association = np.sqrt(
        np.prod(persistence_observations.association_probability[:, local], axis=0)
    )
    physical_variance = np.trace(
        np.sum(physical_observations.covariance_m2[:, local], axis=0),
        axis1=1,
        axis2=2,
    )
    persistence_variance = np.trace(
        np.sum(persistence_observations.covariance_m2[:, local], axis=0),
        axis1=1,
        axis2=2,
    )
    point_weight = np.minimum(
        physical_association, persistence_association
    ) / np.maximum(
        np.maximum(physical_variance, persistence_variance),
        1e-12,
    )
    if not np.any(point_weight > 0.0):
        return CausalResponseBaselineSelection(
            selected_backbone=PERSISTENCE_BACKBONE,
            reason="zero-selector-weight-persistence-default",
            supported_count=len(local),
            physical_pairwise_loss_m=None,
            persistence_pairwise_loss_m=None,
            relative_physical_improvement=None,
        )
    physical_loss = _proposal_pairwise_loss_m(
        physical_observations,
        physical_prediction_m,
        entity_ids,
        local,
        point_weight,
    )
    persistence_loss = _proposal_pairwise_loss_m(
        persistence_observations,
        persistence_prediction_m,
        entity_ids,
        local,
        point_weight,
    )
    relative_improvement = (persistence_loss - physical_loss) / max(
        persistence_loss, 1e-12
    )
    if relative_improvement >= config.baseline_selection_minimum_relative_margin:
        selected = PHYSICAL_BACKBONE
        reason = "physical-pairwise-margin"
    else:
        selected = PERSISTENCE_BACKBONE
        reason = "persistence-insufficient-physical-margin"
    return CausalResponseBaselineSelection(
        selected_backbone=selected,
        reason=reason,
        supported_count=len(local),
        physical_pairwise_loss_m=physical_loss,
        persistence_pairwise_loss_m=persistence_loss,
        relative_physical_improvement=relative_improvement,
    )


def scan_causal_response_event(
    case_id: str,
    physical_prediction_m: np.ndarray,
    schedule: CausalResponseQuerySchedule,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    prefix_depths_m: np.ndarray,
    prefix_object_masks: np.ndarray,
    action_support: np.ndarray,
    tactile_contact_probability: np.ndarray,
    measured_actuator_positions_m: np.ndarray,
    *,
    persistence_prediction_m: np.ndarray | None = None,
    event_config: CausalResponseEventConfig | None = None,
    depth_config: DirectDepthEndpointConfig | None = None,
    admission_config: CausalResponseAdmissionConfig | None = None,
) -> CausalResponseEventScan:
    """Scan prefix endpoints and stop at the first independently admitted event."""

    cfg = event_config or CausalResponseEventConfig()
    physical = np.asarray(physical_prediction_m, dtype=np.float64)
    persistence = (
        physical.copy()
        if persistence_prediction_m is None
        else np.asarray(persistence_prediction_m, dtype=np.float64)
    )
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    depths = np.asarray(prefix_depths_m)
    masks = np.asarray(prefix_object_masks, dtype=bool)
    support = np.asarray(action_support, dtype=np.float64)
    tactile = np.asarray(tactile_contact_probability, dtype=np.float64)
    actuator = np.asarray(measured_actuator_positions_m, dtype=np.float64)
    _require(
        physical.ndim == 3 and physical.shape[2] == 3 and np.all(np.isfinite(physical)),
        "physical prediction must have shape (T, N, 3)",
    )
    _require(
        persistence.shape == physical.shape
        and np.all(np.isfinite(persistence))
        and np.array_equal(persistence[0], physical[0]),
        "persistence prediction differs in shape or frame-zero identity",
    )
    frame_count, node_count, _ = physical.shape
    camera_count = len(schedule.camera_ids)
    allowed_prefix_count = min(
        frame_count - 1,
        cfg.last_candidate_update_frame + 1,
    )
    _require(
        matrices.shape == (camera_count, 3, 3)
        and poses.shape == (camera_count, 4, 4)
        and depths.ndim == 4
        and depths.shape[0] == camera_count
        and masks.shape == depths.shape
        and depths.shape[1] == allowed_prefix_count,
        "causal camera prefix inputs are invalid",
    )
    _require(
        support.shape == (node_count,)
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "action support is invalid",
    )
    _require(
        tactile.shape == (allowed_prefix_count,)
        and np.all(np.isfinite(tactile))
        and np.all((tactile >= 0.0) & (tactile <= 1.0)),
        "tactile contact probability is invalid",
    )
    if actuator.ndim == 2:
        actuator = actuator[:, None]
    _require(
        actuator.ndim == 3
        and actuator.shape[0] == allowed_prefix_count
        and actuator.shape[2] == 3
        and np.all(np.isfinite(actuator)),
        "measured actuator positions must have shape (T, A, 3)",
    )
    proposal_indices = schedule.proposal_camera_indices
    validation_indices = schedule.validation_camera_indices
    depth_cfg = depth_config or DirectDepthEndpointConfig()
    attempts: list[CausalResponseAdmission] = []
    selections: list[CausalResponseBaselineSelection] = []
    selected_proposal: DirectDepthEndpointObservations | None = None
    selected_validation: DirectDepthEndpointObservations | None = None
    final_update = min(
        cfg.last_candidate_update_frame,
        frame_count - 2,
        depths.shape[1] - 1,
    )
    candidate_updates = (
        range(cfg.first_candidate_update_frame, final_update + 1)
        if schedule.admitted
        else ()
    )
    for update in candidate_updates:
        birth = update - cfg.endpoint_lag_frames
        endpoint_frames = np.asarray([birth, update], dtype=np.int64)
        observations: dict[
            str,
            tuple[
                DirectDepthEndpointObservations,
                DirectDepthEndpointObservations,
            ],
        ] = {}
        for backbone_name, backbone in (
            (PHYSICAL_BACKBONE, physical),
            (PERSISTENCE_BACKBONE, persistence),
        ):
            proposal = build_direct_depth_observations_for_entities(
                backbone,
                schedule.entity_ids,
                endpoint_frames,
                matrices[proposal_indices],
                poses[proposal_indices],
                depths[proposal_indices],
                masks[proposal_indices],
                config=depth_cfg,
            )
            validation = build_direct_depth_observations_for_entities(
                backbone,
                schedule.entity_ids,
                endpoint_frames,
                matrices[validation_indices],
                poses[validation_indices],
                depths[validation_indices],
                masks[validation_indices],
                config=depth_cfg,
            )
            observations[backbone_name] = (proposal, validation)
        selection = _select_backbone(
            observations[PHYSICAL_BACKBONE][0],
            observations[PERSISTENCE_BACKBONE][0],
            physical,
            persistence,
            schedule.entity_ids,
            cfg,
        )
        selections.append(selection)
        selected_backbone = (
            physical
            if selection.selected_backbone == PHYSICAL_BACKBONE
            else persistence
        )
        proposal, validation = observations[selection.selected_backbone]
        admission = evaluate_causal_response_admission(
            case_id,
            selected_backbone,
            proposal,
            validation,
            support,
            proposal_camera_ids=schedule.proposal_camera_ids,
            validation_camera_ids=schedule.validation_camera_ids,
            tactile_contact_probability=float(np.max(tactile[birth : update + 1])),
            actuator_displacement_m=_actuator_displacement_m(
                actuator,
                birth,
                update,
            ),
            action_conditioning_positions_m=physical,
            config=admission_config,
        )
        attempts.append(admission)
        if admission.admitted:
            selected_proposal = proposal
            selected_validation = validation
            break
    selected_index = None if selected_proposal is None else len(attempts) - 1
    maximum_observation_frame = 0 if not attempts else attempts[-1].update_frame
    selected_backbone_name = (
        PERSISTENCE_BACKBONE if not selections else selections[-1].selected_backbone
    )
    selected_backbone_values = (
        physical if selected_backbone_name == PHYSICAL_BACKBONE else persistence
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "case_id": str(case_id),
        "config": asdict(cfg),
        "query_artifact_sha256": schedule.artifact_sha256,
        "attempt_artifact_sha256": [attempt.artifact_sha256 for attempt in attempts],
        "baseline_selections": [selection.descriptor() for selection in selections],
        "selected_attempt_index": selected_index,
        "physical_prefix_sha256": array_sha256(
            selected_backbone_values[: maximum_observation_frame + 1]
        ),
        "physical_candidate_prefix_sha256": array_sha256(
            physical[: maximum_observation_frame + 1]
        ),
        "persistence_candidate_prefix_sha256": array_sha256(
            persistence[: maximum_observation_frame + 1]
        ),
        "tactile_prefix_sha256": array_sha256(tactile[: maximum_observation_frame + 1]),
        "actuator_prefix_sha256": array_sha256(
            actuator[: maximum_observation_frame + 1]
        ),
    }
    digest = _canonical_sha256(payload)
    result = CausalResponseEventScan(
        case_id=str(case_id),
        config=cfg,
        query_artifact_sha256=schedule.artifact_sha256,
        attempts=tuple(attempts),
        baseline_selections=tuple(selections),
        selected_attempt_index=selected_index,
        selected_proposal=selected_proposal,
        selected_validation=selected_validation,
        physical_prefix_sha256=payload["physical_prefix_sha256"],
        physical_candidate_prefix_sha256=payload["physical_candidate_prefix_sha256"],
        persistence_candidate_prefix_sha256=payload[
            "persistence_candidate_prefix_sha256"
        ],
        tactile_prefix_sha256=payload["tactile_prefix_sha256"],
        actuator_prefix_sha256=payload["actuator_prefix_sha256"],
        artifact_sha256=digest,
    )
    _require(
        _canonical_sha256(
            {
                "schema_version": 1,
                "contract": CONTRACT,
                "case_id": result.case_id,
                "config": asdict(result.config),
                "query_artifact_sha256": result.query_artifact_sha256,
                "attempt_artifact_sha256": [
                    attempt.artifact_sha256 for attempt in result.attempts
                ],
                "baseline_selections": [
                    selection.descriptor() for selection in result.baseline_selections
                ],
                "selected_attempt_index": result.selected_attempt_index,
                "physical_prefix_sha256": result.physical_prefix_sha256,
                "physical_candidate_prefix_sha256": (
                    result.physical_candidate_prefix_sha256
                ),
                "persistence_candidate_prefix_sha256": (
                    result.persistence_candidate_prefix_sha256
                ),
                "tactile_prefix_sha256": result.tactile_prefix_sha256,
                "actuator_prefix_sha256": result.actuator_prefix_sha256,
            }
        )
        == result.artifact_sha256,
        "event descriptor changed after construction",
    )
    return result


def predict_scanned_causal_response(
    physical_prediction_m: np.ndarray,
    scan: CausalResponseEventScan,
    *,
    persistence_prediction_m: np.ndarray | None = None,
    measurement_config: CausalResponseMeasurementConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the selected update or preserve an exact no-event fallback."""

    physical_input = np.asarray(physical_prediction_m)
    persistence_input = (
        physical_input.copy()
        if persistence_prediction_m is None
        else np.asarray(persistence_prediction_m)
    )
    _require(
        physical_input.ndim == 3
        and physical_input.shape[2] == 3
        and np.all(np.isfinite(physical_input))
        and persistence_input.shape == physical_input.shape
        and np.all(np.isfinite(persistence_input))
        and np.array_equal(persistence_input[0], physical_input[0]),
        "physical or persistence prediction is invalid",
    )
    maximum_frame = scan.maximum_observation_frame
    _require(
        array_sha256(np.asarray(physical_input[: maximum_frame + 1], dtype=np.float64))
        == scan.physical_candidate_prefix_sha256
        and array_sha256(
            np.asarray(
                persistence_input[: maximum_frame + 1],
                dtype=np.float64,
            )
        )
        == scan.persistence_candidate_prefix_sha256,
        "candidate backbones differ from the event scan",
    )
    baseline_input = (
        physical_input
        if scan.selected_backbone == PHYSICAL_BACKBONE
        else persistence_input
    )
    admission = scan.selected_admission
    if admission is None:
        candidate = baseline_input.copy()
        variance = np.zeros(baseline_input.shape, dtype=np.float64)
        return (
            {
                "schema_version": 1,
                "artifact_kind": "Deform360CausalResponseCandidate",
                "event_scan_sha256": scan.artifact_sha256,
                "candidate_applied": False,
                "selected_backbone": scan.selected_backbone,
                "decision": "no-causal-response-exact-baseline-fallback",
                "bit_exact_baseline_fallback": True,
                "future_observation_read": False,
            },
            {
                BASELINE_ARM: baseline_input.copy(),
                CANDIDATE_ARM: candidate,
                "candidate_correction_variance_m2": variance,
            },
        )
    _require(
        scan.selected_proposal is not None,
        "selected scan lacks proposal observations",
    )
    response = build_causal_response_measurements(
        baseline_input,
        scan.selected_proposal,
        admission,
        config=measurement_config,
    )
    report, arrays = predict_causal_response_candidate(
        baseline_input,
        response,
        admission,
        belief_config=belief_config,
    )
    return {
        **report,
        "event_scan_sha256": scan.artifact_sha256,
        "query_artifact_sha256": scan.query_artifact_sha256,
        "selected_backbone": scan.selected_backbone,
        "validation_panel_formed_update": False,
    }, arrays


__all__ = [
    "CONTRACT",
    "PHYSICAL_BACKBONE",
    "PERSISTENCE_BACKBONE",
    "CausalResponseBaselineSelection",
    "CausalResponseEventConfig",
    "CausalResponseEventScan",
    "predict_scanned_causal_response",
    "scan_causal_response_event",
]
