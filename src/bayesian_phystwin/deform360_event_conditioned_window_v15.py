"""Causal, model-independent event-window selection for Deform360 V15."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .observation_belief import array_sha256

CONTRACT = "deform360-event-conditioned-window-v15"


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
        b"deform360-event-conditioned-window-v15\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class EventConditionedWindowConfig:
    """Frozen causal scan and event-support requirements."""

    lag_frames: int = 6
    first_candidate_frame: int = 8
    forecast_horizon_frames: int = 18
    last_candidate_frame: int | None = None
    minimum_camera_support_per_cluster: int = 2
    minimum_supported_cluster_count: int = 3
    minimum_tactile_contact_probability: float = 0.5
    minimum_actuator_displacement_m: float = 0.001
    minimum_panel_nonrigid_rms_m: float = 0.001
    minimum_panel_signal_to_noise: float = 2.0
    minimum_cross_panel_cosine: float = 0.6
    maximum_panel_rms_ratio: float = 3.0
    maximum_cross_panel_reduced_nis: float = 9.0
    temporal_variance_inflation: float = 2.0
    shared_bias_variance_m2: float = 25e-6
    variance_floor_m2: float = 1e-10

    def __post_init__(self) -> None:
        _require(self.lag_frames >= 1, "event lag must be positive")
        _require(
            self.first_candidate_frame >= self.lag_frames,
            "first candidate precedes its causal history",
        )
        _require(
            self.forecast_horizon_frames >= 1,
            "forecast horizon must be positive",
        )
        _require(
            self.last_candidate_frame is None
            or self.last_candidate_frame >= self.first_candidate_frame,
            "last candidate frame is invalid",
        )
        _require(
            self.minimum_camera_support_per_cluster >= 2,
            "cluster support must contain at least two cameras per panel",
        )
        _require(
            self.minimum_supported_cluster_count >= 2,
            "too few independent spatial clusters",
        )
        probabilities = (
            self.minimum_tactile_contact_probability,
            self.minimum_cross_panel_cosine,
        )
        _require(
            all(np.isfinite(value) and 0.0 < value <= 1.0 for value in probabilities),
            "event probabilities or cosines must lie in (0, 1]",
        )
        positive = (
            self.minimum_actuator_displacement_m,
            self.minimum_panel_nonrigid_rms_m,
            self.minimum_panel_signal_to_noise,
            self.maximum_panel_rms_ratio,
            self.maximum_cross_panel_reduced_nis,
            self.temporal_variance_inflation,
            self.shared_bias_variance_m2,
            self.variance_floor_m2,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "event scales must be finite and positive",
        )
        _require(
            self.maximum_panel_rms_ratio >= 1.0,
            "panel RMS ratio cannot be below one",
        )


@dataclass(frozen=True)
class EventPanelEvidence:
    """Cluster-level, gripper-excluded shape evidence from one camera panel."""

    cluster_signature_m: np.ndarray
    variance_m2: np.ndarray
    available: np.ndarray
    camera_support: np.ndarray
    gripper_clear: np.ndarray
    cluster_ids: np.ndarray

    def __post_init__(self) -> None:
        signature = _readonly(self.cluster_signature_m, dtype=np.float64)
        variance = _readonly(self.variance_m2, dtype=np.float64)
        available = _readonly(self.available, dtype=bool)
        camera_support = _readonly(self.camera_support, dtype=np.int64)
        gripper_clear = _readonly(self.gripper_clear, dtype=bool)
        cluster_ids = _readonly(self.cluster_ids, dtype=np.int64)
        _require(
            signature.ndim == 2
            and variance.shape == signature.shape
            and available.shape == signature.shape
            and camera_support.shape == signature.shape
            and gripper_clear.shape == signature.shape,
            "event panel arrays must share shape (T, C)",
        )
        _require(
            cluster_ids.shape == (signature.shape[1],)
            and len(np.unique(cluster_ids)) == len(cluster_ids),
            "event cluster identities are invalid",
        )
        object.__setattr__(self, "cluster_signature_m", signature)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "camera_support", camera_support)
        object.__setattr__(self, "gripper_clear", gripper_clear)
        object.__setattr__(self, "cluster_ids", cluster_ids)

    @property
    def frame_count(self) -> int:
        return int(self.cluster_signature_m.shape[0])

    @property
    def cluster_count(self) -> int:
        return int(self.cluster_signature_m.shape[1])


@dataclass(frozen=True)
class EventConditionedAttempt:
    """One causal event test at a prospective branch frame."""

    birth_frame: int
    branch_frame: int
    admitted: bool
    reason: str
    supported_cluster_count: int
    tactile_contact_probability: float
    actuator_displacement_m: float
    proposal_nonrigid_rms_m: float
    validation_nonrigid_rms_m: float
    proposal_signal_to_noise: float
    validation_signal_to_noise: float
    cross_panel_cosine: float
    panel_rms_ratio: float
    cross_panel_reduced_nis: float

    def __post_init__(self) -> None:
        _require(
            0 <= self.birth_frame < self.branch_frame,
            "event attempt frame order is invalid",
        )
        _require(bool(self.reason.strip()), "event attempt reason is empty")
        _require(
            self.supported_cluster_count >= 0,
            "supported cluster count is negative",
        )
        numeric = (
            self.tactile_contact_probability,
            self.actuator_displacement_m,
            self.proposal_nonrigid_rms_m,
            self.validation_nonrigid_rms_m,
            self.proposal_signal_to_noise,
            self.validation_signal_to_noise,
            self.cross_panel_cosine,
            self.panel_rms_ratio,
            self.cross_panel_reduced_nis,
        )
        _require(
            all(np.isfinite(value) for value in numeric),
            "event attempt metrics are not finite",
        )
        _require(
            0.0 <= self.tactile_contact_probability <= 1.0
            and self.actuator_displacement_m >= 0.0
            and self.proposal_nonrigid_rms_m >= 0.0
            and self.validation_nonrigid_rms_m >= 0.0
            and self.proposal_signal_to_noise >= 0.0
            and self.validation_signal_to_noise >= 0.0
            and -1.0 <= self.cross_panel_cosine <= 1.0
            and self.panel_rms_ratio >= 0.0
            and self.cross_panel_reduced_nis >= 0.0,
            "event attempt metrics are outside their domains",
        )
        _require(
            self.admitted is (self.reason == "admitted"),
            "event admission and reason disagree",
        )


@dataclass(frozen=True)
class EventConditionedWindow:
    """Earliest causal event and its immutable information boundary."""

    case_token: str
    config: EventConditionedWindowConfig
    attempts: tuple[EventConditionedAttempt, ...]
    selected_attempt_index: int | None
    proposal_prefix_sha256: str
    validation_prefix_sha256: str
    tactile_prefix_sha256: str
    actuator_prefix_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.case_token.strip()), "case token is empty")
        frames = tuple(attempt.branch_frame for attempt in self.attempts)
        _require(
            frames == tuple(sorted(set(frames))),
            "event attempts are not a strictly increasing scan",
        )
        admitted = tuple(
            index for index, attempt in enumerate(self.attempts) if attempt.admitted
        )
        if self.selected_attempt_index is None:
            _require(not admitted, "unselected event scan contains an admission")
        else:
            _require(
                admitted == (self.selected_attempt_index,)
                and self.selected_attempt_index == len(self.attempts) - 1,
                "event scan did not stop at its first admission",
            )
        for digest in (
            self.proposal_prefix_sha256,
            self.validation_prefix_sha256,
            self.tactile_prefix_sha256,
            self.actuator_prefix_sha256,
            self.artifact_sha256,
        ):
            _require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                "event-window digest is invalid",
            )
        _require(
            _canonical_sha256(self.descriptor()) == self.artifact_sha256,
            "event-window artifact digest changed",
        )

    @property
    def admitted(self) -> bool:
        return self.selected_attempt_index is not None

    @property
    def selected_attempt(self) -> EventConditionedAttempt | None:
        if self.selected_attempt_index is None:
            return None
        return self.attempts[self.selected_attempt_index]

    @property
    def maximum_observation_frame(self) -> int:
        if not self.attempts:
            return 0
        return self.attempts[-1].branch_frame

    @property
    def forecast_frame_range_half_open(self) -> tuple[int, int] | None:
        selected = self.selected_attempt
        if selected is None:
            return None
        start = selected.branch_frame + 1
        return start, start + self.config.forecast_horizon_frames

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360EventConditionedWindowV15",
            "contract": CONTRACT,
            "case_token": self.case_token,
            "config": asdict(self.config),
            "admitted": self.admitted,
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "selected_attempt_index": self.selected_attempt_index,
            "maximum_object_observation_frame": self.maximum_observation_frame,
            "forecast_frame_range_half_open": (
                None
                if self.forecast_frame_range_half_open is None
                else list(self.forecast_frame_range_half_open)
            ),
            "proposal_prefix_sha256": self.proposal_prefix_sha256,
            "validation_prefix_sha256": self.validation_prefix_sha256,
            "tactile_prefix_sha256": self.tactile_prefix_sha256,
            "actuator_prefix_sha256": self.actuator_prefix_sha256,
            "information_boundary": {
                "selection_rule": "earliest admitted nonrigid response",
                "physical_prediction_used_for_population_selection": False,
                "candidate_update_used_for_population_selection": False,
                "future_object_observation_read": False,
                "future_identity_or_metric_read": False,
                "proposal_and_validation_camera_panels_disjoint": True,
                "shape_signature_is_rigid_motion_invariant": True,
                "gripper_pixels_excluded_upstream": True,
                "cluster_level_evidence_avoids_dense_pixel_accumulation": True,
                "tactile_and_measured_actuation_are_causal_support_only": True,
                "no_event_is_an_explicit_abstention": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def _panel_prefix_sha256(
    panel: EventPanelEvidence,
    stop_frame_exclusive: int,
) -> str:
    components = (
        array_sha256(panel.cluster_signature_m[:stop_frame_exclusive]),
        array_sha256(panel.variance_m2[:stop_frame_exclusive]),
        array_sha256(panel.available[:stop_frame_exclusive]),
        array_sha256(panel.camera_support[:stop_frame_exclusive]),
        array_sha256(panel.gripper_clear[:stop_frame_exclusive]),
        array_sha256(panel.cluster_ids),
    )
    return hashlib.sha256("".join(components).encode("ascii")).hexdigest()


def _actuator_displacement_m(
    positions_m: np.ndarray,
    birth_frame: int,
    branch_frame: int,
) -> float:
    segment = positions_m[birth_frame : branch_frame + 1]
    if not np.all(np.isfinite(segment)):
        return 0.0
    displacement = segment - segment[0]
    return float(np.max(np.linalg.norm(displacement, axis=2)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _attempt_event(
    proposal: EventPanelEvidence,
    validation: EventPanelEvidence,
    tactile: np.ndarray,
    actuator: np.ndarray,
    *,
    branch_frame: int,
    config: EventConditionedWindowConfig,
) -> EventConditionedAttempt:
    birth_frame = branch_frame - config.lag_frames
    endpoint = np.asarray([birth_frame, branch_frame], dtype=np.int64)
    proposal_signature = proposal.cluster_signature_m[endpoint]
    validation_signature = validation.cluster_signature_m[endpoint]
    proposal_variance = proposal.variance_m2[endpoint]
    validation_variance = validation.variance_m2[endpoint]
    supported = (
        np.all(proposal.available[endpoint], axis=0)
        & np.all(validation.available[endpoint], axis=0)
        & np.all(proposal.gripper_clear[endpoint], axis=0)
        & np.all(validation.gripper_clear[endpoint], axis=0)
        & np.all(
            proposal.camera_support[endpoint]
            >= config.minimum_camera_support_per_cluster,
            axis=0,
        )
        & np.all(
            validation.camera_support[endpoint]
            >= config.minimum_camera_support_per_cluster,
            axis=0,
        )
        & np.all(np.isfinite(proposal_signature), axis=0)
        & np.all(np.isfinite(validation_signature), axis=0)
        & np.all(np.isfinite(proposal_variance), axis=0)
        & np.all(np.isfinite(validation_variance), axis=0)
        & np.all(proposal_variance >= 0.0, axis=0)
        & np.all(validation_variance >= 0.0, axis=0)
    )
    local = np.flatnonzero(supported)
    supported_count = int(len(local))
    tactile_segment = tactile[birth_frame : branch_frame + 1]
    valid_tactile = bool(
        np.all(np.isfinite(tactile_segment))
        and np.all((tactile_segment >= 0.0) & (tactile_segment <= 1.0))
    )
    tactile_probability = (
        float(np.max(tactile_segment)) if valid_tactile else 0.0
    )
    actuator_displacement = _actuator_displacement_m(
        actuator,
        birth_frame,
        branch_frame,
    )
    if supported_count:
        proposal_delta = (
            proposal_signature[1, local] - proposal_signature[0, local]
        )
        validation_delta = (
            validation_signature[1, local] - validation_signature[0, local]
        )
        proposal_delta_variance = config.temporal_variance_inflation * np.sum(
            proposal_variance[:, local],
            axis=0,
        )
        validation_delta_variance = config.temporal_variance_inflation * np.sum(
            validation_variance[:, local],
            axis=0,
        )
        proposal_rms = float(np.sqrt(np.mean(np.square(proposal_delta))))
        validation_rms = float(np.sqrt(np.mean(np.square(validation_delta))))
        proposal_noise = float(
            np.sqrt(
                np.mean(
                    np.maximum(
                        proposal_delta_variance,
                        config.variance_floor_m2,
                    )
                )
            )
        )
        validation_noise = float(
            np.sqrt(
                np.mean(
                    np.maximum(
                        validation_delta_variance,
                        config.variance_floor_m2,
                    )
                )
            )
        )
        proposal_snr = proposal_rms / proposal_noise
        validation_snr = validation_rms / validation_noise
        panel_cosine = _cosine(proposal_delta, validation_delta)
        smaller_rms = min(proposal_rms, validation_rms)
        panel_ratio = (
            max(proposal_rms, validation_rms) / smaller_rms
            if smaller_rms > 1e-15
            else 0.0
        )
        cross_variance = (
            proposal_delta_variance
            + validation_delta_variance
            + config.shared_bias_variance_m2
        )
        cross_nis = float(
            np.mean(
                np.square(proposal_delta - validation_delta)
                / np.maximum(cross_variance, config.variance_floor_m2)
            )
        )
    else:
        proposal_rms = 0.0
        validation_rms = 0.0
        proposal_snr = 0.0
        validation_snr = 0.0
        panel_cosine = 0.0
        panel_ratio = 0.0
        cross_nis = 0.0

    checks = (
        (
            supported_count >= config.minimum_supported_cluster_count,
            "insufficient-independent-cluster-support",
        ),
        (valid_tactile, "invalid-tactile-prefix"),
        (
            tactile_probability >= config.minimum_tactile_contact_probability,
            "insufficient-tactile-contact-support",
        ),
        (
            actuator_displacement >= config.minimum_actuator_displacement_m,
            "insufficient-measured-actuator-motion",
        ),
        (
            proposal_rms >= config.minimum_panel_nonrigid_rms_m
            and validation_rms >= config.minimum_panel_nonrigid_rms_m,
            "insufficient-gripper-excluded-nonrigid-response",
        ),
        (
            proposal_snr >= config.minimum_panel_signal_to_noise
            and validation_snr >= config.minimum_panel_signal_to_noise,
            "insufficient-panel-signal-to-noise",
        ),
        (
            panel_cosine >= config.minimum_cross_panel_cosine,
            "cross-panel-direction-disagreement",
        ),
        (
            1.0 <= panel_ratio <= config.maximum_panel_rms_ratio,
            "cross-panel-magnitude-disagreement",
        ),
        (
            cross_nis <= config.maximum_cross_panel_reduced_nis,
            "cross-panel-statistical-disagreement",
        ),
    )
    reason = next((message for passed, message in checks if not passed), "admitted")
    return EventConditionedAttempt(
        birth_frame=birth_frame,
        branch_frame=branch_frame,
        admitted=reason == "admitted",
        reason=reason,
        supported_cluster_count=supported_count,
        tactile_contact_probability=tactile_probability,
        actuator_displacement_m=actuator_displacement,
        proposal_nonrigid_rms_m=proposal_rms,
        validation_nonrigid_rms_m=validation_rms,
        proposal_signal_to_noise=proposal_snr,
        validation_signal_to_noise=validation_snr,
        cross_panel_cosine=panel_cosine,
        panel_rms_ratio=panel_ratio,
        cross_panel_reduced_nis=cross_nis,
    )


def select_event_conditioned_window(
    case_token: str,
    proposal: EventPanelEvidence,
    validation: EventPanelEvidence,
    tactile_contact_probability: np.ndarray,
    measured_actuator_positions_m: np.ndarray,
    *,
    config: EventConditionedWindowConfig | None = None,
) -> EventConditionedWindow:
    """Stop at the earliest observed nonrigid event with a fixed future horizon."""

    cfg = config or EventConditionedWindowConfig()
    _require(
        proposal.frame_count == validation.frame_count
        and proposal.cluster_count == validation.cluster_count
        and np.array_equal(proposal.cluster_ids, validation.cluster_ids),
        "event camera panels do not share a cluster contract",
    )
    frame_count = proposal.frame_count
    tactile = np.asarray(tactile_contact_probability, dtype=np.float64)
    actuator = np.asarray(measured_actuator_positions_m, dtype=np.float64)
    if actuator.ndim == 2:
        actuator = actuator[:, None]
    _require(
        tactile.shape == (frame_count,),
        "tactile stream does not match the event episode",
    )
    _require(
        actuator.ndim == 3
        and actuator.shape[0] == frame_count
        and actuator.shape[2] == 3,
        "actuator stream must have shape (T, A, 3)",
    )
    final_from_horizon = frame_count - cfg.forecast_horizon_frames - 1
    final_candidate = (
        final_from_horizon
        if cfg.last_candidate_frame is None
        else min(final_from_horizon, cfg.last_candidate_frame)
    )
    _require(
        final_candidate >= cfg.first_candidate_frame,
        "episode is too short for the causal scan and fixed forecast horizon",
    )
    attempts: list[EventConditionedAttempt] = []
    selected_index: int | None = None
    for branch_frame in range(cfg.first_candidate_frame, final_candidate + 1):
        attempt = _attempt_event(
            proposal,
            validation,
            tactile,
            actuator,
            branch_frame=branch_frame,
            config=cfg,
        )
        attempts.append(attempt)
        if attempt.admitted:
            selected_index = len(attempts) - 1
            break
    maximum_observation_frame = attempts[-1].branch_frame
    stop = maximum_observation_frame + 1
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360EventConditionedWindowV15",
        "contract": CONTRACT,
        "case_token": str(case_token),
        "config": asdict(cfg),
        "admitted": selected_index is not None,
        "attempts": [asdict(attempt) for attempt in attempts],
        "selected_attempt_index": selected_index,
        "maximum_object_observation_frame": maximum_observation_frame,
        "forecast_frame_range_half_open": (
            None
            if selected_index is None
            else [
                attempts[selected_index].branch_frame + 1,
                attempts[selected_index].branch_frame
                + 1
                + cfg.forecast_horizon_frames,
            ]
        ),
        "proposal_prefix_sha256": _panel_prefix_sha256(proposal, stop),
        "validation_prefix_sha256": _panel_prefix_sha256(validation, stop),
        "tactile_prefix_sha256": array_sha256(tactile[:stop]),
        "actuator_prefix_sha256": array_sha256(actuator[:stop]),
        "information_boundary": {
            "selection_rule": "earliest admitted nonrigid response",
            "physical_prediction_used_for_population_selection": False,
            "candidate_update_used_for_population_selection": False,
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "proposal_and_validation_camera_panels_disjoint": True,
            "shape_signature_is_rigid_motion_invariant": True,
            "gripper_pixels_excluded_upstream": True,
            "cluster_level_evidence_avoids_dense_pixel_accumulation": True,
            "tactile_and_measured_actuation_are_causal_support_only": True,
            "no_event_is_an_explicit_abstention": True,
        },
    }
    digest = _canonical_sha256(payload)
    return EventConditionedWindow(
        case_token=str(case_token),
        config=cfg,
        attempts=tuple(attempts),
        selected_attempt_index=selected_index,
        proposal_prefix_sha256=payload["proposal_prefix_sha256"],
        validation_prefix_sha256=payload["validation_prefix_sha256"],
        tactile_prefix_sha256=payload["tactile_prefix_sha256"],
        actuator_prefix_sha256=payload["actuator_prefix_sha256"],
        artifact_sha256=digest,
    )


__all__ = [
    "CONTRACT",
    "EventConditionedAttempt",
    "EventConditionedWindow",
    "EventConditionedWindowConfig",
    "EventPanelEvidence",
    "select_event_conditioned_window",
]
