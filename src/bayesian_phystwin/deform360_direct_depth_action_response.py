"""Prefix-only action-response admission for direct metric-depth endpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform360_dynamic_tapnextpp_assimilation import BirthAnchoredMeasurements
from .deform360_sentinel_query_schedule import Deform360SentinelQuerySchedule
from .observation_belief import array_sha256
from .phystwin_sentinel_queries import ACTIVE_QUERY_ROLE

CONTRACT = "deform360-direct-depth-action-response-admission-v9"


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
        b"deform360-direct-depth-action-response-admission-v9\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class DirectDepthActionResponseConfig:
    """Frozen support and directional-evidence requirements."""

    spatial_group_count: int = 3
    minimum_supported_active_count: int = 6
    minimum_supported_per_group: int = 2
    minimum_passing_group_count: int = 2
    minimum_actuator_displacement_m: float = 0.002
    minimum_physical_response_rms_m: float = 0.002
    minimum_observed_response_rms_m: float = 0.001
    minimum_association_probability: float = 0.5
    minimum_prior_reliability: float = 0.05
    minimum_response_gain: float = 0.10
    maximum_response_gain: float = 3.0
    minimum_response_gain_lower: float = 0.05
    minimum_direction_cosine: float = 0.50
    minimum_positive_fraction: float = 2.0 / 3.0
    confidence_z: float = 1.645
    maximum_effective_count: float = 3.0
    variance_floor_m2: float = 1e-10

    def __post_init__(self) -> None:
        _require(self.spatial_group_count >= 2, "too few spatial groups")
        _require(
            self.minimum_supported_active_count
            >= self.spatial_group_count * self.minimum_supported_per_group,
            "global support precedes grouped support",
        )
        _require(
            1 <= self.minimum_passing_group_count <= self.spatial_group_count,
            "passing-group count is invalid",
        )
        positive = (
            self.minimum_actuator_displacement_m,
            self.minimum_physical_response_rms_m,
            self.minimum_observed_response_rms_m,
            self.maximum_response_gain,
            self.confidence_z,
            self.maximum_effective_count,
            self.variance_floor_m2,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "action-response scales must be positive",
        )
        probabilities = (
            self.minimum_association_probability,
            self.minimum_prior_reliability,
            self.minimum_direction_cosine,
            self.minimum_positive_fraction,
        )
        _require(
            all(np.isfinite(value) and 0.0 < value <= 1.0 for value in probabilities),
            "action-response fractions must lie in (0, 1]",
        )
        _require(
            0.0
            <= self.minimum_response_gain_lower
            <= self.minimum_response_gain
            < self.maximum_response_gain,
            "response-gain interval is invalid",
        )


@dataclass(frozen=True)
class DirectDepthSpatialGroupEvidence:
    """One balanced spatial subset of active material identities."""

    group_index: int
    entity_ids: tuple[int, ...]
    supported_count: int
    effective_count: float
    response_gain: float
    response_gain_std: float
    response_gain_lower: float
    direction_cosine: float
    positive_fraction: float
    physical_response_rms_m: float
    observed_response_rms_m: float
    passing: bool

    def __post_init__(self) -> None:
        _require(self.group_index >= 0, "group index is negative")
        _require(
            len(self.entity_ids) > 0
            and len(set(self.entity_ids)) == len(self.entity_ids),
            "group entities are empty or repeated",
        )
        _require(
            0 <= self.supported_count <= len(self.entity_ids),
            "group support count is invalid",
        )
        _require(
            0.0 <= self.effective_count <= self.supported_count,
            "group effective count is invalid",
        )
        numeric = (
            self.response_gain,
            self.response_gain_std,
            self.response_gain_lower,
            self.direction_cosine,
            self.positive_fraction,
            self.physical_response_rms_m,
            self.observed_response_rms_m,
        )
        _require(
            all(np.isfinite(value) for value in numeric),
            "group evidence is not finite",
        )
        _require(self.response_gain_std >= 0.0, "gain deviation is negative")
        _require(
            -1.0 <= self.direction_cosine <= 1.0,
            "direction cosine is invalid",
        )
        _require(
            0.0 <= self.positive_fraction <= 1.0,
            "positive fraction is invalid",
        )


@dataclass(frozen=True)
class DirectDepthActionResponseAdmission:
    """Immutable target-free endpoint admission decision."""

    case_id: str
    birth_frame: int
    update_frame: int
    admitted: bool
    reason: str
    sentinel_applied: bool
    actuator_displacement_m: float
    supported_active_count: int
    passing_group_count: int
    config: DirectDepthActionResponseConfig
    groups: tuple[DirectDepthSpatialGroupEvidence, ...]
    physical_prefix_sha256: str
    measurement_prefix_sha256: str
    schedule_sha256: str
    group_assignments: np.ndarray
    artifact_sha256: str

    def __post_init__(self) -> None:
        assignments = _readonly(self.group_assignments, dtype=np.int64)
        _require(bool(self.case_id.strip()), "case ID is empty")
        _require(
            0 <= self.birth_frame < self.update_frame,
            "admission endpoint order is invalid",
        )
        _require(bool(self.reason.strip()), "admission reason is empty")
        _require(
            np.isfinite(self.actuator_displacement_m)
            and self.actuator_displacement_m >= 0.0,
            "actuator displacement is invalid",
        )
        _require(
            assignments.ndim == 1
            and len(assignments) >= self.supported_active_count
            and set(map(int, assignments))
            == set(range(self.config.spatial_group_count)),
            "group assignments are invalid",
        )
        _require(
            len(self.groups) == self.config.spatial_group_count
            and self.passing_group_count
            == sum(group.passing for group in self.groups),
            "group evidence count changed",
        )
        for digest in (
            self.physical_prefix_sha256,
            self.measurement_prefix_sha256,
            self.schedule_sha256,
            self.artifact_sha256,
        ):
            _require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                "admission digest is invalid",
            )
        object.__setattr__(self, "group_assignments", assignments)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360DirectDepthActionResponseAdmission",
            "contract": CONTRACT,
            "case_id": self.case_id,
            "birth_frame": self.birth_frame,
            "update_frame": self.update_frame,
            "admitted": self.admitted,
            "reason": self.reason,
            "sentinel_applied": self.sentinel_applied,
            "actuator_displacement_m": self.actuator_displacement_m,
            "supported_active_count": self.supported_active_count,
            "passing_group_count": self.passing_group_count,
            "config": asdict(self.config),
            "groups": [asdict(group) for group in self.groups],
            "physical_prefix_sha256": self.physical_prefix_sha256,
            "measurement_prefix_sha256": self.measurement_prefix_sha256,
            "schedule_sha256": self.schedule_sha256,
            "group_assignments": self.group_assignments.tolist(),
            "information_boundary": {
                "maximum_observation_frame": self.update_frame,
                "future_observation_read": False,
                "future_identity_read": False,
                "future_metric_read": False,
                "candidate_state_update_read": False,
                "state_innovation_changes_prior_reliability": False,
                "sentinel_common_bias_removed_before_admission": True,
                "final_baseline_relative_regret_guard_required": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def _balanced_spatial_groups(
    positions_m: np.ndarray,
    entity_ids: np.ndarray,
    group_count: int,
) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    entities = np.asarray(entity_ids, dtype=np.int64)
    _require(
        positions.shape == (len(entities), 3)
        and len(entities) >= group_count
        and np.all(np.isfinite(positions))
        and len(np.unique(entities)) == len(entities),
        "active spatial positions are invalid",
    )
    centroid = np.mean(positions, axis=0)
    distance_to_centroid = np.linalg.norm(positions - centroid, axis=1)
    first_candidates = np.flatnonzero(
        np.isclose(
            distance_to_centroid,
            np.max(distance_to_centroid),
            rtol=0.0,
            atol=1e-15,
        )
    )
    first = int(first_candidates[np.argmin(entities[first_candidates])])
    order = [first]
    remaining = set(range(len(entities)))
    remaining.remove(first)
    while remaining:
        candidates = np.asarray(sorted(remaining), dtype=np.int64)
        nearest = np.min(
            np.linalg.norm(
                positions[candidates, None] - positions[np.asarray(order)][None],
                axis=2,
            ),
            axis=1,
        )
        best_distance = float(np.max(nearest))
        tied = candidates[
            np.isclose(nearest, best_distance, rtol=0.0, atol=1e-15)
        ]
        selected = int(tied[np.argmin(entities[tied])])
        order.append(selected)
        remaining.remove(selected)
    assignments = np.empty(len(entities), dtype=np.int64)
    for sequence_index, point_index in enumerate(order):
        assignments[point_index] = sequence_index % group_count
    return assignments


def _effective_count(weights: np.ndarray, maximum: float) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    raw = total * total / float(np.sum(np.square(weights)))
    return float(min(raw, maximum))


def _measurement_prefix_sha256(
    measurements: BirthAnchoredMeasurements,
    stop_frame_exclusive: int,
) -> str:
    digests = (
        array_sha256(
            np.nan_to_num(
                measurements.measurement_m[:stop_frame_exclusive],
                nan=np.finfo(np.float64).max,
            )
        ),
        array_sha256(
            np.nan_to_num(
                measurements.covariance_m2[:stop_frame_exclusive],
                nan=np.finfo(np.float64).max,
            )
        ),
        array_sha256(measurements.prior_reliability[:stop_frame_exclusive]),
        array_sha256(measurements.association_probability[:stop_frame_exclusive]),
        array_sha256(measurements.available[:stop_frame_exclusive]),
        array_sha256(measurements.entity_ids),
    )
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


def _group_evidence(
    group_index: int,
    entity_ids: np.ndarray,
    physical_displacement_m: np.ndarray,
    observed_displacement_m: np.ndarray,
    covariance_m2: np.ndarray,
    reliability: np.ndarray,
    association: np.ndarray,
    supported: np.ndarray,
    config: DirectDepthActionResponseConfig,
) -> DirectDepthSpatialGroupEvidence:
    valid = (
        supported
        & (reliability >= config.minimum_prior_reliability)
        & (association >= config.minimum_association_probability)
    )
    selected_entities = tuple(map(int, entity_ids))
    supported_count = int(np.sum(valid))
    if supported_count < config.minimum_supported_per_group:
        return DirectDepthSpatialGroupEvidence(
            group_index=group_index,
            entity_ids=selected_entities,
            supported_count=supported_count,
            effective_count=float(supported_count),
            response_gain=0.0,
            response_gain_std=1.0,
            response_gain_lower=-config.confidence_z,
            direction_cosine=0.0,
            positive_fraction=0.0,
            physical_response_rms_m=0.0,
            observed_response_rms_m=0.0,
            passing=False,
        )

    physical = physical_displacement_m[valid]
    observed = observed_displacement_m[valid]
    covariance = covariance_m2[valid]
    rel = reliability[valid]
    assoc = association[valid]
    physical_squared = np.sum(np.square(physical), axis=1)
    directional_variance = np.einsum(
        "ni,nij,nj->n",
        physical,
        covariance,
        physical,
        optimize=True,
    ) / np.maximum(np.square(physical_squared), config.variance_floor_m2)
    directional_variance = np.maximum(
        directional_variance,
        config.variance_floor_m2,
    )
    precision = rel * assoc / directional_variance
    weights = precision / np.sum(precision)
    gains = np.sum(physical * observed, axis=1) / np.maximum(
        physical_squared,
        config.variance_floor_m2,
    )
    gain = float(np.sum(weights * gains))
    effective = _effective_count(
        precision,
        min(config.maximum_effective_count, float(supported_count)),
    )
    measurement_variance = float(
        np.sum(np.square(weights) * directional_variance)
        * supported_count
        / max(effective, 1.0)
    )
    scatter_variance = float(
        np.sum(weights * np.square(gains - gain)) / max(effective, 1.0)
    )
    gain_std = float(
        np.sqrt(max(measurement_variance, scatter_variance, 0.0))
    )
    gain_lower = gain - config.confidence_z * gain_std
    dot = float(np.sum(weights * np.sum(physical * observed, axis=1)))
    physical_energy = float(
        np.sum(weights * np.sum(np.square(physical), axis=1))
    )
    observed_energy = float(
        np.sum(weights * np.sum(np.square(observed), axis=1))
    )
    denominator = np.sqrt(physical_energy * observed_energy)
    cosine = 0.0 if denominator <= 0.0 else dot / denominator
    positive_fraction = float(
        np.mean(np.sum(physical * observed, axis=1) > 0.0)
    )
    physical_rms = float(np.sqrt(physical_energy))
    observed_rms = float(np.sqrt(observed_energy))
    passing = bool(
        config.minimum_response_gain
        <= gain
        <= config.maximum_response_gain
        and gain_lower >= config.minimum_response_gain_lower
        and cosine >= config.minimum_direction_cosine
        and positive_fraction >= config.minimum_positive_fraction
        and physical_rms >= config.minimum_physical_response_rms_m
        and observed_rms >= config.minimum_observed_response_rms_m
    )
    return DirectDepthSpatialGroupEvidence(
        group_index=group_index,
        entity_ids=selected_entities,
        supported_count=supported_count,
        effective_count=effective,
        response_gain=gain,
        response_gain_std=gain_std,
        response_gain_lower=gain_lower,
        direction_cosine=float(np.clip(cosine, -1.0, 1.0)),
        positive_fraction=positive_fraction,
        physical_response_rms_m=physical_rms,
        observed_response_rms_m=observed_rms,
        passing=passing,
    )


def evaluate_direct_depth_action_response(
    case_id: str,
    physical_prediction_m: np.ndarray,
    measurements: BirthAnchoredMeasurements,
    schedule: Deform360SentinelQuerySchedule,
    *,
    sentinel_applied: bool,
    actuator_displacement_m: float,
    config: DirectDepthActionResponseConfig | None = None,
) -> DirectDepthActionResponseAdmission:
    """Evaluate one endpoint pair without constructing or scoring a state update."""

    cfg = config or DirectDepthActionResponseConfig()
    physical = np.asarray(physical_prediction_m, dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical)),
        "physical prediction is invalid",
    )
    roles = np.asarray(schedule.query_roles)
    active_ids = np.asarray(
        schedule.entity_ids[roles == ACTIVE_QUERY_ROLE],
        dtype=np.int64,
    )
    _require(
        np.array_equal(measurements.entity_ids, active_ids),
        "measurement identities differ from active schedule identities",
    )
    birth = int(schedule.config.query_birth_frame)
    update = int(schedule.config.query_update_frame)
    _require(update < len(physical), "endpoint lies outside the physical prefix")
    assignments = _balanced_spatial_groups(
        physical[birth, active_ids],
        active_ids,
        cfg.spatial_group_count,
    )
    supported = np.asarray(measurements.available[update, active_ids], dtype=bool)
    observed_positions = np.asarray(
        measurements.measurement_m[update, active_ids],
        dtype=np.float64,
    )
    physical_displacement = (
        physical[update, active_ids] - physical[birth, active_ids]
    )
    observed_displacement = (
        observed_positions - physical[birth, active_ids]
    )
    covariance = np.asarray(
        measurements.covariance_m2[update, active_ids],
        dtype=np.float64,
    )
    reliability = np.asarray(
        measurements.prior_reliability[update, active_ids],
        dtype=np.float64,
    )
    association = np.asarray(
        measurements.association_probability[update, active_ids],
        dtype=np.float64,
    )
    groups = tuple(
        _group_evidence(
            group_index,
            active_ids[assignments == group_index],
            physical_displacement[assignments == group_index],
            observed_displacement[assignments == group_index],
            covariance[assignments == group_index],
            reliability[assignments == group_index],
            association[assignments == group_index],
            supported[assignments == group_index],
            cfg,
        )
        for group_index in range(cfg.spatial_group_count)
    )
    supported_count = int(
        np.sum(
            supported
            & (reliability >= cfg.minimum_prior_reliability)
            & (association >= cfg.minimum_association_probability)
        )
    )
    passing_count = sum(group.passing for group in groups)
    if not sentinel_applied:
        admitted = False
        reason = "sentinel-common-bias-unavailable"
    elif actuator_displacement_m < cfg.minimum_actuator_displacement_m:
        admitted = False
        reason = "insufficient-actuator-displacement"
    elif supported_count < cfg.minimum_supported_active_count:
        admitted = False
        reason = "insufficient-active-support"
    elif passing_count < cfg.minimum_passing_group_count:
        admitted = False
        reason = "insufficient-action-aligned-response"
    else:
        admitted = True
        reason = "action-aligned-direct-depth-response"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "case_id": str(case_id),
        "birth_frame": birth,
        "update_frame": update,
        "admitted": admitted,
        "reason": reason,
        "sentinel_applied": bool(sentinel_applied),
        "actuator_displacement_m": float(actuator_displacement_m),
        "supported_active_count": supported_count,
        "passing_group_count": passing_count,
        "config": asdict(cfg),
        "groups": [asdict(group) for group in groups],
        "physical_prefix_sha256": array_sha256(physical[: update + 1]),
        "measurement_prefix_sha256": _measurement_prefix_sha256(
            measurements,
            update + 1,
        ),
        "schedule_sha256": schedule.artifact_sha256,
        "group_assignments": assignments.tolist(),
    }
    digest = _canonical_sha256(payload)
    return DirectDepthActionResponseAdmission(
        case_id=str(case_id),
        birth_frame=birth,
        update_frame=update,
        admitted=admitted,
        reason=reason,
        sentinel_applied=bool(sentinel_applied),
        actuator_displacement_m=float(actuator_displacement_m),
        supported_active_count=supported_count,
        passing_group_count=passing_count,
        config=cfg,
        groups=groups,
        physical_prefix_sha256=payload["physical_prefix_sha256"],
        measurement_prefix_sha256=payload["measurement_prefix_sha256"],
        schedule_sha256=schedule.artifact_sha256,
        group_assignments=assignments,
        artifact_sha256=digest,
    )


__all__ = [
    "CONTRACT",
    "DirectDepthActionResponseAdmission",
    "DirectDepthActionResponseConfig",
    "DirectDepthSpatialGroupEvidence",
    "evaluate_direct_depth_action_response",
]
