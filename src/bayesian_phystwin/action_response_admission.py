"""Target-free admission for action-supported deformable-state updates.

The certificate tests whether prefix observations contain a spatial response
that is aligned with the causal physical rollout after a shared translation
nuisance is removed. It is deliberately separate from observation reliability,
state innovation scoring, candidate construction, and future regret evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil

import numpy as np

from .complete_belief_selection import CompleteBeliefGuardDecisionV1

ACTION_RESPONSE_ADMISSION_CONTRACT = (
    "bayesian-phystwin-action-response-admission-v1"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite_text(value: str, name: str) -> str:
    result = str(value).strip()
    _require(bool(result), f"{name} must be a nonempty string")
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _coordinate_weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            _weighted_median(values[:, coordinate], weights)
            for coordinate in range(3)
        ],
        dtype=np.float64,
    )


def _directional_displacement_variance(
    current_covariance_m2: np.ndarray,
    initial_covariance_m2: np.ndarray,
    physical_direction_m: np.ndarray,
    *,
    variance_floor_m2: float,
    inflation: float,
) -> float:
    norm = float(np.linalg.norm(physical_direction_m))
    unit = physical_direction_m / norm
    current = max(
        float(unit @ current_covariance_m2 @ unit),
        variance_floor_m2,
    )
    initial = max(
        float(unit @ initial_covariance_m2 @ unit),
        variance_floor_m2,
    )
    displacement = (np.sqrt(current) + np.sqrt(initial)) ** 2
    return float(inflation * displacement / (norm * norm))


@dataclass(frozen=True)
class ActionResponseAdmissionConfig:
    """Source-frozen thresholds for one prefix response certificate."""

    minimum_prefix_frames: int = 3
    minimum_action_displacement_m: float = 0.002
    minimum_identifiable_physical_rms_m: float = 0.0005
    minimum_observed_response_rms_m: float = 0.0005
    minimum_association_probability: float = 0.5
    action_support_threshold: float = 0.5
    minimum_reference_node_count: int = 3
    minimum_independent_group_count: int = 3
    minimum_supported_cluster_count: int = 4
    maximum_effective_cluster_count: float = 8.0
    minimum_response_gain: float = 0.10
    maximum_response_gain: float = 3.0
    minimum_direction_cosine: float = 0.50
    minimum_positive_cluster_fraction: float = 0.60
    minimum_passing_group_fraction: float = 0.75
    maximum_relative_group_gain_spread: float = 1.0
    confidence_z: float = 1.645
    observation_variance_floor_m2: float = 1e-8
    common_bias_covariance_inflation: float = 2.0

    def __post_init__(self) -> None:
        _require(
            self.minimum_prefix_frames >= 3,
            "minimum_prefix_frames must be at least three",
        )
        _require(
            self.minimum_reference_node_count >= 1,
            "minimum_reference_node_count must be positive",
        )
        _require(
            self.minimum_independent_group_count >= 2,
            "minimum_independent_group_count must be at least two",
        )
        _require(
            self.minimum_supported_cluster_count >= 2,
            "minimum_supported_cluster_count must be at least two",
        )
        positive = (
            self.minimum_action_displacement_m,
            self.minimum_identifiable_physical_rms_m,
            self.minimum_observed_response_rms_m,
            self.maximum_effective_cluster_count,
            self.maximum_response_gain,
            self.confidence_z,
            self.observation_variance_floor_m2,
            self.common_bias_covariance_inflation,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "action-response scales must be finite and positive",
        )
        unit_interval = (
            self.minimum_association_probability,
            self.action_support_threshold,
            self.minimum_direction_cosine,
            self.minimum_positive_cluster_fraction,
            self.minimum_passing_group_fraction,
        )
        _require(
            all(np.isfinite(value) and 0.0 < value <= 1.0 for value in unit_interval),
            "action-response probabilities and fractions must lie in (0, 1]",
        )
        _require(
            np.isfinite(self.minimum_response_gain)
            and self.minimum_response_gain >= 0.0
            and self.minimum_response_gain < self.maximum_response_gain,
            "response gain interval is invalid",
        )
        _require(
            np.isfinite(self.maximum_relative_group_gain_spread)
            and self.maximum_relative_group_gain_spread >= 0.0,
            "maximum group gain spread must be finite and nonnegative",
        )


@dataclass(frozen=True)
class ActionResponseGroupEvidence:
    """One unknown-correlation sensor-group response summary."""

    group_id: str
    sensor_count: int
    supported_cluster_count: int
    effective_cluster_count: float
    response_gain: float
    response_gain_std: float
    response_gain_lower: float
    direction_cosine: float
    positive_cluster_fraction: float
    physical_response_rms_m: float
    observed_response_rms_m: float
    mean_prior_reliability: float
    passing: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _finite_text(self.group_id, "group_id"))
        _require(self.sensor_count >= 1, "sensor_count must be positive")
        _require(
            self.supported_cluster_count >= 0,
            "supported_cluster_count must be nonnegative",
        )
        numeric = (
            self.effective_cluster_count,
            self.response_gain,
            self.response_gain_std,
            self.response_gain_lower,
            self.direction_cosine,
            self.positive_cluster_fraction,
            self.physical_response_rms_m,
            self.observed_response_rms_m,
            self.mean_prior_reliability,
        )
        _require(
            all(np.isfinite(value) for value in numeric),
            "group evidence contains non-finite values",
        )
        _require(
            0.0 <= self.effective_cluster_count <= self.supported_cluster_count,
            "effective cluster count is invalid",
        )
        _require(self.response_gain_std >= 0.0, "response gain std is negative")
        _require(
            -1.0 <= self.direction_cosine <= 1.0,
            "direction cosine must lie in [-1, 1]",
        )
        _require(
            0.0 <= self.positive_cluster_fraction <= 1.0,
            "positive cluster fraction must lie in [0, 1]",
        )
        _require(
            self.physical_response_rms_m >= 0.0
            and self.observed_response_rms_m >= 0.0,
            "response RMS must be nonnegative",
        )
        _require(
            0.0 <= self.mean_prior_reliability <= 1.0,
            "mean prior reliability must lie in [0, 1]",
        )


@dataclass(frozen=True)
class ActionResponseAdmissionV1:
    """Immutable prefix-only decision that can gate a later belief update."""

    physical_prefix_id: str
    observation_prefix_id: str
    action_prefix_id: str
    admitted: bool
    reason: str
    shared_bias_mode: str
    action_displacement_m: float
    independent_group_count: int
    passing_group_count: int
    required_passing_group_count: int
    relative_group_gain_spread: float
    config: ActionResponseAdmissionConfig
    groups: tuple[ActionResponseGroupEvidence, ...]

    def __post_init__(self) -> None:
        for name in (
            "physical_prefix_id",
            "observation_prefix_id",
            "action_prefix_id",
            "reason",
            "shared_bias_mode",
        ):
            object.__setattr__(self, name, _finite_text(getattr(self, name), name))
        _require(
            np.isfinite(self.action_displacement_m)
            and self.action_displacement_m >= 0.0,
            "action displacement must be finite and nonnegative",
        )
        _require(
            self.independent_group_count == len(self.groups),
            "independent group count changed",
        )
        _require(
            0 <= self.passing_group_count <= self.independent_group_count,
            "passing group count is invalid",
        )
        _require(
            0
            <= self.required_passing_group_count
            <= self.independent_group_count,
            "required passing group count is invalid",
        )
        _require(
            np.isfinite(self.relative_group_gain_spread)
            and self.relative_group_gain_spread >= 0.0,
            "relative group gain spread is invalid",
        )
        _require(
            tuple(sorted(group.group_id for group in self.groups))
            == tuple(group.group_id for group in self.groups),
            "groups must be sorted by group_id",
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contract": ACTION_RESPONSE_ADMISSION_CONTRACT,
            "physical_prefix_id": self.physical_prefix_id,
            "observation_prefix_id": self.observation_prefix_id,
            "action_prefix_id": self.action_prefix_id,
            "admitted": self.admitted,
            "reason": self.reason,
            "shared_bias_mode": self.shared_bias_mode,
            "action_displacement_m": self.action_displacement_m,
            "independent_group_count": self.independent_group_count,
            "passing_group_count": self.passing_group_count,
            "required_passing_group_count": self.required_passing_group_count,
            "relative_group_gain_spread": self.relative_group_gain_spread,
            "config": asdict(self.config),
            "groups": [asdict(group) for group in self.groups],
            "information_boundary": {
                "prefix_observations_only": True,
                "future_observation_read": False,
                "future_target_read": False,
                "candidate_update_read": False,
                "state_innovation_changes_prior_reliability": False,
                "shared_translation_is_treated_as_nuisance": True,
                "final_regret_guard_required": True,
            },
        }

    @property
    def artifact_id(self) -> str:
        payload = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(
            b"bayesian-phystwin-action-response-admission-v1\0" + payload
        ).hexdigest()
        return f"sha256:{digest}"

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["artifact_id"] = self.artifact_id
        return payload


@dataclass(frozen=True)
class _SensorEvidence:
    group_id: str
    supported_cluster_count: int
    effective_cluster_count: float
    response_gain: float
    response_gain_std: float
    response_gain_lower: float
    direction_cosine: float
    positive_cluster_fraction: float
    physical_response_rms_m: float
    observed_response_rms_m: float
    mean_prior_reliability: float


def _validate_inputs(
    physical_positions_m: np.ndarray,
    observed_positions_m: np.ndarray,
    observation_validity: np.ndarray,
    observation_covariance_m2: np.ndarray,
    prior_reliability: np.ndarray,
    association_probability: np.ndarray,
    actuator_positions_m: np.ndarray,
    sensor_group_ids: Sequence[str],
    correlation_cluster_ids: Sequence[str],
    action_support: np.ndarray,
    shared_bias_reference_mask: np.ndarray | None,
    config: ActionResponseAdmissionConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray | None,
]:
    physical = np.asarray(physical_positions_m, dtype=np.float64)
    observed = np.asarray(observed_positions_m, dtype=np.float64)
    valid = np.asarray(observation_validity, dtype=bool)
    covariance = np.asarray(observation_covariance_m2, dtype=np.float64)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    association = np.asarray(association_probability, dtype=np.float64)
    action = np.asarray(actuator_positions_m, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
    _require(
        observed.ndim == 4 and observed.shape[3] == 3,
        "observed_positions_m must have shape (S, T, N, 3)",
    )
    sensor_count, frame_count, node_count, _ = observed.shape
    _require(
        frame_count >= config.minimum_prefix_frames,
        "physical prefix is too short",
    )
    if physical.ndim == 3:
        _require(
            physical.shape == (frame_count, node_count, 3),
            "shared physical_positions_m shape changed",
        )
        physical = np.broadcast_to(physical, observed.shape).copy()
    else:
        _require(
            physical.ndim == 4 and physical.shape == observed.shape,
            "physical_positions_m must have shape (T, N, 3) or (S, T, N, 3)",
        )
    expected = (sensor_count, frame_count, node_count)
    _require(valid.shape == expected, "observation_validity shape changed")
    _require(
        covariance.shape == (*expected, 3, 3),
        "observation_covariance_m2 shape changed",
    )
    _require(reliability.shape == expected, "prior_reliability shape changed")
    _require(association.shape == expected, "association_probability shape changed")
    _require(
        action.ndim == 3
        and action.shape[0] == frame_count
        and action.shape[2] == 3
        and action.shape[1] >= 1,
        "actuator_positions_m must have shape (T, A, 3)",
    )
    _require(support.shape == (node_count,), "action_support shape changed")
    groups = tuple(_finite_text(value, "sensor_group_id") for value in sensor_group_ids)
    clusters = tuple(
        _finite_text(value, "correlation_cluster_id")
        for value in correlation_cluster_ids
    )
    _require(len(groups) == sensor_count, "sensor_group_ids length changed")
    _require(len(clusters) == node_count, "correlation_cluster_ids length changed")
    _require(np.all(np.isfinite(physical)), "physical positions are not finite")
    _require(np.all(np.isfinite(action)), "actuator positions are not finite")
    _require(
        np.all(np.isfinite(reliability))
        and np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "prior reliability must lie in [0, 1]",
    )
    _require(
        np.all(np.isfinite(association))
        and np.all((association >= 0.0) & (association <= 1.0)),
        "association probability must lie in [0, 1]",
    )
    _require(
        np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "action support must lie in [0, 1]",
    )
    finite_covariance = np.all(np.isfinite(covariance), axis=(3, 4))
    _require(
        np.all(~valid | finite_covariance),
        "valid observation covariance is not finite",
    )
    for sensor, frame, node in np.argwhere(valid):
        matrix = covariance[sensor, frame, node]
        _require(
            np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12),
            "valid observation covariance is not symmetric",
        )
        _require(
            np.min(np.linalg.eigvalsh(matrix))
            >= -config.observation_variance_floor_m2,
            "valid observation covariance is not positive semidefinite",
        )
        _require(
            np.all(np.isfinite(observed[sensor, frame, node])),
            "valid observation position is not finite",
        )
    reference: np.ndarray | None
    if shared_bias_reference_mask is None:
        reference = None
    else:
        reference = np.asarray(shared_bias_reference_mask, dtype=bool)
        _require(reference.shape == (node_count,), "reference mask shape changed")
        _require(
            int(np.sum(reference)) >= config.minimum_reference_node_count,
            "shared-bias reference mask is too small",
        )
    return (
        physical,
        observed,
        valid,
        covariance,
        reliability,
        association,
        action,
        groups,
        clusters,
        support,
        reference,
    )


def _center_responses(
    physical_response: np.ndarray,
    observed_response: np.ndarray,
    valid: np.ndarray,
    reliability: np.ndarray,
    association: np.ndarray,
    support: np.ndarray,
    reference_mask: np.ndarray | None,
    config: ActionResponseAdmissionConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    frame_count, node_count, _ = physical_response.shape
    centered_physical = physical_response.copy()
    centered_observed = observed_response.copy()
    active = support >= config.action_support_threshold
    mode = (
        "reference-residual-translation"
        if reference_mask is not None
        else "translation-invariant"
    )
    for frame in range(1, frame_count):
        eligible = (
            valid[frame]
            & valid[0]
            & (reliability[frame] > 0.0)
            & (association[frame] >= config.minimum_association_probability)
        )
        if reference_mask is not None:
            bias_rows = eligible & reference_mask
            if int(np.sum(bias_rows)) < config.minimum_reference_node_count:
                centered_observed[frame] = np.nan
                continue
            weights = reliability[frame, bias_rows]
            residual = (
                observed_response[frame, bias_rows]
                - physical_response[frame, bias_rows]
            )
            bias = _coordinate_weighted_median(residual, weights)
            centered_observed[frame] -= bias
        else:
            rows = eligible & active
            if int(np.sum(rows)) < config.minimum_supported_cluster_count:
                centered_observed[frame] = np.nan
                centered_physical[frame] = np.nan
                continue
            weights = reliability[frame, rows] * support[rows]
            weights /= np.sum(weights)
            observed_translation = np.sum(
                weights[:, None] * observed_response[frame, rows],
                axis=0,
            )
            physical_translation = np.sum(
                weights[:, None] * physical_response[frame, rows],
                axis=0,
            )
            centered_observed[frame] -= observed_translation
            centered_physical[frame] -= physical_translation
    centered_physical[0] = 0.0
    centered_observed[0] = 0.0
    return centered_physical, centered_observed, mode


def _sensor_evidence(
    sensor_index: int,
    group_id: str,
    physical_response: np.ndarray,
    observed_response: np.ndarray,
    valid: np.ndarray,
    covariance: np.ndarray,
    reliability: np.ndarray,
    association: np.ndarray,
    support: np.ndarray,
    cluster_ids: tuple[str, ...],
    config: ActionResponseAdmissionConfig,
) -> _SensorEvidence:
    cluster_gain: list[float] = []
    cluster_variance: list[float] = []
    cluster_cosine: list[float] = []
    cluster_reliability: list[float] = []
    cluster_physical_energy: list[float] = []
    cluster_observed_energy: list[float] = []
    active = support >= config.action_support_threshold
    for cluster_id in sorted(set(cluster_ids)):
        nodes = np.asarray(
            [
                node
                for node, value in enumerate(cluster_ids)
                if value == cluster_id and active[node]
            ],
            dtype=np.int64,
        )
        if len(nodes) == 0:
            continue
        row_gains: list[float] = []
        gain_variances: list[float] = []
        row_weights: list[float] = []
        dot_sum = 0.0
        physical_sum = 0.0
        observed_sum = 0.0
        for frame in range(1, len(physical_response)):
            for node in nodes:
                if not (
                    valid[sensor_index, frame, node]
                    and valid[sensor_index, 0, node]
                    and reliability[sensor_index, frame, node] > 0.0
                    and association[sensor_index, frame, node]
                    >= config.minimum_association_probability
                ):
                    continue
                physical = physical_response[frame, node]
                observed = observed_response[frame, node]
                if not np.all(np.isfinite(physical)) or not np.all(
                    np.isfinite(observed)
                ):
                    continue
                physical_energy = float(physical @ physical)
                if physical_energy <= 0.0:
                    continue
                row_reliability = float(
                    reliability[sensor_index, frame, node] * support[node]
                )
                gain = float(physical @ observed / physical_energy)
                gain_variance = _directional_displacement_variance(
                    covariance[sensor_index, frame, node],
                    covariance[sensor_index, 0, node],
                    physical,
                    variance_floor_m2=config.observation_variance_floor_m2,
                    inflation=config.common_bias_covariance_inflation,
                )
                row_gains.append(gain)
                gain_variances.append(gain_variance)
                row_weights.append(row_reliability)
                dot_sum += row_reliability * float(physical @ observed)
                physical_sum += row_reliability * physical_energy
                observed_sum += row_reliability * float(observed @ observed)
        if not row_gains:
            continue
        gains_array = np.asarray(row_gains, dtype=np.float64)
        variances_array = np.maximum(
            np.asarray(gain_variances, dtype=np.float64),
            config.observation_variance_floor_m2,
        )
        weights_array = np.asarray(row_weights, dtype=np.float64)
        weights_array /= np.sum(weights_array)
        gain = float(np.sum(weights_array * gains_array))
        covariance_intersection_variance = float(
            1.0 / np.sum(weights_array / variances_array)
        )
        temporal_scatter = float(
            np.sum(weights_array * np.square(gains_array - gain))
        )
        cluster_gain.append(gain)
        cluster_variance.append(
            max(covariance_intersection_variance, temporal_scatter)
        )
        denominator = np.sqrt(max(physical_sum * observed_sum, 0.0))
        cluster_cosine.append(
            float(dot_sum / denominator) if denominator > 0.0 else 0.0
        )
        cluster_reliability.append(float(np.sum(weights_array * row_weights)))
        cluster_physical_energy.append(physical_sum / np.sum(row_weights))
        cluster_observed_energy.append(observed_sum / np.sum(row_weights))
    count = len(cluster_gain)
    if count == 0:
        return _SensorEvidence(
            group_id=group_id,
            supported_cluster_count=0,
            effective_cluster_count=0.0,
            response_gain=0.0,
            response_gain_std=0.0,
            response_gain_lower=0.0,
            direction_cosine=0.0,
            positive_cluster_fraction=0.0,
            physical_response_rms_m=0.0,
            observed_response_rms_m=0.0,
            mean_prior_reliability=0.0,
        )
    cluster_gains_array = np.asarray(cluster_gain, dtype=np.float64)
    variances = np.maximum(
        np.asarray(cluster_variance, dtype=np.float64),
        config.observation_variance_floor_m2,
    )
    precisions = 1.0 / variances
    normalized = precisions / np.sum(precisions)
    gain = _weighted_median(cluster_gains_array, normalized)
    effective = min(float(count), config.maximum_effective_cluster_count)
    covariance_intersection_variance = float(
        1.0 / np.sum(normalized / variances)
    )
    robust_scatter = float(
        np.sum(normalized * np.square(cluster_gains_array - gain))
        / max(effective, 1.0)
    )
    gain_std = float(
        np.sqrt(max(covariance_intersection_variance, robust_scatter))
    )
    return _SensorEvidence(
        group_id=group_id,
        supported_cluster_count=count,
        effective_cluster_count=effective,
        response_gain=gain,
        response_gain_std=gain_std,
        response_gain_lower=gain - config.confidence_z * gain_std,
        direction_cosine=_weighted_median(
            np.asarray(cluster_cosine, dtype=np.float64),
            normalized,
        ),
        positive_cluster_fraction=float(
            np.sum(normalized[cluster_gains_array > 0.0])
        ),
        physical_response_rms_m=float(
            np.sqrt(np.sum(normalized * np.asarray(cluster_physical_energy)))
        ),
        observed_response_rms_m=float(
            np.sqrt(np.sum(normalized * np.asarray(cluster_observed_energy)))
        ),
        mean_prior_reliability=float(
            np.sum(normalized * np.asarray(cluster_reliability))
        ),
    )


def _collapse_sensor_groups(
    sensors: Sequence[_SensorEvidence],
    config: ActionResponseAdmissionConfig,
) -> tuple[ActionResponseGroupEvidence, ...]:
    groups: list[ActionResponseGroupEvidence] = []
    for group_id in sorted({sensor.group_id for sensor in sensors}):
        members = tuple(sensor for sensor in sensors if sensor.group_id == group_id)
        gain = float(np.median([member.response_gain for member in members]))
        gain_std = float(
            max(
                max(member.response_gain_std for member in members),
                np.max(
                    np.abs(
                        np.asarray([member.response_gain for member in members])
                        - gain
                    )
                ),
            )
        )
        lower = min(
            member.response_gain_lower for member in members
        )
        supported = min(member.supported_cluster_count for member in members)
        effective = min(member.effective_cluster_count for member in members)
        cosine = min(member.direction_cosine for member in members)
        positive = min(member.positive_cluster_fraction for member in members)
        physical_rms = min(member.physical_response_rms_m for member in members)
        observed_rms = min(member.observed_response_rms_m for member in members)
        reliability = min(member.mean_prior_reliability for member in members)
        passing = bool(
            supported >= config.minimum_supported_cluster_count
            and physical_rms >= config.minimum_identifiable_physical_rms_m
            and observed_rms >= config.minimum_observed_response_rms_m
            and lower >= config.minimum_response_gain
            and gain <= config.maximum_response_gain
            and cosine >= config.minimum_direction_cosine
            and positive >= config.minimum_positive_cluster_fraction
        )
        groups.append(
            ActionResponseGroupEvidence(
                group_id=group_id,
                sensor_count=len(members),
                supported_cluster_count=supported,
                effective_cluster_count=effective,
                response_gain=gain,
                response_gain_std=gain_std,
                response_gain_lower=lower,
                direction_cosine=cosine,
                positive_cluster_fraction=positive,
                physical_response_rms_m=physical_rms,
                observed_response_rms_m=observed_rms,
                mean_prior_reliability=reliability,
                passing=passing,
            )
        )
    return tuple(groups)


def evaluate_action_response_admission(
    physical_positions_m: np.ndarray,
    observed_positions_m: np.ndarray,
    observation_validity: np.ndarray,
    observation_covariance_m2: np.ndarray,
    prior_reliability: np.ndarray,
    association_probability: np.ndarray,
    actuator_positions_m: np.ndarray,
    sensor_group_ids: Sequence[str],
    correlation_cluster_ids: Sequence[str],
    action_support: np.ndarray,
    *,
    physical_prefix_id: str,
    observation_prefix_id: str,
    action_prefix_id: str,
    shared_bias_reference_mask: np.ndarray | None = None,
    config: ActionResponseAdmissionConfig | None = None,
) -> ActionResponseAdmissionV1:
    """Certify prefix response before a candidate or future loss is available.

    Sensors with the same ``sensor_group_id`` are treated as unknown-correlated
    duplicates: their evidence is collapsed conservatively and never increases
    the independent-group count. ``correlation_cluster_ids`` similarly cap
    repeated spatial or temporal evidence at one material-identity cluster.
    """

    cfg = config or ActionResponseAdmissionConfig()
    (
        physical,
        observed,
        valid,
        covariance,
        reliability,
        association,
        action,
        sensor_groups,
        clusters,
        support,
        reference,
    ) = _validate_inputs(
        physical_positions_m,
        observed_positions_m,
        observation_validity,
        observation_covariance_m2,
        prior_reliability,
        association_probability,
        actuator_positions_m,
        sensor_group_ids,
        correlation_cluster_ids,
        action_support,
        shared_bias_reference_mask,
        cfg,
    )
    physical_response = physical - physical[:, :1]
    observed_response = observed - observed[:, :1]
    action_displacement = float(
        np.max(np.linalg.norm(action - action[0], axis=2))
    )
    sensors: list[_SensorEvidence] = []
    shared_bias_mode = (
        "reference-residual-translation"
        if reference is not None
        else "translation-invariant"
    )
    for sensor_index, group_id in enumerate(sensor_groups):
        centered_physical, centered_observed, shared_bias_mode = _center_responses(
            physical_response[sensor_index],
            observed_response[sensor_index],
            valid[sensor_index],
            reliability[sensor_index],
            association[sensor_index],
            support,
            reference,
            cfg,
        )
        sensors.append(
            _sensor_evidence(
                sensor_index,
                group_id,
                centered_physical,
                centered_observed,
                valid,
                covariance,
                reliability,
                association,
                support,
                clusters,
                cfg,
            )
        )
    groups = _collapse_sensor_groups(sensors, cfg)
    independent_group_count = len(groups)
    required_passing = (
        ceil(cfg.minimum_passing_group_fraction * independent_group_count)
        if independent_group_count
        else 0
    )
    passing_count = int(sum(group.passing for group in groups))
    gains = np.asarray(
        [group.response_gain for group in groups if group.passing],
        dtype=np.float64,
    )
    if len(gains) <= 1:
        relative_spread = (
            0.0
            if len(gains) == 1
            else cfg.maximum_relative_group_gain_spread + 1.0
        )
    else:
        scale = max(abs(float(np.median(gains))), cfg.minimum_response_gain, 1e-12)
        relative_spread = float((np.max(gains) - np.min(gains)) / scale)
    enough_action = action_displacement >= cfg.minimum_action_displacement_m
    enough_groups = independent_group_count >= cfg.minimum_independent_group_count
    enough_passing = passing_count >= required_passing
    consistent_gain = (
        np.isfinite(relative_spread)
        and relative_spread <= cfg.maximum_relative_group_gain_spread
    )
    admitted = bool(enough_action and enough_groups and enough_passing and consistent_gain)
    if not enough_action:
        reason = "insufficient-measured-action"
    elif not enough_groups:
        reason = "insufficient-independent-response-groups"
    elif not enough_passing:
        reason = "insufficient-action-aligned-response"
    elif not consistent_gain:
        reason = "inconsistent-response-gain-across-groups"
    else:
        reason = "admitted-action-aligned-prefix-response"
    return ActionResponseAdmissionV1(
        physical_prefix_id=physical_prefix_id,
        observation_prefix_id=observation_prefix_id,
        action_prefix_id=action_prefix_id,
        admitted=admitted,
        reason=reason,
        shared_bias_mode=shared_bias_mode,
        action_displacement_m=action_displacement,
        independent_group_count=independent_group_count,
        passing_group_count=passing_count,
        required_passing_group_count=required_passing,
        relative_group_gain_spread=relative_spread,
        config=cfg,
        groups=groups,
    )


def build_action_response_guard_decision(
    admission: ActionResponseAdmissionV1,
    *,
    baseline_belief_id: str,
    candidate_belief_id: str,
    common_domain_id: str,
    regret_certificate_id: str,
    numerical_inference_admissible: bool,
    regret_guard_accepted: bool,
) -> CompleteBeliefGuardDecisionV1:
    """Bind response admission and regret acceptance to complete beliefs."""

    inference_admissible = bool(
        admission.admitted and numerical_inference_admissible
    )
    combined_regret_accepted = bool(
        inference_admissible and regret_guard_accepted
    )
    if not admission.admitted:
        reason = f"action-response-rejected:{admission.reason}"
    elif not numerical_inference_admissible:
        reason = "numerical-inference-rejected"
    elif not regret_guard_accepted:
        reason = "baseline-relative-regret-rejected"
    else:
        reason = "action-response-and-regret-accepted"
    return CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline_belief_id,
        candidate_belief_id=candidate_belief_id,
        common_domain_id=common_domain_id,
        certificate_id=regret_certificate_id,
        inference_admissible=inference_admissible,
        regret_guard_accepted=combined_regret_accepted,
        reason=reason,
        metadata={
            "action_response_admission_id": admission.artifact_id,
            "action_response_admitted": admission.admitted,
            "numerical_inference_admissible": numerical_inference_admissible,
            "regret_guard_accepted_before_admission": regret_guard_accepted,
        },
    )


__all__ = [
    "ACTION_RESPONSE_ADMISSION_CONTRACT",
    "ActionResponseAdmissionConfig",
    "ActionResponseAdmissionV1",
    "ActionResponseGroupEvidence",
    "build_action_response_guard_decision",
    "evaluate_action_response_admission",
]
