"""Prior-aware PokeFlex updates from independent D405 depth.

The Kinect checkpoint and registration path propose a causal, action-supported
state subspace.  D405 depth supplies an independent metric observation of that
subspace.  This module keeps point association, perception reliability,
correlation, and innovation handling separate while explicitly marginalizing a
shared translation bias and centered per-view translation biases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .gauge_aware_belief import GaugeAwareBeliefResult
from .observation_belief import ObservationBeliefV1
from .observation_belief_gauge_adapter import (
    centered_view_translation_bias_jacobian,
    global_translation_bias_jacobian,
)
from .physical_linearization import (
    PhysicalLinearizationV1,
    build_gauge_aware_batch_from_artifacts,
)
from .pokeflex_independent_depth import PokeFlexIndependentDepthAnchor
from .prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _points(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2 and result.shape[1] == 3, f"{name} must be Nx3")
    _require(len(result) > 0, f"{name} is empty")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


def _readonly(value: object, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PokeFlexPriorAwareConfigV1:
    """Frozen association, covariance, nuisance, and trust-region settings."""

    assignment_candidates: int = 4
    maximum_association_m: float = 0.025
    assignment_temperature_m: float = 0.008
    minimum_points_per_sensor: int = 16
    effective_samples_per_sensor: float = 16.0
    minimum_local_std_m: float = 0.001
    maximum_calibration_median_residual_m: float = 0.010
    calibration_reliability_scale_m: float = 0.010
    minimum_sensor_reliability: float = 0.05
    prior_nominal_probability: float = 0.90
    state_prior_std_m: float = 0.010
    shared_bias_prior_std_m: float = 0.005
    view_bias_prior_std_m: float = 0.005
    maximum_state_rank: int = 4
    minimum_singular_fraction: float = 1e-3
    maximum_state_update_m: float = 0.030
    maximum_update_to_physical_response_ratio: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            self.maximum_association_m,
            self.assignment_temperature_m,
            self.effective_samples_per_sensor,
            self.minimum_local_std_m,
            self.maximum_calibration_median_residual_m,
            self.calibration_reliability_scale_m,
            self.state_prior_std_m,
            self.shared_bias_prior_std_m,
            self.view_bias_prior_std_m,
            self.maximum_state_update_m,
            self.maximum_update_to_physical_response_ratio,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "PokeFlex prior-aware scales must be positive",
        )
        _require(self.assignment_candidates >= 1, "assignment count must be positive")
        _require(
            self.minimum_points_per_sensor >= 1,
            "minimum sensor support must be positive",
        )
        _require(self.maximum_state_rank >= 1, "state rank must be positive")
        _require(
            0.0 < self.minimum_singular_fraction <= 1.0,
            "minimum singular fraction must lie in (0, 1]",
        )
        for name, value in (
            ("minimum_sensor_reliability", self.minimum_sensor_reliability),
            ("prior_nominal_probability", self.prior_nominal_probability),
        ):
            _require(0.0 < value <= 1.0, f"{name} must lie in (0, 1]")


@dataclass(frozen=True)
class PokeFlexPriorAwareFrameArtifactsV1:
    """Typed observation and physical proposal for one causal source frame."""

    observation_belief: ObservationBeliefV1
    linearization: PhysicalLinearizationV1
    physical_prediction_xyz_m: np.ndarray
    state_prior_covariance_m2: np.ndarray
    shared_bias_jacobian: np.ndarray
    view_bias_jacobian: np.ndarray
    assignment_vertex_indices: np.ndarray
    assignment_weights: np.ndarray
    state_mode_names: tuple[str, ...]
    state_mode_source_fields: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = self.observation_belief.observation_count
        prediction = _readonly(self.physical_prediction_xyz_m)
        state_prior = _readonly(self.state_prior_covariance_m2)
        shared = _readonly(self.shared_bias_jacobian)
        view = _readonly(self.view_bias_jacobian)
        indices = _readonly(self.assignment_vertex_indices, dtype=np.int64)
        weights = _readonly(self.assignment_weights)
        transform = _readonly(self.state_mode_source_fields)
        state_count = self.linearization.state_jacobian.shape[2]
        _require(prediction.shape == (count, 3), "physical prediction shape changed")
        _require(
            state_prior.shape == (state_count, state_count),
            "state prior shape changed",
        )
        _require(shared.shape[:2] == (count, 3), "shared bias row shape changed")
        _require(view.shape[:2] == (count, 3), "view bias row shape changed")
        _require(
            indices.ndim == 2 and indices.shape[0] == count, "assignment shape changed"
        )
        _require(weights.shape == indices.shape, "assignment weight shape changed")
        _require(
            np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-12),
            "assignment weights must sum to one",
        )
        _require(
            len(self.state_mode_names) == state_count,
            "state mode name count changed",
        )
        _require(
            transform.shape[0] == state_count,
            "state mode source-field transform changed",
        )
        _require(
            self.linearization.observation_artifact_id
            == self.observation_belief.artifact_id,
            "observation and linearization provenance differ",
        )
        object.__setattr__(self, "physical_prediction_xyz_m", prediction)
        object.__setattr__(self, "state_prior_covariance_m2", state_prior)
        object.__setattr__(self, "shared_bias_jacobian", shared)
        object.__setattr__(self, "view_bias_jacobian", view)
        object.__setattr__(self, "assignment_vertex_indices", indices)
        object.__setattr__(self, "assignment_weights", weights)
        object.__setattr__(self, "state_mode_source_fields", transform)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PokeFlexPriorAwareInferenceV1:
    """Numerical posterior and uncertainty-bearing target correction."""

    result: GaugeAwareBeliefResult
    candidate_vertices_m: np.ndarray
    candidate_covariance_m2: np.ndarray
    query_update_m: np.ndarray

    def __post_init__(self) -> None:
        candidate = _readonly(self.candidate_vertices_m)
        covariance = _readonly(self.candidate_covariance_m2)
        update = _readonly(self.query_update_m)
        _require(
            candidate.ndim == 2 and candidate.shape[1] == 3, "candidate must be Nx3"
        )
        _require(update.shape == candidate.shape, "query update shape changed")
        _require(
            covariance.shape == (len(candidate), 3, 3),
            "candidate covariance shape changed",
        )
        object.__setattr__(self, "candidate_vertices_m", candidate)
        object.__setattr__(self, "candidate_covariance_m2", covariance)
        object.__setattr__(self, "query_update_m", update)

    def select_or_exact_fallback(self, baseline_vertices_m: np.ndarray) -> np.ndarray:
        """Return the candidate or the exact caller-owned baseline object."""

        baseline = _points(baseline_vertices_m, name="baseline target vertices")
        _require(
            baseline.shape == self.candidate_vertices_m.shape,
            "baseline target shape changed",
        )
        return (
            self.candidate_vertices_m
            if self.result.inference_admissible
            else baseline_vertices_m
        )


def _calibration_vector(
    anchor: PokeFlexIndependentDepthAnchor,
    key: str,
) -> np.ndarray:
    metadata = dict(anchor.metadata or {})
    value = np.asarray(metadata.get(key, ()), dtype=np.float64)
    _require(
        value.shape == (len(anchor.sensor_names),) and np.all(np.isfinite(value)),
        f"anchor {key} inventory changed",
    )
    return value


def _state_modes(
    source_fields: Mapping[str, np.ndarray],
    target_fields: Mapping[str, np.ndarray],
    *,
    point_count: int,
    maximum_rank: int,
    minimum_singular_fraction: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    names = tuple(sorted(source_fields))
    _require(bool(names), "physical correction field inventory is empty")
    _require(set(names) == set(target_fields), "source and target fields differ")
    source_columns = []
    target_columns = []
    for name in names:
        source = _points(source_fields[name], name=f"source correction field {name}")
        target = _points(target_fields[name], name=f"target correction field {name}")
        _require(
            source.shape == target.shape == (point_count, 3),
            f"correction field {name} changed topology",
        )
        source_columns.append(source.reshape(-1))
        target_columns.append(target.reshape(-1))
    source_matrix = np.column_stack(source_columns)
    target_matrix = np.column_stack(target_columns)
    joint = np.vstack((source_matrix, target_matrix))
    _, singular_values, right_transpose = np.linalg.svd(joint, full_matrices=False)
    _require(
        len(singular_values) > 0 and singular_values[0] > 0.0,
        "physical correction fields contain no response",
    )
    retained = min(
        maximum_rank,
        int(np.sum(singular_values >= singular_values[0] * minimum_singular_fraction)),
    )
    _require(retained >= 1, "physical correction span has no retained mode")
    source_modes = np.empty((point_count, 3, retained), dtype=np.float64)
    target_modes = np.empty_like(source_modes)
    transforms = np.empty((retained, len(names)), dtype=np.float64)
    mode_names = []
    for mode in range(retained):
        transform = right_transpose[mode].copy()
        source = (source_matrix @ transform).reshape(point_count, 3)
        target = (target_matrix @ transform).reshape(point_count, 3)
        scale = float(
            max(
                np.max(np.linalg.norm(source, axis=1)),
                np.max(np.linalg.norm(target, axis=1)),
            )
        )
        _require(scale > 0.0, "retained physical correction mode vanished")
        source /= scale
        target /= scale
        transform /= scale
        pivot = int(np.argmax(np.abs(np.concatenate((source.ravel(), target.ravel())))))
        joined = np.concatenate((source.ravel(), target.ravel()))
        if joined[pivot] < 0.0:
            source *= -1.0
            target *= -1.0
            transform *= -1.0
        source_modes[:, :, mode] = source
        target_modes[:, :, mode] = target
        transforms[mode] = transform
        mode_names.append(f"physical-span-mode-{mode}")
    return source_modes, target_modes, tuple(mode_names), transforms


def _assignment_rows(
    anchor: PokeFlexIndependentDepthAnchor,
    baseline_vertices_m: np.ndarray,
    source_modes: np.ndarray,
    calibration_median_m: np.ndarray,
    calibration_p90_m: np.ndarray,
    config: PokeFlexPriorAwareConfigV1,
) -> dict[str, np.ndarray]:
    vertex_count = len(baseline_vertices_m)
    candidates = min(config.assignment_candidates, vertex_count)
    distance, index = cKDTree(baseline_vertices_m).query(
        anchor.points_m,
        k=candidates,
    )
    if candidates == 1:
        distance = np.asarray(distance)[:, None]
        index = np.asarray(index)[:, None]
    distance = np.asarray(distance, dtype=np.float64)
    index = np.asarray(index, dtype=np.int64)
    eligible_sensor = (
        calibration_median_m <= config.maximum_calibration_median_residual_m
    )
    geometrically_supported = distance[:, 0] <= config.maximum_association_m
    keep = geometrically_supported & eligible_sensor[anchor.sensor_index]
    initial_supported_count = int(np.sum(keep))

    # A repeated carrier must not change the grouped likelihood dimension.
    for sensor in range(len(anchor.sensor_names)):
        selected = np.flatnonzero(keep & (anchor.sensor_index == sensor))
        if not len(selected):
            continue
        _, first = np.unique(anchor.points_m[selected], axis=0, return_index=True)
        retained = selected[np.sort(first)]
        dropped = np.setdiff1d(selected, retained, assume_unique=True)
        keep[dropped] = False

    # An exact duplicate camera is provenance, not a second independent sensor.
    sensor_signatures: dict[bytes, int] = {}
    duplicate_sensor_count = 0
    for sensor in range(len(anchor.sensor_names)):
        selected = np.flatnonzero(keep & (anchor.sensor_index == sensor))
        if not len(selected):
            continue
        points = np.asarray(anchor.points_m[selected], dtype=np.float64)
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        signature = np.ascontiguousarray(points[order]).tobytes()
        if signature in sensor_signatures:
            keep[selected] = False
            duplicate_sensor_count += 1
        else:
            sensor_signatures[signature] = sensor

    for sensor in range(len(anchor.sensor_names)):
        sensor_rows = keep & (anchor.sensor_index == sensor)
        if int(np.sum(sensor_rows)) < config.minimum_points_per_sensor:
            keep[sensor_rows] = False
    _require(np.any(keep), "no D405 sensor has admissible association support")

    distance = distance[keep]
    index = index[keep]
    sensor_index = np.asarray(anchor.sensor_index[keep], dtype=np.int64)
    observation = np.asarray(anchor.points_m[keep], dtype=np.float64)
    variance = np.asarray(anchor.variance_m2[keep], dtype=np.float64)
    log_weight = -0.5 * np.square(distance / config.assignment_temperature_m)
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(log_weight)
    weight /= np.sum(weight, axis=1, keepdims=True)

    assigned_vertices = baseline_vertices_m[index]
    prediction = np.einsum("nk,nkc->nc", weight, assigned_vertices)
    state_design = np.einsum("nk,nkcs->ncs", weight, source_modes[index])
    local_covariance = np.empty((len(observation), 3, 3), dtype=np.float64)
    state_variance = config.state_prior_std_m**2
    for row in range(len(observation)):
        vertex_delta = assigned_vertices[row] - prediction[row]
        covariance = np.einsum(
            "k,kc,kd->cd",
            weight[row],
            vertex_delta,
            vertex_delta,
        )
        mode_delta = source_modes[index[row]] - state_design[row][None]
        covariance += state_variance * np.einsum(
            "k,kcs,kds->cd",
            weight[row],
            mode_delta,
            mode_delta,
        )
        covariance += (variance[row] + config.minimum_local_std_m**2) * np.eye(3)
        local_covariance[row] = 0.5 * (covariance + covariance.T)

    support_probability = np.exp(
        -0.5 * np.square(distance[:, 0] / config.maximum_association_m)
    )
    association_probability = np.clip(
        support_probability * np.max(weight, axis=1),
        0.0,
        1.0,
    )
    sensor_reliability = np.clip(
        np.exp(
            -0.5 * np.square(calibration_p90_m / config.calibration_reliability_scale_m)
        ),
        config.minimum_sensor_reliability,
        1.0,
    )
    prior_reliability = sensor_reliability[sensor_index]

    group_ids = np.unique(sensor_index)
    group_weight = np.empty(len(group_ids), dtype=np.float64)
    for position, sensor in enumerate(group_ids):
        count = int(np.sum(sensor_index == sensor))
        group_weight[position] = min(
            1.0,
            config.effective_samples_per_sensor / count,
        )
    return {
        "observation": observation,
        "sensor_index": sensor_index,
        "prediction": prediction,
        "state_design": state_design,
        "local_covariance": local_covariance,
        "association_probability": association_probability,
        "prior_reliability": prior_reliability,
        "assignment_index": index,
        "assignment_weight": weight,
        "group_ids": group_ids,
        "group_weight": group_weight,
        "deduplicated_row_count": np.asarray(
            initial_supported_count - int(np.sum(keep)),
            dtype=np.int64,
        ),
        "duplicate_sensor_count": np.asarray(
            duplicate_sensor_count,
            dtype=np.int64,
        ),
    }


def build_pokeflex_prior_aware_frame_artifacts(
    *,
    anchor: PokeFlexIndependentDepthAnchor,
    baseline_source_vertices_m: np.ndarray,
    baseline_target_vertices_m: np.ndarray,
    source_correction_fields_m: Mapping[str, np.ndarray],
    target_correction_fields_m: Mapping[str, np.ndarray] | None = None,
    baseline_belief_id: str,
    action_prefix_id: str,
    simulator_revision: str,
    source_revision: str,
    source_artifact_sha256: str,
    config: PokeFlexPriorAwareConfigV1 | None = None,
) -> PokeFlexPriorAwareFrameArtifactsV1:
    """Build one source-causal D405 belief and row-bound physical proposal."""

    cfg = config or PokeFlexPriorAwareConfigV1()
    source_vertices = _points(
        baseline_source_vertices_m,
        name="baseline source vertices",
    )
    target_vertices = _points(
        baseline_target_vertices_m,
        name="baseline target vertices",
    )
    _require(
        source_vertices.shape == target_vertices.shape,
        "source and target material topology differ",
    )
    _require(
        anchor.causal_cutoff_frame == anchor.frame_id,
        "D405 anchor is not source-causal",
    )
    source_modes, target_modes, mode_names, transform = _state_modes(
        source_correction_fields_m,
        source_correction_fields_m
        if target_correction_fields_m is None
        else target_correction_fields_m,
        point_count=len(source_vertices),
        maximum_rank=cfg.maximum_state_rank,
        minimum_singular_fraction=cfg.minimum_singular_fraction,
    )
    calibration_median = _calibration_vector(
        anchor,
        "calibration_median_residual_m",
    )
    calibration_p90 = _calibration_vector(
        anchor,
        "calibration_p90_residual_m",
    )
    rows = _assignment_rows(
        anchor,
        source_vertices,
        source_modes,
        calibration_median,
        calibration_p90,
        cfg,
    )
    observation_count = len(rows["observation"])
    group_ids = np.asarray(rows["group_ids"], dtype=np.int64)
    group_position = {int(group): position for position, group in enumerate(group_ids)}
    row_group_weight = np.asarray(
        [
            rows["group_weight"][group_position[int(sensor)]]
            for sensor in rows["sensor_index"]
        ],
        dtype=np.float64,
    )
    correlation_group = np.asarray(rows["sensor_index"], dtype=np.int64)
    target_frame = anchor.frame_id + 1
    observation_belief = ObservationBeliefV1(
        case_id=anchor.take_id,
        stream_id="pokeflex:d405-prior-aware-v1",
        causal_frame_stop=target_frame,
        view_names=tuple(anchor.sensor_names),
        window_names=(f"source-frame-{anchor.frame_id}",),
        factor_names=(),
        source_repository="FlorianPfaff/Bayesian-PhysTwin",
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        declared_frame_ids=np.asarray([anchor.frame_id], dtype=np.int64),
        mean_xyz_m=rows["observation"],
        frame_ids=np.full(observation_count, anchor.frame_id, dtype=np.int64),
        entity_ids=np.arange(observation_count, dtype=np.int64),
        view_indices=rows["sensor_index"],
        window_indices=np.zeros(observation_count, dtype=np.int64),
        correlation_group_ids=correlation_group,
        factor_group_ids=np.zeros(observation_count, dtype=np.int64),
        prior_reliability=rows["prior_reliability"],
        association_probability=rows["association_probability"],
        local_covariance_m2=rows["local_covariance"],
        low_rank_factor_m=np.zeros((observation_count, 3, 0)),
        group_ids=group_ids,
        group_prior_nominal_probability=np.full(
            len(group_ids),
            cfg.prior_nominal_probability,
        ),
        group_composite_weight=np.asarray(rows["group_weight"]),
        metadata={
            "association_semantics": "geometry-only-soft-assignment-v1",
            "assignment_probability_used_as_reliability": False,
            "assignment_mixture_spread_in_covariance": True,
            "calibration_reliability_uses_state_innovation": False,
            "correlation_treatment": "one-capped-group-per-d405-sensor",
            "effective_samples_per_group": cfg.effective_samples_per_sensor,
            "group_composite_weight_semantics": (
                "final-per-row-effective-sample-cap-v1"
            ),
            "source_frame": anchor.frame_id,
            "target_frame": target_frame,
            "sensor_group_weight_by_row_mean": float(np.mean(row_group_weight)),
            "deduplicated_row_count": int(rows["deduplicated_row_count"]),
            "duplicate_sensor_count": int(rows["duplicate_sensor_count"]),
        },
    )
    physical_response = target_vertices - source_vertices
    if np.max(np.linalg.norm(physical_response, axis=1), initial=0.0) <= 0.0:
        physical_response = np.sqrt(np.sum(np.square(target_modes), axis=2))
    linearization = PhysicalLinearizationV1(
        observation_artifact_id=observation_belief.artifact_id,
        baseline_belief_id=baseline_belief_id,
        action_prefix_id=action_prefix_id,
        simulator_revision=simulator_revision,
        frame_ids=observation_belief.frame_ids,
        entity_ids=observation_belief.entity_ids,
        view_indices=observation_belief.view_indices,
        window_indices=observation_belief.window_indices,
        state_jacobian=rows["state_design"],
        query_state_jacobian=target_modes,
        physical_response_m=physical_response,
        metadata={
            "state_mode_names": list(mode_names),
            "source_field_names": sorted(source_correction_fields_m),
            "source_frame": anchor.frame_id,
            "target_frame": target_frame,
            "proposal_only_kinect_innovation_reused_in_likelihood": False,
        },
    )
    shared = global_translation_bias_jacobian(observation_count)
    view = centered_view_translation_bias_jacobian(
        observation_belief.view_indices,
        view_count=len(observation_belief.view_names),
    )
    state_prior = np.eye(len(mode_names)) * cfg.state_prior_std_m**2
    return PokeFlexPriorAwareFrameArtifactsV1(
        observation_belief=observation_belief,
        linearization=linearization,
        physical_prediction_xyz_m=rows["prediction"],
        state_prior_covariance_m2=state_prior,
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        assignment_vertex_indices=rows["assignment_index"],
        assignment_weights=rows["assignment_weight"],
        state_mode_names=mode_names,
        state_mode_source_fields=transform,
        metadata={
            "calibration_median_residual_m": calibration_median.tolist(),
            "calibration_p90_residual_m": calibration_p90.tolist(),
            "active_sensor_indices": group_ids.tolist(),
            "deduplicated_row_count": int(rows["deduplicated_row_count"]),
            "duplicate_sensor_count": int(rows["duplicate_sensor_count"]),
        },
    )


def _spd_covariance_from_precision(precision: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (precision + precision.T)
    factor = np.linalg.cholesky(symmetric)
    inverse_factor = np.linalg.solve(factor, np.eye(len(factor)))
    covariance = inverse_factor.T @ inverse_factor
    return 0.5 * (covariance + covariance.T)


def _apply_known_bias_covariance_floor(
    artifacts: PokeFlexPriorAwareFrameArtifactsV1,
    result: GaugeAwareBeliefResult,
) -> GaugeAwareBeliefResult:
    """Bound confidence using the same final robust responsibilities once."""

    if not result.inference_admissible:
        return result
    belief = artifacts.observation_belief
    state_count = artifacts.linearization.state_jacobian.shape[2]
    precision = _spd_covariance_from_precision(artifacts.state_prior_covariance_m2)
    row_composite = np.asarray(
        [
            belief.group_composite_weight[belief.group_position(int(group_id))]
            for group_id in belief.correlation_group_ids
        ],
        dtype=np.float64,
    )
    row_weight = belief.prior_reliability * row_composite * result.robust_weights
    for row, weight in enumerate(row_weight):
        if weight <= 0.0:
            continue
        factor = np.linalg.cholesky(belief.local_covariance_m2[row])
        white_state = np.linalg.solve(
            factor,
            artifacts.linearization.state_jacobian[row],
        )
        precision += weight * (white_state.T @ white_state)
    known_bias_covariance = _spd_covariance_from_precision(precision)
    covariance = np.asarray(result.posterior_covariance).copy()
    state_covariance = covariance[:state_count, :state_count]
    difference = known_bias_covariance - state_covariance
    difference = 0.5 * (difference + difference.T)
    eigenvalues, eigenvectors = np.linalg.eigh(difference)
    positive_difference = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    covariance[:state_count, :state_count] += positive_difference
    covariance = 0.5 * (covariance + covariance.T)
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "unknown_correlation_covariance_floor": (
                "known-bias-conditional-at-final-robust-responsibilities"
            ),
            "innovation_reprocessed_for_covariance_floor": False,
            "known_bias_conditional_state_covariance_trace_m2": float(
                np.trace(known_bias_covariance)
            ),
            "state_covariance_floor_added_trace_m2": float(
                np.trace(positive_difference)
            ),
        }
    )
    return replace(
        result,
        posterior_covariance=covariance,
        diagnostics=diagnostics,
    )


def infer_pokeflex_prior_aware_frame(
    artifacts: PokeFlexPriorAwareFrameArtifactsV1,
    baseline_target_vertices_m: np.ndarray,
    *,
    config: PokeFlexPriorAwareConfigV1 | None = None,
) -> PokeFlexPriorAwareInferenceV1:
    """Infer a bias-marginalized state update from one typed frame pair."""

    cfg = config or PokeFlexPriorAwareConfigV1()
    baseline = _points(baseline_target_vertices_m, name="baseline target vertices")
    _require(
        len(baseline) == len(artifacts.linearization.query_state_jacobian),
        "baseline target topology differs from the physical query",
    )
    adapted = build_gauge_aware_batch_from_artifacts(
        artifacts.observation_belief,
        artifacts.linearization,
        physical_prediction_xyz_m=artifacts.physical_prediction_xyz_m,
        shared_bias_jacobian=artifacts.shared_bias_jacobian,
        view_bias_jacobian=artifacts.view_bias_jacobian,
        state_prior_covariance_m2=artifacts.state_prior_covariance_m2,
    )
    result = update_prior_aware_gauge_belief(
        adapted.batch,
        config=PriorAwareGaugeConfigV1(
            state_prior_std_m=cfg.state_prior_std_m,
            shared_bias_prior_std_m=cfg.shared_bias_prior_std_m,
            view_bias_prior_std_m=cfg.view_bias_prior_std_m,
            effective_samples_per_correlation_group=(cfg.effective_samples_per_sensor),
            maximum_state_update_m=cfg.maximum_state_update_m,
            maximum_update_to_physical_response_ratio=(
                cfg.maximum_update_to_physical_response_ratio
            ),
        ),
    )
    result = _apply_known_bias_covariance_floor(artifacts, result)
    query = artifacts.linearization.query_state_jacobian
    update = np.einsum("qcs,s->qc", query, result.state_coefficients)
    candidate = baseline + update
    state_count = query.shape[2]
    state_covariance = result.posterior_covariance[:state_count, :state_count]
    candidate_covariance = np.einsum(
        "qcs,st,qdt->qcd",
        query,
        state_covariance,
        query,
    )
    return PokeFlexPriorAwareInferenceV1(
        result=result,
        candidate_vertices_m=candidate,
        candidate_covariance_m2=candidate_covariance,
        query_update_m=update,
    )


__all__ = [
    "PokeFlexPriorAwareConfigV1",
    "PokeFlexPriorAwareFrameArtifactsV1",
    "PokeFlexPriorAwareInferenceV1",
    "build_pokeflex_prior_aware_frame_artifacts",
    "infer_pokeflex_prior_aware_frame",
]
