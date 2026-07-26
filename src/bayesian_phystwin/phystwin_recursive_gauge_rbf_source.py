"""Source-only PhysTwin evaluation for recursive gauge-aware RBF updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._gauge_aware_contracts import GaugeAwareBeliefConfig
from .observation_belief import ObservationBeliefV1
from .phystwin_bayesian_anchor import robust_random_walk_endpoint
from .phystwin_comparison import official_metrics_by_frame
from .phystwin_official_evaluation import _nearest_distances
from .phystwin_online_belief import deterministic_farthest_point_ids
from .phystwin_residual_dynamics import _target_validity
from .phystwin_sparse_identity_observation import (
    SparseIdentityObservations,
)
from .recursive_gauge_rbf_belief import (
    RecursiveGaugeRbfConfig,
    RecursiveGaugeRbfSnapshot,
    decode_recursive_gauge_rbf_belief,
    initialize_recursive_gauge_rbf_belief,
    predict_recursive_gauge_rbf_belief,
    update_recursive_gauge_rbf_belief,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(
    values: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhysTwinRecursiveGaugeRbfSourceConfig:
    """Frozen source-smoke settings independent of future outcomes."""

    center_count: int = 16
    update_count: int = 4
    minimum_center_availability_fraction: float = 0.25
    minimum_rows_per_update: int = 4
    transport_neighbor_count: int = 16
    physical_response_floor_m: float = 0.001
    minimum_prefix_cd_improvement_fraction: float = 0.01
    dense_process_std_m: float = 0.005
    dense_observation_std_m: float = 0.005
    dense_initial_std_m: float = 0.01
    dense_inlier_prior: float = 0.95
    dense_outlier_variance_multiplier: float = 100.0
    dense_cap_quantile: float = 0.95
    dense_cap_multiplier: float = 1.0
    dense_temporal_gamma: float = 0.25
    recursive: RecursiveGaugeRbfConfig = field(
        default_factory=lambda: RecursiveGaugeRbfConfig(
            length_scale_fraction=0.10,
            local_blend=0.25,
            global_prior_std_m=0.10,
            local_prior_std_m=0.02,
            global_process_std_m_per_sqrt_frame=0.003,
            local_process_std_m_per_sqrt_frame=0.003,
            maximum_total_query_correction_m=0.10,
            gauge_update=GaugeAwareBeliefConfig(
                state_prior_std_m=0.02,
                shared_bias_prior_std_m=0.02,
                view_bias_prior_std_m=0.01,
                effective_samples_per_correlation_group=8.0,
                degrees_of_freedom=4.0,
                minimum_robust_weight=0.02,
                maximum_state_update_m=0.05,
                maximum_update_to_physical_response_ratio=2.0,
            ),
        )
    )

    def __post_init__(self) -> None:
        _require(self.center_count >= 1, "center_count must be positive")
        _require(self.update_count >= 1, "update_count must be positive")
        _require(
            0.0 <= self.minimum_center_availability_fraction <= 1.0,
            "minimum center availability must lie in [0, 1]",
        )
        _require(
            1 <= self.minimum_rows_per_update <= self.center_count,
            "minimum_rows_per_update must lie in [1, center_count]",
        )
        _require(
            self.transport_neighbor_count >= 3,
            "transport_neighbor_count must be at least three",
        )
        positive = (
            self.physical_response_floor_m,
            self.dense_process_std_m,
            self.dense_observation_std_m,
            self.dense_initial_std_m,
            self.dense_cap_multiplier,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "source-smoke metric scales must be positive",
        )
        _require(
            0.0 <= self.minimum_prefix_cd_improvement_fraction < 1.0,
            "minimum prefix CD improvement must lie in [0, 1)",
        )
        _require(
            0.0 < self.dense_inlier_prior < 1.0,
            "dense inlier prior must lie in (0, 1)",
        )
        _require(
            self.dense_outlier_variance_multiplier > 1.0,
            "dense outlier multiplier must exceed one",
        )
        _require(
            0.0 < self.dense_cap_quantile < 1.0,
            "dense cap quantile must lie in (0, 1)",
        )
        _require(
            np.isfinite(self.dense_temporal_gamma)
            and self.dense_temporal_gamma >= 0.0,
            "dense temporal gamma must be nonnegative",
        )


@dataclass(frozen=True)
class PhysTwinRecursiveGaugeRbfPrediction:
    """Prediction carrier sealed before future observations are scored."""

    dense_baseline: np.ndarray
    candidate: np.ndarray
    correction_mean_m: np.ndarray
    correction_covariance_m2: np.ndarray
    center_ids: np.ndarray
    prefix_admitted: bool
    prefix_baseline_cd_m: float
    prefix_candidate_cd_m: float
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        baseline = _readonly(self.dense_baseline)
        candidate = _readonly(self.candidate)
        mean = _readonly(self.correction_mean_m, dtype=np.float64)
        covariance = _readonly(
            self.correction_covariance_m2,
            dtype=np.float64,
        )
        centers = _readonly(self.center_ids, dtype=np.int64)
        _require(
            baseline.ndim == 3
            and baseline.shape[2] == 3
            and candidate.shape == baseline.shape,
            "prediction trajectories must share shape (T, N, 3)",
        )
        _require(
            mean.shape == baseline.shape,
            "correction mean must match trajectory shape",
        )
        _require(
            covariance.shape == (*baseline.shape, 3),
            "correction covariance must have shape (T, N, 3, 3)",
        )
        _require(
            np.all(np.isfinite(baseline))
            and np.all(np.isfinite(candidate))
            and np.all(np.isfinite(mean))
            and np.all(np.isfinite(covariance)),
            "prediction carrier contains non-finite values",
        )
        _require(
            centers.ndim == 1
            and len(centers)
            and len(np.unique(centers)) == len(centers),
            "center IDs must be a nonempty unique vector",
        )
        if not self.prefix_admitted:
            _require(
                candidate.dtype == baseline.dtype
                and candidate.tobytes() == baseline.tobytes(),
                "rejected source prediction did not preserve exact fallback",
            )
        object.__setattr__(self, "dense_baseline", baseline)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "correction_mean_m", mean)
        object.__setattr__(self, "correction_covariance_m2", covariance)
        object.__setattr__(self, "center_ids", centers)


def _clip_vectors(values: np.ndarray, maximum_norm_m: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    result *= np.minimum(
        1.0,
        maximum_norm_m / np.maximum(norm, 1e-15),
    )
    return result


def _dense_endpoint_mean(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    state_count: int,
    config: PhysTwinRecursiveGaugeRbfSourceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    posterior = robust_random_walk_endpoint(
        residual_m,
        valid,
        end_frame=end_frame,
        process_variance=config.dense_process_std_m**2,
        observation_variance=config.dense_observation_std_m**2,
        initial_variance=config.dense_initial_std_m**2,
        inlier_prior=config.dense_inlier_prior,
        outlier_variance_multiplier=(
            config.dense_outlier_variance_multiplier
        ),
    )
    original_count = residual_m.shape[1]
    mean = np.zeros((state_count, 3), dtype=np.float64)
    mean[:original_count] = posterior.mean
    updated = posterior.update_count > 0
    return mean, updated


def _capped_dense_mean(
    mean_m: np.ndarray,
    updated: np.ndarray,
    *,
    original_count: int,
    config: PhysTwinRecursiveGaugeRbfSourceConfig,
) -> np.ndarray:
    if not np.any(updated):
        return np.zeros_like(mean_m)
    reference = float(
        np.quantile(
            np.linalg.norm(
                mean_m[:original_count][updated],
                axis=1,
            ),
            config.dense_cap_quantile,
        )
    )
    cap_m = max(config.dense_cap_multiplier * reference, 1e-6)
    return _clip_vectors(mean_m, cap_m)


def _dense_temporal_interval(
    endpoint_mean_m: np.ndarray,
    previous_mean_m: np.ndarray,
    updated: np.ndarray,
    *,
    interval_count: int,
    reference_count: int,
    original_count: int,
    config: PhysTwinRecursiveGaugeRbfSourceConfig,
) -> np.ndarray:
    delta = endpoint_mean_m - previous_mean_m
    correction = np.empty(
        (interval_count, len(endpoint_mean_m), 3),
        dtype=np.float64,
    )
    for offset in range(interval_count):
        ratio = (offset + 1) / max(reference_count, 1)
        predicted = (
            endpoint_mean_m
            + config.dense_temporal_gamma * ratio * delta
        )
        correction[offset] = _capped_dense_mean(
            predicted,
            updated,
            original_count=original_count,
            config=config,
        )
    return correction


def build_dense_temporal_comparator(
    raw_baseline: np.ndarray,
    observed_points_prefix_m: np.ndarray,
    visibility_prefix: np.ndarray,
    motion_valid_prefix: np.ndarray,
    *,
    fit_end_frame: int,
    train_end_frame: int,
    config: PhysTwinRecursiveGaugeRbfSourceConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild the frozen released-dense temporal comparator from its prefix."""

    cfg = config or PhysTwinRecursiveGaugeRbfSourceConfig()
    raw = np.asarray(raw_baseline)
    observed = np.asarray(observed_points_prefix_m, dtype=np.float64)
    visible = np.asarray(visibility_prefix, dtype=bool)
    motion_valid = np.asarray(motion_valid_prefix, dtype=bool)
    _require(
        raw.ndim == 3 and raw.shape[2] == 3,
        "raw baseline must have shape (T, N, 3)",
    )
    _require(
        observed.shape[:2] == visible.shape
        and observed.shape[2] == 3,
        "observed prefix and visibility have inconsistent shapes",
    )
    _require(
        len(observed) == train_end_frame
        and len(motion_valid) == train_end_frame - 1,
        "prefix arrays do not end at train_end_frame",
    )
    _require(
        3 < fit_end_frame < train_end_frame < len(raw),
        "dense comparator requires a nonempty validation and future",
    )
    original_count = observed.shape[1]
    _require(
        raw.shape[1] >= original_count,
        "raw baseline has fewer points than the observation prefix",
    )
    valid = _target_validity(visible, motion_valid)
    residual = observed - raw[:train_end_frame, :original_count]
    inner_end = max(3, fit_end_frame - (train_end_frame - fit_end_frame))
    inner_mean, _ = _dense_endpoint_mean(
        residual,
        valid,
        end_frame=inner_end,
        state_count=raw.shape[1],
        config=cfg,
    )
    fit_mean, fit_updated = _dense_endpoint_mean(
        residual,
        valid,
        end_frame=fit_end_frame,
        state_count=raw.shape[1],
        config=cfg,
    )
    full_mean, full_updated = _dense_endpoint_mean(
        residual,
        valid,
        end_frame=train_end_frame,
        state_count=raw.shape[1],
        config=cfg,
    )
    validation_correction = _dense_temporal_interval(
        fit_mean,
        inner_mean,
        fit_updated,
        interval_count=train_end_frame - fit_end_frame,
        reference_count=fit_end_frame - inner_end,
        original_count=original_count,
        config=cfg,
    )
    future_correction = _dense_temporal_interval(
        full_mean,
        fit_mean,
        full_updated,
        interval_count=len(raw) - train_end_frame,
        reference_count=train_end_frame - fit_end_frame,
        original_count=original_count,
        config=cfg,
    )
    comparator = raw.copy()
    comparator[fit_end_frame:train_end_frame] = (
        raw[fit_end_frame:train_end_frame] + validation_correction
    ).astype(raw.dtype, copy=False)
    comparator[train_end_frame:] = (
        raw[train_end_frame:] + future_correction
    ).astype(raw.dtype, copy=False)
    return comparator, {
        "inner_end_frame_exclusive": inner_end,
        "fit_end_frame_exclusive": fit_end_frame,
        "train_end_frame_exclusive": train_end_frame,
        "dense_temporal_gamma": cfg.dense_temporal_gamma,
        "dense_cap_quantile": cfg.dense_cap_quantile,
        "dense_cap_multiplier": cfg.dense_cap_multiplier,
        "fit_updated_count": int(np.sum(fit_updated)),
        "full_updated_count": int(np.sum(full_updated)),
    }


def _online_dense_prediction(
    raw_baseline: np.ndarray,
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    frame_index: int,
    original_count: int,
    config: PhysTwinRecursiveGaugeRbfSourceConfig,
) -> np.ndarray:
    mean, updated = _dense_endpoint_mean(
        residual_m,
        valid,
        end_frame=frame_index + 1,
        state_count=raw_baseline.shape[1],
        config=config,
    )
    correction = _capped_dense_mean(
        mean,
        updated,
        original_count=original_count,
        config=config,
    )
    return raw_baseline[frame_index] + correction


def fixed_update_frames(end_frame: int, update_count: int) -> np.ndarray:
    """Return deterministic positive frame indices ending at ``end_frame - 1``."""

    _require(end_frame >= 2, "end_frame must contain frame zero and one update")
    _require(update_count >= 1, "update_count must be positive")
    values = [
        max(1, int(round(fraction * (end_frame - 1))))
        for fraction in np.linspace(
            1.0 / update_count,
            1.0,
            update_count,
        )
    ]
    result = np.unique(np.asarray(values, dtype=np.int64))
    result.setflags(write=False)
    return result


def support_adaptive_update_frames(
    valid: np.ndarray,
    *,
    end_frame: int,
    update_count: int,
    minimum_rows: int,
) -> np.ndarray:
    """Select quantiles of causally supported frames, including the last."""

    availability = np.asarray(valid, dtype=bool)
    _require(
        availability.ndim == 2,
        "valid must have shape (T, K)",
    )
    _require(
        2 <= end_frame <= len(availability),
        "end_frame must lie inside the validity sequence",
    )
    _require(update_count >= 1, "update_count must be positive")
    _require(minimum_rows >= 1, "minimum_rows must be positive")
    supported = np.flatnonzero(
        (np.arange(end_frame) > 0)
        & (np.sum(availability[:end_frame], axis=1) >= minimum_rows)
    )
    _require(len(supported) > 0, "no causally supported update frame")
    count = min(update_count, len(supported))
    positions = np.rint(
        np.linspace(0, len(supported) - 1, count)
    ).astype(np.int64)
    result = np.unique(supported[positions])
    result.setflags(write=False)
    return result


def _proper_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cross = source.T @ target
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    return rotation


def action_conditioned_rigid_transition(
    snapshot: RecursiveGaugeRbfSnapshot,
    previous_object_positions_m: np.ndarray,
    current_object_positions_m: np.ndarray,
    *,
    neighbor_count: int,
) -> np.ndarray:
    """Transport local discrepancy vectors with physical local rotations."""

    previous = np.asarray(previous_object_positions_m, dtype=np.float64)
    current = np.asarray(current_object_positions_m, dtype=np.float64)
    _require(
        previous.shape == current.shape
        and previous.ndim == 2
        and previous.shape[1] == 3,
        "physical transport geometries must share shape (N, 3)",
    )
    _require(
        np.all(np.isfinite(previous)) and np.all(np.isfinite(current)),
        "physical transport geometries must be finite",
    )
    _require(neighbor_count >= 3, "neighbor_count must be at least three")
    transition = np.eye(snapshot.state_dimension)
    count = min(neighbor_count, len(previous))
    for center_position, center_id in enumerate(snapshot.center_ids):
        distance = np.linalg.norm(previous - previous[center_id], axis=1)
        neighbors = np.argsort(distance, kind="mergesort")[:count]
        source = previous[neighbors] - np.mean(previous[neighbors], axis=0)
        target = current[neighbors] - np.mean(current[neighbors], axis=0)
        if (
            np.linalg.matrix_rank(source) < 2
            or np.linalg.matrix_rank(target) < 2
        ):
            rotation = np.eye(3)
        else:
            rotation = _proper_rotation(source, target)
        start = 3 + 3 * center_position
        transition[start : start + 3, start : start + 3] = rotation
    return transition


def sparse_frame_observation_belief(
    observations: SparseIdentityObservations,
    *,
    case_id: str,
    frame_index: int,
    entity_ids: np.ndarray,
    source_revision: str,
    source_artifact_sha256: str,
) -> ObservationBeliefV1 | None:
    """Adapt one fused sparse frame without using a physical innovation."""

    identities = np.asarray(entity_ids, dtype=np.int64)
    available = observations.valid[frame_index, identities]
    selected = identities[available]
    if not len(selected):
        return None
    count = len(selected)
    return ObservationBeliefV1(
        case_id=case_id,
        stream_id=f"cotracker3-strict-multiview-frame-{frame_index}",
        causal_frame_stop=frame_index + 1,
        view_names=("strict-multiview-fused",),
        window_names=("causal-prefix",),
        factor_names=(),
        source_repository="FlorianPfaff/Bayesian-PhysTwin",
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        declared_frame_ids=np.asarray([frame_index]),
        mean_xyz_m=observations.points_world_m[frame_index, selected],
        frame_ids=np.full(count, frame_index),
        entity_ids=selected,
        view_indices=np.zeros(count, dtype=np.int64),
        window_indices=np.zeros(count, dtype=np.int64),
        correlation_group_ids=np.zeros(count, dtype=np.int64),
        factor_group_ids=np.zeros(count, dtype=np.int64),
        prior_reliability=observations.prior_reliability[
            frame_index,
            selected,
        ],
        association_probability=np.ones(count),
        local_covariance_m2=observations.observation_covariance_m2[
            frame_index,
            selected,
        ],
        low_rank_factor_m=np.zeros((count, 3, 0)),
        group_ids=np.asarray([0]),
        group_prior_nominal_probability=np.ones(1),
        group_composite_weight=np.ones(1),
        metadata={
            "association": "archived original PhysTwin material identity",
            "reliability_uses_physical_innovation": False,
            "cross_view_correlation": "covariance intersection",
            "shared_bias": "explicit gauge-aware nuisance",
        },
    )


def _center_ids(
    observations: SparseIdentityObservations,
    initial_positions_m: np.ndarray,
    *,
    fit_end_frame: int,
    config: PhysTwinRecursiveGaugeRbfSourceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    availability = np.mean(observations.valid[:fit_end_frame], axis=0)
    eligible = np.flatnonzero(
        np.all(np.isfinite(initial_positions_m), axis=1)
        & (
            availability
            >= config.minimum_center_availability_fraction
        )
    )
    _require(
        len(eligible) >= config.center_count,
        "insufficient strict-multiview center support",
    )
    centers = deterministic_farthest_point_ids(
        initial_positions_m,
        eligible,
        config.center_count,
    )
    return centers, availability


def _run_prefix_updates(
    raw_baseline: np.ndarray,
    observed_points_prefix_m: np.ndarray,
    dense_valid: np.ndarray,
    sparse_observations: SparseIdentityObservations,
    center_ids: np.ndarray,
    *,
    case_id: str,
    end_frame: int,
    source_revision: str,
    source_artifact_sha256: str,
    config: PhysTwinRecursiveGaugeRbfSourceConfig,
) -> tuple[RecursiveGaugeRbfSnapshot, list[dict[str, Any]]]:
    original_count = observed_points_prefix_m.shape[1]
    residual = (
        observed_points_prefix_m
        - raw_baseline[: len(observed_points_prefix_m), :original_count]
    )
    snapshot = initialize_recursive_gauge_rbf_belief(
        center_ids,
        raw_baseline[0, center_ids],
        raw_baseline[0, :original_count],
        config=config.recursive,
    )
    previous_frame = 0
    diagnostics: list[dict[str, Any]] = []
    update_frames = support_adaptive_update_frames(
        sparse_observations.valid[:, center_ids],
        end_frame=end_frame,
        update_count=config.update_count,
        minimum_rows=config.minimum_rows_per_update,
    )
    for frame in update_frames:
        transition = action_conditioned_rigid_transition(
            snapshot,
            raw_baseline[previous_frame, :original_count],
            raw_baseline[frame, :original_count],
            neighbor_count=config.transport_neighbor_count,
        )
        center_positions = raw_baseline[frame, center_ids]
        belief = sparse_frame_observation_belief(
            sparse_observations,
            case_id=case_id,
            frame_index=int(frame),
            entity_ids=center_ids,
            source_revision=source_revision,
            source_artifact_sha256=source_artifact_sha256,
        )
        available_count = 0 if belief is None else belief.observation_count
        response = np.linalg.norm(
            raw_baseline[frame, center_ids]
            - raw_baseline[previous_frame, center_ids],
            axis=1,
        )
        response_scale = max(
            float(np.median(response)),
            config.physical_response_floor_m,
        )
        if (
            belief is None
            or available_count < config.minimum_rows_per_update
        ):
            snapshot = predict_recursive_gauge_rbf_belief(
                snapshot,
                frame_index=int(frame),
                center_positions_m=center_positions,
                config=config.recursive,
                state_transition=transition,
            )
            diagnostics.append(
                {
                    "frame": int(frame),
                    "accepted": False,
                    "reason": "insufficient-observation-rows",
                    "observation_count": available_count,
                    "physical_response_scale_m": response_scale,
                }
            )
        else:
            online_dense = _online_dense_prediction(
                raw_baseline,
                residual,
                dense_valid,
                frame_index=int(frame),
                original_count=original_count,
                config=config,
            )
            selected = belief.entity_ids
            update = update_recursive_gauge_rbf_belief(
                snapshot,
                frame_index=int(frame),
                center_positions_m=center_positions,
                observation_belief=belief,
                physical_prediction_xyz_m=online_dense[selected],
                query_positions_m=center_positions,
                physical_response_scale_m=response_scale,
                config=config.recursive,
                state_transition=transition,
            )
            snapshot = update.posterior_snapshot
            diagnostics.append(
                {
                    "frame": int(frame),
                    "accepted": update.accepted,
                    "reason": update.reason,
                    "observation_count": available_count,
                    "physical_response_scale_m": response_scale,
                    "identifiable_query_state_mode_count": (
                        update.gauge_result.diagnostics.get(
                            "identifiable_query_state_mode_count"
                        )
                    ),
                    "maximum_state_update_m": (
                        update.gauge_result.diagnostics.get(
                            "maximum_state_update_m"
                        )
                    ),
                }
            )
        previous_frame = int(frame)
    return snapshot, diagnostics


def _forecast_from_snapshot(
    snapshot: RecursiveGaugeRbfSnapshot,
    raw_baseline: np.ndarray,
    dense_baseline: np.ndarray,
    *,
    original_count: int,
    start_frame: int,
    end_frame: int,
    config: PhysTwinRecursiveGaugeRbfSourceConfig,
) -> tuple[np.ndarray, np.ndarray, RecursiveGaugeRbfSnapshot]:
    correction = np.zeros_like(
        dense_baseline[start_frame:end_frame],
        dtype=np.float64,
    )
    covariance = np.zeros(
        (*correction.shape, 3),
        dtype=np.float64,
    )
    previous_frame = snapshot.last_update_frame
    _require(
        previous_frame is not None and previous_frame < start_frame,
        "forecast snapshot must precede the requested interval",
    )
    current = snapshot
    for output, frame in enumerate(range(start_frame, end_frame)):
        transition = action_conditioned_rigid_transition(
            current,
            raw_baseline[previous_frame, :original_count],
            raw_baseline[frame, :original_count],
            neighbor_count=config.transport_neighbor_count,
        )
        current = predict_recursive_gauge_rbf_belief(
            current,
            frame_index=frame,
            center_positions_m=raw_baseline[frame, current.center_ids],
            config=config.recursive,
            state_transition=transition,
        )
        prediction = decode_recursive_gauge_rbf_belief(
            current,
            dense_baseline[frame],
            config=config.recursive,
        )
        correction[output] = prediction.mean_m
        covariance[output] = prediction.covariance_m2
        previous_frame = frame
    return correction, covariance, current


def _mean_cd(
    trajectory: np.ndarray,
    observed_points_m: np.ndarray,
    visibility: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> float:
    by_frame: list[float] = []
    for frame in range(start_frame, end_frame):
        target = observed_points_m[frame, visibility[frame]]
        _require(len(target) > 0, "prefix CD frame has no visible target")
        distance, _ = _nearest_distances(
            trajectory[frame, :num_surface_points],
            target,
            p=1,
        )
        by_frame.append(float(np.mean(distance)))
    return float(np.mean(by_frame))


def run_recursive_gauge_rbf_source_prediction(
    raw_baseline: np.ndarray,
    observed_points_prefix_m: np.ndarray,
    visibility_prefix: np.ndarray,
    motion_valid_prefix: np.ndarray,
    sparse_observations: SparseIdentityObservations,
    *,
    case_id: str,
    fit_end_frame: int,
    train_end_frame: int,
    num_surface_points: int,
    source_revision: str,
    source_artifact_sha256: str,
    config: PhysTwinRecursiveGaugeRbfSourceConfig | None = None,
) -> PhysTwinRecursiveGaugeRbfPrediction:
    """Build a sealed future prediction using prefix observations only."""

    cfg = config or PhysTwinRecursiveGaugeRbfSourceConfig()
    raw = np.asarray(raw_baseline)
    observed = np.asarray(observed_points_prefix_m, dtype=np.float64)
    visible = np.asarray(visibility_prefix, dtype=bool)
    motion_valid = np.asarray(motion_valid_prefix, dtype=bool)
    _require(
        len(observed)
        == len(visible)
        == train_end_frame,
        "prefix arrays must end at train_end_frame",
    )
    _require(
        sparse_observations.points_world_m.shape[:2]
        == observed.shape[:2],
        "sparse observations must match the dense prefix identities",
    )
    dense_baseline, dense_diagnostics = build_dense_temporal_comparator(
        raw,
        observed,
        visible,
        motion_valid,
        fit_end_frame=fit_end_frame,
        train_end_frame=train_end_frame,
        config=cfg,
    )
    original_count = observed.shape[1]
    dense_valid = _target_validity(visible, motion_valid)
    centers, availability = _center_ids(
        sparse_observations,
        raw[0, :original_count],
        fit_end_frame=fit_end_frame,
        config=cfg,
    )

    fit_snapshot, fit_updates = _run_prefix_updates(
        raw,
        observed,
        dense_valid,
        sparse_observations,
        centers,
        case_id=case_id,
        end_frame=fit_end_frame,
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        config=cfg,
    )
    validation_correction, _, _ = _forecast_from_snapshot(
        fit_snapshot,
        raw,
        dense_baseline,
        original_count=original_count,
        start_frame=fit_end_frame,
        end_frame=train_end_frame,
        config=cfg,
    )
    validation_candidate = dense_baseline.copy()
    validation_candidate[fit_end_frame:train_end_frame] = (
        dense_baseline[fit_end_frame:train_end_frame]
        + validation_correction
    ).astype(dense_baseline.dtype, copy=False)
    prefix_baseline_cd = _mean_cd(
        dense_baseline,
        observed,
        visible,
        num_surface_points=num_surface_points,
        start_frame=fit_end_frame,
        end_frame=train_end_frame,
    )
    prefix_candidate_cd = _mean_cd(
        validation_candidate,
        observed,
        visible,
        num_surface_points=num_surface_points,
        start_frame=fit_end_frame,
        end_frame=train_end_frame,
    )
    prefix_improvement = (
        0.0
        if prefix_baseline_cd == 0.0 and prefix_candidate_cd == 0.0
        else (
            -np.finfo(np.float64).max
            if prefix_baseline_cd == 0.0
            else 1.0 - prefix_candidate_cd / prefix_baseline_cd
        )
    )
    prefix_admitted = (
        prefix_improvement
        >= cfg.minimum_prefix_cd_improvement_fraction
    )

    full_snapshot, full_updates = _run_prefix_updates(
        raw,
        observed,
        dense_valid,
        sparse_observations,
        centers,
        case_id=case_id,
        end_frame=train_end_frame,
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        config=cfg,
    )
    future_correction, future_covariance, _ = _forecast_from_snapshot(
        full_snapshot,
        raw,
        dense_baseline,
        original_count=original_count,
        start_frame=train_end_frame,
        end_frame=len(raw),
        config=cfg,
    )
    correction = np.zeros_like(dense_baseline, dtype=np.float64)
    covariance = np.zeros(
        (*dense_baseline.shape, 3),
        dtype=np.float64,
    )
    correction[train_end_frame:] = future_correction
    covariance[train_end_frame:] = future_covariance
    if prefix_admitted:
        candidate = dense_baseline.copy()
        candidate[train_end_frame:] = (
            dense_baseline[train_end_frame:] + future_correction
        ).astype(dense_baseline.dtype, copy=False)
    else:
        candidate = dense_baseline.copy()
        correction[train_end_frame:] = 0.0
        covariance[train_end_frame:] = 0.0

    diagnostics = {
        "method": "recursive-gauge-aware-rbf-v1",
        "dense_comparator": dense_diagnostics,
        "center_count": len(centers),
        "center_ids": centers.tolist(),
        "center_availability_fraction": availability[centers].tolist(),
        "fit_updates": fit_updates,
        "full_updates": full_updates,
        "fit_accepted_update_count": int(
            sum(bool(item["accepted"]) for item in fit_updates)
        ),
        "full_accepted_update_count": int(
            sum(bool(item["accepted"]) for item in full_updates)
        ),
        "prefix_gate_uses_manual_tracks": False,
        "prefix_gate_metric": "released pseudo-track Chamfer distance",
        "prefix_cd_improvement_fraction": prefix_improvement,
        "future_object_observations_used": False,
        "future_manual_tracks_used": False,
        "exact_fallback": not prefix_admitted,
    }
    return PhysTwinRecursiveGaugeRbfPrediction(
        dense_baseline=dense_baseline,
        candidate=candidate,
        correction_mean_m=correction,
        correction_covariance_m2=covariance,
        center_ids=centers,
        prefix_admitted=prefix_admitted,
        prefix_baseline_cd_m=prefix_baseline_cd,
        prefix_candidate_cd_m=prefix_candidate_cd,
        diagnostics=diagnostics,
    )


def score_recursive_gauge_rbf_prediction(
    prediction: PhysTwinRecursiveGaugeRbfPrediction,
    observed_points_m: np.ndarray,
    visibility: np.ndarray,
    manual_tracks_m: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    """Score a sealed prediction after the future target is opened."""

    baseline_metrics = official_metrics_by_frame(
        prediction.dense_baseline,
        observed_points_m,
        visibility,
        manual_tracks_m,
        num_surface_points=num_surface_points,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    candidate_metrics = official_metrics_by_frame(
        prediction.candidate,
        observed_points_m,
        visibility,
        manual_tracks_m,
        num_surface_points=num_surface_points,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    baseline_mean = {
        name: float(np.mean(value))
        for name, value in baseline_metrics.items()
    }
    candidate_mean = {
        name: float(np.mean(value))
        for name, value in candidate_metrics.items()
    }
    return {
        "baseline": baseline_mean,
        "candidate": candidate_mean,
        "candidate_relative_change_fraction": {
            name: candidate_mean[name] / baseline_mean[name] - 1.0
            for name in baseline_mean
        },
        "frame_metrics": {
            "baseline": {
                name: value.tolist()
                for name, value in baseline_metrics.items()
            },
            "candidate": {
                name: value.tolist()
                for name, value in candidate_metrics.items()
            },
        },
    }


__all__ = [
    "PhysTwinRecursiveGaugeRbfPrediction",
    "PhysTwinRecursiveGaugeRbfSourceConfig",
    "action_conditioned_rigid_transition",
    "build_dense_temporal_comparator",
    "fixed_update_frames",
    "run_recursive_gauge_rbf_source_prediction",
    "score_recursive_gauge_rbf_prediction",
    "sparse_frame_observation_belief",
    "support_adaptive_update_frames",
]
