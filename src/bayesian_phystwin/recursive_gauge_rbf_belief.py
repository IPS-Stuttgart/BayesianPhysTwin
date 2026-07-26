"""Recursive gauge-aware RBF state beliefs for deformable trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gauge_aware_belief import (
    GaugeAwareBeliefConfig,
    GaugeAwareBeliefResult,
    update_gauge_aware_belief,
)
from .observation_belief import ObservationBeliefV1
from .observation_belief_gauge_adapter import (
    ObservationBeliefGaugeAdapterResult,
    build_gauge_aware_batch_from_observation_belief,
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


def _symmetric_psd(
    values: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-10,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    _require(
        matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1],
        f"{name} must be square",
    )
    _require(np.all(np.isfinite(matrix)), f"{name} must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    _require(
        np.allclose(matrix, symmetric, atol=1e-12, rtol=1e-10),
        f"{name} must be symmetric",
    )
    eigenvalues = np.linalg.eigvalsh(symmetric)
    _require(
        np.min(eigenvalues, initial=0.0) >= -tolerance,
        f"{name} must be positive semidefinite",
    )
    return symmetric


@dataclass(frozen=True)
class RecursiveGaugeRbfConfig:
    """Spatial, process, and acceptance settings for recursive updates."""

    length_scale_fraction: float = 0.10
    local_blend: float = 0.25
    global_prior_std_m: float = 0.10
    local_prior_std_m: float = 0.02
    global_process_std_m_per_sqrt_frame: float = 0.003
    local_process_std_m_per_sqrt_frame: float = 0.003
    minimum_length_scale_m: float = 1e-4
    maximum_total_query_correction_m: float = 0.10
    covariance_eigenvalue_floor_m2: float = 1e-12
    gauge_update: GaugeAwareBeliefConfig = field(
        default_factory=GaugeAwareBeliefConfig
    )

    def __post_init__(self) -> None:
        positive = (
            self.length_scale_fraction,
            self.global_prior_std_m,
            self.local_prior_std_m,
            self.global_process_std_m_per_sqrt_frame,
            self.local_process_std_m_per_sqrt_frame,
            self.minimum_length_scale_m,
            self.maximum_total_query_correction_m,
            self.covariance_eigenvalue_floor_m2,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "recursive gauge-RBF scales must be positive",
        )
        _require(
            0.0 <= self.local_blend <= 1.0,
            "local_blend must lie in [0, 1]",
        )


@dataclass(frozen=True)
class RecursiveGaugeRbfSnapshot:
    """Immutable full-covariance posterior over global and local modes."""

    center_ids: np.ndarray
    center_positions_m: np.ndarray
    coefficient_mean_m: np.ndarray
    coefficient_covariance_m2: np.ndarray
    object_scale_m: float
    last_update_frame: int | None
    accepted_update_count: int

    def __post_init__(self) -> None:
        centers = _readonly(self.center_ids, dtype=np.int64)
        positions = _readonly(self.center_positions_m, dtype=np.float64)
        mean = _readonly(self.coefficient_mean_m, dtype=np.float64)
        covariance = _readonly(
            _symmetric_psd(
                self.coefficient_covariance_m2,
                name="coefficient_covariance_m2",
            ),
            dtype=np.float64,
        )
        _require(
            centers.ndim == 1
            and len(centers)
            and len(np.unique(centers)) == len(centers)
            and np.all(centers >= 0),
            "center_ids must be nonempty, unique, and nonnegative",
        )
        _require(
            positions.shape == (len(centers), 3)
            and np.all(np.isfinite(positions)),
            "center_positions_m must have finite shape (K, 3)",
        )
        dimension = 3 * (len(centers) + 1)
        _require(
            mean.shape == (dimension,) and np.all(np.isfinite(mean)),
            "coefficient_mean_m has changed shape",
        )
        _require(
            covariance.shape == (dimension, dimension),
            "coefficient_covariance_m2 has changed shape",
        )
        _require(
            np.isfinite(self.object_scale_m) and self.object_scale_m > 0.0,
            "object_scale_m must be positive",
        )
        _require(
            self.last_update_frame is None or self.last_update_frame >= 0,
            "last_update_frame must be nonnegative",
        )
        _require(
            self.accepted_update_count >= 0,
            "accepted_update_count must be nonnegative",
        )
        object.__setattr__(self, "center_ids", centers)
        object.__setattr__(self, "center_positions_m", positions)
        object.__setattr__(self, "coefficient_mean_m", mean)
        object.__setattr__(self, "coefficient_covariance_m2", covariance)

    @property
    def state_dimension(self) -> int:
        return len(self.coefficient_mean_m)


@dataclass(frozen=True)
class RecursiveGaugeRbfPrediction:
    """Correction moments decoded at arbitrary material queries."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray

    def __post_init__(self) -> None:
        mean = _readonly(self.mean_m, dtype=np.float64)
        covariance = _readonly(self.covariance_m2, dtype=np.float64)
        _require(
            mean.ndim == 2 and mean.shape[1] == 3,
            "prediction mean must have shape (Q, 3)",
        )
        _require(
            covariance.shape == (len(mean), 3, 3)
            and np.all(np.isfinite(covariance)),
            "prediction covariance must have shape (Q, 3, 3)",
        )
        for index, matrix in enumerate(covariance):
            _symmetric_psd(matrix, name=f"query covariance {index}")
        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "covariance_m2", covariance)


@dataclass(frozen=True)
class RecursiveGaugeRbfUpdate:
    """One causal predict-update step and its exact-fallback decision."""

    accepted: bool
    reason: str
    predicted_snapshot: RecursiveGaugeRbfSnapshot
    posterior_snapshot: RecursiveGaugeRbfSnapshot
    query_prediction: RecursiveGaugeRbfPrediction
    gauge_result: GaugeAwareBeliefResult
    adapter_summary: dict[str, object]


def _object_scale_m(
    object_positions_m: np.ndarray,
    *,
    minimum: float,
) -> float:
    positions = np.asarray(object_positions_m, dtype=np.float64)
    _require(
        positions.ndim == 2 and positions.shape[1] == 3,
        "object_positions_m must have shape (N, 3)",
    )
    finite = np.all(np.isfinite(positions), axis=1)
    _require(np.any(finite), "object_positions_m has no finite row")
    lower = np.quantile(positions[finite], 0.05, axis=0)
    upper = np.quantile(positions[finite], 0.95, axis=0)
    return max(float(np.linalg.norm(upper - lower)), minimum)


def initialize_recursive_gauge_rbf_belief(
    center_ids: np.ndarray,
    center_positions_m: np.ndarray,
    object_positions_m: np.ndarray,
    *,
    config: RecursiveGaugeRbfConfig | None = None,
) -> RecursiveGaugeRbfSnapshot:
    """Create a zero-correction prior using frame-zero geometry only."""

    cfg = config or RecursiveGaugeRbfConfig()
    centers = np.asarray(center_ids, dtype=np.int64)
    positions = np.asarray(center_positions_m, dtype=np.float64)
    _require(
        centers.ndim == 1 and len(centers),
        "center_ids must be a nonempty vector",
    )
    _require(
        positions.shape == (len(centers), 3)
        and np.all(np.isfinite(positions)),
        "center_positions_m must have finite shape (K, 3)",
    )
    dimension = 3 * (len(centers) + 1)
    variance = np.empty(dimension, dtype=np.float64)
    variance[:3] = cfg.global_prior_std_m**2
    variance[3:] = cfg.local_prior_std_m**2
    return RecursiveGaugeRbfSnapshot(
        center_ids=centers,
        center_positions_m=positions,
        coefficient_mean_m=np.zeros(dimension),
        coefficient_covariance_m2=np.diag(variance),
        object_scale_m=_object_scale_m(
            object_positions_m,
            minimum=cfg.minimum_length_scale_m,
        ),
        last_update_frame=None,
        accepted_update_count=0,
    )


def recursive_gauge_rbf_state_jacobian(
    snapshot: RecursiveGaugeRbfSnapshot,
    query_positions_m: np.ndarray,
    *,
    config: RecursiveGaugeRbfConfig | None = None,
) -> np.ndarray:
    """Map latent global/local coefficients to material-point corrections."""

    cfg = config or RecursiveGaugeRbfConfig()
    query = np.asarray(query_positions_m, dtype=np.float64)
    _require(
        query.ndim == 2
        and query.shape[1] == 3
        and np.all(np.isfinite(query)),
        "query_positions_m must have finite shape (Q, 3)",
    )
    length_scale = max(
        cfg.length_scale_fraction * snapshot.object_scale_m,
        cfg.minimum_length_scale_m,
    )
    distance = np.linalg.norm(
        query[:, None] - snapshot.center_positions_m[None],
        axis=2,
    )
    weight = np.exp(-0.5 * np.square(distance / length_scale))
    weight_sum = np.sum(weight, axis=1, keepdims=True)
    normalized = weight / np.maximum(weight_sum, 1e-15)
    normalized[weight_sum[:, 0] < 1e-12] = 0.0

    jacobian = np.zeros((len(query), 3, snapshot.state_dimension))
    identity = np.eye(3)
    jacobian[:, :, :3] = identity
    for center in range(len(snapshot.center_ids)):
        start = 3 + 3 * center
        jacobian[:, :, start : start + 3] = (
            cfg.local_blend * normalized[:, center, None, None] * identity
        )
    return jacobian


def decode_recursive_gauge_rbf_belief(
    snapshot: RecursiveGaugeRbfSnapshot,
    query_positions_m: np.ndarray,
    *,
    config: RecursiveGaugeRbfConfig | None = None,
) -> RecursiveGaugeRbfPrediction:
    """Decode full marginal correction covariance at requested points."""

    jacobian = recursive_gauge_rbf_state_jacobian(
        snapshot,
        query_positions_m,
        config=config,
    )
    mean = np.einsum(
        "qci,i->qc",
        jacobian,
        snapshot.coefficient_mean_m,
        optimize=True,
    )
    covariance = np.einsum(
        "qci,ij,qdj->qcd",
        jacobian,
        snapshot.coefficient_covariance_m2,
        jacobian,
        optimize=True,
    )
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    return RecursiveGaugeRbfPrediction(mean_m=mean, covariance_m2=covariance)


def predict_recursive_gauge_rbf_belief(
    prior: RecursiveGaugeRbfSnapshot,
    *,
    frame_index: int,
    center_positions_m: np.ndarray,
    config: RecursiveGaugeRbfConfig | None = None,
    state_transition: np.ndarray | None = None,
    process_covariance_m2: np.ndarray | None = None,
) -> RecursiveGaugeRbfSnapshot:
    """Propagate a posterior through a caller-supplied physical linearization."""

    cfg = config or RecursiveGaugeRbfConfig()
    _require(frame_index >= 0, "frame_index must be nonnegative")
    _require(
        prior.last_update_frame is None
        or frame_index > prior.last_update_frame,
        "prediction frames must increase strictly",
    )
    positions = np.asarray(center_positions_m, dtype=np.float64)
    _require(
        positions.shape == prior.center_positions_m.shape
        and np.all(np.isfinite(positions)),
        "center_positions_m changed shape or contains non-finite values",
    )
    dimension = prior.state_dimension
    transition = (
        np.eye(dimension)
        if state_transition is None
        else np.asarray(state_transition, dtype=np.float64)
    )
    _require(
        transition.shape == (dimension, dimension)
        and np.all(np.isfinite(transition)),
        "state_transition has changed shape or is non-finite",
    )
    elapsed = (
        0
        if prior.last_update_frame is None
        else frame_index - prior.last_update_frame
    )
    if process_covariance_m2 is None:
        diagonal = np.empty(dimension)
        diagonal[:3] = (
            elapsed * cfg.global_process_std_m_per_sqrt_frame**2
        )
        diagonal[3:] = (
            elapsed * cfg.local_process_std_m_per_sqrt_frame**2
        )
        process = np.diag(diagonal)
    else:
        process = _symmetric_psd(
            process_covariance_m2,
            name="process_covariance_m2",
        )
        _require(
            process.shape == (dimension, dimension),
            "process_covariance_m2 has changed shape",
        )
    covariance = (
        transition
        @ prior.coefficient_covariance_m2
        @ transition.T
        + process
    )
    covariance = 0.5 * (covariance + covariance.T)
    return RecursiveGaugeRbfSnapshot(
        center_ids=prior.center_ids,
        center_positions_m=positions,
        coefficient_mean_m=transition @ prior.coefficient_mean_m,
        coefficient_covariance_m2=covariance,
        object_scale_m=prior.object_scale_m,
        last_update_frame=frame_index,
        accepted_update_count=prior.accepted_update_count,
    )


def _regularize_covariance(
    covariance_m2: np.ndarray,
    *,
    floor_m2: float,
) -> np.ndarray:
    covariance = 0.5 * (covariance_m2 + covariance_m2.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return (
        eigenvectors * np.maximum(eigenvalues, floor_m2)
    ) @ eigenvectors.T


def update_recursive_gauge_rbf_belief(
    prior: RecursiveGaugeRbfSnapshot,
    *,
    frame_index: int,
    center_positions_m: np.ndarray,
    observation_belief: ObservationBeliefV1,
    physical_prediction_xyz_m: np.ndarray,
    query_positions_m: np.ndarray,
    physical_response_scale_m: float,
    config: RecursiveGaugeRbfConfig | None = None,
    state_transition: np.ndarray | None = None,
    process_covariance_m2: np.ndarray | None = None,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    anchor_observation_xyz_m: np.ndarray | None = None,
    anchor_physical_prediction_xyz_m: np.ndarray | None = None,
    anchor_positions_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
) -> RecursiveGaugeRbfUpdate:
    """Apply one causal gauge-aware update to the recursive RBF state."""

    cfg = config or RecursiveGaugeRbfConfig()
    predicted = predict_recursive_gauge_rbf_belief(
        prior,
        frame_index=frame_index,
        center_positions_m=center_positions_m,
        config=cfg,
        state_transition=state_transition,
        process_covariance_m2=process_covariance_m2,
    )
    physical = np.asarray(physical_prediction_xyz_m, dtype=np.float64)
    query = np.asarray(query_positions_m, dtype=np.float64)
    observation_jacobian = recursive_gauge_rbf_state_jacobian(
        predicted,
        physical,
        config=cfg,
    )
    query_jacobian = recursive_gauge_rbf_state_jacobian(
        predicted,
        query,
        config=cfg,
    )
    predicted_observation_correction = np.einsum(
        "nci,i->nc",
        observation_jacobian,
        predicted.coefficient_mean_m,
        optimize=True,
    )

    anchor_values = (
        anchor_observation_xyz_m,
        anchor_physical_prediction_xyz_m,
        anchor_positions_m,
        anchor_covariance_m2,
    )
    has_anchor = any(value is not None for value in anchor_values)
    _require(
        not has_anchor or all(value is not None for value in anchor_values),
        "all independent-anchor arrays must be supplied together",
    )
    anchor_innovation = None
    anchor_jacobian = None
    if has_anchor:
        anchor_observation = np.asarray(
            anchor_observation_xyz_m,
            dtype=np.float64,
        )
        anchor_physical = np.asarray(
            anchor_physical_prediction_xyz_m,
            dtype=np.float64,
        )
        anchor_positions = np.asarray(anchor_positions_m, dtype=np.float64)
        _require(
            anchor_observation.shape
            == anchor_physical.shape
            == anchor_positions.shape,
            "independent-anchor point arrays must share shape",
        )
        anchor_jacobian = recursive_gauge_rbf_state_jacobian(
            predicted,
            anchor_positions,
            config=cfg,
        )
        anchor_prior_correction = np.einsum(
            "aci,i->ac",
            anchor_jacobian,
            predicted.coefficient_mean_m,
            optimize=True,
        )
        anchor_innovation = (
            anchor_observation - anchor_physical - anchor_prior_correction
        )

    adapted: ObservationBeliefGaugeAdapterResult = (
        build_gauge_aware_batch_from_observation_belief(
            observation_belief,
            physical_prediction_xyz_m=(
                physical + predicted_observation_correction
            ),
            state_jacobian=observation_jacobian,
            query_state_jacobian=query_jacobian,
            physical_response_scale_m=physical_response_scale_m,
            shared_bias_jacobian=shared_bias_jacobian,
            view_bias_jacobian=view_bias_jacobian,
            state_prior_covariance_m2=(
                predicted.coefficient_covariance_m2
            ),
            anchor_innovation_m=anchor_innovation,
            anchor_covariance_m2=anchor_covariance_m2,
            anchor_state_jacobian=anchor_jacobian,
        )
    )
    gauge_result = update_gauge_aware_belief(
        adapted.batch,
        config=cfg.gauge_update,
    )
    posterior = predicted
    accepted = gauge_result.accepted
    reason = gauge_result.reason
    if accepted:
        state_dimension = predicted.state_dimension
        posterior_covariance = _regularize_covariance(
            gauge_result.posterior_covariance[
                :state_dimension,
                :state_dimension,
            ],
            floor_m2=cfg.covariance_eigenvalue_floor_m2,
        )
        candidate = RecursiveGaugeRbfSnapshot(
            center_ids=predicted.center_ids,
            center_positions_m=predicted.center_positions_m,
            coefficient_mean_m=(
                predicted.coefficient_mean_m
                + gauge_result.state_coefficients
            ),
            coefficient_covariance_m2=posterior_covariance,
            object_scale_m=predicted.object_scale_m,
            last_update_frame=frame_index,
            accepted_update_count=predicted.accepted_update_count + 1,
        )
        candidate_query = decode_recursive_gauge_rbf_belief(
            candidate,
            query,
            config=cfg,
        )
        maximum = float(
            np.max(np.linalg.norm(candidate_query.mean_m, axis=1), initial=0.0)
        )
        if maximum <= cfg.maximum_total_query_correction_m:
            posterior = candidate
        else:
            accepted = False
            reason = "total-query-correction-limit"

    query_prediction = decode_recursive_gauge_rbf_belief(
        posterior,
        query,
        config=cfg,
    )
    return RecursiveGaugeRbfUpdate(
        accepted=accepted,
        reason=reason,
        predicted_snapshot=predicted,
        posterior_snapshot=posterior,
        query_prediction=query_prediction,
        gauge_result=gauge_result,
        adapter_summary=adapted.summary(),
    )


def select_recursive_gauge_rbf_candidate(
    baseline: np.ndarray,
    update: RecursiveGaugeRbfUpdate,
) -> np.ndarray:
    """Apply an accepted correction or preserve the baseline byte for byte."""

    baseline_input = np.asarray(baseline)
    _require(
        baseline_input.shape == update.query_prediction.mean_m.shape,
        "baseline shape differs from query prediction",
    )
    if not update.accepted:
        selected = baseline_input.copy()
        if selected.tobytes() != baseline_input.tobytes():
            raise AssertionError("recursive gauge-RBF fallback changed bytes")
        return selected
    return (
        baseline_input + update.query_prediction.mean_m
    ).astype(baseline_input.dtype, copy=False)


__all__ = [
    "RecursiveGaugeRbfConfig",
    "RecursiveGaugeRbfPrediction",
    "RecursiveGaugeRbfSnapshot",
    "RecursiveGaugeRbfUpdate",
    "decode_recursive_gauge_rbf_belief",
    "initialize_recursive_gauge_rbf_belief",
    "predict_recursive_gauge_rbf_belief",
    "recursive_gauge_rbf_state_jacobian",
    "select_recursive_gauge_rbf_candidate",
    "update_recursive_gauge_rbf_belief",
]
