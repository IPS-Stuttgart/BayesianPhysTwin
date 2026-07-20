"""Bias-aware Bayesian state updates with certified exact fallback.

Camera innovations are decomposed into physically admissible state modes,
shared spatial bias modes, and per-camera offsets. Independent anchors may
observe the state modes without the camera-bias terms. The update is linear on
purpose: it makes state/bias confounding measurable before a nonlinear digital
twin consumes the posterior correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2, f"{name} must be a matrix")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


def _positive_definite_inverse(value: np.ndarray, name: str) -> np.ndarray:
    matrix = _finite_matrix(value, name)
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    _require(np.allclose(matrix, matrix.T), f"{name} must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return np.linalg.inv(matrix)


@dataclass(frozen=True)
class BiasAwareStateUpdateConfig:
    """Numerical and prior settings for one robust bias-aware update."""

    observation_std_m: float = 0.005
    anchor_std_m: float = 0.002
    state_prior_std_m: float = 0.020
    shared_bias_prior_std_m: float = 0.020
    camera_bias_prior_std_m: float = 0.010
    effective_samples_per_view: float = 64.0
    degrees_of_freedom: float = 4.0
    minimum_robust_weight: float = 0.02
    maximum_iterations: int = 8
    convergence_tolerance: float = 1e-9
    maximum_condition_number: float = 1e12
    maximum_state_update_m: float = 0.10
    ambiguous_subspace_cosine: float = 0.999
    reject_unanchored_ambiguity: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.observation_std_m,
            self.anchor_std_m,
            self.state_prior_std_m,
            self.shared_bias_prior_std_m,
            self.camera_bias_prior_std_m,
            self.effective_samples_per_view,
            self.degrees_of_freedom,
            self.convergence_tolerance,
            self.maximum_condition_number,
            self.maximum_state_update_m,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "bias-aware update scales must be positive",
        )
        _require(
            0.0 < self.minimum_robust_weight <= 1.0,
            "minimum robust weight must lie in (0, 1]",
        )
        _require(self.maximum_iterations >= 1, "maximum iterations must be positive")
        _require(
            0.0 <= self.ambiguous_subspace_cosine <= 1.0,
            "ambiguous subspace cosine must lie in [0, 1]",
        )


@dataclass(frozen=True)
class BiasAwareStateUpdateResult:
    """Posterior coefficients and diagnostics from one state/bias update."""

    accepted: bool
    reason: str
    state_coefficients_m: np.ndarray
    shared_bias_coefficients_m: np.ndarray
    camera_biases_m: np.ndarray
    posterior_covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    robust_weights: np.ndarray
    anchor_robust_weights: np.ndarray
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        arrays = {
            "state_coefficients_m": np.asarray(
                self.state_coefficients_m, dtype=np.float64
            ).copy(),
            "shared_bias_coefficients_m": np.asarray(
                self.shared_bias_coefficients_m, dtype=np.float64
            ).copy(),
            "camera_biases_m": np.asarray(
                self.camera_biases_m, dtype=np.float64
            ).copy(),
            "posterior_covariance_m2": np.asarray(
                self.posterior_covariance_m2, dtype=np.float64
            ).copy(),
            "prior_reliability": np.asarray(
                self.prior_reliability, dtype=np.float64
            ).copy(),
            "robust_weights": np.asarray(
                self.robust_weights, dtype=np.float64
            ).copy(),
            "anchor_robust_weights": np.asarray(
                self.anchor_robust_weights, dtype=np.float64
            ).copy(),
        }
        for name, value in arrays.items():
            _require(np.all(np.isfinite(value)), f"{name} contains non-finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        state = arrays["state_coefficients_m"]
        shared = arrays["shared_bias_coefficients_m"]
        camera = arrays["camera_biases_m"]
        covariance = arrays["posterior_covariance_m2"]
        reliability = arrays["prior_reliability"]
        robust = arrays["robust_weights"]
        anchor_robust = arrays["anchor_robust_weights"]
        _require(state.ndim == 2 and state.shape[1] == 3, "state shape changed")
        _require(
            shared.ndim == 2 and shared.shape[1] == 3,
            "shared bias shape changed",
        )
        _require(
            camera.ndim == 2 and camera.shape[1] == 3,
            "camera bias shape changed",
        )
        dimension = len(state) + len(shared) + len(camera)
        _require(
            covariance.shape == (dimension, dimension),
            "posterior covariance shape changed",
        )
        _require(reliability.ndim == 2, "prior reliability must be a matrix")
        _require(robust.shape == reliability.shape, "robust weight shape changed")
        _require(anchor_robust.ndim == 1, "anchor robust weights must be a vector")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class PhysicalResponseBasis:
    """Low-rank state support derived only from a causal physical response."""

    basis: np.ndarray
    singular_values_m: np.ndarray
    explained_energy_fraction: float
    supported_point_count: int
    maximum_response_m: float

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=np.float64).copy()
        singular_values = np.asarray(
            self.singular_values_m, dtype=np.float64
        ).copy()
        _require(basis.ndim == 2 and basis.shape[1] >= 1, "basis is empty")
        _require(
            singular_values.shape == (basis.shape[1],),
            "singular value count changed",
        )
        _require(
            np.all(np.isfinite(basis)) and np.all(np.isfinite(singular_values)),
            "physical response basis contains non-finite values",
        )
        _require(np.all(singular_values > 0.0), "singular values must be positive")
        _require(
            0.0 < self.explained_energy_fraction <= 1.0,
            "explained energy must lie in (0, 1]",
        )
        _require(self.supported_point_count >= 1, "physical support is empty")
        _require(
            np.isfinite(self.maximum_response_m) and self.maximum_response_m > 0.0,
            "physical response magnitude must be positive",
        )
        basis.setflags(write=False)
        singular_values.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "singular_values_m", singular_values)


@dataclass(frozen=True)
class IdentifiableStateBasis:
    """Reachable state combinations that observation bias cannot reproduce."""

    query_basis: np.ndarray
    observation_basis: np.ndarray
    coefficient_transform: np.ndarray
    identifiable_fractions: np.ndarray

    def __post_init__(self) -> None:
        query = np.asarray(self.query_basis, dtype=np.float64).copy()
        observation = np.asarray(self.observation_basis, dtype=np.float64).copy()
        transform = np.asarray(
            self.coefficient_transform, dtype=np.float64
        ).copy()
        fractions = np.asarray(
            self.identifiable_fractions, dtype=np.float64
        ).copy()
        _require(query.ndim == 2 and query.shape[1] >= 1, "query basis is empty")
        retained = query.shape[1]
        _require(
            observation.ndim == 2 and observation.shape[1] == retained,
            "observation basis shape changed",
        )
        _require(
            transform.ndim == 2 and transform.shape[1] == retained,
            "coefficient transform shape changed",
        )
        _require(
            fractions.shape == (retained,),
            "identifiable fraction count changed",
        )
        _require(
            np.all(np.isfinite(query))
            and np.all(np.isfinite(observation))
            and np.all(np.isfinite(transform))
            and np.all(np.isfinite(fractions)),
            "identifiable state basis contains non-finite values",
        )
        _require(
            np.all((fractions > 0.0) & (fractions <= 1.0 + 1e-12)),
            "identifiable fractions must lie in (0, 1]",
        )
        for name, value in (
            ("query_basis", query),
            ("observation_basis", observation),
            ("coefficient_transform", transform),
            ("identifiable_fractions", fractions),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def build_physical_response_basis(
    causal_physical_response_m: np.ndarray,
    *,
    action_support: np.ndarray | None = None,
    rank: int = 4,
    minimum_response_m: float = 1e-5,
    minimum_singular_fraction: float = 1e-6,
) -> PhysicalResponseBasis:
    """Build state modes from prefix-only simulated response and action support.

    The response may contain one ``(N, 3)`` displacement or a causal history
    ``(T, N, 3)``. Spatial modes are the left singular vectors of the supported
    response matrix. Each mode is deterministically signed and normalized to
    unit maximum amplitude, so its coefficient is measured in metres.
    """

    response = np.asarray(causal_physical_response_m, dtype=np.float64)
    if response.ndim == 2:
        response = response[None]
    _require(
        response.ndim == 3 and response.shape[2] == 3,
        "physical response must have shape (T, N, 3) or (N, 3)",
    )
    _require(np.all(np.isfinite(response)), "physical response is not finite")
    _require(rank >= 1, "physical response rank must be positive")
    _require(
        np.isfinite(minimum_response_m) and minimum_response_m > 0.0,
        "minimum physical response must be positive",
    )
    _require(
        np.isfinite(minimum_singular_fraction)
        and 0.0 < minimum_singular_fraction <= 1.0,
        "minimum singular fraction must lie in (0, 1]",
    )
    point_count = response.shape[1]
    if action_support is None:
        support = np.ones(point_count, dtype=np.float64)
    else:
        support = np.asarray(action_support, dtype=np.float64)
        _require(support.shape == (point_count,), "action support shape changed")
        _require(
            np.all(np.isfinite(support))
            and np.all((support >= 0.0) & (support <= 1.0)),
            "action support must lie in [0, 1]",
        )
    supported_point_count = int(np.sum(support > 0.0))
    _require(supported_point_count >= 1, "action support is empty")
    supported_response = response * support[None, :, None]
    maximum_response = float(
        np.max(np.linalg.norm(supported_response, axis=2))
    )
    _require(
        maximum_response >= minimum_response_m,
        "causal physical response is below the declared support threshold",
    )
    response_matrix = np.transpose(supported_response, (1, 0, 2)).reshape(
        point_count, -1
    )
    left, singular_values, _ = np.linalg.svd(response_matrix, full_matrices=False)
    retained_count = min(
        rank,
        int(
            np.sum(
                singular_values
                >= singular_values[0] * minimum_singular_fraction
            )
        ),
    )
    _require(retained_count >= 1, "physical response has no retained state mode")
    basis = left[:, :retained_count].copy()
    for mode in range(retained_count):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0.0:
            basis[:, mode] *= -1.0
        basis[:, mode] /= np.max(np.abs(basis[:, mode]))
    energy = np.square(singular_values)
    explained = float(np.sum(energy[:retained_count]) / np.sum(energy))
    return PhysicalResponseBasis(
        basis=basis,
        singular_values_m=singular_values[:retained_count],
        explained_energy_fraction=explained,
        supported_point_count=supported_point_count,
        maximum_response_m=maximum_response,
    )


def restrict_state_basis_to_identifiable_subspace(
    query_state_basis: np.ndarray,
    observation_state_basis: np.ndarray,
    observation_bias_design: np.ndarray,
    *,
    minimum_identifiable_fraction: float = 0.10,
) -> IdentifiableStateBasis:
    """Retain physically reachable coefficient directions exposed beyond bias.

    Projection is performed only in observation space. Right singular vectors
    of the projected state design define linear combinations of the original
    physical modes, so every retained query-space mode remains in the supplied
    reachable state span.
    """

    query = _finite_matrix(query_state_basis, "query state basis")
    observation = _finite_matrix(
        observation_state_basis, "observation state basis"
    )
    bias = _finite_matrix(observation_bias_design, "observation bias design")
    _require(query.shape[1] >= 1, "state basis is empty")
    _require(
        observation.shape[1] == query.shape[1],
        "query and observation state modes differ",
    )
    _require(
        bias.shape[0] == observation.shape[0],
        "observation and bias row counts differ",
    )
    _require(
        np.isfinite(minimum_identifiable_fraction)
        and 0.0 < minimum_identifiable_fraction <= 1.0,
        "minimum identifiable fraction must lie in (0, 1]",
    )
    bias_space = (
        np.zeros((len(bias), 0), dtype=np.float64)
        if bias.shape[1] == 0
        else _orthonormal_column_space(bias)
    )
    projected = observation.copy()
    if bias_space.shape[1]:
        projected -= bias_space @ (bias_space.T @ observation)
    projection_floor = (
        max(projected.shape)
        * np.finfo(np.float64).eps
        * max(1.0, float(np.linalg.norm(observation)))
    )
    if np.linalg.norm(projected) <= projection_floor:
        raise ValueError("state response is fully confounded with observation bias")
    _, singular_values, right_transpose = np.linalg.svd(
        projected, full_matrices=False
    )
    if not len(singular_values) or singular_values[0] == 0.0:
        raise ValueError("state response is fully confounded with observation bias")
    tolerance = (
        max(projected.shape)
        * np.finfo(np.float64).eps
        * singular_values[0]
    )
    candidate_count = int(np.sum(singular_values > tolerance))
    transforms = right_transpose[:candidate_count].T
    retained_transforms: list[np.ndarray] = []
    fractions: list[float] = []
    for candidate in range(candidate_count):
        transform = transforms[:, candidate]
        total_norm = float(np.linalg.norm(observation @ transform))
        if total_norm <= tolerance:
            continue
        identifiable_fraction = float(
            np.linalg.norm(projected @ transform) / total_norm
        )
        if identifiable_fraction >= minimum_identifiable_fraction:
            retained_transforms.append(transform)
            fractions.append(min(1.0, identifiable_fraction))
    if not retained_transforms:
        raise ValueError("no physical state mode is identifiable beyond bias")
    coefficient_transform = np.column_stack(retained_transforms)
    retained_query = query @ coefficient_transform
    retained_observation = observation @ coefficient_transform
    for mode in range(retained_query.shape[1]):
        scale = float(np.max(np.abs(retained_query[:, mode])))
        if scale <= tolerance:
            raise ValueError("identifiable state mode vanishes in query space")
        retained_query[:, mode] /= scale
        retained_observation[:, mode] /= scale
        coefficient_transform[:, mode] /= scale
        pivot = int(np.argmax(np.abs(retained_query[:, mode])))
        if retained_query[pivot, mode] < 0.0:
            retained_query[:, mode] *= -1.0
            retained_observation[:, mode] *= -1.0
            coefficient_transform[:, mode] *= -1.0
    return IdentifiableStateBasis(
        query_basis=retained_query,
        observation_basis=retained_observation,
        coefficient_transform=coefficient_transform,
        identifiable_fractions=np.asarray(fractions),
    )


def _orthonormal_column_space(value: np.ndarray) -> np.ndarray:
    left, singular_values, _ = np.linalg.svd(value, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.zeros((len(value), 0), dtype=np.float64)
    tolerance = max(value.shape) * np.finfo(np.float64).eps * singular_values[0]
    return left[:, singular_values > tolerance]


def _subspace_overlap(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape[1] == 0 or second.shape[1] == 0 or len(first) == 0:
        return 0.0
    first_q = _orthonormal_column_space(first)
    second_q = _orthonormal_column_space(second)
    if first_q.shape[1] == 0 or second_q.shape[1] == 0:
        return 0.0
    return float(np.linalg.svd(first_q.T @ second_q, compute_uv=False)[0])


def _student_t_weight(
    residual_m: np.ndarray,
    variance_m2: np.ndarray,
    degrees_of_freedom: float,
    minimum: float,
) -> np.ndarray:
    squared_radius = np.sum(np.square(residual_m), axis=1) / variance_m2
    weight = (degrees_of_freedom + residual_m.shape[1]) / (
        degrees_of_freedom + squared_radius
    )
    return np.clip(weight, minimum, 1.0)


def _fallback_result(
    state_count: int,
    shared_bias_count: int,
    view_count: int,
    prior_reliability: np.ndarray,
    reason: str,
    diagnostics: dict[str, object],
    anchor_count: int = 0,
) -> BiasAwareStateUpdateResult:
    dimension = state_count + shared_bias_count + view_count
    return BiasAwareStateUpdateResult(
        accepted=False,
        reason=reason,
        state_coefficients_m=np.zeros((state_count, 3), dtype=np.float64),
        shared_bias_coefficients_m=np.zeros(
            (shared_bias_count, 3), dtype=np.float64
        ),
        camera_biases_m=np.zeros((view_count, 3), dtype=np.float64),
        posterior_covariance_m2=np.zeros((dimension, dimension), dtype=np.float64),
        prior_reliability=prior_reliability,
        robust_weights=np.zeros_like(prior_reliability),
        anchor_robust_weights=np.zeros(anchor_count, dtype=np.float64),
        diagnostics=diagnostics,
    )


def update_bias_aware_state(
    camera_innovation_m: np.ndarray,
    camera_available: np.ndarray,
    state_basis: np.ndarray,
    shared_bias_basis: np.ndarray,
    *,
    prior_reliability: np.ndarray | None = None,
    observation_variance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_state_basis: np.ndarray | None = None,
    anchor_variance_m2: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    config: BiasAwareStateUpdateConfig | None = None,
) -> BiasAwareStateUpdateResult:
    """Infer physical state modes separately from shared and per-camera bias.

    ``state_basis`` and ``shared_bias_basis`` are defined over the observed
    material points. The caller owns their physical meaning. An action-local
    graph basis, for example, makes unsupported camera motion unavailable to
    the state posterior. Anchor observations see only the state coefficients
    and therefore break common-mode camera ambiguity.
    """

    cfg = config or BiasAwareStateUpdateConfig()
    innovation = np.asarray(camera_innovation_m, dtype=np.float64)
    available = np.asarray(camera_available, dtype=bool)
    _require(
        innovation.ndim == 3 and innovation.shape[2] == 3,
        "camera innovation must have shape (V, N, 3)",
    )
    view_count, point_count, _ = innovation.shape
    _require(available.shape == (view_count, point_count), "camera mask changed")
    state = _finite_matrix(state_basis, "state basis")
    shared_bias = _finite_matrix(shared_bias_basis, "shared bias basis")
    _require(state.shape[0] == point_count, "state basis point count changed")
    _require(
        shared_bias.shape[0] == point_count,
        "shared bias basis point count changed",
    )
    state_count = state.shape[1]
    shared_bias_count = shared_bias.shape[1]
    _require(state_count >= 1, "state basis is empty")

    if prior_reliability is None:
        reliability = np.ones((view_count, point_count), dtype=np.float64)
    else:
        reliability = np.asarray(prior_reliability, dtype=np.float64).copy()
        _require(reliability.shape == available.shape, "prior reliability changed")
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "prior reliability must lie in [0, 1]",
        )
    reliability[~available] = 0.0
    finite_camera = np.all(np.isfinite(innovation), axis=2)
    reliability[~finite_camera] = 0.0
    usable = available & finite_camera & (reliability > 0.0)

    if observation_variance_m2 is None:
        camera_variance = np.full(
            available.shape, cfg.observation_std_m**2, dtype=np.float64
        )
    else:
        camera_variance = np.asarray(
            observation_variance_m2, dtype=np.float64
        ).copy()
        _require(camera_variance.shape == available.shape, "camera variance changed")
        _require(
            np.all(np.isfinite(camera_variance)) and np.all(camera_variance > 0.0),
            "camera variance must be positive",
        )

    active_views = np.flatnonzero(np.any(usable, axis=1))
    diagnostics: dict[str, object] = {
        "view_count": view_count,
        "active_view_count": len(active_views),
        "usable_camera_observation_count": int(np.sum(usable)),
        "state_mode_count": state_count,
        "shared_bias_mode_count": shared_bias_count,
        "correlation_treatment": (
            "effective samples within view and equal-weight covariance "
            "intersection across views"
        ),
        "prior_reliability_uses_innovation": False,
    }
    camera_rows = np.argwhere(usable)
    camera_design = np.zeros(
        (len(camera_rows), state_count + shared_bias_count + view_count),
        dtype=np.float64,
    )
    camera_target = np.empty((len(camera_rows), 3), dtype=np.float64)
    camera_row_variance = np.empty(len(camera_rows), dtype=np.float64)
    camera_base_weight = np.empty(len(camera_rows), dtype=np.float64)
    for row_index, (view_index, point_index) in enumerate(camera_rows):
        camera_design[row_index, :state_count] = state[point_index]
        camera_design[
            row_index, state_count : state_count + shared_bias_count
        ] = shared_bias[point_index]
        camera_design[
            row_index, state_count + shared_bias_count + view_index
        ] = 1.0
        camera_target[row_index] = innovation[view_index, point_index]
        camera_row_variance[row_index] = camera_variance[view_index, point_index]
        count = int(np.sum(usable[view_index]))
        within_view = min(cfg.effective_samples_per_view, float(count)) / count
        camera_base_weight[row_index] = (
            reliability[view_index, point_index]
            * within_view
            / len(active_views)
        )

    has_anchor = anchor_innovation_m is not None
    if has_anchor:
        _require(anchor_state_basis is not None, "anchor state basis is missing")
        anchor_target = np.asarray(anchor_innovation_m, dtype=np.float64)
        anchor_design_state = _finite_matrix(anchor_state_basis, "anchor state basis")
        _require(
            anchor_target.ndim == 2 and anchor_target.shape[1] == 3,
            "anchor innovation must have shape (A, 3)",
        )
        _require(
            anchor_design_state.shape == (len(anchor_target), state_count),
            "anchor state basis changed",
        )
        anchor_finite = np.all(np.isfinite(anchor_target), axis=1)
        anchor_target = anchor_target[anchor_finite]
        anchor_design_state = anchor_design_state[anchor_finite]
        anchor_design = np.zeros(
            (len(anchor_target), camera_design.shape[1]), dtype=np.float64
        )
        anchor_design[:, :state_count] = anchor_design_state
        if anchor_variance_m2 is None:
            anchor_variance = np.full(
                len(anchor_target), cfg.anchor_std_m**2, dtype=np.float64
            )
        else:
            supplied_anchor_variance = np.asarray(
                anchor_variance_m2, dtype=np.float64
            )
            _require(
                supplied_anchor_variance.shape == (len(anchor_finite),),
                "anchor variance changed",
            )
            anchor_variance = supplied_anchor_variance[anchor_finite]
            _require(
                np.all(np.isfinite(anchor_variance))
                and np.all(anchor_variance > 0.0),
                "anchor variance must be positive",
            )
    else:
        _require(anchor_state_basis is None, "anchor innovation is missing")
        _require(anchor_variance_m2 is None, "anchor innovation is missing")
        anchor_target = np.zeros((0, 3), dtype=np.float64)
        anchor_design = np.zeros((0, camera_design.shape[1]), dtype=np.float64)
        anchor_variance = np.zeros(0, dtype=np.float64)

    if not len(camera_rows) and not len(anchor_target):
        return _fallback_result(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            "no-observation-support",
            diagnostics,
        )

    if state_prior_covariance_m2 is None:
        state_precision = np.eye(state_count) / cfg.state_prior_std_m**2
    else:
        supplied = np.asarray(state_prior_covariance_m2, dtype=np.float64)
        _require(
            supplied.shape == (state_count, state_count),
            "state prior covariance changed",
        )
        state_precision = _positive_definite_inverse(
            supplied, "state prior covariance"
        )
    dimension = camera_design.shape[1]
    prior_precision = np.zeros((dimension, dimension), dtype=np.float64)
    prior_precision[:state_count, :state_count] = state_precision
    shared_slice = slice(state_count, state_count + shared_bias_count)
    prior_precision[shared_slice, shared_slice] = (
        np.eye(shared_bias_count) / cfg.shared_bias_prior_std_m**2
    )
    camera_slice = slice(state_count + shared_bias_count, dimension)
    camera_bias_precision = np.full(
        view_count,
        1.0 / cfg.camera_bias_prior_std_m**2,
        dtype=np.float64,
    )
    if len(active_views):
        camera_bias_precision[active_views] /= len(active_views)
    prior_precision[camera_slice, camera_slice] = np.diag(camera_bias_precision)
    diagnostics["per_camera_bias_prior_ci_scaled"] = True

    camera_bias_design = camera_design[:, state_count:]
    camera_overlap_weight = np.sqrt(
        camera_base_weight / camera_row_variance
    )[:, None]
    overlap = _subspace_overlap(
        camera_overlap_weight * camera_design[:, :state_count],
        camera_overlap_weight * camera_bias_design,
    )
    diagnostics["state_bias_subspace_cosine"] = overlap
    diagnostics["independent_anchor_count"] = len(anchor_target)
    if (
        cfg.reject_unanchored_ambiguity
        and not len(anchor_target)
        and overlap >= cfg.ambiguous_subspace_cosine
    ):
        return _fallback_result(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            "unanchored-common-mode-ambiguity",
            diagnostics,
            anchor_count=len(anchor_target),
        )

    solution = np.zeros((dimension, 3), dtype=np.float64)
    camera_robust = np.ones(len(camera_rows), dtype=np.float64)
    anchor_robust = np.ones(len(anchor_target), dtype=np.float64)

    def posterior_system() -> tuple[np.ndarray, np.ndarray]:
        camera_precision_weight = (
            camera_base_weight * camera_robust / camera_row_variance
        )
        posterior_normal = prior_precision + camera_design.T @ (
            camera_precision_weight[:, None] * camera_design
        )
        posterior_right = camera_design.T @ (
            camera_precision_weight[:, None] * camera_target
        )
        if len(anchor_target):
            anchor_precision_weight = anchor_robust / anchor_variance
            posterior_normal += anchor_design.T @ (
                anchor_precision_weight[:, None] * anchor_design
            )
            posterior_right += anchor_design.T @ (
                anchor_precision_weight[:, None] * anchor_target
            )
        return posterior_normal, posterior_right

    normal = prior_precision.copy()
    for iteration in range(cfg.maximum_iterations):
        previous = solution.copy()
        normal, right = posterior_system()
        condition_number = float(np.linalg.cond(normal))
        if not np.isfinite(condition_number) or condition_number > (
            cfg.maximum_condition_number
        ):
            diagnostics["condition_number"] = condition_number
            return _fallback_result(
                state_count,
                shared_bias_count,
                view_count,
                reliability,
                "ill-conditioned-posterior",
                diagnostics,
                anchor_count=len(anchor_target),
            )
        try:
            solution = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            return _fallback_result(
                state_count,
                shared_bias_count,
                view_count,
                reliability,
                "singular-posterior",
                diagnostics,
                anchor_count=len(anchor_target),
            )
        camera_residual = camera_target - camera_design @ solution
        camera_robust = _student_t_weight(
            camera_residual,
            camera_row_variance,
            cfg.degrees_of_freedom,
            cfg.minimum_robust_weight,
        )
        if len(anchor_target):
            anchor_residual = anchor_target - anchor_design @ solution
            anchor_robust = _student_t_weight(
                anchor_residual,
                anchor_variance,
                cfg.degrees_of_freedom,
                cfg.minimum_robust_weight,
            )
        if np.linalg.norm(solution - previous) <= cfg.convergence_tolerance:
            break

    normal, right = posterior_system()
    solution = np.linalg.solve(normal, right)

    state_coefficients = solution[:state_count]
    state_update = state @ state_coefficients
    maximum_update = float(np.max(np.linalg.norm(state_update, axis=1)))
    diagnostics.update(
        {
            "iterations": iteration + 1,
            "condition_number": float(np.linalg.cond(normal)),
            "effective_camera_information_mass": float(
                np.sum(camera_base_weight)
            ),
            "minimum_camera_robust_weight": (
                float(np.min(camera_robust)) if len(camera_robust) else 1.0
            ),
            "downweighted_camera_fraction": (
                float(np.mean(camera_robust < 1.0)) if len(camera_robust) else 0.0
            ),
            "maximum_state_update_m": maximum_update,
        }
    )
    if not np.all(np.isfinite(solution)) or maximum_update > cfg.maximum_state_update_m:
        return _fallback_result(
            state_count,
            shared_bias_count,
            view_count,
            reliability,
            "implausible-state-update",
            diagnostics,
            anchor_count=len(anchor_target),
        )

    covariance = np.linalg.inv(normal)
    bias_indices = np.arange(state_count, dimension)
    state_indices = np.arange(state_count)
    if len(bias_indices):
        cross = covariance[np.ix_(state_indices, bias_indices)]
        scale = np.sqrt(
            covariance[state_indices, state_indices][:, None]
            * covariance[bias_indices, bias_indices][None, :]
        )
        maximum_cross_correlation = float(
            np.max(np.abs(cross / np.maximum(scale, 1e-15)))
        )
    else:
        maximum_cross_correlation = 0.0
    diagnostics["maximum_state_bias_posterior_correlation"] = (
        maximum_cross_correlation
    )

    robust_map = np.zeros((view_count, point_count), dtype=np.float64)
    robust_map[camera_rows[:, 0], camera_rows[:, 1]] = camera_robust
    return BiasAwareStateUpdateResult(
        accepted=True,
        reason="accepted",
        state_coefficients_m=state_coefficients,
        shared_bias_coefficients_m=solution[shared_slice],
        camera_biases_m=solution[camera_slice],
        posterior_covariance_m2=covariance,
        prior_reliability=reliability,
        robust_weights=robust_map,
        anchor_robust_weights=anchor_robust,
        diagnostics=diagnostics,
    )


def decode_bias_aware_state(
    result: BiasAwareStateUpdateResult,
    query_state_basis: np.ndarray,
) -> np.ndarray:
    """Decode state coefficients at arbitrary graph or material points."""

    basis = _finite_matrix(query_state_basis, "query state basis")
    _require(
        basis.shape[1] == len(result.state_coefficients_m),
        "query state basis mode count changed",
    )
    if not result.accepted:
        return np.zeros((len(basis), 3), dtype=np.float64)
    return basis @ result.state_coefficients_m


@dataclass(frozen=True)
class SourceRegretCertificate:
    """Group-cross-fitted upper bound on candidate regret."""

    feature_center: np.ndarray
    feature_scale: np.ndarray
    standardized_feature_lower: np.ndarray
    standardized_feature_upper: np.ndarray
    coefficients: np.ndarray
    upper_residual_quantile: float
    nominal_coverage: float
    minimum_improvement: float
    ridge_penalty: float
    support_margin_std: float
    source_group_count: int
    finite_sample_rank: int
    finite_sample_coverage: float

    def __post_init__(self) -> None:
        center = np.asarray(self.feature_center, dtype=np.float64).copy()
        scale = np.asarray(self.feature_scale, dtype=np.float64).copy()
        lower = np.asarray(
            self.standardized_feature_lower, dtype=np.float64
        ).copy()
        upper = np.asarray(
            self.standardized_feature_upper, dtype=np.float64
        ).copy()
        coefficients = np.asarray(self.coefficients, dtype=np.float64).copy()
        _require(center.ndim == 1, "feature center must be a vector")
        _require(scale.shape == center.shape, "feature scale changed")
        _require(lower.shape == center.shape, "feature lower bound changed")
        _require(upper.shape == center.shape, "feature upper bound changed")
        _require(
            coefficients.shape == (len(center) + 1,),
            "regret coefficients changed",
        )
        _require(
            np.all(np.isfinite(center))
            and np.all(np.isfinite(scale))
            and np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(np.isfinite(coefficients)),
            "regret certificate contains non-finite values",
        )
        _require(np.all(scale > 0.0), "feature scales must be positive")
        _require(np.all(lower <= upper), "feature support bounds are invalid")
        _require(
            np.isfinite(self.upper_residual_quantile),
            "residual quantile must be finite",
        )
        _require(
            0.0 < self.nominal_coverage < 1.0,
            "nominal coverage must lie in (0, 1)",
        )
        _require(self.minimum_improvement >= 0.0, "minimum improvement is negative")
        _require(self.ridge_penalty >= 0.0, "ridge penalty is negative")
        _require(self.support_margin_std >= 0.0, "support margin is negative")
        _require(self.source_group_count >= 3, "too few source groups")
        _require(
            1 <= self.finite_sample_rank <= self.source_group_count,
            "finite-sample rank is invalid",
        )
        _require(
            self.finite_sample_coverage
            == self.finite_sample_rank / (self.source_group_count + 1),
            "finite-sample coverage arithmetic changed",
        )
        for name, value in (
            ("feature_center", center),
            ("feature_scale", scale),
            ("standardized_feature_lower", lower),
            ("standardized_feature_upper", upper),
            ("coefficients", coefficients),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def predict_regret(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype=np.float64)
        _require(vector.shape == self.feature_center.shape, "feature shape changed")
        _require(np.all(np.isfinite(vector)), "features contain non-finite values")
        standardized = (vector - self.feature_center) / self.feature_scale
        return float(self.coefficients[0] + standardized @ self.coefficients[1:])

    def upper_regret(self, features: np.ndarray) -> float:
        if not self.in_source_support(features):
            return float("inf")
        return self.predict_regret(features) + self.upper_residual_quantile

    def in_source_support(self, features: np.ndarray) -> bool:
        vector = np.asarray(features, dtype=np.float64)
        _require(vector.shape == self.feature_center.shape, "feature shape changed")
        _require(np.all(np.isfinite(vector)), "features contain non-finite values")
        standardized = (vector - self.feature_center) / self.feature_scale
        return bool(
            np.all(
                standardized
                >= self.standardized_feature_lower - self.support_margin_std
            )
            and np.all(
                standardized
                <= self.standardized_feature_upper + self.support_margin_std
            )
        )


def _fit_group_weighted_ridge(
    features: np.ndarray,
    regret: np.ndarray,
    groups: np.ndarray,
    ridge_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, counts = np.unique(groups, return_counts=True)
    count_by_group = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.asarray([1.0 / count_by_group[value] for value in groups])
    weights *= len(weights) / np.sum(weights)
    center = np.sum(weights[:, None] * features, axis=0) / np.sum(weights)
    variance = np.sum(
        weights[:, None] * np.square(features - center), axis=0
    ) / np.sum(weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    standardized = (features - center) / scale
    design = np.column_stack((np.ones(len(features)), standardized))
    penalty = np.eye(design.shape[1]) * ridge_penalty
    penalty[0, 0] = 0.0
    normal = design.T @ (weights[:, None] * design) + penalty
    right = design.T @ (weights * regret)
    try:
        coefficients = np.linalg.solve(normal, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(normal) @ right
    return center, scale, coefficients


def fit_source_regret_certificate(
    features: np.ndarray,
    regret: np.ndarray,
    group_ids: Sequence[str],
    *,
    nominal_coverage: float = 0.90,
    within_group_coverage: float = 1.0,
    minimum_improvement: float = 0.0,
    ridge_penalty: float = 10.0,
    support_margin_std: float = 0.0,
) -> SourceRegretCertificate:
    """Fit a group-cross-fitted upper regret certificate on source data only."""

    feature_matrix = _finite_matrix(features, "regret features")
    values = np.asarray(regret, dtype=np.float64)
    groups = np.asarray(tuple(group_ids), dtype=str)
    _require(values.shape == (len(feature_matrix),), "regret shape changed")
    _require(groups.shape == (len(feature_matrix),), "group IDs changed")
    _require(np.all(np.isfinite(values)), "regret contains non-finite values")
    _require(
        0.0 < nominal_coverage < 1.0,
        "nominal coverage must lie in (0, 1)",
    )
    _require(
        0.0 < within_group_coverage <= 1.0,
        "within-group coverage must lie in (0, 1]",
    )
    _require(minimum_improvement >= 0.0, "minimum improvement is negative")
    _require(ridge_penalty >= 0.0, "ridge penalty is negative")
    _require(support_margin_std >= 0.0, "support margin is negative")
    unique_groups = np.unique(groups)
    _require(len(unique_groups) >= 3, "at least three source groups are required")

    group_scores = []
    for held_group in unique_groups:
        training = groups != held_group
        held = ~training
        _require(
            len(np.unique(groups[training])) >= 2,
            "cross-fit training has too few groups",
        )
        center, scale, coefficients = _fit_group_weighted_ridge(
            feature_matrix[training],
            values[training],
            groups[training],
            ridge_penalty,
        )
        held_design = np.column_stack(
            (
                np.ones(np.sum(held)),
                (feature_matrix[held] - center) / scale,
            )
        )
        residual = values[held] - held_design @ coefficients
        rank = min(
            len(residual),
            int(np.ceil((len(residual) + 1) * within_group_coverage)),
        )
        group_scores.append(float(np.partition(residual, rank - 1)[rank - 1]))

    group_scores_array = np.asarray(group_scores, dtype=np.float64)
    group_rank = min(
        len(group_scores_array),
        int(np.ceil((len(group_scores_array) + 1) * nominal_coverage)),
    )
    upper_residual_quantile = float(
        np.partition(group_scores_array, group_rank - 1)[group_rank - 1]
    )
    center, scale, coefficients = _fit_group_weighted_ridge(
        feature_matrix,
        values,
        groups,
        ridge_penalty,
    )
    standardized = (feature_matrix - center) / scale
    return SourceRegretCertificate(
        feature_center=center,
        feature_scale=scale,
        standardized_feature_lower=np.min(standardized, axis=0),
        standardized_feature_upper=np.max(standardized, axis=0),
        coefficients=coefficients,
        upper_residual_quantile=upper_residual_quantile,
        nominal_coverage=nominal_coverage,
        minimum_improvement=minimum_improvement,
        ridge_penalty=ridge_penalty,
        support_margin_std=support_margin_std,
        source_group_count=len(unique_groups),
        finite_sample_rank=group_rank,
        finite_sample_coverage=group_rank / (len(unique_groups) + 1),
    )


@dataclass(frozen=True)
class SourceGroupRegretBound:
    """One-sided finite-sample bound on regret inside a frozen eligibility gate."""

    upper_regret_m: float
    group_scores_m: np.ndarray
    nominal_coverage: float
    finite_sample_rank: int
    finite_sample_coverage: float
    within_group_coverage: float
    minimum_improvement_m: float

    def __post_init__(self) -> None:
        scores = np.asarray(self.group_scores_m, dtype=np.float64).copy()
        _require(scores.ndim == 1 and len(scores) >= 3, "too few group scores")
        _require(np.all(np.isfinite(scores)), "group scores are not finite")
        _require(np.isfinite(self.upper_regret_m), "upper regret is not finite")
        _require(0.0 < self.nominal_coverage < 1.0, "coverage is invalid")
        _require(
            0.0 < self.within_group_coverage <= 1.0,
            "within-group coverage is invalid",
        )
        _require(
            1 <= self.finite_sample_rank <= len(scores),
            "finite-sample rank is invalid",
        )
        _require(
            self.finite_sample_coverage
            == self.finite_sample_rank / (len(scores) + 1),
            "finite-sample coverage arithmetic changed",
        )
        _require(
            self.minimum_improvement_m >= 0.0,
            "minimum improvement is negative",
        )
        scores.setflags(write=False)
        object.__setattr__(self, "group_scores_m", scores)

    @property
    def candidate_certified(self) -> bool:
        return self.upper_regret_m < -self.minimum_improvement_m


def fit_source_group_regret_bound(
    regret_m: np.ndarray,
    group_ids: Sequence[str],
    *,
    nominal_coverage: float = 0.90,
    within_group_coverage: float = 1.0,
    minimum_improvement_m: float = 0.0,
) -> SourceGroupRegretBound:
    """Calibrate regret directly, with groups as the exchangeable units."""

    regret = np.asarray(regret_m, dtype=np.float64)
    groups = np.asarray(tuple(group_ids), dtype=str)
    _require(regret.ndim == 1 and groups.shape == regret.shape, "regret shape changed")
    _require(np.all(np.isfinite(regret)), "regret contains non-finite values")
    _require(0.0 < nominal_coverage < 1.0, "coverage must lie in (0, 1)")
    _require(
        0.0 < within_group_coverage <= 1.0,
        "within-group coverage must lie in (0, 1]",
    )
    _require(minimum_improvement_m >= 0.0, "minimum improvement is negative")
    unique_groups = np.unique(groups)
    _require(len(unique_groups) >= 3, "at least three source groups are required")
    group_scores = []
    for group in unique_groups:
        values = regret[groups == group]
        rank = min(
            len(values),
            int(np.ceil((len(values) + 1) * within_group_coverage)),
        )
        group_scores.append(float(np.partition(values, rank - 1)[rank - 1]))
    scores = np.asarray(group_scores, dtype=np.float64)
    finite_sample_rank = min(
        len(scores),
        int(np.ceil((len(scores) + 1) * nominal_coverage)),
    )
    upper = float(
        np.partition(scores, finite_sample_rank - 1)[finite_sample_rank - 1]
    )
    return SourceGroupRegretBound(
        upper_regret_m=upper,
        group_scores_m=scores,
        nominal_coverage=nominal_coverage,
        finite_sample_rank=finite_sample_rank,
        finite_sample_coverage=finite_sample_rank / (len(scores) + 1),
        within_group_coverage=within_group_coverage,
        minimum_improvement_m=minimum_improvement_m,
    )


def apply_group_regret_bound(
    baseline_value: np.ndarray,
    candidate_value: np.ndarray,
    bound: SourceGroupRegretBound,
) -> GuardedUpdateDecision:
    """Apply a frozen group-level regret bound with exact baseline fallback."""

    baseline = np.asarray(baseline_value)
    candidate = np.asarray(candidate_value)
    _require(baseline.shape == candidate.shape, "candidate shape changed")
    _require(
        np.all(np.isfinite(baseline)) and np.all(np.isfinite(candidate)),
        "candidate values must be finite",
    )
    if bound.candidate_certified:
        selected = candidate.copy()
        reason = "negative-source-group-regret-bound"
    else:
        selected = baseline.copy()
        reason = "source-group-regret-exact-baseline-fallback"
    return GuardedUpdateDecision(
        selected_value=selected,
        candidate_accepted=bound.candidate_certified,
        predicted_regret=bound.upper_regret_m,
        upper_regret=bound.upper_regret_m,
        reason=reason,
    )


@dataclass(frozen=True)
class GuardedUpdateDecision:
    """Candidate routing decision and exact selected value."""

    selected_value: np.ndarray
    candidate_accepted: bool
    predicted_regret: float
    upper_regret: float
    reason: str

    def __post_init__(self) -> None:
        value = np.asarray(self.selected_value).copy()
        _require(np.all(np.isfinite(value)), "selected update contains non-finite values")
        value.setflags(write=False)
        object.__setattr__(self, "selected_value", value)


def apply_regret_guard(
    baseline_value: np.ndarray,
    candidate_value: np.ndarray,
    features: np.ndarray,
    certificate: SourceRegretCertificate,
) -> GuardedUpdateDecision:
    """Select a candidate only when its source-calibrated regret UCB is negative."""

    baseline = np.asarray(baseline_value)
    candidate = np.asarray(candidate_value)
    _require(baseline.shape == candidate.shape, "candidate shape changed")
    _require(
        np.all(np.isfinite(baseline)) and np.all(np.isfinite(candidate)),
        "candidate values must be finite",
    )
    predicted = certificate.predict_regret(features)
    supported = certificate.in_source_support(features)
    upper = certificate.upper_regret(features)
    accepted = supported and upper < -certificate.minimum_improvement
    if accepted:
        selected = candidate.copy()
        reason = "negative-source-calibrated-regret-bound"
    elif not supported:
        selected = baseline.copy()
        reason = "outside-source-support-exact-baseline-fallback"
    else:
        selected = baseline.copy()
        reason = "exact-baseline-fallback"
    return GuardedUpdateDecision(
        selected_value=selected,
        candidate_accepted=accepted,
        predicted_regret=predicted,
        upper_regret=upper,
        reason=reason,
    )
