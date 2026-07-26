"""Prior-aware, group-mixture inference for gauge- and bias-aware updates.

The legacy gauge-aware solver intentionally uses a strict minimax projection.
This module adds the complementary prior-aware mode: nuisance priors enter a
Schur-complement information calculation, and the robust update uses the
paper's group-level nominal/outlier Student-t mixture rather than multiplying
nominal probability into a row weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .causal4d_provider_v1 import GuardDecisionV1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _symmetric(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values + values.T)


def _eigh_psd(
    values: np.ndarray, *, name: str, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    _require(
        matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1],
        f"{name} must be square",
    )
    _require(np.all(np.isfinite(matrix)), f"{name} must be finite")
    _require(
        np.allclose(matrix, matrix.T, atol=1e-12),
        f"{name} must be symmetric",
    )
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(matrix))
    _require(
        np.all(eigenvalues >= -tolerance),
        f"{name} must be positive semidefinite",
    )
    return np.maximum(eigenvalues, 0.0), eigenvectors


def _precision(values: np.ndarray, *, name: str, floor: float) -> np.ndarray:
    eigenvalues, eigenvectors = _eigh_psd(
        values, name=name, tolerance=floor
    )
    if not len(eigenvalues):
        return np.zeros_like(values, dtype=np.float64)
    return (
        eigenvectors * (1.0 / np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T


def _sqrt_psd(values: np.ndarray, *, name: str, floor: float) -> np.ndarray:
    eigenvalues, eigenvectors = _eigh_psd(
        values, name=name, tolerance=floor
    )
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def _block_diagonal(values: list[np.ndarray]) -> np.ndarray:
    dimension = sum(value.shape[0] for value in values)
    result = np.zeros((dimension, dimension), dtype=np.float64)
    offset = 0
    for value in values:
        width = value.shape[0]
        result[offset : offset + width, offset : offset + width] = value
        offset += width
    return result


def _whitener(values: np.ndarray, *, name: str) -> np.ndarray:
    eigenvalues, eigenvectors = _eigh_psd(
        values, name=name, tolerance=1e-14
    )
    _require(np.all(eigenvalues > 0.0), f"{name} must be positive definite")
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def _log_component_density(
    *,
    dimension: int,
    q: float,
    degrees_of_freedom: float,
    covariance_multiplier: float,
) -> float:
    covariance_to_scale = (degrees_of_freedom - 2.0) / degrees_of_freedom
    scale = covariance_to_scale * covariance_multiplier
    return (
        -0.5 * dimension * math.log(covariance_multiplier)
        - 0.5
        * (degrees_of_freedom + dimension)
        * math.log1p(q / (degrees_of_freedom * scale))
    )


def _group_precision_and_responsibility(
    *,
    q: float,
    dimension: int,
    prior_nominal: float,
    config: "PriorAwareGaugeConfigV1",
) -> tuple[float, float]:
    rho = float(
        np.clip(
            prior_nominal,
            config.probability_floor,
            1.0 - config.probability_floor,
        )
    )
    nominal_log = math.log(rho) + _log_component_density(
        dimension=dimension,
        q=q,
        degrees_of_freedom=config.degrees_of_freedom,
        covariance_multiplier=1.0,
    )
    outlier_log = math.log1p(-rho) + _log_component_density(
        dimension=dimension,
        q=q,
        degrees_of_freedom=config.degrees_of_freedom,
        covariance_multiplier=config.outlier_covariance_multiplier,
    )
    mixture_log = float(np.logaddexp(nominal_log, outlier_log))
    responsibility = math.exp(nominal_log - mixture_log)
    scale = (config.degrees_of_freedom - 2.0) / config.degrees_of_freedom
    nominal_precision = (
        config.degrees_of_freedom + dimension
    ) / (config.degrees_of_freedom * scale + q)
    outlier_scale = scale * config.outlier_covariance_multiplier
    outlier_precision = (
        config.degrees_of_freedom + dimension
    ) / (config.degrees_of_freedom * outlier_scale + q)
    precision = (
        responsibility * nominal_precision
        + (1.0 - responsibility) * outlier_precision
    )
    return (
        max(config.minimum_robust_precision, float(precision)),
        responsibility,
    )


@dataclass(frozen=True)
class PriorAwareGaugeConfigV1:
    """Priors, mixture robustness, information thresholds, and trust region."""

    state_prior_std_m: float = 0.020
    shared_bias_prior_std_m: float = 0.020
    view_bias_prior_std_m: float = 0.010
    effective_samples_per_correlation_group: float = 64.0
    degrees_of_freedom: float = 5.0
    outlier_covariance_multiplier: float = 25.0
    probability_floor: float = 1e-6
    minimum_robust_precision: float = 0.01
    maximum_iterations: int = 12
    convergence_tolerance: float = 1e-9
    maximum_condition_number: float = 1e14
    minimum_conditional_information_fraction: float = 1e-4
    minimum_identifiable_fraction: float = 0.10
    minimum_query_sensitivity_fraction: float = 1e-3
    maximum_state_update_m: float = 0.10
    maximum_update_to_physical_response_ratio: float = 2.0
    prior_eigenvalue_floor: float = 1e-12

    def __post_init__(self) -> None:
        positive = (
            self.state_prior_std_m,
            self.shared_bias_prior_std_m,
            self.view_bias_prior_std_m,
            self.effective_samples_per_correlation_group,
            self.outlier_covariance_multiplier,
            self.minimum_robust_precision,
            self.convergence_tolerance,
            self.maximum_condition_number,
            self.maximum_state_update_m,
            self.maximum_update_to_physical_response_ratio,
            self.prior_eigenvalue_floor,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "configuration scales must be positive",
        )
        _require(
            self.degrees_of_freedom > 2.0,
            "degrees_of_freedom must exceed two",
        )
        _require(
            self.outlier_covariance_multiplier > 1.0,
            "outlier covariance multiplier must exceed one",
        )
        _require(
            0.0 < self.probability_floor < 0.5,
            "probability_floor must lie in (0, 0.5)",
        )
        _require(
            self.maximum_iterations >= 1,
            "maximum_iterations must be positive",
        )
        for name, value, lower_open in (
            (
                "minimum_conditional_information_fraction",
                self.minimum_conditional_information_fraction,
                False,
            ),
            (
                "minimum_identifiable_fraction",
                self.minimum_identifiable_fraction,
                True,
            ),
            (
                "minimum_query_sensitivity_fraction",
                self.minimum_query_sensitivity_fraction,
                False,
            ),
        ):
            valid = (
                0.0 < value <= 1.0
                if lower_open
                else 0.0 <= value <= 1.0
            )
            interval = "(0, 1]" if lower_open else "[0, 1]"
            _require(valid, f"{name} must lie in {interval}")


@dataclass(frozen=True)
class PriorAwareGaugeResultV1:
    """Candidate inference result; validity is not a prospective guard decision."""

    candidate_valid: bool
    reason: str
    state_coefficients: np.ndarray
    gauge_delta: np.ndarray
    shared_bias_coefficients: np.ndarray
    view_bias_coefficients: np.ndarray
    working_posterior_covariance: np.ndarray
    identifiable_state_transform: np.ndarray
    conditional_information_eigenvalues: np.ndarray
    identifiable_fractions: np.ndarray
    query_sensitivity_fractions: np.ndarray
    group_ids: tuple[str, ...]
    group_posterior_nominal_probability: np.ndarray
    group_robust_precision: np.ndarray
    anchor_robust_precision: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = _readonly(self.state_coefficients)
        gauge = _readonly(self.gauge_delta)
        shared = _readonly(self.shared_bias_coefficients)
        view = _readonly(self.view_bias_coefficients)
        covariance = _readonly(self.working_posterior_covariance)
        transform = _readonly(self.identifiable_state_transform)
        eigenvalues = _readonly(self.conditional_information_eigenvalues)
        fractions = _readonly(self.identifiable_fractions)
        query_fractions = _readonly(self.query_sensitivity_fractions)
        responsibilities = _readonly(
            self.group_posterior_nominal_probability
        )
        robust = _readonly(self.group_robust_precision)
        anchor = _readonly(self.anchor_robust_precision)
        _require(
            state.ndim == gauge.ndim == shared.ndim == view.ndim == 1,
            "coefficient arrays must be vectors",
        )
        dimension = len(state) + len(gauge) + len(shared) + len(view)
        _require(
            covariance.shape == (dimension, dimension),
            "working posterior covariance shape changed",
        )
        _require(
            transform.ndim == 2 and transform.shape[0] == len(state),
            "identifiable transform shape changed",
        )
        retained = transform.shape[1]
        _require(
            eigenvalues.shape
            == fractions.shape
            == query_fractions.shape
            == (retained,),
            "identifiability vectors changed shape",
        )
        groups = tuple(map(str, self.group_ids))
        _require(
            all(groups) and len(set(groups)) == len(groups),
            "group_ids must be nonempty and unique",
        )
        _require(
            responsibilities.shape == robust.shape == (len(groups),),
            "group diagnostics changed shape",
        )
        _require(
            np.all(
                (responsibilities >= 0.0) & (responsibilities <= 1.0)
            ),
            "group responsibilities must lie in [0, 1]",
        )
        _require(
            np.all(robust >= 0.0) and np.all(anchor >= 0.0),
            "robust precision must be nonnegative",
        )
        for name, values in (
            ("state_coefficients", state),
            ("gauge_delta", gauge),
            ("shared_bias_coefficients", shared),
            ("view_bias_coefficients", view),
            ("working_posterior_covariance", covariance),
            ("identifiable_state_transform", transform),
            ("conditional_information_eigenvalues", eigenvalues),
            ("identifiable_fractions", fractions),
            ("query_sensitivity_fractions", query_fractions),
            (
                "group_posterior_nominal_probability",
                responsibilities,
            ),
            ("group_robust_precision", robust),
            ("anchor_robust_precision", anchor),
        ):
            _require(np.all(np.isfinite(values)), f"{name} must be finite")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def accepted(self) -> bool:
        """Compatibility alias; this means candidate validity, not guard acceptance."""

        return self.candidate_valid


@dataclass(frozen=True)
class GuardedGaugeSelectionV1:
    """Array selection requiring both candidate validity and a guard decision."""

    candidate_valid: bool
    guard_accepted: bool
    selected_candidate: bool
    reason: str
    guard_decision_id: str
    selected_value: np.ndarray

    def __post_init__(self) -> None:
        selected = np.asarray(self.selected_value).copy()
        selected.setflags(write=False)
        object.__setattr__(self, "selected_value", selected)


def _fallback(
    batch: Any,
    *,
    reason: str,
    group_ids: tuple[str, ...],
    diagnostics: Mapping[str, Any],
) -> PriorAwareGaugeResultV1:
    state_count = batch.state_jacobian.shape[2]
    gauge_count = batch.gauge_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    dimension = state_count + gauge_count + shared_count + view_count
    anchor_count = (
        0
        if batch.anchor_innovation_m is None
        else len(batch.anchor_innovation_m)
    )
    return PriorAwareGaugeResultV1(
        candidate_valid=False,
        reason=reason,
        state_coefficients=np.zeros(state_count),
        gauge_delta=np.zeros(gauge_count),
        shared_bias_coefficients=np.zeros(shared_count),
        view_bias_coefficients=np.zeros(view_count),
        working_posterior_covariance=np.zeros((dimension, dimension)),
        identifiable_state_transform=np.zeros((state_count, 0)),
        conditional_information_eigenvalues=np.zeros(0),
        identifiable_fractions=np.zeros(0),
        query_sensitivity_fractions=np.zeros(0),
        group_ids=group_ids,
        group_posterior_nominal_probability=np.zeros(len(group_ids)),
        group_robust_precision=np.zeros(len(group_ids)),
        anchor_robust_precision=np.zeros(anchor_count),
        diagnostics=diagnostics,
    )


def _validate_batch(batch: Any) -> None:
    innovation = np.asarray(batch.innovation_m, dtype=float)
    covariance = np.asarray(batch.observation_covariance_m2, dtype=float)
    state = np.asarray(batch.state_jacobian, dtype=float)
    query = np.asarray(batch.query_state_jacobian, dtype=float)
    count = len(innovation)
    _require(
        innovation.shape == (count, 3),
        "innovation_m must have shape (M, 3)",
    )
    _require(
        covariance.shape == (count, 3, 3),
        "observation covariance shape changed",
    )
    _require(
        state.ndim == 3
        and state.shape[:2] == (count, 3)
        and state.shape[2] >= 1,
        "state Jacobian shape changed",
    )
    _require(
        query.ndim == 3
        and query.shape[1:] == (3, state.shape[2])
        and len(query),
        "query Jacobian shape changed",
    )
    for name in (
        "gauge_jacobian",
        "shared_bias_jacobian",
        "view_bias_jacobian",
    ):
        values = np.asarray(getattr(batch, name), dtype=float)
        _require(
            values.ndim == 3 and values.shape[:2] == (count, 3),
            f"{name} shape changed",
        )
    for index, values in enumerate(covariance):
        _whitener(values, name=f"observation covariance {index}")
    reliability = np.asarray(batch.prior_reliability, dtype=float)
    nominal = np.asarray(batch.prior_nominal_probability, dtype=float)
    composite = np.asarray(batch.composite_weight, dtype=float)
    _require(
        reliability.shape == nominal.shape == composite.shape == (count,),
        "row probability shapes changed",
    )
    _require(
        np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "prior reliability must lie in [0, 1]",
    )
    _require(
        np.all((nominal >= 0.0) & (nominal <= 1.0)),
        "prior nominal probability must lie in [0, 1]",
    )
    _require(
        np.all((composite > 0.0) & (composite <= 1.0)),
        "composite weight must lie in (0, 1]",
    )
    groups = tuple(map(str, batch.correlation_group_ids))
    _require(
        len(groups) == count and all(groups),
        "correlation group IDs changed",
    )


def update_prior_aware_gauge_belief(
    batch: Any,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> PriorAwareGaugeResultV1:
    """Infer a query-relevant candidate using nuisance-prior Schur information."""

    _validate_batch(batch)
    cfg = config or PriorAwareGaugeConfigV1()
    innovation = np.asarray(batch.innovation_m, dtype=np.float64)
    covariance = np.asarray(
        batch.observation_covariance_m2, dtype=np.float64
    )
    state = np.asarray(batch.state_jacobian, dtype=np.float64)
    query = np.asarray(batch.query_state_jacobian, dtype=np.float64)
    gauge = np.asarray(batch.gauge_jacobian, dtype=np.float64)
    shared = np.asarray(batch.shared_bias_jacobian, dtype=np.float64)
    view = np.asarray(batch.view_bias_jacobian, dtype=np.float64)
    nuisance = np.concatenate((gauge, shared, view), axis=2)
    state_count = state.shape[2]
    gauge_count = gauge.shape[2]
    shared_count = shared.shape[2]
    view_count = view.shape[2]
    nuisance_count = nuisance.shape[2]

    group_labels = tuple(map(str, batch.correlation_group_ids))
    group_ids = tuple(dict.fromkeys(group_labels))
    group_indices = {
        group_id: np.asarray(
            [
                index
                for index, value in enumerate(group_labels)
                if value == group_id
            ],
            dtype=np.int64,
        )
        for group_id in group_ids
    }
    row_base = np.zeros(len(innovation), dtype=np.float64)
    group_prior = np.empty(len(group_ids), dtype=np.float64)
    group_composite = np.empty(len(group_ids), dtype=np.float64)
    for position, group_id in enumerate(group_ids):
        selected = group_indices[group_id]
        nominal = np.asarray(batch.prior_nominal_probability)[selected]
        composite = np.asarray(batch.composite_weight)[selected]
        _require(
            np.allclose(nominal, nominal[0], atol=1e-12),
            "prior nominal probability must be constant within a group",
        )
        _require(
            np.allclose(composite, composite[0], atol=1e-12),
            "composite weight must be constant within a group",
        )
        group_prior[position] = float(nominal[0])
        group_composite[position] = float(composite[0])
        cap = min(
            cfg.effective_samples_per_correlation_group,
            float(len(selected)),
        ) / len(selected)
        row_base[selected] = (
            np.asarray(batch.prior_reliability)[selected]
            * group_composite[position]
            * cap
        )

    diagnostics: dict[str, Any] = {
        "identifiability_mode": "prior-aware-schur-v1",
        "robust_likelihood": "grouped nominal/outlier Student-t mixture",
        "posterior_covariance_kind": (
            "working Laplace covariance from converged IRLS/EM system"
        ),
        "association_probability_used_as_reliability": False,
        "prior_nominal_probability_used_inside_mixture": True,
        "observation_count": len(innovation),
        "active_observation_count": int(np.sum(row_base > 0.0)),
        "correlation_group_count": len(group_ids),
    }
    if not np.any(row_base > 0.0):
        return _fallback(
            batch,
            reason="no-observation-support",
            group_ids=group_ids,
            diagnostics=diagnostics,
        )

    whiteners = np.stack(
        [
            _whitener(
                values,
                name=f"observation covariance {index}",
            )
            for index, values in enumerate(covariance)
        ]
    )
    target_white = np.einsum(
        "mij,mj->mi", whiteners, innovation, optimize=True
    )
    state_white = np.einsum(
        "mij,mjs->mis", whiteners, state, optimize=True
    )
    nuisance_white = np.einsum(
        "mij,mjn->min", whiteners, nuisance, optimize=True
    )

    state_prior = (
        np.eye(state_count) * cfg.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2, dtype=np.float64)
    )
    nuisance_prior_blocks: list[np.ndarray] = []
    if gauge_count:
        nuisance_prior_blocks.append(
            np.asarray(batch.gauge_prior_covariance, dtype=np.float64)
        )
    if shared_count:
        nuisance_prior_blocks.append(
            np.eye(shared_count) * cfg.shared_bias_prior_std_m**2
        )
    if view_count:
        nuisance_prior_blocks.append(
            np.eye(view_count) * cfg.view_bias_prior_std_m**2
        )
    nuisance_prior = _block_diagonal(nuisance_prior_blocks)
    nuisance_prior_precision = _precision(
        nuisance_prior,
        name="nuisance prior covariance",
        floor=cfg.prior_eigenvalue_floor,
    )

    expected_group_precision = np.asarray(
        [
            rho
            + (1.0 - rho) / cfg.outlier_covariance_multiplier
            for rho in group_prior
        ],
        dtype=np.float64,
    )
    identification_row_weight = row_base.copy()
    for position, group_id in enumerate(group_ids):
        identification_row_weight[group_indices[group_id]] *= (
            expected_group_precision[position]
        )
    sqrt_identification = np.sqrt(identification_row_weight)
    hx = (
        sqrt_identification[:, None, None] * state_white
    ).reshape(-1, state_count)
    hn = (
        sqrt_identification[:, None, None] * nuisance_white
    ).reshape(-1, nuisance_count)
    a_data = hx.T @ hx
    cross = hx.T @ hn
    d = nuisance_prior_precision + hn.T @ hn

    anchor_target_white = np.zeros((0, 3), dtype=np.float64)
    anchor_state_white = np.zeros(
        (0, 3, state_count), dtype=np.float64
    )
    anchor_whiteners = np.zeros((0, 3, 3), dtype=np.float64)
    if batch.anchor_innovation_m is not None:
        anchor_covariance = np.asarray(
            batch.anchor_covariance_m2, dtype=np.float64
        )
        anchor_whiteners = np.stack(
            [
                _whitener(
                    values,
                    name=f"anchor covariance {index}",
                )
                for index, values in enumerate(anchor_covariance)
            ]
        )
        anchor_target_white = np.einsum(
            "aij,aj->ai",
            anchor_whiteners,
            np.asarray(batch.anchor_innovation_m),
            optimize=True,
        )
        anchor_state_white = np.einsum(
            "aij,ajs->ais",
            anchor_whiteners,
            np.asarray(batch.anchor_state_jacobian),
            optimize=True,
        )
        anchor_flat = anchor_state_white.reshape(-1, state_count)
        a_data += anchor_flat.T @ anchor_flat

    if nuisance_count:
        try:
            conditional_information = (
                a_data - cross @ np.linalg.solve(d, cross.T)
            )
        except np.linalg.LinAlgError:
            return _fallback(
                batch,
                reason="singular-nuisance-information",
                group_ids=group_ids,
                diagnostics=diagnostics,
            )
    else:
        conditional_information = a_data
    conditional_information = _symmetric(conditional_information)
    state_prior_sqrt = _sqrt_psd(
        state_prior,
        name="state prior covariance",
        floor=cfg.prior_eigenvalue_floor,
    )
    prior_standard_information = _symmetric(
        state_prior_sqrt.T
        @ conditional_information
        @ state_prior_sqrt
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        prior_standard_information
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum_information = float(np.max(eigenvalues, initial=0.0))
    query_flat = query.reshape(-1, state_count)
    transforms: list[np.ndarray] = []
    retained_eigenvalues: list[float] = []
    identifiable_fractions: list[float] = []
    query_fractions: list[float] = []
    candidates: list[tuple[np.ndarray, float, float]] = []
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue <= cfg.prior_eigenvalue_floor:
            continue
        direction = state_prior_sqrt @ eigenvectors[:, index]
        query_norm = float(np.linalg.norm(query_flat @ direction))
        candidates.append((direction, query_norm, float(eigenvalue)))
    maximum_query_norm = float(
        np.max([item[1] for item in candidates], initial=0.0)
    )
    for direction, query_norm, eigenvalue in candidates:
        known_nuisance_information = float(
            direction @ a_data @ direction
        )
        conditional_value = float(
            direction @ conditional_information @ direction
        )
        identifiable_fraction = conditional_value / max(
            known_nuisance_information,
            cfg.prior_eigenvalue_floor,
        )
        query_fraction = (
            query_norm / maximum_query_norm
            if maximum_query_norm > 0.0
            else 0.0
        )
        information_fraction = (
            eigenvalue / maximum_information
            if maximum_information > 0.0
            else 0.0
        )
        if (
            information_fraction
            >= cfg.minimum_conditional_information_fraction
            and identifiable_fraction >= cfg.minimum_identifiable_fraction
            and query_fraction >= cfg.minimum_query_sensitivity_fraction
        ):
            transforms.append(direction)
            retained_eigenvalues.append(eigenvalue)
            identifiable_fractions.append(
                min(1.0, max(0.0, identifiable_fraction))
            )
            query_fractions.append(
                min(1.0, max(0.0, query_fraction))
            )
    if not transforms:
        diagnostics.update(
            {
                "maximum_conditional_information_eigenvalue": (
                    maximum_information
                ),
                "maximum_query_sensitivity_norm": maximum_query_norm,
            }
        )
        return _fallback(
            batch,
            reason="no-identifiable-query-state",
            group_ids=group_ids,
            diagnostics=diagnostics,
        )
    transform = np.column_stack(transforms)
    reduced_state_count = transform.shape[1]
    diagnostics.update(
        {
            "identifiable_query_state_mode_count": reduced_state_count,
            "maximum_conditional_information_eigenvalue": (
                maximum_information
            ),
            "maximum_query_sensitivity_norm": maximum_query_norm,
        }
    )

    reduced_state = np.einsum(
        "mcs,sr->mcr", state_white, transform, optimize=True
    )
    observation_design = np.concatenate(
        (reduced_state, nuisance_white), axis=2
    )
    raw_reduced_state = np.einsum(
        "mcs,sr->mcr", state, transform, optimize=True
    )
    raw_design = np.concatenate((raw_reduced_state, nuisance), axis=2)
    joint_dimension = reduced_state_count + nuisance_count
    reduced_state_prior = transform.T @ state_prior @ transform
    prior_covariance = _block_diagonal(
        [reduced_state_prior, nuisance_prior]
        if nuisance_count
        else [reduced_state_prior]
    )
    prior_precision = _precision(
        prior_covariance,
        name="joint prior covariance",
        floor=cfg.prior_eigenvalue_floor,
    )

    anchor_design = np.zeros(
        (len(anchor_state_white), 3, joint_dimension),
        dtype=np.float64,
    )
    raw_anchor_design = np.zeros_like(anchor_design)
    if len(anchor_state_white):
        anchor_design[:, :, :reduced_state_count] = np.einsum(
            "acs,sr->acr",
            anchor_state_white,
            transform,
            optimize=True,
        )
        raw_anchor_design[:, :, :reduced_state_count] = np.einsum(
            "acs,sr->acr",
            np.asarray(batch.anchor_state_jacobian),
            transform,
            optimize=True,
        )

    group_precision = expected_group_precision.copy()
    group_responsibility = group_prior.copy()
    anchor_precision = np.ones(
        len(anchor_target_white), dtype=np.float64
    )
    solution = np.zeros(joint_dimension, dtype=np.float64)

    def posterior_system() -> tuple[np.ndarray, np.ndarray]:
        normal = prior_precision.copy()
        right = np.zeros(joint_dimension, dtype=np.float64)
        row_precision = np.zeros(len(innovation), dtype=np.float64)
        for position, group_id in enumerate(group_ids):
            row_precision[group_indices[group_id]] = (
                group_precision[position]
            )
        effective = row_base * row_precision
        normal += np.einsum(
            "m,mci,mcj->ij",
            effective,
            observation_design,
            observation_design,
            optimize=True,
        )
        right += np.einsum(
            "m,mci,mc->i",
            effective,
            observation_design,
            target_white,
            optimize=True,
        )
        if len(anchor_target_white):
            normal += np.einsum(
                "a,aci,acj->ij",
                anchor_precision,
                anchor_design,
                anchor_design,
                optimize=True,
            )
            right += np.einsum(
                "a,aci,ac->i",
                anchor_precision,
                anchor_design,
                anchor_target_white,
                optimize=True,
            )
        return normal, right

    iteration = 0
    for iteration in range(cfg.maximum_iterations):
        previous = solution.copy()
        normal, right = posterior_system()
        condition_number = float(np.linalg.cond(normal))
        if (
            not np.isfinite(condition_number)
            or condition_number > cfg.maximum_condition_number
        ):
            diagnostics["condition_number"] = condition_number
            return _fallback(
                batch,
                reason="ill-conditioned-posterior",
                group_ids=group_ids,
                diagnostics=diagnostics,
            )
        try:
            solution = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            return _fallback(
                batch,
                reason="singular-posterior",
                group_ids=group_ids,
                diagnostics=diagnostics,
            )
        residual = innovation - np.einsum(
            "mci,i->mc", raw_design, solution, optimize=True
        )
        white_residual = np.einsum(
            "mij,mj->mi", whiteners, residual, optimize=True
        )
        for position, group_id in enumerate(group_ids):
            selected = group_indices[group_id]
            q = float(
                np.sum(
                    np.asarray(batch.prior_reliability)[selected, None]
                    * np.square(white_residual[selected])
                )
            )
            dimension = 3 * len(selected)
            (
                group_precision[position],
                group_responsibility[position],
            ) = _group_precision_and_responsibility(
                q=q,
                dimension=dimension,
                prior_nominal=group_prior[position],
                config=cfg,
            )
        if len(anchor_target_white):
            anchor_residual = np.asarray(
                batch.anchor_innovation_m
            ) - np.einsum(
                "aci,i->ac", raw_anchor_design, solution, optimize=True
            )
            white_anchor_residual = np.einsum(
                "aij,aj->ai",
                anchor_whiteners,
                anchor_residual,
                optimize=True,
            )
            scale = (
                cfg.degrees_of_freedom - 2.0
            ) / cfg.degrees_of_freedom
            q = np.sum(np.square(white_anchor_residual), axis=1)
            anchor_precision = np.maximum(
                cfg.minimum_robust_precision,
                (cfg.degrees_of_freedom + 3.0)
                / (cfg.degrees_of_freedom * scale + q),
            )
        if np.linalg.norm(solution - previous) <= cfg.convergence_tolerance:
            break

    normal, right = posterior_system()
    try:
        solution = np.linalg.solve(normal, right)
        reduced_covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return _fallback(
            batch,
            reason="singular-final-posterior",
            group_ids=group_ids,
            diagnostics=diagnostics,
        )
    full_dimension = state_count + nuisance_count
    mapping = np.zeros(
        (full_dimension, joint_dimension), dtype=np.float64
    )
    mapping[:state_count, :reduced_state_count] = transform
    if nuisance_count:
        mapping[state_count:, reduced_state_count:] = np.eye(
            nuisance_count
        )
    full_solution = mapping @ solution
    full_covariance = mapping @ reduced_covariance @ mapping.T
    state_coefficients = full_solution[:state_count]
    gauge_slice = slice(state_count, state_count + gauge_count)
    shared_slice = slice(gauge_slice.stop, gauge_slice.stop + shared_count)
    view_slice = slice(shared_slice.stop, shared_slice.stop + view_count)
    query_update = np.einsum(
        "qcs,s->qc", query, state_coefficients, optimize=True
    )
    maximum_query_update = float(
        np.max(np.linalg.norm(query_update, axis=1), initial=0.0)
    )
    relative_limit = (
        cfg.maximum_update_to_physical_response_ratio
        * float(batch.physical_response_scale_m)
    )
    update_limit = min(cfg.maximum_state_update_m, relative_limit)
    candidate_valid = (
        np.all(np.isfinite(full_solution))
        and maximum_query_update <= update_limit
    )
    reason = (
        "candidate-valid"
        if candidate_valid
        else "implausible-state-update"
    )
    diagnostics.update(
        {
            "iterations": iteration + 1,
            "condition_number": float(np.linalg.cond(normal)),
            "maximum_query_state_update_m": maximum_query_update,
            "active_state_update_limit_m": update_limit,
            "physical_response_relative_limit_m": relative_limit,
            "candidate_valid": bool(candidate_valid),
        }
    )
    return PriorAwareGaugeResultV1(
        candidate_valid=bool(candidate_valid),
        reason=reason,
        state_coefficients=(
            state_coefficients
            if candidate_valid
            else np.zeros_like(state_coefficients)
        ),
        gauge_delta=(
            full_solution[gauge_slice]
            if candidate_valid
            else np.zeros(gauge_count)
        ),
        shared_bias_coefficients=(
            full_solution[shared_slice]
            if candidate_valid
            else np.zeros(shared_count)
        ),
        view_bias_coefficients=(
            full_solution[view_slice]
            if candidate_valid
            else np.zeros(view_count)
        ),
        working_posterior_covariance=(
            full_covariance
            if candidate_valid
            else np.zeros_like(full_covariance)
        ),
        identifiable_state_transform=transform,
        conditional_information_eigenvalues=np.asarray(
            retained_eigenvalues
        ),
        identifiable_fractions=np.asarray(identifiable_fractions),
        query_sensitivity_fractions=np.asarray(query_fractions),
        group_ids=group_ids,
        group_posterior_nominal_probability=group_responsibility,
        group_robust_precision=group_precision,
        anchor_robust_precision=anchor_precision,
        diagnostics=diagnostics,
    )


def decode_prior_aware_query(
    result: PriorAwareGaugeResultV1,
    query_state_jacobian: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_state_jacobian, dtype=np.float64)
    _require(
        query.ndim == 3
        and query.shape[1:] == (3, len(result.state_coefficients)),
        "query state Jacobian shape changed",
    )
    if not result.candidate_valid:
        return np.zeros(query.shape[:2], dtype=np.float64)
    return np.einsum(
        "qcs,s->qc", query, result.state_coefficients, optimize=True
    )


def select_guarded_gauge_candidate(
    baseline: np.ndarray,
    candidate: np.ndarray,
    result: PriorAwareGaugeResultV1,
    guard: GuardDecisionV1,
) -> GuardedGaugeSelectionV1:
    """Require the prospective guard before selecting an array candidate."""

    baseline_array = np.asarray(baseline)
    candidate_array = np.asarray(candidate)
    _require(
        candidate_array.shape == baseline_array.shape,
        "candidate shape changed",
    )
    _require(
        guard.candidate_valid == result.candidate_valid,
        "guard candidate validity does not match inference",
    )
    selected_candidate = result.candidate_valid and guard.guard_accepted
    selected = (
        candidate_array.copy()
        if selected_candidate
        else baseline_array.copy()
    )
    if (
        not selected_candidate
        and selected.tobytes() != baseline_array.tobytes()
    ):
        raise AssertionError("guarded fallback changed baseline bytes")
    reason = (
        "guard-accepted"
        if selected_candidate
        else (
            "candidate-invalid"
            if not result.candidate_valid
            else "guard-rejected"
        )
    )
    return GuardedGaugeSelectionV1(
        candidate_valid=result.candidate_valid,
        guard_accepted=guard.guard_accepted,
        selected_candidate=selected_candidate,
        reason=reason,
        guard_decision_id=guard.decision_id,
        selected_value=selected,
    )


__all__ = [
    "GuardedGaugeSelectionV1",
    "PriorAwareGaugeConfigV1",
    "PriorAwareGaugeResultV1",
    "decode_prior_aware_query",
    "select_guarded_gauge_candidate",
    "update_prior_aware_gauge_belief",
]
