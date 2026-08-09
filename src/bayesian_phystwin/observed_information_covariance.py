"""Exact local mixture-curvature covariance for prior-aware updates.

This prospective post-processor reconstructs the reduced robust objective at an
already admitted prior-aware solution. It verifies the existing working
Gauss--Newton covariance, forms the exact observed mixture Hessian, and maps its
inverse back to the complete state and nuisance domain without changing the
point estimate or any frozen solver.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    literal_lower_hex,
    plain_json,
)
from ._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    _block_diagonal,
    _regularized_precision,
)
from ._portable_contracts import content_id
from ._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _full_covariance,
    _group_layout,
    _prior_covariances,
    _spd_covariance,
    _student_t_mixture_statistics,
    _whiten,
)
from .posterior_covariance_semantics import PosteriorCovarianceSemanticsV1

OBSERVED_INFORMATION_COVARIANCE_SCHEMA = (
    "bayesian_phystwin.observed_information_covariance"
)
OBSERVED_INFORMATION_COVARIANCE_VERSION = 1
LIKELIHOOD_POWER_SEMANTICS = "grouped-student-t-generalized-bayes-power-v1"
_PRIOR_STANDARDIZATION_ATOL = 1e-8
_PRIOR_NULLSPACE_ATOL = 1e-10

FloatArray: TypeAlias = NDArray[np.float64]


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _finite_matrix(value: object, *, name: str) -> FloatArray:
    try:
        matrix: FloatArray = np.array(
            value,
            dtype=np.float64,
            copy=True,
            order="C",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real matrix") from error
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix")
    return matrix


def _symmetric_matrix(value: object, *, name: str) -> FloatArray:
    matrix = _finite_matrix(value, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-11, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (matrix + matrix.T)


def _finite_vector(value: object, *, name: str) -> FloatArray:
    try:
        vector: FloatArray = np.array(
            value,
            dtype=np.float64,
            copy=True,
            order="C",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real vector") from error
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector")
    return vector


def _immutable(array: FloatArray) -> FloatArray:
    canonical: FloatArray = np.asarray(
        array,
        dtype=np.dtype("<f8"),
        order="C",
    )
    frozen: FloatArray = np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=np.dtype("<f8"),
    ).reshape(canonical.shape)
    return frozen


def _array_record(array: FloatArray) -> dict[str, object]:
    canonical: FloatArray = np.asarray(
        array,
        dtype=np.dtype("<f8"),
        order="C",
    )
    return {
        "dtype": "float64-le",
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _group_ids(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(
        _canonical_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _finite_real(value: object, *, name: str, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class ObservedInformationCovarianceResultV1:
    """Content-addressed exact local mixture-curvature covariance."""

    working_information: FloatArray
    observed_information: FloatArray
    reduced_covariance: FloatArray
    full_covariance: FloatArray
    state_prior_covariance: FloatArray
    state_mapping: FloatArray
    observation_group_ids: Sequence[str]
    anchor_group_ids: Sequence[str]
    observation_group_power: FloatArray
    anchor_group_power: FloatArray
    observation_group_expected_precision: FloatArray
    anchor_group_expected_precision: FloatArray
    observation_group_precision_derivative: FloatArray
    anchor_group_precision_derivative: FloatArray
    prior_eigenvalue_floor: float
    condition_number: float
    covariance_semantics: PosteriorCovarianceSemanticsV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        working = _symmetric_matrix(
            self.working_information,
            name="working_information",
        )
        observed = _symmetric_matrix(
            self.observed_information,
            name="observed_information",
        )
        reduced = _symmetric_matrix(
            self.reduced_covariance,
            name="reduced_covariance",
        )
        full = _symmetric_matrix(self.full_covariance, name="full_covariance")
        state_prior = _symmetric_matrix(
            self.state_prior_covariance,
            name="state_prior_covariance",
        )
        mapping = _finite_matrix(self.state_mapping, name="state_mapping")
        if working.shape != observed.shape or reduced.shape != observed.shape:
            raise ValueError("reduced information and covariance shapes changed")
        if not observed.shape[0]:
            raise ValueError("observed information must be nonempty")
        try:
            np.linalg.cholesky(working)
        except np.linalg.LinAlgError as error:
            raise ValueError("working_information must be positive definite") from error
        try:
            np.linalg.cholesky(observed)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "observed_information must be positive definite"
            ) from error
        identity = observed @ reduced
        if not np.allclose(
            identity,
            np.eye(len(observed)),
            atol=1e-8,
            rtol=1e-8,
        ):
            raise ValueError("reduced_covariance does not invert observed_information")

        state_count = len(state_prior)
        if mapping.ndim != 2 or mapping.shape[0] != state_count:
            raise ValueError("state_mapping row count must match state prior")
        retained = mapping.shape[1]
        prior_eigenvalue_floor = _finite_real(
            self.prior_eigenvalue_floor,
            name="prior_eigenvalue_floor",
            minimum=0.0,
        )
        if prior_eigenvalue_floor <= 0.0:
            raise ValueError("prior_eigenvalue_floor must be positive")
        prior_eigenvalues, prior_eigenvectors = np.linalg.eigh(state_prior)
        if np.any(prior_eigenvalues < -prior_eigenvalue_floor):
            raise ValueError("state_prior_covariance must be positive semidefinite")
        if retained == 0:
            raise ValueError("state_mapping must retain at least one state direction")
        positive_prior = prior_eigenvalues > prior_eigenvalue_floor
        if retained > int(np.count_nonzero(positive_prior)):
            raise ValueError(
                "state_mapping retains more directions than the prior supports"
            )
        positive_basis = prior_eigenvectors[:, positive_prior]
        standardized_mapping = (positive_basis.T @ mapping) / np.sqrt(
            prior_eigenvalues[positive_prior]
        )[:, None]
        if not np.allclose(
            standardized_mapping.T @ standardized_mapping,
            np.eye(retained),
            atol=_PRIOR_STANDARDIZATION_ATOL,
            rtol=_PRIOR_STANDARDIZATION_ATOL,
        ):
            raise ValueError("state_mapping is not standardized by the state prior")
        null_basis = prior_eigenvectors[:, ~positive_prior]
        null_component = null_basis.T @ mapping
        null_tolerance = _PRIOR_NULLSPACE_ATOL * max(
            1.0,
            float(np.linalg.norm(mapping, ord=2)),
        )
        if null_component.size and not np.allclose(
            null_component,
            np.zeros_like(null_component),
            atol=null_tolerance,
            rtol=0.0,
        ):
            raise ValueError("state_mapping leaks into the state-prior nullspace")
        if np.any(np.linalg.eigvalsh(full) < -prior_eigenvalue_floor):
            raise ValueError("full_covariance must be positive semidefinite")
        nuisance_count = len(observed) - retained
        if nuisance_count < 0:
            raise ValueError("state_mapping is wider than observed information")
        expected_full = _full_covariance(
            state_prior,
            mapping,
            reduced,
            nuisance_count,
        )
        if full.shape != expected_full.shape or not np.allclose(
            full,
            expected_full,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("full_covariance does not match the reduced mapping")

        observation_ids = _group_ids(
            self.observation_group_ids,
            name="observation_group_ids",
        )
        anchor_ids = _group_ids(
            self.anchor_group_ids,
            name="anchor_group_ids",
        )
        observation_arrays = (
            ("observation_group_power", self.observation_group_power),
            (
                "observation_group_expected_precision",
                self.observation_group_expected_precision,
            ),
            (
                "observation_group_precision_derivative",
                self.observation_group_precision_derivative,
            ),
        )
        anchor_arrays = (
            ("anchor_group_power", self.anchor_group_power),
            (
                "anchor_group_expected_precision",
                self.anchor_group_expected_precision,
            ),
            (
                "anchor_group_precision_derivative",
                self.anchor_group_precision_derivative,
            ),
        )
        validated: dict[str, FloatArray] = {}
        for name, value in (*observation_arrays, *anchor_arrays):
            validated[name] = _finite_vector(value, name=name)
        if any(
            len(validated[name]) != len(observation_ids)
            for name, _ in observation_arrays
        ):
            raise ValueError("observation group arrays changed length")
        if any(len(validated[name]) != len(anchor_ids) for name, _ in anchor_arrays):
            raise ValueError("anchor group arrays changed length")
        if np.any(validated["observation_group_power"] < 0.0) or np.any(
            validated["anchor_group_power"] < 0.0
        ):
            raise ValueError("group powers must be nonnegative")
        if np.any(validated["observation_group_expected_precision"] < 0.0) or np.any(
            validated["anchor_group_expected_precision"] < 0.0
        ):
            raise ValueError("group expected precisions must be nonnegative")

        condition_number = _finite_real(
            self.condition_number,
            name="condition_number",
            minimum=1.0,
        )
        expected_condition = float(np.linalg.cond(observed))
        if not np.isclose(
            condition_number,
            expected_condition,
            atol=0.0,
            rtol=1e-12,
        ):
            raise ValueError("condition_number does not match observed_information")

        semantics = self.covariance_semantics
        if not isinstance(semantics, PosteriorCovarianceSemanticsV1):
            raise ValueError(
                "covariance_semantics must be a PosteriorCovarianceSemanticsV1"
            )
        if semantics.method != "laplace_observed_information":
            raise ValueError("covariance_semantics method must be observed information")
        if semantics.dimension != len(full):
            raise ValueError("covariance_semantics dimension changed")
        if not semantics.mixture_curvature_exact:
            raise ValueError(
                "covariance_semantics must declare exact mixture curvature"
            )
        if semantics.group_score_correction or semantics.calibrated:
            raise ValueError("observed-information covariance flags are inconsistent")

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="observed information covariance metadata",
        )
        object.__setattr__(self, "working_information", _immutable(working))
        object.__setattr__(self, "observed_information", _immutable(observed))
        object.__setattr__(self, "reduced_covariance", _immutable(reduced))
        object.__setattr__(self, "full_covariance", _immutable(full))
        object.__setattr__(
            self,
            "state_prior_covariance",
            _immutable(state_prior),
        )
        object.__setattr__(self, "state_mapping", _immutable(mapping))
        object.__setattr__(self, "observation_group_ids", observation_ids)
        object.__setattr__(self, "anchor_group_ids", anchor_ids)
        for name, value in validated.items():
            object.__setattr__(self, name, _immutable(value))
        object.__setattr__(
            self,
            "prior_eigenvalue_floor",
            prior_eigenvalue_floor,
        )
        object.__setattr__(self, "condition_number", condition_number)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = literal_lower_hex(
                supplied_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match covariance content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def full_dimension(self) -> int:
        return len(self.full_covariance)

    @property
    def reduced_dimension(self) -> int:
        return len(self.reduced_covariance)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": OBSERVED_INFORMATION_COVARIANCE_SCHEMA,
            "schema_version": OBSERVED_INFORMATION_COVARIANCE_VERSION,
            "working_information": _array_record(self.working_information),
            "observed_information": _array_record(self.observed_information),
            "reduced_covariance": _array_record(self.reduced_covariance),
            "full_covariance": _array_record(self.full_covariance),
            "state_prior_covariance": _array_record(self.state_prior_covariance),
            "state_mapping": _array_record(self.state_mapping),
            "prior_eigenvalue_floor": self.prior_eigenvalue_floor,
            "full_dimension": self.full_dimension,
            "reduced_dimension": self.reduced_dimension,
            "observation_group_ids": list(self.observation_group_ids),
            "anchor_group_ids": list(self.anchor_group_ids),
            "observation_group_power": _array_record(self.observation_group_power),
            "anchor_group_power": _array_record(self.anchor_group_power),
            "observation_group_expected_precision": _array_record(
                self.observation_group_expected_precision
            ),
            "anchor_group_expected_precision": _array_record(
                self.anchor_group_expected_precision
            ),
            "observation_group_precision_derivative": _array_record(
                self.observation_group_precision_derivative
            ),
            "anchor_group_precision_derivative": _array_record(
                self.anchor_group_precision_derivative
            ),
            "condition_number": self.condition_number,
            "covariance_semantics_id": self.covariance_semantics.artifact_id,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def _validate_result_dimensions(
    batch: GaugeAwareObservationBatch,
    result: GaugeAwareBeliefResult,
) -> tuple[int, int, int, int, int]:
    state_count = batch.state_jacobian.shape[2]
    gauge_count = batch.gauge_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    observed = (
        len(result.state_coefficients),
        len(result.gauge_delta),
        len(result.shared_bias_coefficients),
        len(result.view_bias_coefficients),
        len(result.anchor_bias_coefficients),
    )
    expected = (
        state_count,
        gauge_count,
        shared_count,
        view_count,
        anchor_bias_count,
    )
    if observed != expected:
        raise ValueError("result coefficient dimensions do not match batch")
    return expected


def observed_information_covariance_from_prior_aware_result(
    batch: GaugeAwareObservationBatch,
    result: GaugeAwareBeliefResult,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ObservedInformationCovarianceResultV1:
    """Reconstruct and invert the exact local grouped-mixture Hessian.

    The function accepts only an inference-admissible result from the exact
    prior-aware mixture objective. It independently reconstructs the working
    normal matrix, verifies the result covariance and robust group precisions,
    then adds the exact responsibility/Student-t curvature terms.
    """

    if not isinstance(batch, GaugeAwareObservationBatch):
        raise TypeError("batch must be a GaugeAwareObservationBatch")
    if not isinstance(result, GaugeAwareBeliefResult):
        raise TypeError("result must be a GaugeAwareBeliefResult")
    if not result.inference_admissible:
        raise ValueError("observed information requires an admissible result")
    cfg = config or PriorAwareGaugeConfigV1()
    if cfg.minimum_robust_precision != 0.0:
        raise ValueError(
            "exact observed information requires minimum_robust_precision=0"
        )
    if batch.prior_nominal_probability is None or batch.composite_weight is None:
        raise ValueError("validated observation mixture metadata is missing")

    (
        state_count,
        gauge_count,
        shared_count,
        view_count,
        anchor_bias_count,
    ) = _validate_result_dimensions(batch, result)
    nuisance_count = gauge_count + shared_count + view_count + anchor_bias_count
    ordinary_nuisance = np.concatenate(
        (
            batch.gauge_jacobian,
            batch.shared_bias_jacobian,
            batch.view_bias_jacobian,
            np.zeros((len(batch.innovation_m), 3, anchor_bias_count)),
        ),
        axis=2,
    )

    anchor_count = (
        0 if batch.anchor_innovation_m is None else len(batch.anchor_innovation_m)
    )
    if anchor_count:
        if (
            batch.anchor_innovation_m is None
            or batch.anchor_covariance_m2 is None
            or batch.anchor_state_jacobian is None
            or batch.anchor_correlation_group_ids is None
            or batch.anchor_prior_reliability is None
            or batch.anchor_prior_nominal_probability is None
            or batch.anchor_composite_weight is None
        ):
            raise ValueError("validated anchor mixture metadata is missing")
        anchor_innovation = np.asarray(batch.anchor_innovation_m)
        anchor_covariance = np.asarray(batch.anchor_covariance_m2)
        anchor_state = np.asarray(batch.anchor_state_jacobian)
        anchor_groups_input = batch.anchor_correlation_group_ids
        anchor_reliability = np.asarray(batch.anchor_prior_reliability)
        anchor_nominal = np.asarray(batch.anchor_prior_nominal_probability)
        anchor_composite = np.asarray(batch.anchor_composite_weight)
        anchor_bias = (
            np.zeros((anchor_count, 3, anchor_bias_count))
            if batch.anchor_bias_jacobian is None
            else np.asarray(batch.anchor_bias_jacobian)
        )
    else:
        anchor_innovation = np.zeros((0, 3), dtype=np.float64)
        anchor_covariance = np.zeros((0, 3, 3), dtype=np.float64)
        anchor_state = np.zeros((0, 3, state_count), dtype=np.float64)
        anchor_groups_input = ()
        anchor_reliability = np.zeros(0, dtype=np.float64)
        anchor_nominal = np.zeros(0, dtype=np.float64)
        anchor_composite = np.zeros(0, dtype=np.float64)
        anchor_bias = np.zeros((0, 3, anchor_bias_count), dtype=np.float64)
    anchor_nuisance = np.concatenate(
        (
            np.zeros(
                (
                    anchor_count,
                    3,
                    gauge_count + shared_count + view_count,
                ),
                dtype=np.float64,
            ),
            anchor_bias,
        ),
        axis=2,
    )

    _, (state_white, nuisance_white), whiteners = _whiten(
        batch.innovation_m,
        batch.observation_covariance_m2,
        (batch.state_jacobian, ordinary_nuisance),
        name="observation",
    )
    if anchor_count:
        (
            _,
            (anchor_state_white, anchor_nuisance_white),
            anchor_whiteners,
        ) = _whiten(
            anchor_innovation,
            anchor_covariance,
            (anchor_state, anchor_nuisance),
            name="anchor",
        )
    else:
        anchor_state_white = anchor_state
        anchor_nuisance_white = anchor_nuisance
        anchor_whiteners = np.zeros((0, 3, 3), dtype=np.float64)

    association_probability = np.asarray(batch.association_probability)
    observation_row_weight = batch.prior_reliability * association_probability
    (
        observation_groups,
        observation_indices,
        observation_base,
        observation_prior,
        observation_group_power,
    ) = _group_layout(
        batch.correlation_group_ids,
        observation_row_weight,
        np.asarray(batch.prior_nominal_probability),
        np.asarray(batch.composite_weight),
        cfg.effective_samples_per_correlation_group,
        composite_weight_mode=batch.composite_weight_mode,
    )
    if anchor_count:
        (
            anchor_groups,
            anchor_indices,
            anchor_base,
            anchor_prior,
            anchor_group_power,
        ) = _group_layout(
            anchor_groups_input,
            anchor_reliability,
            anchor_nominal,
            anchor_composite,
            cfg.effective_samples_per_anchor_correlation_group,
            composite_weight_mode=batch.anchor_composite_weight_mode,
        )
    else:
        anchor_groups, anchor_indices = (), ()
        anchor_base = np.zeros(0, dtype=np.float64)
        anchor_prior = np.zeros(0, dtype=np.float64)
        anchor_group_power = np.zeros(0, dtype=np.float64)

    state_mapping = np.asarray(result.identifiable_state_transform)
    retained = state_mapping.shape[1]
    if retained == 0:
        raise ValueError("admissible result has no retained state direction")
    observation_design = np.concatenate(
        (
            np.einsum("mcs,sr->mcr", state_white, state_mapping),
            nuisance_white,
        ),
        axis=2,
    )
    anchor_design = np.concatenate(
        (
            np.einsum("acs,sr->acr", anchor_state_white, state_mapping),
            anchor_nuisance_white,
        ),
        axis=2,
    )

    nuisance_solution = np.concatenate(
        (
            result.gauge_delta,
            result.shared_bias_coefficients,
            result.view_bias_coefficients,
            result.anchor_bias_coefficients,
        )
    )
    full_solution = np.concatenate((result.state_coefficients, nuisance_solution))
    raw_observation_design = np.concatenate(
        (batch.state_jacobian, ordinary_nuisance),
        axis=2,
    )
    raw_anchor_design = np.concatenate(
        (anchor_state, anchor_nuisance),
        axis=2,
    )
    residual = batch.innovation_m - np.einsum(
        "mci,i->mc",
        raw_observation_design,
        full_solution,
    )
    white_residual = np.einsum("mij,mj->mi", whiteners, residual)
    anchor_residual = anchor_innovation - np.einsum(
        "aci,i->ac",
        raw_anchor_design,
        full_solution,
    )
    white_anchor_residual = np.einsum(
        "aij,aj->ai",
        anchor_whiteners,
        anchor_residual,
    )

    observation_precision: FloatArray = np.zeros(
        len(observation_groups), dtype=np.float64
    )
    observation_derivative: FloatArray = np.zeros(
        len(observation_groups), dtype=np.float64
    )
    observation_row_precision: FloatArray = np.zeros(
        len(batch.innovation_m), dtype=np.float64
    )
    for position, selected in enumerate(observation_indices):
        active = selected[observation_row_weight[selected] > 0.0]
        if len(active):
            squared = float(
                np.sum(
                    observation_row_weight[active]
                    * np.sum(np.square(white_residual[active]), axis=1)
                )
            )
            statistics = _student_t_mixture_statistics(
                squared,
                3 * len(active),
                float(observation_prior[position]),
                cfg,
            )
            observation_precision[position] = statistics.expected_precision
            observation_derivative[position] = statistics.expected_precision_derivative
        observation_row_precision[selected] = observation_precision[position]

    anchor_precision: FloatArray = np.zeros(len(anchor_groups), dtype=np.float64)
    anchor_derivative: FloatArray = np.zeros(len(anchor_groups), dtype=np.float64)
    anchor_row_precision: FloatArray = np.zeros(anchor_count, dtype=np.float64)
    for position, selected in enumerate(anchor_indices):
        active = selected[anchor_reliability[selected] > 0.0]
        if len(active):
            squared = float(
                np.sum(
                    anchor_reliability[active]
                    * np.sum(np.square(white_anchor_residual[active]), axis=1)
                )
            )
            statistics = _student_t_mixture_statistics(
                squared,
                3 * len(active),
                float(anchor_prior[position]),
                cfg,
            )
            anchor_precision[position] = statistics.expected_precision
            anchor_derivative[position] = statistics.expected_precision_derivative
        anchor_row_precision[selected] = anchor_precision[position]

    if not np.allclose(
        result.robust_weights,
        observation_row_precision,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("result robust weights do not match the exact solution")
    if not np.allclose(
        result.anchor_robust_weights,
        anchor_row_precision,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("result anchor robust weights do not match the solution")

    state_prior, nuisance_prior, _ = _prior_covariances(batch, cfg)
    reduced_prior = _block_diagonal(
        [np.eye(retained), nuisance_prior] if nuisance_count else [np.eye(retained)]
    )
    prior_precision = _regularized_precision(
        reduced_prior,
        "reduced prior covariance",
        eigenvalue_floor=cfg.prior_eigenvalue_floor,
    )
    ordinary_weight = observation_base * observation_row_precision
    independent_weight = anchor_base * anchor_row_precision
    working_information = prior_precision.copy()
    working_information += np.einsum(
        "m,mci,mcj->ij",
        ordinary_weight,
        observation_design,
        observation_design,
    )
    if anchor_count:
        working_information += np.einsum(
            "a,aci,acj->ij",
            independent_weight,
            anchor_design,
            anchor_design,
        )
    working_information = 0.5 * (working_information + working_information.T)

    working_reduced_covariance = _spd_covariance(working_information)
    working_full_covariance = _full_covariance(
        state_prior,
        state_mapping,
        working_reduced_covariance,
        nuisance_count,
    )
    if not np.allclose(
        working_full_covariance,
        result.posterior_covariance,
        atol=1e-9,
        rtol=1e-9,
    ):
        raise ValueError("working covariance does not reproduce the solver result")

    observed_information = working_information.copy()
    for position, selected in enumerate(observation_indices):
        active = selected[observation_row_weight[selected] > 0.0]
        if not len(active):
            continue
        score_direction = np.einsum(
            "m,mci,mc->i",
            observation_row_weight[active],
            observation_design[active],
            white_residual[active],
        )
        observed_information += (
            2.0
            * observation_group_power[position]
            * observation_derivative[position]
            * np.outer(score_direction, score_direction)
        )
    for position, selected in enumerate(anchor_indices):
        active = selected[anchor_reliability[selected] > 0.0]
        if not len(active):
            continue
        score_direction = np.einsum(
            "a,aci,ac->i",
            anchor_reliability[active],
            anchor_design[active],
            white_anchor_residual[active],
        )
        observed_information += (
            2.0
            * anchor_group_power[position]
            * anchor_derivative[position]
            * np.outer(score_direction, score_direction)
        )
    observed_information = 0.5 * (observed_information + observed_information.T)

    eigenvalues = np.linalg.eigvalsh(observed_information)
    minimum_eigenvalue = float(np.min(eigenvalues))
    maximum_eigenvalue = float(np.max(eigenvalues))
    diagnostic_minimum = result.diagnostics.get(
        "exact_reduced_mixture_hessian_minimum_eigenvalue"
    )
    diagnostic_maximum = result.diagnostics.get(
        "exact_reduced_mixture_hessian_maximum_eigenvalue"
    )
    if diagnostic_minimum is not None:
        diagnostic_minimum_value = float(diagnostic_minimum)
        if (
            not np.isfinite(diagnostic_minimum_value)
            or diagnostic_minimum_value <= 0.0
            or not np.isclose(
                minimum_eigenvalue,
                diagnostic_minimum_value,
                atol=1e-10,
                rtol=1e-10,
            )
        ):
            raise ValueError("reconstructed minimum Hessian eigenvalue changed")
    if diagnostic_maximum is not None:
        diagnostic_maximum_value = float(diagnostic_maximum)
        if (
            not np.isfinite(diagnostic_maximum_value)
            or diagnostic_maximum_value <= 0.0
            or not np.isclose(
                maximum_eigenvalue,
                diagnostic_maximum_value,
                atol=1e-10,
                rtol=1e-10,
            )
        ):
            raise ValueError("reconstructed maximum Hessian eigenvalue changed")
    if minimum_eigenvalue <= 0.0:
        raise ValueError("exact observed mixture information is not positive definite")
    condition_number = float(np.linalg.cond(observed_information))
    if (
        not np.isfinite(condition_number)
        or condition_number > cfg.maximum_condition_number
    ):
        raise ValueError("exact observed mixture information is ill-conditioned")

    reduced_covariance = _spd_covariance(observed_information)
    full_covariance = _full_covariance(
        state_prior,
        state_mapping,
        reduced_covariance,
        nuisance_count,
    )
    semantics_metadata = {
        "source_objective": "prior-aware-group-mixture-v1",
        "reduced_dimension": len(observed_information),
        "retained_state_dimension": retained,
        "observation_group_count": len(observation_groups),
        "anchor_group_count": len(anchor_groups),
        "working_covariance_parity_verified": True,
        "minimum_robust_precision": cfg.minimum_robust_precision,
    }
    semantics = PosteriorCovarianceSemanticsV1(
        method="laplace_observed_information",
        dimension=len(full_covariance),
        likelihood_power_semantics=LIKELIHOOD_POWER_SEMANTICS,
        prior_included=True,
        generalized_bayes=True,
        mixture_curvature_exact=True,
        group_score_correction=False,
        calibrated=False,
        metadata=semantics_metadata,
    )
    result_metadata = {
        **({} if metadata is None else dict(metadata)),
        "input_lineage": plain_json(result.input_lineage),
        "working_covariance_kind": result.diagnostics.get("posterior_covariance_kind"),
        "exact_objective": result.diagnostics.get("robust_likelihood_objective"),
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": maximum_eigenvalue,
    }
    return ObservedInformationCovarianceResultV1(
        working_information=working_information,
        observed_information=observed_information,
        reduced_covariance=reduced_covariance,
        full_covariance=full_covariance,
        state_prior_covariance=state_prior,
        state_mapping=state_mapping,
        observation_group_ids=observation_groups,
        anchor_group_ids=anchor_groups,
        observation_group_power=observation_group_power,
        anchor_group_power=anchor_group_power,
        observation_group_expected_precision=observation_precision,
        anchor_group_expected_precision=anchor_precision,
        observation_group_precision_derivative=observation_derivative,
        anchor_group_precision_derivative=anchor_derivative,
        prior_eigenvalue_floor=cfg.prior_eigenvalue_floor,
        condition_number=condition_number,
        covariance_semantics=semantics,
        metadata=result_metadata,
    )


__all__ = [
    "LIKELIHOOD_POWER_SEMANTICS",
    "OBSERVED_INFORMATION_COVARIANCE_SCHEMA",
    "OBSERVED_INFORMATION_COVARIANCE_VERSION",
    "ObservedInformationCovarianceResultV1",
    "observed_information_covariance_from_prior_aware_result",
]
