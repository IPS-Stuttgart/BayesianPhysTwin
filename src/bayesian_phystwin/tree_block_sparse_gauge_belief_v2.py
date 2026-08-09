"""Strict admission for the exact block-tree gauge solver.

The version-1 block-tree solver remains available as a numerical reproduction
surface. This module adds a closed, fail-closed admission boundary for
prospective and claim-bearing callers. Every rejection reconstructs the exact
physical and tree-gauge prior instead of trusting candidate corrections or a
candidate covariance returned by the underlying solver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np

from ._gauge_aware_contracts import GaugeAwareObservationBatch, _require
from ._prior_aware_gauge_math import PriorAwareGaugeConfigV1, _whiten
from .sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
    _whiten_sparse_observations,
)
from .tree_block_sparse_gauge_belief import (
    TreeBlockGaugeAwareBeliefResultV1,
    TreeBlockPosteriorCovarianceV1,
    _fallback_result,
    _prior_covariance,
    update_tree_block_sparse_prior_aware_gauge_belief,
)

TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_SCHEMA: Final = (
    "bayesian_phystwin.tree_block_sparse_gauge_belief"
)
TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_VERSION: Final = 2
TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION: Final = (
    "tree-block-group-mixture-strict-admission-v2"
)
TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_CLAIM_BOUNDARY: Final = (
    "Prospective numerical-admission implementation only. It preserves the exact "
    "physical and tree-gauge prior for every rejection and validates one closed "
    "admission certificate against the retained solver diagnostics. It does not "
    "establish provider competence, uncertainty calibration, physical-query "
    "benefit, intervention benefit, deployment safety, or state of the art."
)

_EXACT_MIXTURE_OBJECTIVE: Final = "exact-group-mixture-gradient"
_EXACT_POSTERIOR_SOLVER: Final = "tree-block-leaf-schur-cholesky-v1"
_PASS_REASON: Final = "strict-admission-passed"
_UNDERLYING_REJECTION_REASON: Final = "underlying-inference-rejected"
_FIXED_POINT_REASON: Final = "strict-v2-fixed-point-not-converged"
_NONEXACT_REASON: Final = "strict-v2-non-exact-mixture-objective"
_INVALID_DIAGNOSTICS_REASON: Final = "strict-v2-invalid-admission-diagnostics"
_CERTIFICATE_SCHEMA: Final = (
    "bayesian_phystwin.tree_block_strict_admission_certificate"
)
_CERTIFICATE_VERSION: Final = 1
_CERTIFICATE_KEY: Final = "strict_admission_certificate"
_CERTIFICATE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "underlying_inference_admissible",
        "underlying_inference_reason",
        "exact_mixture_objective",
        "fixed_point_converged",
        "exact_tree_block_solver",
        "diagnostics_valid",
        "mixture_solution_delta",
        "mixture_stationarity_norm",
        "maximum_eliminated_node_condition_number",
        "global_schur_condition_number",
        "passed",
        "reason",
    }
)


class TreeBlockGaugeAwareBeliefResultV2(TreeBlockGaugeAwareBeliefResultV1):
    """Tree-block result bound to one reconstructed strict-v2 certificate."""

    __slots__ = ()

    implementation_schema: Final = TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_SCHEMA
    implementation_version: Final = TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_VERSION
    implementation_id: Final = TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_tagged_certificate_diagnostics(
            self.diagnostics,
            inference_admissible=self.inference_admissible,
            result_reason=self.reason,
        )


@dataclass(frozen=True)
class _TreeBlockAdmissionCertificateV1:
    underlying_inference_admissible: bool
    underlying_inference_reason: str
    exact_mixture_objective: bool
    fixed_point_converged: bool
    exact_tree_block_solver: bool
    diagnostics_valid: bool
    mixture_solution_delta: float | None
    mixture_stationarity_norm: float | None
    maximum_eliminated_node_condition_number: float | None
    global_schur_condition_number: float | None
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        expected_reason = _admission_reason(
            underlying_inference_admissible=self.underlying_inference_admissible,
            exact_mixture_objective=self.exact_mixture_objective,
            fixed_point_converged=self.fixed_point_converged,
            diagnostics_valid=self.diagnostics_valid,
        )
        _require(
            self.reason == expected_reason,
            "tree-block admission certificate reason violates its decision invariant",
        )
        _require(
            self.passed == (expected_reason == _PASS_REASON),
            "tree-block admission certificate pass flag violates its decision invariant",
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": _CERTIFICATE_SCHEMA,
            "schema_version": _CERTIFICATE_VERSION,
            "underlying_inference_admissible": self.underlying_inference_admissible,
            "underlying_inference_reason": self.underlying_inference_reason,
            "exact_mixture_objective": self.exact_mixture_objective,
            "fixed_point_converged": self.fixed_point_converged,
            "exact_tree_block_solver": self.exact_tree_block_solver,
            "diagnostics_valid": self.diagnostics_valid,
            "mixture_solution_delta": self.mixture_solution_delta,
            "mixture_stationarity_norm": self.mixture_stationarity_norm,
            "maximum_eliminated_node_condition_number": (
                self.maximum_eliminated_node_condition_number
            ),
            "global_schur_condition_number": self.global_schur_condition_number,
            "passed": self.passed,
            "reason": self.reason,
        }


def _finite_nonnegative(
    diagnostics: Mapping[str, object],
    name: str,
) -> float | None:
    value = diagnostics.get(name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        return None
    return result


def _finite_positive(
    diagnostics: Mapping[str, object],
    name: str,
) -> float | None:
    result = _finite_nonnegative(diagnostics, name)
    return result if result is not None and result > 0.0 else None


def _admission_reason(
    *,
    underlying_inference_admissible: bool,
    exact_mixture_objective: bool,
    fixed_point_converged: bool,
    diagnostics_valid: bool,
) -> str:
    if not underlying_inference_admissible:
        return _UNDERLYING_REJECTION_REASON
    if not exact_mixture_objective:
        return _NONEXACT_REASON
    if not fixed_point_converged:
        return _FIXED_POINT_REASON
    if not diagnostics_valid:
        return _INVALID_DIAGNOSTICS_REASON
    return _PASS_REASON


def _build_admission_certificate(
    result: TreeBlockGaugeAwareBeliefResultV1,
) -> _TreeBlockAdmissionCertificateV1:
    diagnostics = result.diagnostics
    exact_objective = (
        diagnostics.get("robust_likelihood_objective") == _EXACT_MIXTURE_OBJECTIVE
    )
    fixed_point_converged = (
        diagnostics.get("mixture_fixed_point_converged") is True
    )
    exact_solver = diagnostics.get("posterior_solver") == _EXACT_POSTERIOR_SOLVER
    solution_delta = _finite_nonnegative(diagnostics, "mixture_solution_delta")
    stationarity = _finite_nonnegative(
        diagnostics,
        "mixture_stationarity_norm",
    )
    maximum_eliminated_condition = _finite_positive(
        diagnostics,
        "maximum_eliminated_node_condition_number",
    )
    global_condition = _finite_positive(
        diagnostics,
        "global_schur_condition_number",
    )
    diagnostics_valid = bool(
        exact_solver
        and solution_delta is not None
        and stationarity is not None
        and maximum_eliminated_condition is not None
        and global_condition is not None
    )
    reason = _admission_reason(
        underlying_inference_admissible=result.inference_admissible,
        exact_mixture_objective=exact_objective,
        fixed_point_converged=fixed_point_converged,
        diagnostics_valid=diagnostics_valid,
    )
    return _TreeBlockAdmissionCertificateV1(
        underlying_inference_admissible=result.inference_admissible,
        underlying_inference_reason=result.reason,
        exact_mixture_objective=exact_objective,
        fixed_point_converged=fixed_point_converged,
        exact_tree_block_solver=exact_solver,
        diagnostics_valid=diagnostics_valid,
        mixture_solution_delta=solution_delta,
        mixture_stationarity_norm=stationarity,
        maximum_eliminated_node_condition_number=maximum_eliminated_condition,
        global_schur_condition_number=global_condition,
        passed=reason == _PASS_REASON,
        reason=reason,
    )


def _certificate_optional_real(
    certificate: Mapping[str, object],
    name: str,
) -> float | None:
    value = certificate.get(name)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            f"tree-block admission certificate field {name!r} "
            "must be a finite real or null"
        )
    result = float(value)
    _require(
        np.isfinite(result),
        f"tree-block admission certificate field {name!r} must be finite",
    )
    return result


def _validate_certificate_mapping(certificate: Mapping[str, object]) -> bool:
    _require(
        set(certificate) == _CERTIFICATE_FIELDS,
        "tree-block admission certificate fields changed",
    )
    _require(
        type(certificate.get("schema")) is str
        and certificate.get("schema") == _CERTIFICATE_SCHEMA,
        "tree-block admission certificate has an unsupported schema",
    )
    schema_version = certificate.get("schema_version")
    _require(
        type(schema_version) is int and schema_version == _CERTIFICATE_VERSION,
        "tree-block admission certificate has an unsupported schema_version",
    )
    underlying_reason = certificate.get("underlying_inference_reason")
    _require(
        type(underlying_reason) is str and bool(underlying_reason),
        "tree-block admission certificate underlying reason must be nonempty text",
    )
    for name in (
        "underlying_inference_admissible",
        "exact_mixture_objective",
        "fixed_point_converged",
        "exact_tree_block_solver",
        "diagnostics_valid",
    ):
        _require(
            type(certificate.get(name)) is bool,
            f"tree-block admission certificate field {name!r} must be a bool",
        )

    solution_delta = _certificate_optional_real(
        certificate,
        "mixture_solution_delta",
    )
    stationarity = _certificate_optional_real(
        certificate,
        "mixture_stationarity_norm",
    )
    maximum_eliminated_condition = _certificate_optional_real(
        certificate,
        "maximum_eliminated_node_condition_number",
    )
    global_condition = _certificate_optional_real(
        certificate,
        "global_schur_condition_number",
    )
    expected_diagnostics_valid = bool(
        certificate["exact_tree_block_solver"] is True
        and solution_delta is not None
        and solution_delta >= 0.0
        and stationarity is not None
        and stationarity >= 0.0
        and maximum_eliminated_condition is not None
        and maximum_eliminated_condition > 0.0
        and global_condition is not None
        and global_condition > 0.0
    )
    _require(
        certificate["diagnostics_valid"] is expected_diagnostics_valid,
        "tree-block admission certificate diagnostic invariant is inconsistent",
    )
    expected_reason = _admission_reason(
        underlying_inference_admissible=bool(
            certificate["underlying_inference_admissible"]
        ),
        exact_mixture_objective=bool(certificate["exact_mixture_objective"]),
        fixed_point_converged=bool(certificate["fixed_point_converged"]),
        diagnostics_valid=expected_diagnostics_valid,
    )
    expected_passed = expected_reason == _PASS_REASON
    _require(
        type(certificate.get("passed")) is bool
        and certificate.get("passed") == expected_passed,
        "tree-block admission certificate pass invariant is inconsistent",
    )
    _require(
        type(certificate.get("reason")) is str
        and certificate.get("reason") == expected_reason,
        "tree-block admission certificate reason invariant is inconsistent",
    )
    return expected_passed


def _validate_tagged_certificate_diagnostics(
    diagnostics: Mapping[str, object],
    *,
    inference_admissible: bool,
    result_reason: str,
) -> bool:
    certificate = diagnostics.get(_CERTIFICATE_KEY)
    if not isinstance(certificate, Mapping):
        raise ValueError("tree-block diagnostics do not contain an admission certificate")
    certificate_passed = _validate_certificate_mapping(certificate)

    raw_exact_objective = (
        diagnostics.get("robust_likelihood_objective") == _EXACT_MIXTURE_OBJECTIVE
    )
    raw_fixed_point = diagnostics.get("mixture_fixed_point_converged") is True
    raw_exact_solver = diagnostics.get("posterior_solver") == _EXACT_POSTERIOR_SOLVER
    raw_solution_delta = _finite_nonnegative(
        diagnostics,
        "mixture_solution_delta",
    )
    raw_stationarity = _finite_nonnegative(
        diagnostics,
        "mixture_stationarity_norm",
    )
    raw_maximum_eliminated_condition = _finite_positive(
        diagnostics,
        "maximum_eliminated_node_condition_number",
    )
    raw_global_condition = _finite_positive(
        diagnostics,
        "global_schur_condition_number",
    )
    expected_raw = {
        "exact_mixture_objective": raw_exact_objective,
        "fixed_point_converged": raw_fixed_point,
        "exact_tree_block_solver": raw_exact_solver,
        "mixture_solution_delta": raw_solution_delta,
        "mixture_stationarity_norm": raw_stationarity,
        "maximum_eliminated_node_condition_number": (
            raw_maximum_eliminated_condition
        ),
        "global_schur_condition_number": raw_global_condition,
    }
    for name, expected in expected_raw.items():
        _require(
            certificate[name] == expected,
            f"tree-block admission certificate field {name!r} "
            "differs from retained solver diagnostics",
        )

    for name in (
        "strict_admission_passed",
        "strict_admission_reason",
        "underlying_inference_admissible",
        "underlying_inference_reason",
    ):
        certificate_name = {
            "strict_admission_passed": "passed",
            "strict_admission_reason": "reason",
            "underlying_inference_admissible": "underlying_inference_admissible",
            "underlying_inference_reason": "underlying_inference_reason",
        }[name]
        _require(
            diagnostics.get(name) == certificate[certificate_name],
            f"tree-block diagnostic {name!r} differs from its certificate",
        )

    _require(
        type(inference_admissible) is bool
        and inference_admissible == certificate_passed,
        "tree-block inference admissibility differs from its certificate",
    )
    underlying_admissible = certificate["underlying_inference_admissible"] is True
    expected_result_reason = (
        certificate["underlying_inference_reason"]
        if certificate_passed or not underlying_admissible
        else certificate["reason"]
    )
    _require(
        type(result_reason) is str and result_reason == expected_result_reason,
        "tree-block result reason differs from its selected certificate path",
    )
    return certificate_passed


def _strict_failure(diagnostics: Mapping[str, object]) -> str | None:
    if diagnostics.get("robust_likelihood_objective") != _EXACT_MIXTURE_OBJECTIVE:
        return _NONEXACT_REASON
    if diagnostics.get("mixture_fixed_point_converged") is not True:
        return _FIXED_POINT_REASON
    if diagnostics.get("posterior_solver") != _EXACT_POSTERIOR_SOLVER:
        return _INVALID_DIAGNOSTICS_REASON
    if (
        _finite_nonnegative(diagnostics, "mixture_solution_delta") is None
        or _finite_nonnegative(diagnostics, "mixture_stationarity_norm") is None
        or _finite_positive(
            diagnostics,
            "maximum_eliminated_node_condition_number",
        )
        is None
        or _finite_positive(diagnostics, "global_schur_condition_number") is None
    ):
        return _INVALID_DIAGNOSTICS_REASON
    return None


def _tag_diagnostics(
    diagnostics: Mapping[str, object],
    *,
    certificate: _TreeBlockAdmissionCertificateV1,
) -> dict[str, object]:
    tagged = dict(diagnostics)
    tagged.update(
        {
            "implementation_schema": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_SCHEMA,
            "implementation_version": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_VERSION,
            "implementation_id": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION,
            "strict_admission_version": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_VERSION,
            "strict_admission_passed": certificate.passed,
            "strict_admission_reason": certificate.reason,
            "underlying_inference_admissible": (
                certificate.underlying_inference_admissible
            ),
            "underlying_inference_reason": certificate.underlying_inference_reason,
            _CERTIFICATE_KEY: certificate.as_mapping(),
            "exact_mixture_objective_required": True,
            "fixed_point_convergence_required": True,
            "finite_stationarity_diagnostics_required": True,
            "positive_cholesky_condition_diagnostics_required": True,
            "implicit_jitter": False,
            "eigenvalue_clipping": False,
            "pseudoinverse_fallback": False,
            "claim_boundary": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_CLAIM_BOUNDARY,
        }
    )
    return tagged


def _as_v2(
    result: TreeBlockGaugeAwareBeliefResultV1,
    diagnostics: Mapping[str, object],
) -> TreeBlockGaugeAwareBeliefResultV2:
    return TreeBlockGaugeAwareBeliefResultV2(
        inference_admissible=result.inference_admissible,
        reason=result.reason,
        state_coefficients=result.state_coefficients,
        gauge_delta=result.gauge_delta,
        shared_bias_coefficients=result.shared_bias_coefficients,
        view_bias_coefficients=result.view_bias_coefficients,
        anchor_bias_coefficients=result.anchor_bias_coefficients,
        covariance=result.covariance,
        identifiable_state_transform=result.identifiable_state_transform,
        identifiable_fractions=result.identifiable_fractions,
        query_sensitivity_fractions=result.query_sensitivity_fractions,
        robust_weights=result.robust_weights,
        anchor_robust_weights=result.anchor_robust_weights,
        diagnostics=diagnostics,
        input_lineage=result.input_lineage,
    )


def _exact_prior_covariance(
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    config: PriorAwareGaugeConfigV1,
) -> TreeBlockPosteriorCovarianceV1:
    (
        _,
        state_white,
        local_gauge_white,
        shared_white,
        view_white,
        _,
    ) = _whiten_sparse_observations(batch, gauge)
    state_count = batch.state_jacobian.shape[2]
    anchor_count = (
        0 if batch.anchor_innovation_m is None else len(batch.anchor_innovation_m)
    )
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    if anchor_count:
        if batch.anchor_innovation_m is None:
            raise ValueError("validated anchor innovation is missing")
        if batch.anchor_covariance_m2 is None:
            raise ValueError("validated anchor covariance is missing")
        if batch.anchor_state_jacobian is None:
            raise ValueError("validated anchor state Jacobian is missing")
        anchor_bias = (
            np.zeros((anchor_count, 3, anchor_bias_count), dtype=np.float64)
            if batch.anchor_bias_jacobian is None
            else np.asarray(batch.anchor_bias_jacobian)
        )
        (
            _,
            (anchor_state_white, anchor_bias_white),
            _,
        ) = _whiten(
            np.asarray(batch.anchor_innovation_m),
            np.asarray(batch.anchor_covariance_m2),
            (np.asarray(batch.anchor_state_jacobian), anchor_bias),
            name="anchor",
        )
    else:
        anchor_state_white = np.zeros((0, 3, state_count), dtype=np.float64)
        anchor_bias_white = np.zeros(
            (0, 3, anchor_bias_count),
            dtype=np.float64,
        )
    state_prior = (
        np.eye(state_count, dtype=np.float64) * config.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2)
    )
    return _prior_covariance(
        batch=batch,
        gauge=gauge,
        state_prior=state_prior,
        state_design=state_white,
        local_gauge=local_gauge_white,
        shared=shared_white,
        view=view_white,
        anchor_state=anchor_state_white,
        anchor_bias=anchor_bias_white,
        config=config,
    )


def _exact_fallback(
    *,
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    config: PriorAwareGaugeConfigV1,
    reason: str,
    diagnostics: Mapping[str, object],
) -> TreeBlockGaugeAwareBeliefResultV2:
    fallback = _fallback_result(
        batch=batch,
        gauge=gauge,
        reason=reason,
        diagnostics=diagnostics,
        covariance=_exact_prior_covariance(batch, gauge, config),
    )
    return _as_v2(fallback, diagnostics)


def update_tree_block_sparse_prior_aware_gauge_belief_v2(
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> TreeBlockGaugeAwareBeliefResultV2:
    """Run block-tree inference and fail closed through one certificate."""

    if not isinstance(batch, GaugeAwareObservationBatch):
        raise TypeError("batch must be a GaugeAwareObservationBatch")
    if not isinstance(gauge, TreeSparseGaugeDesignV1):
        raise TypeError("gauge must be a TreeSparseGaugeDesignV1")
    if config is not None and not isinstance(config, PriorAwareGaugeConfigV1):
        raise TypeError("config must be a PriorAwareGaugeConfigV1")
    cfg = config or PriorAwareGaugeConfigV1()
    result = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=cfg,
    )
    certificate = _build_admission_certificate(result)
    diagnostics = _tag_diagnostics(
        result.diagnostics,
        certificate=certificate,
    )
    if certificate.passed:
        return _as_v2(result, diagnostics)
    fallback_reason = (
        result.reason
        if not certificate.underlying_inference_admissible
        else certificate.reason
    )
    return _exact_fallback(
        batch=batch,
        gauge=gauge,
        config=cfg,
        reason=fallback_reason,
        diagnostics=diagnostics,
    )


__all__ = [
    "TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_CLAIM_BOUNDARY",
    "TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION",
    "TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_SCHEMA",
    "TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_VERSION",
    "TreeBlockGaugeAwareBeliefResultV2",
    "update_tree_block_sparse_prior_aware_gauge_belief_v2",
]
