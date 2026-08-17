"""Prospective strict admission for prior-aware grouped-mixture updates.

The version-1 dense and native-sparse solvers remain unchanged for historical
reproduction. This module calls those solvers and then applies one additional,
fail-closed admission boundary before an update can be returned as admissible.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Final, TypeAlias

import numpy as np

from ._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    _fallback_result,
    _require,
)
from ._prior_aware_gauge_math import (
    PriorAwareGaugeConfigV1,
    _prior_covariances,
)
from .prior_aware_gauge_belief import update_prior_aware_gauge_belief
from .sparse_prior_aware_gauge_belief import (
    SparseGaugeDesignV1,
    TreeSparseGaugeDesignV1,
    _sparse_fallback_result,
    update_sparse_prior_aware_gauge_belief,
    update_sparse_prior_aware_gauge_belief_structured,
)
from .sparse_prior_aware_gauge_belief import (
    _prior_covariances as _sparse_prior_covariances,
)
from .structured_gauge_aware_result import StructuredGaugeAwareBeliefResultV1

PRIOR_AWARE_GAUGE_BELIEF_V2_SCHEMA: Final = "bayesian_phystwin.prior_aware_gauge_belief"
PRIOR_AWARE_GAUGE_BELIEF_V2_VERSION: Final = 2
PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION: Final = (
    "prior-aware-group-mixture-strict-admission-v2"
)
PRIOR_AWARE_GAUGE_BELIEF_V2_CLAIM_BOUNDARY: Final = (
    "Prospective numerical-admission implementation only. It does not alter "
    "historical v1 evidence and requires a separately frozen protocol, guard, "
    "calibration, and target-access record before claim-bearing use."
)

_EXACT_MIXTURE_OBJECTIVE: Final = "exact-group-mixture-gradient"
_PASS_REASON: Final = "strict-admission-passed"
_UNDERLYING_REJECTION_REASON: Final = "underlying-inference-rejected"
_FIXED_POINT_REASON: Final = "strict-v2-fixed-point-not-converged"
_NONEXACT_REASON: Final = "strict-v2-non-exact-mixture-objective"
_INVALID_DIAGNOSTICS_REASON: Final = "strict-v2-invalid-admission-diagnostics"
_NONPOSITIVE_CURVATURE_REASON: Final = "strict-v2-non-positive-exact-mixture-curvature"
_ILL_CONDITIONED_CURVATURE_REASON: Final = (
    "strict-v2-ill-conditioned-exact-mixture-curvature"
)
_CERTIFICATE_SCHEMA: Final = "bayesian_phystwin.prior_aware_gauge_admission_certificate"
_CERTIFICATE_VERSION: Final = 1
_CERTIFICATE_KEY: Final = "strict_admission_certificate"
_CERTIFICATE_BOOL_FIELDS: Final = (
    "underlying_inference_admissible",
    "exact_mixture_objective",
    "fixed_point_converged",
    "diagnostics_valid",
    "positive_exact_mixture_curvature",
    "condition_number_within_limit",
)
_CERTIFICATE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "underlying_inference_admissible",
        "underlying_inference_reason",
        "exact_mixture_objective",
        "fixed_point_converged",
        "diagnostics_valid",
        "positive_exact_mixture_curvature",
        "condition_number_within_limit",
        "mixture_solution_delta",
        "mixture_stationarity_norm",
        "exact_hessian_minimum_eigenvalue",
        "exact_hessian_maximum_eigenvalue",
        "exact_hessian_condition_number",
        "maximum_exact_hessian_condition_number",
        "passed",
        "reason",
    }
)

GaugeDesignV1: TypeAlias = SparseGaugeDesignV1 | TreeSparseGaugeDesignV1
AdmissionInputResult: TypeAlias = (
    GaugeAwareBeliefResult | StructuredGaugeAwareBeliefResultV1
)


@dataclass(frozen=True)
class PriorAwareGaugeAdmissionConfigV2:
    """Strict post-solver admission settings for the prospective v2 path."""

    maximum_exact_hessian_condition_number: float = 1.0e14

    def __post_init__(self) -> None:
        raw = self.maximum_exact_hessian_condition_number
        _require(
            not isinstance(raw, (bool, np.bool_)) and isinstance(raw, Real),
            "maximum exact-Hessian condition number must be a real number",
        )
        value = float(raw)
        _require(
            np.isfinite(value) and value >= 1.0,
            "maximum exact-Hessian condition number must be finite and at least one",
        )
        object.__setattr__(self, "maximum_exact_hessian_condition_number", value)


@dataclass(frozen=True)
class _PriorAwareGaugeAdmissionCertificateV1:
    """One immutable decision record shared by every strict-v2 solver path."""

    underlying_inference_admissible: bool
    underlying_inference_reason: str
    exact_mixture_objective: bool
    fixed_point_converged: bool
    diagnostics_valid: bool
    positive_exact_mixture_curvature: bool
    condition_number_within_limit: bool
    mixture_solution_delta: float | None
    mixture_stationarity_norm: float | None
    exact_hessian_minimum_eigenvalue: float | None
    exact_hessian_maximum_eigenvalue: float | None
    exact_hessian_condition_number: float | None
    maximum_exact_hessian_condition_number: float
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        expected_reason = _admission_reason(
            underlying_inference_admissible=(self.underlying_inference_admissible),
            exact_mixture_objective=self.exact_mixture_objective,
            fixed_point_converged=self.fixed_point_converged,
            diagnostics_valid=self.diagnostics_valid,
            positive_exact_mixture_curvature=(self.positive_exact_mixture_curvature),
            condition_number_within_limit=self.condition_number_within_limit,
        )
        _require(
            self.reason == expected_reason,
            "strict admission certificate reason violates the decision invariant",
        )
        _require(
            self.passed == (expected_reason == _PASS_REASON),
            "strict admission certificate pass flag violates the decision invariant",
        )

    def as_mapping(self) -> dict[str, object]:
        """Return the finite JSON payload embedded in result diagnostics."""

        return {
            "schema": _CERTIFICATE_SCHEMA,
            "schema_version": _CERTIFICATE_VERSION,
            "underlying_inference_admissible": (self.underlying_inference_admissible),
            "underlying_inference_reason": self.underlying_inference_reason,
            "exact_mixture_objective": self.exact_mixture_objective,
            "fixed_point_converged": self.fixed_point_converged,
            "diagnostics_valid": self.diagnostics_valid,
            "positive_exact_mixture_curvature": (self.positive_exact_mixture_curvature),
            "condition_number_within_limit": self.condition_number_within_limit,
            "mixture_solution_delta": self.mixture_solution_delta,
            "mixture_stationarity_norm": self.mixture_stationarity_norm,
            "exact_hessian_minimum_eigenvalue": (self.exact_hessian_minimum_eigenvalue),
            "exact_hessian_maximum_eigenvalue": (self.exact_hessian_maximum_eigenvalue),
            "exact_hessian_condition_number": (self.exact_hessian_condition_number),
            "maximum_exact_hessian_condition_number": (
                self.maximum_exact_hessian_condition_number
            ),
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PriorAwareGaugeBeliefResultV2(GaugeAwareBeliefResult):
    """Gauge-aware result explicitly bound to strict prospective admission."""

    implementation_schema: str = field(
        init=False,
        default=PRIOR_AWARE_GAUGE_BELIEF_V2_SCHEMA,
    )
    implementation_version: int = field(
        init=False,
        default=PRIOR_AWARE_GAUGE_BELIEF_V2_VERSION,
    )
    implementation_id: str = field(
        init=False,
        default=PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _require(
            self.diagnostics.get("implementation_id") == self.implementation_id,
            "v2 diagnostics do not identify the v2 implementation",
        )
        _require(
            self.diagnostics.get("strict_admission_version")
            == self.implementation_version,
            "v2 diagnostics do not identify strict admission version 2",
        )
        certificate_passed = _validate_tagged_certificate_diagnostics(
            self.diagnostics,
            inference_admissible=self.inference_admissible,
            result_reason=self.reason,
        )
        if not certificate_passed:
            for name in (
                "state_coefficients",
                "gauge_delta",
                "shared_bias_coefficients",
                "view_bias_coefficients",
                "anchor_bias_coefficients",
            ):
                _require(
                    np.count_nonzero(getattr(self, name)) == 0,
                    "rejected v2 results must preserve zero candidate coefficients",
                )


def _finite_diagnostic(
    diagnostics: Mapping[str, object],
    name: str,
) -> float | None:
    value = diagnostics.get(name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _admission_reason(
    *,
    underlying_inference_admissible: bool,
    exact_mixture_objective: bool,
    fixed_point_converged: bool,
    diagnostics_valid: bool,
    positive_exact_mixture_curvature: bool,
    condition_number_within_limit: bool,
) -> str:
    if not underlying_inference_admissible:
        return _UNDERLYING_REJECTION_REASON
    if not exact_mixture_objective:
        return _NONEXACT_REASON
    if not fixed_point_converged:
        return _FIXED_POINT_REASON
    if not diagnostics_valid:
        return _INVALID_DIAGNOSTICS_REASON
    if not positive_exact_mixture_curvature:
        return _NONPOSITIVE_CURVATURE_REASON
    if not condition_number_within_limit:
        return _ILL_CONDITIONED_CURVATURE_REASON
    return _PASS_REASON


def _build_admission_certificate(
    result: AdmissionInputResult,
    admission: PriorAwareGaugeAdmissionConfigV2,
) -> _PriorAwareGaugeAdmissionCertificateV1:
    diagnostics = result.diagnostics
    exact_objective = (
        diagnostics.get("robust_likelihood_objective") == _EXACT_MIXTURE_OBJECTIVE
    )
    fixed_point_converged = diagnostics.get("mixture_fixed_point_converged") is True
    solution_delta = _finite_diagnostic(diagnostics, "mixture_solution_delta")
    stationarity = _finite_diagnostic(diagnostics, "mixture_stationarity_norm")
    minimum = _finite_diagnostic(
        diagnostics,
        "exact_reduced_mixture_hessian_minimum_eigenvalue",
    )
    maximum = _finite_diagnostic(
        diagnostics,
        "exact_reduced_mixture_hessian_maximum_eigenvalue",
    )
    declared_positive = diagnostics.get(
        "exact_reduced_mixture_hessian_positive_definite"
    )
    diagnostics_valid = (
        solution_delta is not None
        and solution_delta >= 0.0
        and stationarity is not None
        and stationarity >= 0.0
        and minimum is not None
        and maximum is not None
        and maximum >= minimum
        and type(declared_positive) is bool
        and declared_positive == (minimum > 0.0)
    )
    condition_number: float | None = None
    if (
        diagnostics_valid
        and minimum is not None
        and maximum is not None
        and minimum > 0.0
    ):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            candidate_condition = float(np.divide(maximum, minimum))
        if np.isfinite(candidate_condition):
            condition_number = candidate_condition
        else:
            diagnostics_valid = False
    positive_curvature = bool(
        diagnostics_valid and minimum is not None and minimum > 0.0
    )
    condition_number_within_limit = bool(
        positive_curvature
        and condition_number is not None
        and condition_number <= admission.maximum_exact_hessian_condition_number
    )
    reason = _admission_reason(
        underlying_inference_admissible=result.inference_admissible,
        exact_mixture_objective=exact_objective,
        fixed_point_converged=fixed_point_converged,
        diagnostics_valid=diagnostics_valid,
        positive_exact_mixture_curvature=positive_curvature,
        condition_number_within_limit=condition_number_within_limit,
    )
    return _PriorAwareGaugeAdmissionCertificateV1(
        underlying_inference_admissible=result.inference_admissible,
        underlying_inference_reason=result.reason,
        exact_mixture_objective=exact_objective,
        fixed_point_converged=fixed_point_converged,
        diagnostics_valid=diagnostics_valid,
        positive_exact_mixture_curvature=positive_curvature,
        condition_number_within_limit=condition_number_within_limit,
        mixture_solution_delta=solution_delta,
        mixture_stationarity_norm=stationarity,
        exact_hessian_minimum_eigenvalue=minimum,
        exact_hessian_maximum_eigenvalue=maximum,
        exact_hessian_condition_number=condition_number,
        maximum_exact_hessian_condition_number=(
            admission.maximum_exact_hessian_condition_number
        ),
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
            f"v2 admission certificate field {name!r} must be a finite real or null"
        )
    result = float(value)
    _require(
        np.isfinite(result),
        f"v2 admission certificate field {name!r} must be finite",
    )
    return result


def _validate_certificate_mapping(certificate: Mapping[str, object]) -> bool:
    _require(
        set(certificate) == _CERTIFICATE_FIELDS,
        "v2 admission certificate fields changed",
    )
    _require(
        type(certificate.get("schema")) is str
        and certificate.get("schema") == _CERTIFICATE_SCHEMA,
        "v2 admission certificate has an unsupported schema",
    )
    schema_version = certificate.get("schema_version")
    _require(
        type(schema_version) is int and schema_version == _CERTIFICATE_VERSION,
        "v2 admission certificate has an unsupported schema_version",
    )
    underlying_reason = certificate.get("underlying_inference_reason")
    _require(
        type(underlying_reason) is str and bool(underlying_reason),
        "v2 admission certificate underlying reason must be nonempty text",
    )
    for name in _CERTIFICATE_BOOL_FIELDS:
        _require(
            type(certificate.get(name)) is bool,
            f"v2 admission certificate field {name!r} must be a bool",
        )

    solution_delta = _certificate_optional_real(
        certificate,
        "mixture_solution_delta",
    )
    stationarity = _certificate_optional_real(
        certificate,
        "mixture_stationarity_norm",
    )
    minimum = _certificate_optional_real(
        certificate,
        "exact_hessian_minimum_eigenvalue",
    )
    maximum = _certificate_optional_real(
        certificate,
        "exact_hessian_maximum_eigenvalue",
    )
    condition_number = _certificate_optional_real(
        certificate,
        "exact_hessian_condition_number",
    )
    maximum_condition_number = _certificate_optional_real(
        certificate,
        "maximum_exact_hessian_condition_number",
    )
    if maximum_condition_number is None or maximum_condition_number < 1.0:
        raise ValueError(
            "v2 admission certificate maximum condition number must be finite and at least one"
        )

    diagnostics_valid = certificate["diagnostics_valid"] is True
    if diagnostics_valid:
        _require(
            solution_delta is not None
            and solution_delta >= 0.0
            and stationarity is not None
            and stationarity >= 0.0
            and minimum is not None
            and maximum is not None
            and maximum >= minimum,
            "v2 admission certificate numeric diagnostics are inconsistent",
        )
    expected_positive_curvature = bool(
        diagnostics_valid and minimum is not None and minimum > 0.0
    )
    _require(
        certificate["positive_exact_mixture_curvature"] is expected_positive_curvature,
        "v2 admission certificate curvature invariant is inconsistent",
    )

    if expected_positive_curvature:
        assert minimum is not None
        assert maximum is not None
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            expected_condition_number = float(np.divide(maximum, minimum))
        _require(
            np.isfinite(expected_condition_number)
            and condition_number is not None
            and condition_number == expected_condition_number,
            "v2 admission certificate condition number is inconsistent with its Hessian spectrum",
        )
    else:
        _require(
            condition_number is None,
            "v2 admission certificate condition number requires positive valid curvature",
        )

    expected_within_limit = bool(
        expected_positive_curvature
        and condition_number is not None
        and condition_number <= maximum_condition_number
    )
    _require(
        certificate["condition_number_within_limit"] is expected_within_limit,
        "v2 admission certificate condition-number limit invariant is inconsistent",
    )
    expected_passed = all(bool(certificate[name]) for name in _CERTIFICATE_BOOL_FIELDS)
    _require(
        type(certificate.get("passed")) is bool
        and certificate.get("passed") == expected_passed,
        "v2 admission certificate pass invariant is inconsistent",
    )
    expected_reason = _admission_reason(
        underlying_inference_admissible=bool(
            certificate["underlying_inference_admissible"]
        ),
        exact_mixture_objective=bool(certificate["exact_mixture_objective"]),
        fixed_point_converged=bool(certificate["fixed_point_converged"]),
        diagnostics_valid=diagnostics_valid,
        positive_exact_mixture_curvature=expected_positive_curvature,
        condition_number_within_limit=expected_within_limit,
    )
    _require(
        type(certificate.get("reason")) is str
        and certificate.get("reason") == expected_reason,
        "v2 admission certificate reason invariant is inconsistent",
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
        raise ValueError("v2 diagnostics do not contain an admission certificate")
    certificate_passed = _validate_certificate_mapping(certificate)
    strict_passed = diagnostics.get("strict_admission_passed")
    _require(
        type(strict_passed) is bool and strict_passed == certificate_passed,
        "v2 strict-admission flag differs from its certificate",
    )
    _require(
        type(inference_admissible) is bool
        and inference_admissible == certificate_passed,
        "v2 inference admissibility differs from its certificate",
    )
    _require(
        diagnostics.get("strict_admission_reason") == certificate["reason"],
        "v2 strict-admission reason differs from its certificate",
    )
    for name, certificate_name in (
        ("underlying_inference_admissible", "underlying_inference_admissible"),
        ("underlying_inference_reason", "underlying_inference_reason"),
        ("strict_exact_hessian_condition_number", "exact_hessian_condition_number"),
        (
            "maximum_exact_hessian_condition_number",
            "maximum_exact_hessian_condition_number",
        ),
    ):
        _require(
            diagnostics.get(name) == certificate[certificate_name],
            f"v2 diagnostic {name!r} differs from its certificate",
        )
    underlying_admissible = certificate["underlying_inference_admissible"] is True
    expected_result_reason = (
        certificate["underlying_inference_reason"]
        if certificate_passed or not underlying_admissible
        else certificate["reason"]
    )
    _require(
        type(result_reason) is str and result_reason == expected_result_reason,
        "v2 result reason differs from its selected certificate path",
    )
    return certificate_passed


def _tag_diagnostics(
    diagnostics: Mapping[str, object],
    *,
    admission: PriorAwareGaugeAdmissionConfigV2,
    certificate: _PriorAwareGaugeAdmissionCertificateV1,
) -> dict[str, object]:
    tagged = dict(diagnostics)
    tagged.update(
        {
            "implementation_schema": PRIOR_AWARE_GAUGE_BELIEF_V2_SCHEMA,
            "implementation_version": PRIOR_AWARE_GAUGE_BELIEF_V2_VERSION,
            "implementation_id": PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION,
            "strict_admission_version": PRIOR_AWARE_GAUGE_BELIEF_V2_VERSION,
            "strict_admission_passed": certificate.passed,
            "strict_admission_reason": certificate.reason,
            "underlying_inference_admissible": (
                certificate.underlying_inference_admissible
            ),
            "underlying_inference_reason": certificate.underlying_inference_reason,
            "strict_exact_hessian_condition_number": (
                certificate.exact_hessian_condition_number
            ),
            _CERTIFICATE_KEY: certificate.as_mapping(),
            "exact_mixture_objective_required": True,
            "fixed_point_convergence_required": True,
            "positive_exact_mixture_curvature_required": True,
            "maximum_exact_hessian_condition_number": (
                admission.maximum_exact_hessian_condition_number
            ),
            "implicit_jitter": False,
            "eigenvalue_clipping": False,
            "pseudoinverse_fallback": False,
            "claim_boundary": PRIOR_AWARE_GAUGE_BELIEF_V2_CLAIM_BOUNDARY,
        }
    )
    return tagged


def _as_v2(
    result: GaugeAwareBeliefResult,
    diagnostics: Mapping[str, object],
) -> PriorAwareGaugeBeliefResultV2:
    return PriorAwareGaugeBeliefResultV2(
        inference_admissible=result.inference_admissible,
        reason=result.reason,
        state_coefficients=result.state_coefficients,
        gauge_delta=result.gauge_delta,
        shared_bias_coefficients=result.shared_bias_coefficients,
        view_bias_coefficients=result.view_bias_coefficients,
        anchor_bias_coefficients=result.anchor_bias_coefficients,
        posterior_covariance=result.posterior_covariance,
        identifiable_state_transform=result.identifiable_state_transform,
        identifiable_fractions=result.identifiable_fractions,
        query_sensitivity_fractions=result.query_sensitivity_fractions,
        robust_weights=result.robust_weights,
        anchor_robust_weights=result.anchor_robust_weights,
        diagnostics=diagnostics,
        input_lineage=result.input_lineage,
    )


def _as_structured_v2(
    result: StructuredGaugeAwareBeliefResultV1,
    diagnostics: Mapping[str, object],
) -> StructuredGaugeAwareBeliefResultV1:
    _validate_tagged_certificate_diagnostics(
        diagnostics,
        inference_admissible=result.inference_admissible,
        result_reason=result.reason,
    )
    return StructuredGaugeAwareBeliefResultV1(
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


def _strict_admission_payload(
    result: AdmissionInputResult,
    admission: PriorAwareGaugeAdmissionConfigV2,
) -> tuple[
    _PriorAwareGaugeAdmissionCertificateV1,
    dict[str, object],
    str,
]:
    certificate = _build_admission_certificate(result, admission)
    tagged = _tag_diagnostics(
        result.diagnostics,
        admission=admission,
        certificate=certificate,
    )
    fallback_reason = (
        result.reason if not result.inference_admissible else certificate.reason
    )
    return certificate, tagged, fallback_reason


def _apply_strict_admission(
    result: GaugeAwareBeliefResult,
    *,
    admission: PriorAwareGaugeAdmissionConfigV2,
    fallback: Callable[[str, Mapping[str, object]], GaugeAwareBeliefResult],
) -> PriorAwareGaugeBeliefResultV2:
    certificate, tagged, fallback_reason = _strict_admission_payload(
        result,
        admission,
    )
    selected = result if certificate.passed else fallback(fallback_reason, tagged)
    return _as_v2(selected, tagged)


def _apply_structured_strict_admission(
    result: StructuredGaugeAwareBeliefResultV1,
    *,
    admission: PriorAwareGaugeAdmissionConfigV2,
    fallback: Callable[
        [str, Mapping[str, object]],
        StructuredGaugeAwareBeliefResultV1,
    ],
) -> StructuredGaugeAwareBeliefResultV1:
    certificate, tagged, fallback_reason = _strict_admission_payload(
        result,
        admission,
    )
    if certificate.passed:
        return _as_structured_v2(result, tagged)
    selected = fallback(fallback_reason, tagged)
    _validate_tagged_certificate_diagnostics(
        selected.diagnostics,
        inference_admissible=selected.inference_admissible,
        result_reason=selected.reason,
    )
    return selected


def update_prior_aware_gauge_belief_v2(
    batch: GaugeAwareObservationBatch,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
    admission_config: PriorAwareGaugeAdmissionConfigV2 | None = None,
) -> PriorAwareGaugeBeliefResultV2:
    """Run dense prior-aware inference and fail closed on strict v2 checks."""

    if not isinstance(batch, GaugeAwareObservationBatch):
        raise TypeError("batch must be a GaugeAwareObservationBatch")
    if config is not None and not isinstance(config, PriorAwareGaugeConfigV1):
        raise TypeError("config must be a PriorAwareGaugeConfigV1")
    if admission_config is not None and not isinstance(
        admission_config,
        PriorAwareGaugeAdmissionConfigV2,
    ):
        raise TypeError("admission_config must be a PriorAwareGaugeAdmissionConfigV2")
    cfg = config or PriorAwareGaugeConfigV1()
    admission = admission_config or PriorAwareGaugeAdmissionConfigV2()
    result = update_prior_aware_gauge_belief(batch, config=cfg)

    def fallback(
        reason: str,
        diagnostics: Mapping[str, object],
    ) -> GaugeAwareBeliefResult:
        _, _, prior = _prior_covariances(batch, cfg)
        return _fallback_result(
            batch,
            reason,
            diagnostics,
            prior_covariance=prior,
        )

    return _apply_strict_admission(
        result,
        admission=admission,
        fallback=fallback,
    )


def update_sparse_prior_aware_gauge_belief_v2(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
    admission_config: PriorAwareGaugeAdmissionConfigV2 | None = None,
) -> PriorAwareGaugeBeliefResultV2:
    """Run native-sparse prior-aware inference with strict v2 admission."""

    if not isinstance(batch, GaugeAwareObservationBatch):
        raise TypeError("batch must be a GaugeAwareObservationBatch")
    if not isinstance(gauge, (SparseGaugeDesignV1, TreeSparseGaugeDesignV1)):
        raise TypeError(
            "gauge must be a SparseGaugeDesignV1 or TreeSparseGaugeDesignV1"
        )
    if config is not None and not isinstance(config, PriorAwareGaugeConfigV1):
        raise TypeError("config must be a PriorAwareGaugeConfigV1")
    if admission_config is not None and not isinstance(
        admission_config,
        PriorAwareGaugeAdmissionConfigV2,
    ):
        raise TypeError("admission_config must be a PriorAwareGaugeAdmissionConfigV2")
    cfg = config or PriorAwareGaugeConfigV1()
    admission = admission_config or PriorAwareGaugeAdmissionConfigV2()
    result = update_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=cfg,
    )

    def fallback(
        reason: str,
        diagnostics: Mapping[str, object],
    ) -> GaugeAwareBeliefResult:
        _, _, prior = _sparse_prior_covariances(batch, gauge, cfg)
        fallback_result = _sparse_fallback_result(
            batch,
            gauge,
            reason,
            diagnostics,
            prior_covariance=prior,
        )
        if not isinstance(fallback_result, GaugeAwareBeliefResult):
            raise RuntimeError("dense strict-v2 fallback returned a structured result")
        return fallback_result

    return _apply_strict_admission(
        result,
        admission=admission,
        fallback=fallback,
    )


def update_sparse_prior_aware_gauge_belief_structured_v2(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
    admission_config: PriorAwareGaugeAdmissionConfigV2 | None = None,
) -> StructuredGaugeAwareBeliefResultV1:
    """Run structured sparse inference and retain exact prior on strict rejection."""

    if not isinstance(batch, GaugeAwareObservationBatch):
        raise TypeError("batch must be a GaugeAwareObservationBatch")
    if not isinstance(gauge, (SparseGaugeDesignV1, TreeSparseGaugeDesignV1)):
        raise TypeError(
            "gauge must be a SparseGaugeDesignV1 or TreeSparseGaugeDesignV1"
        )
    if config is not None and not isinstance(config, PriorAwareGaugeConfigV1):
        raise TypeError("config must be a PriorAwareGaugeConfigV1")
    if admission_config is not None and not isinstance(
        admission_config,
        PriorAwareGaugeAdmissionConfigV2,
    ):
        raise TypeError("admission_config must be a PriorAwareGaugeAdmissionConfigV2")
    cfg = config or PriorAwareGaugeConfigV1()
    admission = admission_config or PriorAwareGaugeAdmissionConfigV2()
    result = update_sparse_prior_aware_gauge_belief_structured(
        batch,
        gauge,
        config=cfg,
    )

    def fallback(
        reason: str,
        diagnostics: Mapping[str, object],
    ) -> StructuredGaugeAwareBeliefResultV1:
        _, _, prior = _sparse_prior_covariances(batch, gauge, cfg)
        fallback_diagnostics = {
            **diagnostics,
            "result_covariance_representation": prior.representation,
            "result_dense_covariance_materialized": False,
            "result_estimated_dense_covariance_bytes": prior.estimated_dense_bytes,
            "result_stored_covariance_bytes_before_materialization": (
                prior.stored_nbytes
            ),
        }
        return StructuredGaugeAwareBeliefResultV1(
            inference_admissible=False,
            reason=reason,
            state_coefficients=np.zeros_like(result.state_coefficients),
            gauge_delta=np.zeros_like(result.gauge_delta),
            shared_bias_coefficients=np.zeros_like(result.shared_bias_coefficients),
            view_bias_coefficients=np.zeros_like(result.view_bias_coefficients),
            anchor_bias_coefficients=np.zeros_like(result.anchor_bias_coefficients),
            covariance=prior,
            identifiable_state_transform=np.zeros(
                (len(result.state_coefficients), 0),
                dtype=np.float64,
            ),
            identifiable_fractions=np.zeros(0, dtype=np.float64),
            query_sensitivity_fractions=np.zeros(0, dtype=np.float64),
            robust_weights=np.zeros_like(result.robust_weights),
            anchor_robust_weights=np.zeros_like(result.anchor_robust_weights),
            diagnostics=fallback_diagnostics,
            input_lineage=result.input_lineage,
        )

    return _apply_structured_strict_admission(
        result,
        admission=admission,
        fallback=fallback,
    )


__all__ = [
    "PRIOR_AWARE_GAUGE_BELIEF_V2_CLAIM_BOUNDARY",
    "PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION",
    "PRIOR_AWARE_GAUGE_BELIEF_V2_SCHEMA",
    "PRIOR_AWARE_GAUGE_BELIEF_V2_VERSION",
    "PriorAwareGaugeAdmissionConfigV2",
    "PriorAwareGaugeBeliefResultV2",
    "update_prior_aware_gauge_belief_v2",
    "update_sparse_prior_aware_gauge_belief_structured_v2",
    "update_sparse_prior_aware_gauge_belief_v2",
]
