"""Prospective strict admission for prior-aware grouped-mixture updates.

The version-1 dense and native-sparse solvers remain unchanged for historical
reproduction. This module calls those solvers and then applies one additional,
fail-closed admission boundary before an update can be returned as admissible.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Final

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
_FIXED_POINT_REASON: Final = "strict-v2-fixed-point-not-converged"
_NONEXACT_REASON: Final = "strict-v2-non-exact-mixture-objective"
_INVALID_DIAGNOSTICS_REASON: Final = "strict-v2-invalid-admission-diagnostics"
_NONPOSITIVE_CURVATURE_REASON: Final = "strict-v2-non-positive-exact-mixture-curvature"
_ILL_CONDITIONED_CURVATURE_REASON: Final = (
    "strict-v2-ill-conditioned-exact-mixture-curvature"
)

GaugeDesignV1 = SparseGaugeDesignV1 | TreeSparseGaugeDesignV1
AdmissionInputResult = GaugeAwareBeliefResult | StructuredGaugeAwareBeliefResultV1


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


def _finite_diagnostic(
    diagnostics: Mapping[str, object],
    name: str,
) -> float | None:
    value = diagnostics.get(name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _tag_diagnostics(
    diagnostics: Mapping[str, object],
    *,
    admission: PriorAwareGaugeAdmissionConfigV2,
    passed: bool,
    reason: str,
    underlying_result: AdmissionInputResult,
) -> dict[str, object]:
    tagged = dict(diagnostics)
    tagged.update(
        {
            "implementation_schema": PRIOR_AWARE_GAUGE_BELIEF_V2_SCHEMA,
            "implementation_version": PRIOR_AWARE_GAUGE_BELIEF_V2_VERSION,
            "implementation_id": PRIOR_AWARE_GAUGE_BELIEF_V2_IMPLEMENTATION,
            "strict_admission_version": PRIOR_AWARE_GAUGE_BELIEF_V2_VERSION,
            "strict_admission_passed": passed,
            "strict_admission_reason": reason,
            "underlying_inference_admissible": (
                underlying_result.inference_admissible
            ),
            "underlying_inference_reason": underlying_result.reason,
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


def _strict_failure(
    diagnostics: dict[str, object],
    admission: PriorAwareGaugeAdmissionConfigV2,
) -> str | None:
    if diagnostics.get("robust_likelihood_objective") != _EXACT_MIXTURE_OBJECTIVE:
        return _NONEXACT_REASON
    if diagnostics.get("mixture_fixed_point_converged") is not True:
        return _FIXED_POINT_REASON

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
    if (
        solution_delta is None
        or solution_delta < 0.0
        or stationarity is None
        or stationarity < 0.0
        or minimum is None
        or maximum is None
        or maximum < minimum
        or type(declared_positive) is not bool
        or declared_positive != (minimum > 0.0)
    ):
        return _INVALID_DIAGNOSTICS_REASON
    if minimum <= 0.0:
        return _NONPOSITIVE_CURVATURE_REASON

    condition_number = maximum / minimum
    if not np.isfinite(condition_number):
        return _INVALID_DIAGNOSTICS_REASON
    diagnostics["strict_exact_hessian_condition_number"] = condition_number
    if condition_number > admission.maximum_exact_hessian_condition_number:
        return _ILL_CONDITIONED_CURVATURE_REASON
    return None


def _apply_strict_admission(
    result: GaugeAwareBeliefResult,
    *,
    admission: PriorAwareGaugeAdmissionConfigV2,
    fallback: Callable[[str, Mapping[str, object]], GaugeAwareBeliefResult],
) -> PriorAwareGaugeBeliefResultV2:
    if not result.inference_admissible:
        diagnostics = _tag_diagnostics(
            result.diagnostics,
            admission=admission,
            passed=False,
            reason="underlying-inference-rejected",
            underlying_result=result,
        )
        return _as_v2(result, diagnostics)

    diagnostics = dict(result.diagnostics)
    failure = _strict_failure(diagnostics, admission)
    if failure is None:
        tagged = _tag_diagnostics(
            diagnostics,
            admission=admission,
            passed=True,
            reason="strict-admission-passed",
            underlying_result=result,
        )
        return _as_v2(result, tagged)

    tagged = _tag_diagnostics(
        diagnostics,
        admission=admission,
        passed=False,
        reason=failure,
        underlying_result=result,
    )
    return _as_v2(fallback(failure, tagged), tagged)


def _apply_structured_strict_admission(
    result: StructuredGaugeAwareBeliefResultV1,
    *,
    admission: PriorAwareGaugeAdmissionConfigV2,
    fallback: Callable[
        [str, Mapping[str, object]],
        StructuredGaugeAwareBeliefResultV1,
    ],
) -> StructuredGaugeAwareBeliefResultV1:
    if not result.inference_admissible:
        diagnostics = _tag_diagnostics(
            result.diagnostics,
            admission=admission,
            passed=False,
            reason="underlying-inference-rejected",
            underlying_result=result,
        )
        return _as_structured_v2(result, diagnostics)

    diagnostics = dict(result.diagnostics)
    failure = _strict_failure(diagnostics, admission)
    if failure is None:
        tagged = _tag_diagnostics(
            diagnostics,
            admission=admission,
            passed=True,
            reason="strict-admission-passed",
            underlying_result=result,
        )
        return _as_structured_v2(result, tagged)

    tagged = _tag_diagnostics(
        diagnostics,
        admission=admission,
        passed=False,
        reason=failure,
        underlying_result=result,
    )
    return fallback(failure, tagged)


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
            shared_bias_coefficients=np.zeros_like(
                result.shared_bias_coefficients
            ),
            view_bias_coefficients=np.zeros_like(result.view_bias_coefficients),
            anchor_bias_coefficients=np.zeros_like(
                result.anchor_bias_coefficients
            ),
            covariance=prior,
            identifiable_state_transform=np.zeros(
                (len(result.state_coefficients), 0),
                dtype=np.float64,
            ),
            identifiable_fractions=np.zeros(0, dtype=np.float64),
            query_sensitivity_fractions=np.zeros(0, dtype=np.float64),
            robust_weights=np.zeros_like(result.robust_weights),
            anchor_robust_weights=np.zeros_like(
                result.anchor_robust_weights
            ),
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
