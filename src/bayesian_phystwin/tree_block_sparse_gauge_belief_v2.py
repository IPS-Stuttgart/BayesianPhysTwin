"""Strict admission for the exact block-tree gauge solver.

The version-1 block-tree solver remains available as a numerical reproduction
surface. This module adds the fail-closed boundary required by prospective and
claim-bearing callers: an exhausted or diagnostically invalid robust fixed point
cannot modify the physical belief.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Final

import numpy as np

from ._gauge_aware_contracts import GaugeAwareObservationBatch
from ._prior_aware_gauge_math import PriorAwareGaugeConfigV1, _whiten
from .sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
    _whiten_sparse_observations,
)
from .tree_block_claim_contract import validate_tree_block_result
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
    "physical and tree-gauge prior whenever the robust fixed point is unfinished "
    "or its admission diagnostics are invalid. It does not establish provider "
    "competence, uncertainty calibration, physical-query benefit, intervention "
    "benefit, deployment safety, or state of the art."
)

_EXACT_MIXTURE_OBJECTIVE: Final = "exact-group-mixture-gradient"
_FIXED_POINT_REASON: Final = "strict-v2-fixed-point-not-converged"
_NONEXACT_REASON: Final = "strict-v2-non-exact-mixture-objective"
_INVALID_DIAGNOSTICS_REASON: Final = "strict-v2-invalid-admission-diagnostics"


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


def _strict_failure(diagnostics: Mapping[str, object]) -> str | None:
    if diagnostics.get("robust_likelihood_objective") != _EXACT_MIXTURE_OBJECTIVE:
        return _NONEXACT_REASON
    if diagnostics.get("mixture_fixed_point_converged") is not True:
        return _FIXED_POINT_REASON
    if diagnostics.get("posterior_solver") != "tree-block-leaf-schur-cholesky-v1":
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
    passed: bool,
    reason: str,
    underlying_result: TreeBlockGaugeAwareBeliefResultV1,
) -> dict[str, object]:
    tagged = dict(diagnostics)
    tagged.update(
        {
            "implementation_schema": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_SCHEMA,
            "implementation_version": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_VERSION,
            "implementation_id": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION,
            "strict_admission_version": TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_VERSION,
            "strict_admission_passed": passed,
            "strict_admission_reason": reason,
            "underlying_inference_admissible": (underlying_result.inference_admissible),
            "underlying_inference_reason": underlying_result.reason,
            "exact_mixture_objective_required": True,
            "fixed_point_convergence_required": True,
            "finite_stationarity_diagnostics_required": True,
            "positive_cholesky_condition_diagnostics_required": True,
            "implicit_jitter": False,
            "eigenvalue_clipping": False,
            "pseudoinverse_fallback": False,
            "claim_boundary": (TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_CLAIM_BOUNDARY),
        }
    )
    return tagged


def _copy_result(
    result: TreeBlockGaugeAwareBeliefResultV1,
    diagnostics: Mapping[str, object],
) -> TreeBlockGaugeAwareBeliefResultV1:
    return TreeBlockGaugeAwareBeliefResultV1(
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


def update_tree_block_sparse_prior_aware_gauge_belief_v2(
    batch: GaugeAwareObservationBatch,
    gauge: TreeSparseGaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> TreeBlockGaugeAwareBeliefResultV1:
    """Run block-tree inference and fail closed on strict-v2 admission checks."""

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
    validate_tree_block_result(result)
    if not result.inference_admissible:
        diagnostics = _tag_diagnostics(
            result.diagnostics,
            passed=False,
            reason="underlying-inference-rejected",
            underlying_result=result,
        )
        fallback = _fallback_result(
            batch=batch,
            gauge=gauge,
            reason=result.reason,
            diagnostics=diagnostics,
            covariance=_exact_prior_covariance(batch, gauge, cfg),
        )
        return validate_tree_block_result(
            fallback,
            require_strict_admission=True,
        )

    failure = _strict_failure(result.diagnostics)
    if failure is None:
        admitted = _copy_result(
            result,
            _tag_diagnostics(
                result.diagnostics,
                passed=True,
                reason="strict-admission-passed",
                underlying_result=result,
            ),
        )
        return validate_tree_block_result(
            admitted,
            require_strict_admission=True,
        )

    diagnostics = _tag_diagnostics(
        result.diagnostics,
        passed=False,
        reason=failure,
        underlying_result=result,
    )
    fallback = _fallback_result(
        batch=batch,
        gauge=gauge,
        reason=failure,
        diagnostics=diagnostics,
        covariance=_exact_prior_covariance(batch, gauge, cfg),
    )
    return validate_tree_block_result(
        fallback,
        require_strict_admission=True,
    )


__all__ = [
    "TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_CLAIM_BOUNDARY",
    "TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION",
    "TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_SCHEMA",
    "TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_VERSION",
    "update_tree_block_sparse_prior_aware_gauge_belief_v2",
]
