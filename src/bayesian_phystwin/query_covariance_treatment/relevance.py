"""Numerical query-space relevance diagnostics for shared covariance."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .._canonical_contracts import genuine_boolean
from ._common import array_sha256, real_array, symmetric_psd
from ._relevance_types import (
    QueryCovarianceRelevanceCertificateV1,
    QueryCovarianceRelevancePolicyV1,
    relevance_reasons,
    shared_covariance_material,
)


def certify_query_covariance_relevance(
    *,
    query_id: str,
    covariance_artifact_id: str,
    jacobian_artifact_id: str,
    calibration_partition_id: str,
    statistical_unit: str,
    local_covariance: object,
    shared_factor: object,
    query_jacobian: object,
    query_noise_covariance: object | None,
    policy: QueryCovarianceRelevancePolicyV1,
    frozen_before_target_outcomes: bool,
    target_outcomes_used_for_selection: bool,
    calibration_groups_independent: bool,
    metadata: Mapping[str, Any] | None = None,
) -> QueryCovarianceRelevanceCertificateV1:
    """Project local and shared covariance into one registered query space."""

    if not isinstance(policy, QueryCovarianceRelevancePolicyV1):
        raise TypeError("policy must be QueryCovarianceRelevancePolicyV1")
    local = symmetric_psd(local_covariance, name="local_covariance")
    factor = real_array(shared_factor, name="shared_factor", ndim=2)
    jacobian = real_array(query_jacobian, name="query_jacobian", ndim=2)
    state_dimension = local.shape[0]
    if factor.shape[0] != state_dimension:
        raise ValueError("shared_factor state dimension does not match covariance")
    if jacobian.shape[1] != state_dimension or jacobian.shape[0] < 1:
        raise ValueError("query_jacobian has incompatible dimensions")
    query_dimension = jacobian.shape[0]
    if query_noise_covariance is None:
        noise = np.zeros((query_dimension, query_dimension), dtype=np.float64)
    else:
        noise = symmetric_psd(
            query_noise_covariance,
            name="query_noise_covariance",
        )
        if noise.shape != (query_dimension, query_dimension):
            raise ValueError("query noise dimension does not match the query")

    with np.errstate(over="ignore", invalid="ignore"):
        local_query = jacobian @ local @ jacobian.T
        projected_factor = jacobian @ factor
        shared_query = projected_factor @ projected_factor.T
        reference_query = local_query + noise
    for name, array in (
        ("local query covariance", local_query),
        ("projected shared factor", projected_factor),
        ("shared query covariance", shared_query),
        ("reference query covariance", reference_query),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} overflowed finite float64 representation")

    local_trace = max(float(np.trace(local_query)), 0.0)
    shared_trace = max(float(np.trace(shared_query)), 0.0)
    noise_trace = max(float(np.trace(noise)), 0.0)
    total_trace = local_trace + shared_trace + noise_trace
    if not math.isfinite(total_trace) or total_trace <= 0.0:
        raise ValueError("query covariance must have positive finite trace")
    shared_trace_fraction = shared_trace / total_trace

    shared_rank = factor.shape[1]
    if shared_rank:
        try:
            singular_values = np.linalg.svd(
                projected_factor,
                compute_uv=False,
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "projected shared-factor rank could not be evaluated"
            ) from error
        if not np.all(np.isfinite(singular_values)):
            raise ValueError("projected shared-factor singular values must be finite")
        maximum_singular = float(np.max(singular_values, initial=0.0))
        rank_threshold = policy.rank_relative_tolerance * max(maximum_singular, 1.0)
        effective_query_rank = int(np.sum(singular_values > rank_threshold))
        mode_norms = np.linalg.norm(projected_factor, axis=0)
        if not np.all(np.isfinite(mode_norms)):
            raise ValueError("projected shared-mode norms must be finite")
        maximum_mode_norm = float(np.max(mode_norms, initial=0.0))
        mode_threshold = policy.mode_response_relative_tolerance * max(
            maximum_mode_norm,
            1.0,
        )
        null_mode_fraction = float(np.mean(mode_norms <= mode_threshold))
    else:
        effective_query_rank = 0
        null_mode_fraction = 1.0

    scale = max(float(np.max(np.abs(reference_query))), 1.0)
    regularized = 0.5 * (reference_query + reference_query.T)
    regularized += (
        policy.covariance_jitter
        * scale
        * np.eye(
            query_dimension,
            dtype=np.float64,
        )
    )
    try:
        cholesky = np.linalg.cholesky(regularized)
        left = np.linalg.solve(cholesky, shared_query)
        whitened = np.linalg.solve(cholesky, left.T).T
        eigenvalues = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    except np.linalg.LinAlgError as error:
        raise ValueError("query covariance relevance solve failed") from error
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("query covariance relevance eigenvalues must be finite")
    maximum_generalized_eigenvalue = max(
        float(np.max(eigenvalues, initial=0.0)),
        0.0,
    )

    frozen = genuine_boolean(
        frozen_before_target_outcomes,
        name="frozen_before_target_outcomes",
    )
    target_used = genuine_boolean(
        target_outcomes_used_for_selection,
        name="target_outcomes_used_for_selection",
    )
    independent = genuine_boolean(
        calibration_groups_independent,
        name="calibration_groups_independent",
    )
    reasons = relevance_reasons(
        shared_trace_fraction=shared_trace_fraction,
        effective_query_rank=effective_query_rank,
        null_mode_fraction=null_mode_fraction,
        maximum_generalized_eigenvalue=maximum_generalized_eigenvalue,
        policy=policy,
        frozen_before_target_outcomes=frozen,
        target_outcomes_used_for_selection=target_used,
        calibration_groups_independent=independent,
    )
    return QueryCovarianceRelevanceCertificateV1(
        query_id=query_id,
        covariance_artifact_id=covariance_artifact_id,
        jacobian_artifact_id=jacobian_artifact_id,
        calibration_partition_id=calibration_partition_id,
        statistical_unit=statistical_unit,
        state_dimension=state_dimension,
        query_dimension=query_dimension,
        shared_rank=shared_rank,
        local_covariance_sha256=array_sha256(local),
        shared_factor_sha256=array_sha256(factor),
        query_jacobian_sha256=array_sha256(jacobian),
        query_noise_covariance_sha256=array_sha256(noise),
        shared_trace_fraction=shared_trace_fraction,
        effective_query_rank=effective_query_rank,
        null_mode_fraction=null_mode_fraction,
        maximum_generalized_eigenvalue=maximum_generalized_eigenvalue,
        shared_covariance_material=shared_covariance_material(
            shared_trace_fraction=shared_trace_fraction,
            effective_query_rank=effective_query_rank,
            null_mode_fraction=null_mode_fraction,
            maximum_generalized_eigenvalue=maximum_generalized_eigenvalue,
            policy=policy,
        ),
        reasons=reasons,
        policy=policy,
        frozen_before_target_outcomes=frozen,
        target_outcomes_used_for_selection=target_used,
        calibration_groups_independent=independent,
        metadata={} if metadata is None else metadata,
    )
