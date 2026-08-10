"""Factorized linear queries of structured discrepancy beliefs."""

from __future__ import annotations

import numpy as np

from ._structured_discrepancy_contracts import (
    StructuredDiscrepancyBeliefV1,
    StructuredDiscrepancyPosteriorV1,
    StructuredDiscrepancyPredictionV1,
    StructuredDiscrepancyQueryMomentsV1,
)


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def _validated_query_jacobian(
    query_jacobian: np.ndarray,
    *,
    track_count: int,
) -> np.ndarray:
    raw = np.asarray(query_jacobian)
    if raw.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("query_jacobian must be numeric")
    query = np.asarray(raw, dtype=np.float64)
    if query.ndim == 2 and query.shape[1] == 3 * track_count:
        query = query.reshape(len(query), track_count, 3)
    if query.ndim != 3 or query.shape[1:] != (track_count, 3):
        raise ValueError(
            "query_jacobian must have shape (queries, tracks, 3) or "
            "(queries, 3 * tracks)"
        )
    if len(query) < 1 or not np.all(np.isfinite(query)):
        raise ValueError("query_jacobian must be finite and nonempty")
    return query


def structured_discrepancy_query_moments(
    belief: StructuredDiscrepancyBeliefV1,
    query_jacobian: np.ndarray,
) -> StructuredDiscrepancyQueryMomentsV1:
    """Evaluate an exact linear query without materializing dense field covariance."""

    if not isinstance(
        belief,
        (StructuredDiscrepancyPosteriorV1, StructuredDiscrepancyPredictionV1),
    ):
        raise TypeError("belief must be a structured discrepancy belief")
    query = _validated_query_jacobian(
        query_jacobian,
        track_count=len(belief.spatial_basis),
    )
    mean = np.einsum("qnc,nc->q", query, belief.mean_m)
    covariance = np.zeros((len(query), len(query)), dtype=np.float64)
    for index, weight in enumerate(belief.component_weights):
        within = np.zeros_like(covariance)
        coefficient_covariance = belief.component_coefficient_covariance_m2[index]
        local_variance = belief.component_local_variance_m2[index]
        for coordinate in range(3):
            coordinate_query = query[:, :, coordinate]
            coefficient_query = coordinate_query @ belief.spatial_basis
            within += coefficient_query @ coefficient_covariance @ coefficient_query.T
            within += (coordinate_query * local_variance[None, :]) @ (
                coordinate_query.T
            )
        component_query_mean = np.einsum(
            "qnc,nc->q",
            query,
            belief.component_mean_m[index],
        )
        centered = component_query_mean - mean
        covariance += weight * (within + np.outer(centered, centered))
    covariance = _symmetric(covariance)
    return StructuredDiscrepancyQueryMomentsV1(
        mean=mean,
        covariance=covariance,
    )


__all__ = ["structured_discrepancy_query_moments"]
