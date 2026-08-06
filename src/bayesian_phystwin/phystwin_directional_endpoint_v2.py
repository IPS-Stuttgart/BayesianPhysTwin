"""Prospective SPD-safe directional discrepancy filtering for PhysTwin cues.

This module is deliberately versioned separately from the frozen v1 endpoint.
It uses Cholesky solves, Joseph-form component covariance updates, exact Gaussian
mixture moment matching, and fail-closed numerical admission. It never adds
jitter, clips eigenvalues, substitutes a pseudoinverse, or replaces a full
covariance by its trace-average isotropic approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ._phystwin_directional_endpoint_v2_solver import (
    DirectionalEndpointConfigV2,
    DirectionalEndpointNumericalError,
    admit_spd_system,
    immutable_float64,
    immutable_int64,
    robust_linear_update_v2,
    validate_filter_parameters,
)
from .spd_system import SPD_SYSTEM_SCHEMA, SPD_SYSTEM_VERSION, SPDSystemError

PHYSTWIN_DIRECTIONAL_ENDPOINT_SCHEMA: Final = (
    "bayesian_phystwin.phystwin_directional_endpoint"
)
PHYSTWIN_DIRECTIONAL_ENDPOINT_VERSION: Final = 2


@dataclass(frozen=True, slots=True)
class DirectionalEndpointPosteriorV2:
    """Per-identity posterior with retained covariance and audit diagnostics."""

    mean: np.ndarray
    covariance: np.ndarray
    variance: np.ndarray
    final_inlier_probability: np.ndarray
    update_count: np.ndarray
    source_update_count: np.ndarray
    tangent_update_count: np.ndarray
    maximum_innovation_condition_number: np.ndarray
    maximum_posterior_condition_number: np.ndarray

    def diagnostics(self) -> dict[str, object]:
        """Return JSON-compatible prospective numerical semantics."""

        return {
            "schema": PHYSTWIN_DIRECTIONAL_ENDPOINT_SCHEMA,
            "schema_version": PHYSTWIN_DIRECTIONAL_ENDPOINT_VERSION,
            "spd_backend_schema": SPD_SYSTEM_SCHEMA,
            "spd_backend_version": SPD_SYSTEM_VERSION,
            "component_covariance_update": "joseph-form",
            "mixture_covariance_update": "exact-moment-matching",
            "full_source_covariance_retained": True,
            "trace_average_isotropization": False,
            "implicit_jitter": False,
            "eigenvalue_clipping": False,
            "pseudoinverse_fallback": False,
            "maximum_innovation_condition_number": float(
                np.max(self.maximum_innovation_condition_number, initial=0.0)
            ),
            "maximum_posterior_condition_number": float(
                np.max(self.maximum_posterior_condition_number, initial=0.0)
            ),
        }


def robust_directional_endpoint_v2(
    source_residual: np.ndarray,
    source_valid: np.ndarray,
    multiview_residual: np.ndarray,
    multiview_valid: np.ndarray,
    tangent_projectors: np.ndarray,
    priority_identities: np.ndarray,
    *,
    end_frame: int,
    process_variance: float,
    observation_variance: float,
    initial_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
    config: DirectionalEndpointConfigV2 | None = None,
) -> DirectionalEndpointPosteriorV2:
    """Filter directional innovations while retaining exact mixture covariance."""

    numerical_config = config or DirectionalEndpointConfigV2()
    source = np.asarray(source_residual, dtype=np.float64)
    multiview = np.asarray(multiview_residual, dtype=np.float64)
    source_mask = np.asarray(source_valid, dtype=bool)
    multiview_mask = np.asarray(multiview_valid, dtype=bool)
    projectors = np.asarray(tangent_projectors, dtype=np.float64)
    priority = np.asarray(priority_identities, dtype=bool)

    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source residual must have shape (frame, point, 3)")
    if multiview.shape != source.shape:
        raise ValueError("multiview residual must match source residual")
    if source_mask.shape != source.shape[:2]:
        raise ValueError("source validity must match the residual axes")
    if multiview_mask.shape != source.shape[:2]:
        raise ValueError("multiview validity must match the residual axes")
    if projectors.shape != (source.shape[1], 3, 3):
        raise ValueError("tangent projectors must have shape (point, 3, 3)")
    if priority.shape != (source.shape[1],):
        raise ValueError("priority identities must match the point axis")
    if np.any(source_mask & ~np.isfinite(source).all(axis=2)):
        raise ValueError("valid source residuals must be finite")
    if np.any(multiview_mask & ~np.isfinite(multiview).all(axis=2)):
        raise ValueError("valid multiview residuals must be finite")
    if not np.all(np.isfinite(projectors)):
        raise ValueError("tangent projectors must be finite")
    if not np.allclose(
        projectors,
        np.swapaxes(projectors, 1, 2),
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("tangent projectors must be symmetric")
    if not np.allclose(
        np.einsum("nij,njk->nik", projectors, projectors),
        projectors,
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("tangent projectors must be idempotent")
    (
        process,
        observation,
        initial,
        inlier,
        outlier,
    ) = validate_filter_parameters(
        end_frame=end_frame,
        frame_count=len(source),
        process_variance=process_variance,
        observation_variance=observation_variance,
        initial_variance=initial_variance,
        inlier_prior=inlier_prior,
        outlier_variance_multiplier=outlier_variance_multiplier,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(projectors)
    if not np.allclose(eigenvalues[:, 0], 0.0, atol=1e-7, rtol=0.0):
        raise ValueError("each tangent projector must have one null direction")
    if not np.allclose(eigenvalues[:, 1:], 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("each tangent projector must have rank two")
    normal_basis = np.swapaxes(eigenvectors[:, :, :1], 1, 2)
    tangent_basis = np.swapaxes(eigenvectors[:, :, 1:], 1, 2)

    point_count = source.shape[1]
    mean = np.zeros((point_count, 3), dtype=np.float64)
    identity_3 = np.eye(3, dtype=np.float64)
    covariance = np.repeat((initial * identity_3)[None], point_count, axis=0)
    final_probability = np.full(point_count, inlier, dtype=np.float64)
    source_update_count = np.zeros(point_count, dtype=np.int64)
    tangent_update_count = np.zeros(point_count, dtype=np.int64)
    maximum_innovation_condition = np.zeros(point_count, dtype=np.float64)
    maximum_posterior_condition = np.ones(point_count, dtype=np.float64)

    for frame in range(int(end_frame)):
        covariance += process * identity_3[None]
        if not np.all(np.isfinite(covariance)):
            raise DirectionalEndpointNumericalError(
                f"process covariance overflowed at frame {frame}"
            )

        full_indices = np.flatnonzero(source_mask[frame] & ~priority)
        if len(full_indices):
            full_matrix = np.repeat(identity_3[None], len(full_indices), axis=0)
            (
                mean[full_indices],
                covariance[full_indices],
                final_probability[full_indices],
                innovation_condition,
                posterior_condition,
            ) = robust_linear_update_v2(
                mean[full_indices],
                covariance[full_indices],
                source[frame, full_indices],
                full_matrix,
                observation_variance=observation,
                inlier_prior=inlier,
                outlier_variance_multiplier=outlier,
                name=f"frame {frame} full-source",
                config=numerical_config,
            )
            maximum_innovation_condition[full_indices] = np.maximum(
                maximum_innovation_condition[full_indices],
                innovation_condition,
            )
            maximum_posterior_condition[full_indices] = np.maximum(
                maximum_posterior_condition[full_indices],
                posterior_condition,
            )
            source_update_count[full_indices] += 1

        normal_indices = np.flatnonzero(source_mask[frame] & priority)
        if len(normal_indices):
            normal_matrix = normal_basis[normal_indices]
            normal_observation = np.einsum(
                "mij,mj->mi",
                normal_matrix,
                source[frame, normal_indices],
            )
            (
                mean[normal_indices],
                covariance[normal_indices],
                final_probability[normal_indices],
                innovation_condition,
                posterior_condition,
            ) = robust_linear_update_v2(
                mean[normal_indices],
                covariance[normal_indices],
                normal_observation,
                normal_matrix,
                observation_variance=observation,
                inlier_prior=inlier,
                outlier_variance_multiplier=outlier,
                name=f"frame {frame} source-normal",
                config=numerical_config,
            )
            maximum_innovation_condition[normal_indices] = np.maximum(
                maximum_innovation_condition[normal_indices],
                innovation_condition,
            )
            maximum_posterior_condition[normal_indices] = np.maximum(
                maximum_posterior_condition[normal_indices],
                posterior_condition,
            )
            source_update_count[normal_indices] += 1

        tangent_indices = np.flatnonzero(multiview_mask[frame] & priority)
        if len(tangent_indices):
            tangent_matrix = tangent_basis[tangent_indices]
            tangent_observation = np.einsum(
                "mij,mj->mi",
                tangent_matrix,
                multiview[frame, tangent_indices],
            )
            (
                mean[tangent_indices],
                covariance[tangent_indices],
                final_probability[tangent_indices],
                innovation_condition,
                posterior_condition,
            ) = robust_linear_update_v2(
                mean[tangent_indices],
                covariance[tangent_indices],
                tangent_observation,
                tangent_matrix,
                observation_variance=observation,
                inlier_prior=inlier,
                outlier_variance_multiplier=outlier,
                name=f"frame {frame} multiview-tangent",
                config=numerical_config,
            )
            maximum_innovation_condition[tangent_indices] = np.maximum(
                maximum_innovation_condition[tangent_indices],
                innovation_condition,
            )
            maximum_posterior_condition[tangent_indices] = np.maximum(
                maximum_posterior_condition[tangent_indices],
                posterior_condition,
            )
            tangent_update_count[tangent_indices] += 1

    for index in range(point_count):
        try:
            posterior = admit_spd_system(
                covariance[index],
                name=f"final point {index} covariance",
                config=numerical_config,
            )
        except SPDSystemError as error:
            raise DirectionalEndpointNumericalError(
                f"final point {index} covariance failed SPD admission"
            ) from error
        covariance[index] = posterior.matrix
        maximum_posterior_condition[index] = max(
            maximum_posterior_condition[index],
            posterior.condition_number,
        )

    eigenvalues = np.linalg.eigvalsh(covariance)
    conservative_variance = eigenvalues[:, -1]
    update_count = source_update_count + tangent_update_count
    return DirectionalEndpointPosteriorV2(
        mean=immutable_float64(mean),
        covariance=immutable_float64(covariance),
        variance=immutable_float64(conservative_variance),
        final_inlier_probability=immutable_float64(final_probability),
        update_count=immutable_int64(update_count),
        source_update_count=immutable_int64(source_update_count),
        tangent_update_count=immutable_int64(tangent_update_count),
        maximum_innovation_condition_number=immutable_float64(
            maximum_innovation_condition
        ),
        maximum_posterior_condition_number=immutable_float64(
            maximum_posterior_condition
        ),
    )


__all__ = [
    "DirectionalEndpointConfigV2",
    "DirectionalEndpointNumericalError",
    "DirectionalEndpointPosteriorV2",
    "PHYSTWIN_DIRECTIONAL_ENDPOINT_SCHEMA",
    "PHYSTWIN_DIRECTIONAL_ENDPOINT_VERSION",
    "robust_directional_endpoint_v2",
]
