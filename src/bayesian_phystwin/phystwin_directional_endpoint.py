"""Robust directional discrepancy filtering for multiview PhysTwin cues."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionalEndpointPosterior:
    """Per-identity posterior with a conservative scalar variance readout."""

    mean: np.ndarray
    covariance: np.ndarray
    variance: np.ndarray
    final_inlier_probability: np.ndarray
    update_count: np.ndarray
    source_update_count: np.ndarray
    tangent_update_count: np.ndarray


def _boolean_array(value: object, *, name: str) -> np.ndarray:
    """Copy a caller-provided boolean array without truth-value coercion."""

    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise ValueError(f"{name} must have boolean dtype")
    return np.array(raw, dtype=np.bool_, copy=True, order="C")


def _validate_filter_parameters(
    *,
    end_frame: int,
    frame_count: int,
    process_variance: float,
    observation_variance: float,
    initial_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
) -> None:
    if not 0 < end_frame <= frame_count:
        raise ValueError("end_frame must lie inside the residual sequence")
    variances = (process_variance, observation_variance, initial_variance)
    if not all(np.isfinite(value) for value in variances):
        raise ValueError("filter variances must be finite")
    if process_variance < 0.0 or observation_variance <= 0.0:
        raise ValueError(
            "process variance must be nonnegative and observation positive"
        )
    if initial_variance <= 0.0:
        raise ValueError("initial_variance must be positive")
    if not np.isfinite(inlier_prior) or not 0.0 < inlier_prior < 1.0:
        raise ValueError("inlier_prior must be finite and lie in (0, 1)")
    if (
        not np.isfinite(outlier_variance_multiplier)
        or outlier_variance_multiplier <= 1.0
    ):
        raise ValueError("outlier_variance_multiplier must be finite and exceed one")


def _cholesky_factor_and_logdet(
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor a batch of SPD matrices and return stable log determinants."""

    factor = np.linalg.cholesky(covariance)
    diagonal = np.diagonal(factor, axis1=1, axis2=2)
    log_determinant = 2.0 * np.sum(np.log(diagonal), axis=1)
    return factor, log_determinant


def _cholesky_solve(factor: np.ndarray, right: np.ndarray) -> np.ndarray:
    intermediate = np.linalg.solve(factor, right)
    return np.linalg.solve(np.swapaxes(factor, 1, 2), intermediate)


def _joseph_covariance_update(
    covariance: np.ndarray,
    observation_matrix: np.ndarray,
    gain: np.ndarray,
    *,
    observation_variance: float,
) -> np.ndarray:
    """Apply a covariance-stable Joseph-form linear Gaussian update."""

    state_dimension = covariance.shape[1]
    identity = np.eye(state_dimension, dtype=float)[None]
    residual_map = identity - np.einsum("mij,mjk->mik", gain, observation_matrix)
    propagated = np.einsum(
        "mij,mjk,mnk->min",
        residual_map,
        covariance,
        residual_map,
    )
    noise = observation_variance * np.einsum(
        "mij,mkj->mik",
        gain,
        gain,
    )
    return propagated + noise


def _repair_roundoff_psd(covariance: np.ndarray) -> np.ndarray:
    """Clip only roundoff-scale negative eigenvalues and fail on real defects."""

    symmetric = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    if len(symmetric) == 0:
        return symmetric
    if not np.all(np.isfinite(symmetric)):
        raise FloatingPointError(
            "directional endpoint covariance contains non-finite values"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = np.maximum(
        np.max(np.abs(eigenvalues), axis=1),
        np.finfo(np.float64).tiny,
    )
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise np.linalg.LinAlgError(
            "directional endpoint covariance lost positive semidefiniteness"
        )
    affected = np.any(eigenvalues < 0.0, axis=1)
    if not np.any(affected):
        return symmetric
    repaired = symmetric.copy()
    clipped = np.maximum(eigenvalues[affected], 0.0)
    affected_vectors = eigenvectors[affected]
    repaired[affected] = np.einsum(
        "mik,mk,mjk->mij",
        affected_vectors,
        clipped,
        affected_vectors,
    )
    return 0.5 * (repaired + np.swapaxes(repaired, 1, 2))


def _robust_linear_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    observation: np.ndarray,
    observation_matrix: np.ndarray,
    *,
    observation_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = observation.shape[1]
    predicted_observation = np.einsum(
        "mij,mj->mi",
        observation_matrix,
        mean,
    )
    innovation = observation - predicted_observation
    projected = np.einsum(
        "mij,mjk->mik",
        observation_matrix,
        covariance,
    )
    projected_covariance = np.einsum(
        "mij,mkj->mik",
        projected,
        observation_matrix,
    )
    identity = np.eye(dimension, dtype=float)[None]
    projected_covariance = 0.5 * (
        projected_covariance + np.swapaxes(projected_covariance, 1, 2)
    )
    inlier_innovation_covariance = (
        projected_covariance + observation_variance * identity
    )
    outlier_observation_variance = observation_variance * outlier_variance_multiplier
    outlier_innovation_covariance = (
        projected_covariance + outlier_observation_variance * identity
    )

    inlier_factor, inlier_log_determinant = _cholesky_factor_and_logdet(
        inlier_innovation_covariance
    )
    outlier_factor, outlier_log_determinant = _cholesky_factor_and_logdet(
        outlier_innovation_covariance
    )
    inlier_precision_innovation = _cholesky_solve(
        inlier_factor,
        innovation[:, :, None],
    )[:, :, 0]
    outlier_precision_innovation = _cholesky_solve(
        outlier_factor,
        innovation[:, :, None],
    )[:, :, 0]
    inlier_quadratic = np.einsum(
        "mi,mi->m",
        innovation,
        inlier_precision_innovation,
    )
    outlier_quadratic = np.einsum(
        "mi,mi->m",
        innovation,
        outlier_precision_innovation,
    )
    normalizer = dimension * np.log(2.0 * np.pi)
    log_inlier = np.log(inlier_prior) - 0.5 * (
        normalizer + inlier_log_determinant + inlier_quadratic
    )
    log_outlier = np.log1p(-inlier_prior) - 0.5 * (
        normalizer + outlier_log_determinant + outlier_quadratic
    )
    probability = np.exp(log_inlier - np.logaddexp(log_inlier, log_outlier))

    covariance_times_observation = np.einsum(
        "mij,mkj->mik",
        covariance,
        observation_matrix,
    )
    inlier_gain = np.swapaxes(
        _cholesky_solve(
            inlier_factor,
            np.swapaxes(covariance_times_observation, 1, 2),
        ),
        1,
        2,
    )
    outlier_gain = np.swapaxes(
        _cholesky_solve(
            outlier_factor,
            np.swapaxes(covariance_times_observation, 1, 2),
        ),
        1,
        2,
    )
    inlier_mean = mean + np.einsum(
        "mij,mj->mi",
        inlier_gain,
        innovation,
    )
    outlier_mean = mean + np.einsum(
        "mij,mj->mi",
        outlier_gain,
        innovation,
    )
    updated_mean = (
        probability[:, None] * inlier_mean + (1.0 - probability)[:, None] * outlier_mean
    )
    inlier_covariance = _joseph_covariance_update(
        covariance,
        observation_matrix,
        inlier_gain,
        observation_variance=observation_variance,
    )
    outlier_covariance = _joseph_covariance_update(
        covariance,
        observation_matrix,
        outlier_gain,
        observation_variance=outlier_observation_variance,
    )
    inlier_offset = inlier_mean - updated_mean
    outlier_offset = outlier_mean - updated_mean
    updated_covariance = probability[:, None, None] * (
        inlier_covariance + np.einsum("mi,mj->mij", inlier_offset, inlier_offset)
    ) + (1.0 - probability)[:, None, None] * (
        outlier_covariance + np.einsum("mi,mj->mij", outlier_offset, outlier_offset)
    )
    return updated_mean, _repair_roundoff_psd(updated_covariance), probability


def robust_directional_endpoint(
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
) -> DirectionalEndpointPosterior:
    """Filter source-normal and multiview-tangent innovations once each."""

    source = np.asarray(source_residual, dtype=float)
    multiview = np.asarray(multiview_residual, dtype=float)
    source_mask = _boolean_array(source_valid, name="source_valid")
    multiview_mask = _boolean_array(multiview_valid, name="multiview_valid")
    projectors = np.asarray(tangent_projectors, dtype=float)
    priority = _boolean_array(priority_identities, name="priority_identities")

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
    if not np.allclose(projectors, np.swapaxes(projectors, 1, 2), atol=1e-8):
        raise ValueError("tangent projectors must be symmetric")
    if not np.allclose(
        np.einsum("nij,njk->nik", projectors, projectors),
        projectors,
        atol=1e-7,
    ):
        raise ValueError("tangent projectors must be idempotent")
    _validate_filter_parameters(
        end_frame=end_frame,
        frame_count=len(source),
        process_variance=process_variance,
        observation_variance=observation_variance,
        initial_variance=initial_variance,
        inlier_prior=inlier_prior,
        outlier_variance_multiplier=outlier_variance_multiplier,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(projectors)
    if not np.allclose(eigenvalues[:, 0], 0.0, atol=1e-7):
        raise ValueError("each tangent projector must have one null direction")
    if not np.allclose(eigenvalues[:, 1:], 1.0, atol=1e-7):
        raise ValueError("each tangent projector must have rank two")
    normal_basis = np.swapaxes(eigenvectors[:, :, :1], 1, 2)
    tangent_basis = np.swapaxes(eigenvectors[:, :, 1:], 1, 2)

    point_count = source.shape[1]
    mean = np.zeros((point_count, 3), dtype=float)
    identity_3 = np.eye(3, dtype=float)
    covariance = np.repeat(
        (initial_variance * identity_3)[None],
        point_count,
        axis=0,
    )
    final_probability = np.zeros(point_count, dtype=float)
    source_update_count = np.zeros(point_count, dtype=np.int64)
    tangent_update_count = np.zeros(point_count, dtype=np.int64)

    for frame in range(end_frame):
        covariance += process_variance * identity_3[None]

        full_indices = np.flatnonzero(source_mask[frame] & ~priority)
        if len(full_indices):
            full_matrix = np.repeat(
                identity_3[None],
                len(full_indices),
                axis=0,
            )
            (
                mean[full_indices],
                covariance[full_indices],
                final_probability[full_indices],
            ) = _robust_linear_update(
                mean[full_indices],
                covariance[full_indices],
                source[frame, full_indices],
                full_matrix,
                observation_variance=observation_variance,
                inlier_prior=inlier_prior,
                outlier_variance_multiplier=outlier_variance_multiplier,
            )
            scalar_variance = (
                np.trace(
                    covariance[full_indices],
                    axis1=1,
                    axis2=2,
                )
                / 3.0
            )
            covariance[full_indices] = scalar_variance[:, None, None] * identity_3[None]
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
            ) = _robust_linear_update(
                mean[normal_indices],
                covariance[normal_indices],
                normal_observation,
                normal_matrix,
                observation_variance=observation_variance,
                inlier_prior=inlier_prior,
                outlier_variance_multiplier=outlier_variance_multiplier,
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
            ) = _robust_linear_update(
                mean[tangent_indices],
                covariance[tangent_indices],
                tangent_observation,
                tangent_matrix,
                observation_variance=observation_variance,
                inlier_prior=inlier_prior,
                outlier_variance_multiplier=outlier_variance_multiplier,
            )
            tangent_update_count[tangent_indices] += 1

    covariance = _repair_roundoff_psd(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    conservative_variance = np.maximum(eigenvalues[:, -1], 0.0)
    update_count = source_update_count + tangent_update_count
    return DirectionalEndpointPosterior(
        mean=mean,
        covariance=covariance,
        variance=conservative_variance,
        final_inlier_probability=final_probability,
        update_count=update_count,
        source_update_count=source_update_count,
        tangent_update_count=tangent_update_count,
    )
