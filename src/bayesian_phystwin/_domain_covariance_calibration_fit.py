"""Numerical fitting primitives for domain covariance calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._domain_covariance_calibration_common import (
    _COORDINATE_COVERAGE_Z90,
    _GAUSSIAN_CONSTANT,
    DOMAIN_COVARIANCE_DATA_SCHEMA,
    DOMAIN_COVARIANCE_DATA_VERSION,
    DomainCovarianceCalibrationConfigV1,
    _array_record,
    _canonical_float_array,
    _canonical_string,
    _canonical_strings,
)
from ._portable_contracts import content_id


@dataclass(frozen=True, slots=True)
class _CalibrationGroup:
    group_id: str
    domain_id: str
    sample_ids: tuple[str, ...]
    residuals: np.ndarray
    covariances: np.ndarray
    eigenvalues: np.ndarray
    projected_residual_sq: np.ndarray


@dataclass(frozen=True, slots=True)
class _FitResult:
    scale: float
    floor_variance: float
    reference_variance: float
    score: float


def _prepare_group(
    *,
    group_id: str,
    domain_id: str,
    sample_ids: Sequence[str],
    residuals: object,
    covariances: object,
    config: DomainCovarianceCalibrationConfigV1,
    expected_dimension: int | None,
) -> _CalibrationGroup:
    canonical_group = _canonical_string(group_id, name="group_id")
    canonical_domain = _canonical_string(domain_id, name="domain_id")
    samples = _canonical_strings(sample_ids, name="sample_ids")
    if len(set(samples)) != len(samples):
        raise ValueError(f"sample_ids for {canonical_group!r} contain duplicates")
    residual_array = _canonical_float_array(
        residuals,
        name=f"residuals[{canonical_group}]",
        ndim=2,
    )
    covariance_array = _canonical_float_array(
        covariances,
        name=f"covariances[{canonical_group}]",
        ndim=3,
    )
    if residual_array.shape[0] != len(samples):
        raise ValueError("sample_ids and residual rows must have equal lengths")
    dimension = residual_array.shape[1]
    if dimension == 0:
        raise ValueError("residual dimension must be positive")
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError("all calibration groups must have the same dimension")
    if covariance_array.shape != (len(samples), dimension, dimension):
        raise ValueError(
            "each covariance array must have shape (sample_count, dimension, dimension)"
        )
    transpose = np.swapaxes(covariance_array, -1, -2)
    if not np.allclose(
        covariance_array,
        transpose,
        rtol=0.0,
        atol=config.symmetry_tolerance,
    ):
        raise ValueError("calibration covariances must be symmetric")
    covariance_array = 0.5 * (covariance_array + transpose)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_array)
    if np.any(eigenvalues <= config.minimum_eigenvalue):
        raise ValueError("calibration covariances must be positive definite")
    projected = np.einsum(
        "nij,ni->nj",
        eigenvectors,
        residual_array,
        optimize=True,
    )
    order = tuple(sorted(range(len(samples)), key=samples.__getitem__))
    canonical_samples = tuple(samples[index] for index in order)
    residual_array = residual_array[np.asarray(order)]
    covariance_array = covariance_array[np.asarray(order)]
    eigenvalues = eigenvalues[np.asarray(order)]
    projected_sq = np.square(projected[np.asarray(order)])
    return _CalibrationGroup(
        group_id=canonical_group,
        domain_id=canonical_domain,
        sample_ids=canonical_samples,
        residuals=residual_array,
        covariances=covariance_array,
        eigenvalues=eigenvalues,
        projected_residual_sq=projected_sq,
    )


def _reference_variance(groups: Sequence[_CalibrationGroup]) -> float:
    group_medians = [
        float(np.median(np.sum(group.eigenvalues, axis=1) / group.residuals.shape[1]))
        for group in groups
    ]
    result = float(np.median(np.asarray(group_medians, dtype=np.float64)))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("calibration covariance reference variance is invalid")
    return result


def _group_nll(
    group: _CalibrationGroup,
    *,
    scale: float,
    floor_variance: float,
) -> float:
    eigenvalues = scale * group.eigenvalues + floor_variance
    dimension = group.residuals.shape[1]
    values = 0.5 * (
        np.sum(np.log(eigenvalues), axis=1)
        + np.sum(group.projected_residual_sq / eigenvalues, axis=1)
        + dimension * _GAUSSIAN_CONSTANT
    )
    return float(np.mean(values) / dimension)


def _group_normalized_energy(
    group: _CalibrationGroup,
    *,
    scale: float,
    floor_variance: float,
) -> float:
    eigenvalues = scale * group.eigenvalues + floor_variance
    dimension = group.residuals.shape[1]
    return float(
        np.mean(np.sum(group.projected_residual_sq / eigenvalues, axis=1)) / dimension
    )


def _group_coordinate_coverage(
    group: _CalibrationGroup,
    *,
    scale: float,
    floor_variance: float,
) -> float:
    variances = (
        scale * np.diagonal(group.covariances, axis1=-2, axis2=-1) + floor_variance
    )
    covered = np.abs(group.residuals) <= (_COORDINATE_COVERAGE_Z90 * np.sqrt(variances))
    return float(np.mean(covered))


def _group_balanced_score(
    groups: Sequence[_CalibrationGroup],
    *,
    scale: float,
    floor_variance: float,
) -> float:
    return float(
        np.mean(
            np.asarray(
                [
                    _group_nll(
                        group,
                        scale=scale,
                        floor_variance=floor_variance,
                    )
                    for group in groups
                ],
                dtype=np.float64,
            )
        )
    )


def _group_balanced_energy(
    groups: Sequence[_CalibrationGroup],
    *,
    scale: float,
    floor_variance: float,
) -> float:
    return float(
        np.mean(
            np.asarray(
                [
                    _group_normalized_energy(
                        group,
                        scale=scale,
                        floor_variance=floor_variance,
                    )
                    for group in groups
                ],
                dtype=np.float64,
            )
        )
    )


def _fit_scale_and_floor(
    groups: Sequence[_CalibrationGroup],
    config: DomainCovarianceCalibrationConfigV1,
) -> _FitResult:
    if not groups:
        raise ValueError("at least one training group is required")
    reference = _reference_variance(groups)
    best: _FitResult | None = None
    for floor_ratio in config.floor_ratio_grid():
        floor_variance = floor_ratio * reference
        for scale in config.scale_grid():
            score = _group_balanced_score(
                groups,
                scale=scale,
                floor_variance=floor_variance,
            )
            candidate = _FitResult(
                scale=scale,
                floor_variance=floor_variance,
                reference_variance=reference,
                score=score,
            )
            if best is None or score < best.score - config.score_tolerance:
                best = candidate
    if best is None:
        raise AssertionError("covariance calibration grid was empty")
    return best


def _calibration_data_id(
    *,
    calibration_partition_id: str,
    statistical_unit: str,
    residual_definition: str,
    covariance_definition: str,
    dimension: int,
    groups: Sequence[_CalibrationGroup],
) -> str:
    records = [
        {
            "group_id": group.group_id,
            "domain_id": group.domain_id,
            "sample_ids": list(group.sample_ids),
            "residuals": _array_record(group.residuals),
            "covariances": _array_record(group.covariances),
        }
        for group in sorted(groups, key=lambda item: item.group_id)
    ]
    return content_id(
        {
            "schema": DOMAIN_COVARIANCE_DATA_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_DATA_VERSION,
            "calibration_partition_id": calibration_partition_id,
            "statistical_unit": statistical_unit,
            "residual_definition": residual_definition,
            "covariance_definition": covariance_definition,
            "dimension": dimension,
            "records": records,
        }
    )
