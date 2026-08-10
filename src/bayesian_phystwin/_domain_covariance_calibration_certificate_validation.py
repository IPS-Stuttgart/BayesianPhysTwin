"""Validation helpers for domain covariance calibration certificates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
)
from ._domain_covariance_calibration_common import (
    DOMAIN_COVARIANCE_CALIBRATION_SCHEMA,
    DOMAIN_COVARIANCE_CALIBRATION_VERSION,
    DOMAIN_COVARIANCE_GUARD_METRIC,
    DomainCovarianceCalibrationConfigV1,
    _canonical_string,
    _matches_frozen_grid,
)
from ._domain_covariance_calibration_fold import DomainCovarianceFoldV1
from ._domain_covariance_calibration_transform import DomainCovarianceTransformV1
from ._portable_contracts import sha256_digest
from .calibration_domain_guard import (
    CalibrationDomainGuardCertificateV1,
    fit_calibration_domain_guard,
)


def normalize_certificate_fields(certificate: Any) -> tuple[object, ...]:
    """Validate and canonicalize fields shared by the certificate record."""

    predictor_id = sha256_digest(certificate.predictor_id, name="predictor_id")
    predictor_frozen = genuine_boolean(
        certificate.predictor_frozen_before_calibration_outcomes,
        name="predictor_frozen_before_calibration_outcomes",
    )
    partition_id = sha256_digest(
        certificate.calibration_partition_id,
        name="calibration_partition_id",
    )
    statistical_unit = _canonical_string(
        certificate.statistical_unit,
        name="statistical_unit",
    )
    residual_definition = _canonical_string(
        certificate.residual_definition,
        name="residual_definition",
    )
    covariance_definition = _canonical_string(
        certificate.covariance_definition,
        name="covariance_definition",
    )
    dimension = genuine_integer(certificate.dimension, name="dimension", minimum=1)
    if not isinstance(certificate.config, DomainCovarianceCalibrationConfigV1):
        raise TypeError("config must be a DomainCovarianceCalibrationConfigV1")
    calibration_data_id = sha256_digest(
        certificate.calibration_data_id,
        name="calibration_data_id",
    )
    transforms = tuple(certificate.transforms)
    if not transforms or any(
        not isinstance(item, DomainCovarianceTransformV1) for item in transforms
    ):
        raise TypeError("transforms must contain DomainCovarianceTransformV1 records")
    if len({item.domain_id for item in transforms}) != len(transforms):
        raise ValueError("transforms must not contain duplicate domains")
    if any(item.dimension != dimension for item in transforms):
        raise ValueError("all transforms must match the certificate dimension")
    transforms = tuple(sorted(transforms, key=lambda item: item.domain_id))

    folds = tuple(certificate.fold_records)
    if not folds or any(not isinstance(item, DomainCovarianceFoldV1) for item in folds):
        raise TypeError("fold_records must contain DomainCovarianceFoldV1 records")
    if len({item.held_out_group_id for item in folds}) != len(folds):
        raise ValueError("fold_records must have unique held-out groups")
    folds = tuple(sorted(folds, key=lambda item: item.held_out_group_id))
    _validate_rosters_and_grid(certificate.config, dimension, transforms, folds)
    metadata = frozen_finite_json_mapping(
        certificate.metadata,
        name="domain covariance calibration metadata",
    )
    return (
        predictor_id,
        predictor_frozen,
        partition_id,
        statistical_unit,
        residual_definition,
        covariance_definition,
        dimension,
        calibration_data_id,
        transforms,
        folds,
        metadata,
    )


def _validate_rosters_and_grid(
    config: DomainCovarianceCalibrationConfigV1,
    dimension: int,
    transforms: tuple[DomainCovarianceTransformV1, ...],
    folds: tuple[DomainCovarianceFoldV1, ...],
) -> None:
    transform_domains = {item.domain_id for item in transforms}
    if {item.domain_id for item in folds} != transform_domains:
        raise ValueError("fold and transform domain rosters must match")
    scale_grid = config.scale_grid()
    floor_grid = config.floor_ratio_grid()
    domain_groups: dict[str, set[str]] = {}
    for transform in transforms:
        if transform.dimension != dimension:
            raise ValueError("all transforms must match the certificate dimension")
        fold_groups = {
            item.held_out_group_id
            for item in folds
            if item.domain_id == transform.domain_id
        }
        if fold_groups != set(transform.group_ids):
            raise ValueError(
                "each transform group roster must match its held-out folds"
            )
        if not _matches_frozen_grid(transform.scale, scale_grid):
            raise ValueError("transform scale is outside the frozen grid")
        floor_ratio = transform.isotropic_floor_variance / transform.reference_variance
        if not _matches_frozen_grid(floor_ratio, floor_grid):
            raise ValueError("transform floor is outside the frozen grid")
        if (
            transform.calibrated_group_balanced_nll_per_dimension
            > transform.raw_group_balanced_nll_per_dimension + config.score_tolerance
        ):
            raise ValueError("final covariance transform worsens fitted NLL")
        domain_groups[transform.domain_id] = set(transform.group_ids)
    for fold in folds:
        expected_training = domain_groups[fold.domain_id] - {fold.held_out_group_id}
        if set(fold.training_group_ids) != expected_training:
            raise ValueError(
                "fold training roster must contain every other domain group"
            )
        if not _matches_frozen_grid(fold.scale, scale_grid):
            raise ValueError("fold scale is outside the frozen grid")
        floor_ratio = fold.isotropic_floor_variance / fold.reference_variance
        if not _matches_frozen_grid(floor_ratio, floor_grid):
            raise ValueError("fold floor is outside the frozen grid")


def validate_guard_binding(certificate: Any) -> None:
    """Rebuild the embedded domain guard and verify every numerical binding."""

    guard = certificate.guard_certificate
    if not isinstance(guard, CalibrationDomainGuardCertificateV1):
        raise TypeError(
            "guard_certificate must be a CalibrationDomainGuardCertificateV1"
        )
    transform_domains = {item.domain_id for item in certificate.transforms}
    if guard.calibration_partition_id != certificate.calibration_partition_id:
        raise ValueError("guard and covariance calibration partitions differ")
    if guard.statistical_unit != certificate.statistical_unit:
        raise ValueError("guard and covariance statistical units differ")
    if guard.metric != DOMAIN_COVARIANCE_GUARD_METRIC:
        raise ValueError("guard metric does not match covariance calibration")
    if {item.domain_id for item in guard.decisions} != transform_domains:
        raise ValueError("guard and transform domain rosters must match")

    folds = certificate.fold_records
    expected = fit_calibration_domain_guard(
        calibration_partition_id=certificate.calibration_partition_id,
        statistical_unit=certificate.statistical_unit,
        metric=DOMAIN_COVARIANCE_GUARD_METRIC,
        group_ids=tuple(item.held_out_group_id for item in folds),
        domain_ids=tuple(item.domain_id for item in folds),
        candidate_losses=np.asarray([item.loss_ratio for item in folds]),
        fallback_losses=np.ones(len(folds), dtype=np.float64),
        guard_frozen_before_application_outcomes=(
            guard.guard_frozen_before_application_outcomes
        ),
        application_outcomes_used_for_guard_selection=(
            guard.application_outcomes_used_for_guard_selection
        ),
        calibration_groups_independent=guard.calibration_groups_independent,
        config=guard.config,
        metadata={
            "predictor_id": certificate.predictor_id,
            "predictor_frozen_before_calibration_outcomes": (
                certificate.predictor_frozen_before_calibration_outcomes
            ),
            "covariance_calibration_data_id": certificate.calibration_data_id,
            "covariance_calibration_schema": DOMAIN_COVARIANCE_CALIBRATION_SCHEMA,
            "covariance_calibration_version": DOMAIN_COVARIANCE_CALIBRATION_VERSION,
            "fit_config": certificate.config.descriptor(),
        },
    )
    if expected.artifact_id != guard.artifact_id:
        raise ValueError("guard certificate is not bound to the covariance folds")
    for fold in folds:
        expected_ratio = math.exp(
            float(
                np.clip(
                    fold.log_loss_ratio,
                    -certificate.config.log_loss_ratio_clip,
                    certificate.config.log_loss_ratio_clip,
                )
            )
        )
        if not math.isclose(
            fold.loss_ratio,
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1e-12 * max(1.0, expected_ratio),
        ):
            raise ValueError("fold loss_ratio does not match the frozen clip")
