"""Build cross-fitted transforms and guards from prepared calibration groups."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._domain_covariance_calibration_certificate_record import (
    DomainCovarianceCalibrationCertificateV1,
)
from ._domain_covariance_calibration_common import (
    DOMAIN_COVARIANCE_CALIBRATION_SCHEMA,
    DOMAIN_COVARIANCE_CALIBRATION_VERSION,
    DOMAIN_COVARIANCE_GUARD_METRIC,
    DomainCovarianceCalibrationConfigV1,
)
from ._domain_covariance_calibration_fit import (
    _CalibrationGroup,
    _calibration_data_id,
    _fit_scale_and_floor,
    _group_balanced_energy,
    _group_balanced_score,
    _group_coordinate_coverage,
    _group_nll,
    _group_normalized_energy,
)
from ._domain_covariance_calibration_fold import DomainCovarianceFoldV1
from ._domain_covariance_calibration_transform import DomainCovarianceTransformV1
from .calibration_domain_guard import (
    CalibrationDomainGuardConfigV1,
    fit_calibration_domain_guard,
)


def build_calibration_certificate(
    *,
    partition_id: str,
    statistical_unit: str,
    residual_definition: str,
    covariance_definition: str,
    dimension: int,
    groups: Sequence[_CalibrationGroup],
    fit_config: DomainCovarianceCalibrationConfigV1,
    guard_config: CalibrationDomainGuardConfigV1,
    frozen_before: bool,
    application_used: bool,
    independent: bool,
    metadata: Mapping[str, Any] | None,
) -> DomainCovarianceCalibrationCertificateV1:
    """Build final and leave-one-group-out records without application outcomes."""

    by_domain = {
        domain: tuple(group for group in groups if group.domain_id == domain)
        for domain in sorted({group.domain_id for group in groups})
    }
    undersized = [
        domain for domain, domain_groups in by_domain.items() if len(domain_groups) < 2
    ]
    if undersized:
        raise ValueError(
            "each domain needs at least two groups for leave-one-group-out "
            f"calibration: {undersized}"
        )
    data_id = _calibration_data_id(
        calibration_partition_id=partition_id,
        statistical_unit=statistical_unit,
        residual_definition=residual_definition,
        covariance_definition=covariance_definition,
        dimension=dimension,
        groups=groups,
    )
    folds: list[DomainCovarianceFoldV1] = []
    transforms: list[DomainCovarianceTransformV1] = []
    for domain, domain_groups in by_domain.items():
        for held_out in domain_groups:
            training = tuple(group for group in domain_groups if group is not held_out)
            fitted = _fit_scale_and_floor(training, fit_config)
            raw_nll = _group_nll(held_out, scale=1.0, floor_variance=0.0)
            calibrated_nll = _group_nll(
                held_out,
                scale=fitted.scale,
                floor_variance=fitted.floor_variance,
            )
            log_ratio = calibrated_nll - raw_nll
            clipped_ratio = float(
                np.clip(
                    log_ratio,
                    -fit_config.log_loss_ratio_clip,
                    fit_config.log_loss_ratio_clip,
                )
            )
            folds.append(
                DomainCovarianceFoldV1(
                    domain_id=domain,
                    held_out_group_id=held_out.group_id,
                    training_group_ids=tuple(group.group_id for group in training),
                    scale=fitted.scale,
                    isotropic_floor_variance=fitted.floor_variance,
                    reference_variance=fitted.reference_variance,
                    raw_nll_per_dimension=raw_nll,
                    calibrated_nll_per_dimension=calibrated_nll,
                    log_loss_ratio=log_ratio,
                    loss_ratio=math.exp(clipped_ratio),
                    raw_normalized_energy=_group_normalized_energy(
                        held_out,
                        scale=1.0,
                        floor_variance=0.0,
                    ),
                    calibrated_normalized_energy=_group_normalized_energy(
                        held_out,
                        scale=fitted.scale,
                        floor_variance=fitted.floor_variance,
                    ),
                    raw_coordinate_coverage_90=_group_coordinate_coverage(
                        held_out,
                        scale=1.0,
                        floor_variance=0.0,
                    ),
                    calibrated_coordinate_coverage_90=_group_coordinate_coverage(
                        held_out,
                        scale=fitted.scale,
                        floor_variance=fitted.floor_variance,
                    ),
                )
            )
        transforms.append(
            _final_transform(domain, domain_groups, dimension, fit_config)
        )
    folds.sort(key=lambda item: item.held_out_group_id)
    guard = fit_calibration_domain_guard(
        calibration_partition_id=partition_id,
        statistical_unit=statistical_unit,
        metric=DOMAIN_COVARIANCE_GUARD_METRIC,
        group_ids=tuple(item.held_out_group_id for item in folds),
        domain_ids=tuple(item.domain_id for item in folds),
        candidate_losses=np.asarray([item.loss_ratio for item in folds]),
        fallback_losses=np.ones(len(folds), dtype=np.float64),
        guard_frozen_before_application_outcomes=frozen_before,
        application_outcomes_used_for_guard_selection=application_used,
        calibration_groups_independent=independent,
        config=guard_config,
        metadata={
            "covariance_calibration_data_id": data_id,
            "covariance_calibration_schema": DOMAIN_COVARIANCE_CALIBRATION_SCHEMA,
            "covariance_calibration_version": DOMAIN_COVARIANCE_CALIBRATION_VERSION,
            "fit_config": fit_config.descriptor(),
        },
    )
    return DomainCovarianceCalibrationCertificateV1(
        calibration_partition_id=partition_id,
        statistical_unit=statistical_unit,
        residual_definition=residual_definition,
        covariance_definition=covariance_definition,
        dimension=dimension,
        config=fit_config,
        calibration_data_id=data_id,
        transforms=transforms,
        fold_records=folds,
        guard_certificate=guard,
        metadata={} if metadata is None else metadata,
    )


def _final_transform(
    domain: str,
    groups: Sequence[_CalibrationGroup],
    dimension: int,
    config: DomainCovarianceCalibrationConfigV1,
) -> DomainCovarianceTransformV1:
    fitted = _fit_scale_and_floor(groups, config)
    return DomainCovarianceTransformV1(
        domain_id=domain,
        dimension=dimension,
        group_ids=tuple(group.group_id for group in groups),
        scale=fitted.scale,
        isotropic_floor_variance=fitted.floor_variance,
        reference_variance=fitted.reference_variance,
        raw_group_balanced_nll_per_dimension=_group_balanced_score(
            groups,
            scale=1.0,
            floor_variance=0.0,
        ),
        calibrated_group_balanced_nll_per_dimension=fitted.score,
        raw_mean_normalized_energy=_group_balanced_energy(
            groups,
            scale=1.0,
            floor_variance=0.0,
        ),
        calibrated_mean_normalized_energy=_group_balanced_energy(
            groups,
            scale=fitted.scale,
            floor_variance=fitted.floor_variance,
        ),
    )
