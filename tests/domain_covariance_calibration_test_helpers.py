from __future__ import annotations

import numpy as np

from bayesian_phystwin.domain_covariance_calibration import (
    DomainCovarianceCalibrationCertificateV1,
    fit_domain_covariance_calibration,
)

PARTITION_ID = "d" * 64


def _inputs() -> dict[str, object]:
    dynamic = (
        np.asarray(
            [
                [3.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, -3.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, -3.0],
            ]
        ),
        np.asarray(
            [
                [2.8, 0.3, 0.0],
                [-2.8, -0.3, 0.0],
                [0.0, 3.2, 0.2],
                [0.0, -3.2, -0.2],
                [0.2, 0.0, 2.9],
                [-0.2, 0.0, -2.9],
            ]
        ),
        np.asarray(
            [
                [3.1, 0.1, 0.1],
                [-3.1, -0.1, -0.1],
                [0.1, 2.9, 0.2],
                [-0.1, -2.9, -0.2],
                [0.2, 0.1, 3.0],
                [-0.2, -0.1, -3.0],
            ]
        ),
    )
    static = (
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        np.asarray(
            [
                [0.9, 0.1, 0.0],
                [-0.9, -0.1, 0.0],
                [0.0, 1.1, 0.1],
                [0.0, -1.1, -0.1],
                [0.1, 0.0, 1.0],
                [-0.1, 0.0, -1.0],
            ]
        ),
        np.asarray(
            [
                [1.05, 0.05, 0.0],
                [-1.05, -0.05, 0.0],
                [0.0, 0.95, 0.05],
                [0.0, -0.95, -0.05],
                [0.05, 0.0, 1.0],
                [-0.05, 0.0, -1.0],
            ]
        ),
    )
    groups: list[str] = []
    domains: list[str] = []
    samples: list[tuple[str, ...]] = []
    residuals: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for domain, domain_values in (("dynamic", dynamic), ("quasi-static", static)):
        for index, values in enumerate(domain_values):
            groups.append(f"{domain}-{index}")
            domains.append(domain)
            samples.append(tuple(f"sample-{item}" for item in range(len(values))))
            residuals.append(values)
            covariances.append(np.tile(np.eye(3), (len(values), 1, 1)))
    return {
        "calibration_partition_id": PARTITION_ID,
        "statistical_unit": "independent-physical-trial",
        "residual_definition": "prediction-minus-observation-m",
        "covariance_definition": "raw-predictive-covariance-m2",
        "group_ids": tuple(groups),
        "domain_ids": tuple(domains),
        "sample_ids": tuple(samples),
        "residuals": tuple(residuals),
        "covariances": tuple(covariances),
        "guard_frozen_before_application_outcomes": True,
        "application_outcomes_used_for_guard_selection": False,
        "calibration_groups_independent": True,
    }


def _certificate(**updates: object) -> DomainCovarianceCalibrationCertificateV1:
    arguments = _inputs()
    arguments.update(updates)
    return fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]


def _certificate_arguments(
    certificate: DomainCovarianceCalibrationCertificateV1,
) -> dict[str, object]:
    return {
        "calibration_partition_id": certificate.calibration_partition_id,
        "statistical_unit": certificate.statistical_unit,
        "residual_definition": certificate.residual_definition,
        "covariance_definition": certificate.covariance_definition,
        "dimension": certificate.dimension,
        "config": certificate.config,
        "calibration_data_id": certificate.calibration_data_id,
        "transforms": certificate.transforms,
        "fold_records": certificate.fold_records,
        "guard_certificate": certificate.guard_certificate,
        "metadata": certificate.metadata,
    }


def _guard_for_folds(
    certificate: DomainCovarianceCalibrationCertificateV1,
    folds: object,
    *,
    calibration_partition_id: str | None = None,
    statistical_unit: str | None = None,
    metric: str | None = None,
    domain_ids: tuple[str, ...] | None = None,
):
    from bayesian_phystwin.calibration_domain_guard import (
        fit_calibration_domain_guard,
    )
    from bayesian_phystwin.domain_covariance_calibration import (
        DOMAIN_COVARIANCE_CALIBRATION_GUARD_METRIC,
    )

    records = tuple(folds)
    domains = (
        tuple(item.domain_id for item in records) if domain_ids is None else domain_ids
    )
    metadata = {
        "covariance_calibration_data_id": certificate.calibration_data_id,
        "covariance_calibration_schema": "bayesian_phystwin.domain_covariance_calibration",
        "covariance_calibration_version": 1,
        "fit_config": certificate.config.descriptor(),
    }
    guard = certificate.guard_certificate
    return fit_calibration_domain_guard(
        calibration_partition_id=(
            certificate.calibration_partition_id
            if calibration_partition_id is None
            else calibration_partition_id
        ),
        statistical_unit=(
            certificate.statistical_unit
            if statistical_unit is None
            else statistical_unit
        ),
        metric=(
            DOMAIN_COVARIANCE_CALIBRATION_GUARD_METRIC if metric is None else metric
        ),
        group_ids=tuple(item.held_out_group_id for item in records),
        domain_ids=domains,
        candidate_losses=np.asarray([item.loss_ratio for item in records]),
        fallback_losses=np.ones(len(records)),
        guard_frozen_before_application_outcomes=(
            guard.guard_frozen_before_application_outcomes
        ),
        application_outcomes_used_for_guard_selection=(
            guard.application_outcomes_used_for_guard_selection
        ),
        calibration_groups_independent=guard.calibration_groups_independent,
        config=guard.config,
        metadata=metadata,
    )
