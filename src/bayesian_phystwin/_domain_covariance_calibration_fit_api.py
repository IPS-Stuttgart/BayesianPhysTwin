"""Public fit entry point for calibration-frozen domain covariance transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._canonical_contracts import genuine_boolean
from ._domain_covariance_calibration_build import build_calibration_certificate
from ._domain_covariance_calibration_certificate_record import (
    DomainCovarianceCalibrationCertificateV1,
)
from ._domain_covariance_calibration_common import (
    DomainCovarianceCalibrationConfigV1,
    _canonical_string,
    _canonical_strings,
)
from ._domain_covariance_calibration_fit import _CalibrationGroup, _prepare_group
from ._portable_contracts import sha256_digest
from .calibration_domain_guard import CalibrationDomainGuardConfigV1


def fit_domain_covariance_calibration(
    *,
    predictor_id: str,
    predictor_frozen_before_calibration_outcomes: bool,
    calibration_partition_id: str,
    statistical_unit: str,
    residual_definition: str,
    covariance_definition: str,
    group_ids: Sequence[str],
    domain_ids: Sequence[str],
    sample_ids: Sequence[Sequence[str]],
    residuals: Sequence[object],
    covariances: Sequence[object],
    guard_frozen_before_application_outcomes: bool,
    application_outcomes_used_for_guard_selection: bool,
    calibration_groups_independent: bool,
    config: DomainCovarianceCalibrationConfigV1 | None = None,
    guard_config: CalibrationDomainGuardConfigV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DomainCovarianceCalibrationCertificateV1:
    """Fit source-only domain transforms and a leave-one-group-out guard."""

    fit_config = DomainCovarianceCalibrationConfigV1() if config is None else config
    if not isinstance(fit_config, DomainCovarianceCalibrationConfigV1):
        raise TypeError("config must be a DomainCovarianceCalibrationConfigV1")
    domain_guard_config = (
        CalibrationDomainGuardConfigV1() if guard_config is None else guard_config
    )
    if not isinstance(domain_guard_config, CalibrationDomainGuardConfigV1):
        raise TypeError("guard_config must be a CalibrationDomainGuardConfigV1")
    predictor = sha256_digest(predictor_id, name="predictor_id")
    predictor_frozen = genuine_boolean(
        predictor_frozen_before_calibration_outcomes,
        name="predictor_frozen_before_calibration_outcomes",
    )
    partition_id = sha256_digest(
        calibration_partition_id,
        name="calibration_partition_id",
    )
    unit = _canonical_string(statistical_unit, name="statistical_unit")
    residual_name = _canonical_string(
        residual_definition,
        name="residual_definition",
    )
    covariance_name = _canonical_string(
        covariance_definition,
        name="covariance_definition",
    )
    frozen_before = genuine_boolean(
        guard_frozen_before_application_outcomes,
        name="guard_frozen_before_application_outcomes",
    )
    application_used = genuine_boolean(
        application_outcomes_used_for_guard_selection,
        name="application_outcomes_used_for_guard_selection",
    )
    independent = genuine_boolean(
        calibration_groups_independent,
        name="calibration_groups_independent",
    )
    groups_input = _canonical_strings(group_ids, name="group_ids")
    domains_input = _canonical_strings(domain_ids, name="domain_ids")
    if len(set(groups_input)) != len(groups_input):
        raise ValueError("group_ids must not contain duplicates")
    sample_input = tuple(sample_ids)
    residual_input = tuple(residuals)
    covariance_input = tuple(covariances)
    count = len(groups_input)
    if not (
        count
        == len(domains_input)
        == len(sample_input)
        == len(residual_input)
        == len(covariance_input)
    ):
        raise ValueError(
            "group_ids, domain_ids, sample_ids, residuals, and covariances "
            "must have equal lengths"
        )
    prepared: list[_CalibrationGroup] = []
    dimension: int | None = None
    for index in range(count):
        group = _prepare_group(
            group_id=groups_input[index],
            domain_id=domains_input[index],
            sample_ids=sample_input[index],
            residuals=residual_input[index],
            covariances=covariance_input[index],
            config=fit_config,
            expected_dimension=dimension,
        )
        if dimension is None:
            dimension = group.residuals.shape[1]
        prepared.append(group)
    if dimension is None:
        raise AssertionError("calibration groups were unexpectedly empty")
    groups = tuple(sorted(prepared, key=lambda item: item.group_id))
    return build_calibration_certificate(
        predictor_id=predictor,
        predictor_frozen_before_calibration_outcomes=predictor_frozen,
        partition_id=partition_id,
        statistical_unit=unit,
        residual_definition=residual_name,
        covariance_definition=covariance_name,
        dimension=dimension,
        groups=groups,
        fit_config=fit_config,
        guard_config=domain_guard_config,
        frozen_before=frozen_before,
        application_used=application_used,
        independent=independent,
        metadata=metadata,
    )
