from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.domain_covariance_calibration import (
    DomainCovarianceCalibrationCertificateV1,
    DomainCovarianceCalibrationConfigV1,
    apply_domain_covariance_calibration,
    fit_domain_covariance_calibration,
)
from domain_covariance_calibration_test_helpers import (
    _certificate,
    _certificate_arguments,
    _guard_for_folds,
    _inputs,
)

def test_certificate_and_fold_loss_identities_reject_forgery() -> None:
    certificate = _certificate()
    assert (
        replace(certificate, artifact_id=certificate.artifact_id)
        == certificate
    )
    with pytest.raises(ValueError, match="artifact_id"):
        replace(certificate, artifact_id="0" * 64)

    folds = list(certificate.fold_records)
    folds[0] = replace(folds[0], loss_ratio=0.99, artifact_id=None)
    guard = _guard_for_folds(certificate, folds)
    arguments = _certificate_arguments(certificate)
    arguments["fold_records"] = folds
    arguments["guard_certificate"] = guard
    with pytest.raises(ValueError, match="loss_ratio"):
        DomainCovarianceCalibrationCertificateV1(  # type: ignore[arg-type]
            **arguments
        )

def test_application_record_contract_rejects_inconsistent_ids() -> None:
    from bayesian_phystwin.domain_covariance_calibration import (
        DomainCovarianceCalibrationApplicationV1,
    )

    certificate = _certificate()
    _, applied = apply_domain_covariance_calibration(
        np.eye(3),
        certificate,
        domain_id="dynamic",
    )
    assert replace(applied, artifact_id=applied.artifact_id) == applied
    with pytest.raises(ValueError, match="artifact_id"):
        replace(applied, artifact_id="0" * 64)
    with pytest.raises(ValueError, match="requires decision"):
        replace(applied, domain_decision_id=None, artifact_id=None)

    _, rejected = apply_domain_covariance_calibration(
        np.eye(3),
        certificate,
        domain_id="quasi-static",
    )
    with pytest.raises(ValueError, match="preserve covariance ID"):
        replace(rejected, output_covariance_id="f" * 64, artifact_id=None)
    with pytest.raises(ValueError, match="canonical string"):
        DomainCovarianceCalibrationApplicationV1(
            certificate_id="a" * 64,
            domain_id="domain",
            domain_decision_id=None,
            transform_id=None,
            input_covariance_id="b" * 64,
            output_covariance_id="b" * 64,
            calibration_applied=False,
            reason="",
        )

def test_application_covariance_validation_fails_closed() -> None:
    certificate = _certificate()
    malformed = (
        (np.asarray([["x"] * 3] * 3), "numeric"),
        (np.full((3, 3), np.nan), "finite"),
        (np.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, 1.0, 0.0]), "positive definite"),
    )
    for covariance, match in malformed:
        with pytest.raises(ValueError, match=match):
            apply_domain_covariance_calibration(
                covariance,
                certificate,
                domain_id="dynamic",
            )
    with pytest.raises(TypeError, match="certificate must"):
        apply_domain_covariance_calibration(  # type: ignore[arg-type]
            np.eye(3),
            object(),
            domain_id="dynamic",
        )
