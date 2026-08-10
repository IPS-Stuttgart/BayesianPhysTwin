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
from domain_covariance_calibration_test_helpers import _certificate, _inputs

def test_batch_covariance_application_preserves_shape_and_psd() -> None:
    raw = np.stack((np.eye(3), 2.0 * np.eye(3)))

    calibrated, record = apply_domain_covariance_calibration(
        raw,
        _certificate(),
        domain_id="dynamic",
    )

    assert record.calibration_applied
    assert calibrated.shape == raw.shape
    assert np.all(np.linalg.eigvalsh(calibrated) > 0.0)

@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("group_ids", ("duplicate",) * 6, "duplicates"),
        ("domain_ids", ("dynamic",), "equal lengths"),
        ("sample_ids", (("sample", "sample"),) * 6, "duplicates"),
    ),
)
def test_malformed_rosters_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    arguments = _inputs()
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

def test_non_positive_definite_calibration_covariance_is_rejected() -> None:
    arguments = _inputs()
    covariances = list(arguments["covariances"])
    malformed = np.asarray(covariances[0]).copy()
    malformed[0, 2, 2] = 0.0
    covariances[0] = malformed
    arguments["covariances"] = tuple(covariances)

    with pytest.raises(ValueError, match="positive definite"):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

def test_single_group_domain_cannot_be_cross_fitted() -> None:
    arguments = _inputs()
    arguments["domain_ids"] = (
        "singleton",
        "dynamic",
        "dynamic",
        "quasi-static",
        "quasi-static",
        "quasi-static",
    )

    with pytest.raises(ValueError, match="at least two groups"):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

def test_forged_fold_loss_ratio_breaks_guard_binding() -> None:
    certificate = _certificate()
    folds = list(certificate.fold_records)
    folds[0] = replace(folds[0], loss_ratio=0.99, artifact_id=None)

    with pytest.raises(ValueError, match="guard certificate|loss_ratio"):
        DomainCovarianceCalibrationCertificateV1(
            calibration_partition_id=certificate.calibration_partition_id,
            statistical_unit=certificate.statistical_unit,
            residual_definition=certificate.residual_definition,
            covariance_definition=certificate.covariance_definition,
            dimension=certificate.dimension,
            config=certificate.config,
            calibration_data_id=certificate.calibration_data_id,
            transforms=certificate.transforms,
            fold_records=folds,
            guard_certificate=certificate.guard_certificate,
            metadata=certificate.metadata,
        )

def test_config_rejects_deflation_and_invalid_floor_grid() -> None:
    with pytest.raises(ValueError, match="minimum_scale"):
        DomainCovarianceCalibrationConfigV1(minimum_scale=0.9)
    with pytest.raises(ValueError, match="minimum_positive_floor_ratio"):
        DomainCovarianceCalibrationConfigV1(
            floor_grid_size=2,
            minimum_positive_floor_ratio=0.0,
        )

def test_application_rejects_nonarray_and_dimension_mismatch() -> None:
    certificate = _certificate()
    with pytest.raises(TypeError, match="NumPy array"):
        apply_domain_covariance_calibration(  # type: ignore[arg-type]
            [[1.0, 0.0], [0.0, 1.0]],
            certificate,
            domain_id="dynamic",
        )
    with pytest.raises(ValueError, match="trailing axes"):
        apply_domain_covariance_calibration(
            np.eye(2),
            certificate,
            domain_id="dynamic",
        )
