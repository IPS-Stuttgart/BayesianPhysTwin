"""Apply-or-exact-fallback API for domain covariance calibration."""

from __future__ import annotations

import numpy as np

from ._canonical_contracts import immutable_array
from ._domain_covariance_calibration_application_record import (
    DomainCovarianceCalibrationApplicationV1,
)
from ._domain_covariance_calibration_certificate_record import (
    DomainCovarianceCalibrationCertificateV1,
)
from ._domain_covariance_calibration_common import _array_id, _canonical_string


def _validate_application_covariance(
    covariance: np.ndarray,
    *,
    certificate: DomainCovarianceCalibrationCertificateV1,
) -> np.ndarray:
    if not isinstance(covariance, np.ndarray):
        raise TypeError("covariance must be a NumPy array for exact fallback")
    raw = np.asarray(covariance)
    if raw.ndim < 2 or raw.shape[-2:] != (
        certificate.dimension,
        certificate.dimension,
    ):
        raise ValueError("covariance trailing axes do not match the certificate")
    if raw.dtype.kind not in "iuf":
        raise ValueError("covariance must be numeric")
    canonical = np.array(raw, dtype=np.dtype("<f8"), copy=True, order="C")
    if not np.all(np.isfinite(canonical)):
        raise ValueError("covariance must contain only finite values")
    transpose = np.swapaxes(canonical, -1, -2)
    if not np.allclose(
        canonical,
        transpose,
        rtol=0.0,
        atol=certificate.config.symmetry_tolerance,
    ):
        raise ValueError("covariance must be symmetric")
    symmetric = 0.5 * (canonical + transpose)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.any(eigenvalues <= certificate.config.minimum_eigenvalue):
        raise ValueError("covariance must be positive definite")
    return symmetric


def apply_domain_covariance_calibration(
    covariance: np.ndarray,
    certificate: DomainCovarianceCalibrationCertificateV1,
    *,
    domain_id: str,
) -> tuple[np.ndarray, DomainCovarianceCalibrationApplicationV1]:
    """Apply an authorized transform or return the exact input array object."""

    if not isinstance(certificate, DomainCovarianceCalibrationCertificateV1):
        raise TypeError(
            "certificate must be a DomainCovarianceCalibrationCertificateV1"
        )
    domain = _canonical_string(domain_id, name="domain_id")
    symmetric = _validate_application_covariance(covariance, certificate=certificate)
    input_id = _array_id(np.asarray(covariance))
    transform = certificate.transform_for_domain(domain)
    decision = certificate.guard_certificate.decision_for_domain(domain)
    supported = bool(decision is not None and decision.calibration_supported)
    applied = bool(
        transform is not None and supported and certificate.deployment_admissible
    )
    if transform is None or decision is None:
        reason = "unknown-calibration-domain"
    elif not certificate.deployment_admissible:
        reason = "calibration-information-boundary-rejected"
    elif not supported:
        reason = "calibration-domain-rejected"
    else:
        reason = "domain-covariance-calibration-applied"
    if applied:
        identity = np.eye(certificate.dimension, dtype=np.float64)
        calibrated = (
            transform.scale * symmetric
            + transform.isotropic_floor_variance * identity
        )
        selected = immutable_array(calibrated, dtype=np.dtype("<f8"))
        output_id = _array_id(selected)
    else:
        selected = covariance
        output_id = input_id
        if selected is not covariance:
            raise AssertionError("rejected calibration did not reuse input array")
    application = DomainCovarianceCalibrationApplicationV1(
        certificate_id=str(certificate.artifact_id),
        domain_id=domain,
        domain_decision_id=None if decision is None else decision.artifact_id,
        transform_id=None if transform is None else transform.artifact_id,
        input_covariance_id=input_id,
        output_covariance_id=output_id,
        calibration_applied=applied,
        reason=reason,
    )
    return selected, application
