"""Calibration-frozen domain covariance inflation with exact array fallback.

The fitted transform is intentionally conservative::

    covariance_calibrated = scale * covariance_raw + floor_variance * I

``scale`` is constrained to be at least one and ``floor_variance`` is
nonnegative. Parameters are selected by group-balanced Gaussian negative log
likelihood. Domain support is then decided from leave-one-group-out predictive
loss ratios through :mod:`bayesian_phystwin.calibration_domain_guard`.
"""

from ._domain_covariance_calibration_application_record import (
    DomainCovarianceCalibrationApplicationV1,
)
from ._domain_covariance_calibration_apply_api import (
    apply_domain_covariance_calibration,
)
from ._domain_covariance_calibration_certificate_record import (
    DomainCovarianceCalibrationCertificateV1,
)
from ._domain_covariance_calibration_common import (
    DOMAIN_COVARIANCE_APPLICATION_SCHEMA,
    DOMAIN_COVARIANCE_APPLICATION_VERSION,
    DOMAIN_COVARIANCE_CALIBRATION_SCHEMA,
    DOMAIN_COVARIANCE_CALIBRATION_VERSION,
    DOMAIN_COVARIANCE_DATA_SCHEMA,
    DOMAIN_COVARIANCE_DATA_VERSION,
    DOMAIN_COVARIANCE_FOLD_SCHEMA,
    DOMAIN_COVARIANCE_FOLD_VERSION,
    DOMAIN_COVARIANCE_GUARD_METRIC,
    DOMAIN_COVARIANCE_TRANSFORM_SCHEMA,
    DOMAIN_COVARIANCE_TRANSFORM_VERSION,
    DomainCovarianceCalibrationConfigV1,
)
from ._domain_covariance_calibration_fit_api import (
    fit_domain_covariance_calibration,
)
from ._domain_covariance_calibration_fold import DomainCovarianceFoldV1
from ._domain_covariance_calibration_transform import DomainCovarianceTransformV1

__all__ = [
    "DOMAIN_COVARIANCE_APPLICATION_SCHEMA",
    "DOMAIN_COVARIANCE_APPLICATION_VERSION",
    "DOMAIN_COVARIANCE_CALIBRATION_SCHEMA",
    "DOMAIN_COVARIANCE_CALIBRATION_VERSION",
    "DOMAIN_COVARIANCE_DATA_SCHEMA",
    "DOMAIN_COVARIANCE_DATA_VERSION",
    "DOMAIN_COVARIANCE_FOLD_SCHEMA",
    "DOMAIN_COVARIANCE_FOLD_VERSION",
    "DOMAIN_COVARIANCE_GUARD_METRIC",
    "DOMAIN_COVARIANCE_TRANSFORM_SCHEMA",
    "DOMAIN_COVARIANCE_TRANSFORM_VERSION",
    "DomainCovarianceCalibrationApplicationV1",
    "DomainCovarianceCalibrationCertificateV1",
    "DomainCovarianceCalibrationConfigV1",
    "DomainCovarianceFoldV1",
    "DomainCovarianceTransformV1",
    "apply_domain_covariance_calibration",
    "fit_domain_covariance_calibration",
]
