"""Supported uncertainty contracts and estimators.

This namespace groups stable uncertainty-facing interfaces without expanding the
package root with every research module.
"""

from .group_sandwich_covariance import (
    GroupSandwichCovarianceResultV1,
    estimate_group_sandwich_covariance,
)
from .observed_information_covariance import (
    ObservedInformationCovarianceResultV1,
    observed_information_covariance_from_prior_aware_result,
)
from .posterior_covariance_portfolio import (
    PosteriorCovarianceSourceV1,
    PosteriorQueryCovariancePortfolioV1,
    build_posterior_query_covariance_portfolio,
    exact_prior_fallback_covariance_source,
    group_sandwich_covariance_source,
    observed_information_covariance_source,
    working_covariance_source,
)
from .posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
    working_irls_covariance_semantics,
)
from .posterior_uncertainty import (
    FiniteGroupCoverageStatusV1,
    PosteriorQueryUncertaintyV1,
    finite_group_coverage_status,
)
from .query_calibration import (
    QueryCalibrationV1,
    calibrate_query_covariance,
    fit_query_calibration,
    group_mahalanobis_nonconformity,
    load_query_calibration,
    query_group_is_covered,
    save_query_calibration,
)

__all__ = [
    "FiniteGroupCoverageStatusV1",
    "GroupSandwichCovarianceResultV1",
    "ObservedInformationCovarianceResultV1",
    "PosteriorCovarianceSemanticsV1",
    "PosteriorCovarianceSourceV1",
    "PosteriorQueryCovariancePortfolioV1",
    "PosteriorQueryUncertaintyV1",
    "QueryCalibrationV1",
    "build_posterior_query_covariance_portfolio",
    "calibrate_query_covariance",
    "estimate_group_sandwich_covariance",
    "exact_prior_fallback_covariance_source",
    "finite_group_coverage_status",
    "fit_query_calibration",
    "group_mahalanobis_nonconformity",
    "group_sandwich_covariance_source",
    "load_query_calibration",
    "observed_information_covariance_from_prior_aware_result",
    "observed_information_covariance_source",
    "query_group_is_covered",
    "save_query_calibration",
    "working_covariance_source",
    "working_irls_covariance_semantics",
]
