from __future__ import annotations

import bayesian_phystwin
from bayesian_phystwin import uncertainty


def test_uncertainty_namespace_is_narrow_and_explicit() -> None:
    assert uncertainty.__all__ == [
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
    assert not hasattr(bayesian_phystwin, "PosteriorQueryUncertaintyV1")
    assert not hasattr(bayesian_phystwin, "PosteriorQueryCovariancePortfolioV1")
