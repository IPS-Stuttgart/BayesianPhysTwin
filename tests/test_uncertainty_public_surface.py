from __future__ import annotations

import bayesian_phystwin
from bayesian_phystwin import uncertainty


def test_uncertainty_namespace_is_narrow_and_explicit() -> None:
    assert uncertainty.__all__ == [
        "FiniteGroupCoverageStatusV1",
        "GroupSandwichCovarianceResultV1",
        "ObservedInformationCovarianceResultV1",
        "PosteriorCovarianceSemanticsV1",
        "PosteriorQueryUncertaintyV1",
        "QueryCalibrationV1",
        "calibrate_query_covariance",
        "estimate_group_sandwich_covariance",
        "finite_group_coverage_status",
        "fit_query_calibration",
        "group_mahalanobis_nonconformity",
        "load_query_calibration",
        "observed_information_covariance_from_prior_aware_result",
        "query_group_is_covered",
        "save_query_calibration",
        "working_irls_covariance_semantics",
    ]
    assert not hasattr(bayesian_phystwin, "PosteriorQueryUncertaintyV1")
