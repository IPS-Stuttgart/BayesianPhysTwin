from __future__ import annotations

import bayesian_phystwin
import bayesian_phystwin.observed_information_covariance as observed_information


def test_observed_information_covariance_remains_explicitly_opt_in() -> None:
    assert (
        "observed_information_covariance_from_prior_aware_result"
        in observed_information.__all__
    )
    assert "ObservedInformationCovarianceResultV1" in observed_information.__all__
    assert not hasattr(
        bayesian_phystwin,
        "observed_information_covariance_from_prior_aware_result",
    )
    assert not hasattr(
        bayesian_phystwin,
        "ObservedInformationCovarianceResultV1",
    )
