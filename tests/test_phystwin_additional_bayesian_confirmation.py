from bayesian_phystwin.phystwin_additional_bayesian_confirmation import (
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_PROCESS_STD_M,
)


def test_additional_bayesian_hyperparameters_are_frozen() -> None:
    assert FIXED_PROCESS_STD_M == 0.005
    assert FIXED_OBSERVATION_STD_M == 0.001
    assert FIXED_INLIER_PRIOR == 0.95
