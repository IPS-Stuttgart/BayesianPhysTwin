from bayesian_phystwin.phystwin_bayesian_confirmation import (
    BayesianAnchorConfirmationProtocol,
)


def test_bayesian_confirmation_protocol_freezes_reliability_grid() -> None:
    protocol = BayesianAnchorConfirmationProtocol()

    assert protocol.maximum_residual_m == 0.01
    assert protocol.process_std_candidates_m[-1] == 0.005
    assert protocol.observation_std_candidates_m == (0.001, 0.0025, 0.005)
    assert protocol.inlier_prior == 0.95
