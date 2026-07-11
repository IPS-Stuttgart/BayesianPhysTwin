import numpy as np

from bayesian_phystwin.phystwin_motioncrafter_assimilation_evaluation import (
    _exact_sign_test,
    _paired_summary,
)


def test_exact_sign_test_counts_wins_losses_and_ties() -> None:
    result = _exact_sign_test(np.array([-1.0, -2.0, 3.0, 0.0]))

    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["ties"] == 1
    assert result["two_sided_p"] == 1.0


def test_paired_summary_uses_candidate_minus_reference_direction() -> None:
    result = _paired_summary(
        np.array([2.0, 4.0, 6.0]),
        np.array([1.0, 3.0, 5.0]),
        bootstrap_samples=1000,
        bootstrap_seed=7,
    )

    assert result["candidate_minus_reference_mean_m"] == -1.0
    assert result["relative_change_percent"] == -25.0
    assert result["sign_test"]["wins"] == 3
