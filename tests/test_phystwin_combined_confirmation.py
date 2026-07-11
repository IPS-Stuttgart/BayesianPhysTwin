import numpy as np
import pytest

from bayesian_phystwin.phystwin_combined_confirmation import (
    balanced_profile_temperatures,
    combined_profile_fit_end,
    matched_hierarchical_trajectory,
)


def test_balanced_profile_temperatures_normalize_fit_lengths() -> None:
    temperatures = balanced_profile_temperatures(
        {"single": 43, "double": 31, "stretch": 99}
    )

    assert temperatures["single"] == pytest.approx(0.7456647399)
    assert temperatures["double"] == pytest.approx(0.5375722543)
    assert temperatures["stretch"] == pytest.approx(1.716763006)
    assert sum(temperatures.values()) / 3.0 == pytest.approx(1.0)


def test_combined_profile_end_preserves_locked_cohort_contracts() -> None:
    assert (
        combined_profile_fit_end(
            81,
            cohort="main",
            main_fit_fraction=0.75,
            additional_holdout_frames=1,
        )
        == 60
    )
    assert (
        combined_profile_fit_end(
            57,
            cohort="additional",
            main_fit_fraction=0.75,
            additional_holdout_frames=1,
        )
        == 56
    )


def test_combined_profile_end_rejects_unknown_cohort() -> None:
    with pytest.raises(ValueError, match="main or additional"):
        combined_profile_fit_end(
            20,
            cohort="other",
            main_fit_fraction=0.75,
            additional_holdout_frames=1,
        )


def test_matched_hierarchy_is_identity_at_zero_parameter_update() -> None:
    released = np.arange(24, dtype=float).reshape(2, 4, 3)
    zero = released + 0.4
    posterior = zero.copy()

    matched = matched_hierarchical_trajectory(released, zero, posterior)

    np.testing.assert_allclose(matched, released, atol=2e-15)


def test_matched_hierarchy_transports_only_the_paired_parameter_delta() -> None:
    released = np.zeros((3, 2, 3), dtype=float)
    zero = np.full_like(released, 2.0)
    posterior = zero.copy()
    posterior[:, :, 1] += 0.25

    matched = matched_hierarchical_trajectory(released, zero, posterior)

    np.testing.assert_allclose(matched[:, :, 0], 0.0)
    np.testing.assert_allclose(matched[:, :, 1], 0.25)
