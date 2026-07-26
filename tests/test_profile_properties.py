import numpy as np
from hypothesis import assume, given, strategies as st

from bayesian_phystwin.phystwin_profile import truncate_profile_prediction_weights


@given(
    raw_weights=st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=9,
        max_size=9,
    ),
    retained_integer=st.integers(min_value=1, max_value=1_000),
)
def test_profile_truncation_accounts_for_every_unit_of_posterior_mass(
    raw_weights: list[int],
    retained_integer: int,
) -> None:
    assume(any(raw_weights))
    weights = np.asarray(raw_weights, dtype=np.float64).reshape(3, 3)
    retained_mass = retained_integer / 1_000.0

    truncated, kept_mass, count = truncate_profile_prediction_weights(
        weights,
        retained_mass=retained_mass,
    )

    normalized = weights.reshape(-1) / np.sum(weights)
    order = np.argsort(-normalized, kind="stable")
    cumulative = np.cumsum(normalized[order])
    expected_count = min(
        int(np.searchsorted(cumulative, retained_mass, side="left")) + 1,
        len(normalized),
    )
    selected = order[:expected_count]
    omitted = order[expected_count:]
    flat_truncated = truncated.reshape(-1)

    np.testing.assert_allclose(np.sum(truncated), 1.0)
    assert count == expected_count
    np.testing.assert_allclose(kept_mass, np.sum(normalized[selected]))
    np.testing.assert_allclose(kept_mass, 1.0 - np.sum(normalized[omitted]))
    assert kept_mass >= retained_mass - 1e-15
    np.testing.assert_allclose(flat_truncated[omitted], 0.0)
    np.testing.assert_allclose(
        flat_truncated[selected],
        normalized[selected] / kept_mass,
    )
    if count > 1:
        assert cumulative[count - 2] < retained_mass + 1e-15
