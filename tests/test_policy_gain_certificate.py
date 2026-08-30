from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.policy_gain_certificate import (
    apply_policy_gain_guard,
    calibrate_policy_gain_lower_bound,
    fit_local_policy_gain_predictor,
    predict_distance_weighted_local_policy_gain,
    predict_local_policy_gain,
)
from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_v1 import (
    bias_invariant_features,
    posterior_policy_action,
)


def _predictor():
    return fit_local_policy_gain_predictor(
        reference_ids=("c", "a", "b"),
        reference_features=np.asarray([[2.0], [0.0], [1.0]]),
        reference_action_gains=np.asarray(
            [[0.2, -0.2], [0.0, 0.4], [0.1, 0.3]]
        ),
        neighbor_count=2,
    )


def test_local_predictor_is_canonical_and_uses_candidate_action() -> None:
    predictor = _predictor()
    prediction = predict_local_policy_gain(
        predictor,
        query_features=np.asarray([[0.1], [1.9]]),
        candidate_actions=np.asarray([1, 0]),
    )

    assert predictor.reference_ids == ("a", "b", "c")
    assert prediction.neighbor_indices.tolist() == [[0, 1], [2, 1]]
    assert prediction.predicted_gain == pytest.approx([0.35, 0.15])
    assert not prediction.predicted_gain.flags.writeable
    assert not prediction.neighbor_indices.flags.writeable


def test_reference_row_order_does_not_change_local_prediction() -> None:
    first = _predictor()
    second = fit_local_policy_gain_predictor(
        reference_ids=("b", "c", "a"),
        reference_features=np.asarray([[1.0], [2.0], [0.0]]),
        reference_action_gains=np.asarray(
            [[0.1, 0.3], [0.2, -0.2], [0.0, 0.4]]
        ),
        neighbor_count=2,
    )
    kwargs = {
        "query_features": np.asarray([[0.1], [1.9]]),
        "candidate_actions": np.asarray([1, 0]),
    }

    assert predict_local_policy_gain(first, **kwargs).predicted_gain.tobytes() == (
        predict_local_policy_gain(second, **kwargs).predicted_gain.tobytes()
    )


def test_distance_weighted_prediction_uses_local_distance_and_candidate_action() -> None:
    predictor = _predictor()
    prediction = predict_distance_weighted_local_policy_gain(
        predictor,
        query_features=np.asarray([[0.25], [1.75]]),
        candidate_actions=np.asarray([1, 0]),
    )

    # Standardization is a common scale factor, so the 1:3 distance ratio gives
    # weights 3:1 for the nearer and farther rows.
    assert prediction.neighbor_indices.tolist() == [[0, 1], [2, 1]]
    assert prediction.predicted_gain == pytest.approx([0.375, 0.175])
    assert not prediction.predicted_gain.flags.writeable


def test_distance_weighted_prediction_averages_only_exact_matches() -> None:
    predictor = fit_local_policy_gain_predictor(
        reference_ids=("a", "b", "c"),
        reference_features=np.asarray([[0.0], [0.0], [1.0]]),
        reference_action_gains=np.asarray([[0.2, 0.0], [0.4, 0.0], [0.9, 0.0]]),
        neighbor_count=3,
    )

    prediction = predict_distance_weighted_local_policy_gain(
        predictor,
        query_features=np.asarray([[0.0]]),
        candidate_actions=np.asarray([0]),
    )

    assert prediction.predicted_gain == pytest.approx([0.3])


def test_split_conformal_rank_and_guard_are_exact() -> None:
    predicted = np.arange(19, dtype=np.float64) / 100.0
    realized = np.zeros(19, dtype=np.float64)
    calibration = calibrate_policy_gain_lower_bound(
        predicted_gain=predicted,
        realized_gain=realized,
        miscoverage=0.10,
    )
    decision = apply_policy_gain_guard(
        candidate_actions=np.asarray([2, 3]),
        predicted_gain=np.asarray([0.20, 0.10]),
        calibration=calibration,
        fallback_action=5,
        harm_margin=0.002,
    )

    assert calibration.rank == 18
    assert calibration.offset == pytest.approx(0.17)
    assert decision.lower_gain_bound == pytest.approx([0.03, -0.07])
    assert decision.accepted_mask.tolist() == [True, False]
    assert decision.selected_actions.tolist() == [2, 5]
    assert not decision.selected_actions.flags.writeable


def test_unsupported_conformal_level_fails_closed() -> None:
    with pytest.raises(ValueError, match="cannot support"):
        calibrate_policy_gain_lower_bound(
            predicted_gain=np.zeros(8),
            realized_gain=np.zeros(8),
            miscoverage=0.10,
        )


def test_harmful_admitted_update_is_a_lower_bound_failure() -> None:
    calibration = calibrate_policy_gain_lower_bound(
        predicted_gain=np.linspace(0.0, 0.18, 19),
        realized_gain=np.zeros(19),
        miscoverage=0.10,
    )
    predicted = np.asarray([0.20, 0.30])
    realized = np.asarray([0.01, -0.01])
    decision = apply_policy_gain_guard(
        candidate_actions=np.asarray([1, 1]),
        predicted_gain=predicted,
        calibration=calibration,
        fallback_action=0,
        harm_margin=0.002,
    )

    harmful = decision.accepted_mask & (realized < -0.002)
    lower_bound_failure = realized < decision.lower_gain_bound
    assert np.all(~harmful | lower_bound_failure)


def test_slingshot_features_cancel_shared_xyz_bias() -> None:
    rng = np.random.default_rng(260930)
    observation = rng.normal(size=(3, 4, 3))
    shifted = observation + np.asarray([0.1, -0.2, 0.3])

    np.testing.assert_allclose(
        bias_invariant_features(observation),
        bias_invariant_features(shifted),
        rtol=0.0,
        atol=5e-16,
    )
    assert bias_invariant_features(observation).shape == (51,)


def test_posterior_policy_uses_registered_action_order() -> None:
    losses = np.asarray(
        [
            [0.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0],
        ]
    )

    assert posterior_policy_action(losses).tolist() == [0, 6]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference_ids": ("a", "a")}, "duplicates"),
        ({"neighbor_count": 3}, "supported"),
        ({"reference_features": [[0.0], [np.nan]]}, "finite"),
    ],
)
def test_local_predictor_rejects_invalid_inputs(kwargs: dict, message: str) -> None:
    values = {
        "reference_ids": ("a", "b"),
        "reference_features": [[0.0], [1.0]],
        "reference_action_gains": [[0.0, 1.0], [1.0, 0.0]],
        "neighbor_count": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        fit_local_policy_gain_predictor(**values)
