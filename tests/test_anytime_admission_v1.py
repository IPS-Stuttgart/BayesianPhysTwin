import math

import numpy as np
import pytest

from bayesian_phystwin.anytime_admission_v1 import (
    AnytimeAdmissionConfig,
    AnytimeAdmissionController,
    BernoulliHarmMixtureEProcess,
    BoundedGainMixtureEProcess,
    GeometricAlphaSpending,
)


def _config(**overrides: object) -> AnytimeAdmissionConfig:
    values: dict[str, object] = {
        "loss_cap": 0.02,
        "minimum_mean_gain": 0.00025,
        "harmful_margin": 0.0,
        "maximum_harm_rate": 0.10,
        "total_alpha_gain": 0.05,
        "total_alpha_harm": 0.05,
        "epoch_alpha_continuation": 0.5,
        "minimum_resolved_trials": 5,
    }
    values.update(overrides)
    return AnytimeAdmissionConfig(**values)


def test_geometric_alpha_spending_is_bounded_over_many_epochs() -> None:
    schedule = GeometricAlphaSpending(total_alpha=0.05, continuation=0.5)

    allocated = sum(schedule.alpha_for_epoch(index) for index in range(1000))

    assert allocated == pytest.approx(0.05)
    assert schedule.alpha_for_epoch(0) == pytest.approx(0.025)
    assert schedule.alpha_for_epoch(1) == pytest.approx(0.0125)
    assert schedule.cumulative_alpha_through(9) < 0.05


def test_bounded_gain_process_matches_registered_mixture() -> None:
    bets = (0.25, 0.50)
    process = BoundedGainMixtureEProcess(bets)
    scores = (0.4, -0.2, 0.6)

    for score in scores:
        process.update(score)

    component_wealth = [
        math.prod(1.0 + bet * score for score in scores) for bet in bets
    ]
    expected = sum(component_wealth) / len(component_wealth)
    assert math.exp(process.log_e_value) == pytest.approx(expected)
    assert process.count == len(scores)
    assert process.maximum_log_e_value >= process.log_e_value


def test_harm_process_is_boundary_fair_for_each_component() -> None:
    ceiling = 0.20
    fractions = (0.25, 0.50, 0.75)
    process = BernoulliHarmMixtureEProcess(
        maximum_harm_rate=ceiling,
        alternative_fractions=fractions,
    )

    for fraction in fractions:
        alternative = ceiling * fraction
        harmful_factor = alternative / ceiling
        safe_factor = (1.0 - alternative) / (1.0 - ceiling)
        expectation_at_boundary = (
            ceiling * harmful_factor + (1.0 - ceiling) * safe_factor
        )
        assert expectation_at_boundary == pytest.approx(1.0)

    process.update(False)
    process.update(True)
    assert process.count == 2
    assert process.harm_count == 1


def test_controller_rejects_early_and_duplicate_resolution() -> None:
    controller = AnytimeAdmissionController(_config())
    controller.issue_trial(trial_id="trial-0", issued_step=0, maturity_step=3)

    with pytest.raises(ValueError, match="before maturity"):
        controller.resolve_trial(
            trial_id="trial-0",
            resolved_step=2,
            candidate_loss=0.01,
            fallback_loss=0.02,
        )

    controller.resolve_trial(
        trial_id="trial-0",
        resolved_step=3,
        candidate_loss=0.01,
        fallback_loss=0.02,
    )
    with pytest.raises(ValueError, match="already resolved"):
        controller.resolve_trial(
            trial_id="trial-0",
            resolved_step=4,
            candidate_loss=0.01,
            fallback_loss=0.02,
        )


def test_late_outcome_from_closed_epoch_cannot_update_new_epoch() -> None:
    controller = AnytimeAdmissionController(_config(minimum_resolved_trials=1))
    controller.issue_trial(trial_id="old", issued_step=0, maturity_step=2)
    controller.start_new_epoch(reason="declared-domain-shift")

    resolved = controller.resolve_trial(
        trial_id="old",
        resolved_step=2,
        candidate_loss=0.0,
        fallback_loss=0.02,
    )
    snapshot = controller.snapshot()

    assert resolved.used_for_current_epoch is False
    assert snapshot.resolved_current_epoch_count == 0
    assert snapshot.ignored_closed_epoch_outcome_count == 1
    assert snapshot.gain_log_e_value == pytest.approx(0.0)
    assert snapshot.harm_log_e_value == pytest.approx(0.0)
    assert snapshot.authorized is False


def test_strong_benefit_with_no_harm_eventually_authorizes() -> None:
    controller = AnytimeAdmissionController(_config())

    for index in range(80):
        controller.issue_trial(
            trial_id=f"positive-{index}",
            issued_step=2 * index,
            maturity_step=2 * index + 1,
        )
        controller.resolve_trial(
            trial_id=f"positive-{index}",
            resolved_step=2 * index + 1,
            candidate_loss=0.002,
            fallback_loss=0.018,
        )
        if controller.authorized:
            break

    snapshot = controller.snapshot()
    assert snapshot.authorized is True
    assert snapshot.ever_authorized_in_epoch is True
    assert snapshot.utility_evidence_passed is True
    assert snapshot.harm_evidence_passed is True
    assert snapshot.minimum_evidence_passed is True
    assert snapshot.harm_count == 0


def test_persistently_harmful_candidate_never_authorizes() -> None:
    controller = AnytimeAdmissionController(_config())

    for index in range(100):
        controller.issue_trial(
            trial_id=f"harm-{index}",
            issued_step=2 * index,
            maturity_step=2 * index + 1,
        )
        controller.resolve_trial(
            trial_id=f"harm-{index}",
            resolved_step=2 * index + 1,
            candidate_loss=0.018,
            fallback_loss=0.002,
        )

    snapshot = controller.snapshot()
    assert snapshot.authorized is False
    assert snapshot.ever_authorized_in_epoch is False
    assert snapshot.harm_count == 100
    assert snapshot.empirical_harm_fraction == pytest.approx(1.0)


def test_gain_score_is_bounded_after_loss_clipping() -> None:
    controller = AnytimeAdmissionController(_config(minimum_resolved_trials=1))
    controller.issue_trial(trial_id="clip", issued_step=0, maturity_step=1)

    resolved = controller.resolve_trial(
        trial_id="clip",
        resolved_step=1,
        candidate_loss=1000.0,
        fallback_loss=0.0,
    )

    assert -1.0 <= resolved.bounded_gain_score <= 1.0
    assert resolved.harmful is True


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="loss_cap"):
        _config(loss_cap=0.0)
    with pytest.raises(ValueError, match="maximum_harm_rate"):
        _config(maximum_harm_rate=1.0)
    with pytest.raises(ValueError, match="gain bet fractions"):
        _config(gain_bet_fractions=(0.2, 1.0))
    with pytest.raises(ValueError, match="minimum_resolved_trials"):
        _config(minimum_resolved_trials=0)


def test_null_gain_simulation_does_not_systematically_cross() -> None:
    """Development smoke test, not the claim-bearing type-I experiment."""

    rng = np.random.default_rng(20260902)
    crossing_count = 0
    world_count = 200
    for world in range(world_count):
        controller = AnytimeAdmissionController(_config(minimum_resolved_trials=10))
        scores = rng.choice((-0.01, 0.01), size=120)
        for index, signed_gain in enumerate(scores):
            fallback = 0.01
            candidate = fallback - float(signed_gain)
            controller.issue_trial(
                trial_id=f"{world}-{index}",
                issued_step=2 * index,
                maturity_step=2 * index + 1,
            )
            controller.resolve_trial(
                trial_id=f"{world}-{index}",
                resolved_step=2 * index + 1,
                candidate_loss=candidate,
                fallback_loss=fallback,
            )
            if controller.authorized:
                crossing_count += 1
                break

    assert crossing_count <= 20
