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


def test_fail_closed_validation_and_defensive_branches() -> None:
    import bayesian_phystwin.anytime_admission_v1 as admission

    with pytest.raises(ValueError, match="positive and finite"):
        admission._positive(float("nan"), label="positive")
    with pytest.raises(ValueError, match="nonnegative and finite"):
        admission._nonnegative(-1.0, label="nonnegative")
    with pytest.raises(ValueError, match="positive"):
        admission._literal_positive_integer(0, label="count")
    with pytest.raises(ValueError, match="nonnegative literal integer"):
        admission._literal_nonnegative_integer(True, label="index")
    with pytest.raises(ValueError, match="at least one component"):
        admission._weights(0)
    with pytest.raises(ValueError, match="aligned one-dimensional"):
        admission._log_mixture(
            np.asarray([0.0]),
            np.asarray([0.5, 0.5]),
        )
    assert (
        admission._log_mixture(
            np.asarray([-math.inf]),
            np.asarray([1.0]),
        )
        == -math.inf
    )


def test_configuration_rejects_empty_duplicate_and_invalid_mixtures() -> None:
    with pytest.raises(ValueError, match="gain_bet_fractions must not be empty"):
        _config(gain_bet_fractions=())
    with pytest.raises(
        ValueError, match="harm_alternative_fractions must not be empty"
    ):
        _config(harm_alternative_fractions=())
    with pytest.raises(ValueError, match="gain bet fractions must be unique"):
        _config(gain_bet_fractions=(0.2, 0.2))
    with pytest.raises(ValueError, match="harm alternative fractions must be unique"):
        _config(harm_alternative_fractions=(0.2, 0.2))
    with pytest.raises(ValueError, match="harm alternative fractions"):
        _config(harm_alternative_fractions=(0.0, 0.5))
    with pytest.raises(ValueError, match="minimum_mean_gain"):
        _config(minimum_mean_gain=-0.1)
    with pytest.raises(ValueError, match="harmful_margin"):
        _config(harmful_margin=-0.1)
    with pytest.raises(ValueError, match="total_alpha_gain"):
        _config(total_alpha_gain=float("nan"))


def test_e_process_constructors_and_updates_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="bet_fractions must not be empty"):
        BoundedGainMixtureEProcess(())
    with pytest.raises(ValueError, match="finite vector"):
        BoundedGainMixtureEProcess((0.0,))
    with pytest.raises(ValueError, match="alternative_fractions must not be empty"):
        BernoulliHarmMixtureEProcess(
            maximum_harm_rate=0.1,
            alternative_fractions=(),
        )
    with pytest.raises(ValueError, match="alternative fractions"):
        BernoulliHarmMixtureEProcess(
            maximum_harm_rate=0.1,
            alternative_fractions=(1.0,),
        )

    gain = BoundedGainMixtureEProcess((0.5,))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        gain.update(1.1)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        gain.update(float("nan"))

    harm = BernoulliHarmMixtureEProcess(
        maximum_harm_rate=0.1,
        alternative_fractions=(0.5,),
    )
    with pytest.raises(ValueError, match="literal bool"):
        harm.update(1)  # type: ignore[arg-type]

    gain.__dict__["_bets"] = np.asarray((2.0,))
    with pytest.raises(ValueError, match="not positive"):
        gain.update(-1.0)
    harm.__dict__["_alternatives"] = np.asarray((float("nan"),))
    with pytest.raises(ValueError, match="invalid"):
        harm.update(False)


def test_controller_rejects_malformed_epoch_and_trial_operations() -> None:
    with pytest.raises(TypeError, match="AnytimeAdmissionConfig"):
        AnytimeAdmissionController(object())  # type: ignore[arg-type]

    controller = AnytimeAdmissionController(_config())
    assert controller.epoch_index == 0
    assert controller.resolved_trials == ()
    with pytest.raises(ValueError, match="nonempty literal string"):
        controller.start_new_epoch(reason="")
    with pytest.raises(ValueError, match="nonempty literal string"):
        controller.start_new_epoch(reason=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty literal string"):
        controller.issue_trial(trial_id="", issued_step=0, maturity_step=1)
    with pytest.raises(ValueError, match="nonnegative literal integer"):
        controller.issue_trial(trial_id="negative", issued_step=-1, maturity_step=1)
    with pytest.raises(ValueError, match="nonnegative literal integer"):
        controller.issue_trial(trial_id="bool", issued_step=True, maturity_step=1)
    with pytest.raises(ValueError, match="strictly after"):
        controller.issue_trial(trial_id="same", issued_step=1, maturity_step=1)

    controller.issue_trial(trial_id="duplicate", issued_step=0, maturity_step=1)
    with pytest.raises(ValueError, match="already registered"):
        controller.issue_trial(trial_id="duplicate", issued_step=0, maturity_step=1)
    with pytest.raises(ValueError, match="unknown pending"):
        controller.resolve_trial(
            trial_id="unknown",
            resolved_step=1,
            candidate_loss=0.0,
            fallback_loss=0.0,
        )
    with pytest.raises(ValueError, match="candidate_loss"):
        controller.resolve_trial(
            trial_id="duplicate",
            resolved_step=1,
            candidate_loss=-1.0,
            fallback_loss=0.0,
        )
    resolved = controller.resolve_trial(
        trial_id="duplicate",
        resolved_step=1,
        candidate_loss=0.0,
        fallback_loss=0.0,
    )
    assert resolved.trial_id == "duplicate"
    assert len(controller.resolved_trials) == 1
