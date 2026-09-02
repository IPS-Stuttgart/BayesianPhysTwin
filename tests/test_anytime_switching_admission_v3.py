import pytest

from bayesian_phystwin.anytime_switching_admission_v3 import (
    SwitchingAdmissionConfigV3,
    SwitchingAdmissionContractV3,
    SwitchingUnionAdmissionControllerV3,
    bounded_gain_score,
    bounded_harm_score,
    robust_switching_score,
)


def _contract(**overrides: str) -> SwitchingAdmissionContractV3:
    values = {
        "candidate_id": "candidate-sha256",
        "fallback_id": "fallback-sha256",
        "gain_score_id": "capped-coordinate-l1-v1",
        "harm_definition_id": "candidate-loss-gt-fallback-plus-margin-v1",
        "information_set_id": "prefix-and-planned-action-v1",
        "reveal_policy_id": "paired-delayed-shadow-outcome-v1",
    }
    values.update(overrides)
    return SwitchingAdmissionContractV3(**values)


def _config(**overrides: object) -> SwitchingAdmissionConfigV3:
    values: dict[str, object] = {
        "loss_cap": 1.0,
        "minimum_mean_gain": 0.0,
        "harmful_margin": 0.0,
        "maximum_harm_rate": 0.10,
        "total_alpha": 0.05,
        "epoch_alpha_continuation": 0.5,
        "minimum_resolved_trials": 1,
    }
    values.update(overrides)
    return SwitchingAdmissionConfigV3(**values)


def _resolve(
    controller: SwitchingUnionAdmissionControllerV3[object],
    *,
    index: int,
    candidate_loss: float,
    fallback_loss: float,
) -> None:
    trial_id = f"trial-{index}"
    controller.issue_trial(
        trial_id=trial_id,
        issued_step=2 * index,
        maturity_step=2 * index + 1,
    )
    controller.resolve_trial(
        trial_id=trial_id,
        resolved_step=2 * index + 1,
        candidate_loss=candidate_loss,
        fallback_loss=fallback_loss,
    )


def test_harm_score_is_bounded_and_boundary_fair() -> None:
    ceiling = 0.10
    safe = bounded_harm_score(harmful=False, maximum_harm_rate=ceiling)
    harmful = bounded_harm_score(harmful=True, maximum_harm_rate=ceiling)

    assert -1.0 <= harmful < 0.0 < safe <= 1.0
    assert ceiling * harmful + (1.0 - ceiling) * safe == pytest.approx(0.0)
    with pytest.raises(ValueError, match="literal bool"):
        bounded_harm_score(harmful=1, maximum_harm_rate=ceiling)  # type: ignore[arg-type]


def test_robust_score_is_the_pointwise_lower_envelope() -> None:
    assert robust_switching_score(gain_score=0.2, harm_score=0.1) == 0.1
    assert robust_switching_score(gain_score=-0.3, harm_score=0.1) == -0.3
    with pytest.raises(ValueError, match="component scores"):
        robust_switching_score(gain_score=2.0, harm_score=0.0)


def test_bounded_gain_score_respects_cap_and_margin() -> None:
    assert bounded_gain_score(
        candidate_loss=0.0,
        fallback_loss=1.0,
        loss_cap=1.0,
        minimum_mean_gain=0.0,
    ) == pytest.approx(1.0)
    assert bounded_gain_score(
        candidate_loss=100.0,
        fallback_loss=0.0,
        loss_cap=1.0,
        minimum_mean_gain=0.0,
    ) == pytest.approx(-1.0)
    assert bounded_gain_score(
        candidate_loss=0.4,
        fallback_loss=0.5,
        loss_cap=1.0,
        minimum_mean_gain=0.1,
    ) == pytest.approx(0.0)


def test_strong_joint_benefit_authorizes() -> None:
    controller = SwitchingUnionAdmissionControllerV3(_config(), _contract())

    for index in range(200):
        _resolve(
            controller,
            index=index,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        if controller.authorized:
            break

    assert controller.authorized is True
    snapshot = controller.snapshot()
    assert snapshot.maximum_log_e_value >= snapshot.log_threshold
    assert snapshot.selected_artifact_id == "candidate-sha256"


def test_fixed_path_can_fluctuate_above_one_without_authorizing() -> None:
    controller = SwitchingUnionAdmissionControllerV3(_config(), _contract())

    # This fixed path contains occasional favorable paired outcomes followed by
    # sustained unfavorable outcomes. An e-process is not pathwise monotone:
    # it may rise above one transiently under a null-compatible distribution.
    # The guarantee controls crossing of the registered 1/alpha boundary, not
    # every excursion above its initial value.
    index = 0
    for _ in range(30):
        _resolve(
            controller,
            index=index,
            candidate_loss=0.0,
            fallback_loss=0.5,
        )
        index += 1
        _resolve(
            controller,
            index=index,
            candidate_loss=0.05,
            fallback_loss=0.0,
        )
        index += 1
    for _ in range(120):
        _resolve(
            controller,
            index=index,
            candidate_loss=0.55,
            fallback_loss=0.50,
        )
        index += 1

    snapshot = controller.snapshot()
    assert controller.authorized is False
    assert snapshot.maximum_log_e_value > 0.0
    assert snapshot.maximum_log_e_value < snapshot.log_threshold
    assert snapshot.log_e_value < 0.0


def test_old_epoch_outcome_is_retained_but_ignored() -> None:
    controller = SwitchingUnionAdmissionControllerV3(_config(), _contract())
    controller.issue_trial(trial_id="old", issued_step=0, maturity_step=2)
    controller.start_new_epoch(reason="declared-shift")

    result = controller.resolve_trial(
        trial_id="old",
        resolved_step=2,
        candidate_loss=0.0,
        fallback_loss=1.0,
    )
    snapshot = controller.snapshot()

    assert result.used_for_current_epoch is False
    assert snapshot.resolved_current_epoch_count == 0
    assert snapshot.ignored_closed_epoch_outcome_count == 1


def test_selection_returns_exact_registered_object() -> None:
    controller: SwitchingUnionAdmissionControllerV3[object] = (
        SwitchingUnionAdmissionControllerV3(_config(), _contract())
    )
    fallback = object()
    candidate = object()

    assert controller.select(
        fallback=fallback,
        candidate=candidate,
        fallback_id="fallback-sha256",
        candidate_id="candidate-sha256",
    ) is fallback
    with pytest.raises(ValueError, match="candidate_id"):
        controller.select(
            fallback=fallback,
            candidate=candidate,
            fallback_id="fallback-sha256",
            candidate_id="wrong",
        )


def test_contract_and_configuration_are_content_addressed() -> None:
    first = SwitchingUnionAdmissionControllerV3(_config(), _contract())
    second = SwitchingUnionAdmissionControllerV3(
        _config(maximum_harm_rate=0.20),
        _contract(),
    )

    assert len(first.decision_contract_digest) == 64
    assert first.decision_contract_digest != second.decision_contract_digest
    assert first.theorem_boundary()["lifetime_false_admission_bound"] == 0.05
    with pytest.raises(ValueError, match="must differ"):
        _contract(candidate_id="same", fallback_id="same")
