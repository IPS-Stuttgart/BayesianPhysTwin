import pytest

from bayesian_phystwin.anytime_factor_envelope_controller_v4 import (
    FactorEnvelopeAdmissionConfigV4,
    FactorEnvelopeAdmissionContractV4,
    FactorEnvelopeAdmissionControllerV4,
)


def _contract(**overrides: str) -> FactorEnvelopeAdmissionContractV4:
    values = {
        "candidate_id": "candidate-sha256",
        "fallback_id": "fallback-sha256",
        "gain_score_id": "capped-coordinate-l1-v1",
        "harm_definition_id": "candidate-loss-gt-fallback-plus-margin-v1",
        "information_set_id": "prefix-and-planned-action-v1",
        "reveal_policy_id": "paired-delayed-shadow-outcome-v1",
        "factor_family_id": "lower-envelope-cartesian-v1",
    }
    values.update(overrides)
    return FactorEnvelopeAdmissionContractV4(**values)


def _config(**overrides: object) -> FactorEnvelopeAdmissionConfigV4:
    values: dict[str, object] = {
        "loss_cap": 1.0,
        "minimum_mean_gain": 0.0,
        "harmful_margin": 0.1,
        "maximum_harm_rate": 0.10,
        "total_alpha": 0.05,
        "epoch_alpha_continuation": 0.5,
        "minimum_resolved_trials": 1,
        "gain_bet_fractions": (0.1, 0.4, 0.8),
        "harm_alternative_fractions": (0.1, 0.5, 0.9),
    }
    values.update(overrides)
    return FactorEnvelopeAdmissionConfigV4(**values)


def _resolve(
    controller: FactorEnvelopeAdmissionControllerV4[object],
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


def test_strong_safe_benefit_authorizes_and_returns_exact_candidate() -> None:
    controller: FactorEnvelopeAdmissionControllerV4[object] = (
        FactorEnvelopeAdmissionControllerV4(_config(), _contract())
    )
    fallback = object()
    candidate = object()

    for index in range(300):
        _resolve(
            controller,
            index=index,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        if controller.authorized:
            break

    snapshot = controller.snapshot()
    assert controller.authorized is True
    assert snapshot.maximum_log_e_value >= snapshot.log_threshold
    assert snapshot.factor_component_count == 9
    assert snapshot.selected_artifact_id == "candidate-sha256"
    assert (
        controller.select(
            fallback=fallback,
            candidate=candidate,
            fallback_id="fallback-sha256",
            candidate_id="candidate-sha256",
        )
        is candidate
    )


def test_unadmitted_selection_returns_exact_registered_fallback() -> None:
    controller: FactorEnvelopeAdmissionControllerV4[object] = (
        FactorEnvelopeAdmissionControllerV4(_config(), _contract())
    )
    fallback = object()
    candidate = object()

    assert (
        controller.select(
            fallback=fallback,
            candidate=candidate,
            fallback_id="fallback-sha256",
            candidate_id="candidate-sha256",
        )
        is fallback
    )
    with pytest.raises(ValueError, match="fallback_id"):
        controller.select(
            fallback=fallback,
            candidate=candidate,
            fallback_id="wrong",
            candidate_id="candidate-sha256",
        )


def test_malformed_reveal_is_atomic_and_retryable() -> None:
    controller = FactorEnvelopeAdmissionControllerV4(_config(), _contract())
    controller.issue_trial(trial_id="retry", issued_step=0, maturity_step=1)

    with pytest.raises(ValueError, match="candidate_loss"):
        controller.resolve_trial(
            trial_id="retry",
            resolved_step=1,
            candidate_loss=-1.0,
            fallback_loss=1.0,
        )

    rejected = controller.snapshot()
    assert rejected.pending_trial_count == 1
    assert rejected.resolved_current_epoch_count == 0
    result = controller.resolve_trial(
        trial_id="retry",
        resolved_step=1,
        candidate_loss=0.0,
        fallback_loss=1.0,
    )
    assert result.used_for_current_epoch is True
    assert controller.snapshot().pending_trial_count == 0


def test_outcome_from_closed_epoch_is_retained_but_cannot_update() -> None:
    controller = FactorEnvelopeAdmissionControllerV4(_config(), _contract())
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
    assert result.log_e_value is None
    assert snapshot.resolved_current_epoch_count == 0
    assert snapshot.ignored_closed_epoch_outcome_count == 1
    assert snapshot.pending_trial_count == 0


def test_switching_invalidity_does_not_splice_certificates() -> None:
    controller = FactorEnvelopeAdmissionControllerV4(_config(), _contract())
    index = 0

    for _ in range(10):
        for _ in range(4):
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
            candidate_loss=0.2,
            fallback_loss=0.0,
        )
        index += 1

    for _ in range(450):
        _resolve(
            controller,
            index=index,
            candidate_loss=0.55,
            fallback_loss=0.50,
        )
        index += 1

    snapshot = controller.snapshot()
    assert controller.authorized is False
    assert snapshot.maximum_log_e_value < snapshot.log_threshold


def test_contract_digest_binds_factor_family_and_parameter_grids() -> None:
    first = FactorEnvelopeAdmissionControllerV4(_config(), _contract())
    second = FactorEnvelopeAdmissionControllerV4(
        _config(gain_bet_fractions=(0.2, 0.5)),
        _contract(),
    )
    third = FactorEnvelopeAdmissionControllerV4(
        _config(),
        _contract(factor_family_id="different-family"),
    )

    assert len(first.decision_contract_digest) == 64
    assert first.decision_contract_digest != second.decision_contract_digest
    assert first.decision_contract_digest != third.decision_contract_digest
    boundary = first.theorem_boundary()
    assert boundary["lifetime_false_admission_bound"] == 0.05
    assert "caller-owned registered fallback object" in str(
        boundary["fallback_rule"]
    )


def test_duplicate_and_premature_trials_fail_closed() -> None:
    controller = FactorEnvelopeAdmissionControllerV4(_config(), _contract())
    controller.issue_trial(trial_id="once", issued_step=1, maturity_step=3)

    with pytest.raises(ValueError, match="already registered"):
        controller.issue_trial(trial_id="once", issued_step=2, maturity_step=4)
    with pytest.raises(ValueError, match="before maturity"):
        controller.resolve_trial(
            trial_id="once",
            resolved_step=2,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
    assert controller.snapshot().pending_trial_count == 1
