import math

import pytest

from bayesian_phystwin.anytime_admission_v1 import (
    AnytimeAdmissionConfig as SplitAdmissionConfig,
)
from bayesian_phystwin.anytime_admission_v1 import (
    AnytimeAdmissionController as SplitAdmissionController,
)
from bayesian_phystwin.anytime_joint_admission_v2 import (
    AdmissionContractV2,
    ChangePointBoundedGainEProcess,
    JointAdmissionConfigV2,
    JointAnytimeAdmissionControllerV2,
)


def _contract(**overrides: str) -> AdmissionContractV2:
    values = {
        "candidate_id": "candidate-sha256",
        "fallback_id": "fallback-sha256",
        "score_id": "capped-coordinate-l1-v1",
        "harm_definition_id": "candidate-loss-gt-fallback-plus-margin-v1",
        "information_set_id": "prefix-and-planned-action-v1",
        "reveal_policy_id": "paired-delayed-shadow-outcome-v1",
    }
    values.update(overrides)
    return AdmissionContractV2(**values)


def _config(**overrides: object) -> JointAdmissionConfigV2:
    values: dict[str, object] = {
        "loss_cap": 1.0,
        "minimum_mean_gain": 0.0,
        "harmful_margin": 0.0,
        "maximum_harm_rate": 0.10,
        "total_alpha": 0.05,
        "total_beta": 0.05,
        "epoch_alpha_continuation": 0.5,
        "minimum_resolved_trials": 1,
        "allow_reentry": True,
    }
    values.update(overrides)
    return JointAdmissionConfigV2(**values)


def _resolve(
    controller: JointAnytimeAdmissionControllerV2[object],
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


def test_contract_digest_binds_every_information_identity() -> None:
    baseline = _contract()

    assert len(baseline.digest) == 64
    for field, value in {
        "candidate_id": "candidate-other",
        "fallback_id": "fallback-other",
        "score_id": "score-other",
        "harm_definition_id": "harm-other",
        "information_set_id": "information-other",
        "reveal_policy_id": "reveal-other",
    }.items():
        assert _contract(**{field: value}).digest != baseline.digest

    with pytest.raises(ValueError, match="must differ"):
        _contract(candidate_id="same", fallback_id="same")
    with pytest.raises(ValueError, match="nonempty literal string"):
        _contract(score_id="")


def test_shared_iut_uses_threshold_40_instead_of_split_threshold_80() -> None:
    joint = JointAnytimeAdmissionControllerV2(_config(), _contract())
    snapshot = joint.snapshot()

    split = SplitAdmissionController(
        SplitAdmissionConfig(
            loss_cap=1.0,
            maximum_harm_rate=0.10,
            total_alpha_gain=0.025,
            total_alpha_harm=0.025,
            epoch_alpha_continuation=0.5,
            minimum_resolved_trials=1,
        )
    )
    split_snapshot = split.snapshot()

    assert snapshot.current_epoch_alpha == pytest.approx(0.025)
    assert math.exp(snapshot.shared_log_threshold) == pytest.approx(40.0)
    assert math.exp(split_snapshot.gain_log_threshold) == pytest.approx(80.0)
    assert math.exp(split_snapshot.harm_log_threshold) == pytest.approx(80.0)


def test_latched_component_crossings_authorize_asynchronously() -> None:
    controller = JointAnytimeAdmissionControllerV2(_config(), _contract())

    saw_gain_without_harm = False
    for index in range(100):
        _resolve(
            controller,
            index=index,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        snapshot = controller.snapshot()
        saw_gain_without_harm |= (
            snapshot.gain_evidence_ever_passed
            and not snapshot.harm_evidence_ever_passed
        )
        if controller.authorized:
            break

    assert saw_gain_without_harm is True
    assert controller.authorized is True
    assert controller.snapshot().gain_evidence_ever_passed is True
    assert controller.snapshot().harm_evidence_ever_passed is True


def test_shared_iut_authorizes_no_later_than_equal_budget_split_gate() -> None:
    joint = JointAnytimeAdmissionControllerV2(_config(), _contract())
    split = SplitAdmissionController(
        SplitAdmissionConfig(
            loss_cap=1.0,
            maximum_harm_rate=0.10,
            total_alpha_gain=0.025,
            total_alpha_harm=0.025,
            epoch_alpha_continuation=0.5,
            minimum_resolved_trials=1,
        )
    )
    joint_crossing = None
    split_crossing = None

    for index in range(120):
        _resolve(
            joint,
            index=index,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        split.issue_trial(
            trial_id=f"split-{index}",
            issued_step=2 * index,
            maturity_step=2 * index + 1,
        )
        split.resolve_trial(
            trial_id=f"split-{index}",
            resolved_step=2 * index + 1,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        if joint_crossing is None and joint.authorized:
            joint_crossing = index + 1
        if split_crossing is None and split.authorized:
            split_crossing = index + 1
        if joint_crossing is not None and split_crossing is not None:
            break

    assert joint_crossing is not None
    assert split_crossing is not None
    assert joint_crossing < split_crossing


def test_nonbeneficial_or_high_harm_candidate_never_authorizes() -> None:
    harmful = JointAnytimeAdmissionControllerV2(_config(), _contract())
    neutral = JointAnytimeAdmissionControllerV2(_config(), _contract())

    for index in range(200):
        _resolve(
            harmful,
            index=index,
            candidate_loss=1.0,
            fallback_loss=0.0,
        )
        _resolve(
            neutral,
            index=index,
            candidate_loss=0.5,
            fallback_loss=0.5,
        )

    assert harmful.authorized is False
    assert neutral.authorized is False


def test_change_point_reverse_gain_revokes_after_late_shift() -> None:
    controller = JointAnytimeAdmissionControllerV2(_config(), _contract())
    index = 0
    while not controller.authorized and index < 120:
        _resolve(
            controller,
            index=index,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        index += 1
    assert controller.authorized is True
    promoted_epoch = controller.epoch_index

    while controller.authorized and index < 220:
        _resolve(
            controller,
            index=index,
            candidate_loss=1.0,
            fallback_loss=0.0,
        )
        index += 1

    assert controller.authorized is False
    assert controller.epoch_index == promoted_epoch + 1
    assert any(record.event == "revoke" for record in controller.resolved_trials)


def test_old_epoch_delayed_outcome_cannot_update_new_epoch() -> None:
    controller = JointAnytimeAdmissionControllerV2(_config(), _contract())
    controller.issue_trial(trial_id="old", issued_step=0, maturity_step=2)
    controller.start_new_epoch(reason="declared-domain-shift")

    result = controller.resolve_trial(
        trial_id="old",
        resolved_step=2,
        candidate_loss=0.0,
        fallback_loss=1.0,
    )
    snapshot = controller.snapshot()

    assert result.used_for_current_epoch is False
    assert result.event == "closed-epoch-outcome"
    assert snapshot.resolved_current_epoch_admission_count == 0
    assert snapshot.ignored_closed_epoch_outcome_count == 1
    assert snapshot.authorized is False


def test_selection_returns_exact_registered_object_and_checks_ids() -> None:
    controller: JointAnytimeAdmissionControllerV2[object] = (
        JointAnytimeAdmissionControllerV2(_config(), _contract())
    )
    fallback = object()
    candidate = object()

    selected = controller.select(
        fallback=fallback,
        candidate=candidate,
        fallback_id="fallback-sha256",
        candidate_id="candidate-sha256",
    )
    assert selected is fallback

    with pytest.raises(ValueError, match="fallback_id"):
        controller.select(
            fallback=fallback,
            candidate=candidate,
            fallback_id="wrong",
            candidate_id="candidate-sha256",
        )

    for index in range(120):
        _resolve(
            controller,
            index=index,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        if controller.authorized:
            break
    selected = controller.select(
        fallback=fallback,
        candidate=candidate,
        fallback_id="fallback-sha256",
        candidate_id="candidate-sha256",
    )
    assert selected is candidate


def test_decision_digest_changes_with_configuration() -> None:
    first = JointAnytimeAdmissionControllerV2(_config(), _contract())
    second = JointAnytimeAdmissionControllerV2(
        _config(maximum_harm_rate=0.20),
        _contract(),
    )

    assert first.decision_contract_digest != second.decision_contract_digest
    assert first.snapshot().selected_artifact_id == "fallback-sha256"
    assert first.theorem_boundary()["lifetime_false_admission_bound"] == 0.05


def test_change_point_process_starts_at_one_and_detects_reverse_evidence() -> None:
    process = ChangePointBoundedGainEProcess((0.2, 0.5, 0.8))

    assert process.log_e_value == pytest.approx(0.0)
    for _ in range(80):
        process.update(1.0)
        if process.maximum_log_e_value >= math.log(40.0):
            break

    assert process.count > 0
    assert process.maximum_log_e_value >= math.log(40.0)
    snapshot = process.snapshot()
    assert snapshot["future_start_mass"] == pytest.approx(1.0 / (process.count + 1))


def test_invalid_config_and_trial_operations_fail_closed() -> None:
    with pytest.raises(ValueError, match="total_alpha"):
        _config(total_alpha=1.0)
    with pytest.raises(ValueError, match="unique finite"):
        _config(gain_bet_fractions=(0.2, 0.2))
    with pytest.raises(ValueError, match="positive"):
        _config(loss_cap=0.0)

    controller = JointAnytimeAdmissionControllerV2(_config(), _contract())
    with pytest.raises(ValueError, match="strictly after"):
        controller.issue_trial(trial_id="bad", issued_step=1, maturity_step=1)
    controller.issue_trial(trial_id="once", issued_step=0, maturity_step=1)
    with pytest.raises(ValueError, match="already registered"):
        controller.issue_trial(trial_id="once", issued_step=0, maturity_step=1)
    with pytest.raises(ValueError, match="before maturity"):
        controller.resolve_trial(
            trial_id="once",
            resolved_step=0,
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
