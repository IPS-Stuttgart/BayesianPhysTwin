import math

import numpy as np
import pytest

from bayesian_phystwin_experiments.anytime_valid_admission_v1 import (
    AnytimeAdmissionConfig,
    AnytimeAdmissionController,
    DeploymentState,
    MixtureBettingEProcess,
    clipped_gain,
    epoch_budget,
    margin_evidence,
)


def test_epoch_alpha_spending_is_summable() -> None:
    total = 0.05
    spent = sum(
        epoch_budget(total, epoch, allow_reentry=True)
        for epoch in range(1, 100_001)
    )
    assert spent < total
    assert total - spent < 1e-6
    assert epoch_budget(total, 1, allow_reentry=False) == total


def test_one_step_supermartingale_factor_under_rademacher_null() -> None:
    for betting_fraction in (0.01, 0.1, 0.5, 0.9):
        factors = [
            1.0 + betting_fraction * evidence for evidence in (-1.0, 1.0)
        ]
        assert np.mean(factors) == pytest.approx(1.0)
        negative_null_factors = [
            1.0 + betting_fraction * evidence
            for evidence in (-1.0, 0.0)
        ]
        assert np.mean(negative_null_factors) < 1.0


def test_mixture_e_process_stays_one_on_zero_evidence() -> None:
    process = MixtureBettingEProcess((0.01, 0.1, 0.5, 0.9))
    for _ in range(100):
        record = process.update(0.0)
        assert record.e_value == pytest.approx(1.0)
    assert process.observation_count == 100


def test_margin_transform_encodes_practical_gain_null() -> None:
    margin = 0.2
    assert margin_evidence(margin, margin) == pytest.approx(0.0)
    assert margin_evidence(-1.0, margin) == pytest.approx(-1.0)
    assert -1.0 <= margin_evidence(1.0, margin) <= 1.0


def test_losses_are_paired_clipped_and_audited() -> None:
    raw, normalized, clipped = clipped_gain(
        candidate_loss=0.0,
        fallback_loss=3.0,
        loss_cap=1.0,
    )
    assert raw == 3.0
    assert normalized == 1.0
    assert clipped is True

    with pytest.raises(ValueError, match="nonnegative"):
        clipped_gain(candidate_loss=-1.0, fallback_loss=0.0, loss_cap=1.0)


def test_controller_returns_exact_fallback_identity_until_admission() -> None:
    fallback = {"belief": "physical", "covariance": object()}
    candidate = {"belief": "corrected", "covariance": object()}
    controller = AnytimeAdmissionController(
        AnytimeAdmissionConfig(alpha=0.2, allow_reentry=False),
        candidate_id="fixed-candidate-v1",
    )

    assert controller.select(fallback=fallback, candidate=candidate) is fallback
    record = controller.observe(candidate_loss=1.0, fallback_loss=1.0)
    assert record.event == "remain-fallback"
    assert controller.select(fallback=fallback, candidate=candidate) is fallback


def test_strong_paired_gain_eventually_admits_for_next_decision() -> None:
    controller = AnytimeAdmissionController(
        AnytimeAdmissionConfig(
            alpha=0.05,
            beta=0.05,
            allow_reentry=False,
            lambdas=(0.2, 0.5, 0.9),
        ),
        candidate_id="fixed-candidate-v1",
    )

    admission = None
    for _ in range(100):
        assert controller.state is DeploymentState.FALLBACK
        record = controller.observe(candidate_loss=0.0, fallback_loss=1.0)
        if record.event == "admit":
            admission = record
            break

    assert admission is not None
    assert admission.state_before == DeploymentState.FALLBACK.value
    assert admission.state_after == DeploymentState.CANDIDATE.value
    assert admission.e_value >= admission.boundary
    assert controller.state is DeploymentState.CANDIDATE


def test_harm_monitor_revokes_after_abrupt_shift() -> None:
    controller = AnytimeAdmissionController(
        AnytimeAdmissionConfig(
            alpha=0.1,
            beta=0.1,
            allow_reentry=True,
            lambdas=(0.2, 0.5, 0.9),
        ),
        candidate_id="fixed-candidate-v1",
    )

    for _ in range(100):
        record = controller.observe(candidate_loss=0.0, fallback_loss=1.0)
        if record.event == "admit":
            break
    else:
        pytest.fail("favorable stream did not admit")

    first_epoch = controller.epoch
    for _ in range(100):
        record = controller.observe(candidate_loss=1.0, fallback_loss=0.0)
        if record.event == "revoke":
            break
    else:
        pytest.fail("harmful shifted stream did not revoke")

    assert record.epoch == first_epoch
    assert controller.state is DeploymentState.FALLBACK
    assert controller.epoch == first_epoch + 1
    assert controller.current_alpha < 0.1 / 2.0


def test_nonreentrant_controller_closes_after_revocation() -> None:
    controller = AnytimeAdmissionController(
        AnytimeAdmissionConfig(
            alpha=0.2,
            beta=0.2,
            allow_reentry=False,
            lambdas=(0.5, 0.9),
        ),
        candidate_id="fixed-candidate-v1",
    )
    for _ in range(100):
        if controller.observe(candidate_loss=0.0, fallback_loss=1.0).event == "admit":
            break
    for _ in range(100):
        if controller.observe(candidate_loss=1.0, fallback_loss=0.0).event == "revoke":
            break

    snapshot = controller.snapshot()
    assert snapshot["closed"] is True
    with pytest.raises(RuntimeError, match="closed"):
        controller.observe(candidate_loss=1.0, fallback_loss=0.0)


def test_snapshot_is_json_compatible_and_records_claim_boundary() -> None:
    controller = AnytimeAdmissionController(
        AnytimeAdmissionConfig(),
        candidate_id="fixed-candidate-v1",
    )
    controller.observe(candidate_loss=0.8, fallback_loss=1.0)
    snapshot = controller.snapshot()

    assert snapshot["contract"] == "anytime-valid-simulator-admission-state-v1"
    assert snapshot["candidate_id"] == "fixed-candidate-v1"
    assert snapshot["history"][0]["raw_gain"] == pytest.approx(0.2)
    assert "not a deployment-safety" in snapshot["guarantee_boundary"]
    assert math.isfinite(snapshot["promotion_e_process"]["e_value"])
