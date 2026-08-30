"""Synthetic integration contracts for the active-probe cloth runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.tracking_cloth_deformation_v1.active_probe_run import (
    active_mask,
    arm_specs,
    array_digest,
    belief_digest,
    build_belief_arms,
    calibrated_residuals,
    loss_vector,
    posterior_temperature,
    validate_protocol,
    weighted_belief,
)
from experiments.tracking_cloth_deformation_v1.data import Inputs
from experiments.tracking_cloth_deformation_v1.model import Predictions

BASE = (
    Path(__file__).resolve().parents[1]
    / "experiments/tracking_cloth_deformation_v1"
)


def protocol() -> dict:
    value = json.loads((BASE / "active_probe_protocol.json").read_text())
    validate_protocol(value)
    return value


def prediction() -> Predictions:
    time_count, markers, models = 8, 4, 9
    times = np.arange(time_count, dtype=np.float64)
    cutoff = 2
    prefix = np.zeros((cutoff + 1, markers, 3), dtype=np.float64)
    boundary = np.zeros((time_count, 2, 3), dtype=np.float64)
    inputs = Inputs(
        times,
        prefix,
        boundary,
        np.arange(markers),
        np.array([0, markers - 1]),
        cutoff,
        0.0,
        1.0,
    )
    bank = np.zeros((models, time_count, markers, 3), dtype=np.float64)
    for model in range(models):
        bank[model, :, 1:-1, 0] = (
            model * np.arange(time_count)[:, None] * 0.001
        )
    return Predictions(inputs, bank[4].copy(), bank)


def test_protocol_registers_decisive_budget_and_information_boundary() -> None:
    value = protocol()
    assert value["fold_rule"] == "leave-one-material-out"
    assert value["probe_budgets"] == [0, 1, 2, 4]
    assert value["primary_budget"] == 1
    assert value["held_material_candidate_inputs_used_for_selection"] is False
    assert value["held_material_twist_inputs_used_for_selection"] is False
    assert value["paper_claim_authorized"] is False


def test_weighted_belief_and_content_digests() -> None:
    candidate = prediction()
    weights = np.ones(9) / 9
    mean, variance = weighted_belief(candidate, weights, [1e-4] * 3)
    assert mean.shape == variance.shape == candidate.nominal.shape
    assert np.all(variance > 0.0)
    assert len(array_digest(mean)) == 64
    assert len(belief_digest(mean, variance)) == 64


def test_complete_arm_roster_and_common_endpoint_parity() -> None:
    value = protocol()
    candidate = prediction()
    weights = (np.ones(9) / 9).tolist()
    states = {
        policy: {
            str(budget): {
                "weights": weights,
                "selected_actions": value["probe_conditions"][:budget],
                "steps": [],
            }
            for budget in value["probe_budgets"]
        }
        for policy in value["probe_policies"]
    }
    specimen = {
        "policy_states": states,
        "single_probe_weights": {
            condition: weights for condition in value["probe_conditions"]
        },
    }
    fold = {
        "prior_weights": weights,
        "source_residual_variance_m2": {
            "bayesian": [1e-4] * 3,
            "nominal_physics": [2e-4] * 3,
            "last_residual": [3e-4] * 3,
        },
    }
    arms = build_belief_arms(
        candidate,
        prefix_last=candidate.inputs.prefix[-1],
        boundary=candidate.inputs.boundary,
        fold=fold,
        specimen=specimen,
        protocol=value,
    )
    assert set(arms) == set(arm_specs(value))
    assert len(arms) == 18
    for budget in (0, 4):
        reference = arms[f"fixed_order_k{budget}"]
        for policy in ("parameter_information", "task_directed"):
            np.testing.assert_array_equal(
                reference[0], arms[f"{policy}_k{budget}"][0]
            )
            np.testing.assert_array_equal(
                reference[1], arms[f"{policy}_k{budget}"][1]
            )


def test_loss_temperature_residual_calibration_and_mask() -> None:
    value = protocol()
    candidate = prediction()
    truth = candidate.bank[3].copy()
    losses = loss_vector(candidate, truth)
    assert losses[3] == 0.0
    assert posterior_temperature(np.stack([losses, losses]), 0.001) > 0.0
    residual = calibrated_residuals(
        [(candidate, truth)], np.ones(9) / 9, value
    )
    assert set(residual) == {
        "bayesian",
        "nominal_physics",
        "last_residual",
    }
    assert all(number > 0.0 for values in residual.values() for number in values)
    mask = active_mask(candidate.inputs)
    assert mask.shape == (8, 4)
    assert int(mask.sum()) == 10
