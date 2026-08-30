"""Synthetic contracts for the Tracking Cloth active-probe CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.tracking_cloth_deformation_v1.active_probe_cli import (
    active_arms,
    aggregate,
    persistence_belief,
)
from experiments.tracking_cloth_deformation_v1.data import Inputs
from experiments.tracking_cloth_deformation_v1.model import Predictions

BASE = Path(__file__).resolve().parents[1] / "experiments/tracking_cloth_deformation_v1"


def protocol() -> dict:
    return json.loads((BASE / "active_probe_protocol.json").read_text())


def prediction() -> Predictions:
    times = np.arange(8, dtype=np.float64)
    prefix = np.zeros((3, 4, 3), dtype=np.float64)
    prefix[-1, :, 0] = np.arange(4)
    boundary = np.zeros((8, 2, 3), dtype=np.float64)
    boundary[:, 0, 0] = np.arange(8)
    boundary[:, 1, 0] = 10 + np.arange(8)
    inputs = Inputs(
        times,
        prefix,
        boundary,
        np.arange(4),
        np.array([0, 3]),
        2,
        0.0,
        1.0,
    )
    bank = np.zeros((9, 8, 4, 3), dtype=np.float64)
    return Predictions(inputs, bank[0].copy(), bank)


def test_active_roster_adds_the_strong_persistence_control() -> None:
    arms = active_arms(protocol())
    assert arms[0] == "persistence"
    assert len(arms) == 19
    assert len(set(arms)) == len(arms)


def test_persistence_belief_preserves_boundary_and_positive_variance() -> None:
    candidate = prediction()
    mean, variance = persistence_belief(candidate, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        mean[:, candidate.inputs.corners], candidate.inputs.boundary
    )
    np.testing.assert_array_equal(mean[3:, 1], candidate.inputs.prefix[-1, 1])
    assert mean.shape == variance.shape == candidate.nominal.shape
    assert np.all(variance > 0.0)


def test_aggregate_reports_policy_disagreement_without_pseudoreplication() -> None:
    value = protocol()
    arms = active_arms(value)
    specimens = [
        f"{material}_{size}"
        for material in value["materials"]
        for size in value["sizes"]
    ]
    arm_rmse = {arm: 20.0 for arm in arms}
    arm_rmse.update(
        {
            "task_directed_k1": 10.0,
            "parameter_information_k1": 12.0,
            "fixed_order_k1": 13.0,
            "fixed_order_k0": 14.0,
            "persistence": 15.0,
        }
    )
    for index, condition in enumerate(value["probe_conditions"]):
        arm_rmse[f"single_probe_{condition}"] = 11.0 + index

    rows = []
    source_specimens = {}
    for specimen_index, specimen in enumerate(specimens):
        material, size = specimen.split("_", maxsplit=1)
        task_choice = value["probe_conditions"][specimen_index % 2]
        parameter_choice = value["probe_conditions"][2 + specimen_index % 2]
        source_specimens[specimen] = {
            "policy_states": {
                "task_directed": {
                    "1": {"selected_actions": [task_choice]},
                },
                "parameter_information": {
                    "1": {"selected_actions": [parameter_choice]},
                },
                "fixed_order": {
                    "1": {
                        "selected_actions": [value["fixed_probe_order"][0]],
                    },
                },
            }
        }
        for recording in range(4):
            for arm in arms:
                rows.append(
                    {
                        "recording": f"{specimen}-{recording}",
                        "specimen": specimen,
                        "material": material,
                        "size": size,
                        "speed": "fast",
                        "grasp": "hands",
                        "arm": arm,
                        "rmse_mm": arm_rmse[arm],
                        "mean_marker_error_mm": arm_rmse[arm],
                        "coordinate_nll": arm_rmse[arm] / 100.0,
                        "coordinate_90_coverage": 0.9,
                        "mean_full_90_width_mm": 30.0,
                    }
                )

    table, metrics = aggregate(
        rows,
        {"specimens": source_specimens},
        value,
        arms,
    )
    assert len(table) == 8 * len(arms)
    assert metrics["selection"]["task_vs_parameter_disagreement_count"] == 8
    assert metrics["mechanism_gate"][
        "task_directed_beats_parameter_information_overall"
    ]
    assert metrics["mechanism_gate"]["task_directed_beats_persistence"]
