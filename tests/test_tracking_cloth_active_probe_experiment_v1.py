from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.tracking_cloth_deformation_v1.active_probe_experiment import (
    _assert_endpoint_parity,
    aggregate,
)
from experiments.tracking_cloth_deformation_v1.active_probe_run import arm_specs


def protocol() -> dict:
    return {
        "materials": ["cotton", "denim", "polyester", "wool"],
        "sizes": ["A2", "A3"],
        "probe_conditions": [
            "fast_hands",
            "fast_hanger",
            "slow_hands",
            "slow_hanger",
        ],
        "fixed_probe_order": [
            "fast_hands",
            "fast_hanger",
            "slow_hands",
            "slow_hanger",
        ],
        "probe_policies": [
            "fixed_order",
            "parameter_information",
            "task_directed",
        ],
        "probe_budgets": [0, 1, 2, 4],
        "primary_budget": 1,
        "bootstrap_repetitions": 100,
        "bootstrap_seed": 4,
        "held_material_candidate_inputs_used_for_selection": False,
        "held_material_twist_inputs_used_for_selection": False,
        "paper_claim_authorized": False,
    }


def source_fit() -> dict:
    specimens = {}
    actions = protocol()["probe_conditions"]
    for material in protocol()["materials"]:
        for size_index, size in enumerate(protocol()["sizes"]):
            specimen = f"{material}_{size}"
            states = {}
            for policy_index, policy in enumerate(protocol()["probe_policies"]):
                first = actions[(size_index + policy_index) % len(actions)]
                states[policy] = {
                    "1": {"selected_actions": [first]},
                }
            specimens[specimen] = {"policy_states": states}
    return {"specimens": specimens}


def arm_rmse(arm: str) -> float:
    values = {
        "nominal_physics": 10.0,
        "last_residual": 9.5,
        "single_probe_fast_hands": 9.2,
        "single_probe_fast_hanger": 9.1,
        "single_probe_slow_hands": 9.0,
        "single_probe_slow_hanger": 8.9,
        "fixed_order_k0": 10.0,
        "fixed_order_k1": 9.0,
        "fixed_order_k2": 8.0,
        "fixed_order_k4": 7.0,
        "parameter_information_k0": 10.0,
        "parameter_information_k1": 8.5,
        "parameter_information_k2": 7.5,
        "parameter_information_k4": 7.0,
        "task_directed_k0": 10.0,
        "task_directed_k1": 7.5,
        "task_directed_k2": 7.0,
        "task_directed_k4": 7.0,
    }
    return values[arm]


def rows() -> list[dict]:
    result = []
    p = protocol()
    for material in p["materials"]:
        for size in p["sizes"]:
            specimen = f"{material}_{size}"
            for condition in p["probe_conditions"]:
                recording = f"{specimen}_{condition}.csv"
                for arm in arm_specs(p):
                    value = arm_rmse(arm)
                    result.append(
                        {
                            "recording": recording,
                            "specimen": specimen,
                            "material": material,
                            "arm": arm,
                            "rmse_mm": value,
                            "mean_marker_error_mm": value * 0.8,
                            "coordinate_nll": value / 10.0,
                            "coordinate_90_coverage": 0.9,
                            "mean_full_90_width_mm": value * 2.0,
                        }
                    )
    return result


def test_aggregate_reports_registered_primary_contrast() -> None:
    table, metrics = aggregate(rows(), protocol(), source_fit())
    assert len(table) == 8 * 18
    primary = metrics["primary_contrasts"]["parameter_information_k1"]
    assert primary["candidate_minus_comparator_rmse_mm"] == pytest.approx(-1.0)
    assert primary["specimen_wins"] == 8
    assert primary["specimen_losses"] == 0
    assert metrics["endpoint_parity"] == {"K0": True, "K4": True}
    assert metrics["task_vs_parameter_first_probe_disagreements"] == 8


def test_aggregate_refuses_partial_or_endpoint_inconsistent_results() -> None:
    partial = rows()[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(partial, protocol(), source_fit())
    mismatched_source = source_fit()
    mismatched_source["specimens"].pop("cotton_A2")
    with pytest.raises(ValueError, match="specimen rosters"):
        aggregate(rows(), protocol(), mismatched_source)
    inconsistent = copy.deepcopy(rows())
    entry = next(row for row in inconsistent if row["arm"] == "task_directed_k4")
    entry["rmse_mm"] += 1.0
    with pytest.raises(ValueError, match="endpoint parity"):
        aggregate(inconsistent, protocol(), source_fit())


def test_complete_belief_endpoint_parity_is_exact() -> None:
    p = protocol()
    mean_zero = np.zeros((2, 3, 3))
    variance_zero = np.ones((2, 3, 3))
    mean_full = np.full((2, 3, 3), 2.0)
    variance_full = np.full((2, 3, 3), 3.0)
    beliefs = {}
    for policy in p["probe_policies"]:
        beliefs[f"{policy}_k0"] = (mean_zero.copy(), variance_zero.copy())
        beliefs[f"{policy}_k4"] = (mean_full.copy(), variance_full.copy())
    _assert_endpoint_parity(beliefs, p)
    beliefs["task_directed_k4"][0][0, 0, 0] = 3.0
    with pytest.raises(ValueError, match="K=4"):
        _assert_endpoint_parity(beliefs, p)
