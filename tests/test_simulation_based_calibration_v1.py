from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.simulation_based_calibration_v1 import (
    compact_summary,
    randomized_discrete_pit,
    run_simulation_based_calibration,
    seal_protocol,
    validate_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/locks/simulation_based_calibration_v1.json"
REGISTERED_DECISION = (
    "exact-model-calibration-not-rejected-and-misspecification-detected"
)


def _registered_protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _small_protocol() -> dict[str, object]:
    protocol = _registered_protocol()
    protocol.pop("protocol_id")
    protocol["seed_start"] = 91000
    protocol["replicate_count"] = 32
    config = protocol["benchmark_config"]
    assert isinstance(config, dict)
    config.update(
        {
            "node_count": 4,
            "step_count": 30,
            "train_step_count": 20,
            "stiffness_count": 5,
            "damping_count": 4,
            "control_scale_count": 3,
        }
    )
    return seal_protocol(protocol)


def _aggregate_by_key(result: dict[str, object]) -> dict[tuple[str, str, str], dict]:
    aggregate = result["aggregate"]
    assert isinstance(aggregate, list)
    return {
        (row["action_mode"], row["condition"], row["quantity"]): row
        for row in aggregate
    }


def test_randomized_discrete_pit_accounts_for_ties() -> None:
    values = np.array([0.0, 1.0, 1.0, 2.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    assert randomized_discrete_pit(values, weights, 1.0, 0.0) == pytest.approx(0.1)
    assert randomized_discrete_pit(values, weights, 1.0, 0.5) == pytest.approx(0.35)
    assert randomized_discrete_pit(values, weights, 1.0, 1.0) == pytest.approx(0.6)

    with pytest.raises(ValueError, match="absent"):
        randomized_discrete_pit(values, weights, 1.5, 0.5)


def test_registered_protocol_is_content_addressed_and_target_free() -> None:
    protocol = validate_protocol(_registered_protocol())
    assert protocol["replicate_count"] == 512
    assert protocol["seed_start"] == 31000
    boundary = protocol["information_boundary"]
    assert isinstance(boundary, dict)
    assert not any(boundary.values())

    changed = copy.deepcopy(protocol)
    changed.pop("protocol_id")
    changed_boundary = changed["information_boundary"]
    assert isinstance(changed_boundary, dict)
    changed_boundary["target_outcomes_used"] = True
    with pytest.raises(ValueError, match="forbidden information use"):
        seal_protocol(changed)


def test_small_study_is_deterministic_and_row_bound() -> None:
    protocol = _small_protocol()
    first = run_simulation_based_calibration(protocol)
    second = run_simulation_based_calibration(protocol)
    assert first["result_id"] == second["result_id"]
    assert first["replicate_rows"] == second["replicate_rows"]
    assert len(first["replicate_rows"]) == 32 * 2 * 2

    summary = compact_summary(first)
    assert summary["replicate_row_count"] == 128
    assert len(summary["replicate_rows_sha256"]) == 64
    for row in first["replicate_rows"]:
        for quantity in row["quantities"].values():
            assert 0.0 <= quantity["randomized_pit"] <= 1.0


def test_summary_rejects_result_mutation() -> None:
    result = run_simulation_based_calibration(_small_protocol())
    changed = copy.deepcopy(result)
    changed["decision"] = "mutated"
    with pytest.raises(ValueError, match="result_id"):
        compact_summary(changed)


def test_registered_study_separates_exact_model_from_misspecification() -> None:
    result = run_simulation_based_calibration(_registered_protocol())
    assert result["decision"] == REGISTERED_DECISION
    assert result["exact_model_calibration_not_rejected"] is True
    assert result["correlated_misspecification_detected"] is True
    assert result["correlated_failed_test_fraction"] == 1.0
    assert result["familywise_test_count"] == 8
    assert len(result["replicate_rows"]) == 512 * 2 * 2

    rows = _aggregate_by_key(result)
    clean = [row for key, row in rows.items() if key[1] == "clean"]
    correlated = [row for key, row in rows.items() if key[1] == "correlated"]
    threshold = clean[0]["bonferroni_dkw_95_threshold"]
    assert threshold == pytest.approx(0.07505415359895051)
    assert max(row["pit_ks_distance"] for row in clean) < threshold
    assert min(row["pit_ks_distance"] for row in correlated) > 0.5

    dynamic_control = rows[("dynamic", "clean", "control_scale")]
    quasi_control = rows[("quasi_static", "clean", "control_scale")]
    assert dynamic_control["mean_absolute_error"] < 1.0e-4
    assert quasi_control["mean_posterior_std"] > 50.0 * dynamic_control[
        "mean_posterior_std"
    ]

    dynamic_query = rows[
        ("dynamic", "correlated", "terminal_last_node_displacement")
    ]
    assert dynamic_query["coverage"]["0.9"]["rate"] < 0.05
    assert dynamic_query["pit_ks_distance"] > 0.8
