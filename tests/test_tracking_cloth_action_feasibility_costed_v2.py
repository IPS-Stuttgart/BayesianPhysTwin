from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    act_sense_fallback_certificate,
)
from experiments.tracking_cloth_action_feasibility_costed_v2._decision import (
    decision_grid_v2,
    support_robust_decision,
)
from experiments.tracking_cloth_action_feasibility_costed_v2.run import (
    read_v2_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments"
    / "tracking_cloth_action_feasibility_costed_v2"
    / "protocol.json"
)


def _parent_certificate():
    prior = np.asarray([0.5, 0.5])
    quotient = np.asarray([1.0])
    classes = np.asarray([0, 0])
    losses = np.asarray([[0.0, 2.0], [2.0, 0.0]])
    outcomes = np.asarray([[0, 1]])
    return act_sense_fallback_certificate(
        prior,
        quotient,
        classes,
        losses,
        outcomes,
        [0.25],
        fallback_action_index=0,
        regret_tolerance=0.5,
        probe_names=["probe"],
    )


def test_zero_support_miss_matches_represented_plan_certificate() -> None:
    parent = _parent_certificate()
    robust = support_robust_decision(
        parent,
        probe_costs=np.asarray([0.25]),
        support_miss_probability=0.0,
        unknown_action_loss_lower=np.asarray([0.0, 0.0]),
        unknown_action_loss_upper=np.asarray([3.0, 3.0]),
        regret_tolerance=0.5,
    )

    np.testing.assert_allclose(
        robust.pairwise_worst_case_loss_gap,
        parent.plan_certificate.pairwise_worst_case_loss_gap,
    )
    np.testing.assert_allclose(
        robust.worst_case_regret,
        parent.plan_certificate.worst_case_regret,
    )


def test_support_miss_formula_matches_declared_plan_loss_box() -> None:
    parent = _parent_certificate()
    epsilon = 0.2
    robust = support_robust_decision(
        parent,
        probe_costs=np.asarray([0.25]),
        support_miss_probability=epsilon,
        unknown_action_loss_lower=np.asarray([0.0, 0.5]),
        unknown_action_loss_upper=np.asarray([2.0, 3.0]),
        regret_tolerance=1.0,
    )

    represented = parent.plan_certificate.pairwise_worst_case_loss_gap
    unknown_gap = (
        robust.unknown_plan_loss_upper[:, None]
        - robust.unknown_plan_loss_lower[None, :]
    )
    expected = represented + epsilon * np.maximum(0.0, unknown_gap - represented)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(robust.pairwise_worst_case_loss_gap, expected)


def test_same_plan_width_does_not_reduce_support_miss_budget() -> None:
    parent = act_sense_fallback_certificate(
        [1.0],
        [1.0],
        [0],
        [[0.0, 1.0]],
        np.empty((0, 1), dtype=np.int64),
        [],
        fallback_action_index=1,
        regret_tolerance=0.5,
    )
    robust = support_robust_decision(
        parent,
        probe_costs=np.empty(0),
        support_miss_probability=0.0,
        unknown_action_loss_lower=np.asarray([0.0, 0.0]),
        unknown_action_loss_upper=np.asarray([10.0, 0.0]),
        regret_tolerance=0.5,
    )

    # The budget of plan 1 is constrained only by comparison to the other plan;
    # an interval width for a plan compared with itself must remain irrelevant.
    assert robust.maximum_admissible_support_miss[1] == pytest.approx(1.0)


def _source_protocol(probe_cost: float) -> dict:
    return {
        "materials": ["m0", "m1"],
        "interactions": ["a0", "a1", "hold"],
        "probe_cost_grid": [probe_cost],
        "regret_tolerance_grid": [0.5],
        "max_plan_count": 1000,
    }


def _v2_protocol() -> dict:
    return {
        "support_robustness": {
            "support_miss_probability_grid": [0.0],
            "primary_support_miss_probability": 0.0,
            "unknown_terminal_loss_lower_normalized": [0.0, 0.0, 0.0],
            "unknown_terminal_loss_upper_normalized": [2.0, 2.0, 2.0],
        },
        "source_gate": {"minimum_relative_objective_gain": -1.0},
    }


def test_sensing_source_objective_explicitly_charges_probe_cost() -> None:
    blocks = [("m0", 1), ("m0", 2), ("m1", 1), ("m1", 2)]
    losses = np.asarray(
        [
            [0.0, 4.0, 1.5],
            [4.0, 0.0, 1.5],
            [0.0, 4.0, 1.5],
            [4.0, 0.0, 1.5],
        ]
    )
    probes = np.asarray(
        [
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    records, summary = decision_grid_v2(
        blocks,
        losses,
        probes,
        _source_protocol(0.2),
        _v2_protocol(),
    )
    record = records[0]

    assert record["mode_counts"]["sense"] == 2
    assert record["mean_source_terminal_loss"] == pytest.approx(0.0)
    assert record["mean_source_objective_loss"] == pytest.approx(
        0.2 * summary["loss_scale"]
    )
    assert record["relative_objective_gain_vs_fallback"] < (
        record["relative_terminal_gain_vs_fallback"]
    )


def test_primary_setting_serializes_complete_probe_policy() -> None:
    blocks = [("m0", 1), ("m0", 2), ("m1", 1), ("m1", 2)]
    losses = np.asarray(
        [
            [0.0, 4.0, 1.5],
            [4.0, 0.0, 1.5],
            [0.0, 4.0, 1.5],
            [4.0, 0.0, 1.5],
        ]
    )
    probes = np.asarray(
        [[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64
    )
    _, summary = decision_grid_v2(
        blocks,
        losses,
        probes,
        _source_protocol(0.2),
        _v2_protocol(),
    )
    selected = summary["selected_primary_source_setting"]

    assert selected["mode_counts"] == {"act": 0, "sense": 2, "fallback": 0}
    for output in selected["outputs"]:
        assert output["selected_probe"] == "a0"
        assert output["terminal_action_by_probe_outcome"] == ["a0", "a1"]
        assert output["output_plan_support_miss_budget"] >= 0.0


def test_v2_protocol_keeps_rep3_closed_and_marks_bounds_assumed() -> None:
    protocol = read_v2_protocol(PROTOCOL)
    boundary = protocol["information_boundary"]
    robustness = protocol["support_robustness"]

    assert protocol["source_repetitions"] == [1, 2]
    assert protocol["reserved_target_repetition"] == 3
    assert boundary["rep3_numeric_outcomes_read"] is False
    assert boundary["rep3_protocol_authorized"] is False
    assert robustness["bound_is_assumed_not_estimated"] is True
    assert robustness["target_tuning"] is False
