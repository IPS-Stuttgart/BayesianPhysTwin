from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.robust_interventional_cause_attribution_v1 import (
    ROBUST_ATTRIBUTION_CLAIM_BOUNDARY,
    RobustAttributionPlanV1,
    RobustAttributionStatus,
    RobustCauseModelV1,
    RobustObservationDesignV1,
)


def _cause(
    name: str,
    values: list[float],
    *,
    signature_error: float = 0.02,
    tolerance: float = 0.3,
) -> RobustCauseModelV1:
    interventions = tuple(f"u{index}" for index in range(len(values)))
    return RobustCauseModelV1(
        cause_id=name,
        intervention_ids=interventions,
        response_blocks=tuple(np.asarray([[value]]) for value in values),
        query_map=np.eye(1),
        signature_error_bounds=(signature_error,) * len(values),
        coefficient_norm_bound=2.0,
        query_error_tolerance=tolerance,
        minimum_effect_norm=0.1,
    )


def _design(
    count: int,
    *,
    noise: float = 0.01,
    costs: tuple[float, ...] | None = None,
) -> RobustObservationDesignV1:
    interventions = tuple(f"u{index}" for index in range(count))
    return RobustObservationDesignV1(
        intervention_ids=interventions,
        nuisance_blocks=tuple(np.empty((1, 0)) for _ in interventions),
        observation_noise_radii=(noise,) * count,
        nuisance_signature_error_bounds=(0.0,) * count,
        nuisance_coefficient_norm_bound=0.0,
        intervention_costs=costs or (1.0,) * count,
    )


def _plan(
    left: RobustCauseModelV1,
    right: RobustCauseModelV1,
    design: RobustObservationDesignV1,
) -> RobustAttributionPlanV1:
    causes = tuple(sorted((left, right), key=lambda item: item.cause_id))
    return RobustAttributionPlanV1(
        observation_design=design,
        cause_models=causes,
        cause_family_id="test-family",
    )


def test_changed_intervention_resolves_source_action_confounding() -> None:
    plan = _plan(_cause("a", [1.0, 1.0]), _cause("b", [1.0, -1.0]), _design(2))

    assert plan.result_for("a").minimum_robust_intervention_count == 2
    decision = plan.evaluate((np.asarray([0.5]), np.asarray([1.9])))

    assert (
        decision.result_for("a").status
        is RobustAttributionStatus.ROBUSTLY_ATTRIBUTABLE
    )


def test_query_bound_covers_signature_and_observation_error() -> None:
    plan = _plan(_cause("a", [1.0, 1.0]), _cause("b", [1.0, -1.0]), _design(2))
    truth_a = 1.2
    truth_b = -0.7
    observed = (
        np.asarray([1.01 * truth_a + truth_b + 0.005]),
        np.asarray([0.99 * truth_a - 0.99 * truth_b - 0.004]),
    )

    result = plan.evaluate(observed).result_for("a")

    assert abs(float(result.query_estimate[0]) - truth_a) <= (
        result.query_error_bound + 1e-12
    )


def test_nominal_identifiability_does_not_override_uncertainty() -> None:
    plan = _plan(
        _cause("a", [1.0, 1.0], signature_error=0.4, tolerance=0.05),
        _cause("b", [1.0, -1.0], signature_error=0.4, tolerance=0.05),
        _design(2),
    )

    result = plan.evaluate((np.asarray([0.5]), np.asarray([1.9])))

    assert (
        result.result_for("a").status
        is RobustAttributionStatus.IDENTIFIABLE_BUT_UNSTABLE
    )


def test_open_set_closure_rejects_response_outside_family_span() -> None:
    interventions = ("u0", "u1")
    left = RobustCauseModelV1(
        "a",
        interventions,
        (np.asarray([[1.0], [0.0]]), np.asarray([[0.0], [1.0]])),
        np.eye(1),
        (0.001, 0.001),
        2.0,
        0.1,
    )
    right = RobustCauseModelV1(
        "b",
        interventions,
        (np.asarray([[1.0], [0.0]]), np.asarray([[1.0], [0.0]])),
        np.eye(1),
        (0.001, 0.001),
        2.0,
        0.1,
    )
    design = RobustObservationDesignV1(
        interventions,
        (np.empty((2, 0)), np.empty((2, 0))),
        (0.001, 0.001),
        (0.0, 0.0),
        0.0,
        (1.0, 1.0),
    )

    decision = _plan(left, right, design).evaluate(
        (np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]))
    )

    assert decision.registered_family_falsified
    assert all(
        result.status is RobustAttributionStatus.UNREGISTERED_CAUSE
        for result in decision.cause_decisions
    )


def test_cost_aware_intervention_plan_is_outcome_independent() -> None:
    plan = _plan(
        _cause("a", [1.0, 1.0, 1.0], signature_error=0.005, tolerance=0.1),
        _cause("b", [1.0, -1.0, 2.0], signature_error=0.005, tolerance=0.1),
        _design(3, costs=(1.0, 10.0, 2.0)),
    )
    identity = plan.plan_id
    first = plan.evaluate(
        (np.asarray([0.2]), np.asarray([0.1]), np.asarray([1000.0]))
    )
    second = plan.evaluate(
        (np.asarray([-50.0]), np.asarray([80.0]), np.asarray([-1000.0]))
    )

    result = plan.result_for("a")
    assert first.plan_id == second.plan_id == identity
    assert result.minimum_robust_intervention_count == 2
    assert result.minimum_cost_robust_intervention_sets == (("u0", "u2"),)


def test_immutable_arrays_and_bounded_claim() -> None:
    source = np.asarray([[1.0]])
    left = RobustCauseModelV1(
        "a",
        ("u0", "u1"),
        (source, np.asarray([[1.0]])),
        np.eye(1),
        (0.02, 0.02),
        2.0,
        0.3,
    )
    plan = _plan(left, _cause("b", [1.0, -1.0]), _design(2))
    identity = plan.plan_id
    source[0, 0] = 99.0

    assert left.response_blocks[0][0, 0] == 1.0
    with pytest.raises(ValueError):
        left.response_blocks[0].setflags(write=True)
    assert plan.plan_id == identity
    assert "does not prove family completeness" in ROBUST_ATTRIBUTION_CLAIM_BOUNDARY
