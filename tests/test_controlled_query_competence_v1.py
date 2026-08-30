from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin_experiments.controlled_query_competence_v1 import (
    ACTION_NAMES,
    CONFIRMATION_COUNT_PER_TOPOLOGY,
    CONFIRMATION_SEED_BASE,
    HARM_MARGIN,
    HORIZONS,
    QUERY_NAMES,
    SOURCE_BUNDLE_SHA256,
    TOPOLOGY_NAMES,
    FrozenLogisticRiskModelV1,
    QueryOutcomeV1,
    build_actions_v1,
    build_hypotheses_v1,
    build_objects_v1,
    evaluate_selective_policy_v1,
    experiment_protocol_v1,
    feature_names_v1,
    fit_logistic_risk_model_v1,
    fit_screen_calibration_v1,
    generate_partition_v1,
    preoutcome_route_v1,
    query_value_v1,
    risk_model_feature_sets_v1,
    select_threshold_v1,
    simulate_trajectory_v1,
)


def _synthetic_outcomes(count: int = 120) -> tuple[QueryOutcomeV1, ...]:
    names = feature_names_v1()
    entropy_index = names.index("normalized_posterior_entropy")
    uncertainty_index = names.index("one_minus_maximum_posterior")
    outcomes = []
    for index in range(count):
        harmful = index < 10
        feature = np.zeros(len(names))
        feature[entropy_index] = 0.9 if harmful else 0.1 + index / 10_000
        feature[uncertainty_index] = 0.8 if harmful else 0.05 + index / 20_000
        candidate_loss = 0.2 if harmful else 0.05
        fallback_loss = 0.05 if harmful else 0.15
        outcomes.append(
            QueryOutcomeV1(
                group_id=f"synthetic:{index:04d}",
                partition="synthetic",
                topology=TOPOLOGY_NAMES[index % len(TOPOLOGY_NAMES)],
                action=ACTION_NAMES[index % len(ACTION_NAMES)],
                horizon_step_count=HORIZONS[index % len(HORIZONS)],
                query_name=QUERY_NAMES[index % len(QUERY_NAMES)],
                true_hypothesis_index=index % 4,
                candidate_model_index=1,
                feature_vector=feature,
                model_losses=np.asarray((fallback_loss, candidate_loss, 0.3, 0.4)),
                candidate_loss=candidate_loss,
                fallback_loss=fallback_loss,
                harmful_candidate=harmful,
            )
        )
    return tuple(outcomes)


def test_protocol_freezes_disjoint_source_and_confirmation_partitions() -> None:
    protocol = experiment_protocol_v1()
    partitions = protocol["partitions"]
    seed_bases = [value["seed_base"] for value in partitions.values()]

    assert len(seed_bases) == len(set(seed_bases))
    assert partitions["confirmation"] == {
        "seed_base": CONFIRMATION_SEED_BASE,
        "count_per_topology": CONFIRMATION_COUNT_PER_TOPOLOGY,
        "opened_by_source_stage": False,
    }
    assert protocol["source_bundle_sha256"] == SOURCE_BUNDLE_SHA256
    assert protocol["source_gate_required_before_confirmation"] is True
    assert protocol["confirmation_attempt_limit"] == 1
    assert protocol["no_reselection_or_retry"] is True
    assert protocol["unseen_topology_transfer_claimed"] is False
    assert protocol["prob4d_used"] is False
    assert protocol["protected_artifacts_used"] is False
    assert len(protocol["protocol_id"]) == 64


def test_spring_graph_and_query_functionals_are_finite() -> None:
    graph_object = build_objects_v1()[1]
    action = build_actions_v1(graph_object)[4]
    hypothesis = build_hypotheses_v1()[2]
    trajectory = simulate_trajectory_v1(
        graph_object,
        action,
        graph_object.nominal_parameters,
        hypothesis,
        nonlinearity=0.18,
    )

    assert trajectory.shape == (56, 9, 2)
    assert np.all(np.isfinite(trajectory))
    for query_name in QUERY_NAMES:
        value = query_value_v1(graph_object, trajectory, HORIZONS[-1], query_name)
        assert value.ndim == 1
        assert np.all(np.isfinite(value))

    with pytest.raises(ValueError, match="horizon"):
        query_value_v1(graph_object, trajectory, 13, QUERY_NAMES[0])
    with pytest.raises(ValueError, match="functional"):
        query_value_v1(graph_object, trajectory, HORIZONS[0], "unknown")


def test_preoutcome_route_has_no_future_outcome_argument() -> None:
    posterior = np.asarray((0.1, 0.6, 0.2, 0.1))
    outputs = tuple(np.asarray((float(index), 0.5 * index)) for index in range(4))

    candidate, feature = preoutcome_route_v1(
        topology_index=0,
        action_index=1,
        horizon=HORIZONS[1],
        query_index=2,
        posterior=posterior,
        query_outputs=outputs,
    )
    repeated_candidate, repeated_feature = preoutcome_route_v1(
        topology_index=0,
        action_index=1,
        horizon=HORIZONS[1],
        query_index=2,
        posterior=posterior,
        query_outputs=outputs,
    )

    assert candidate == repeated_candidate
    assert np.array_equal(feature, repeated_feature)
    assert feature.shape == (len(feature_names_v1()),)
    with pytest.raises(ValueError, match="posterior shape"):
        preoutcome_route_v1(
            topology_index=0,
            action_index=1,
            horizon=HORIZONS[1],
            query_index=2,
            posterior=np.asarray((0.5, 0.5)),
            query_outputs=outputs,
        )


def test_partition_generation_is_deterministic_and_group_disjoint() -> None:
    calibration = fit_screen_calibration_v1(count_per_topology=4, seed_base=17)
    first = generate_partition_v1(
        partition="tiny-source",
        count_per_topology=3,
        seed_base=101,
        calibration=calibration,
    )
    repeated = generate_partition_v1(
        partition="tiny-source",
        count_per_topology=3,
        seed_base=101,
        calibration=calibration,
    )
    different = generate_partition_v1(
        partition="tiny-gate",
        count_per_topology=3,
        seed_base=202,
        calibration=calibration,
    )

    assert len(first) == 9
    assert [item.group_id for item in first] == [item.group_id for item in repeated]
    assert np.array_equal(first[0].feature_vector, repeated[0].feature_vector)
    assert first[0].candidate_loss == repeated[0].candidate_loss
    assert set(item.group_id for item in first).isdisjoint(
        item.group_id for item in different
    )
    with pytest.raises(ValueError, match="canonical"):
        generate_partition_v1(
            partition=" bad",
            count_per_topology=1,
            seed_base=1,
            calibration=calibration,
        )


def test_logistic_risk_model_roundtrip_and_tamper_rejection() -> None:
    outcomes = _synthetic_outcomes()
    selected = risk_model_feature_sets_v1()["uncertainty_only"]
    model = fit_logistic_risk_model_v1(
        outcomes,
        model_name="uncertainty_only",
        selected_feature_names=selected,
    )
    repeated = fit_logistic_risk_model_v1(
        outcomes,
        model_name="uncertainty_only",
        selected_feature_names=selected,
    )
    loaded = FrozenLogisticRiskModelV1.from_record(model.to_record())

    assert model.artifact_id == repeated.artifact_id == loaded.artifact_id
    assert model.converged is True
    assert model.score(outcomes[0].feature_vector) > model.score(
        outcomes[-1].feature_vector
    )

    tampered = model.to_record()
    tampered["coefficients"] = list(tampered["coefficients"])
    tampered["coefficients"][0] += 1.0
    with pytest.raises(ValueError, match="identity"):
        FrozenLogisticRiskModelV1.from_record(tampered)

    with pytest.raises(ValueError, match="too few"):
        fit_logistic_risk_model_v1(
            outcomes[:20],
            model_name="uncertainty_only",
            selected_feature_names=selected,
        )


def test_threshold_selection_and_finite_group_gate() -> None:
    outcomes = _synthetic_outcomes()
    scores = np.asarray(
        [0.9 if outcome.harmful_candidate else 0.01 for outcome in outcomes]
    )
    selection = select_threshold_v1(outcomes, scores)

    assert selection["selection_passed"] is True
    assert selection["selected_threshold"] == 0.5
    evaluation = evaluate_selective_policy_v1(
        outcomes,
        scores,
        float(selection["selected_threshold"]),
        bootstrap_seed=123,
    )
    assert evaluation["accepted_count"] == 110
    assert evaluation["harmful_accepted_count"] == 0
    assert evaluation["exact_one_sided_95_harm_upper_bound"] < 0.03
    assert evaluation["selected_regret"]["ci95_upper"] < 0.0
    assert evaluation["gate_passed"] is True
    assert all(value >= 20 for value in evaluation["accepted_per_query"].values())

    with pytest.raises(ValueError, match="risk score"):
        evaluate_selective_policy_v1(
            outcomes,
            np.zeros(len(outcomes) - 1),
            0.5,
            bootstrap_seed=123,
        )


def test_harm_margin_is_practical_and_not_zero() -> None:
    outcomes = _synthetic_outcomes()
    safe = outcomes[-1]
    close = replace(
        safe,
        candidate_loss=safe.fallback_loss + 0.5 * HARM_MARGIN,
        harmful_candidate=False,
    )
    harmful = replace(
        safe,
        candidate_loss=safe.fallback_loss + 1.5 * HARM_MARGIN,
        harmful_candidate=True,
    )

    assert close.regret < HARM_MARGIN
    assert harmful.regret > HARM_MARGIN
