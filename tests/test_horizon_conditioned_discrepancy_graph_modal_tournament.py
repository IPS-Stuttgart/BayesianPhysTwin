from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.discrepancy_candidate_tournament import (
    DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
    CandidateSpec,
    parse_discrepancy_candidate_tournament,
)
from bayesian_phystwin.graph_dynamic_discrepancy import (
    GraphDynamicDiscrepancyBeliefV1,
)
from bayesian_phystwin.graph_dynamic_discrepancy_tournament import (
    GraphDynamicTournamentScoringPolicyV1,
    build_graph_dynamic_tournament_prediction_bundle,
    candidate_spec_dict,
    score_graph_dynamic_tournament_prediction_bundle,
    seal_graph_dynamic_tournament_prediction,
    tournament_record_dict,
)

REVISION = "1" * 40
BARRIER = "2" * 64
GRAPH_CONFIG = "3" * 64
LAST_CONFIG = "4" * 64


def _forecast(
    mean_x: float,
    variance: float,
    *,
    node_count: int = 2,
):
    mean = np.zeros((node_count, 3))
    mean[:, 0] = mean_x
    belief = GraphDynamicDiscrepancyBeliefV1.from_independent_endpoint_posterior(
        mean,
        np.full(node_count, variance),
        process_std_m=0.0,
    )
    return belief, belief.forecast([1, 3])


def _prediction(
    group_index: int,
    *,
    candidate_id: str = "graph_modal",
    mean_x: float = 0.4,
    configuration_sha256: str = GRAPH_CONFIG,
    accepted: bool = True,
):
    belief, forecast = _forecast(mean_x, 0.04)
    return seal_graph_dynamic_tournament_prediction(
        forecast,
        selected_horizon_index=0,
        candidate_id=candidate_id,
        unit_id=f"group-{group_index}-endpoint",
        group_id=f"group-{group_index}",
        horizon="endpoint",
        source_revision=REVISION,
        configuration_sha256=configuration_sha256,
        prediction_barrier_sha256=BARRIER,
        physical_fallback_mean_m=np.zeros((2, 3)),
        physical_fallback_covariance_m2=np.eye(6),
        graph_rank=belief.graph_basis.shape[1],
        parameter_count=14 if candidate_id == "graph_modal" else 0,
        runtime_milliseconds=1.0 if candidate_id == "graph_modal" else 0.1,
        accepted=accepted,
        reason="prediction-admissible" if accepted else "source-guard-rejected",
    )


def _bundle(
    *,
    candidate_id: str = "graph_modal",
    mean_x: float = 0.4,
    configuration_sha256: str = GRAPH_CONFIG,
):
    return build_graph_dynamic_tournament_prediction_bundle(
        [
            _prediction(
                index,
                candidate_id=candidate_id,
                mean_x=mean_x,
                configuration_sha256=configuration_sha256,
            )
            for index in range(3)
        ]
    )


def test_prediction_seal_binds_complete_forecast_before_scoring() -> None:
    belief, forecast = _forecast(0.25, 0.09)
    prediction = seal_graph_dynamic_tournament_prediction(
        forecast,
        selected_horizon_index=1,
        candidate_id="graph_modal",
        unit_id="object-a-late",
        group_id="object-a",
        horizon="late",
        source_revision=REVISION,
        configuration_sha256=GRAPH_CONFIG,
        prediction_barrier_sha256=BARRIER,
        physical_fallback_mean_m=np.zeros((2, 3)),
        physical_fallback_covariance_m2=np.eye(6),
        graph_rank=belief.graph_basis.shape[1],
        parameter_count=14,
        runtime_milliseconds=2.5,
        accepted=True,
        reason="prediction-admissible",
    )

    np.testing.assert_array_equal(prediction.mean_m, forecast.mean_m[1])
    np.testing.assert_array_equal(
        prediction.covariance_m2,
        forecast.joint_covariance_m2[6:12, 6:12],
    )
    assert prediction.horizon_step == 3
    assert prediction.to_record()["prediction_sealed_before_scoring"] is True
    assert len(prediction.artifact_id or "") == 64
    with pytest.raises(ValueError):
        prediction.source_mean_m.setflags(write=True)

    changed_forecast = np.asarray(prediction.source_mean_m).copy()
    changed_forecast[1, 0, 0] += 1.0
    with pytest.raises(ValueError, match="artifact_id"):
        replace(prediction, source_mean_m=changed_forecast)


def test_bundle_identity_is_order_invariant_and_binds_candidate_spec() -> None:
    predictions = [_prediction(index) for index in range(3)]
    first = build_graph_dynamic_tournament_prediction_bundle(predictions)
    second = build_graph_dynamic_tournament_prediction_bundle(
        tuple(reversed(predictions))
    )

    assert first.artifact_id == second.artifact_id
    assert first.candidate.prediction_artifact_sha256 == first.artifact_id
    assert first.candidate.state_dimension == 12
    assert first.candidate.parameter_count == 14
    assert first.candidate.runtime_milliseconds == pytest.approx(3.0)
    assert first.candidate.covariance_bytes == sum(
        prediction.source_joint_covariance_m2.nbytes
        for prediction in first.predictions
    )
    assert len(first.physical_fallback_artifact_sha256) == 64

    with pytest.raises(ValueError, match="artifact_id"):
        replace(first, artifact_id="0" * 64)


def test_rejected_graph_record_deploys_the_exact_physical_fallback() -> None:
    prediction = _prediction(0, accepted=False)
    bundle = build_graph_dynamic_tournament_prediction_bundle([prediction])
    target = np.full((2, 3), 0.25)
    scored = score_graph_dynamic_tournament_prediction_bundle(
        bundle,
        [target],
    )

    candidate = scored.candidate_records[0]
    fallback = scored.physical_fallback_records[0]
    assert candidate.accepted is False
    assert candidate.deployed_point_loss == fallback.point_loss
    assert candidate.deployed_proper_score == fallback.proper_score
    assert candidate.interval_covered == fallback.interval_covered
    assert candidate.interval_width == fallback.interval_width
    assert candidate.fallback_point_loss == fallback.point_loss
    assert candidate.fallback_proper_score == fallback.proper_score


def test_scored_bundle_is_content_addressed_and_owns_targets() -> None:
    bundle = _bundle()
    targets = [np.full((2, 3), 0.45) for _ in bundle.predictions]
    policy = GraphDynamicTournamentScoringPolicyV1()
    first = score_graph_dynamic_tournament_prediction_bundle(
        bundle,
        targets,
        scoring_policy=policy,
    )
    second = score_graph_dynamic_tournament_prediction_bundle(
        bundle,
        [target.copy() for target in targets],
        scoring_policy=policy,
    )

    assert first.artifact_id == second.artifact_id
    assert first.candidate == bundle.candidate
    assert len(first.candidate_records) == 3
    assert len(first.physical_fallback_records) == 3
    assert len(first.tournament_records) == 6
    with pytest.raises(ValueError):
        first.targets_m[0].setflags(write=True)
    with pytest.raises(ValueError, match="artifact_id"):
        replace(first, artifact_id="0" * 64)
    with pytest.raises(ValueError, match="complete prediction roster"):
        score_graph_dynamic_tournament_prediction_bundle(bundle, targets[:2])
    with pytest.raises(ValueError, match="shape"):
        score_graph_dynamic_tournament_prediction_bundle(
            bundle,
            [np.zeros((1, 3)), *targets[1:]],
        )


def test_scoring_policy_supports_registered_intervals_or_none() -> None:
    interval = GraphDynamicTournamentScoringPolicyV1()
    disabled = GraphDynamicTournamentScoringPolicyV1(
        nominal_interval_coverage=None,
        marginal_standard_score=None,
    )

    assert interval.interval_semantics_id == "coordinatewise-marginal-90-v1"
    assert disabled.interval_semantics_id == "interval-disabled-v1"
    assert interval.artifact_id != disabled.artifact_id
    with pytest.raises(ValueError, match="both be present or absent"):
        GraphDynamicTournamentScoringPolicyV1(
            nominal_interval_coverage=None,
        )
    with pytest.raises(ValueError, match="artifact_id"):
        replace(interval, artifact_id="0" * 64)


def test_graph_and_nested_reference_emit_a_valid_matched_tournament() -> None:
    graph_bundle = _bundle()
    reference_bundle = _bundle(
        candidate_id="last_residual",
        mean_x=0.0,
        configuration_sha256=LAST_CONFIG,
    )
    assert (
        graph_bundle.physical_fallback_artifact_sha256
        == reference_bundle.physical_fallback_artifact_sha256
    )
    targets = [np.full((2, 3), 0.5) for _ in graph_bundle.predictions]
    policy = GraphDynamicTournamentScoringPolicyV1()
    graph = score_graph_dynamic_tournament_prediction_bundle(
        graph_bundle,
        targets,
        scoring_policy=policy,
    )
    reference = score_graph_dynamic_tournament_prediction_bundle(
        reference_bundle,
        targets,
        scoring_policy=policy,
    )
    for graph_fallback, reference_fallback in zip(
        graph.physical_fallback_records,
        reference.physical_fallback_records,
        strict=True,
    ):
        assert tournament_record_dict(graph_fallback) == tournament_record_dict(
            reference_fallback
        )

    physical_candidate = CandidateSpec(
        candidate_id="physical_fallback",
        family="physical",
        state_dimension=0,
        parameter_count=0,
        runtime_milliseconds=0.0,
        covariance_bytes=sum(
            prediction.physical_fallback_covariance_m2.nbytes
            for prediction in graph_bundle.predictions
        ),
        source_revision=REVISION,
        configuration_sha256=graph_bundle.physical_fallback_artifact_sha256,
        prediction_artifact_sha256=(
            graph_bundle.physical_fallback_artifact_sha256
        ),
    )
    payload = {
        "contract": DISCREPANCY_TOURNAMENT_INPUT_CONTRACT,
        "schema_version": 1,
        "protocol_id": "graph-modal-source-v1",
        "statistical_unit": "physical-object-or-session",
        "split": "source-only",
        "reference_candidate": "last_residual",
        "physical_fallback_candidate": "physical_fallback",
        "information_boundary": {
            "candidate_predictions_sealed_before_scoring": True,
            "candidate_generation_used_scored_targets": False,
            "future_observations_used": False,
            "confirmation_payloads_opened": False,
            "replacement_allowed": False,
        },
        "evaluation": {
            "evaluator_revision": "5" * 40,
            "scoring_policy_sha256": policy.artifact_id,
            "scored_unit_roster_sha256": "6" * 64,
            "physical_fallback_artifact_sha256": (
                graph_bundle.physical_fallback_artifact_sha256
            ),
            "prediction_barrier_sha256": BARRIER,
            "point_loss_id": policy.point_loss_id,
            "proper_score_id": policy.proper_score_id,
            "interval_semantics_id": policy.interval_semantics_id,
        },
        "selection": {
            "minimum_group_count": 3,
            "minimum_relative_point_improvement": 0.0,
            "maximum_worst_group_relative_regression": 1.0,
            "maximum_harmful_accepted_count": 3,
            "maximum_mean_proper_score_regression": 1e12,
            "require_paired_point_upper_bound_nonpositive": False,
            "bootstrap_samples": 100,
            "bootstrap_seed": 20260810,
            "require_crossfit_stability": False,
            "nominal_interval_coverage": 0.9,
            "maximum_interval_coverage_shortfall": 0.9,
            "numerical_tolerance": 1e-12,
        },
        "candidates": [
            candidate_spec_dict(physical_candidate),
            candidate_spec_dict(reference.candidate),
            candidate_spec_dict(graph.candidate),
        ],
        "records": [
            *[
                tournament_record_dict(record)
                for record in graph.physical_fallback_records
            ],
            *[
                tournament_record_dict(record)
                for record in reference.candidate_records
            ],
            *[
                tournament_record_dict(record)
                for record in graph.candidate_records
            ],
        ],
    }

    evidence = parse_discrepancy_candidate_tournament(payload)

    assert len(evidence.candidates) == 3
    assert len(evidence.records) == 9
    assert evidence.reference_candidate == "last_residual"
    assert evidence.physical_fallback_candidate == "physical_fallback"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"accepted": False, "reason": "prediction-admissible"},
            "rejection reason",
        ),
        (
            {"accepted": True, "reason": "guard-rejected"},
            "prediction-admissible",
        ),
    ],
)
def test_prediction_admission_reason_fails_closed(
    kwargs: dict[str, object],
    message: str,
) -> None:
    belief, forecast = _forecast(0.1, 0.01)
    with pytest.raises(ValueError, match=message):
        seal_graph_dynamic_tournament_prediction(
            forecast,
            selected_horizon_index=0,
            candidate_id="graph_modal",
            unit_id="object-a-endpoint",
            group_id="object-a",
            horizon="endpoint",
            source_revision=REVISION,
            configuration_sha256=GRAPH_CONFIG,
            prediction_barrier_sha256=BARRIER,
            physical_fallback_mean_m=np.zeros((2, 3)),
            physical_fallback_covariance_m2=np.eye(6),
            graph_rank=belief.graph_basis.shape[1],
            parameter_count=14,
            runtime_milliseconds=1.0,
            **kwargs,
        )
