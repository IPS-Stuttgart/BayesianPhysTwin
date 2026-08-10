from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin._graph_dynamic_tournament_common import (
    canonical_string,
    finite_real,
    float_array,
    genuine_boolean,
    genuine_integer,
    identifier,
    integer_vector,
    validated_covariance,
)
from bayesian_phystwin._graph_dynamic_tournament_score import _regularized_score
from bayesian_phystwin.graph_dynamic_discrepancy import (
    GraphDynamicDiscrepancyBeliefV1,
)
from bayesian_phystwin.graph_dynamic_discrepancy_tournament import (
    GraphDynamicTournamentPredictionBundleV1,
    GraphDynamicTournamentScoredBundleV1,
    GraphDynamicTournamentScoringPolicyV1,
    build_graph_dynamic_tournament_prediction_bundle,
    score_graph_dynamic_tournament_prediction_bundle,
    seal_graph_dynamic_tournament_prediction,
)

REVISION = "1" * 40
CONFIGURATION = "2" * 64
BARRIER = "3" * 64


def _valid_prediction(*, unit_id: str = "object-a-endpoint"):
    belief = GraphDynamicDiscrepancyBeliefV1.from_independent_endpoint_posterior(
        np.zeros((2, 3)),
        np.ones(2),
        process_std_m=0.0,
    )
    forecast = belief.forecast([1, 2])
    return seal_graph_dynamic_tournament_prediction(
        forecast,
        selected_horizon_index=0,
        candidate_id="graph_modal",
        unit_id=unit_id,
        group_id="object-a",
        horizon="endpoint",
        source_revision=REVISION,
        configuration_sha256=CONFIGURATION,
        prediction_barrier_sha256=BARRIER,
        physical_fallback_mean_m=np.zeros((2, 3)),
        physical_fallback_covariance_m2=np.eye(6),
        graph_rank=belief.graph_basis.shape[1],
        parameter_count=12,
        runtime_milliseconds=1.0,
        accepted=True,
        reason="prediction-admissible",
        metadata={"fixture": "coverage"},
    )


def _valid_bundle():
    return build_graph_dynamic_tournament_prediction_bundle([_valid_prediction()])


@pytest.mark.parametrize("value", [None, "", " padded ", 1])
def test_common_rejects_noncanonical_strings(value: object) -> None:
    with pytest.raises(ValueError, match="canonical string"):
        canonical_string(value, name="value")


def test_common_rejects_non_identifier_string() -> None:
    with pytest.raises(ValueError, match="lowercase identifier"):
        identifier("Not Valid", name="identifier")


@pytest.mark.parametrize("value", [1, "true", None])
def test_common_rejects_non_boolean_values(value: object) -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        genuine_boolean(value, name="value")


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_common_rejects_non_integer_values(value: object) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        genuine_integer(value, name="value")


def test_common_rejects_integer_below_minimum() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        genuine_integer(1, name="value", minimum=2)


@pytest.mark.parametrize("value", [True, "1", None, np.nan, np.inf])
def test_common_rejects_nonfinite_real_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite number"):
        finite_real(value, name="value")


def test_common_rejects_real_outside_declared_bounds() -> None:
    with pytest.raises(ValueError, match="at least 0.0"):
        finite_real(-1.0, name="value", minimum=0.0)
    with pytest.raises(ValueError, match="at most 1.0"):
        finite_real(2.0, name="value", maximum=1.0)


def test_common_rejects_invalid_float_arrays() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        float_array(np.asarray(["1"]), name="value")
    with pytest.raises(ValueError, match="finite"):
        float_array(np.asarray([np.nan]), name="value")


def test_common_rejects_nonvector_integer_array() -> None:
    with pytest.raises(ValueError, match="must be a vector"):
        integer_vector(np.ones((1, 1), dtype=np.int64), name="value")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.ones((1, 2)), "shape changed"),
        (np.asarray([[1.0, 1.0], [0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, -1.0]), "positive semidefinite"),
    ],
)
def test_common_rejects_invalid_covariances(
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validated_covariance(value, name="covariance", dimension=2)


def test_scoring_policy_covers_all_fail_closed_branches() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        GraphDynamicTournamentScoringPolicyV1(covariance_floor_m2=0.0)
    for coverage in (0.0, 1.0):
        with pytest.raises(ValueError, match="strictly inside"):
            GraphDynamicTournamentScoringPolicyV1(
                nominal_interval_coverage=coverage,
            )
    with pytest.raises(ValueError, match="must be positive"):
        GraphDynamicTournamentScoringPolicyV1(marginal_standard_score=0.0)

    policy = GraphDynamicTournamentScoringPolicyV1()
    reconstructed = replace(policy, artifact_id=policy.artifact_id)
    assert reconstructed.artifact_id == policy.artifact_id


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_horizon_steps": np.asarray([], dtype=np.int64)}, "positive"),
        ({"source_horizon_steps": np.asarray([1, 1])}, "strictly increasing"),
        ({"node_indices": np.asarray([], dtype=np.int64)}, "nonempty"),
        ({"node_indices": np.asarray([0, 0])}, "unique"),
        ({"source_mean_m": np.zeros((1, 2, 3))}, "shape changed"),
        ({"selected_horizon_index": 2}, "outside the forecast"),
        ({"physical_fallback_mean_m": np.zeros((1, 3))}, "shape changed"),
    ],
)
def test_prediction_rejects_shape_and_roster_drift(
    changes: dict[str, Any],
    message: str,
) -> None:
    prediction = _valid_prediction()
    with pytest.raises(ValueError, match=message):
        replace(prediction, **changes, artifact_id=None)


def test_prediction_accepts_matching_explicit_artifact_identity() -> None:
    prediction = _valid_prediction()
    reconstructed = replace(prediction, artifact_id=prediction.artifact_id)
    assert reconstructed.artifact_id == prediction.artifact_id


def test_seal_rejects_nonforecast_value() -> None:
    with pytest.raises(TypeError, match="GraphDynamicDiscrepancyForecastV1"):
        seal_graph_dynamic_tournament_prediction(
            cast(Any, object()),
            selected_horizon_index=0,
            candidate_id="graph_modal",
            unit_id="object-a-endpoint",
            group_id="object-a",
            horizon="endpoint",
            source_revision=REVISION,
            configuration_sha256=CONFIGURATION,
            prediction_barrier_sha256=BARRIER,
            physical_fallback_mean_m=np.zeros((2, 3)),
            physical_fallback_covariance_m2=np.eye(6),
            graph_rank=2,
            parameter_count=12,
            runtime_milliseconds=1.0,
            accepted=True,
            reason="prediction-admissible",
        )


def test_bundle_rejects_malformed_prediction_sequences() -> None:
    with pytest.raises(ValueError, match="must be a sequence"):
        GraphDynamicTournamentPredictionBundleV1(predictions=cast(Any, "not-a-roster"))
    with pytest.raises(ValueError, match="must contain"):
        GraphDynamicTournamentPredictionBundleV1(predictions=[])
    with pytest.raises(ValueError, match="must contain"):
        GraphDynamicTournamentPredictionBundleV1(predictions=cast(Any, [object()]))


def test_bundle_rejects_duplicate_units_and_candidate_drift() -> None:
    prediction = _valid_prediction()
    with pytest.raises(ValueError, match="unit_id values must be unique"):
        GraphDynamicTournamentPredictionBundleV1(predictions=[prediction, prediction])

    changed = replace(
        prediction,
        candidate_id="other_candidate",
        unit_id="object-b-endpoint",
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="changed candidate"):
        GraphDynamicTournamentPredictionBundleV1(predictions=[prediction, changed])


def test_bundle_accepts_matching_explicit_artifact_identity() -> None:
    bundle = _valid_bundle()
    reconstructed = replace(bundle, artifact_id=bundle.artifact_id)
    assert reconstructed.artifact_id == bundle.artifact_id


def test_regularized_score_covers_disabled_and_numerical_failure_paths() -> None:
    disabled = GraphDynamicTournamentScoringPolicyV1(
        nominal_interval_coverage=None,
        marginal_standard_score=None,
    )
    point, proper, covered, width = _regularized_score(
        np.zeros((1, 3)),
        np.eye(3),
        np.zeros((1, 3)),
        disabled,
    )
    assert np.isfinite(point)
    assert np.isfinite(proper)
    assert covered is None
    assert width is None

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="score is not finite"):
            _regularized_score(
                np.full((1, 3), 1.0e308),
                np.eye(3),
                np.full((1, 3), -1.0e308),
                GraphDynamicTournamentScoringPolicyV1(),
            )
        with pytest.raises(ValueError, match="interval width is not finite"):
            _regularized_score(
                np.zeros((1, 3)),
                np.eye(3) * 1.0e150,
                np.zeros((1, 3)),
                GraphDynamicTournamentScoringPolicyV1(
                    marginal_standard_score=1.0e308,
                ),
            )


def test_scored_bundle_rejects_invalid_constructor_types() -> None:
    policy = GraphDynamicTournamentScoringPolicyV1()
    bundle = _valid_bundle()
    target = np.zeros((2, 3))

    with pytest.raises(TypeError, match="prediction_bundle"):
        GraphDynamicTournamentScoredBundleV1(
            prediction_bundle=cast(Any, object()),
            targets_m=[],
            scoring_policy=policy,
        )
    with pytest.raises(TypeError, match="scoring_policy"):
        GraphDynamicTournamentScoredBundleV1(
            prediction_bundle=bundle,
            targets_m=[target],
            scoring_policy=cast(Any, object()),
        )
    with pytest.raises(ValueError, match="targets_m must be a sequence"):
        GraphDynamicTournamentScoredBundleV1(
            prediction_bundle=bundle,
            targets_m=cast(Any, "not-targets"),
            scoring_policy=policy,
        )


def test_scored_bundle_accepts_matching_explicit_artifact_identity() -> None:
    scored = score_graph_dynamic_tournament_prediction_bundle(
        _valid_bundle(),
        [np.zeros((2, 3))],
    )
    reconstructed = replace(scored, artifact_id=scored.artifact_id)
    assert reconstructed.artifact_id == scored.artifact_id
