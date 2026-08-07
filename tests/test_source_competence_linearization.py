from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin._prob4d_stream_binding import (
    prob4d_observation_identity_summary,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.physical_linearization import (
    PhysicalLinearizationV1,
    validate_observation_linearization_alignment,
)
from bayesian_phystwin.source_competence_linearization import (
    SOURCE_COMPETENCE_LINEARIZATION_BINDING,
    rebind_physical_linearization_to_source_competence,
)
from bayesian_phystwin.source_competence_reliability import (
    SourceCompetenceEvidenceV1,
    refine_observation_source_competence,
)


def _observation(*, case_id: str = "case-a") -> ObservationBeliefV1:
    frame_ids = np.asarray([0, 1], dtype=np.int64)
    return ObservationBeliefV1(
        case_id=case_id,
        stream_id="stream-a",
        causal_frame_stop=2,
        view_names=("cam-0",),
        window_names=("window-0",),
        factor_names=(),
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="1" * 40,
        source_artifact_sha256="2" * 64,
        declared_frame_ids=frame_ids,
        mean_xyz_m=np.asarray(
            [[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]],
            dtype=np.float64,
        ),
        frame_ids=frame_ids,
        entity_ids=np.asarray([10, 10], dtype=np.int64),
        view_indices=np.zeros(2, dtype=np.int64),
        window_indices=np.zeros(2, dtype=np.int64),
        correlation_group_ids=np.zeros(2, dtype=np.int64),
        factor_group_ids=np.zeros(2, dtype=np.int64),
        prior_reliability=np.asarray([0.8, 0.8], dtype=np.float64),
        association_probability=np.asarray([0.9, 0.9], dtype=np.float64),
        local_covariance_m2=np.broadcast_to(
            1e-4 * np.eye(3),
            (2, 3, 3),
        ).copy(),
        low_rank_factor_m=np.zeros((2, 3, 0), dtype=np.float64),
        group_ids=np.asarray([0], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([0.9], dtype=np.float64),
        group_composite_weight=np.asarray([1.0], dtype=np.float64),
        metadata={"producer": "source-competence linearization test"},
    )


def _evidence(observation: ObservationBeliefV1) -> SourceCompetenceEvidenceV1:
    _, _, identity = prob4d_observation_identity_summary(observation)
    return SourceCompetenceEvidenceV1(
        observation_artifact_id=observation.artifact_id,
        observation_identity_sha256=identity,
        source_feature_artifact_id="3" * 64,
        source_reliability_model_id="4" * 64,
        causal_frame_stop=observation.causal_frame_stop,
        feature_names=("overlap_disagreement",),
        sequence_ids=("track-10", "track-10"),
        time_values=np.asarray([0.0, 1.0], dtype=np.float64),
        log_competent_density=np.asarray([1.0, -3.0], dtype=np.float64),
        log_incompetent_density=np.zeros(2, dtype=np.float64),
        metadata={"uses_truth": False},
    )


def _linearization(
    observation: ObservationBeliefV1,
    *,
    metadata: dict[str, object] | None = None,
) -> PhysicalLinearizationV1:
    return PhysicalLinearizationV1(
        observation_artifact_id=observation.artifact_id,
        baseline_belief_id="5" * 64,
        action_prefix_id="6" * 64,
        simulator_revision="source-competence-linearization-test",
        frame_ids=observation.frame_ids,
        entity_ids=observation.entity_ids,
        view_indices=observation.view_indices,
        window_indices=observation.window_indices,
        state_jacobian=np.asarray(
            [
                [[1.0], [0.0], [0.0]],
                [[1.0], [0.0], [0.0]],
            ],
            dtype=np.float64,
        ),
        query_state_jacobian=np.asarray(
            [[[1.0], [0.0], [0.0]]],
            dtype=np.float64,
        ),
        physical_response_m=np.asarray([[0.01, 0.0, 0.0]], dtype=np.float64),
        metadata=metadata or {"protocol": "source-competence-test"},
    )


def test_rebind_changes_only_observation_identity_and_metadata() -> None:
    observation = _observation()
    update = refine_observation_source_competence(
        observation,
        _evidence(observation),
    )
    linearization = _linearization(observation)
    rebound = rebind_physical_linearization_to_source_competence(
        update,
        linearization,
    )

    assert rebound.observation_artifact_id == update.refined_observation.artifact_id
    assert rebound.artifact_id != linearization.artifact_id
    assert rebound.baseline_belief_id == linearization.baseline_belief_id
    assert rebound.action_prefix_id == linearization.action_prefix_id
    assert rebound.simulator_revision == linearization.simulator_revision
    for name in (
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "state_jacobian",
        "query_state_jacobian",
        "physical_response_m",
    ):
        assert np.array_equal(getattr(rebound, name), getattr(linearization, name))
    assert (
        rebound.metadata["source_competence_linearization_binding"]
        == SOURCE_COMPETENCE_LINEARIZATION_BINDING
    )
    assert rebound.metadata["source_competence_rows_changed"] is False
    assert rebound.metadata["source_competence_jacobians_changed"] is False
    validate_observation_linearization_alignment(
        update.refined_observation,
        rebound,
    )
    with pytest.raises(ValueError, match="does not identify"):
        validate_observation_linearization_alignment(
            update.refined_observation,
            linearization,
        )


def test_rebind_rejects_linearization_for_another_source_observation() -> None:
    observation = _observation()
    other = _observation(case_id="case-b")
    update = refine_observation_source_competence(
        observation,
        _evidence(observation),
    )
    with pytest.raises(ValueError, match="does not identify"):
        rebind_physical_linearization_to_source_competence(
            update,
            _linearization(other),
        )


def test_rebind_rejects_conflicting_existing_metadata() -> None:
    observation = _observation()
    update = refine_observation_source_competence(
        observation,
        _evidence(observation),
    )
    linearization = _linearization(
        observation,
        metadata={
            "source_competence_update_id": "7" * 64,
        },
    )
    with pytest.raises(ValueError, match="conflicts with source_competence_update_id"):
        rebind_physical_linearization_to_source_competence(
            update,
            linearization,
        )


def test_rebind_rejects_wrong_types() -> None:
    observation = _observation()
    update = refine_observation_source_competence(
        observation,
        _evidence(observation),
    )
    linearization = _linearization(observation)
    with pytest.raises(TypeError, match="SourceCompetenceReliabilityUpdateV1"):
        rebind_physical_linearization_to_source_competence(  # type: ignore[arg-type]
            object(),
            linearization,
        )
    with pytest.raises(TypeError, match="PhysicalLinearizationV1"):
        rebind_physical_linearization_to_source_competence(  # type: ignore[arg-type]
            update,
            object(),
        )


def test_rebound_artifact_is_deterministic() -> None:
    observation = _observation()
    update = refine_observation_source_competence(
        observation,
        _evidence(observation),
    )
    first = rebind_physical_linearization_to_source_competence(
        update,
        _linearization(observation),
    )
    second = rebind_physical_linearization_to_source_competence(
        update,
        replace(_linearization(observation)),
    )
    assert first.artifact_id == second.artifact_id
    assert first.descriptor() == second.descriptor()
