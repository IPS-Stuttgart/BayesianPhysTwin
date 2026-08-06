from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin._prob4d_stream_binding import (
    prob4d_observation_identity_summary,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.source_competence_reliability import (
    SourceCompetenceEvidenceV1,
    SourceCompetenceMarkovConfigV1,
    load_source_competence_evidence,
    refine_observation_source_competence,
    write_source_competence_evidence,
)
from bayesian_phystwin.structured_reliability import (
    MARKOV_TIME_MODE_INTEGER_STEPS,
)


def _observation(*, prior: float = 0.65) -> ObservationBeliefV1:
    count = 6
    frames = np.arange(count, dtype=np.int64)
    return ObservationBeliefV1(
        case_id="case-a",
        stream_id="stream-a",
        causal_frame_stop=count + 1,
        view_names=("cam-0",),
        window_names=("window-0",),
        factor_names=("shared-gauge",),
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="1" * 40,
        source_artifact_sha256="2" * 64,
        declared_frame_ids=frames,
        mean_xyz_m=np.column_stack(
            (
                0.01 * frames,
                np.zeros(count),
                np.ones(count),
            )
        ).astype(np.float64),
        frame_ids=frames,
        entity_ids=np.asarray([10, 10, 10, 11, 11, 11], dtype=np.int64),
        view_indices=np.zeros(count, dtype=np.int64),
        window_indices=np.zeros(count, dtype=np.int64),
        correlation_group_ids=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        factor_group_ids=np.zeros(count, dtype=np.int64),
        prior_reliability=np.full(count, prior, dtype=np.float64),
        association_probability=np.asarray(
            [0.95, 0.94, 0.93, 0.92, 0.91, 0.90],
            dtype=np.float64,
        ),
        local_covariance_m2=np.broadcast_to(
            1e-4 * np.eye(3),
            (count, 3, 3),
        ).copy(),
        low_rank_factor_m=np.full((count, 3, 1), 1e-3, dtype=np.float64),
        group_ids=np.asarray([0, 1], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([0.8, 0.75]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata={"producer": "source-competence test"},
    )


def _evidence(
    observation: ObservationBeliefV1,
    *,
    time_values: np.ndarray | None = None,
    competent: np.ndarray | None = None,
    incompetent: np.ndarray | None = None,
    **overrides: object,
) -> SourceCompetenceEvidenceV1:
    _, _, identity = prob4d_observation_identity_summary(observation)
    count = observation.observation_count
    values = {
        "observation_artifact_id": observation.artifact_id,
        "observation_identity_sha256": identity,
        "source_feature_artifact_id": "3" * 64,
        "source_reliability_model_id": "4" * 64,
        "causal_frame_stop": observation.causal_frame_stop,
        "feature_names": (
            "overlap_disagreement",
            "triangulation_condition",
            "track_age",
        ),
        "sequence_ids": (
            "track-10",
            "track-10",
            "track-10",
            "track-11",
            "track-11",
            "track-11",
        ),
        "time_values": (
            np.arange(count, dtype=np.float64) if time_values is None else time_values
        ),
        "log_competent_density": (
            np.asarray([2.0, 1.5, -5.0, 2.0, -4.0, -4.0], dtype=np.float64)
            if competent is None
            else competent
        ),
        "log_incompetent_density": (
            np.zeros(count, dtype=np.float64) if incompetent is None else incompetent
        ),
        "metadata": {
            "feature_semantics": "target-blind source-only features",
            "uses_truth": False,
        },
    }
    values.update(overrides)
    return SourceCompetenceEvidenceV1(**values)  # type: ignore[arg-type]


def test_temporal_source_competence_only_reduces_provider_reliability() -> None:
    observation = _observation()
    evidence = _evidence(observation)
    update = refine_observation_source_competence(observation, evidence)

    assert update.refined_observation is not observation
    assert np.all(update.deployed_prior_reliability <= observation.prior_reliability)
    assert np.any(update.deployed_prior_reliability < observation.prior_reliability)
    assert np.array_equal(
        update.refined_observation.prior_reliability,
        update.deployed_prior_reliability,
    )
    assert np.array_equal(
        observation.prior_reliability,
        np.full(observation.observation_count, 0.65),
    )
    assert set(update.sequence_log_evidence) == {"track-10", "track-11"}

    unchanged = (
        "declared_frame_ids",
        "mean_xyz_m",
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "correlation_group_ids",
        "factor_group_ids",
        "association_probability",
        "local_covariance_m2",
        "low_rank_factor_m",
        "group_ids",
        "group_prior_nominal_probability",
        "group_composite_weight",
    )
    for name in unchanged:
        assert np.array_equal(
            getattr(update.refined_observation, name),
            getattr(observation, name),
        )
    metadata = update.refined_observation.metadata
    assert metadata["source_competence_evidence_id"] == evidence.artifact_id
    assert metadata["source_competence_covariance_changed"] is False
    assert metadata["source_competence_association_probability_changed"] is False
    assert metadata["source_competence_uses_physical_innovation"] is False


def test_strong_competence_evidence_cannot_exceed_provider_prior() -> None:
    observation = _observation(prior=0.2)
    evidence = _evidence(
        observation,
        competent=np.full(6, 20.0, dtype=np.float64),
        incompetent=np.zeros(6, dtype=np.float64),
    )
    update = refine_observation_source_competence(observation, evidence)
    np.testing.assert_allclose(
        update.deployed_prior_reliability,
        observation.prior_reliability,
    )
    assert np.all(update.posterior_competence_probability > 0.99)


def test_evidence_roundtrip_and_arrays_are_irreversibly_immutable(tmp_path) -> None:
    observation = _observation()
    evidence = _evidence(observation)
    path = tmp_path / "source-competence.json"
    write_source_competence_evidence(evidence, path)
    restored = load_source_competence_evidence(path)
    assert restored.artifact_id == evidence.artifact_id
    assert restored.identity_record() == evidence.identity_record()
    for array in (
        restored.time_values,
        restored.log_competent_density,
        restored.log_incompetent_density,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)

    update = refine_observation_source_competence(observation, restored)
    for array in (
        update.posterior_competence_probability,
        update.deployed_prior_reliability,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_row_identity_artifact_and_cutoff_mismatch_fail_closed() -> None:
    observation = _observation()
    evidence = _evidence(observation)
    with pytest.raises(ValueError, match="another observation"):
        refine_observation_source_competence(
            observation,
            replace(evidence, observation_artifact_id="5" * 64, artifact_id=None),
        )
    with pytest.raises(ValueError, match="row identity digest"):
        refine_observation_source_competence(
            observation,
            replace(
                evidence,
                observation_identity_sha256="6" * 64,
                artifact_id=None,
            ),
        )
    with pytest.raises(ValueError, match="causal frame cutoff"):
        refine_observation_source_competence(
            observation,
            replace(
                evidence,
                causal_frame_stop=observation.causal_frame_stop + 1,
                artifact_id=None,
            ),
        )


@pytest.mark.parametrize(
    "field",
    [
        "uses_target_outcomes",
        "uses_physical_innovation",
        "uses_posterior_responsibility",
        "uses_association_probability_as_label",
    ],
)
def test_forbidden_information_sources_are_rejected(field: str) -> None:
    observation = _observation()
    with pytest.raises(ValueError, match=field):
        _evidence(observation, **{field: True})


def test_integer_step_gaps_are_supported_but_fractional_gaps_fail() -> None:
    observation = _observation()
    config = SourceCompetenceMarkovConfigV1(
        time_delta_mode=MARKOV_TIME_MODE_INTEGER_STEPS,
        time_step=1.0,
    )
    evidence = _evidence(
        observation,
        time_values=np.asarray([0.0, 2.0, 5.0, 0.0, 3.0, 7.0]),
    )
    update = refine_observation_source_competence(
        observation,
        evidence,
        config=config,
    )
    assert update.config.config_id == config.config_id

    fractional = _evidence(
        observation,
        time_values=np.asarray([0.0, 1.5, 3.0, 0.0, 2.0, 4.0]),
    )
    with pytest.raises(ValueError, match="integer multiples"):
        refine_observation_source_competence(
            observation,
            fractional,
            config=config,
        )


def test_config_and_evidence_content_id_reject_tampering() -> None:
    observation = _observation()
    evidence = _evidence(observation)
    with pytest.raises(ValueError, match="evidence ID mismatch"):
        replace(
            evidence,
            log_competent_density=np.zeros(6, dtype=np.float64),
        )
    config = SourceCompetenceMarkovConfigV1()
    with pytest.raises(ValueError, match="config ID mismatch"):
        replace(config, inlier_persistence=0.9)


def test_duplicate_json_keys_and_non_boolean_declarations_are_rejected(
    tmp_path,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"a","schema":"b"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_source_competence_evidence(path)

    observation = _observation()
    with pytest.raises(ValueError, match="must be a boolean"):
        _evidence(observation, uses_target_outcomes=0)


def test_wrong_config_type_is_rejected() -> None:
    observation = _observation()
    evidence = _evidence(observation)
    with pytest.raises(TypeError, match="SourceCompetenceMarkovConfigV1"):
        refine_observation_source_competence(
            observation,
            evidence,
            config=object(),  # type: ignore[arg-type]
        )
