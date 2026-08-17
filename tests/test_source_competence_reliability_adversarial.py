from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import test_source_competence_reliability as cases

import bayesian_phystwin.source_competence_reliability as source


def test_scalar_sequence_and_literal_json_validators_fail_closed() -> None:
    for value in (True, np.bool_(False), "1.0", [1.0]):
        with pytest.raises(ValueError, match="finite real"):
            source._finite_real(value, name="value")
    with pytest.raises(ValueError, match="finite"):
        source._finite_real(np.nan, name="value")
    with pytest.raises(ValueError, match="positive"):
        source._finite_real(0.0, name="value", positive=True)

    with pytest.raises(ValueError, match="length 2"):
        source._sequence_ids(("only",), count=2)

    for value in (None, [], (1.0,)):
        with pytest.raises(ValueError, match="nonempty JSON array"):
            source._literal_json_float_array(value, name="values")
    for value in ([True], ["1"]):
        with pytest.raises(ValueError, match="literal JSON number"):
            source._literal_json_float_array(value, name="values")
    with pytest.raises(ValueError, match="finite JSON number"):
        source._literal_json_float_array([10**10000], name="values")
    with pytest.raises(ValueError, match="finite"):
        source._literal_json_float_array([float("inf")], name="values")


def test_markov_config_rejects_invalid_probability_and_semantic_contracts() -> None:
    for field, value, match in (
        ("inlier_persistence", 0.0, "inlier_persistence"),
        ("outlier_persistence", 1.0, "outlier_persistence"),
        ("probability_floor", 0.5, "probability_floor"),
        ("time_delta_mode", "unsupported", "time_delta_mode"),
        ("composition", "promote-provider-prior", "composition"),
    ):
        with pytest.raises(ValueError, match=match):
            source.SourceCompetenceMarkovConfigV1(**{field: value})


def test_evidence_constructor_rejects_shape_finiteness_and_feature_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = cases._observation()
    with pytest.raises(ValueError, match="feature_names must be unique"):
        cases._evidence(observation, feature_names=("same", "same"))
    with pytest.raises(ValueError, match="time_values must be"):
        cases._evidence(
            observation,
            time_values=np.arange(observation.observation_count, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="shape"):
        cases._evidence(
            observation,
            competent=np.zeros((observation.observation_count, 1), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros(observation.observation_count, dtype=np.float64)
        bad[0] = np.nan
        cases._evidence(observation, incompetent=bad)

    monkeypatch.setattr(source, "_sequence_ids", lambda _value, *, count: ())
    with pytest.raises(ValueError, match="must contain rows"):
        source.SourceCompetenceEvidenceV1(
            observation_artifact_id=observation.artifact_id,
            observation_identity_sha256="1" * 64,
            source_feature_artifact_id="2" * 64,
            source_reliability_model_id="3" * 64,
            causal_frame_stop=observation.causal_frame_stop,
            feature_names=("feature",),
            sequence_ids=(),
            time_values=np.asarray([], dtype=np.float64),
            log_competent_density=np.asarray([], dtype=np.float64),
            log_incompetent_density=np.asarray([], dtype=np.float64),
        )


def test_evidence_mapping_loader_rejects_schema_and_container_drift() -> None:
    observation = cases._observation()
    record = cases._evidence(observation).to_record()
    for key, value, match in (
        ("schema", "wrong", "schema differs"),
        ("schema_version", 2, "version differs"),
        ("claim_boundary", "wrong", "claim boundary differs"),
    ):
        payload = dict(record)
        payload[key] = value
        with pytest.raises(ValueError, match=match):
            source.SourceCompetenceEvidenceV1.from_mapping(payload)

    payload = dict(record)
    payload["feature_names"] = "feature"
    with pytest.raises(ValueError, match="sequence fields must be lists"):
        source.SourceCompetenceEvidenceV1.from_mapping(payload)
    payload = dict(record)
    payload["sequence_ids"] = "sequence"
    with pytest.raises(ValueError, match="sequence fields must be lists"):
        source.SourceCompetenceEvidenceV1.from_mapping(payload)
    payload = dict(record)
    payload["metadata"] = []
    with pytest.raises(ValueError, match="metadata must be an object"):
        source.SourceCompetenceEvidenceV1.from_mapping(payload)


def test_update_constructor_rejects_types_arrays_lineage_and_identity_drift() -> None:
    observation = cases._observation()
    evidence = cases._evidence(observation)
    update = source.refine_observation_source_competence(observation, evidence)

    for field, value, match in (
        ("source_observation", object(), "source_observation"),
        ("refined_observation", object(), "refined_observation"),
        ("evidence", object(), "evidence"),
        ("config", object(), "config"),
    ):
        with pytest.raises(TypeError, match=match):
            replace(update, **{field: value, "update_id": None})

    with pytest.raises(ValueError, match="float64 with shape"):
        replace(
            update,
            posterior_competence_probability=np.zeros(
                observation.observation_count, dtype=np.float32
            ),
            update_id=None,
        )
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        replace(
            update,
            deployed_prior_reliability=np.full(
                observation.observation_count, 2.0, dtype=np.float64
            ),
            update_id=None,
        )
    with pytest.raises(TypeError, match="sequence_log_evidence"):
        replace(update, sequence_log_evidence=[], update_id=None)

    other = cases._observation(prior=0.5)
    with pytest.raises(ValueError, match="different source observation"):
        replace(
            update,
            evidence=cases._evidence(other),
            update_id=None,
        )

    short = replace(
        evidence,
        sequence_ids=evidence.sequence_ids[:-1],
        time_values=evidence.time_values[:-1],
        log_competent_density=evidence.log_competent_density[:-1],
        log_incompetent_density=evidence.log_incompetent_density[:-1],
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="row count differs"):
        replace(update, evidence=short, update_id=None)

    identity_drift = replace(
        evidence,
        observation_identity_sha256="f" * 64,
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="row identity digest differs"):
        replace(update, evidence=identity_drift, update_id=None)

    cutoff_drift = replace(
        evidence,
        causal_frame_stop=evidence.causal_frame_stop + 1,
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="causal frame cutoff differs"):
        replace(update, evidence=cutoff_drift, update_id=None)

    with pytest.raises(ValueError, match="update ID mismatch"):
        replace(update, update_id="0" * 64)


def test_public_validation_and_persistence_type_guards() -> None:
    observation = cases._observation()
    evidence = cases._evidence(observation)
    with pytest.raises(TypeError, match="observation"):
        source.validate_source_competence_evidence(object(), evidence)
    with pytest.raises(TypeError, match="evidence"):
        source.validate_source_competence_evidence(observation, object())
    with pytest.raises(TypeError, match="evidence"):
        source.write_source_competence_evidence(object(), "unused.json")
