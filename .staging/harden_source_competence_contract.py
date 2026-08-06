from __future__ import annotations

from pathlib import Path

SOURCE = Path("src/bayesian_phystwin/source_competence_reliability.py")
TESTS = Path("tests/test_source_competence_reliability.py")

source = SOURCE.read_text(encoding="utf-8")
if "def _literal_json_float_array" not in source:
    old_sequence = '''def _sequence_ids(value: Sequence[str], *, count: int) -> tuple[str, ...]:
    result = canonical_string_tuple(
        value,
        name="sequence_ids",
        allow_empty=False,
    )
    if len(result) != count:
        raise ValueError(f"sequence_ids must have length {count}")
    return result


@dataclass(frozen=True, slots=True)
class SourceCompetenceMarkovConfigV1:
'''
    new_sequence = '''def _sequence_ids(value: Sequence[str], *, count: int) -> tuple[str, ...]:
    result = canonical_string_tuple(
        value,
        name="sequence_ids",
        allow_empty=False,
    )
    if len(result) != count:
        raise ValueError(f"sequence_ids must have length {count}")
    return result


def _literal_json_float_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty JSON array")
    result = np.empty(len(value), dtype=np.float64)
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name}[{index}] must be a literal JSON number")
        try:
            number = float(raw)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{name}[{index}] must be a finite JSON number") from error
        if not np.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result[index] = number
    return result


@dataclass(frozen=True, slots=True)
class SourceCompetenceMarkovConfigV1:
'''
    if old_sequence not in source:
        raise SystemExit("sequence helper anchor changed")
    source = source.replace(old_sequence, new_sequence, 1)

    old_loader = '''            time_values=np.asarray(value.get("time_values"), dtype=np.float64),
            log_competent_density=np.asarray(
                value.get("log_competent_density"), dtype=np.float64
            ),
            log_incompetent_density=np.asarray(
                value.get("log_incompetent_density"), dtype=np.float64
            ),
'''
    new_loader = '''            time_values=_literal_json_float_array(
                value.get("time_values"), name="time_values"
            ),
            log_competent_density=_literal_json_float_array(
                value.get("log_competent_density"),
                name="log_competent_density",
            ),
            log_incompetent_density=_literal_json_float_array(
                value.get("log_incompetent_density"),
                name="log_incompetent_density",
            ),
'''
    if old_loader not in source:
        raise SystemExit("strict loader anchor changed")
    source = source.replace(old_loader, new_loader, 1)

    old_update = '''        if np.any(deployed > self.source_observation.prior_reliability):
            raise ValueError("deployed reliability exceeds the provider prior")
        if not np.array_equal(
            deployed,
            self.refined_observation.prior_reliability,
        ):
            raise ValueError(
                "refined observation does not contain deployed reliability"
            )
        if self.evidence.observation_artifact_id != self.source_observation.artifact_id:
            raise ValueError("evidence identifies a different source observation")
        if not isinstance(self.sequence_log_evidence, Mapping):
            raise TypeError("sequence_log_evidence must be a mapping")
        evidence_mapping = frozen_finite_json_mapping(
            self.sequence_log_evidence,
            name="source-competence sequence log evidence",
        )
'''
    new_update = '''        if not isinstance(self.sequence_log_evidence, Mapping):
            raise TypeError("sequence_log_evidence must be a mapping")
        if self.evidence.observation_artifact_id != self.source_observation.artifact_id:
            raise ValueError("evidence identifies a different source observation")
        _, evidence_count, identity_sha = prob4d_observation_identity_summary(
            self.source_observation
        )
        if self.evidence.observation_count != evidence_count:
            raise ValueError("source-competence evidence row count differs")
        if self.evidence.observation_identity_sha256 != identity_sha:
            raise ValueError("source-competence row identity digest differs")
        if (
            self.evidence.causal_frame_stop
            != self.source_observation.causal_frame_stop
        ):
            raise ValueError("source-competence causal frame cutoff differs")

        expected_smoothed = smooth_markov_reliability(
            self.source_observation.prior_reliability,
            self.evidence.log_competent_density,
            self.evidence.log_incompetent_density,
            self.evidence.sequence_ids,
            self.evidence.time_values,
            config=self.config.as_markov_config(),
        )
        expected_posterior = np.asarray(
            expected_smoothed.posterior_inlier_probability,
            dtype=np.float64,
        )
        if not np.array_equal(posterior, expected_posterior):
            raise ValueError(
                "posterior competence probability differs from the bound model"
            )
        expected_deployed = np.minimum(
            self.source_observation.prior_reliability,
            expected_posterior,
        )
        if not np.array_equal(deployed, expected_deployed):
            raise ValueError(
                "deployed reliability does not equal the conservative composition"
            )
        provided_evidence_mapping = frozen_finite_json_mapping(
            self.sequence_log_evidence,
            name="source-competence sequence log evidence",
        )
        expected_evidence_mapping = frozen_finite_json_mapping(
            expected_smoothed.sequence_log_evidence,
            name="expected source-competence sequence log evidence",
        )
        if plain_json(provided_evidence_mapping) != plain_json(
            expected_evidence_mapping
        ):
            raise ValueError(
                "sequence log evidence differs from the bound Markov model"
            )

        expected_metadata = dict(plain_json(self.source_observation.metadata))
        expected_metadata.update(
            {
                "source_competence_evidence_id": self.evidence.artifact_id,
                "source_competence_feature_artifact_id": (
                    self.evidence.source_feature_artifact_id
                ),
                "source_competence_reliability_model_id": (
                    self.evidence.source_reliability_model_id
                ),
                "source_competence_markov_config_id": self.config.config_id,
                "source_competence_original_observation_id": (
                    self.source_observation.artifact_id
                ),
                "source_competence_composition": SOURCE_COMPETENCE_COMPOSITION,
                "source_competence_covariance_changed": False,
                "source_competence_association_probability_changed": False,
                "source_competence_uses_target_outcomes": False,
                "source_competence_uses_physical_innovation": False,
                "source_competence_uses_posterior_responsibility": False,
                "source_competence_uses_association_probability_as_label": False,
                "source_competence_claim_boundary": (
                    SOURCE_COMPETENCE_CLAIM_BOUNDARY
                ),
            }
        )
        expected_refined = replace(
            self.source_observation,
            prior_reliability=expected_deployed,
            metadata=expected_metadata,
        )
        if self.refined_observation.artifact_id != expected_refined.artifact_id:
            raise ValueError(
                "refined observation differs from the exact metadata-bound refinement"
            )
        evidence_mapping = provided_evidence_mapping
'''
    if old_update not in source:
        raise SystemExit("update semantic anchor changed")
    source = source.replace(old_update, new_update, 1)

    old_refine = '''    validate_source_competence_evidence(observation, evidence)
    cfg = SourceCompetenceMarkovConfigV1() if config is None else config
'''
    new_refine = '''    validate_source_competence_evidence(observation, evidence)
    if "source_competence_original_observation_id" in observation.metadata:
        raise ValueError(
            "source observation already carries source-competence refinement"
        )
    cfg = SourceCompetenceMarkovConfigV1() if config is None else config
'''
    if old_refine not in source:
        raise SystemExit("reapplication guard anchor changed")
    source = source.replace(old_refine, new_refine, 1)
    SOURCE.write_text(source, encoding="utf-8")


tests = TESTS.read_text(encoding="utf-8")
if "test_strict_loader_rejects_coercion_dependent_array_scalars" not in tests:
    old_import = '''from __future__ import annotations

from dataclasses import replace

import numpy as np
'''
    new_import = '''from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
'''
    if old_import not in tests:
        raise SystemExit("test import anchor changed")
    tests = tests.replace(old_import, new_import, 1)
    tests += '''

@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("time_values", "0.0"),
        ("time_values", True),
        ("log_competent_density", "1.0"),
        ("log_incompetent_density", False),
    ],
)
def test_strict_loader_rejects_coercion_dependent_array_scalars(
    tmp_path,
    field: str,
    bad_value: object,
) -> None:
    observation = _observation()
    record = _evidence(observation).to_record()
    values = record[field]
    assert isinstance(values, list)
    values[0] = bad_value
    path = tmp_path / f"bad-{field}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="literal JSON number"):
        load_source_competence_evidence(path)


def test_update_contract_recomputes_all_claim_bearing_semantics() -> None:
    observation = _observation()
    update = refine_observation_source_competence(
        observation,
        _evidence(observation),
    )

    with pytest.raises(ValueError, match="posterior competence probability"):
        replace(
            update,
            posterior_competence_probability=np.zeros(
                observation.observation_count,
                dtype=np.float64,
            ),
            update_id=None,
        )

    lower = 0.5 * update.deployed_prior_reliability
    lower_refined = replace(
        update.refined_observation,
        prior_reliability=lower,
    )
    with pytest.raises(ValueError, match="conservative composition"):
        replace(
            update,
            refined_observation=lower_refined,
            deployed_prior_reliability=lower,
            update_id=None,
        )

    altered_mean = np.array(
        update.refined_observation.mean_xyz_m,
        dtype=np.float64,
        copy=True,
    )
    altered_mean[0, 0] += 1e-6
    with pytest.raises(ValueError, match="exact metadata-bound refinement"):
        replace(
            update,
            refined_observation=replace(
                update.refined_observation,
                mean_xyz_m=altered_mean,
            ),
            update_id=None,
        )

    altered_evidence = dict(update.sequence_log_evidence)
    altered_evidence["track-10"] = float(altered_evidence["track-10"]) + 1.0
    with pytest.raises(ValueError, match="sequence log evidence"):
        replace(
            update,
            sequence_log_evidence=altered_evidence,
            update_id=None,
        )


def test_source_competence_refinement_cannot_be_reapplied() -> None:
    observation = _observation()
    update = refine_observation_source_competence(
        observation,
        _evidence(observation),
    )
    refined = update.refined_observation
    with pytest.raises(ValueError, match="already carries"):
        refine_observation_source_competence(
            refined,
            _evidence(refined),
        )
'''
    TESTS.write_text(tests, encoding="utf-8")
