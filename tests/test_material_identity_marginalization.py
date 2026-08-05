from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.material_identity_marginalization import (
    IDENTITY_LIKELIHOOD_SEMANTICS,
    PROB4D_MATERIAL_IDENTITY_CLAIM_BOUNDARY,
    PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA,
    PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_VERSION,
    PROB4D_MATERIAL_IDENTITY_MIXTURE_SCHEMA,
    PROB4D_MATERIAL_IDENTITY_MIXTURE_VERSION,
    PROB4D_MATERIAL_IDENTITY_NULL_SEMANTICS,
    PROB4D_MATERIAL_IDENTITY_WEIGHT_SEMANTICS,
    MaterialIdentityLikelihoodEvidenceV1,
    MaterialIdentityStatePosteriorV1,
    Prob4DMaterialIdentityMixtureV1,
    load_prob4d_material_identity_mixture,
    marginalize_material_identity_state,
    material_identity_candidate_lineage,
    validate_prob4d_material_identity_mixture,
)


def _content_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _candidate(
    *,
    target: dict[str, Any],
    source: dict[str, Any] | None,
    weight: float,
    association: str | None = None,
    score: float | None = None,
) -> dict[str, Any]:
    kind = "null" if source is None else "linked"
    identity = {
        "schema": PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA,
        "schema_version": PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_VERSION,
        "target_endpoint": target,
        "kind": kind,
        "source_endpoint": source,
        "association_result_id": association,
    }
    return {
        "candidate_id": _content_id(identity),
        "kind": kind,
        "source_endpoint": source,
        "association_result_id": association,
        "source_score": score,
        "calibrated_log_weight": weight,
        "metadata": {"portable": True},
    }


def _valid_record(*, linked: int = 2) -> dict[str, Any]:
    target = {"window_id": "w2", "track_id": 7}
    candidates = [_candidate(target=target, source=None, weight=np.log(0.4))]
    if linked >= 1:
        candidates.append(
            _candidate(
                target=target,
                source={"window_id": "w0", "track_id": 3},
                association="a" * 64,
                score=0.91,
                weight=np.log(0.35),
            )
        )
    if linked >= 2:
        candidates.append(
            _candidate(
                target=target,
                source={"window_id": "w1", "track_id": 5},
                association="b" * 64,
                score=0.82,
                weight=np.log(0.25),
            )
        )
    identity = {
        "schema": PROB4D_MATERIAL_IDENTITY_MIXTURE_SCHEMA,
        "schema_version": PROB4D_MATERIAL_IDENTITY_MIXTURE_VERSION,
        "target_endpoint": target,
        "window_order": ["w0", "w1", "w2"],
        "causal_frame_stop": 11,
        "association_rule_id": "c" * 64,
        "calibration_id": "d" * 64,
        "tracklet_producer_revision": "e" * 40,
        "association_revision": "f" * 40,
        "weight_semantics": PROB4D_MATERIAL_IDENTITY_WEIGHT_SEMANTICS,
        "null_hypothesis_semantics": PROB4D_MATERIAL_IDENTITY_NULL_SEMANTICS,
        "candidates": candidates,
        "metadata": {"split": "source-only", "nested": {"values": [1, 2]}},
        "claim_boundary": PROB4D_MATERIAL_IDENTITY_CLAIM_BOUNDARY,
    }
    return {**identity, "mixture_id": _content_id(identity)}


def _rewrite_mixture_id(record: dict[str, Any]) -> None:
    identity = {key: value for key, value in record.items() if key != "mixture_id"}
    record["mixture_id"] = _content_id(identity)


def _evidence(
    mixture: Prob4DMaterialIdentityMixtureV1,
    values: np.ndarray | None = None,
    *,
    power: float = 1.0,
) -> MaterialIdentityLikelihoodEvidenceV1:
    return MaterialIdentityLikelihoodEvidenceV1(
        mixture_id=mixture.mixture_id,
        common_state_domain_id="1" * 64,
        candidate_ids=mixture.candidate_ids,
        log_likelihoods=(
            np.zeros(len(mixture.candidate_ids)) if values is None else values
        ),
        calibration_id="2" * 64,
        likelihood_power=power,
        target_outcomes_used=False,
        metadata={"partition": "calibration-only"},
    )


@dataclass
class _Result:
    inference_admissible: bool
    reason: str
    state_coefficients: np.ndarray
    posterior_covariance: np.ndarray
    input_lineage: dict[str, Any]


def _results(
    mixture: Prob4DMaterialIdentityMixtureV1,
    evidence: MaterialIdentityLikelihoodEvidenceV1,
    *,
    means: tuple[tuple[float, ...], ...] | None = None,
    variances: tuple[float, ...] | None = None,
    inadmissible: int | None = None,
) -> dict[str, _Result]:
    count = len(mixture.candidate_ids)
    if means is None:
        means = tuple((float(index), -float(index)) for index in range(count))
    if variances is None:
        variances = tuple(0.1 + 0.1 * index for index in range(count))
    result: dict[str, _Result] = {}
    for index, candidate_id in enumerate(mixture.candidate_ids):
        state = np.asarray(means[index], dtype=np.float64)
        dimension = len(state)
        full = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
        full[:dimension, :dimension] = np.eye(dimension) * variances[index]
        full[-1, -1] = 0.5
        result[candidate_id] = _Result(
            inference_admissible=index != inadmissible,
            reason="ok" if index != inadmissible else "no-identifiable-query-state",
            state_coefficients=state,
            posterior_covariance=full,
            input_lineage=dict(
                material_identity_candidate_lineage(
                    mixture,
                    candidate_id=candidate_id,
                    common_state_domain_id=evidence.common_state_domain_id,
                    metadata={"case": index},
                )
            ),
        )
    return result


def test_loads_exact_prob4d_contract_and_owns_values(tmp_path: Path) -> None:
    record = _valid_record()
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    mixture = load_prob4d_material_identity_mixture(path)

    assert mixture.mixture_id == record["mixture_id"]
    assert mixture.candidate_ids == tuple(
        candidate["candidate_id"] for candidate in record["candidates"]
    )
    np.testing.assert_allclose(mixture.source_probabilities, [0.4, 0.35, 0.25])
    assert mixture.source_probabilities.flags.writeable is False
    assert mixture.metadata["nested"]["values"] == [1, 2]
    with pytest.raises(TypeError, match="immutable"):
        mixture.metadata["nested"]["values"].append(3)


def test_validate_rejects_top_level_contract_drift() -> None:
    record = _valid_record()
    record["unexpected"] = True
    with pytest.raises(ValueError, match="fields changed"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["schema"] = "other"
    with pytest.raises(ValueError, match="unsupported.*schema"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["claim_boundary"] += " changed"
    with pytest.raises(ValueError, match="claim boundary changed"):
        validate_prob4d_material_identity_mixture(record)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(window_order=[]), "window_order"),
        (lambda value: value.update(window_order=["w0", "w2", "w2"]), "unique"),
        (lambda value: value.update(window_order=["w0", "w1", "other"]), "last"),
        (lambda value: value.update(causal_frame_stop=0), "causal_frame_stop"),
        (lambda value: value.update(association_rule_id="bad"), "association_rule_id"),
        (lambda value: value.update(calibration_id="bad"), "calibration_id"),
        (
            lambda value: value.update(tracklet_producer_revision="main"),
            "tracklet_producer_revision",
        ),
        (
            lambda value: value.update(association_revision="main"),
            "association_revision",
        ),
        (lambda value: value.update(weight_semantics="other"), "weight semantics"),
        (
            lambda value: value.update(null_hypothesis_semantics="other"),
            "null-hypothesis",
        ),
        (lambda value: value.update(metadata=[]), "metadata must be"),
    ],
)
def test_validate_rejects_invalid_mixture_fields(mutate: Any, message: str) -> None:
    record = _valid_record()
    mutate(record)
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match=message):
        validate_prob4d_material_identity_mixture(record)


def test_validate_rejects_candidate_contract_and_identity_drift() -> None:
    record = _valid_record()
    record["candidates"][1]["unexpected"] = 1
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="fields changed"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["candidates"][1]["candidate_id"] = "0" * 64
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="candidate ID mismatch"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["mixture_id"] = "0" * 64
    with pytest.raises(ValueError, match="mixture ID mismatch"):
        validate_prob4d_material_identity_mixture(record)


def test_validate_rejects_noncanonical_and_invalid_candidates() -> None:
    record = _valid_record()
    record["candidates"] = [record["candidates"][1], record["candidates"][0]]
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="canonical"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["candidates"][0]["association_result_id"] = "a" * 64
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="null hypothesis"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["candidates"][1]["kind"] = "null"
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="kind does not match"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["candidates"][1]["source_score"] = 1.1
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="source_score"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record()
    record["candidates"][1]["metadata"] = []
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="metadata must be"):
        validate_prob4d_material_identity_mixture(record)


def test_validate_rejects_duplicate_or_future_linked_endpoints() -> None:
    record = _valid_record()
    record["candidates"][2]["source_endpoint"] = deepcopy(
        record["candidates"][1]["source_endpoint"]
    )
    record["candidates"][2]["association_result_id"] = "b" * 64
    target = record["target_endpoint"]
    record["candidates"][2]["candidate_id"] = _content_id(
        {
            "schema": PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA,
            "schema_version": 1,
            "target_endpoint": target,
            "kind": "linked",
            "source_endpoint": record["candidates"][2]["source_endpoint"],
            "association_result_id": "b" * 64,
        }
    )
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="unique"):
        validate_prob4d_material_identity_mixture(record)

    record = _valid_record(linked=1)
    record["candidates"][1]["source_endpoint"]["window_id"] = "w2"
    record["candidates"][1]["candidate_id"] = _content_id(
        {
            "schema": PROB4D_MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA,
            "schema_version": 1,
            "target_endpoint": record["target_endpoint"],
            "kind": "linked",
            "source_endpoint": record["candidates"][1]["source_endpoint"],
            "association_result_id": "a" * 64,
        }
    )
    _rewrite_mixture_id(record)
    with pytest.raises(ValueError, match="precede"):
        validate_prob4d_material_identity_mixture(record)


def test_strict_loader_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_prob4d_material_identity_mixture(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_prob4d_material_identity_mixture(nonfinite)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read"):
        load_prob4d_material_identity_mixture(malformed)


def test_candidate_lineage_binds_common_domain_and_rejects_conflicts() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    candidate_id = mixture.candidate_ids[1]

    lineage = material_identity_candidate_lineage(
        mixture,
        candidate_id=candidate_id,
        common_state_domain_id="1" * 64,
        metadata={"physical_linearization_artifact_id": "3" * 64},
    )

    assert lineage["prob4d_material_identity_mixture_id"] == mixture.mixture_id
    assert lineage["prob4d_material_identity_candidate_id"] == candidate_id
    assert lineage["material_identity_common_state_domain_id"] == "1" * 64
    assert lineage["prob4d_material_identity_causal_frame_stop"] == 11

    with pytest.raises(ValueError, match="not present"):
        material_identity_candidate_lineage(
            mixture,
            candidate_id="9" * 64,
            common_state_domain_id="1" * 64,
        )
    with pytest.raises(ValueError, match="reserved"):
        material_identity_candidate_lineage(
            mixture,
            candidate_id=candidate_id,
            common_state_domain_id="1" * 64,
            metadata={"prob4d_material_identity_mixture_id": "9" * 64},
        )


def test_likelihood_evidence_is_content_addressed_and_read_only() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    evidence = _evidence(mixture, np.array([-0.5, -1.5]), power=0.75)

    assert len(evidence.evidence_id or "") == 64
    assert evidence.log_likelihoods.flags.writeable is False
    assert evidence.semantics == IDENTITY_LIKELIHOOD_SEMANTICS
    assert evidence.identity_record()["target_outcomes_used"] is False

    with pytest.raises(ValueError, match="evidence ID mismatch"):
        MaterialIdentityLikelihoodEvidenceV1(
            mixture_id=mixture.mixture_id,
            common_state_domain_id="1" * 64,
            candidate_ids=mixture.candidate_ids,
            log_likelihoods=np.zeros(2),
            calibration_id="2" * 64,
            evidence_id="0" * 64,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_ids": ("a" * 64, "a" * 64)}, "unique"),
        ({"log_likelihoods": np.array([0.0])}, "match candidate_ids"),
        ({"log_likelihoods": np.array([np.nan, 0.0])}, "NaN"),
        ({"log_likelihoods": np.array([np.inf, 0.0])}, "positive infinity"),
        ({"likelihood_power": -1.0}, "likelihood_power"),
        ({"likelihood_power": True}, "real number"),
        ({"target_outcomes_used": True}, "may not use target outcomes"),
        ({"target_outcomes_used": 0}, "must be a boolean"),
        ({"semantics": "other"}, "unsupported"),
    ],
)
def test_likelihood_evidence_rejects_invalid_inputs(
    kwargs: dict[str, Any], message: str
) -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    values: dict[str, Any] = {
        "mixture_id": mixture.mixture_id,
        "common_state_domain_id": "1" * 64,
        "candidate_ids": mixture.candidate_ids,
        "log_likelihoods": np.zeros(2),
        "calibration_id": "2" * 64,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        MaterialIdentityLikelihoodEvidenceV1(**values)


def test_null_only_mixture_reproduces_reference_exactly() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=0))
    evidence = _evidence(mixture)
    results = _results(
        mixture,
        evidence,
        means=((0.25, -0.5),),
        variances=(0.2,),
    )

    posterior = marginalize_material_identity_state(mixture, evidence, results)

    assert posterior.identity_marginalization_admissible is True
    assert posterior.deployed_reference_only is True
    assert posterior.reason == "null-only-mixture"
    assert posterior.posterior_probabilities.tolist() == [1.0]
    np.testing.assert_array_equal(
        posterior.state_mean,
        results[mixture.candidate_ids[0]].state_coefficients,
    )
    np.testing.assert_array_equal(posterior.state_covariance, np.eye(2) * 0.2)
    np.testing.assert_array_equal(
        posterior.between_identity_covariance,
        np.zeros((2, 2)),
    )


def test_marginalization_adds_between_identity_covariance() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    evidence = _evidence(mixture, np.zeros(2))
    results = _results(
        mixture,
        evidence,
        means=((0.0, 0.0), (2.0, 0.0)),
        variances=(1.0, 3.0),
    )

    posterior = marginalize_material_identity_state(mixture, evidence, results)

    probabilities = np.array([0.4, 0.35])
    probabilities /= probabilities.sum()
    expected_mean = np.array([2.0 * probabilities[1], 0.0])
    expected_within = np.eye(2) * (probabilities[0] + 3.0 * probabilities[1])
    centered = np.array([[0.0, 0.0], [2.0, 0.0]]) - expected_mean
    expected_between = np.einsum(
        "i,ij,ik->jk", probabilities, centered, centered
    )
    np.testing.assert_allclose(posterior.posterior_probabilities, probabilities)
    np.testing.assert_allclose(posterior.state_mean, expected_mean)
    np.testing.assert_allclose(posterior.within_identity_covariance, expected_within)
    np.testing.assert_allclose(posterior.between_identity_covariance, expected_between)
    np.testing.assert_allclose(
        posterior.state_covariance,
        expected_within + expected_between,
    )
    assert posterior.deployed_reference_only is False
    assert posterior.reason == "identity-marginalized"
    assert posterior.posterior_id is not None


def test_likelihood_power_zero_uses_source_prior() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    evidence = _evidence(mixture, np.array([-100.0, 100.0]), power=0.0)
    posterior = marginalize_material_identity_state(
        mixture,
        evidence,
        _results(mixture, evidence),
    )
    expected = np.array([0.4, 0.35])
    expected /= expected.sum()
    np.testing.assert_allclose(posterior.posterior_probabilities, expected)


def test_inadmissible_candidate_returns_exact_null_state() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=2))
    evidence = _evidence(mixture, np.array([-2.0, 5.0, 1.0]))
    results = _results(
        mixture,
        evidence,
        means=((0.25, -0.5), (20.0, 20.0), (-3.0, 1.0)),
        variances=(0.2, 5.0, 2.0),
        inadmissible=1,
    )

    posterior = marginalize_material_identity_state(mixture, evidence, results)

    assert posterior.identity_marginalization_admissible is False
    assert posterior.deployed_reference_only is True
    assert posterior.reason.startswith("candidate-inference-inadmissible:")
    assert posterior.posterior_probabilities.tolist() == [1.0, 0.0, 0.0]
    np.testing.assert_array_equal(posterior.state_mean, [0.25, -0.5])
    np.testing.assert_array_equal(posterior.state_covariance, np.eye(2) * 0.2)


def test_impossible_or_null_only_posterior_uses_exact_reference() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    impossible = _evidence(mixture, np.array([-np.inf, -np.inf]))
    result = marginalize_material_identity_state(
        mixture,
        impossible,
        _results(mixture, impossible),
    )
    assert result.identity_marginalization_admissible is False
    assert result.reason == "all-candidate-likelihoods-impossible"

    null_only = _evidence(mixture, np.array([0.0, -np.inf]))
    result = marginalize_material_identity_state(
        mixture,
        null_only,
        _results(mixture, null_only),
    )
    assert result.identity_marginalization_admissible is True
    assert result.reason == "posterior-null-reference"
    assert result.posterior_probabilities.tolist() == [1.0, 0.0]


def test_marginalization_rejects_contract_lineage_and_domain_mismatch() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    evidence = _evidence(mixture)
    results = _results(mixture, evidence)

    with pytest.raises(ValueError, match="candidate result IDs changed"):
        marginalize_material_identity_state(
            mixture,
            evidence,
            {mixture.candidate_ids[0]: results[mixture.candidate_ids[0]]},
        )

    bad = deepcopy(results)
    bad[mixture.candidate_ids[1]].input_lineage[
        "material_identity_common_state_domain_id"
    ] = "9" * 64
    with pytest.raises(ValueError, match="does not bind"):
        marginalize_material_identity_state(mixture, evidence, bad)

    other_record = _valid_record(linked=1)
    other_record["metadata"]["other"] = True
    _rewrite_mixture_id(other_record)
    other = validate_prob4d_material_identity_mixture(other_record)
    other_evidence = MaterialIdentityLikelihoodEvidenceV1(
        mixture_id=other.mixture_id,
        common_state_domain_id="1" * 64,
        candidate_ids=other.candidate_ids,
        log_likelihoods=np.zeros(2),
        calibration_id="2" * 64,
    )
    with pytest.raises(ValueError, match="different mixture"):
        marginalize_material_identity_state(mixture, other_evidence, results)


def test_marginalization_rejects_malformed_candidate_state() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    evidence = _evidence(mixture)
    results = _results(mixture, evidence)

    results[mixture.candidate_ids[1]].state_coefficients = np.array([1.0])
    with pytest.raises(ValueError, match="share one domain"):
        marginalize_material_identity_state(mixture, evidence, results)

    results = _results(mixture, evidence)
    results[mixture.candidate_ids[1]].posterior_covariance[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        marginalize_material_identity_state(mixture, evidence, results)

    results = _results(mixture, evidence)
    results[mixture.candidate_ids[1]].state_coefficients[0] = np.nan
    with pytest.raises(ValueError, match="finite non-empty"):
        marginalize_material_identity_state(mixture, evidence, results)


def test_posterior_rejects_inconsistent_direct_construction() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    evidence = _evidence(mixture)
    valid = marginalize_material_identity_state(
        mixture,
        evidence,
        _results(mixture, evidence),
    )
    values = {
        "mixture_id": valid.mixture_id,
        "likelihood_evidence_id": valid.likelihood_evidence_id,
        "common_state_domain_id": valid.common_state_domain_id,
        "candidate_ids": valid.candidate_ids,
        "candidate_inference_admissible": valid.candidate_inference_admissible,
        "identity_marginalization_admissible": (
            valid.identity_marginalization_admissible
        ),
        "deployed_reference_only": valid.deployed_reference_only,
        "reason": valid.reason,
        "posterior_probabilities": valid.posterior_probabilities,
        "state_mean": valid.state_mean,
        "state_covariance": valid.state_covariance,
        "within_identity_covariance": valid.within_identity_covariance,
        "between_identity_covariance": valid.between_identity_covariance,
        "identity_entropy_nats": valid.identity_entropy_nats,
        "effective_hypothesis_count": valid.effective_hypothesis_count,
    }

    with pytest.raises(ValueError, match="must equal"):
        MaterialIdentityStatePosteriorV1(
            **{**values, "state_covariance": valid.state_covariance + np.eye(2)}
        )
    with pytest.raises(ValueError, match="posterior ID mismatch"):
        MaterialIdentityStatePosteriorV1(
            **values,
            posterior_id="0" * 64,
        )


def test_public_type_checks_fail_cleanly() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    evidence = _evidence(mixture)
    with pytest.raises(TypeError, match="mixture"):
        material_identity_candidate_lineage(  # type: ignore[arg-type]
            object(),
            candidate_id="0" * 64,
            common_state_domain_id="1" * 64,
        )
    with pytest.raises(TypeError, match="mixture"):
        marginalize_material_identity_state(  # type: ignore[arg-type]
            object(), evidence, {}
        )
    with pytest.raises(TypeError, match="evidence"):
        marginalize_material_identity_state(  # type: ignore[arg-type]
            mixture, object(), {}
        )


def test_internal_numerical_guards_cover_unreachable_adversarial_shapes() -> None:
    import bayesian_phystwin.material_identity_marginalization as module

    with pytest.raises(ValueError, match="non-empty vector"):
        module._logsumexp(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="positive infinity"):
        module._logsumexp(np.array([np.inf]))
    with pytest.raises(ValueError, match="square"):
        module._validated_state_covariance(np.zeros((1, 2)), name="covariance")
    with pytest.raises(ValueError, match="finite"):
        module._validated_state_covariance(
            np.array([[np.nan]]),
            name="covariance",
        )
    with pytest.raises(ValueError, match="symmetric"):
        module._validated_state_covariance(
            np.array([[1.0, 1.0], [0.0, 1.0]]),
            name="covariance",
        )


def test_loader_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be"):
        load_prob4d_material_identity_mixture(path)


def test_direct_contract_construction_rejects_invalid_container_types() -> None:
    import bayesian_phystwin.material_identity_marginalization as module

    with pytest.raises(ValueError, match="JSON object"):
        module.Prob4DLocalTrackEndpointV1.from_mapping([], name="endpoint")
    with pytest.raises(ValueError, match="unsupported"):
        module.Prob4DMaterialIdentityCandidateV1(
            candidate_id="0" * 64,
            kind="other",  # type: ignore[arg-type]
            source_endpoint=None,
            association_result_id=None,
            source_score=None,
            calibrated_log_weight=0.0,
        )

    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    values = {
        "mixture_id": mixture.mixture_id,
        "target_endpoint": mixture.target_endpoint,
        "window_order": mixture.window_order,
        "causal_frame_stop": mixture.causal_frame_stop,
        "association_rule_id": mixture.association_rule_id,
        "calibration_id": mixture.calibration_id,
        "tracklet_producer_revision": mixture.tracklet_producer_revision,
        "association_revision": mixture.association_revision,
        "candidates": mixture.candidates,
    }
    with pytest.raises(ValueError, match="window_order"):
        module.Prob4DMaterialIdentityMixtureV1(
            **{**values, "window_order": []},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="candidates"):
        module.Prob4DMaterialIdentityMixtureV1(
            **{**values, "candidates": []},  # type: ignore[arg-type]
        )


def test_malformed_result_protocol_and_covariance_fail_closed() -> None:
    mixture = validate_prob4d_material_identity_mixture(_valid_record(linked=1))
    evidence = _evidence(mixture)
    results = _results(mixture, evidence)

    class Empty:
        pass

    malformed: dict[str, Any] = dict(results)
    malformed[mixture.candidate_ids[1]] = Empty()
    with pytest.raises(TypeError, match="GaugeAwareStateResult"):
        marginalize_material_identity_state(mixture, evidence, malformed)

    results = _results(mixture, evidence)
    results[mixture.candidate_ids[1]].posterior_covariance = np.zeros((1, 2))
    with pytest.raises(ValueError, match="square"):
        marginalize_material_identity_state(mixture, evidence, results)

    results = _results(mixture, evidence)
    results[mixture.candidate_ids[1]].posterior_covariance = np.zeros((1, 1))
    with pytest.raises(ValueError, match="smaller"):
        marginalize_material_identity_state(mixture, evidence, results)

    results = _results(mixture, evidence)
    results[mixture.candidate_ids[1]].input_lineage = []  # type: ignore[assignment]
    with pytest.raises(ValueError, match="must be a mapping"):
        marginalize_material_identity_state(mixture, evidence, results)
