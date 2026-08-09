from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.observation_belief import (
    ObservationBeliefV1,
    array_sha256,
    load_observation_belief,
    save_observation_belief,
)
from bayesian_phystwin.observation_contract_bundle import (
    OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256,
    invalid_observation_contract_vector,
    observation_contract_array_sha256,
    observation_contract_artifact_id,
    observation_contract_bundle_manifest,
    observation_contract_invalid_cases,
    observation_contract_schema,
    observation_contract_vector,
)


def _construct(descriptor, arrays) -> ObservationBeliefV1:
    values = dict(descriptor)
    values.pop("schema_name", None)
    values.pop("schema_version", None)
    values.pop("artifact_id", None)
    return ObservationBeliefV1(**values, **arrays)


def _write(
    path: Path,
    descriptor,
    arrays,
    *,
    artifact_id: str | None = None,
) -> None:
    payload = dict(descriptor)
    payload["artifact_id"] = (
        observation_contract_artifact_id(payload, arrays)
        if artifact_id is None
        else artifact_id
    )
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        **arrays,
    )


def test_bundle_is_content_locked_and_normative() -> None:
    manifest = observation_contract_bundle_manifest()
    schema = observation_contract_schema()

    assert manifest["bundle_sha256"] == OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256
    assert manifest["canonical_repository"] == "FlorianPfaff/Prob4D"
    assert schema["contract_id"] == "phys4d.observation_belief.v1"
    assert schema["descriptor"]["closed"] is True
    assert schema["arrays"]["closed"] is True


@pytest.mark.parametrize("vector_name", ("minimal", "zero_rank"))
def test_valid_vectors_match_reference_hash_and_round_trip(
    vector_name: str,
    tmp_path: Path,
) -> None:
    vector = observation_contract_vector(vector_name)
    belief = _construct(vector.descriptor, vector.arrays)

    assert belief.artifact_id == vector.expected_artifact_id
    for _name, values in belief._arrays().items():
        assert array_sha256(values) == observation_contract_array_sha256(values)

    path = tmp_path / f"{vector_name}.npz"
    save_observation_belief(path, belief)
    restored = load_observation_belief(path)
    assert restored.artifact_id == vector.expected_artifact_id
    assert restored.mean_xyz_m.flags.writeable is False


@pytest.mark.parametrize(
    "case_id",
    [case["id"] for case in observation_contract_invalid_cases()],
)
def test_invalid_corpus_is_rejected(case_id: str, tmp_path: Path) -> None:
    invalid = invalid_observation_contract_vector(case_id)

    if invalid.mode == "semantic":
        with pytest.raises(ValueError):
            _construct(invalid.descriptor, invalid.arrays)
        return

    path = tmp_path / f"{case_id}.npz"
    artifact_id = (
        invalid.original_artifact_id
        if invalid.mode == "digest_mismatch"
        else observation_contract_artifact_id(
            invalid.descriptor,
            invalid.arrays,
        )
    )
    _write(
        path,
        invalid.descriptor,
        invalid.arrays,
        artifact_id=artifact_id,
    )
    with pytest.raises(ValueError):
        load_observation_belief(path)


def test_bundle_report_and_unknown_names_fail_closed(capsys) -> None:
    from bayesian_phystwin.observation_contract_bundle import main

    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bundle_sha256"] == OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256
    with pytest.raises(KeyError):
        observation_contract_vector("unknown")
    with pytest.raises(KeyError):
        invalid_observation_contract_vector("unknown")


def test_loader_rejects_non_scalar_descriptor(tmp_path: Path) -> None:
    vector = observation_contract_vector("minimal")
    payload = dict(vector.descriptor)
    payload["artifact_id"] = vector.expected_artifact_id
    path = tmp_path / "descriptor-array.npz"
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            [json.dumps(payload, sort_keys=True, separators=(",", ":"))]
        ),
        **vector.arrays,
    )
    with pytest.raises(ValueError, match="scalar"):
        load_observation_belief(path)
