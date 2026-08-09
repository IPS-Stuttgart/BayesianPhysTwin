from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.observation_contract_bundle as contract_module
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
    observation_contract_canonical_json_sha256,
    observation_contract_invalid_cases,
    observation_contract_schema,
    observation_contract_vector,
)


def _construct(descriptor: Any, arrays: Any) -> ObservationBeliefV1:
    values = dict(descriptor)
    values.pop("schema_name", None)
    values.pop("schema_version", None)
    values.pop("artifact_id", None)
    return ObservationBeliefV1(**values, **arrays)


def _write(
    path: Path,
    descriptor: Any,
    arrays: Any,
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


def _raw_descriptor_path(
    tmp_path: Path,
    descriptor_member: Any,
    *,
    name: str,
) -> Path:
    vector = observation_contract_vector("minimal")
    path = tmp_path / f"{name}.npz"
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(descriptor_member),
        **vector.arrays,
    )
    return path


class _FakeMember:
    def __init__(
        self,
        content: bytes = b"",
        *,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error

    def read_bytes(self) -> bytes:
        if self.error is not None:
            raise self.error
        return self.content

    def read_text(self, *, encoding: str) -> str:
        if self.error is not None:
            raise self.error
        return self.content.decode(encoding)


def _synthetic_manifest(
    files: dict[str, str],
    *,
    bundle_name: str = contract_module.OBSERVATION_BELIEF_CONTRACT_BUNDLE,
    bundle_version: int = 1,
) -> dict[str, Any]:
    descriptor = {
        "bundle_name": bundle_name,
        "bundle_version": bundle_version,
        "canonical_repository": "IPS-Stuttgart/Prob4D",
        "files": files,
    }
    bundle_sha256 = hashlib.sha256(
        contract_module._canonical_json(descriptor)
    ).hexdigest()
    return {**descriptor, "bundle_sha256": bundle_sha256}


def _patch_manifest(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
    members: dict[str, _FakeMember] | None = None,
) -> None:
    monkeypatch.setattr(
        contract_module,
        "_read_json",
        lambda relative_path: copy.deepcopy(payload),
    )
    if members is not None:
        monkeypatch.setattr(
            contract_module,
            "_bundle_member",
            lambda relative_path: members[relative_path],
        )


def _patch_vector_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
) -> None:
    monkeypatch.setattr(
        contract_module,
        "observation_contract_bundle_manifest",
        lambda: {},
    )
    monkeypatch.setattr(
        contract_module,
        "_read_json",
        lambda relative_path: copy.deepcopy(payload),
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


def test_bundle_report_hash_helpers_and_unknown_names_fail_closed(capsys) -> None:
    from bayesian_phystwin.observation_contract_bundle import main

    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bundle_sha256"] == OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256
    assert observation_contract_canonical_json_sha256({"b": 2, "a": 1}) == (
        hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    )

    vector = observation_contract_vector("minimal")
    descriptor = dict(vector.descriptor)
    descriptor["artifact_id"] = "0" * 64
    assert observation_contract_artifact_id(descriptor, vector.arrays) == (
        vector.expected_artifact_id
    )

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


def test_loader_accepts_utf8_bytes_descriptor(tmp_path: Path) -> None:
    vector = observation_contract_vector("minimal")
    payload = dict(vector.descriptor)
    payload["artifact_id"] = vector.expected_artifact_id
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path = _raw_descriptor_path(tmp_path, encoded, name="bytes-descriptor")

    assert load_observation_belief(path).artifact_id == vector.expected_artifact_id


@pytest.mark.parametrize(
    ("member", "message"),
    [
        (b"\xff", "valid UTF-8"),
        (1, "contain a string"),
        ("{", "valid JSON"),
        ("[]", "JSON object"),
    ],
)
def test_loader_rejects_malformed_descriptor_payloads(
    tmp_path: Path,
    member: Any,
    message: str,
) -> None:
    path = _raw_descriptor_path(tmp_path, member, name=message.replace(" ", "-"))
    with pytest.raises(ValueError, match=message):
        load_observation_belief(path)


def test_loader_rejects_changed_descriptor_field_set(tmp_path: Path) -> None:
    vector = observation_contract_vector("minimal")
    payload = dict(vector.descriptor)
    payload["artifact_id"] = vector.expected_artifact_id
    payload["unexpected"] = True
    path = _raw_descriptor_path(
        tmp_path,
        json.dumps(payload),
        name="changed-fields",
    )
    with pytest.raises(ValueError, match="fields changed"):
        load_observation_belief(path)


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("missing"), TypeError("wrong type"), ValueError("bad JSON")],
)
def test_bundle_json_reader_wraps_member_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(
        contract_module,
        "_bundle_member",
        lambda relative_path: _FakeMember(error=error),
    )
    with pytest.raises(ValueError, match="bundle member"):
        contract_module._read_json("broken.json")


def test_bundle_json_reader_wraps_json_parse_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contract_module,
        "_bundle_member",
        lambda relative_path: _FakeMember(b"{"),
    )
    with pytest.raises(ValueError, match="bundle member"):
        contract_module._read_json("broken.json")


@pytest.mark.parametrize("digest", ("short", "A" * 64))
def test_bundle_digest_validator_rejects_noncanonical_values(digest: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        contract_module._validate_sha256(digest, name="digest")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not-object", "JSON object"),
        ("fields", "fields changed"),
        ("name", "bundle name"),
        ("version", "bundle version"),
        ("files-type", "no files"),
        ("files-empty", "no files"),
    ],
)
def test_manifest_rejects_invalid_headers(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    payload: Any = copy.deepcopy(contract_module._read_json("manifest.json"))
    if case == "not-object":
        payload = []
    elif case == "fields":
        payload["unexpected"] = True
    elif case == "name":
        payload["bundle_name"] = "other"
    elif case == "version":
        payload["bundle_version"] = 2
    elif case == "files-type":
        payload["files"] = []
    else:
        payload["files"] = {}
    _patch_manifest(monkeypatch, payload)

    with pytest.raises(ValueError, match=message):
        observation_contract_bundle_manifest()


@pytest.mark.parametrize(
    "unsafe_path",
    ("", "/absolute", "windows\\path", ".", "a/../b", "a//b"),
)
def test_manifest_rejects_each_unsafe_path_form(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    payload = _synthetic_manifest({unsafe_path: "0" * 64})
    _patch_manifest(monkeypatch, payload)

    with pytest.raises(ValueError, match="unsafe path"):
        observation_contract_bundle_manifest()


def test_manifest_rejects_invalid_file_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _synthetic_manifest({"schema.json": "invalid"})
    _patch_manifest(monkeypatch, payload)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        observation_contract_bundle_manifest()


def test_manifest_rejects_missing_member(monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"schema"
    digest = hashlib.sha256(content).hexdigest()
    payload = _synthetic_manifest({"schema.json": digest})
    _patch_manifest(
        monkeypatch,
        payload,
        {"schema.json": _FakeMember(error=FileNotFoundError("missing"))},
    )

    with pytest.raises(ValueError, match="is missing"):
        observation_contract_bundle_manifest()


def test_manifest_rejects_member_content_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _synthetic_manifest({"schema.json": "0" * 64})
    _patch_manifest(
        monkeypatch,
        payload,
        {"schema.json": _FakeMember(b"different")},
    )

    with pytest.raises(ValueError, match="content lock"):
        observation_contract_bundle_manifest()


def test_manifest_rejects_aggregate_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"schema"
    digest = hashlib.sha256(content).hexdigest()
    payload = _synthetic_manifest({"schema.json": digest})
    payload["bundle_sha256"] = "0" * 64
    _patch_manifest(
        monkeypatch,
        payload,
        {"schema.json": _FakeMember(content)},
    )

    with pytest.raises(ValueError, match="does not match its manifest"):
        observation_contract_bundle_manifest()


def test_manifest_rejects_code_lock_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"schema"
    digest = hashlib.sha256(content).hexdigest()
    payload = _synthetic_manifest({"schema.json": digest})
    _patch_manifest(
        monkeypatch,
        payload,
        {"schema.json": _FakeMember(content)},
    )
    monkeypatch.setattr(
        contract_module,
        "OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256",
        "0" * 64,
    )

    with pytest.raises(ValueError, match="differs from the code lock"):
        observation_contract_bundle_manifest()


def test_synthetic_manifest_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"schema"
    digest = hashlib.sha256(content).hexdigest()
    payload = _synthetic_manifest({"schema.json": digest})
    _patch_manifest(
        monkeypatch,
        payload,
        {"schema.json": _FakeMember(content)},
    )
    monkeypatch.setattr(
        contract_module,
        "OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256",
        payload["bundle_sha256"],
    )

    assert observation_contract_bundle_manifest()["files"] == {
        "schema.json": digest
    }


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not-object", "JSON object"),
        ("contract", "identity changed"),
        ("schema", "identity changed"),
        ("version", "identity changed"),
    ],
)
def test_schema_rejects_identity_corruption(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    payload: Any = copy.deepcopy(contract_module._read_json("schema.json"))
    if case == "not-object":
        payload = []
    elif case == "contract":
        payload["contract_id"] = "other"
    elif case == "schema":
        payload["schema_name"] = "other"
    else:
        payload["schema_version"] = 2
    monkeypatch.setattr(
        contract_module,
        "observation_contract_bundle_manifest",
        lambda: {},
    )
    monkeypatch.setattr(contract_module, "_read_json", lambda path: payload)

    with pytest.raises(ValueError, match=message):
        observation_contract_schema()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not-object", "fields changed"),
        ("fields", "fields changed"),
        ("version", "vector version"),
        ("descriptor", "payload is invalid"),
        ("records", "payload is invalid"),
        ("record-type", "array record"),
        ("record-fields", "array record"),
        ("shape", "has shape"),
        ("digest-format", "lowercase SHA-256"),
        ("digest-value", "invalid artifact ID"),
    ],
)
def test_vector_rejects_corrupt_payloads(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    payload: Any = copy.deepcopy(
        contract_module._read_json("vectors/minimal.json")
    )
    first_name = next(iter(payload["arrays"]))
    if case == "not-object":
        payload = []
    elif case == "fields":
        payload["unexpected"] = True
    elif case == "version":
        payload["vector_version"] = 2
    elif case == "descriptor":
        payload["descriptor"] = []
    elif case == "records":
        payload["arrays"] = []
    elif case == "record-type":
        payload["arrays"][first_name] = []
    elif case == "record-fields":
        del payload["arrays"][first_name]["values"]
    elif case == "shape":
        payload["arrays"][first_name]["shape"] = [999]
    elif case == "digest-format":
        payload["expected_artifact_id"] = "invalid"
    else:
        payload["expected_artifact_id"] = "0" * 64
    _patch_vector_payload(monkeypatch, payload)

    with pytest.raises(ValueError, match=message):
        observation_contract_vector("minimal")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("not-object", "fields changed"),
        ("fields", "fields changed"),
        ("version", "invalid-case version"),
        ("base", "unavailable base vector"),
        ("cases-type", "no invalid cases"),
        ("cases-empty", "no invalid cases"),
        ("case-type", "invalid case changed"),
        ("case-fields", "invalid case changed"),
        ("empty-id", "unique and nonempty"),
        ("duplicate-id", "unique and nonempty"),
        ("mutations-type", "no mutations"),
        ("mutations-empty", "no mutations"),
    ],
)
def test_invalid_case_registry_rejects_corruption(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    payload: Any = copy.deepcopy(contract_module._read_json("invalid_cases.json"))
    if case == "not-object":
        payload = []
    elif case == "fields":
        payload["unexpected"] = True
    elif case == "version":
        payload["invalid_case_version"] = 2
    elif case == "base":
        payload["base_vector"] = "unknown"
    elif case == "cases-type":
        payload["cases"] = {}
    elif case == "cases-empty":
        payload["cases"] = []
    elif case == "case-type":
        payload["cases"][0] = []
    elif case == "case-fields":
        payload["cases"][0]["unexpected"] = True
    elif case == "empty-id":
        payload["cases"][0]["id"] = ""
    elif case == "duplicate-id":
        payload["cases"][1]["id"] = payload["cases"][0]["id"]
    elif case == "mutations-type":
        payload["cases"][0]["mutations"] = {}
    else:
        payload["cases"][0]["mutations"] = []
    _patch_vector_payload(monkeypatch, payload)

    with pytest.raises(ValueError, match=message):
        observation_contract_invalid_cases()


def _patch_custom_invalid_case(
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
) -> None:
    base = observation_contract_vector("minimal")
    case = {
        "id": "custom",
        "mode": "loader",
        "mutations": [mutation],
    }
    monkeypatch.setattr(
        contract_module,
        "observation_contract_invalid_cases",
        lambda: (case,),
    )
    monkeypatch.setattr(
        contract_module,
        "observation_contract_vector",
        lambda name: base,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"target": "descriptor", "op": "set"}, "no path"),
        (
            {
                "target": "descriptor",
                "op": "replace",
                "path": ["case_id"],
            },
            "unsupported descriptor mutation",
        ),
        (
            {
                "target": "array",
                "op": "set",
                "name": "unknown",
                "index": [0],
                "value": 0,
            },
            "unavailable member",
        ),
        (
            {
                "target": "array",
                "op": "replace",
                "name": "frame_ids",
            },
            "unsupported array mutation",
        ),
        (
            {
                "target": "arrays",
                "op": "add",
                "name": "",
                "value": [0],
                "dtype": "int64",
                "shape": [1],
            },
            "added array name is invalid",
        ),
        (
            {
                "target": "arrays",
                "op": "add",
                "name": "frame_ids",
                "value": [0],
                "dtype": "int64",
                "shape": [1],
            },
            "added array name is invalid",
        ),
        (
            {
                "target": "arrays",
                "op": "add",
                "name": "new_array",
                "value": [0],
                "dtype": "int64",
                "shape": [2],
            },
            "added array shape is invalid",
        ),
        ({"target": "other", "op": "set"}, "unsupported.*target"),
    ],
)
def test_invalid_vector_interpreter_rejects_bad_mutations(
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
    message: str,
) -> None:
    _patch_custom_invalid_case(monkeypatch, mutation)
    with pytest.raises(ValueError, match=message):
        invalid_observation_contract_vector("custom")


def test_invalid_vector_interpreter_adds_well_formed_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutation = {
        "target": "arrays",
        "op": "add",
        "name": "new_array",
        "value": [1, 2],
        "dtype": "int64",
        "shape": [2],
    }
    _patch_custom_invalid_case(monkeypatch, mutation)

    invalid = invalid_observation_contract_vector("custom")
    np.testing.assert_array_equal(invalid.arrays["new_array"], [1, 2])
    assert invalid.arrays["new_array"].flags.writeable is False


def test_module_entrypoint_reports_manifest(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        runpy.run_module(
            "bayesian_phystwin.observation_contract_bundle",
            run_name="__main__",
        )
    assert raised.value.code == 0
    assert OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256 in capsys.readouterr().out
