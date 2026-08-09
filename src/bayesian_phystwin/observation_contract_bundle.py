"""Normative conformance bundle for ``phys4d.observation_belief`` version 1.

The JSON bundle is intentionally data-only. Prob4D, Bayesian-PhysTwin, and
Causal4D carry the same byte-for-byte corpus while retaining independent
provider/consumer implementations. This module verifies the content lock and
provides neutral hashing and conformance helpers without importing another
repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

OBSERVATION_BELIEF_CONTRACT_BUNDLE = "phys4d.observation_belief.v1"
OBSERVATION_BELIEF_CONTRACT_BUNDLE_VERSION = 1
OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256 = (
    "a62c693a14c227daa1f4c8db850e691a1d0081df0c853cf0174c33d0b8504ce9"
)

_BUNDLE_DIRECTORY = ("contract_data", "observation_belief_v1")
_VECTOR_NAMES = frozenset({"minimal", "zero_rank"})


@dataclass(frozen=True)
class ObservationContractVector:
    """One validated descriptor/array conformance vector."""

    name: str
    descriptor: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    expected_artifact_id: str


@dataclass(frozen=True)
class InvalidObservationContractVector:
    """One declaratively mutated invalid vector."""

    case_id: str
    mode: str
    descriptor: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    original_artifact_id: str


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def observation_contract_array_sha256(values: np.ndarray) -> str:
    """Hash one array according to the version-1 contract."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def observation_contract_canonical_json_sha256(
    value: Mapping[str, Any],
) -> str:
    """Hash finite JSON data using contract canonicalization."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def observation_contract_artifact_id(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Compute the portable content address.

    A serialized ``artifact_id`` member is ignored so the same helper can hash
    both in-memory descriptors and descriptors read from NPZ.
    """

    payload = dict(descriptor)
    payload.pop("artifact_id", None)
    digest = hashlib.sha256()
    digest.update(_canonical_json(payload))
    for name, values in sorted(arrays.items()):
        digest.update(str(name).encode("utf-8"))
        digest.update(observation_contract_array_sha256(values).encode("ascii"))
    return digest.hexdigest()


def _bundle_root():
    root = resources.files(__package__)
    for component in _BUNDLE_DIRECTORY:
        root = root.joinpath(component)
    return root


def _bundle_member(relative_path: str):
    member = _bundle_root()
    for component in relative_path.split("/"):
        member = member.joinpath(component)
    return member


def _read_json(relative_path: str) -> Any:
    try:
        text = _bundle_member(relative_path).read_text(encoding="utf-8")
        return json.loads(text)
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise ValueError(
            f"observation-contract bundle member {relative_path!r} is invalid"
        ) from error


def observation_contract_bundle_manifest() -> dict[str, Any]:
    """Load and verify the complete content-addressed bundle manifest."""

    payload = _read_json("manifest.json")
    if not isinstance(payload, dict):
        raise ValueError("observation-contract manifest must be a JSON object")
    required = {
        "bundle_name",
        "bundle_version",
        "bundle_sha256",
        "canonical_repository",
        "files",
    }
    if set(payload) != required:
        raise ValueError("observation-contract manifest fields changed")
    if payload["bundle_name"] != OBSERVATION_BELIEF_CONTRACT_BUNDLE:
        raise ValueError("unexpected observation-contract bundle name")
    if int(payload["bundle_version"]) != OBSERVATION_BELIEF_CONTRACT_BUNDLE_VERSION:
        raise ValueError("unsupported observation-contract bundle version")

    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("observation-contract manifest has no files")
    normalized_files: dict[str, str] = {}
    for relative_path, expected_digest in sorted(files.items()):
        path = str(relative_path)
        if (
            not path
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("observation-contract manifest contains an unsafe path")
        expected = _validate_sha256(
            str(expected_digest),
            name=f"bundle file digest for {path}",
        )
        try:
            content = _bundle_member(path).read_bytes()
        except FileNotFoundError as error:
            raise ValueError(
                f"observation-contract bundle is missing {path}"
            ) from error
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(
                f"observation-contract bundle member {path} failed its content lock"
            )
        normalized_files[path] = expected

    descriptor = {
        "bundle_name": payload["bundle_name"],
        "bundle_version": int(payload["bundle_version"]),
        "canonical_repository": str(payload["canonical_repository"]),
        "files": normalized_files,
    }
    expected_bundle_digest = _validate_sha256(
        str(payload["bundle_sha256"]),
        name="bundle_sha256",
    )
    actual_bundle_digest = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    if actual_bundle_digest != expected_bundle_digest:
        raise ValueError(
            "observation-contract bundle digest does not match its manifest"
        )
    if actual_bundle_digest != OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256:
        raise ValueError(
            "installed observation-contract bundle differs from the code lock"
        )
    return {**descriptor, "bundle_sha256": actual_bundle_digest}


def observation_contract_schema() -> dict[str, Any]:
    """Return the verified normative schema description."""

    observation_contract_bundle_manifest()
    payload = _read_json("schema.json")
    if not isinstance(payload, dict):
        raise ValueError("observation-contract schema must be a JSON object")
    if (
        payload.get("contract_id") != OBSERVATION_BELIEF_CONTRACT_BUNDLE
        or payload.get("schema_name") != "phys4d.observation_belief"
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError("observation-contract schema identity changed")
    return payload


def observation_contract_vector(name: str) -> ObservationContractVector:
    """Load one verified valid conformance vector."""

    vector_name = str(name)
    if vector_name not in _VECTOR_NAMES:
        raise KeyError(f"unknown observation-contract vector {vector_name!r}")
    observation_contract_bundle_manifest()
    payload = _read_json(f"vectors/{vector_name}.json")
    if not isinstance(payload, dict) or set(payload) != {
        "vector_version",
        "descriptor",
        "arrays",
        "expected_artifact_id",
    }:
        raise ValueError("observation-contract vector fields changed")
    if int(payload["vector_version"]) != 1:
        raise ValueError("unsupported observation-contract vector version")
    descriptor = payload["descriptor"]
    records = payload["arrays"]
    if not isinstance(descriptor, dict) or not isinstance(records, dict):
        raise ValueError("observation-contract vector payload is invalid")

    arrays: dict[str, np.ndarray] = {}
    for array_name, record in records.items():
        if not isinstance(record, dict) or set(record) != {
            "dtype",
            "shape",
            "values",
        }:
            raise ValueError(f"array record {array_name!r} changed")
        array = np.asarray(record["values"], dtype=np.dtype(str(record["dtype"])))
        expected_shape = tuple(int(value) for value in record["shape"])
        if array.shape != expected_shape:
            raise ValueError(
                f"array record {array_name!r} has shape {array.shape}, "
                f"expected {expected_shape}"
            )
        array.setflags(write=False)
        arrays[str(array_name)] = array

    expected_artifact_id = _validate_sha256(
        str(payload["expected_artifact_id"]),
        name="expected_artifact_id",
    )
    if observation_contract_artifact_id(descriptor, arrays) != expected_artifact_id:
        raise ValueError("observation-contract vector has an invalid artifact ID")
    return ObservationContractVector(
        name=vector_name,
        descriptor=copy.deepcopy(descriptor),
        arrays=arrays,
        expected_artifact_id=expected_artifact_id,
    )


def observation_contract_invalid_cases() -> tuple[dict[str, Any], ...]:
    """Return verified declarative invalid cases for loader conformance."""

    observation_contract_bundle_manifest()
    payload = _read_json("invalid_cases.json")
    if not isinstance(payload, dict) or set(payload) != {
        "invalid_case_version",
        "base_vector",
        "cases",
    }:
        raise ValueError("observation-contract invalid-case fields changed")
    if int(payload["invalid_case_version"]) != 1:
        raise ValueError("unsupported invalid-case version")
    if payload["base_vector"] not in _VECTOR_NAMES:
        raise ValueError("invalid cases reference an unavailable base vector")
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("observation-contract bundle has no invalid cases")
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id",
            "mode",
            "mutations",
        }:
            raise ValueError("observation-contract invalid case changed")
        identifier = str(case["id"])
        if not identifier or identifier in identifiers:
            raise ValueError("invalid-case identifiers must be unique and nonempty")
        mutations = case["mutations"]
        if not isinstance(mutations, list) or not mutations:
            raise ValueError("invalid case has no mutations")
        identifiers.add(identifier)
        result.append(copy.deepcopy(case))
    return tuple(result)


def _set_path(value: Any, path: list[Any], replacement: Any) -> None:
    current = value
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = replacement


def _delete_path(value: Any, path: list[Any]) -> None:
    current = value
    for component in path[:-1]:
        current = current[component]
    del current[path[-1]]


def invalid_observation_contract_vector(
    case_id: str,
) -> InvalidObservationContractVector:
    """Materialize one invalid vector from the declarative corpus."""

    cases = {case["id"]: case for case in observation_contract_invalid_cases()}
    identifier = str(case_id)
    if identifier not in cases:
        raise KeyError(f"unknown observation-contract invalid case {identifier!r}")
    base = observation_contract_vector("minimal")
    descriptor = copy.deepcopy(dict(base.descriptor))
    arrays = {name: values.copy() for name, values in base.arrays.items()}
    case = cases[identifier]

    for mutation in case["mutations"]:
        target = mutation.get("target")
        operation = mutation.get("op")
        if target == "descriptor":
            path = list(mutation.get("path", ()))
            if not path:
                raise ValueError("descriptor mutation has no path")
            if operation == "set":
                _set_path(descriptor, path, copy.deepcopy(mutation.get("value")))
            elif operation == "delete":
                _delete_path(descriptor, path)
            else:
                raise ValueError("unsupported descriptor mutation")
        elif target == "array":
            name = str(mutation.get("name", ""))
            if name not in arrays:
                raise ValueError("array mutation references an unavailable member")
            if operation == "set":
                index = tuple(int(value) for value in mutation.get("index", ()))
                arrays[name][index] = mutation.get("value")
            elif operation == "astype":
                arrays[name] = arrays[name].astype(str(mutation.get("dtype")))
            else:
                raise ValueError("unsupported array mutation")
        elif target == "arrays" and operation == "add":
            name = str(mutation.get("name", ""))
            if not name or name in arrays:
                raise ValueError("added array name is invalid")
            array = np.asarray(
                mutation.get("value"),
                dtype=np.dtype(str(mutation.get("dtype"))),
            )
            shape = tuple(int(value) for value in mutation.get("shape", ()))
            if array.shape != shape:
                raise ValueError("added array shape is invalid")
            arrays[name] = array
        else:
            raise ValueError("unsupported observation-contract mutation target")

    for array in arrays.values():
        array.setflags(write=False)
    return InvalidObservationContractVector(
        case_id=identifier,
        mode=str(case["mode"]),
        descriptor=descriptor,
        arrays=arrays,
        original_artifact_id=base.expected_artifact_id,
    )


def main(argv: list[str] | None = None) -> int:
    """Print the verified bundle manifest."""

    parser = argparse.ArgumentParser(
        description="Verify and report the Phys4D observation-contract bundle."
    )
    parser.parse_args(argv)
    print(
        json.dumps(
            observation_contract_bundle_manifest(),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OBSERVATION_BELIEF_CONTRACT_BUNDLE",
    "OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256",
    "OBSERVATION_BELIEF_CONTRACT_BUNDLE_VERSION",
    "InvalidObservationContractVector",
    "ObservationContractVector",
    "invalid_observation_contract_vector",
    "main",
    "observation_contract_array_sha256",
    "observation_contract_artifact_id",
    "observation_contract_bundle_manifest",
    "observation_contract_canonical_json_sha256",
    "observation_contract_invalid_cases",
    "observation_contract_schema",
    "observation_contract_vector",
]
