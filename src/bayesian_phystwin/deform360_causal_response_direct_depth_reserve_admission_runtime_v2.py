"""Checksum-only runtime repair for V14 reserve source admission."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .deform360_causal_response_direct_depth_method_hash_runtime_v2 import (
    correct_v14_method_config_sha256,
    legacy_v14_method_config_sha256,
)
from .deform360_object_exclusion import file_sha256

RUNTIME_KIND = (
    "Deform360CausalResponseDirectDepthReserveAdmissionMethodHashRuntimeV14V2"
)
RUNTIME_CONTRACT = (
    "deform360-causal-response-direct-depth-reserve-admission-"
    "method-hash-runtime-v14-v2"
)
RUNTIME_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-admission-"
    "method-hash-runtime-v2"
)
METHOD_PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-source"
RESERVE_ADMISSION_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-admission-prelock"
)
RUNTIME_NAMESPACE = (
    b"deform360-causal-response-direct-depth-reserve-admission-"
    b"method-hash-runtime-v14-v2\0"
)
AMENDMENT_ARGUMENT = "--reserve-method-hash-runtime-v2"
PARENT_RUNNER = (
    "scripts/remote/run_deform360_causal_response_direct_depth_v14_admission.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read V14 reserve admission runtime artifact: {source}"
        ) from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        RUNTIME_NAMESPACE
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_v14_reserve_admission_runtime_v2(
    path: str | Path,
    *,
    repository: str | Path,
    method_protocol_path: str | Path,
    admission_prelock_path: str | Path,
) -> dict[str, Any]:
    """Validate the reserve-only checksum repair and every frozen parent."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RUNTIME_KIND
        and payload.get("contract") == RUNTIME_CONTRACT
        and payload.get("protocol_id") == RUNTIME_PROTOCOL_ID
        and payload.get("status")
        == (
            "locked_after_checksum_defect_discovery_before_first_"
            "reserve_source_admission"
        )
        and payload.get("config_sha256") == _canonical_sha256(payload),
        "V14 reserve admission runtime identity or checksum changed",
    )
    root = Path(repository).resolve()
    method_path = Path(method_protocol_path).resolve()
    admission_path = Path(admission_prelock_path).resolve()
    method = _read_json(method_path)
    admission = _read_json(admission_path)
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents) == {"method_protocol", "reserve_admission_prelock"},
        "V14 reserve admission runtime parent ledger changed",
    )
    for role, source, artifact in (
        ("method_protocol", method_path, method),
        ("reserve_admission_prelock", admission_path, admission),
    ):
        record = parents[role]
        _require(
            isinstance(record, Mapping)
            and isinstance(record.get("path"), str)
            and _valid_digest(record.get("semantic_sha256"))
            and _valid_digest(record.get("file_sha256"))
            and source == (root / record["path"]).resolve()
            and artifact.get("config_sha256") == record["semantic_sha256"]
            and file_sha256(source) == record["file_sha256"],
            f"V14 reserve admission runtime parent changed: {role}",
        )
    _require(
        method.get("protocol_id") == METHOD_PROTOCOL_ID
        and correct_v14_method_config_sha256(method) == method["config_sha256"]
        and admission.get("protocol_id") == RESERVE_ADMISSION_PROTOCOL_ID,
        "V14 reserve admission method or prelock identity changed",
    )

    implementation = payload.get("implementation")
    files = implementation.get("files") if isinstance(implementation, Mapping) else None
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and isinstance(files, Mapping)
        and set(files) == {"runtime_module", "wrapper", "admission_parent"},
        "V14 reserve admission runtime implementation ledger changed",
    )
    for name, record in files.items():
        _require(
            isinstance(record, Mapping)
            and isinstance(record.get("path"), str)
            and _valid_digest(record.get("file_sha256")),
            f"V14 reserve admission runtime file record is invalid: {name}",
        )
        implementation_path = (root / record["path"]).resolve()
        _require(
            implementation_path.is_file()
            and file_sha256(implementation_path) == record["file_sha256"],
            f"V14 reserve admission runtime implementation changed: {name}",
        )
    _require(
        files["admission_parent"]["path"] == PARENT_RUNNER,
        "V14 reserve admission runtime parent role changed",
    )

    trigger = payload.get("trigger")
    _require(
        isinstance(trigger, Mapping)
        and trigger.get("registered_namespaced_sha256") == method["config_sha256"]
        and trigger.get("correct_namespaced_sha256")
        == correct_v14_method_config_sha256(method)
        and trigger.get("legacy_unnamespaced_sha256")
        == legacy_v14_method_config_sha256(method)
        and trigger.get("legacy_matches_registered") is False
        and trigger.get("first_attempt_stage")
        == "method_checksum_validation_before_output_directory_creation"
        and trigger.get("reserve_source_admission_artifact_created") is False
        and trigger.get("source_outcome_read") is False,
        "V14 reserve admission runtime trigger changed",
    )
    amendment = payload.get("amendment")
    _require(
        isinstance(amendment, Mapping)
        and amendment.get("patched_symbol") == "_canonical_config_sha256"
        and amendment.get("method_protocol_bytes_changed") is False
        and amendment.get("admission_prelock_bytes_changed") is False
        and amendment.get("estimator_or_gate_changed") is False
        and amendment.get("only_adds_registered_hash_namespace") is True,
        "V14 reserve admission runtime amendment scope changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("object_observation_read") is False
        and boundary.get("tactile_or_robot_prefix_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve admission runtime crossed its information boundary",
    )
    return payload


def _argument_path(name: str, *, remove: bool) -> Path:
    indices = [index for index, value in enumerate(sys.argv) if value == name]
    _require(len(indices) == 1, f"{name} must appear exactly once")
    index = indices[0]
    _require(index + 1 < len(sys.argv), f"{name} lacks a value")
    path = Path(sys.argv[index + 1]).resolve()
    if remove:
        del sys.argv[index : index + 2]
    return path


def _load_parent(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bayesian_phystwin_v14_reserve_admission_parent",
        path,
    )
    _require(
        spec is not None and spec.loader is not None,
        f"cannot load V14 reserve admission parent: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_v14_reserve_admission_runtime_v2(
    *,
    wrapper_path: str | Path,
) -> int:
    """Validate, patch, and execute the reserve-capable admission parent."""

    runtime_path = _argument_path(AMENDMENT_ARGUMENT, remove=True)
    method_path = _argument_path("--method-protocol", remove=False)
    admission_path = _argument_path("--admission-prelock", remove=False)
    wrapper = Path(wrapper_path).resolve()
    repository = wrapper.parents[2]
    runtime = load_v14_reserve_admission_runtime_v2(
        runtime_path,
        repository=repository,
        method_protocol_path=method_path,
        admission_prelock_path=admission_path,
    )
    files = runtime["implementation"]["files"]
    _require(
        file_sha256(wrapper) == files["wrapper"]["file_sha256"],
        "V14 reserve admission runtime wrapper changed",
    )
    parent_path = (repository / PARENT_RUNNER).resolve()
    parent = _load_parent(parent_path)
    method = _read_json(method_path)
    _require(
        parent._canonical_config_sha256(method)
        == runtime["trigger"]["legacy_unnamespaced_sha256"],
        "V14 reserve admission parent no longer exhibits the registered defect",
    )
    parent._canonical_config_sha256 = correct_v14_method_config_sha256
    _require(
        parent._canonical_config_sha256(method) == method["config_sha256"],
        "V14 reserve admission runtime did not repair the registered checksum",
    )
    return int(parent.main())


__all__ = [
    "AMENDMENT_ARGUMENT",
    "RUNTIME_CONTRACT",
    "RUNTIME_KIND",
    "RUNTIME_NAMESPACE",
    "RUNTIME_PROTOCOL_ID",
    "load_v14_reserve_admission_runtime_v2",
    "run_v14_reserve_admission_runtime_v2",
]
