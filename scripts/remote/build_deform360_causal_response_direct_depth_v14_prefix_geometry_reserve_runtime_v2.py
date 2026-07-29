#!/usr/bin/env python3
"""Repair the reserve geometry parent-alias adapter before reconstruction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from bayesian_phystwin.deform360_causal_response_direct_depth_assets import (
    canonical_sha256,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_v14 import (
    load_v14_reserve_batch_protocol,
    load_v14_reserve_geometry_protocol,
    validate_v14_reserve_geometry_mask_input,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

RUNTIME_KIND = (
    "Deform360CausalResponseDirectDepthReserveGeometryRuntimeV14V2"
)
RUNTIME_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-geometry-runtime-v2"
)
APPLICATION_KIND = (
    "Deform360CausalResponseDirectDepthReserveGeometryApplicationV14V2"
)
APPLICATION_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-geometry-application-v2"
)
PARENT_BUILDER = (
    "build_deform360_causal_response_direct_depth_v14_prefix_geometry.py"
)
PARENT_RESERVE_WRAPPER = (
    "build_deform360_causal_response_direct_depth_v14_prefix_geometry_reserve_v1.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read V14 reserve runtime JSON: {source}") from error
    _require(isinstance(payload, dict), "V14 reserve runtime JSON is not an object")
    return payload


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _runtime_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-reserve-geometry-runtime-"
        b"v14-v2\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _argument_value(name: str, *, remove: bool) -> Path:
    indices = [index for index, value in enumerate(sys.argv) if value == name]
    _require(len(indices) == 1, f"{name} must appear exactly once")
    index = indices[0]
    _require(index + 1 < len(sys.argv), f"{name} lacks a value")
    value = Path(sys.argv[index + 1]).resolve()
    if remove:
        del sys.argv[index : index + 2]
    return value


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(
        spec is not None and spec.loader is not None,
        f"cannot load V14 reserve runtime source: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runtime(
    path: Path,
    *,
    geometry_path: Path,
    wrapper_path: Path,
) -> dict[str, Any]:
    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RUNTIME_KIND
        and payload.get("protocol_id") == RUNTIME_ID
        and payload.get("status")
        == "locked_after_parent_alias_failure_before_geometry",
        "V14 reserve geometry runtime-v2 identity changed",
    )
    _require(
        payload.get("config_sha256") == _runtime_sha256(payload),
        "V14 reserve geometry runtime-v2 checksum changed",
    )
    parent = payload.get("parent_reserve_geometry")
    _require(
        isinstance(parent, Mapping)
        and parent.get("config_sha256")
        == _read_json(geometry_path).get("config_sha256")
        and parent.get("file_sha256") == file_sha256(geometry_path),
        "V14 reserve geometry runtime-v2 uses another parent lock",
    )
    trigger = payload.get("trigger")
    _require(
        isinstance(trigger, Mapping)
        and trigger.get("failure_type") == "KeyError"
        and trigger.get("failure_key") == "parent_prefix_assets"
        and trigger.get("geometry_artifact_created") is False
        and trigger.get("result_artifact_created") is False
        and trigger.get("object_prefix_materialized") is False
        and trigger.get("method_or_gate_changed") is False,
        "V14 reserve geometry runtime-v2 trigger changed",
    )
    implementation = payload.get("implementation_file_sha256")
    _require(
        isinstance(implementation, Mapping)
        and set(implementation)
        == {
            "geometry_builder",
            "reserve_module",
            "reserve_wrapper_v1",
            "runtime_wrapper_v2",
        }
        and all(_valid_digest(value) for value in implementation.values())
        and file_sha256(wrapper_path) == implementation["runtime_wrapper_v2"],
        "V14 reserve geometry runtime-v2 implementation changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve geometry runtime-v2 crossed its information boundary",
    )
    return payload


def _write_application(
    result_path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    reserve_batch: Mapping[str, Any],
    reserve_batch_path: Path,
    runtime: Mapping[str, Any],
    runtime_path: Path,
    wrapper_path: Path,
) -> None:
    result = _read_json(result_path)
    _require(
        result.get("artifact_sha256")
        == canonical_sha256(
            result,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-"
                b"result-v14\0"
            ),
            digest_key="artifact_sha256",
        )
        and result.get("geometry_protocol_config_sha256")
        == protocol["config_sha256"],
        "V14 reserve geometry result changed before runtime-v2 binding",
    )
    application_path = result_path.with_name(
        f"{result_path.stem}.reserve-runtime-v2.json"
    )
    _require(
        not application_path.exists(),
        "refusing to replace V14 reserve runtime-v2 application",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": APPLICATION_KIND,
        "protocol_id": APPLICATION_ID,
        "status": "reserve_geometry_runtime_v2_applied",
        "queue_rank": result["queue_rank"],
        "object_hash": result["object_hash"],
        "case_hash": result["case_hash"],
        "reserve_geometry_config_sha256": protocol["config_sha256"],
        "reserve_geometry_file_sha256": file_sha256(protocol_path),
        "reserve_batch_config_sha256": reserve_batch["config_sha256"],
        "reserve_batch_file_sha256": file_sha256(reserve_batch_path),
        "runtime_v2_config_sha256": runtime["config_sha256"],
        "runtime_v2_file_sha256": file_sha256(runtime_path),
        "geometry_result_artifact_sha256": result["artifact_sha256"],
        "geometry_result_file_sha256": file_sha256(result_path),
        "runtime_wrapper_v2_sha256": file_sha256(wrapper_path),
        "information_boundary": {
            "maximum_object_observation_frame": 57,
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "source_outcome_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    payload["artifact_sha256"] = canonical_sha256(
        payload,
        namespace=(
            b"deform360-causal-response-direct-depth-reserve-geometry-"
            b"application-v14-v2\0"
        ),
        digest_key="artifact_sha256",
    )
    temporary = application_path.with_name(f".{application_path.name}.tmp")
    _require(
        not temporary.exists(),
        "V14 reserve runtime-v2 application scratch exists",
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(application_path)


def main() -> int:
    runtime_path = _argument_value("--reserve-runtime-v2", remove=True)
    reserve_batch_path = _argument_value("--reserve-batch", remove=True)
    result_path = _argument_value("--result", remove=True)
    sys.argv.extend(["--result", str(result_path)])
    repository = _argument_value("--repo", remove=False)
    geometry_path = _argument_value("--geometry-protocol", remove=False)
    asset_path = _argument_value("--asset-protocol", remove=False)
    queue_path = _argument_value("--queue", remove=False)
    wrapper_path = Path(__file__).resolve()
    runtime = _load_runtime(
        runtime_path,
        geometry_path=geometry_path,
        wrapper_path=wrapper_path,
    )
    reserve_batch = load_v14_reserve_batch_protocol(
        reserve_batch_path,
        asset_protocol_path=asset_path,
        queue_path=queue_path,
    )
    protocol = load_v14_reserve_geometry_protocol(
        geometry_path,
        reserve_batch_path=reserve_batch_path,
        asset_protocol_path=asset_path,
        queue_path=queue_path,
    )
    implementation = runtime["implementation_file_sha256"]
    parent_builder_path = wrapper_path.with_name(PARENT_BUILDER)
    parent_reserve_wrapper = wrapper_path.with_name(PARENT_RESERVE_WRAPPER)
    _require(
        file_sha256(parent_builder_path) == implementation["geometry_builder"]
        and file_sha256(parent_reserve_wrapper)
        == implementation["reserve_wrapper_v1"]
        and file_sha256(
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_reserve_v14.py"
        )
        == implementation["reserve_module"],
        "V14 reserve geometry runtime-v2 source binding changed",
    )
    parent = _load_module(
        parent_builder_path,
        "_v14_reserve_geometry_runtime_v2_parent",
    )
    builder_protocol = dict(protocol)
    builder_protocol["parent_prefix_assets"] = {
        "protocol_id": "deform360-causal-response-direct-depth-v14-prefix-assets",
        "config_sha256": protocol["parent_artifacts"]["prefix_assets"][
            "config_sha256"
        ],
        "file_sha256": protocol["parent_artifacts"]["prefix_assets"][
            "file_sha256"
        ],
    }

    def load_protocol(path: str | Path) -> dict[str, Any]:
        _require(
            Path(path).resolve() == geometry_path,
            "V14 reserve runtime-v2 received another geometry protocol",
        )
        return builder_protocol

    parent.load_v14_prefix_geometry_protocol = load_protocol
    parent.validate_v14_geometry_mask_input = (
        validate_v14_reserve_geometry_mask_input
    )
    return_code = int(parent.main())
    _require(result_path.is_file(), "V14 reserve runtime-v2 result is missing")
    _write_application(
        result_path,
        protocol=protocol,
        protocol_path=geometry_path,
        reserve_batch=reserve_batch,
        reserve_batch_path=reserve_batch_path,
        runtime=runtime,
        runtime_path=runtime_path,
        wrapper_path=wrapper_path,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
