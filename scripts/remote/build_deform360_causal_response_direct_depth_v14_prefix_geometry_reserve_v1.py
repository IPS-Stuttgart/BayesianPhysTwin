#!/usr/bin/env python3
"""Run the fixed V14 reserve geometry batch with direct runtime custody."""

from __future__ import annotations

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
    RESERVE_GEOMETRY_APPLICATION_ID,
    RESERVE_GEOMETRY_APPLICATION_KIND,
    load_v14_reserve_batch_protocol,
    load_v14_reserve_geometry_protocol,
    validate_v14_reserve_geometry_mask_input,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

PARENT_BUILDER = (
    "build_deform360_causal_response_direct_depth_v14_prefix_geometry.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read V14 reserve artifact: {path}") from error
    _require(isinstance(payload, dict), "V14 reserve artifact is not an object")
    return payload


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
        f"cannot load V14 reserve source: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_application(
    result_path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    reserve_batch: Mapping[str, Any],
    reserve_batch_path: Path,
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
        "V14 reserve geometry result changed before child binding",
    )
    application_path = result_path.with_name(
        f"{result_path.stem}.reserve-v1.json"
    )
    _require(
        not application_path.exists(),
        "refusing to replace V14 reserve geometry application",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESERVE_GEOMETRY_APPLICATION_KIND,
        "protocol_id": RESERVE_GEOMETRY_APPLICATION_ID,
        "status": "reserve_geometry_child_applied",
        "queue_rank": result["queue_rank"],
        "object_hash": result["object_hash"],
        "case_hash": result["case_hash"],
        "reserve_geometry_config_sha256": protocol["config_sha256"],
        "reserve_geometry_file_sha256": file_sha256(protocol_path),
        "reserve_batch_config_sha256": reserve_batch["config_sha256"],
        "reserve_batch_file_sha256": file_sha256(reserve_batch_path),
        "geometry_result_artifact_sha256": result["artifact_sha256"],
        "geometry_result_file_sha256": file_sha256(result_path),
        "reserve_wrapper_sha256": file_sha256(wrapper_path),
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
            b"application-v14-v1\0"
        ),
        digest_key="artifact_sha256",
    )
    temporary = application_path.with_name(f".{application_path.name}.tmp")
    _require(not temporary.exists(), "V14 reserve application scratch exists")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(application_path)


def main() -> int:
    reserve_batch_path = _argument_value("--reserve-batch", remove=True)
    result_path = _argument_value("--result", remove=True)
    sys.argv.extend(["--result", str(result_path)])
    repository = _argument_value("--repo", remove=False)
    geometry_path = _argument_value("--geometry-protocol", remove=False)
    asset_path = _argument_value("--asset-protocol", remove=False)
    queue_path = _argument_value("--queue", remove=False)
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
    wrapper_path = Path(__file__).resolve()
    parent_builder_path = wrapper_path.with_name(PARENT_BUILDER)
    implementation = protocol["implementation_file_sha256"]
    _require(
        file_sha256(parent_builder_path) == implementation["geometry_builder"]
        and file_sha256(
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_prefix_geometry.py"
        )
        == implementation["geometry_module"]
        and file_sha256(
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_reserve_v14.py"
        )
        == implementation["reserve_module"]
        and file_sha256(wrapper_path) == implementation["reserve_wrapper"],
        "V14 reserve geometry implementation changed",
    )
    parent = _load_module(
        parent_builder_path,
        "_v14_reserve_geometry_parent_builder",
    )

    def load_protocol(path: str | Path) -> dict[str, Any]:
        _require(
            Path(path).resolve() == geometry_path,
            "V14 reserve builder received another geometry protocol",
        )
        return protocol

    parent.load_v14_prefix_geometry_protocol = load_protocol
    parent.validate_v14_geometry_mask_input = (
        validate_v14_reserve_geometry_mask_input
    )
    return_code = int(parent.main())
    _require(result_path.is_file(), "V14 reserve geometry result is missing")
    _write_application(
        result_path,
        protocol=protocol,
        protocol_path=geometry_path,
        reserve_batch=reserve_batch,
        reserve_batch_path=reserve_batch_path,
        wrapper_path=wrapper_path,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
