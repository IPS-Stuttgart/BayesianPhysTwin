#!/usr/bin/env python3
"""Apply the frozen V14 gsplat rebuild amendment to prefix geometry."""

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
from bayesian_phystwin.deform360_object_exclusion import file_sha256

AMENDMENT_KIND = "Deform360CausalDirectDepthPrefixGeometryRuntimeV14"
AMENDMENT_ID = (
    "deform360-causal-response-direct-depth-v14-prefix-geometry-runtime"
)
APPLICATION_KIND = (
    "Deform360CausalDirectDepthPrefixGeometryRuntimeApplicationV14"
)
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
        raise ValueError(f"cannot read JSON artifact: {path}") from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _extract_argument(name: str) -> Path:
    indices = [index for index, value in enumerate(sys.argv) if value == name]
    _require(len(indices) == 1, f"{name} must appear exactly once")
    index = indices[0]
    _require(index + 1 < len(sys.argv), f"{name} lacks a value")
    value = Path(sys.argv[index + 1]).resolve()
    del sys.argv[index : index + 2]
    return value


def _load_amendment(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == AMENDMENT_KIND
        and payload.get("protocol_id") == AMENDMENT_ID
        and payload.get("status")
        == "locked_after_runtime_preflight_failure_before_geometry",
        "V14 geometry runtime amendment identity changed",
    )
    _require(
        payload.get("config_sha256")
        == canonical_sha256(
            payload,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-"
                b"runtime-v14\0"
            ),
            digest_key="config_sha256",
        ),
        "V14 geometry runtime amendment checksum changed",
    )
    trigger = payload.get("trigger")
    _require(
        isinstance(trigger, Mapping)
        and trigger.get("attempted_queue_rank") == 3
        and trigger.get("failure_type") == "ValueError"
        and trigger.get("failure_message") == "gsplat CUDA extension changed"
        and trigger.get("geometry_artifact_created") is False
        and trigger.get("result_artifact_created") is False
        and trigger.get("candidate_remains_prelock_unattempted") is True
        and trigger.get("future_identity_or_metric_read") is False
        and trigger.get("target_object_or_outcome_read") is False
        and trigger.get("held_v8_access") is False,
        "V14 geometry runtime amendment trigger changed",
    )
    runtime = payload.get("runtime_amendment")
    _require(
        isinstance(runtime, Mapping)
        and runtime.get("parent_extension_sha256")
        == "58c95816cdf011dbbd13a71f1d98312c9e661ef34c95592cc00ff93c72cab89b"
        and runtime.get("rebuilt_extension_sha256")
        == "99733266b0b679eef3e21ee748582266584d38c72921330f8118c3f5f4eddf63"
        and runtime.get("extension_path")
        == (
            "/home/florianpfaff/.cache/torch_extensions/py310_cu121/"
            "gsplat_cuda/gsplat_cuda.so"
        )
        and runtime.get("semantic_runtime_contract_changed") is False,
        "V14 geometry runtime amendment changed",
    )
    return payload


def _load_parent_builder(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bayesian_phystwin_v14_prefix_geometry_parent",
        path,
    )
    _require(
        spec is not None and spec.loader is not None,
        "cannot load the V14 parent geometry builder",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_application(
    result_path: Path,
    *,
    amendment: Mapping[str, Any],
    amendment_path: Path,
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
        ),
        "V14 geometry result checksum changed before runtime binding",
    )
    application_path = result_path.with_name(
        f"{result_path.stem}.runtime.json"
    )
    _require(
        not application_path.exists(),
        f"refusing to replace V14 runtime application: {application_path}",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": APPLICATION_KIND,
        "protocol_id": AMENDMENT_ID,
        "runtime_amendment_config_sha256": amendment["config_sha256"],
        "runtime_amendment_file_sha256": file_sha256(amendment_path),
        "runtime_wrapper_sha256": file_sha256(wrapper_path),
        "geometry_result_artifact_sha256": result["artifact_sha256"],
        "geometry_result_file_sha256": file_sha256(result_path),
        "queue_rank": result["queue_rank"],
        "object_hash": result["object_hash"],
        "case_hash": result["case_hash"],
        "status": "runtime_amendment_applied",
        "information_boundary": {
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    payload["artifact_sha256"] = canonical_sha256(
        payload,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-"
            b"runtime-application-v14\0"
        ),
        digest_key="artifact_sha256",
    )
    temporary = application_path.with_name(f".{application_path.name}.tmp")
    _require(
        not temporary.exists(),
        f"temporary V14 runtime application exists: {temporary}",
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(application_path)


def main() -> int:
    amendment_path = _extract_argument("--runtime-amendment")
    result_path = _extract_argument("--result")
    sys.argv.extend(["--result", str(result_path)])
    amendment = _load_amendment(amendment_path)
    wrapper_path = Path(__file__).resolve()
    repository = wrapper_path.parents[2]
    parent_protocol_path = repository / amendment["parent_geometry_protocol"]["path"]
    parent_builder_path = wrapper_path.with_name(PARENT_BUILDER)
    _require(
        file_sha256(parent_protocol_path)
        == amendment["parent_geometry_protocol"]["file_sha256"]
        and file_sha256(parent_builder_path)
        == amendment["parent_geometry_protocol"]["geometry_builder_sha256"]
        and file_sha256(wrapper_path)
        == amendment["implementation_file_sha256"]["runtime_wrapper"],
        "V14 geometry runtime amendment source binding changed",
    )
    parent = _load_parent_builder(parent_builder_path)
    original_validate_runtime = parent._validate_runtime

    def amended_validate_runtime(runtime: Mapping[str, Any]) -> dict[str, str]:
        _require(
            runtime.get("gsplat_extension_sha256")
            == amendment["runtime_amendment"]["parent_extension_sha256"],
            "V14 parent runtime contract no longer matches the amendment",
        )
        amended = dict(runtime)
        amended["gsplat_extension_sha256"] = amendment["runtime_amendment"][
            "rebuilt_extension_sha256"
        ]
        probe = original_validate_runtime(amended)
        probe.update(
            {
                "runtime_amendment_protocol_id": AMENDMENT_ID,
                "runtime_amendment_config_sha256": amendment["config_sha256"],
                "runtime_amendment_file_sha256": file_sha256(amendment_path),
                "runtime_wrapper_sha256": file_sha256(wrapper_path),
                "parent_extension_sha256": runtime["gsplat_extension_sha256"],
            }
        )
        return probe

    parent._validate_runtime = amended_validate_runtime
    return_code = int(parent.main())
    _require(result_path.is_file(), "V14 geometry builder did not write a result")
    _write_application(
        result_path,
        amendment=amendment,
        amendment_path=amendment_path,
        wrapper_path=wrapper_path,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
