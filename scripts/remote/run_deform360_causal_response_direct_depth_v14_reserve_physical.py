#!/usr/bin/env python3
"""Run the frozen V14 physical backend under the reserve child ledgers."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from bayesian_phystwin import (
    deform360_causal_response_direct_depth_physical as physical_module,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_physical_v14 import (
    load_v14_reserve_physical_prelock,
    load_v14_reserve_physical_runtime,
    v14_reserve_physical_case_record,
    validate_v14_reserve_geometry_bundle_v2,
    validate_v14_reserve_physical_action,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256

PARENT_RUNNER = "run_deform360_causal_response_direct_depth_v14_physical.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _argument_value(name: str) -> Path:
    indices = [index for index, value in enumerate(sys.argv) if value == name]
    _require(len(indices) == 1, f"{name} must appear exactly once")
    index = indices[0]
    _require(index + 1 < len(sys.argv), f"{name} lacks a value")
    return Path(sys.argv[index + 1]).resolve()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(
        spec is not None and spec.loader is not None,
        f"cannot load V14 reserve physical source: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_bound_reserve_geometry(
    *,
    protocol: Mapping[str, Any],
    case_record: Mapping[str, Any],
    geometry_protocol_path: Path,
    runtime_v1_path: Path,
    validation_v1_path: Path,
    runtime_v2_path: Path,
    validation_v2_path: Path,
    manifest_path: Path,
    result_path: Path,
    runtime_application_path: Path,
    geometry_episode: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    del runtime_v2_path, validation_v2_path
    manifest, result, application = validate_v14_reserve_geometry_bundle_v2(
        manifest_path=manifest_path,
        result_path=result_path,
        application_path=runtime_application_path,
        geometry_protocol_path=geometry_protocol_path,
        reserve_batch_path=runtime_v1_path,
        runtime_v2_path=validation_v1_path,
        geometry_episode=geometry_episode,
    )
    rank = int(case_record["queue_rank"])
    expected = next(
        record
        for record in protocol["geometry_cases"]
        if int(record["queue_rank"]) == rank
    )
    observed = {
        "object_hash": manifest["object_hash"],
        "case_hash": manifest["case_hash"],
        "physical_node_count": int(manifest["physical_node_count"]),
        "successful_camera_count": len(manifest["cameras"]),
        "geometry_manifest_artifact_sha256": manifest["artifact_sha256"],
        "geometry_manifest_file_sha256": file_sha256(manifest_path),
        "geometry_result_artifact_sha256": result["artifact_sha256"],
        "geometry_result_file_sha256": file_sha256(result_path),
        "runtime_application_artifact_sha256": application["artifact_sha256"],
        "runtime_application_file_sha256": file_sha256(
            runtime_application_path
        ),
        "runtime_contract_version": "reserve-v2",
    }
    _require(
        all(expected.get(key) == value for key, value in observed.items()),
        "V14 reserve physical geometry differs from its prelock ledger",
    )
    _require(
        observed["object_hash"] == case_record["object_hash"]
        and observed["case_hash"] == case_record["case_hash"]
        and observed["physical_node_count"] == case_record["physical_node_count"],
        "V14 reserve physical geometry differs from its case record",
    )
    return manifest, result, application


def main() -> int:
    repository = _argument_value("--repo")
    prelock_path = _argument_value("--prelock-protocol")
    runtime_path = _argument_value("--physical-runtime-v2")
    protocol = load_v14_reserve_physical_prelock(prelock_path)
    runtime = load_v14_reserve_physical_runtime(
        runtime_path,
        parent_prelock_path=prelock_path,
    )
    wrapper_path = Path(__file__).resolve()
    parent_path = wrapper_path.with_name(PARENT_RUNNER)
    implementation = protocol["implementation"]["file_sha256"]
    runtime_implementation = runtime["implementation"]["file_sha256"]
    _require(
        file_sha256(wrapper_path) == implementation["reserve_physical_runner"]
        and file_sha256(wrapper_path) == runtime_implementation["reserve_runner"]
        and file_sha256(parent_path) == implementation["parent_physical_runner"]
        and file_sha256(parent_path) == runtime_implementation["physical_runner"]
        and file_sha256(
            repository
            / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_reserve_physical_v14.py"
        )
        == implementation["reserve_physical_module"]
        and file_sha256(
            repository
            / "scripts/remote/"
            "build_deform360_causal_response_direct_depth_v14_automatic_twin.py"
        )
        == implementation["automatic_twin"],
        "V14 reserve physical implementation changed",
    )
    parent = _load_module(parent_path, "_v14_reserve_physical_parent")
    parent.load_v14_physical_prelock_protocol = (
        load_v14_reserve_physical_prelock
    )
    parent.v14_physical_case_record = v14_reserve_physical_case_record
    parent.load_v14_physical_runtime_v2 = load_v14_reserve_physical_runtime
    parent.validate_v14_physical_action_v2 = (
        validate_v14_reserve_physical_action
    )
    parent._validate_bound_geometry = _validate_bound_reserve_geometry
    physical_module.load_v14_physical_prelock_protocol = (
        load_v14_reserve_physical_prelock
    )
    original_writer = parent.write_v14_physical_artifacts

    def write_with_reserve_provenance(
        *args: Any,
        runtime_provenance: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        provenance = dict(runtime_provenance)
        provenance.update(
            {
                "reserve_physical_prelock_config_sha256": protocol[
                    "config_sha256"
                ],
                "reserve_physical_runtime_config_sha256": runtime[
                    "config_sha256"
                ],
                "reserve_physical_runner_sha256": file_sha256(wrapper_path),
            }
        )
        return original_writer(
            *args,
            runtime_provenance=provenance,
            **kwargs,
        )

    parent.write_v14_physical_artifacts = write_with_reserve_provenance
    return int(parent.main())


if __name__ == "__main__":
    raise SystemExit(main())
