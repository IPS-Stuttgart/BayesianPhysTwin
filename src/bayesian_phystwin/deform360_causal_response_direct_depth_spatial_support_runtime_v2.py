"""Crash-to-rejection runtime amendment for sparse V14 spatial support."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .deform360_causal_response_direct_depth_reserve_prediction_v14 import (
    PARENT_PREDICTION_RUNNER,
    load_v14_composite_physical_custody,
    load_v14_reserve_prediction_runtime,
    validate_v14_mixed_physical_artifacts,
)
from .deform360_object_exclusion import file_sha256

RUNTIME_KIND = "Deform360CausalResponseDirectDepthSparseSpatialSupportRuntimeV14V2"
RUNTIME_CONTRACT = (
    "deform360-causal-response-direct-depth-sparse-spatial-support-runtime-v14-v2"
)
RUNTIME_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-sparse-spatial-support-runtime-v2"
)
RUNTIME_NAMESPACE = (
    b"deform360-causal-response-direct-depth-sparse-spatial-support-runtime-v14-v2\0"
)
AMENDMENT_ARGUMENT = "--spatial-support-runtime-v2"
ADMISSION_RELATIVE_PATH = "src/bayesian_phystwin/deform360_causal_response_admission.py"


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
            f"cannot read V14 sparse-spatial runtime artifact: {source}"
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


def load_v14_sparse_spatial_support_runtime_v2(
    path: str | Path,
    *,
    repository: str | Path,
    method_protocol_path: str | Path,
    source_lock_path: str | Path,
) -> dict[str, Any]:
    """Validate the narrow crash-to-rejection amendment and its trigger."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RUNTIME_KIND
        and payload.get("contract") == RUNTIME_CONTRACT
        and payload.get("protocol_id") == RUNTIME_PROTOCOL_ID
        and payload.get("status")
        == (
            "locked_after_first_prediction_technical_failure_"
            "before_first_prediction_seal"
        )
        and payload.get("config_sha256") == _canonical_sha256(payload),
        "V14 sparse-spatial runtime identity or checksum changed",
    )
    root = Path(repository).resolve()
    method_path = Path(method_protocol_path).resolve()
    source_path = Path(source_lock_path).resolve()
    method = _read_json(method_path)
    source = _read_json(source_path)
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents) == {"method_protocol", "source_lock"},
        "V14 sparse-spatial runtime parent ledger changed",
    )
    for role, artifact_path, artifact, semantic_key in (
        ("method_protocol", method_path, method, "config_sha256"),
        ("source_lock", source_path, source, "artifact_sha256"),
    ):
        record = parents[role]
        _require(
            isinstance(record, Mapping)
            and _valid_digest(record.get("semantic_sha256"))
            and _valid_digest(record.get("file_sha256"))
            and artifact.get(semantic_key) == record["semantic_sha256"]
            and file_sha256(artifact_path) == record["file_sha256"],
            f"V14 sparse-spatial runtime parent changed: {role}",
        )
    amendment = payload.get("amendment")
    _require(
        isinstance(amendment, Mapping)
        and amendment.get("relative_path") == ADMISSION_RELATIVE_PATH
        and amendment.get("old_file_sha256")
        == method["implementation_file_sha256"][ADMISSION_RELATIVE_PATH]
        and amendment.get("old_file_sha256")
        == "ff1feda3026a5fbc893181b4f7c4ba9d4df7076a9dee76531e2e58d17a626be5"
        and amendment.get("new_file_sha256")
        == file_sha256(root / ADMISSION_RELATIVE_PATH)
        and amendment.get("new_file_sha256")
        == "5976207fc98989a161b026238967bdfd2ac5ad248cc96f68a613219f1a56703e"
        and amendment.get("adequately_supported_cases_numerically_changed") is False
        and amendment.get("gate_threshold_changed") is False
        and amendment.get("crash_becomes_registered_rejection") is True,
        "V14 sparse-spatial amendment scope changed",
    )
    implementation = payload.get("implementation")
    files = (
        implementation.get("file_sha256")
        if isinstance(implementation, Mapping)
        else None
    )
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and isinstance(files, Mapping)
        and set(files)
        == {
            "reserve_prediction_module",
            "runtime_module",
            "runtime_wrapper",
        }
        and all(_valid_digest(value) for value in files.values()),
        "V14 sparse-spatial runtime implementation ledger changed",
    )
    implementation_paths = {
        "reserve_prediction_module": (
            root / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_reserve_prediction_v14.py"
        ),
        "runtime_module": Path(__file__).resolve(),
        "runtime_wrapper": (
            root / "scripts/remote/"
            "run_deform360_causal_response_direct_depth_v14_"
            "reserve_prediction_runtime_v2.py"
        ),
    }
    _require(
        all(
            path.is_file() and file_sha256(path) == files[name]
            for name, path in implementation_paths.items()
        ),
        "V14 sparse-spatial runtime implementation changed",
    )
    trigger = payload.get("trigger")
    _require(
        isinstance(trigger, Mapping)
        and trigger.get("queue_rank") == 3
        and trigger.get("failure_stage")
        == "admission_dataclass_before_prediction_artifact"
        and trigger.get("exception")
        == "selected entities or spatial groups are invalid"
        and trigger.get("prediction_artifact_created") is False
        and trigger.get("source_outcome_read") is False
        and _valid_digest(trigger.get("failed_runtime_config_sha256"))
        and _valid_digest(trigger.get("failed_runtime_file_sha256"))
        and _valid_digest(trigger.get("sealed_prefix_result_sha256"))
        and _valid_digest(trigger.get("sealed_prefix_report_file_sha256")),
        "V14 sparse-spatial runtime trigger changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("permitted_prefix_read") is True
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 sparse-spatial runtime crossed its information boundary",
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
        "bayesian_phystwin_v14_sparse_spatial_prediction_parent",
        path,
    )
    _require(
        spec is not None and spec.loader is not None,
        f"cannot load V14 sparse-spatial prediction parent: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_v14_sparse_spatial_support_prediction(
    *,
    wrapper_path: str | Path,
) -> int:
    """Apply the registered method-ledger amendment and run frozen V14."""

    amendment_path = _argument_path(AMENDMENT_ARGUMENT, remove=True)
    wrapper = Path(wrapper_path).resolve()
    repository = wrapper.parents[2]
    parent = _load_parent((repository / PARENT_PREDICTION_RUNNER).resolve())
    args = parent._parse_args()
    amendment = load_v14_sparse_spatial_support_runtime_v2(
        amendment_path,
        repository=repository,
        method_protocol_path=args.method_protocol.resolve(),
        source_lock_path=args.source_lock.resolve(),
    )
    runtime = load_v14_reserve_prediction_runtime(
        args.prediction_runtime.resolve(),
        repository=repository,
        method_protocol_path=args.method_protocol.resolve(),
        source_lock_path=args.source_lock.resolve(),
        admission_custody_path=args.admission_prelock.resolve(),
        physical_custody_path=args.physical_prelock.resolve(),
    )
    physical_custody = load_v14_composite_physical_custody(
        args.physical_prelock.resolve(),
        repository=repository,
    )
    original_json_loader = parent._load_json

    def load_json(path: str | Path) -> dict[str, Any]:
        payload = original_json_loader(path)
        if Path(path).resolve() == args.method_protocol.resolve():
            payload = json.loads(json.dumps(payload))
            ledger = payload["implementation_file_sha256"]
            _require(
                ledger[ADMISSION_RELATIVE_PATH]
                == amendment["amendment"]["old_file_sha256"],
                "V14 method ledger no longer contains the registered old hash",
            )
            ledger[ADMISSION_RELATIVE_PATH] = amendment["amendment"]["new_file_sha256"]
        return payload

    def load_runtime(
        path: str | Path,
        *,
        method_protocol_path: str | Path,
        source_lock_path: str | Path,
        admission_prelock_path: str | Path,
        physical_prelock_path: str | Path,
    ) -> dict[str, Any]:
        return load_v14_reserve_prediction_runtime(
            path,
            repository=repository,
            method_protocol_path=method_protocol_path,
            source_lock_path=source_lock_path,
            admission_custody_path=admission_prelock_path,
            physical_custody_path=physical_prelock_path,
        )

    def validate_physical(
        output_dir: str | Path,
        *,
        prelock_protocol_path: str | Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require(
            Path(prelock_protocol_path).resolve() == args.physical_prelock.resolve(),
            "V14 sparse-spatial physical custody path changed",
        )
        return validate_v14_mixed_physical_artifacts(
            output_dir,
            custody=physical_custody,
            repository=repository,
        )

    parent._parse_args = lambda: args
    parent._load_json = load_json
    parent.load_v14_prediction_runtime = load_runtime
    parent.validate_v14_physical_artifacts = validate_physical
    _require(
        runtime["information_boundary"]["source_outcome_read"] is False,
        "V14 sparse-spatial runtime crossed the source outcome boundary",
    )
    return int(parent.main())


__all__ = [
    "ADMISSION_RELATIVE_PATH",
    "AMENDMENT_ARGUMENT",
    "RUNTIME_CONTRACT",
    "RUNTIME_KIND",
    "RUNTIME_NAMESPACE",
    "RUNTIME_PROTOCOL_ID",
    "load_v14_sparse_spatial_support_runtime_v2",
    "run_v14_sparse_spatial_support_prediction",
]
