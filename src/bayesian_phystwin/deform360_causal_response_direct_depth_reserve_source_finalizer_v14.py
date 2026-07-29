"""Mixed-custody finalization for the prospective V14 source panel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .deform360_causal_response_direct_depth_admission_v14 import (
    load_v14_admission_prelock_protocol,
    validate_v14_admission_report,
)
from .deform360_causal_response_direct_depth_method_hash_runtime_v2 import (
    correct_v14_method_config_sha256,
    legacy_v14_method_config_sha256,
)
from .deform360_causal_response_direct_depth_reserve_admission_runtime_v2 import (
    load_v14_reserve_admission_runtime_v2,
)
from .deform360_object_exclusion import file_sha256

CUSTODY_KIND = "Deform360CausalResponseDirectDepthCompositeAdmissionCustodyV14V1"
CUSTODY_CONTRACT = (
    "deform360-causal-response-direct-depth-composite-admission-custody-v14-v1"
)
CUSTODY_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-composite-admission-custody-v1"
)
CUSTODY_NAMESPACE = (
    b"deform360-causal-response-direct-depth-composite-admission-custody-v14-v1\0"
)
FINALIZER_KIND = "Deform360CausalResponseDirectDepthReserveSourceFinalizerProtocolV14V1"
FINALIZER_CONTRACT = (
    "deform360-causal-response-direct-depth-reserve-source-finalizer-v14-v1"
)
FINALIZER_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-source-finalizer-v1"
)
FINALIZER_NAMESPACE = (
    b"deform360-causal-response-direct-depth-reserve-source-finalizer-v14-v1\0"
)
METHOD_PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-source"
PARENT_RUNNER = (
    "scripts/remote/finalize_deform360_causal_response_direct_depth_v14_source.py"
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
            f"cannot read V14 reserve finalizer artifact: {source}"
        ) from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {source}")
    return payload


def _canonical_sha256(payload: Mapping[str, Any], *, namespace: bytes) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        namespace
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_parent_record(
    *,
    root: Path,
    record: Mapping[str, Any],
    artifact: Mapping[str, Any],
    path: Path,
    role: str,
) -> None:
    _require(
        isinstance(record.get("path"), str)
        and _valid_digest(record.get("semantic_sha256"))
        and _valid_digest(record.get("file_sha256"))
        and path == (root / record["path"]).resolve()
        and artifact.get("config_sha256") == record["semantic_sha256"]
        and file_sha256(path) == record["file_sha256"],
        f"V14 reserve finalizer parent changed: {role}",
    )


def load_v14_composite_admission_custody(
    path: str | Path,
    *,
    repository: str | Path,
) -> dict[str, Any]:
    """Validate the two admission prelocks and their immutable rank scopes."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == CUSTODY_KIND
        and payload.get("contract") == CUSTODY_CONTRACT
        and payload.get("protocol_id") == CUSTODY_PROTOCOL_ID
        and payload.get("status")
        == "locked_after_twelfth_admission_before_source_finalization"
        and payload.get("config_sha256")
        == _canonical_sha256(payload, namespace=CUSTODY_NAMESPACE),
        "V14 composite admission custody identity or checksum changed",
    )
    root = Path(repository).resolve()
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents)
        == {
            "method_protocol",
            "original_admission_prelock",
            "original_method_hash_runtime",
            "reserve_admission_prelock",
            "reserve_admission_runtime",
        },
        "V14 composite admission custody parent ledger changed",
    )
    artifacts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for role, record in parents.items():
        _require(
            isinstance(record, Mapping) and isinstance(record.get("path"), str),
            f"V14 composite admission custody parent is invalid: {role}",
        )
        parent_path = (root / record["path"]).resolve()
        artifact = _read_json(parent_path)
        _validate_parent_record(
            root=root,
            record=record,
            artifact=artifact,
            path=parent_path,
            role=role,
        )
        artifacts[role] = (parent_path, artifact)
    method_path, method = artifacts["method_protocol"]
    original_path, original = artifacts["original_admission_prelock"]
    reserve_path, reserve = artifacts["reserve_admission_prelock"]
    _require(
        method.get("protocol_id") == METHOD_PROTOCOL_ID
        and correct_v14_method_config_sha256(method) == method["config_sha256"],
        "V14 composite admission custody method changed",
    )
    load_v14_admission_prelock_protocol(original_path)
    load_v14_admission_prelock_protocol(reserve_path)
    reserve_runtime_path, _ = artifacts["reserve_admission_runtime"]
    load_v14_reserve_admission_runtime_v2(
        reserve_runtime_path,
        repository=root,
        method_protocol_path=method_path,
        admission_prelock_path=reserve_path,
    )

    ranks = payload.get("rank_contract")
    _require(
        isinstance(ranks, Mapping)
        and ranks.get("technical_window_disposition_ranks") == [1, 2]
        and ranks.get("original_admission_rank_range_inclusive") == [3, 14]
        and ranks.get("reserve_admission_rank_range_inclusive") == [15, 18]
        and ranks.get("final_queue_rank") == 18
        and ranks.get("required_selected_source_count") == 12
        and ranks.get("original_admission_prelock_config_sha256")
        == original["config_sha256"]
        and ranks.get("reserve_admission_prelock_config_sha256")
        == reserve["config_sha256"]
        and ranks.get("original_repository_revision")
        == "ac416f49f3d5f464348c843c0f918b052ee54874"
        and ranks.get("reserve_repository_revision")
        == "08e8869a07d8356f51d4fe7240ea0d4939d7cf24"
        and ranks.get("prepared_but_unselected_reserve_ranks") == [19, 20, 21, 22],
        "V14 composite admission custody rank contract changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("admission_dispositions_read") is True
        and boundary.get("prefix_or_future_object_response_read") is False
        and boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 composite admission custody crossed its information boundary",
    )
    return payload


def load_v14_reserve_source_finalizer_protocol(
    path: str | Path,
    *,
    repository: str | Path,
    composite_custody_path: str | Path,
) -> dict[str, Any]:
    """Validate the reserve-aware finalizer and frozen parent implementation."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == FINALIZER_KIND
        and payload.get("contract") == FINALIZER_CONTRACT
        and payload.get("protocol_id") == FINALIZER_PROTOCOL_ID
        and payload.get("status")
        == "locked_after_twelfth_admission_before_source_finalization"
        and payload.get("config_sha256")
        == _canonical_sha256(payload, namespace=FINALIZER_NAMESPACE),
        "V14 reserve source finalizer identity or checksum changed",
    )
    root = Path(repository).resolve()
    custody_path = Path(composite_custody_path).resolve()
    custody = load_v14_composite_admission_custody(
        custody_path,
        repository=root,
    )
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents)
        == {
            "admission_prelock",
            "exclusion_manifest",
            "method_protocol",
            "staging_queue",
            "synthetic_control",
        }
        and parents["admission_prelock"].get("semantic_sha256")
        == custody["config_sha256"]
        and parents["admission_prelock"].get("file_sha256")
        == file_sha256(custody_path),
        "V14 reserve source finalizer parent ledger changed",
    )
    parent_finalizer = payload.get("parent_finalizer")
    _require(
        isinstance(parent_finalizer, Mapping)
        and isinstance(parent_finalizer.get("path"), str)
        and _valid_digest(parent_finalizer.get("semantic_sha256"))
        and _valid_digest(parent_finalizer.get("file_sha256")),
        "V14 reserve source finalizer historical parent is invalid",
    )
    parent_finalizer_path = (root / parent_finalizer["path"]).resolve()
    parent_finalizer_payload = _read_json(parent_finalizer_path)
    _validate_parent_record(
        root=root,
        record=parent_finalizer,
        artifact=parent_finalizer_payload,
        path=parent_finalizer_path,
        role="parent_finalizer",
    )

    implementation = payload.get("implementation")
    files = (
        implementation.get("file_sha256")
        if isinstance(implementation, Mapping)
        else None
    )
    expected_files = {
        "admission_module",
        "finalizer_runner",
        "runtime_module",
        "runtime_wrapper",
        "selection_module",
        "source_lock_module",
    }
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and isinstance(files, Mapping)
        and set(files) == expected_files
        and all(_valid_digest(value) for value in files.values()),
        "V14 reserve source finalizer implementation ledger changed",
    )
    implementation_paths = {
        "admission_module": (
            root / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_admission_v14.py"
        ),
        "finalizer_runner": root / PARENT_RUNNER,
        "runtime_module": Path(__file__).resolve(),
        "runtime_wrapper": (
            root / "scripts/remote/"
            "finalize_deform360_causal_response_direct_depth_v14_reserve.py"
        ),
        "selection_module": (
            root / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_selection_v14.py"
        ),
        "source_lock_module": (
            root / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_source_lock.py"
        ),
    }
    _require(
        all(
            path.is_file() and file_sha256(path) == files[name]
            for name, path in implementation_paths.items()
        ),
        "V14 reserve source finalizer implementation changed",
    )
    trigger = payload.get("trigger")
    _require(
        isinstance(trigger, Mapping)
        and trigger.get("admitted_source_count") == 12
        and trigger.get("final_queue_rank") == 18
        and trigger.get("source_lock_artifact_created") is False
        and trigger.get("source_prediction_started") is False
        and trigger.get("source_outcome_read") is False,
        "V14 reserve source finalizer trigger changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("admission_dispositions_read") is True
        and boundary.get("prefix_or_future_object_response_read") is False
        and boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve source finalizer crossed its information boundary",
    )
    return payload


def expected_admission_prelock(
    custody: Mapping[str, Any],
    *,
    queue_rank: int,
) -> tuple[str, str]:
    """Return the expected prelock checksum and repository revision by rank."""

    ranks = custody["rank_contract"]
    original_start, original_end = ranks["original_admission_rank_range_inclusive"]
    reserve_start, reserve_end = ranks["reserve_admission_rank_range_inclusive"]
    if original_start <= queue_rank <= original_end:
        return (
            str(ranks["original_admission_prelock_config_sha256"]),
            str(ranks["original_repository_revision"]),
        )
    if reserve_start <= queue_rank <= reserve_end:
        return (
            str(ranks["reserve_admission_prelock_config_sha256"]),
            str(ranks["reserve_repository_revision"]),
        )
    raise ValueError("V14 admission report rank is outside composite custody")


def _argument_path(name: str) -> Path:
    indices = [index for index, value in enumerate(sys.argv) if value == name]
    _require(len(indices) == 1, f"{name} must appear exactly once")
    index = indices[0]
    _require(index + 1 < len(sys.argv), f"{name} lacks a value")
    return Path(sys.argv[index + 1]).resolve()


def _load_parent(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bayesian_phystwin_v14_reserve_source_finalizer_parent",
        path,
    )
    _require(
        spec is not None and spec.loader is not None,
        f"cannot load V14 reserve source finalizer parent: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_v14_reserve_source_finalizer(*, wrapper_path: str | Path) -> int:
    """Patch custody and checksum validation, then execute the frozen finalizer."""

    finalizer_path = _argument_path("--finalizer-protocol")
    method_path = _argument_path("--method-protocol")
    custody_path = _argument_path("--admission-prelock")
    wrapper = Path(wrapper_path).resolve()
    repository = wrapper.parents[2]
    finalizer = load_v14_reserve_source_finalizer_protocol(
        finalizer_path,
        repository=repository,
        composite_custody_path=custody_path,
    )
    custody = load_v14_composite_admission_custody(
        custody_path,
        repository=repository,
    )
    parent_path = (repository / PARENT_RUNNER).resolve()
    parent = _load_parent(parent_path)
    method = _read_json(method_path)
    _require(
        parent._canonical_config_sha256(method)
        == legacy_v14_method_config_sha256(method),
        "V14 source finalizer no longer exhibits the registered hash defect",
    )
    parent._canonical_config_sha256 = correct_v14_method_config_sha256
    _require(
        parent._canonical_config_sha256(method) == method["config_sha256"],
        "V14 reserve source finalizer did not repair the method checksum",
    )
    original_report_validator = validate_v14_admission_report
    original_failure_validator = parent._validate_window_failure

    def load_finalizer(path: str | Path) -> dict[str, Any]:
        _require(
            Path(path).resolve() == finalizer_path,
            "V14 reserve source finalizer path changed",
        )
        return finalizer

    def load_custody(path: str | Path) -> dict[str, Any]:
        _require(
            Path(path).resolve() == custody_path,
            "V14 composite admission custody path changed",
        )
        return custody

    def validate_report(path: str | Path) -> dict[str, Any]:
        report = original_report_validator(path)
        expected_prelock, expected_revision = expected_admission_prelock(
            custody,
            queue_rank=int(report["queue_rank"]),
        )
        _require(
            report["admission_prelock_config_sha256"] == expected_prelock
            and report["repository_revision"] == expected_revision,
            "V14 admission report differs from its rank-scoped custody",
        )
        return report

    def validate_window_failure(path: Path) -> dict[str, Any]:
        failure = original_failure_validator(path)
        _require(
            int(failure["queue_rank"])
            in custody["rank_contract"]["technical_window_disposition_ranks"],
            "V14 technical disposition rank differs from composite custody",
        )
        return failure

    parent.load_v14_source_finalizer_protocol = load_finalizer
    parent.load_v14_admission_prelock_protocol = load_custody
    parent.validate_v14_admission_report = validate_report
    parent._validate_window_failure = validate_window_failure
    return int(parent.main())


__all__ = [
    "CUSTODY_CONTRACT",
    "CUSTODY_KIND",
    "CUSTODY_NAMESPACE",
    "CUSTODY_PROTOCOL_ID",
    "FINALIZER_CONTRACT",
    "FINALIZER_KIND",
    "FINALIZER_NAMESPACE",
    "FINALIZER_PROTOCOL_ID",
    "expected_admission_prelock",
    "load_v14_composite_admission_custody",
    "load_v14_reserve_source_finalizer_protocol",
    "run_v14_reserve_source_finalizer",
]
