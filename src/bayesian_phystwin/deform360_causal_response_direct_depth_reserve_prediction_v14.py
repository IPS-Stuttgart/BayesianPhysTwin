"""Mixed-custody prediction runtime for the prospective V14 source panel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from . import deform360_causal_response_direct_depth_physical as physical_module
from .deform360_causal_response_direct_depth_admission_v14 import (
    ADMISSION_REPORT_FILENAME,
    validate_v14_admission_report,
)
from .deform360_causal_response_direct_depth_method_hash_runtime_v2 import (
    correct_v14_method_config_sha256,
)
from .deform360_causal_response_direct_depth_physical import (
    PHYSICAL_ARCHIVE_FILENAME,
    PHYSICAL_MANIFEST_FILENAME,
    load_v14_physical_prelock_protocol,
)
from .deform360_causal_response_direct_depth_prediction_v14 import (
    ACTUATOR_POSITION_FIELD,
    PREDICTION_FRAME_COUNT,
    PREFIX_FRAME_COUNT,
    TACTILE_AGGREGATION,
)
from .deform360_causal_response_direct_depth_reserve_physical_v14 import (
    load_v14_reserve_physical_prelock,
)
from .deform360_causal_response_direct_depth_reserve_source_finalizer_v14 import (
    expected_admission_prelock,
    load_v14_composite_admission_custody,
)
from .deform360_causal_response_direct_depth_source_lock import (
    validate_adaptive_direct_depth_source_lock_v14,
)
from .deform360_object_exclusion import file_sha256

PHYSICAL_CUSTODY_KIND = (
    "Deform360CausalResponseDirectDepthCompositePhysicalCustodyV14V1"
)
PHYSICAL_CUSTODY_CONTRACT = (
    "deform360-causal-response-direct-depth-composite-physical-custody-v14-v1"
)
PHYSICAL_CUSTODY_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-composite-physical-custody-v1"
)
PHYSICAL_CUSTODY_NAMESPACE = (
    b"deform360-causal-response-direct-depth-composite-physical-custody-v14-v1\0"
)
RUNTIME_KIND = "Deform360CausalResponseDirectDepthReservePredictionRuntimeV14V1"
RUNTIME_CONTRACT = (
    "deform360-causal-response-direct-depth-reserve-prediction-runtime-v14-v1"
)
RUNTIME_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-reserve-prediction-runtime-v1"
)
RUNTIME_NAMESPACE = (
    b"deform360-causal-response-direct-depth-reserve-prediction-runtime-v14-v1\0"
)
METHOD_PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-source"
PARENT_PREDICTION_RUNNER = (
    "scripts/remote/run_deform360_causal_response_direct_depth_v14_prediction.py"
)
PARENT_RUNTIME_BUILDER = (
    "scripts/remote/"
    "prepare_deform360_causal_response_direct_depth_v14_prediction_runtime.py"
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
            f"cannot read V14 reserve prediction artifact: {source}"
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


def load_v14_composite_physical_custody(
    path: str | Path,
    *,
    repository: str | Path,
) -> dict[str, Any]:
    """Validate the original and reserve physical prelocks and rank scopes."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == PHYSICAL_CUSTODY_KIND
        and payload.get("contract") == PHYSICAL_CUSTODY_CONTRACT
        and payload.get("protocol_id") == PHYSICAL_CUSTODY_PROTOCOL_ID
        and payload.get("status") == "locked_after_source_selection_before_prefix_scan"
        and payload.get("config_sha256")
        == _canonical_sha256(payload, namespace=PHYSICAL_CUSTODY_NAMESPACE),
        "V14 composite physical custody identity or checksum changed",
    )
    root = Path(repository).resolve()
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents)
        == {
            "original_physical_prelock",
            "original_physical_runtime",
            "reserve_physical_prelock",
            "reserve_physical_runtime",
        },
        "V14 composite physical custody parent ledger changed",
    )
    artifacts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for role, record in parents.items():
        _require(
            isinstance(record, Mapping)
            and isinstance(record.get("path"), str)
            and _valid_digest(record.get("semantic_sha256"))
            and _valid_digest(record.get("file_sha256")),
            f"V14 composite physical custody parent is invalid: {role}",
        )
        parent_path = (root / record["path"]).resolve()
        artifact = _read_json(parent_path)
        _require(
            artifact.get("config_sha256") == record["semantic_sha256"]
            and file_sha256(parent_path) == record["file_sha256"],
            f"V14 composite physical custody parent changed: {role}",
        )
        artifacts[role] = (parent_path, artifact)
    original_path, original = artifacts["original_physical_prelock"]
    reserve_path, reserve = artifacts["reserve_physical_prelock"]
    load_v14_physical_prelock_protocol(original_path)
    load_v14_reserve_physical_prelock(reserve_path)
    ranks = payload.get("rank_contract")
    _require(
        isinstance(ranks, Mapping)
        and ranks.get("original_selected_ranks") == [3, 6, 7, 8, 9, 10, 12, 14]
        and ranks.get("reserve_selected_ranks") == [15, 16, 17, 18]
        and ranks.get("original_physical_prelock_config_sha256")
        == original["config_sha256"]
        and ranks.get("reserve_physical_prelock_config_sha256")
        == reserve["config_sha256"]
        and ranks.get("original_repository_revision")
        == "ac416f49f3d5f464348c843c0f918b052ee54874"
        and ranks.get("reserve_repository_revision")
        == "45ff13ca258fc3544ed08cb90b2f02ec203333c7"
        and ranks.get("selected_source_count") == 12,
        "V14 composite physical custody rank contract changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("prefix_object_response_read") is False
        and boundary.get("future_object_observation_read") is False
        and boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 composite physical custody crossed its information boundary",
    )
    return payload


def expected_physical_prelock(
    custody: Mapping[str, Any],
    *,
    queue_rank: int,
    repository: str | Path,
) -> tuple[Path, str, str]:
    """Return the physical prelock path, checksum, and revision by rank."""

    root = Path(repository).resolve()
    ranks = custody["rank_contract"]
    parents = custody["parent_artifacts"]
    if queue_rank in ranks["original_selected_ranks"]:
        record = parents["original_physical_prelock"]
        return (
            (root / record["path"]).resolve(),
            str(ranks["original_physical_prelock_config_sha256"]),
            str(ranks["original_repository_revision"]),
        )
    if queue_rank in ranks["reserve_selected_ranks"]:
        record = parents["reserve_physical_prelock"]
        return (
            (root / record["path"]).resolve(),
            str(ranks["reserve_physical_prelock_config_sha256"]),
            str(ranks["reserve_repository_revision"]),
        )
    raise ValueError("V14 physical rank is outside composite custody")


def validate_v14_mixed_physical_artifacts(
    output_dir: str | Path,
    *,
    custody: Mapping[str, Any],
    repository: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one physical carrier with its rank-scoped prelock."""

    root = Path(output_dir).resolve()
    report = _read_json(root / PHYSICAL_MANIFEST_FILENAME)
    rank = int(report["queue_rank"])
    prelock_path, prelock_sha, revision = expected_physical_prelock(
        custody,
        queue_rank=rank,
        repository=repository,
    )
    original_loader = physical_module.load_v14_physical_prelock_protocol
    try:
        physical_module.load_v14_physical_prelock_protocol = (
            load_v14_reserve_physical_prelock
            if rank in custody["rank_contract"]["reserve_selected_ranks"]
            else load_v14_physical_prelock_protocol
        )
        manifest, arrays = physical_module.validate_v14_physical_artifacts(
            root,
            prelock_protocol_path=prelock_path,
        )
    finally:
        physical_module.load_v14_physical_prelock_protocol = original_loader
    _require(
        manifest["physical_prelock_config_sha256"] == prelock_sha
        and manifest["code_revision"] == revision,
        "V14 physical carrier differs from its rank-scoped custody",
    )
    return manifest, arrays


def _validate_implementation(
    implementation_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    expected = {
        "prediction_module",
        "prediction_runner",
        "preflight_module",
        "runtime_builder",
        "reserve_prediction_module",
        "reserve_prediction_runner",
        "reserve_runtime_builder",
    }
    normalized = {
        str(name): Path(path).resolve() for name, path in implementation_paths.items()
    }
    _require(
        set(normalized) == expected
        and all(path.is_file() for path in normalized.values()),
        "V14 reserve prediction implementation paths are incomplete",
    )
    return normalized


def build_v14_reserve_prediction_runtime(
    *,
    repository_revision: str,
    repository: str | Path,
    method_protocol_path: str | Path,
    source_lock_path: str | Path,
    admission_custody_path: str | Path,
    physical_custody_path: str | Path,
    admission_root: str | Path,
    physical_root: str | Path,
    implementation_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Build the outcome-blind mixed-custody runtime after source locking."""

    _require(
        len(repository_revision) == 40
        and all(character in "0123456789abcdef" for character in repository_revision),
        "V14 reserve prediction runtime revision is invalid",
    )
    repo = Path(repository).resolve()
    method_path = Path(method_protocol_path).resolve()
    method = _read_json(method_path)
    _require(
        method.get("protocol_id") == METHOD_PROTOCOL_ID
        and correct_v14_method_config_sha256(method) == method["config_sha256"],
        "V14 reserve prediction method changed",
    )
    source_path = Path(source_lock_path).resolve()
    source_lock = validate_adaptive_direct_depth_source_lock_v14(source_path)
    _require(
        method["config_sha256"] == source_lock.method_config_sha256,
        "V14 reserve prediction source lock belongs to another method",
    )
    admission_path = Path(admission_custody_path).resolve()
    admission_custody = load_v14_composite_admission_custody(
        admission_path,
        repository=repo,
    )
    physical_path = Path(physical_custody_path).resolve()
    physical_custody = load_v14_composite_physical_custody(
        physical_path,
        repository=repo,
    )
    normalized_paths = _validate_implementation(implementation_paths)
    locked_by_case = {case.case_hash: case for case in source_lock.cases}

    admitted: dict[str, tuple[dict[str, Any], Path]] = {}
    for directory in sorted(Path(admission_root).glob("rank-*")):
        if not (directory / ADMISSION_REPORT_FILENAME).is_file():
            continue
        report = validate_v14_admission_report(directory)
        if report["case_hash"] not in locked_by_case:
            continue
        prelock_sha, revision = expected_admission_prelock(
            admission_custody,
            queue_rank=int(report["queue_rank"]),
        )
        _require(
            report["status"] == "admitted"
            and report["admission_prelock_config_sha256"] == prelock_sha
            and report["repository_revision"] == revision
            and report["case_hash"] not in admitted,
            "V14 reserve prediction admission is rejected or custody-mismatched",
        )
        admitted[report["case_hash"]] = (report, directory)
    _require(
        set(admitted) == set(locked_by_case),
        "V14 reserve prediction runtime lacks an admitted source case",
    )

    physical_by_case: dict[str, tuple[dict[str, Any], Path]] = {}
    for directory in sorted(Path(physical_root).glob("rank-*")):
        if not (directory / PHYSICAL_MANIFEST_FILENAME).is_file():
            continue
        manifest, _ = validate_v14_mixed_physical_artifacts(
            directory,
            custody=physical_custody,
            repository=repo,
        )
        if manifest["case_hash"] not in locked_by_case:
            continue
        _require(
            manifest["case_hash"] not in physical_by_case,
            "V14 reserve prediction physical carrier is duplicated",
        )
        physical_by_case[manifest["case_hash"]] = (manifest, directory)
    _require(
        set(physical_by_case) == set(locked_by_case),
        "V14 reserve prediction runtime lacks a physical carrier",
    )

    cases: list[dict[str, Any]] = []
    for case_hash, locked in locked_by_case.items():
        admission_report, admission_dir = admitted[case_hash]
        physical_manifest, physical_dir = physical_by_case[case_hash]
        rank = int(admission_report["queue_rank"])
        admission_prelock, _ = expected_admission_prelock(
            admission_custody,
            queue_rank=rank,
        )
        _, physical_prelock, _ = expected_physical_prelock(
            physical_custody,
            queue_rank=rank,
            repository=repo,
        )
        _require(
            admission_report["object_hash"] == locked.object_hash
            and physical_manifest["object_hash"] == locked.object_hash
            and rank == int(physical_manifest["queue_rank"])
            and admission_report["physical_artifact_sha256"]
            == physical_manifest["artifact_sha256"],
            "V14 reserve prediction source components do not agree",
        )
        cases.append(
            {
                "queue_rank": rank,
                "case_hash": case_hash,
                "object_hash": locked.object_hash,
                "admission_prelock_config_sha256": admission_prelock,
                "admission_artifact_sha256": admission_report["artifact_sha256"],
                "admission_file_sha256": file_sha256(
                    admission_dir / ADMISSION_REPORT_FILENAME
                ),
                "physical_prelock_config_sha256": physical_prelock,
                "physical_artifact_sha256": physical_manifest["artifact_sha256"],
                "physical_manifest_file_sha256": file_sha256(
                    physical_dir / PHYSICAL_MANIFEST_FILENAME
                ),
                "physical_archive_file_sha256": file_sha256(
                    physical_dir / PHYSICAL_ARCHIVE_FILENAME
                ),
            }
        )
    cases.sort(key=lambda record: int(record["queue_rank"]))
    _require(
        len(cases) == 12 and len({int(record["queue_rank"]) for record in cases}) == 12,
        "V14 reserve prediction runtime case ranks are invalid",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RUNTIME_KIND,
        "contract": RUNTIME_CONTRACT,
        "protocol_id": RUNTIME_PROTOCOL_ID,
        "status": "locked_after_source_selection_before_prefix_scan",
        "parent_artifacts": {
            "method_protocol": {
                "semantic_sha256": method["config_sha256"],
                "file_sha256": file_sha256(method_path),
            },
            "source_lock": {
                "semantic_sha256": source_lock.artifact_sha256,
                "file_sha256": file_sha256(source_path),
            },
            "admission_prelock": {
                "semantic_sha256": admission_custody["config_sha256"],
                "file_sha256": file_sha256(admission_path),
            },
            "physical_prelock": {
                "semantic_sha256": physical_custody["config_sha256"],
                "file_sha256": file_sha256(physical_path),
            },
        },
        "implementation": {
            "parent_commit": repository_revision,
            "file_sha256": {
                name: file_sha256(path)
                for name, path in sorted(normalized_paths.items())
            },
        },
        "numerical_contract": {
            "prefix_frame_count": PREFIX_FRAME_COUNT,
            "prediction_frame_count": PREDICTION_FRAME_COUNT,
            "depth_scale_to_m": 0.001,
            "tactile_aggregation": TACTILE_AGGREGATION,
            "tactile_values_are_calibrated_probabilities": False,
            "actuator_position_field": ACTUATOR_POSITION_FIELD,
        },
        "cases": cases,
        "information_boundary": {
            "maximum_object_observation_frame": PREFIX_FRAME_COUNT - 1,
            "future_object_observation_read": False,
            "future_identity_or_metric_read": False,
            "source_outcome_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    payload["config_sha256"] = _canonical_sha256(
        payload,
        namespace=RUNTIME_NAMESPACE,
    )
    return payload


def load_v14_reserve_prediction_runtime(
    path: str | Path,
    *,
    repository: str | Path,
    method_protocol_path: str | Path,
    source_lock_path: str | Path,
    admission_custody_path: str | Path,
    physical_custody_path: str | Path,
) -> dict[str, Any]:
    """Validate the mixed-custody runtime without reading prefix responses."""

    source = Path(path).resolve()
    payload = _read_json(source)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RUNTIME_KIND
        and payload.get("contract") == RUNTIME_CONTRACT
        and payload.get("protocol_id") == RUNTIME_PROTOCOL_ID
        and payload.get("status") == "locked_after_source_selection_before_prefix_scan"
        and payload.get("config_sha256")
        == _canonical_sha256(payload, namespace=RUNTIME_NAMESPACE),
        "V14 reserve prediction runtime identity or checksum changed",
    )
    repo = Path(repository).resolve()
    method_path = Path(method_protocol_path).resolve()
    method = _read_json(method_path)
    source_path = Path(source_lock_path).resolve()
    source_lock = validate_adaptive_direct_depth_source_lock_v14(source_path)
    admission_path = Path(admission_custody_path).resolve()
    admission = load_v14_composite_admission_custody(
        admission_path,
        repository=repo,
    )
    physical_path = Path(physical_custody_path).resolve()
    physical = load_v14_composite_physical_custody(
        physical_path,
        repository=repo,
    )
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and parents.get("method_protocol", {}).get("semantic_sha256")
        == method.get("config_sha256")
        and correct_v14_method_config_sha256(method) == method.get("config_sha256")
        and parents["method_protocol"].get("file_sha256") == file_sha256(method_path)
        and parents.get("source_lock", {}).get("semantic_sha256")
        == source_lock.artifact_sha256
        and parents["source_lock"].get("file_sha256") == file_sha256(source_path)
        and parents.get("admission_prelock", {}).get("semantic_sha256")
        == admission["config_sha256"]
        and parents["admission_prelock"].get("file_sha256")
        == file_sha256(admission_path)
        and parents.get("physical_prelock", {}).get("semantic_sha256")
        == physical["config_sha256"]
        and parents["physical_prelock"].get("file_sha256")
        == file_sha256(physical_path),
        "V14 reserve prediction runtime parent changed",
    )
    implementation = payload.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and isinstance(implementation.get("file_sha256"), Mapping)
        and set(implementation["file_sha256"])
        == {
            "prediction_module",
            "prediction_runner",
            "preflight_module",
            "runtime_builder",
            "reserve_prediction_module",
            "reserve_prediction_runner",
            "reserve_runtime_builder",
        }
        and all(
            _valid_digest(value) for value in implementation["file_sha256"].values()
        ),
        "V14 reserve prediction runtime implementation changed",
    )
    implementation_paths = {
        "prediction_module": (
            repo / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_prediction_v14.py"
        ),
        "prediction_runner": repo / PARENT_PREDICTION_RUNNER,
        "preflight_module": (
            repo / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_preflight.py"
        ),
        "runtime_builder": repo / PARENT_RUNTIME_BUILDER,
        "reserve_prediction_module": Path(__file__).resolve(),
        "reserve_prediction_runner": (
            repo / "scripts/remote/"
            "run_deform360_causal_response_direct_depth_v14_reserve_prediction.py"
        ),
        "reserve_runtime_builder": (
            repo / "scripts/remote/"
            "prepare_deform360_causal_response_direct_depth_v14_reserve_prediction_runtime.py"
        ),
    }
    _require(
        all(
            path.is_file() and file_sha256(path) == implementation["file_sha256"][name]
            for name, path in implementation_paths.items()
        ),
        "V14 reserve prediction runtime implementation bytes changed",
    )
    numerical = payload.get("numerical_contract")
    _require(
        isinstance(numerical, Mapping)
        and numerical.get("prefix_frame_count") == PREFIX_FRAME_COUNT
        and numerical.get("prediction_frame_count") == PREDICTION_FRAME_COUNT
        and numerical.get("depth_scale_to_m") == 0.001
        and numerical.get("tactile_aggregation") == TACTILE_AGGREGATION
        and numerical.get("actuator_position_field") == ACTUATOR_POSITION_FIELD
        and numerical.get("tactile_values_are_calibrated_probabilities") is False,
        "V14 reserve prediction numerical contract changed",
    )
    locked_by_case = {case.case_hash: case for case in source_lock.cases}
    cases = payload.get("cases")
    _require(
        isinstance(cases, list)
        and len(cases) == 12
        and {record.get("case_hash") for record in cases} == set(locked_by_case)
        and len({int(record.get("queue_rank", 0)) for record in cases}) == 12
        and all(
            record.get("object_hash")
            == locked_by_case[record.get("case_hash")].object_hash
            and all(
                _valid_digest(record.get(key))
                for key in (
                    "case_hash",
                    "object_hash",
                    "admission_prelock_config_sha256",
                    "admission_artifact_sha256",
                    "admission_file_sha256",
                    "physical_prelock_config_sha256",
                    "physical_artifact_sha256",
                    "physical_manifest_file_sha256",
                    "physical_archive_file_sha256",
                )
            )
            for record in cases
        ),
        "V14 reserve prediction runtime case ledger changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("maximum_object_observation_frame") == PREFIX_FRAME_COUNT - 1
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 reserve prediction runtime crossed its information boundary",
    )
    return payload


def write_v14_reserve_prediction_runtime(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    repository: str | Path,
    method_protocol_path: str | Path,
    source_lock_path: str | Path,
    admission_custody_path: str | Path,
    physical_custody_path: str | Path,
) -> None:
    """Write and validate one mixed-custody prediction runtime."""

    output = Path(path)
    _require(not output.exists(), "V14 reserve prediction runtime already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_v14_reserve_prediction_runtime(
        output,
        repository=repository,
        method_protocol_path=method_protocol_path,
        source_lock_path=source_lock_path,
        admission_custody_path=admission_custody_path,
        physical_custody_path=physical_custody_path,
    )


def _load_parent(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bayesian_phystwin_v14_reserve_prediction_parent",
        path,
    )
    _require(
        spec is not None and spec.loader is not None,
        f"cannot load V14 reserve prediction parent: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_v14_reserve_prediction(*, wrapper_path: str | Path) -> int:
    """Validate mixed custody, patch loaders, and run the frozen estimator."""

    wrapper = Path(wrapper_path).resolve()
    repository = wrapper.parents[2]
    parent_path = (repository / PARENT_PREDICTION_RUNNER).resolve()
    parent = _load_parent(parent_path)
    original_args = parent._parse_args()
    runtime = load_v14_reserve_prediction_runtime(
        original_args.prediction_runtime.resolve(),
        repository=repository,
        method_protocol_path=original_args.method_protocol.resolve(),
        source_lock_path=original_args.source_lock.resolve(),
        admission_custody_path=original_args.admission_prelock.resolve(),
        physical_custody_path=original_args.physical_prelock.resolve(),
    )
    _require(
        file_sha256(wrapper)
        == runtime["implementation"]["file_sha256"]["reserve_prediction_runner"],
        "V14 reserve prediction wrapper changed",
    )
    physical_custody = load_v14_composite_physical_custody(
        original_args.physical_prelock.resolve(),
        repository=repository,
    )

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
            Path(prelock_protocol_path).resolve()
            == original_args.physical_prelock.resolve(),
            "V14 reserve prediction physical custody path changed",
        )
        return validate_v14_mixed_physical_artifacts(
            output_dir,
            custody=physical_custody,
            repository=repository,
        )

    parent._parse_args = lambda: original_args
    parent.load_v14_prediction_runtime = load_runtime
    parent.validate_v14_physical_artifacts = validate_physical
    return int(parent.main())


__all__ = [
    "PHYSICAL_CUSTODY_CONTRACT",
    "PHYSICAL_CUSTODY_KIND",
    "PHYSICAL_CUSTODY_NAMESPACE",
    "PHYSICAL_CUSTODY_PROTOCOL_ID",
    "RUNTIME_CONTRACT",
    "RUNTIME_KIND",
    "RUNTIME_NAMESPACE",
    "RUNTIME_PROTOCOL_ID",
    "build_v14_reserve_prediction_runtime",
    "expected_physical_prelock",
    "load_v14_composite_physical_custody",
    "load_v14_reserve_prediction_runtime",
    "run_v14_reserve_prediction",
    "validate_v14_mixed_physical_artifacts",
    "write_v14_reserve_prediction_runtime",
]
