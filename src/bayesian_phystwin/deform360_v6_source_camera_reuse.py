"""Outcome-blind camera reuse for the sealed public Deform360 v6 source batch.

The v6 source execution selected eight public cameras per object before opening
any source suffix.  Its v5 runner nevertheless converted one camera-local
metric-gauge failure into an object-wide fallback.  This module binds the
already-produced all-camera MotionCrafter and robot-metric products to the
frozen v5.2 per-camera recovery rule.  It performs no new provider inference
and reads no suffix, confirmation, or target outcome.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._canonical_contracts import genuine_integer, plain_json
from ._portable_contracts import (
    canonical_relative_posix_path,
    content_id,
    exact_revision,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
)
from .deform360_joint_sparse_camera_recovery_v5_2 import (
    AUDIT_INFORMATION_BOUNDARY,
    CAMERA_REUSE_POLICY,
    MAXIMUM_ADDITIONAL_CAMERAS,
    MINIMUM_PASSING_CAMERAS,
    rank_deform360_metric_camera_support_v5_2,
    summarize_deform360_metric_camera_support_v5_2,
    validate_deform360_joint_sparse_camera_audit_v5_2,
    validate_deform360_metric_camera_support_v5_2,
)
from .deform360_joint_sparse_geometric_common_v4 import (
    METRIC_BATCH_SCHEMA,
    METRIC_BATCH_SEMANTICS,
    METRIC_BATCH_VERSION,
    METRIC_PLAN_SCHEMA,
    METRIC_PLAN_SEMANTICS,
    METRIC_PLAN_VERSION,
)
from .deform360_joint_sparse_source_evidence_v5 import (
    validate_deform360_joint_sparse_source_prediction_batch_v5,
)
from .deform360_joint_sparse_source_runner_v5 import (
    build_deform360_joint_sparse_source_prediction_plan_v5,
    validate_deform360_joint_sparse_source_prediction_plan_v5,
    validate_deform360_joint_sparse_source_prediction_receipt_v5,
)
from .deform360_joint_sparse_source_runner_v5_2 import (
    CAMERA_REUSE_ARTIFACT_NAMES,
    validate_deform360_joint_sparse_source_prediction_plan_v5_2,
    validate_deform360_joint_sparse_source_prediction_receipt_v5_2,
)

AMENDMENT_SCHEMA: Final = "bayesian-phystwin.deform360-v6-source-camera-reuse-amendment"
AMENDMENT_VERSION: Final = 1
AMENDMENT_SEMANTICS: Final = (
    "additive-source-prefix-camera-failure-granularity-repair-v1"
)
AMENDMENT_ID: Final = "5cc43432eb509b98442d289ec884b30780ff26c76ab8654826d000bb4832e3b3"
EXECUTION_LOCK_ID: Final = (
    "76b74483790ace51d642889be2e3dbb22149e30f7919b5855a18066434e25189"
)
PREFLIGHT_SCHEMA: Final = "bayesian-phystwin.deform360-v6-source-camera-reuse-preflight"
PREFLIGHT_VERSION: Final = 1
PREFLIGHT_SEMANTICS: Final = (
    "metric-only-existing-camera-product-reuse-before-source-suffix-v1"
)
REUSE_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-v6-source-camera-reuse-receipt"
)
REUSE_RECEIPT_VERSION: Final = 1
REUSE_RECEIPT_SEMANTICS: Final = (
    "integrity-bound-existing-camera-product-reuse-before-source-suffix-v1"
)
EXECUTION_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-v6-source-camera-reuse-execution-receipt"
)
EXECUTION_RECEIPT_VERSION: Final = 1
EXECUTION_RECEIPT_SEMANTICS: Final = (
    "sealed-versioned-source-panel-without-source-suffix-access-v1"
)
TECHNICAL_FAILURE_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-v6-source-camera-reuse-technical-failure"
)
TECHNICAL_FAILURE_RECEIPT_VERSION: Final = 1
TECHNICAL_FAILURE_RECEIPT_SEMANTICS: Final = (
    "retained-one-shot-source-camera-reuse-failure-without-replacement-v1"
)
BASE_SOURCE_EXECUTION: Final = {
    "run_id": 31585420194,
    "run_attempt": 1,
    "artifact_id": 9137481740,
    "artifact_name": "deform360-v6-source-prediction-dual-runtime-31585420194-1",
    "artifact_digest_sha256": (
        "d811ff1ea4d6ad22a6e7476d2911602af1ee71f81339c0737e1cc60fd5883f9d"
    ),
    "head_sha": "6bb16bb307349c50535b1b368c60dfb4d5d17ab9",
}
BASE_EXECUTION_RECEIPT_ID: Final = (
    "cf3ebb9e69eb3c15051ba4ae39e2d0338ec244e0c49e587a277f7b36344c5f3d"
)
BASE_EXECUTION_RECEIPT_FILE_SHA256: Final = (
    "f1cd4ccfb8281a167718a30e5a6af1caaf740ba7a9d49081638efaabdeaf8441"
)

INFORMATION_BOUNDARY: Final = {
    **AUDIT_INFORMATION_BOUNDARY,
    "new_provider_inference_run": False,
    "existing_source_provider_products_reused": True,
}
CLAIM_BOUNDARY: Final = (
    "Public Deform360 source-prefix camera admission and archive reuse only. "
    "This artifact opens no source suffix, confirmation payload, or target "
    "outcome and establishes no prediction, calibration, Causal4D, safety, "
    "or state-of-the-art claim."
)
EXECUTION_INFORMATION_BOUNDARY: Final = {
    **INFORMATION_BOUNDARY,
    "base_prediction_batch_mutated": False,
    "new_prediction_batch_versioned": True,
    "source_suffix_opened": False,
}

_FILE_FIELDS = frozenset({"path", "sha256"})
_AMENDMENT_FIELDS = frozenset(
    {
        "all_camera_sources",
        "amendment_id",
        "base_execution_lock",
        "base_source_execution",
        "base_source_evidence",
        "claim_boundary",
        "information_boundary",
        "policy",
        "schema",
        "schema_version",
        "semantics",
        "status",
    }
)
_PREFLIGHT_FIELDS = frozenset(
    {
        "base_camera_audit_file_sha256",
        "base_camera_audit_id",
        "base_source_plan_file_sha256",
        "base_source_plan_id",
        "execution_lock_id",
        "information_boundary",
        "metric_batch_result_file_sha256",
        "metric_batch_result_id",
        "metric_prefix_plan_file_sha256",
        "metric_prefix_plan_id",
        "objects",
        "policy",
        "preflight_id",
        "schema",
        "schema_version",
        "semantics",
    }
)
_PREFLIGHT_OBJECT_FIELDS = frozenset(
    {
        "base_attempted_camera_ids",
        "base_passing_camera_ids",
        "candidate_metric_support",
        "object_id",
        "ranked_eligible_camera_ids",
        "recovery_required",
        "selected_reuse_camera_ids",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "base_source_plan_id",
        "combined_source_plan_id",
        "information_boundary",
        "metric_prefix_plan_id",
        "preflight_id",
        "receipt_id",
        "reused_cameras",
        "reused_camera_count",
        "schema",
        "schema_version",
        "semantics",
    }
)
_REUSED_CAMERA_FIELDS = frozenset(
    {
        "camera_id",
        "decoded_uniform_sha256",
        "metric_prefix_sha256",
        "object_id",
        "prediction_manifest_sha256",
    }
)
_METRIC_RESULT_FIELDS = frozenset(
    {
        "admission_id",
        "admitted_stream_count",
        "claim_boundary",
        "implementation_revision",
        "information_boundary",
        "jobs",
        "object_count",
        "plan_emitted",
        "plan_file",
        "production_result_id",
        "result_id",
        "schema",
        "schema_version",
        "semantics",
        "source_artifacts",
        "status",
        "support_negative_stream_count",
        "supported_object_count",
        "supported_stream_count",
        "technical_failure_stream_count",
    }
)
_EXECUTION_BASE_SOURCE_FIELDS = frozenset(
    {
        "artifact_digest_sha256",
        "artifact_id",
        "artifact_name",
        "head_sha",
        "run_attempt",
        "run_id",
    }
)
_AMENDMENT_BASE_SOURCE_EXECUTION_FIELDS = frozenset(
    {
        *_EXECUTION_BASE_SOURCE_FIELDS,
        "execution_receipt_file_sha256",
        "execution_receipt_id",
    }
)
_EXECUTION_RECEIPT_FIELDS = frozenset(
    {
        "amendment_id",
        "artifacts",
        "base_source_execution",
        "claim_authorized",
        "execution_lock_id",
        "independent_confirmation_authorized",
        "information_boundary",
        "prediction_batch_id",
        "prediction_record_count",
        "receipt_id",
        "runner_name",
        "schema",
        "schema_version",
        "semantics",
        "source_plan_id",
        "source_prediction_receipt_id",
        "source_prediction_seal_file_sha256",
        "source_revision",
        "source_suffix_access_authorized",
        "status",
        "workflow_run_attempt",
        "workflow_run_id",
    }
)
_TECHNICAL_FAILURE_RECEIPT_FIELDS = frozenset(
    {
        "amendment_id",
        "base_source_execution",
        "claim_authorized",
        "execution_lock_id",
        "exit_code",
        "independent_confirmation_authorized",
        "information_boundary",
        "receipt_id",
        "retained_artifacts",
        "runner_name",
        "schema",
        "schema_version",
        "semantics",
        "source_revision",
        "source_suffix_access_authorized",
        "status",
        "terminal_stage",
        "workflow_run_attempt",
        "workflow_run_id",
    }
)
EXECUTION_ARTIFACT_NAMES: Final = frozenset(
    {
        *CAMERA_REUSE_ARTIFACT_NAMES,
        "base_execution_receipt",
        "camera_reuse_lineage",
        "execution_lock",
        "source_plan",
        "source_prediction_batch",
        "source_prediction_receipt",
        "visual_production_result",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ValueError(f"{name} must be a canonical nonempty string")
    return value


def _normalize_base_source_execution(value: object) -> dict[str, Any]:
    base = _mapping(value, name="base source execution")
    require_exact_fields(
        base,
        expected=_EXECUTION_BASE_SOURCE_FIELDS,
        name="base source execution",
    )
    normalized = {
        "run_id": genuine_integer(base.get("run_id"), name="base run ID", minimum=1),
        "run_attempt": genuine_integer(
            base.get("run_attempt"), name="base run attempt", minimum=1
        ),
        "artifact_id": genuine_integer(
            base.get("artifact_id"), name="base artifact ID", minimum=1
        ),
        "artifact_name": _identifier(
            base.get("artifact_name"), name="base artifact name"
        ),
        "artifact_digest_sha256": sha256_digest(
            base.get("artifact_digest_sha256"), name="base artifact digest"
        ),
        "head_sha": exact_revision(base.get("head_sha"), name="base head SHA"),
    }
    _require(
        normalized == BASE_SOURCE_EXECUTION,
        "base source execution changed from the frozen successful v6 run",
    )
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _content_addressed(value: Mapping[str, Any], *, identity: str) -> None:
    declared = sha256_digest(value.get(identity), name=identity)
    descriptor = dict(value)
    descriptor.pop(identity)
    _require(content_id(descriptor) == declared, f"{identity} changed")


def _ordinary_root(path: str | Path, *, name: str) -> Path:
    requested = Path(path).absolute()
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        f"{name} must be an ordinary non-symlink directory",
    )
    return requested.resolve(strict=True)


def _verified_file(
    root: Path,
    value: object,
    *,
    name: str,
) -> Path:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    relative = canonical_relative_posix_path(record.get("path"), name=f"{name}.path")
    digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    requested = root / relative
    _require(
        requested.is_file()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        f"{name} must be an ordinary file",
    )
    resolved = requested.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes its root") from error
    _require(_sha256_file(resolved) == digest, f"{name} SHA-256 changed")
    return resolved


def _relative_record(path: Path, *, root: Path, name: str) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} is outside the results root") from error
    return {"path": relative.as_posix(), "sha256": _sha256_file(resolved)}


def _binding(value: object, *, name: str, id_field: str) -> dict[str, str]:
    record = _mapping(value, name=name)
    require_exact_fields(
        record,
        expected=frozenset({id_field, "file_sha256"}),
        name=name,
    )
    return {
        id_field: sha256_digest(record.get(id_field), name=f"{name}.{id_field}"),
        "file_sha256": sha256_digest(
            record.get("file_sha256"), name=f"{name}.file_sha256"
        ),
    }


def validate_deform360_v6_source_camera_reuse_amendment(
    value: object,
) -> dict[str, Any]:
    """Validate the locked source-only repair amendment."""

    amendment = _mapping(value, name="camera reuse amendment")
    require_exact_fields(
        amendment, expected=_AMENDMENT_FIELDS, name="camera reuse amendment"
    )
    _content_addressed(amendment, identity="amendment_id")
    _require(
        amendment.get("schema") == AMENDMENT_SCHEMA
        and amendment.get("schema_version") == AMENDMENT_VERSION
        and amendment.get("semantics") == AMENDMENT_SEMANTICS
        and amendment.get("amendment_id") == AMENDMENT_ID
        and amendment.get("status") == "locked-before-source-prefix-camera-audit",
        "camera reuse amendment contract changed",
    )
    base_lock = _binding(
        amendment.get("base_execution_lock"),
        name="base execution lock",
        id_field="execution_lock_id",
    )
    base_execution = _mapping(
        amendment.get("base_source_execution"), name="base source execution"
    )
    require_exact_fields(
        base_execution,
        expected=_AMENDMENT_BASE_SOURCE_EXECUTION_FIELDS,
        name="base source execution",
    )
    normalized_base_execution = _normalize_base_source_execution(
        {
            key: value
            for key, value in base_execution.items()
            if key in _EXECUTION_BASE_SOURCE_FIELDS
        }
    )
    _require(
        normalized_base_execution == BASE_SOURCE_EXECUTION
        and base_execution.get("execution_receipt_id") == BASE_EXECUTION_RECEIPT_ID
        and base_execution.get("execution_receipt_file_sha256")
        == BASE_EXECUTION_RECEIPT_FILE_SHA256,
        "base execution receipt binding changed",
    )
    source = _mapping(amendment.get("base_source_evidence"), name="base evidence")
    expected_source = {
        "source_plan": "plan_id",
        "prediction_batch": "prediction_batch_id",
        "prediction_receipt": "receipt_id",
    }
    _require(set(source) == set(expected_source), "base source evidence roster changed")
    for field, id_field in expected_source.items():
        _binding(source[field], name=field, id_field=id_field)
    all_camera = _mapping(
        amendment.get("all_camera_sources"), name="all-camera sources"
    )
    expected_all_camera = {
        "metric_prefix_plan": "plan_id",
        "metric_batch_result": "result_id",
        "visual_production": "result_id",
    }
    _require(
        set(all_camera) == set(expected_all_camera),
        "all-camera source roster changed",
    )
    for field, id_field in expected_all_camera.items():
        _binding(all_camera[field], name=field, id_field=id_field)
    _require(bool(base_lock), "base lock is missing")
    _require(amendment.get("policy") == CAMERA_REUSE_POLICY, "reuse policy changed")
    _require(
        amendment.get("information_boundary") == INFORMATION_BOUNDARY,
        "reuse amendment boundary changed",
    )
    _require(
        amendment.get("claim_boundary") == CLAIM_BOUNDARY,
        "reuse amendment claim boundary changed",
    )
    return cast(dict[str, Any], plain_json(amendment))


def validate_deform360_v6_source_camera_reuse_amendment_bindings(
    amendment: Mapping[str, Any],
    *,
    execution_lock_id: str,
    execution_lock_file_sha256: str,
    base_source_plan_id: str,
    base_source_plan_file_sha256: str,
    base_prediction_batch_id: str,
    base_prediction_batch_file_sha256: str,
    base_prediction_receipt_id: str,
    base_prediction_receipt_file_sha256: str,
    metric_prefix_plan_id: str,
    metric_prefix_plan_file_sha256: str,
    metric_batch_result_id: str,
    metric_batch_result_file_sha256: str,
    visual_production_result_id: str,
    visual_production_result_file_sha256: str,
) -> dict[str, Any]:
    """Require an amendment to name every exact reusable source artifact."""

    normalized = validate_deform360_v6_source_camera_reuse_amendment(amendment)
    expected_lock = {
        "execution_lock_id": sha256_digest(execution_lock_id, name="execution_lock_id"),
        "file_sha256": sha256_digest(
            execution_lock_file_sha256, name="execution_lock_file_sha256"
        ),
    }
    expected_source = {
        "source_plan": {
            "plan_id": sha256_digest(base_source_plan_id, name="base plan ID"),
            "file_sha256": sha256_digest(
                base_source_plan_file_sha256, name="base plan file SHA-256"
            ),
        },
        "prediction_batch": {
            "prediction_batch_id": sha256_digest(
                base_prediction_batch_id, name="base prediction batch ID"
            ),
            "file_sha256": sha256_digest(
                base_prediction_batch_file_sha256,
                name="base prediction batch file SHA-256",
            ),
        },
        "prediction_receipt": {
            "receipt_id": sha256_digest(
                base_prediction_receipt_id, name="base prediction receipt ID"
            ),
            "file_sha256": sha256_digest(
                base_prediction_receipt_file_sha256,
                name="base prediction receipt file SHA-256",
            ),
        },
    }
    expected_all_camera = {
        "metric_prefix_plan": {
            "plan_id": sha256_digest(metric_prefix_plan_id, name="metric plan ID"),
            "file_sha256": sha256_digest(
                metric_prefix_plan_file_sha256, name="metric plan file SHA-256"
            ),
        },
        "metric_batch_result": {
            "result_id": sha256_digest(metric_batch_result_id, name="metric result ID"),
            "file_sha256": sha256_digest(
                metric_batch_result_file_sha256,
                name="metric result file SHA-256",
            ),
        },
        "visual_production": {
            "result_id": sha256_digest(
                visual_production_result_id, name="visual production result ID"
            ),
            "file_sha256": sha256_digest(
                visual_production_result_file_sha256,
                name="visual production result file SHA-256",
            ),
        },
    }
    _require(
        normalized["base_execution_lock"] == expected_lock
        and normalized["base_source_evidence"] == expected_source
        and normalized["all_camera_sources"] == expected_all_camera,
        "camera reuse amendment binds another source execution",
    )
    return normalized


def _validate_metric_plan(value: object) -> dict[str, Any]:
    plan = _mapping(value, name="metric-prefix plan")
    _require(
        plan.get("schema") == METRIC_PLAN_SCHEMA
        and plan.get("schema_version") == METRIC_PLAN_VERSION
        and plan.get("semantics") == METRIC_PLAN_SEMANTICS,
        "metric-prefix plan contract changed",
    )
    _content_addressed(plan, identity="plan_id")
    seen_objects: set[str] = set()
    for raw_case in _sequence(plan.get("cases"), name="metric cases"):
        case = _mapping(raw_case, name="metric case")
        object_id = _identifier(case.get("object_id"), name="object_id")
        _require(object_id not in seen_objects, "metric plan repeats an object")
        seen_objects.add(object_id)
        cameras: set[str] = set()
        for raw_stream in _sequence(case.get("streams"), name="metric streams"):
            stream = _mapping(raw_stream, name="metric stream")
            camera = _identifier(stream.get("camera_id"), name="camera_id")
            _require(camera not in cameras, "metric plan repeats a camera")
            cameras.add(camera)
            for field in ("metric_prefix", "prediction_manifest"):
                record = _mapping(stream.get(field), name=field)
                canonical_relative_posix_path(record.get("path"), name=f"{field}.path")
                sha256_digest(record.get("sha256"), name=f"{field}.sha256")
    _require(len(seen_objects) == 10, "metric-prefix plan cohort changed")
    return cast(dict[str, Any], plain_json(plan))


def _validate_metric_result(
    value: object,
    *,
    metric_plan: Mapping[str, Any],
) -> dict[str, Any]:
    result = _mapping(value, name="metric-batch result")
    require_exact_fields(
        result, expected=_METRIC_RESULT_FIELDS, name="metric-batch result"
    )
    _content_addressed(result, identity="result_id")
    _require(
        result.get("schema") == METRIC_BATCH_SCHEMA
        and result.get("schema_version") == METRIC_BATCH_VERSION
        and result.get("semantics") == METRIC_BATCH_SEMANTICS
        and result.get("status") == "target-free-visible-streams-supported",
        "metric-batch result contract changed",
    )
    plan_file = _mapping(result.get("plan_file"), name="metric result plan file")
    _require(
        plan_file.get("path") == "metric-prefix-plan.json"
        and plan_file.get("sha256") is not None,
        "metric result plan binding changed",
    )
    sha256_digest(plan_file.get("sha256"), name="metric plan file SHA-256")
    _require(
        result.get("production_result_id")
        == metric_plan.get("visual_production_result_id")
        and result.get("plan_emitted") is True,
        "metric result source boundary changed",
    )
    _require(
        result.get("object_count") == 10
        and result.get("admitted_stream_count") == 324
        and result.get("supported_stream_count") == 313
        and result.get("support_negative_stream_count") == 11
        and result.get("technical_failure_stream_count") == 0
        and result.get("supported_object_count") == 10,
        "metric result accounting changed",
    )
    boundary = _mapping(result.get("information_boundary"), name="metric boundary")
    _require(
        boundary.get("confirmation_payloads_opened") is False
        and boundary.get("future_frames_used") is False
        and boundary.get("target_outcomes_used") is False,
        "metric result crosses the source boundary",
    )
    return cast(dict[str, Any], plain_json(result))


def _plan_objects(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(value.get("objects"), name="source objects"):
        row = _mapping(raw, name="source object")
        object_id = _identifier(row.get("object_id"), name="object_id")
        _require(object_id not in result, "source plan repeats an object")
        result[object_id] = row
    return result


def _metric_cases(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], value["cases"])
    }


def build_deform360_v6_source_camera_reuse_preflight(
    *,
    lock: Mapping[str, Any],
    base_source_plan: Mapping[str, Any],
    base_source_plan_file_sha256: str,
    base_camera_audit: Mapping[str, Any],
    base_camera_audit_file_sha256: str,
    metric_prefix_plan: Mapping[str, Any],
    metric_prefix_plan_file_sha256: str,
    metric_batch_result: Mapping[str, Any],
    metric_batch_result_file_sha256: str,
    metric_files_root: str | Path,
) -> dict[str, Any]:
    """Rank existing extra cameras using source-prefix robot geometry only."""

    base = validate_deform360_joint_sparse_source_prediction_plan_v5(
        base_source_plan, lock=lock
    )
    audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        base_camera_audit, lock=lock
    )
    _require(
        audit["base_source_plan_id"] == base["plan_id"],
        "base camera audit uses another source plan",
    )
    metric_plan = _validate_metric_plan(metric_prefix_plan)
    metric_result = _validate_metric_result(
        metric_batch_result, metric_plan=metric_plan
    )
    _require(
        _mapping(metric_result["plan_file"], name="plan file").get("sha256")
        == sha256_digest(
            metric_prefix_plan_file_sha256,
            name="metric_prefix_plan_file_sha256",
        ),
        "metric result binds another metric-prefix plan file",
    )
    root = _ordinary_root(metric_files_root, name="metric files root")
    base_objects = _plan_objects(base)
    audit_objects = _plan_objects(audit)
    cases = _metric_cases(metric_plan)
    _require(
        set(base_objects) == set(audit_objects) == set(cases),
        "camera-reuse source rosters differ",
    )
    objects: list[dict[str, Any]] = []
    for object_id in sorted(base_objects):
        source = base_objects[object_id]
        audited = audit_objects[object_id]
        attempted = tuple(cast(Sequence[str], audited["attempted_camera_ids"]))
        passing = tuple(cast(Sequence[str], audited["passing_camera_ids"]))
        reserved = set(cast(Sequence[str], source["reserved_endpoint_camera_ids"]))
        recovery_required = len(passing) < MINIMUM_PASSING_CAMERAS
        supports: list[dict[str, Any]] = []
        if recovery_required:
            for stream in cast(
                Sequence[Mapping[str, Any]], cases[object_id]["streams"]
            ):
                camera = cast(str, stream["camera_id"])
                if camera in reserved or camera in attempted:
                    continue
                metric_record = _mapping(stream["metric_prefix"], name="metric prefix")
                metric_path = _verified_file(
                    root,
                    {
                        "path": metric_record["path"],
                        "sha256": metric_record["sha256"],
                    },
                    name=f"{object_id}/{camera} metric prefix",
                )
                support = summarize_deform360_metric_camera_support_v5_2(
                    metric_path.parent
                )
                _require(
                    support["camera_id"] == camera
                    and support["metric_prefix_file_sha256"] == metric_record["sha256"],
                    "metric support differs from its frozen stream",
                )
                supports.append(support)
        supports.sort(key=lambda row: cast(str, row["camera_id"]))
        ranked = list(rank_deform360_metric_camera_support_v5_2(supports))
        objects.append(
            {
                "object_id": object_id,
                "base_attempted_camera_ids": list(attempted),
                "base_passing_camera_ids": list(passing),
                "recovery_required": recovery_required,
                "candidate_metric_support": supports,
                "ranked_eligible_camera_ids": ranked,
                "selected_reuse_camera_ids": ranked[:MAXIMUM_ADDITIONAL_CAMERAS],
            }
        )
    identity: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "schema_version": PREFLIGHT_VERSION,
        "semantics": PREFLIGHT_SEMANTICS,
        "execution_lock_id": lock["execution_lock_id"],
        "base_source_plan_id": base["plan_id"],
        "base_source_plan_file_sha256": sha256_digest(
            base_source_plan_file_sha256, name="base_source_plan_file_sha256"
        ),
        "base_camera_audit_id": audit["audit_id"],
        "base_camera_audit_file_sha256": sha256_digest(
            base_camera_audit_file_sha256, name="base_camera_audit_file_sha256"
        ),
        "metric_prefix_plan_id": metric_plan["plan_id"],
        "metric_prefix_plan_file_sha256": sha256_digest(
            metric_prefix_plan_file_sha256,
            name="metric_prefix_plan_file_sha256",
        ),
        "metric_batch_result_id": metric_result["result_id"],
        "metric_batch_result_file_sha256": sha256_digest(
            metric_batch_result_file_sha256,
            name="metric_batch_result_file_sha256",
        ),
        "policy": dict(CAMERA_REUSE_POLICY),
        "objects": objects,
        "information_boundary": dict(INFORMATION_BOUNDARY),
    }
    result = {**identity, "preflight_id": content_id(identity)}
    return validate_deform360_v6_source_camera_reuse_preflight(
        result,
        lock=lock,
        base_source_plan=base,
        base_camera_audit=audit,
        metric_prefix_plan=metric_plan,
    )


def validate_deform360_v6_source_camera_reuse_preflight(
    value: object,
    *,
    lock: Mapping[str, Any],
    base_source_plan: Mapping[str, Any],
    base_camera_audit: Mapping[str, Any],
    metric_prefix_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one deterministic camera-reuse preflight."""

    base = validate_deform360_joint_sparse_source_prediction_plan_v5(
        base_source_plan, lock=lock
    )
    audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        base_camera_audit, lock=lock
    )
    metric_plan = _validate_metric_plan(metric_prefix_plan)
    preflight = _mapping(value, name="camera reuse preflight")
    require_exact_fields(
        preflight, expected=_PREFLIGHT_FIELDS, name="camera reuse preflight"
    )
    _content_addressed(preflight, identity="preflight_id")
    _require(
        preflight.get("schema") == PREFLIGHT_SCHEMA
        and preflight.get("schema_version") == PREFLIGHT_VERSION
        and preflight.get("semantics") == PREFLIGHT_SEMANTICS,
        "camera reuse preflight contract changed",
    )
    _require(
        preflight.get("execution_lock_id") == lock.get("execution_lock_id")
        and preflight.get("base_source_plan_id") == base.get("plan_id")
        and preflight.get("base_camera_audit_id") == audit.get("audit_id")
        and preflight.get("metric_prefix_plan_id") == metric_plan.get("plan_id"),
        "camera reuse preflight lineage changed",
    )
    for field in (
        "base_source_plan_file_sha256",
        "base_camera_audit_file_sha256",
        "metric_prefix_plan_file_sha256",
        "metric_batch_result_id",
        "metric_batch_result_file_sha256",
    ):
        sha256_digest(preflight.get(field), name=field)
    _require(preflight.get("policy") == CAMERA_REUSE_POLICY, "reuse policy changed")
    _require(
        preflight.get("information_boundary") == INFORMATION_BOUNDARY,
        "reuse preflight boundary changed",
    )
    base_objects = _plan_objects(base)
    audit_objects = _plan_objects(audit)
    metric_cases = _metric_cases(metric_plan)
    seen: set[str] = set()
    for raw in _sequence(preflight.get("objects"), name="preflight objects"):
        row = _mapping(raw, name="preflight object")
        require_exact_fields(
            row, expected=_PREFLIGHT_OBJECT_FIELDS, name="preflight object"
        )
        object_id = _identifier(row.get("object_id"), name="object_id")
        _require(
            object_id in base_objects
            and object_id in audit_objects
            and object_id in metric_cases
            and object_id not in seen,
            "camera reuse preflight object roster changed",
        )
        seen.add(object_id)
        audited = audit_objects[object_id]
        attempted = tuple(cast(Sequence[str], row["base_attempted_camera_ids"]))
        passing = tuple(cast(Sequence[str], row["base_passing_camera_ids"]))
        _require(
            attempted == tuple(audited["attempted_camera_ids"])
            and passing == tuple(audited["passing_camera_ids"]),
            "camera reuse preflight changed the base audit",
        )
        required = len(passing) < MINIMUM_PASSING_CAMERAS
        _require(
            row.get("recovery_required") is required,
            "camera reuse trigger changed",
        )
        supports = [
            validate_deform360_metric_camera_support_v5_2(item)
            for item in _sequence(
                row.get("candidate_metric_support"), name="candidate support"
            )
        ]
        support_ids = [cast(str, item["camera_id"]) for item in supports]
        _require(
            support_ids == sorted(set(support_ids)),
            "camera reuse support roster is not unique and sorted",
        )
        available = {
            cast(str, stream["camera_id"])
            for stream in cast(
                Sequence[Mapping[str, Any]], metric_cases[object_id]["streams"]
            )
        }
        reserved = set(
            cast(Sequence[str], base_objects[object_id]["reserved_endpoint_camera_ids"])
        )
        expected = available - reserved - set(attempted) if required else set()
        _require(
            set(support_ids) == expected,
            "camera reuse candidate set changed",
        )
        ranked = list(rank_deform360_metric_camera_support_v5_2(supports))
        _require(
            row.get("ranked_eligible_camera_ids") == ranked
            and row.get("selected_reuse_camera_ids")
            == ranked[:MAXIMUM_ADDITIONAL_CAMERAS],
            "camera reuse ranking changed",
        )
    _require(set(seen) == set(base_objects), "camera reuse cohort changed")
    return cast(dict[str, Any], plain_json(preflight))


def _prediction_archive(
    *,
    prediction_root: Path,
    stream: Mapping[str, Any],
    object_id: str,
    camera_id: str,
) -> tuple[Path, str]:
    manifest_record = _mapping(stream.get("prediction_manifest"), name="manifest")
    manifest_path = _verified_file(
        prediction_root,
        {
            "path": manifest_record["path"],
            "sha256": manifest_record["sha256"],
        },
        name=f"{object_id}/{camera_id} prediction manifest",
    )
    manifest = load_strict_json_object(manifest_path, label="prediction manifest")
    _require(manifest.get("format_version") == 1, "prediction format changed")
    archive_name = canonical_relative_posix_path(
        manifest.get("disjoint_baseline"), name="disjoint_baseline"
    )
    archive = (manifest_path.parent / archive_name).resolve(strict=True)
    _require(
        archive.is_file()
        and not archive.is_symlink()
        and archive.parent == manifest_path.parent,
        "disjoint baseline is missing or escaped its view",
    )
    integrity = _mapping(manifest.get("artifact_integrity"), name="integrity")
    members = _sequence(integrity.get("members"), name="integrity members")
    matched = [
        _mapping(member, name="integrity member")
        for member in members
        if _mapping(member, name="integrity member").get("path") == archive_name
    ]
    _require(len(matched) == 1, "disjoint baseline is not integrity bound")
    member = matched[0]
    digest = _sha256_file(archive)
    _require(
        member.get("sha256") == digest
        and member.get("bytes") == archive.stat().st_size,
        "disjoint baseline integrity changed",
    )
    return archive, cast(str, manifest_record["sha256"])


def build_deform360_v6_source_camera_reuse_plan(
    *,
    lock: Mapping[str, Any],
    base_source_plan: Mapping[str, Any],
    base_camera_audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    metric_prefix_plan: Mapping[str, Any],
    results_root: str | Path,
    prediction_root: str | Path,
    metric_files_root: str | Path,
    implementation_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append selected existing cameras and bind every reused archive."""

    base = validate_deform360_joint_sparse_source_prediction_plan_v5(
        base_source_plan, lock=lock
    )
    metric_plan = _validate_metric_plan(metric_prefix_plan)
    normalized_preflight = validate_deform360_v6_source_camera_reuse_preflight(
        preflight,
        lock=lock,
        base_source_plan=base,
        base_camera_audit=base_camera_audit,
        metric_prefix_plan=metric_plan,
    )
    result_root = _ordinary_root(results_root, name="results root")
    predictions = _ordinary_root(prediction_root, name="prediction root")
    metrics = _ordinary_root(metric_files_root, name="metric files root")
    for child, name in ((predictions, "prediction root"), (metrics, "metric root")):
        try:
            child.relative_to(result_root)
        except ValueError as error:
            raise ValueError(f"{name} is outside the results root") from error
    preflight_objects = _plan_objects(normalized_preflight)
    cases = _metric_cases(metric_plan)
    streams = {
        (object_id, cast(str, stream["camera_id"])): stream
        for object_id, case in cases.items()
        for stream in cast(Sequence[Mapping[str, Any]], case["streams"])
    }
    attempted: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    for base_object in cast(Sequence[Mapping[str, Any]], base["objects"]):
        object_id = cast(str, base_object["object_id"])
        windows = [
            cast(dict[str, Any], plain_json(window))
            for window in cast(
                Sequence[Mapping[str, Any]], base_object["visual_windows"]
            )
        ]
        selected = cast(
            Sequence[str], preflight_objects[object_id]["selected_reuse_camera_ids"]
        )
        for camera in selected:
            stream = streams[(object_id, camera)]
            archive, manifest_digest = _prediction_archive(
                prediction_root=predictions,
                stream=stream,
                object_id=object_id,
                camera_id=camera,
            )
            metric_record = _mapping(stream["metric_prefix"], name="metric prefix")
            metric = _verified_file(
                metrics,
                {
                    "path": metric_record["path"],
                    "sha256": metric_record["sha256"],
                },
                name=f"{object_id}/{camera} metric prefix",
            )
            decoded_record = _relative_record(
                archive, root=result_root, name="decoded uniform"
            )
            metric_output = _relative_record(
                metric, root=result_root, name="metric prefix"
            )
            windows.append(
                {
                    "camera_id": camera,
                    "decoded_uniform": decoded_record,
                    "metric_prefix": metric_output,
                }
            )
            reused.append(
                {
                    "object_id": object_id,
                    "camera_id": camera,
                    "prediction_manifest_sha256": manifest_digest,
                    "decoded_uniform_sha256": decoded_record["sha256"],
                    "metric_prefix_sha256": metric_output["sha256"],
                }
            )
        windows.sort(key=lambda row: cast(str, row["camera_id"]))
        _require(
            len(windows) == len({cast(str, row["camera_id"]) for row in windows}),
            "combined camera plan repeats a camera",
        )
        attempted.append(
            {
                **{
                    key: plain_json(item)
                    for key, item in base_object.items()
                    if key != "visual_windows"
                },
                "visual_windows": windows,
            }
        )
    combined = build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision=implementation_revision,
        objects=attempted,
    )
    reused.sort(
        key=lambda row: (cast(str, row["object_id"]), cast(str, row["camera_id"]))
    )
    identity: dict[str, Any] = {
        "schema": REUSE_RECEIPT_SCHEMA,
        "schema_version": REUSE_RECEIPT_VERSION,
        "semantics": REUSE_RECEIPT_SEMANTICS,
        "base_source_plan_id": base["plan_id"],
        "preflight_id": normalized_preflight["preflight_id"],
        "metric_prefix_plan_id": metric_plan["plan_id"],
        "combined_source_plan_id": combined["plan_id"],
        "reused_camera_count": len(reused),
        "reused_cameras": reused,
        "information_boundary": dict(INFORMATION_BOUNDARY),
    }
    receipt = {**identity, "receipt_id": content_id(identity)}
    return combined, validate_deform360_v6_source_camera_reuse_receipt(receipt)


def validate_deform360_v6_source_camera_reuse_receipt(
    value: object,
) -> dict[str, Any]:
    """Validate the content-addressed existing-product reuse receipt."""

    receipt = _mapping(value, name="camera reuse receipt")
    require_exact_fields(receipt, expected=_RECEIPT_FIELDS, name="reuse receipt")
    _content_addressed(receipt, identity="receipt_id")
    _require(
        receipt.get("schema") == REUSE_RECEIPT_SCHEMA
        and receipt.get("schema_version") == REUSE_RECEIPT_VERSION
        and receipt.get("semantics") == REUSE_RECEIPT_SEMANTICS,
        "camera reuse receipt contract changed",
    )
    for field in (
        "base_source_plan_id",
        "preflight_id",
        "metric_prefix_plan_id",
        "combined_source_plan_id",
    ):
        sha256_digest(receipt.get(field), name=field)
    _require(
        receipt.get("amendment_id") == AMENDMENT_ID
        and receipt.get("execution_lock_id") == EXECUTION_LOCK_ID,
        "camera reuse execution lock identity changed",
    )
    rows = _sequence(receipt.get("reused_cameras"), name="reused cameras")
    normalized: list[tuple[str, str]] = []
    for raw in rows:
        row = _mapping(raw, name="reused camera")
        require_exact_fields(row, expected=_REUSED_CAMERA_FIELDS, name="reused camera")
        key = (
            _identifier(row.get("object_id"), name="object_id"),
            _identifier(row.get("camera_id"), name="camera_id"),
        )
        normalized.append(key)
        for field in (
            "prediction_manifest_sha256",
            "decoded_uniform_sha256",
            "metric_prefix_sha256",
        ):
            sha256_digest(row.get(field), name=field)
    _require(
        normalized == sorted(set(normalized))
        and receipt.get("reused_camera_count") == len(rows),
        "camera reuse receipt accounting changed",
    )
    _require(
        receipt.get("information_boundary") == INFORMATION_BOUNDARY,
        "camera reuse receipt boundary changed",
    )
    return cast(dict[str, Any], plain_json(receipt))


def build_deform360_v6_source_camera_reuse_lineage(
    *,
    lock: Mapping[str, Any],
    execution_lock_file_sha256: str,
    amendment: Mapping[str, Any],
    amendment_file_sha256: str,
    base_source_plan: Mapping[str, Any],
    base_source_plan_file_sha256: str,
    base_prediction_batch: Mapping[str, Any],
    base_prediction_batch_file_sha256: str,
    base_prediction_receipt: Mapping[str, Any],
    base_prediction_receipt_file_sha256: str,
    base_camera_audit: Mapping[str, Any],
    base_camera_audit_file_sha256: str,
    preflight: Mapping[str, Any],
    preflight_file_sha256: str,
    reuse_receipt: Mapping[str, Any],
    reuse_receipt_file_sha256: str,
    combined_plan: Mapping[str, Any],
    combined_plan_file_sha256: str,
    final_camera_audit: Mapping[str, Any],
    final_camera_audit_file_sha256: str,
    metric_prefix_plan: Mapping[str, Any],
    metric_prefix_plan_file_sha256: str,
    metric_batch_result: Mapping[str, Any],
    metric_batch_result_file_sha256: str,
) -> dict[str, Any]:
    """Bind the reuse path to the original sealed batch and final audit."""

    normalized_amendment = validate_deform360_v6_source_camera_reuse_amendment(
        amendment
    )
    base = validate_deform360_joint_sparse_source_prediction_plan_v5(
        base_source_plan, lock=lock
    )
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        base_prediction_batch, lock
    )
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5(
        base_prediction_receipt,
        lock=lock,
        plan=base,
        prediction_batch=batch,
        prediction_batch_file_sha256=base_prediction_batch_file_sha256,
    )
    base_audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        base_camera_audit, lock=lock
    )
    metric_plan = _validate_metric_plan(metric_prefix_plan)
    metric_result = _validate_metric_result(
        metric_batch_result, metric_plan=metric_plan
    )
    normalized_preflight = validate_deform360_v6_source_camera_reuse_preflight(
        preflight,
        lock=lock,
        base_source_plan=base,
        base_camera_audit=base_audit,
        metric_prefix_plan=metric_plan,
    )
    reuse = validate_deform360_v6_source_camera_reuse_receipt(reuse_receipt)
    combined = validate_deform360_joint_sparse_source_prediction_plan_v5(
        combined_plan, lock=lock
    )
    final_audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        final_camera_audit, lock=lock
    )
    _require(
        base_audit["base_source_plan_id"] == base["plan_id"]
        and reuse["base_source_plan_id"] == base["plan_id"]
        and reuse["preflight_id"] == normalized_preflight["preflight_id"]
        and reuse["combined_source_plan_id"] == combined["plan_id"]
        and final_audit["base_source_plan_id"] == combined["plan_id"],
        "camera reuse semantic lineage changed",
    )
    source_evidence = cast(
        Mapping[str, Any], normalized_amendment["base_source_evidence"]
    )
    _require(
        source_evidence["source_plan"]
        == {"plan_id": base["plan_id"], "file_sha256": base_source_plan_file_sha256}
        and source_evidence["prediction_batch"]
        == {
            "prediction_batch_id": batch["prediction_batch_id"],
            "file_sha256": base_prediction_batch_file_sha256,
        }
        and source_evidence["prediction_receipt"]
        == {
            "receipt_id": receipt["receipt_id"],
            "file_sha256": base_prediction_receipt_file_sha256,
        },
        "camera reuse amendment binds another source batch",
    )
    all_camera = cast(Mapping[str, Any], normalized_amendment["all_camera_sources"])
    _require(
        all_camera["metric_prefix_plan"]
        == {
            "plan_id": metric_plan["plan_id"],
            "file_sha256": metric_prefix_plan_file_sha256,
        }
        and all_camera["metric_batch_result"]
        == {
            "result_id": metric_result["result_id"],
            "file_sha256": metric_batch_result_file_sha256,
        },
        "camera reuse amendment binds another metric source",
    )
    validate_deform360_v6_source_camera_reuse_amendment_bindings(
        normalized_amendment,
        execution_lock_id=cast(str, lock["execution_lock_id"]),
        execution_lock_file_sha256=execution_lock_file_sha256,
        base_source_plan_id=cast(str, base["plan_id"]),
        base_source_plan_file_sha256=base_source_plan_file_sha256,
        base_prediction_batch_id=cast(str, batch["prediction_batch_id"]),
        base_prediction_batch_file_sha256=base_prediction_batch_file_sha256,
        base_prediction_receipt_id=cast(str, receipt["receipt_id"]),
        base_prediction_receipt_file_sha256=base_prediction_receipt_file_sha256,
        metric_prefix_plan_id=cast(str, metric_plan["plan_id"]),
        metric_prefix_plan_file_sha256=metric_prefix_plan_file_sha256,
        metric_batch_result_id=cast(str, metric_result["result_id"]),
        metric_batch_result_file_sha256=metric_batch_result_file_sha256,
        visual_production_result_id=cast(
            str, metric_plan["visual_production_result_id"]
        ),
        visual_production_result_file_sha256=cast(
            str,
            _mapping(metric_result["source_artifacts"], name="metric source artifacts")[
                "visual-production-result.json"
            ],
        ),
    )
    paths = {
        "amendment": amendment_file_sha256,
        "base_camera_audit": base_camera_audit_file_sha256,
        "base_prediction_batch": base_prediction_batch_file_sha256,
        "base_prediction_receipt": base_prediction_receipt_file_sha256,
        "base_source_plan": base_source_plan_file_sha256,
        "camera_reuse_preflight": preflight_file_sha256,
        "camera_reuse_receipt": reuse_receipt_file_sha256,
        "combined_camera_audit_plan": combined_plan_file_sha256,
        "final_camera_audit": final_camera_audit_file_sha256,
        "metric_batch_result": metric_batch_result_file_sha256,
        "metric_prefix_plan": metric_prefix_plan_file_sha256,
    }
    identities = {
        "amendment": normalized_amendment["amendment_id"],
        "base_camera_audit": base_audit["audit_id"],
        "base_prediction_batch": batch["prediction_batch_id"],
        "base_prediction_receipt": receipt["receipt_id"],
        "base_source_plan": base["plan_id"],
        "camera_reuse_preflight": normalized_preflight["preflight_id"],
        "camera_reuse_receipt": reuse["receipt_id"],
        "combined_camera_audit_plan": combined["plan_id"],
        "final_camera_audit": final_audit["audit_id"],
        "metric_batch_result": metric_result["result_id"],
        "metric_prefix_plan": metric_plan["plan_id"],
    }
    _require(
        set(paths) == set(identities) == CAMERA_REUSE_ARTIFACT_NAMES,
        "camera reuse artifact roster changed",
    )
    return {
        "camera_recovery": {
            "artifact_ids": dict(identities),
            "source_artifacts": dict(
                source_artifact_mapping(paths, name="camera reuse source artifacts")
            ),
            "policy": dict(CAMERA_REUSE_POLICY),
            "base_prediction_batch_preserved": True,
        }
    }


def build_deform360_v6_source_camera_reuse_execution_receipt(
    *,
    amendment: Mapping[str, Any],
    lock: Mapping[str, Any],
    source_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    base_source_execution: Mapping[str, Any],
    artifact_file_sha256: Mapping[str, str],
    source_plan: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    source_prediction_receipt: Mapping[str, Any],
    source_prediction_seal_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Seal one target-closed execution of the camera-reuse source panel."""

    normalized_amendment = validate_deform360_v6_source_camera_reuse_amendment(
        amendment
    )
    revision = exact_revision(source_revision, name="source_revision")
    normalized_plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        source_plan, lock=lock
    )
    lock_id = sha256_digest(lock.get("execution_lock_id"), name="execution lock ID")
    _require(
        lock_id == normalized_amendment["base_execution_lock"]["execution_lock_id"],
        "camera reuse execution lock changed from the amendment",
    )
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch, lock
    )
    prediction_batch_digest = sha256_digest(
        artifact_file_sha256.get("source_prediction_batch"),
        name="source prediction batch file SHA-256",
    )
    panel_receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5_2(
        source_prediction_receipt,
        lock=lock,
        plan=normalized_plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=prediction_batch_digest,
    )
    _require(
        normalized_plan["implementation_revision"]
        == batch["implementation_revision"]
        == panel_receipt["implementation_revision"]
        == revision,
        "camera reuse execution revision changed",
    )
    normalized_base = _normalize_base_source_execution(base_source_execution)
    artifacts = source_artifact_mapping(
        artifact_file_sha256, name="camera reuse execution artifacts"
    )
    _require(
        set(artifacts) == EXECUTION_ARTIFACT_NAMES,
        "camera reuse execution artifact roster changed",
    )
    _require(
        artifacts["base_execution_receipt"] == BASE_EXECUTION_RECEIPT_FILE_SHA256,
        "base execution receipt changed",
    )
    amendment_sources = _mapping(
        normalized_amendment["all_camera_sources"], name="all-camera sources"
    )
    _require(
        artifacts["execution_lock"]
        == normalized_amendment["base_execution_lock"]["file_sha256"]
        and artifacts["visual_production_result"]
        == amendment_sources["visual_production"]["file_sha256"],
        "camera reuse execution source artifacts changed",
    )
    plan_recovery = _mapping(
        normalized_plan.get("camera_recovery"), name="source plan camera recovery"
    )
    lineage_artifacts = source_artifact_mapping(
        _mapping(
            plan_recovery.get("source_artifacts"),
            name="source plan camera-recovery artifacts",
        ),
        name="source plan camera-recovery artifacts",
    )
    _require(
        set(lineage_artifacts) == CAMERA_REUSE_ARTIFACT_NAMES
        and all(
            artifacts[name] == digest for name, digest in lineage_artifacts.items()
        ),
        "execution artifacts do not match the frozen source-plan lineage",
    )
    seals = source_artifact_mapping(
        _mapping(
            panel_receipt["source_prediction_seal_file_sha256"],
            name="source prediction seal digests",
        ),
        name="source prediction seal digests",
    )
    observed_seals = source_artifact_mapping(
        source_prediction_seal_file_sha256,
        name="observed source prediction seal digests",
    )
    _require(
        len(seals) == 100 and observed_seals == seals,
        "camera reuse execution must bind the 100 observed source seals",
    )
    identity: dict[str, Any] = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "schema_version": EXECUTION_RECEIPT_VERSION,
        "semantics": EXECUTION_RECEIPT_SEMANTICS,
        "status": "source-camera-reuse-predictions-sealed",
        "amendment_id": normalized_amendment["amendment_id"],
        "execution_lock_id": lock["execution_lock_id"],
        "source_revision": revision,
        "runner_name": _identifier(runner_name, name="runner name"),
        "workflow_run_id": genuine_integer(workflow_run_id, name="workflow run ID"),
        "workflow_run_attempt": genuine_integer(
            workflow_run_attempt, name="workflow run attempt"
        ),
        "base_source_execution": normalized_base,
        "source_plan_id": normalized_plan["plan_id"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "source_prediction_receipt_id": panel_receipt["receipt_id"],
        "prediction_record_count": panel_receipt["prediction_record_count"],
        "source_prediction_seal_file_sha256": dict(seals),
        "artifacts": dict(artifacts),
        "information_boundary": dict(EXECUTION_INFORMATION_BOUNDARY),
        "source_suffix_access_authorized": False,
        "independent_confirmation_authorized": False,
        "claim_authorized": False,
    }
    result = {**identity, "receipt_id": content_id(identity)}
    return validate_deform360_v6_source_camera_reuse_execution_receipt(result)


def validate_deform360_v6_source_camera_reuse_execution_receipt(
    value: object,
) -> dict[str, Any]:
    """Validate the compact receipt without opening source outcomes."""

    receipt = _mapping(value, name="camera reuse execution receipt")
    require_exact_fields(
        receipt,
        expected=_EXECUTION_RECEIPT_FIELDS,
        name="camera reuse execution receipt",
    )
    _content_addressed(receipt, identity="receipt_id")
    _require(
        receipt.get("schema") == EXECUTION_RECEIPT_SCHEMA
        and receipt.get("schema_version") == EXECUTION_RECEIPT_VERSION
        and receipt.get("semantics") == EXECUTION_RECEIPT_SEMANTICS
        and receipt.get("status") == "source-camera-reuse-predictions-sealed",
        "camera reuse execution receipt contract changed",
    )
    for field in (
        "amendment_id",
        "execution_lock_id",
        "source_plan_id",
        "prediction_batch_id",
        "source_prediction_receipt_id",
    ):
        sha256_digest(receipt.get(field), name=field)
    exact_revision(receipt.get("source_revision"), name="source_revision")
    _identifier(receipt.get("runner_name"), name="runner name")
    _require(
        genuine_integer(receipt.get("workflow_run_id"), name="workflow run ID") > 0
        and genuine_integer(
            receipt.get("workflow_run_attempt"), name="workflow run attempt"
        )
        > 0,
        "workflow identifiers must be positive",
    )
    base = _mapping(receipt.get("base_source_execution"), name="base execution")
    require_exact_fields(
        base, expected=_EXECUTION_BASE_SOURCE_FIELDS, name="base execution"
    )
    _require(
        genuine_integer(base.get("run_id"), name="base run ID") > 0
        and genuine_integer(base.get("run_attempt"), name="base run attempt") > 0
        and genuine_integer(base.get("artifact_id"), name="base artifact ID") > 0,
        "base execution identifiers must be positive",
    )
    _identifier(base.get("artifact_name"), name="base artifact name")
    sha256_digest(base.get("artifact_digest_sha256"), name="base artifact digest")
    exact_revision(base.get("head_sha"), name="base head SHA")
    artifacts = source_artifact_mapping(
        _mapping(receipt.get("artifacts"), name="execution artifacts"),
        name="execution artifacts",
    )
    seals = source_artifact_mapping(
        _mapping(
            receipt.get("source_prediction_seal_file_sha256"),
            name="source prediction seal digests",
        ),
        name="source prediction seal digests",
    )
    _require(
        set(artifacts) == EXECUTION_ARTIFACT_NAMES
        and dict(base) == BASE_SOURCE_EXECUTION
        and artifacts["base_execution_receipt"] == BASE_EXECUTION_RECEIPT_FILE_SHA256
        and len(seals) == 100
        and receipt.get("prediction_record_count") == 100,
        "camera reuse execution accounting changed",
    )
    _require(
        receipt.get("information_boundary") == EXECUTION_INFORMATION_BOUNDARY
        and receipt.get("source_suffix_access_authorized") is False
        and receipt.get("independent_confirmation_authorized") is False
        and receipt.get("claim_authorized") is False,
        "camera reuse execution boundary changed",
    )
    return cast(dict[str, Any], plain_json(receipt))


def build_deform360_v6_source_camera_reuse_technical_failure_receipt(
    *,
    amendment: Mapping[str, Any],
    lock: Mapping[str, Any],
    source_revision: str,
    runner_name: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    base_source_execution: Mapping[str, Any],
    terminal_stage: str,
    exit_code: int,
    retained_artifact_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Retain one failed source-only execution without authorizing replacement."""

    normalized_amendment = validate_deform360_v6_source_camera_reuse_amendment(
        amendment
    )
    lock_id = sha256_digest(lock.get("execution_lock_id"), name="execution lock ID")
    _require(
        lock_id == normalized_amendment["base_execution_lock"]["execution_lock_id"],
        "camera reuse execution lock changed from the amendment",
    )
    normalized_base = _normalize_base_source_execution(base_source_execution)
    retained = source_artifact_mapping(
        retained_artifact_file_sha256,
        name="retained technical-failure artifacts",
        allow_empty=True,
    )
    code = genuine_integer(exit_code, name="technical-failure exit code", minimum=1)
    identity: dict[str, Any] = {
        "schema": TECHNICAL_FAILURE_RECEIPT_SCHEMA,
        "schema_version": TECHNICAL_FAILURE_RECEIPT_VERSION,
        "semantics": TECHNICAL_FAILURE_RECEIPT_SEMANTICS,
        "status": "source-camera-reuse-technical-failure-retained",
        "amendment_id": normalized_amendment["amendment_id"],
        "execution_lock_id": lock_id,
        "source_revision": exact_revision(source_revision, name="source revision"),
        "runner_name": _identifier(runner_name, name="runner name"),
        "workflow_run_id": genuine_integer(
            workflow_run_id, name="workflow run ID", minimum=1
        ),
        "workflow_run_attempt": genuine_integer(
            workflow_run_attempt, name="workflow run attempt", minimum=1
        ),
        "base_source_execution": normalized_base,
        "terminal_stage": _identifier(terminal_stage, name="terminal stage"),
        "exit_code": code,
        "retained_artifacts": dict(retained),
        "information_boundary": dict(EXECUTION_INFORMATION_BOUNDARY),
        "source_suffix_access_authorized": False,
        "independent_confirmation_authorized": False,
        "claim_authorized": False,
    }
    result = {**identity, "receipt_id": content_id(identity)}
    return validate_deform360_v6_source_camera_reuse_technical_failure_receipt(result)


def validate_deform360_v6_source_camera_reuse_technical_failure_receipt(
    value: object,
) -> dict[str, Any]:
    """Validate a retained failure without interpreting source outcomes."""

    receipt = _mapping(value, name="camera reuse technical-failure receipt")
    require_exact_fields(
        receipt,
        expected=_TECHNICAL_FAILURE_RECEIPT_FIELDS,
        name="camera reuse technical-failure receipt",
    )
    _content_addressed(receipt, identity="receipt_id")
    _require(
        receipt.get("schema") == TECHNICAL_FAILURE_RECEIPT_SCHEMA
        and receipt.get("schema_version") == TECHNICAL_FAILURE_RECEIPT_VERSION
        and receipt.get("semantics") == TECHNICAL_FAILURE_RECEIPT_SEMANTICS
        and receipt.get("status") == "source-camera-reuse-technical-failure-retained",
        "camera reuse technical-failure contract changed",
    )
    sha256_digest(receipt.get("amendment_id"), name="amendment ID")
    sha256_digest(receipt.get("execution_lock_id"), name="execution lock ID")
    _require(
        receipt.get("amendment_id") == AMENDMENT_ID
        and receipt.get("execution_lock_id") == EXECUTION_LOCK_ID,
        "camera reuse technical-failure lock identity changed",
    )
    exact_revision(receipt.get("source_revision"), name="source revision")
    _identifier(receipt.get("runner_name"), name="runner name")
    _identifier(receipt.get("terminal_stage"), name="terminal stage")
    genuine_integer(receipt.get("workflow_run_id"), name="workflow run ID", minimum=1)
    genuine_integer(
        receipt.get("workflow_run_attempt"),
        name="workflow run attempt",
        minimum=1,
    )
    genuine_integer(
        receipt.get("exit_code"), name="technical-failure exit code", minimum=1
    )
    _normalize_base_source_execution(receipt.get("base_source_execution"))
    source_artifact_mapping(
        _mapping(receipt.get("retained_artifacts"), name="retained artifacts"),
        name="retained artifacts",
        allow_empty=True,
    )
    _require(
        receipt.get("information_boundary") == EXECUTION_INFORMATION_BOUNDARY
        and receipt.get("source_suffix_access_authorized") is False
        and receipt.get("independent_confirmation_authorized") is False
        and receipt.get("claim_authorized") is False,
        "camera reuse technical-failure boundary changed",
    )
    return cast(dict[str, Any], plain_json(receipt))


__all__ = [
    "AMENDMENT_ID",
    "AMENDMENT_SCHEMA",
    "BASE_EXECUTION_RECEIPT_FILE_SHA256",
    "BASE_EXECUTION_RECEIPT_ID",
    "BASE_SOURCE_EXECUTION",
    "CLAIM_BOUNDARY",
    "EXECUTION_ARTIFACT_NAMES",
    "EXECUTION_INFORMATION_BOUNDARY",
    "EXECUTION_LOCK_ID",
    "EXECUTION_RECEIPT_SCHEMA",
    "INFORMATION_BOUNDARY",
    "PREFLIGHT_SCHEMA",
    "REUSE_RECEIPT_SCHEMA",
    "TECHNICAL_FAILURE_RECEIPT_SCHEMA",
    "build_deform360_v6_source_camera_reuse_technical_failure_receipt",
    "build_deform360_v6_source_camera_reuse_lineage",
    "build_deform360_v6_source_camera_reuse_plan",
    "build_deform360_v6_source_camera_reuse_preflight",
    "build_deform360_v6_source_camera_reuse_execution_receipt",
    "validate_deform360_v6_source_camera_reuse_amendment",
    "validate_deform360_v6_source_camera_reuse_amendment_bindings",
    "validate_deform360_v6_source_camera_reuse_preflight",
    "validate_deform360_v6_source_camera_reuse_receipt",
    "validate_deform360_v6_source_camera_reuse_execution_receipt",
    "validate_deform360_v6_source_camera_reuse_technical_failure_receipt",
]
