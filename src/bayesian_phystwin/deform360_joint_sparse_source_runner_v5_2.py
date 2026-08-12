"""Versioned source publication after outcome-blind camera recovery.

This module is deliberately separate from the byte-locked v5 runner. It keeps
the original batch immutable, admits cameras independently using public prefix
measurements, and emits a new 10-by-10 prediction batch before any development
suffix is opened.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from . import deform360_joint_sparse_source_runner_v5 as _v5
from ._canonical_contracts import canonical_relative_posix_path, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
)
from .deform360_joint_sparse_camera_recovery_v5_2 import (
    CAMERA_REUSE_POLICY,
    RECOVERY_POLICY,
    validate_deform360_joint_sparse_camera_audit_v5_2,
)
from .deform360_joint_sparse_endpoint_v5 import select_reserved_endpoint_views_v5
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparsePrefixFitV5,
    materialize_deform360_joint_sparse_prediction_v5,
)
from .deform360_joint_sparse_prediction_artifacts_v5 import (
    PREDICTION_SEAL_FILENAME,
    load_deform360_joint_sparse_prediction_v5,
    publish_deform360_joint_sparse_prediction_v5,
)
from .deform360_joint_sparse_prediction_v5 import (
    RAW_METHOD_IDS,
    run_deform360_joint_sparse_prediction_v5,
)
from .deform360_joint_sparse_public_inputs_v5 import (
    estimate_deform360_last_causal_residual_v5,
    prepare_deform360_joint_sparse_contact_rows_v5,
    prepare_deform360_joint_sparse_visual_window_v5,
)
from .deform360_joint_sparse_source_evidence_v5 import (
    build_deform360_joint_sparse_source_prediction_batch_v5,
    build_deform360_joint_sparse_source_prediction_seal_v5,
    publish_deform360_joint_sparse_source_prediction_batch_v5,
    validate_deform360_joint_sparse_source_prediction_batch_v5,
    validate_deform360_joint_sparse_source_prediction_seal_v5,
)
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from .deform360_public_contact_prefix import validate_deform360_public_contact_prefix

SOURCE_PLAN_SCHEMA: Final = _v5.SOURCE_PLAN_SCHEMA
SOURCE_PLAN_VERSION: Final = 6
SOURCE_PLAN_SEMANTICS: Final = (
    "public-prefix-only-camera-recovery-nested-source-prediction-plan-v1"
)
SOURCE_PANEL_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-source-prediction-receipt"
)
SOURCE_PANEL_RECEIPT_VERSION: Final = 2
SOURCE_PLAN_BOUNDARY: Final = {
    **_v5.SOURCE_PLAN_BOUNDARY,
    "prefix_camera_quality_used_for_admission": True,
    "base_prediction_batch_replaced": False,
    "new_prediction_batch_versioned": True,
}

CONTACT_PREFIX_AVAILABLE: Final = _v5.CONTACT_PREFIX_AVAILABLE
CONTACT_PREFIX_UNAVAILABLE: Final = _v5.CONTACT_PREFIX_UNAVAILABLE
CONTACT_AXIS_IDENTITY_UNAVAILABLE_REASON: Final = (
    _v5.CONTACT_AXIS_IDENTITY_UNAVAILABLE_REASON
)

_FILE_FIELDS = frozenset({"path", "sha256"})
_PHYSICAL_FIELDS = frozenset({"path", "physical_mode", "sha256"})
_CONTACT_FIELDS = frozenset(
    {
        "manifest_file_sha256",
        "materialization_id",
        "path",
        "status",
        "unavailable_reason",
    }
)
_VISUAL_FIELDS = frozenset({"camera_id", "decoded_uniform", "metric_prefix"})
_OBJECT_FIELDS = frozenset(
    {
        "all_camera_ids",
        "camera_admission",
        "contact_prefix",
        "episode_id",
        "object_id",
        "physical",
        "raw_prefix_range_half_open",
        "reserved_endpoint_camera_ids",
        "stratum",
        "visual_windows",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "camera_recovery",
        "cohort_selection_sha256",
        "execution_lock_id",
        "implementation_revision",
        "information_boundary",
        "objects",
        "plan_id",
        "schema",
        "schema_version",
        "semantics",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "execution_lock_id",
        "implementation_revision",
        "information_boundary",
        "plan_id",
        "prediction_batch_file_sha256",
        "prediction_batch_id",
        "prediction_record_count",
        "receipt_id",
        "schema",
        "schema_version",
        "source_prediction_seal_file_sha256",
    }
)
_CAMERA_ADMISSION_FIELDS = frozenset(
    {
        "admitted",
        "attempted_camera_ids",
        "exact_physical_fallback_required",
        "failed_camera_ids",
        "final_camera_audit_id",
        "minimum_passing_camera_count",
        "passing_camera_ids",
    }
)
_CAMERA_RECOVERY_FIELDS = frozenset(
    {
        "artifact_ids",
        "base_prediction_batch_preserved",
        "policy",
        "source_artifacts",
    }
)
CAMERA_RECOVERY_ARTIFACT_NAMES: Final = frozenset(
    {
        "amendment",
        "base_camera_audit",
        "base_prediction_batch",
        "base_prediction_receipt",
        "combined_camera_audit_plan",
        "final_camera_audit",
        "recovery_preflight",
        "recovery_provider_plan",
        "recovery_provider_run",
    }
)
CAMERA_REUSE_ARTIFACT_NAMES: Final = frozenset(
    {
        "amendment",
        "base_camera_audit",
        "base_prediction_batch",
        "base_prediction_receipt",
        "base_source_plan",
        "camera_reuse_preflight",
        "camera_reuse_receipt",
        "combined_camera_audit_plan",
        "final_camera_audit",
        "metric_batch_result",
        "metric_prefix_plan",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
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
    result = nonempty_string(value, name=name)
    _require(result == result.strip() and "\x00" not in result, f"invalid {name}")
    return result


def _file_record(value: object, *, name: str) -> dict[str, str]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    return {
        "path": canonical_relative_posix_path(record.get("path"), name=f"{name}.path"),
        "sha256": sha256_digest(record.get("sha256"), name=f"{name}.sha256"),
    }


def _contact_record(value: object) -> dict[str, Any]:
    record = _mapping(value, name="contact_prefix")
    require_exact_fields(record, expected=_CONTACT_FIELDS, name="contact_prefix")
    status = record.get("status")
    _require(
        status in {CONTACT_PREFIX_AVAILABLE, CONTACT_PREFIX_UNAVAILABLE},
        "contact-prefix status changed",
    )
    if status == CONTACT_PREFIX_AVAILABLE:
        _require(
            record.get("unavailable_reason") is None,
            "available contact prefix has an unavailable reason",
        )
        return {
            "status": CONTACT_PREFIX_AVAILABLE,
            "path": canonical_relative_posix_path(
                record.get("path"), name="contact_prefix.path"
            ),
            "manifest_file_sha256": sha256_digest(
                record.get("manifest_file_sha256"),
                name="contact_prefix.manifest_file_sha256",
            ),
            "materialization_id": sha256_digest(
                record.get("materialization_id"),
                name="contact_prefix.materialization_id",
            ),
            "unavailable_reason": None,
        }
    _require(
        record.get("path") is None
        and record.get("manifest_file_sha256") is None
        and record.get("materialization_id") is None
        and record.get("unavailable_reason")
        == CONTACT_AXIS_IDENTITY_UNAVAILABLE_REASON,
        "unavailable contact-prefix record changed",
    )
    return {
        "status": CONTACT_PREFIX_UNAVAILABLE,
        "path": None,
        "manifest_file_sha256": None,
        "materialization_id": None,
        "unavailable_reason": CONTACT_AXIS_IDENTITY_UNAVAILABLE_REASON,
    }


def _normalized_object(
    value: object,
    *,
    cohort: Mapping[str, tuple[int, str]],
    minimum_visual_cameras: int,
) -> dict[str, Any]:
    row = _mapping(value, name="source plan object")
    expected = (
        _OBJECT_FIELDS
        if "camera_admission" in row
        else _OBJECT_FIELDS - {"camera_admission"}
    )
    require_exact_fields(row, expected=expected, name="source plan object")
    object_id = _identifier(row.get("object_id"), name="object_id")
    _require(object_id in cohort, "source plan object is outside the cohort")
    episode_id, stratum = cohort[object_id]
    _require(
        row.get("episode_id") == episode_id and row.get("stratum") == stratum,
        "source object identity changed",
    )
    prefix = _sequence(
        row.get("raw_prefix_range_half_open"), name="raw_prefix_range_half_open"
    )
    _require(
        len(prefix) == 2
        and all(type(item) is int for item in prefix)
        and 0 <= prefix[0] < prefix[1]
        and prefix[1] - prefix[0] == 58,
        "source prefix must contain exactly 58 causal frames",
    )
    all_cameras = tuple(
        _identifier(item, name="camera_id")
        for item in _sequence(row.get("all_camera_ids"), name="all_camera_ids")
    )
    _require(
        all_cameras == tuple(sorted(set(all_cameras))) and len(all_cameras) >= 4,
        "all_camera_ids must contain at least four unique sorted cameras",
    )
    reserved = select_reserved_endpoint_views_v5(object_id, all_cameras, count=2)
    _require(
        tuple(row.get("reserved_endpoint_camera_ids", ())) == reserved,
        "reserved endpoint camera identities changed",
    )
    physical_raw = _mapping(row.get("physical"), name="physical")
    require_exact_fields(physical_raw, expected=_PHYSICAL_FIELDS, name="physical")
    physical_mode = physical_raw.get("physical_mode")
    _require(
        physical_mode in {"warp_twin", "persistence_fallback"},
        "physical mode changed",
    )
    physical = {
        **_file_record(
            {"path": physical_raw.get("path"), "sha256": physical_raw.get("sha256")},
            name="physical archive",
        ),
        "physical_mode": physical_mode,
    }
    windows: list[dict[str, Any]] = []
    for index, raw_window in enumerate(
        _sequence(row.get("visual_windows"), name="visual_windows")
    ):
        window = _mapping(raw_window, name=f"visual_windows[{index}]")
        require_exact_fields(
            window, expected=_VISUAL_FIELDS, name=f"visual_windows[{index}]"
        )
        camera = _identifier(window.get("camera_id"), name="camera_id")
        _require(
            camera in all_cameras and camera not in reserved,
            "visual camera is unavailable or reserved",
        )
        windows.append(
            {
                "camera_id": camera,
                "decoded_uniform": _file_record(
                    window.get("decoded_uniform"),
                    name=f"visual_windows[{index}].decoded_uniform",
                ),
                "metric_prefix": _file_record(
                    window.get("metric_prefix"),
                    name=f"visual_windows[{index}].metric_prefix",
                ),
            }
        )
    windows.sort(key=lambda item: cast(str, item["camera_id"]))
    camera_ids = [cast(str, item["camera_id"]) for item in windows]
    _require(
        type(minimum_visual_cameras) is int
        and minimum_visual_cameras >= 0
        and len(windows) >= minimum_visual_cameras
        and len(camera_ids) == len(set(camera_ids)),
        f"source plan needs at least {minimum_visual_cameras} unique visual cameras",
    )
    result = {
        "object_id": object_id,
        "episode_id": episode_id,
        "stratum": stratum,
        "raw_prefix_range_half_open": list(prefix),
        "all_camera_ids": list(all_cameras),
        "reserved_endpoint_camera_ids": list(reserved),
        "physical": physical,
        "visual_windows": windows,
        "contact_prefix": _contact_record(row.get("contact_prefix")),
    }
    if "camera_admission" in row:
        result["camera_admission"] = row["camera_admission"]
    return result


def _normalized_camera_recovery(value: object) -> dict[str, Any]:
    recovery = _mapping(value, name="camera_recovery")
    require_exact_fields(
        recovery, expected=_CAMERA_RECOVERY_FIELDS, name="camera_recovery"
    )
    artifact_ids = source_artifact_mapping(
        _mapping(recovery.get("artifact_ids"), name="camera recovery artifact IDs"),
        name="camera recovery artifact IDs",
    )
    source_artifacts = source_artifact_mapping(
        _mapping(recovery.get("source_artifacts"), name="camera recovery artifacts"),
        name="camera recovery artifacts",
    )
    artifact_names = set(artifact_ids)
    _require(
        artifact_names in {CAMERA_RECOVERY_ARTIFACT_NAMES, CAMERA_REUSE_ARTIFACT_NAMES}
        and set(source_artifacts) == artifact_names,
        "camera recovery artifact roster changed",
    )
    expected_policy = (
        RECOVERY_POLICY
        if artifact_names == CAMERA_RECOVERY_ARTIFACT_NAMES
        else CAMERA_REUSE_POLICY
    )
    _require(
        recovery.get("policy") == expected_policy,
        "camera recovery policy changed",
    )
    _require(
        recovery.get("base_prediction_batch_preserved") is True,
        "base prediction batch must remain preserved",
    )
    return {
        "artifact_ids": dict(artifact_ids),
        "source_artifacts": dict(source_artifacts),
        "policy": dict(expected_policy),
        "base_prediction_batch_preserved": True,
    }


def _normalized_camera_admission(
    value: object,
    *,
    visual_camera_ids: Sequence[str],
    final_camera_audit_id: str,
) -> dict[str, Any]:
    admission = _mapping(value, name="camera_admission")
    require_exact_fields(
        admission, expected=_CAMERA_ADMISSION_FIELDS, name="camera_admission"
    )
    attempted = tuple(
        _identifier(item, name="attempted camera")
        for item in _sequence(
            admission.get("attempted_camera_ids"), name="attempted cameras"
        )
    )
    passing = tuple(
        _identifier(item, name="passing camera")
        for item in _sequence(
            admission.get("passing_camera_ids"), name="passing cameras"
        )
    )
    failed = tuple(
        _identifier(item, name="failed camera")
        for item in _sequence(admission.get("failed_camera_ids"), name="failed cameras")
    )
    _require(
        attempted == tuple(sorted(set(attempted)))
        and passing == tuple(sorted(set(passing)))
        and failed == tuple(sorted(set(failed)))
        and set(passing).isdisjoint(failed)
        and set(passing) | set(failed) == set(attempted),
        "camera admission accounting changed",
    )
    admitted = len(passing) >= 2
    _require(
        admission.get("minimum_passing_camera_count") == 2
        and admission.get("admitted") is admitted
        and admission.get("exact_physical_fallback_required") is (not admitted),
        "camera admission decision changed",
    )
    _require(
        tuple(visual_camera_ids) == (passing if admitted else ()),
        "visual windows differ from automatic camera admission",
    )
    _require(
        admission.get("final_camera_audit_id") == final_camera_audit_id,
        "camera admission uses a different final audit",
    )
    return {
        "attempted_camera_ids": list(attempted),
        "passing_camera_ids": list(passing),
        "failed_camera_ids": list(failed),
        "minimum_passing_camera_count": 2,
        "admitted": admitted,
        "exact_physical_fallback_required": not admitted,
        "final_camera_audit_id": final_camera_audit_id,
    }


def _require_public_contact_policy(
    lock: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]
) -> None:
    measurements = _mapping(lock.get("public_measurements"), name="public_measurements")
    _require(
        measurements.get("tactile_axis_identity_policy")
        == "unavailable-in-release-exact-no-contact-fallback",
        "source lock changed the tactile-axis policy",
    )
    _require(
        all(
            cast(Mapping[str, Any], item["contact_prefix"])["status"]
            == CONTACT_PREFIX_UNAVAILABLE
            for item in objects
        ),
        "recovery source plan invents an unregistered tactile-to-robot axis identity",
    )


def build_deform360_joint_sparse_source_prediction_plan_v5_2(
    *,
    lock: Mapping[str, Any],
    implementation_revision: str,
    attempted_objects: Sequence[Mapping[str, Any]],
    final_camera_audit: Mapping[str, Any],
    camera_recovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the source plan after independent prefix-only camera admission."""

    cohort = _v5._cohort(lock)  # noqa: SLF001
    recovery = _normalized_camera_recovery(camera_recovery)
    audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        final_camera_audit, lock=lock
    )
    final_audit_id = cast(str, audit["audit_id"])
    _require(
        recovery["artifact_ids"]["final_camera_audit"] == final_audit_id,
        "camera recovery does not bind the final audit",
    )
    audit_rows = {
        cast(str, row["object_id"]): row
        for row in cast(Sequence[Mapping[str, Any]], audit["objects"])
    }
    normalized: list[dict[str, Any]] = []
    for raw_object in attempted_objects:
        attempted = _normalized_object(
            raw_object, cohort=cohort, minimum_visual_cameras=2
        )
        object_id = cast(str, attempted["object_id"])
        audit_row = audit_rows[object_id]
        attempted_cameras = tuple(
            cast(Sequence[str], audit_row["attempted_camera_ids"])
        )
        window_map = {
            cast(str, window["camera_id"]): window
            for window in cast(Sequence[Mapping[str, Any]], attempted["visual_windows"])
        }
        _require(
            tuple(sorted(window_map)) == attempted_cameras,
            "attempted windows differ from the final camera audit",
        )
        audit_results = {
            cast(str, result["camera_id"]): result
            for result in cast(Sequence[Mapping[str, Any]], audit_row["camera_results"])
        }
        _require(
            set(audit_results) == set(window_map)
            and all(
                audit_results[camera]["decoded_uniform_sha256"]
                == window_map[camera]["decoded_uniform"]["sha256"]
                and audit_results[camera]["metric_prefix_sha256"]
                == window_map[camera]["metric_prefix"]["sha256"]
                for camera in window_map
            ),
            "final camera audit does not bind the attempted provider archives",
        )
        passing = tuple(cast(Sequence[str], audit_row["passing_camera_ids"]))
        admitted = len(passing) >= 2
        selected = [window_map[camera] for camera in passing] if admitted else []
        admission = _normalized_camera_admission(
            {
                "attempted_camera_ids": list(attempted_cameras),
                "passing_camera_ids": list(passing),
                "failed_camera_ids": list(audit_row["failed_camera_ids"]),
                "minimum_passing_camera_count": 2,
                "admitted": admitted,
                "exact_physical_fallback_required": not admitted,
                "final_camera_audit_id": final_audit_id,
            },
            visual_camera_ids=[cast(str, row["camera_id"]) for row in selected],
            final_camera_audit_id=final_audit_id,
        )
        normalized.append(
            {**attempted, "visual_windows": selected, "camera_admission": admission}
        )
    normalized.sort(key=lambda item: cast(str, item["object_id"]))
    _require_public_contact_policy(lock, normalized)
    _require(
        [item["object_id"] for item in normalized] == sorted(cohort),
        "recovery source plan differs from the exact cohort",
    )
    cohort_record = _mapping(lock.get("cohort"), name="cohort")
    identity: dict[str, Any] = {
        "schema": SOURCE_PLAN_SCHEMA,
        "schema_version": SOURCE_PLAN_VERSION,
        "semantics": SOURCE_PLAN_SEMANTICS,
        "execution_lock_id": sha256_digest(
            lock.get("execution_lock_id"), name="execution_lock_id"
        ),
        "cohort_selection_sha256": sha256_digest(
            cohort_record.get("selection_sha256"), name="selection_sha256"
        ),
        "implementation_revision": exact_revision(
            implementation_revision, name="implementation_revision"
        ),
        "objects": normalized,
        "camera_recovery": recovery,
        "information_boundary": dict(SOURCE_PLAN_BOUNDARY),
    }
    plan = {**identity, "plan_id": content_id(identity)}
    validate_deform360_joint_sparse_source_prediction_plan_v5_2(plan, lock=lock)
    return plan


def validate_deform360_joint_sparse_source_prediction_plan_v5_2(
    value: object,
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one content-addressed v5.2 source plan."""

    plan = _mapping(value, name="recovery source prediction plan")
    require_exact_fields(plan, expected=_PLAN_FIELDS, name="recovery source plan")
    _require(
        plan.get("schema") == SOURCE_PLAN_SCHEMA
        and plan.get("schema_version") == SOURCE_PLAN_VERSION
        and plan.get("semantics") == SOURCE_PLAN_SEMANTICS,
        "recovery source-plan contract changed",
    )
    _require(
        plan.get("information_boundary") == SOURCE_PLAN_BOUNDARY,
        "recovery source-plan boundary changed",
    )
    _require(
        plan.get("execution_lock_id") == lock.get("execution_lock_id"),
        "recovery source plan uses a different execution lock",
    )
    cohort_record = _mapping(lock.get("cohort"), name="cohort")
    _require(
        plan.get("cohort_selection_sha256") == cohort_record.get("selection_sha256"),
        "recovery source-plan cohort changed",
    )
    exact_revision(plan.get("implementation_revision"), name="implementation revision")
    recovery = _normalized_camera_recovery(plan.get("camera_recovery"))
    final_audit_id = cast(str, recovery["artifact_ids"]["final_camera_audit"])
    cohort = _v5._cohort(lock)  # noqa: SLF001
    normalized: list[dict[str, Any]] = []
    for raw_object in _sequence(plan.get("objects"), name="objects"):
        item = _normalized_object(raw_object, cohort=cohort, minimum_visual_cameras=0)
        visual_cameras = [
            cast(str, window["camera_id"])
            for window in cast(Sequence[Mapping[str, Any]], item["visual_windows"])
        ]
        admission = _normalized_camera_admission(
            item.get("camera_admission"),
            visual_camera_ids=visual_cameras,
            final_camera_audit_id=final_audit_id,
        )
        _require(
            set(admission["attempted_camera_ids"]).issubset(item["all_camera_ids"])
            and set(admission["attempted_camera_ids"]).isdisjoint(
                item["reserved_endpoint_camera_ids"]
            ),
            "camera admission includes an unavailable or reserved camera",
        )
        item["camera_admission"] = admission
        normalized.append(item)
    normalized.sort(key=lambda item: cast(str, item["object_id"]))
    _require_public_contact_policy(lock, normalized)
    _require(
        [item["object_id"] for item in normalized] == sorted(cohort),
        "recovery source-plan cohort changed",
    )
    identity = {key: item for key, item in plan.items() if key != "plan_id"}
    _require(plan.get("plan_id") == content_id(identity), "recovery plan ID changed")
    _require(
        plain_json(plan["objects"]) == plain_json(normalized)
        and plain_json(plan["camera_recovery"]) == recovery,
        "recovery source-plan normalization changed",
    )
    return cast(dict[str, Any], plain_json(plan))


def validate_deform360_joint_sparse_source_prediction_receipt_v5_2(
    value: object,
    *,
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    prediction_batch_file_sha256: str,
) -> dict[str, Any]:
    """Validate the complete v5.2 receipt before suffix access."""

    normalized_plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        plan, lock=lock
    )
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        prediction_batch, lock
    )
    receipt = _mapping(value, name="source prediction receipt")
    require_exact_fields(receipt, expected=_RECEIPT_FIELDS, name="source receipt")
    _require(
        receipt.get("schema") == SOURCE_PANEL_RECEIPT_SCHEMA
        and receipt.get("schema_version") == SOURCE_PANEL_RECEIPT_VERSION
        and receipt.get("information_boundary") == SOURCE_PLAN_BOUNDARY,
        "v5.2 source receipt contract changed",
    )
    _require(
        receipt.get("execution_lock_id") == lock.get("execution_lock_id")
        and receipt.get("plan_id") == normalized_plan["plan_id"]
        and receipt.get("prediction_batch_id") == batch["prediction_batch_id"]
        and receipt.get("implementation_revision")
        == batch["implementation_revision"]
        == normalized_plan["implementation_revision"],
        "v5.2 source receipt lineage changed",
    )
    _require(
        receipt.get("prediction_batch_file_sha256")
        == sha256_digest(
            prediction_batch_file_sha256, name="prediction_batch_file_sha256"
        ),
        "v5.2 receipt prediction-batch digest changed",
    )
    _require(
        receipt.get("prediction_record_count") == 100,
        "v5.2 source receipt must bind exactly 100 predictions",
    )
    seal_digests = source_artifact_mapping(
        _mapping(
            receipt.get("source_prediction_seal_file_sha256"),
            name="source prediction seal digests",
        ),
        name="source prediction seal digests",
    )
    expected = {
        f"{outer_index:02d}-{target_index:02d}.json"
        for outer_index in range(10)
        for target_index in range(10)
    }
    _require(set(seal_digests) == expected, "v5.2 receipt seal roster changed")
    identity = {key: item for key, item in receipt.items() if key != "receipt_id"}
    _require(
        receipt.get("receipt_id") == content_id(identity),
        "v5.2 receipt content identity changed",
    )
    return cast(dict[str, Any], plain_json(receipt))


def _verified_contact_directory(root: Path, record: Mapping[str, Any]) -> Path:
    relative = canonical_relative_posix_path(
        record.get("path"), name="contact_prefix.path"
    )
    requested = root / relative
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        "contact prefix must be an ordinary directory",
    )
    path = requested.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("contact prefix escapes the input root") from error
    _require(
        _v5._sha256_file(path / "contact-prefix.json")  # noqa: SLF001
        == record.get("manifest_file_sha256"),
        "contact-prefix manifest SHA-256 changed",
    )
    manifest = validate_deform360_public_contact_prefix(path)
    _require(
        manifest["materialization_id"] == record.get("materialization_id"),
        "contact-prefix materialization identity changed",
    )
    return path


def publish_deform360_joint_sparse_source_prediction_panel_v5_2(
    *,
    execution_lock_path: str | Path,
    source_plan_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Publish a new 100-forecast batch without opening the suffix."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(execution_lock_path)
    plan_path = Path(source_plan_path).absolute().resolve(strict=True)
    plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        load_strict_json_object(plan_path, label="v5.2 source prediction plan"),
        lock=lock,
    )
    root = _v5._ordinary_root(input_root)  # noqa: SLF001
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=True)
    _require(output.is_dir() and not output.is_symlink(), "output root is invalid")
    cohort = _v5._cohort(lock)  # noqa: SLF001
    revision = cast(str, plan["implementation_revision"])
    execution_lock_id = cast(str, plan["execution_lock_id"])
    recovery = cast(Mapping[str, Any], plan["camera_recovery"])
    recovery_sources = cast(Mapping[str, str], recovery["source_artifacts"])
    common_sources = source_artifact_mapping(
        {
            "locks/source-execution-v5.json": _v5._sha256_file(  # noqa: SLF001
                Path(execution_lock_path).resolve(strict=True)
            ),
            "plans/source-prediction-plan-v5-2.json": _v5._sha256_file(  # noqa: SLF001
                plan_path
            ),
            **{
                f"camera-recovery/{name}.json": digest
                for name, digest in recovery_sources.items()
            },
        },
        name="v5.2 source panel common artifacts",
    )
    object_rows = {
        cast(str, row["object_id"]): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Mapping[str, Any]], plan["objects"])
    }
    prepared: dict[str, tuple[Any, ...]] = {}
    base_fit = Deform360JointSparsePrefixFitV5(
        fit_object_ids=tuple(sorted(cohort)), source_artifact_ids=common_sources
    )
    for object_id in sorted(cohort):
        row = object_rows[object_id]
        physical_record = cast(Mapping[str, Any], row["physical"])
        physical_path = _v5._verified_file(  # noqa: SLF001
            root, physical_record, name=f"physical archive for {object_id}"
        )
        physical_mode = cast(str, physical_record["physical_mode"])
        physical, persistence = _v5._load_physical_archive(  # noqa: SLF001
            physical_path, physical_mode=physical_mode
        )
        prefix = cast(Sequence[int], row["raw_prefix_range_half_open"])
        raw_prefix = (int(prefix[0]), int(prefix[1]))
        object_sources = {
            **dict(common_sources),
            f"physical/{object_id}.npz": _v5._sha256_file(physical_path),  # noqa: SLF001
        }
        visual_inputs: list[tuple[str, Path, Path]] = []
        for visual in cast(Sequence[Mapping[str, Any]], row["visual_windows"]):
            camera_id = cast(str, visual["camera_id"])
            decoded = _v5._verified_file(  # noqa: SLF001
                root,
                cast(Mapping[str, Any], visual["decoded_uniform"]),
                name=f"decoded uniform for {object_id}/{camera_id}",
            )
            metric = _v5._verified_file(  # noqa: SLF001
                root,
                cast(Mapping[str, Any], visual["metric_prefix"]),
                name=f"metric prefix for {object_id}/{camera_id}",
            )
            visual_inputs.append((camera_id, decoded, metric))
            object_sources.update(
                {
                    f"visual/{object_id}/{camera_id}/decoded-uniform.npz": (
                        _v5._sha256_file(decoded)  # noqa: SLF001
                    ),
                    f"visual/{object_id}/{camera_id}/metric-prefix.npz": (
                        _v5._sha256_file(metric)  # noqa: SLF001
                    ),
                }
            )
        contact_record = cast(Mapping[str, Any], row["contact_prefix"])
        if contact_record["status"] == CONTACT_PREFIX_AVAILABLE:
            contact_path: Path | None = _verified_contact_directory(
                root, contact_record
            )
            object_sources.update(
                {
                    f"contact/{object_id}/contact-prefix.json": cast(
                        str, contact_record["manifest_file_sha256"]
                    ),
                    f"contact/{object_id}/materialization-id": cast(
                        str, contact_record["materialization_id"]
                    ),
                }
            )
        else:
            contact_path = None
            reason = cast(str, contact_record["unavailable_reason"])
            object_sources[f"contact/{object_id}/unavailable-policy"] = hashlib.sha256(
                reason.encode("ascii")
            ).hexdigest()
        episode_id, _stratum = cohort[object_id]
        admission = cast(Mapping[str, Any], row["camera_admission"])
        technical_failure: tuple[str, Exception] | None = None
        visual_rows = []
        if admission["exact_physical_fallback_required"]:
            technical_failure = (
                "camera_admission",
                ValueError(
                    "fewer than two independently passing cameras after v5.2 recovery"
                ),
            )
            contact = None
            residual = np.zeros(physical.shape[1:], dtype=np.float64)
        else:
            try:
                for camera_id, decoded, metric in visual_inputs:
                    rows, _gauge = prepare_deform360_joint_sparse_visual_window_v5(
                        camera_id=camera_id,
                        decoded_uniform_path=decoded,
                        metric_prefix_path=metric,
                        raw_prefix_range_half_open=raw_prefix,
                        fit=base_fit,
                        source_artifact_ids=object_sources,
                    )
                    visual_rows.append(rows)
                contact = (
                    None
                    if contact_path is None
                    else prepare_deform360_joint_sparse_contact_rows_v5(
                        contact_prefix_directory=contact_path,
                        object_id=object_id,
                        episode_id=episode_id,
                        raw_prefix_range_half_open=raw_prefix,
                        physical_prediction_m=physical,
                        source_artifact_ids=object_sources,
                    )
                )
                residual = estimate_deform360_last_causal_residual_v5(
                    visual_windows=tuple(visual_rows),
                    physical_prediction_m=physical,
                    causal_frame_stop=58,
                )
            except (
                OSError,
                ValueError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as error:
                technical_failure = ("prefix_provider", error)
                contact = None
                residual = np.zeros(physical.shape[1:], dtype=np.float64)
        prepared[object_id] = (
            physical,
            persistence,
            physical_mode,
            tuple(visual_rows),
            contact,
            residual,
            object_sources,
            technical_failure,
        )

    prediction_root = output / "predictions"
    source_seal_root = output / "source-seals"
    prediction_root.mkdir(parents=True, exist_ok=True)
    source_seal_root.mkdir(parents=True, exist_ok=True)
    source_seals: list[dict[str, Any]] = []
    seal_file_digests: dict[str, str] = {}
    ordered_ids = tuple(sorted(cohort))
    for outer_index, outer_id in enumerate(ordered_ids):
        for target_index, target_id in enumerate(ordered_ids):
            fit_ids = _v5._fit_object_ids(  # noqa: SLF001
                cohort, outer_object_id=outer_id, target_object_id=target_id
            )
            fit = Deform360JointSparsePrefixFitV5(
                fit_object_ids=fit_ids, source_artifact_ids=common_sources
            )
            (
                physical,
                persistence,
                physical_mode,
                windows,
                contact,
                residual,
                sources,
                technical_failure,
            ) = prepared[target_id]
            episode_id, stratum = cohort[target_id]
            if technical_failure is None:
                materialized = materialize_deform360_joint_sparse_prediction_v5(
                    object_id=target_id,
                    episode_id=episode_id,
                    stratum=cast(Any, stratum),
                    physical_prediction_m=cast(np.ndarray, physical),
                    persistence_m=cast(np.ndarray, persistence),
                    last_causal_residual_m=cast(np.ndarray, residual),
                    physical_mode=cast(str, physical_mode),
                    causal_frame_stop=58,
                    evaluation_frame_range_half_open=(58, 76),
                    visual_windows=cast(Sequence[Any], windows),
                    contact_rows=cast(Any, contact),
                    fit=fit,
                    implementation_revision=revision,
                    source_artifact_ids=cast(Mapping[str, str], sources),
                )
                problem = materialized.problem
            else:
                failure_stage, failure = cast(tuple[str, Exception], technical_failure)
                problem = _v5._technical_fallback_problem(  # noqa: SLF001
                    object_id=target_id,
                    episode_id=episode_id,
                    stratum=stratum,
                    physical_prediction_m=cast(np.ndarray, physical),
                    persistence_m=cast(np.ndarray, persistence),
                    physical_mode=cast(str, physical_mode),
                    implementation_revision=revision,
                    source_artifact_ids=cast(Mapping[str, str], sources),
                    failure_stage=failure_stage,
                    failure=failure,
                )
            result = run_deform360_joint_sparse_prediction_v5(problem)
            relative = f"{outer_index:02d}-{outer_id}/{target_index:02d}-{target_id}"
            prediction_directory = prediction_root / relative
            if prediction_directory.exists():
                prediction_seal, existing_result = (
                    load_deform360_joint_sparse_prediction_v5(prediction_directory)
                )
                _require(
                    prediction_seal["input_id"] == problem.input_id
                    and existing_result.result_id == result.result_id
                    and prediction_seal["prediction_fit_artifact_id"]
                    == fit.fit_artifact_id,
                    "existing v5.2 source prediction differs",
                )
            else:
                prediction_seal = publish_deform360_joint_sparse_prediction_v5(
                    problem,
                    result,
                    prediction_directory,
                    execution_lock_id=execution_lock_id,
                    implementation_revision=revision,
                    prediction_fit_artifact_id=fit.fit_artifact_id,
                    prediction_fit_object_ids=fit_ids,
                )
            method_ids = cast(Mapping[str, str], prediction_seal["method_artifact_ids"])
            features = cast(
                Mapping[str, float], prediction_seal["predicted_loss_features_m"]
            )
            source_seal = build_deform360_joint_sparse_source_prediction_seal_v5(
                lock=lock,
                implementation_revision=revision,
                outer_held_out_object_id=outer_id,
                record_role="held_out" if outer_id == target_id else "training",
                object_id=target_id,
                factor_admitted=bool(prediction_seal["factor_admitted"]),
                technical_failure=technical_failure is not None,
                physical_mode=cast(str, prediction_seal["physical_mode"]),
                risk_score=float(prediction_seal["risk_score"]),
                prediction_fit_artifact_id=fit.fit_artifact_id,
                prediction_fit_object_ids=fit_ids,
                methods={
                    method_id: {
                        "artifact_id": method_ids[method_id],
                        "predicted_loss_mm": 1000.0 * float(features[method_id]),
                    }
                    for method_id in RAW_METHOD_IDS
                },
                source_artifacts={
                    **dict(
                        cast(Mapping[str, str], prediction_seal["source_artifact_ids"])
                    ),
                    f"predictions/{relative}/{PREDICTION_SEAL_FILENAME}": (
                        _v5._sha256_file(  # noqa: SLF001
                            prediction_directory / PREDICTION_SEAL_FILENAME
                        )
                    ),
                },
            )
            validate_deform360_joint_sparse_source_prediction_seal_v5(source_seal, lock)
            source_seal_path = (
                source_seal_root / f"{outer_index:02d}-{target_index:02d}.json"
            )
            _v5._publish_or_validate_json(  # noqa: SLF001
                source_seal, source_seal_path, label="v5.2 source prediction seal"
            )
            source_seals.append(source_seal)
            seal_file_digests[source_seal_path.name] = _v5._sha256_file(  # noqa: SLF001
                source_seal_path
            )

    batch = build_deform360_joint_sparse_source_prediction_batch_v5(source_seals, lock)
    batch_path = output / "source-prediction-batch.json"
    if batch_path.exists():
        _v5._publish_or_validate_json(  # noqa: SLF001
            batch, batch_path, label="v5.2 source prediction batch"
        )
    else:
        publish_deform360_joint_sparse_source_prediction_batch_v5(
            batch, lock=lock, output_path=batch_path
        )
    receipt_identity: dict[str, Any] = {
        "schema": SOURCE_PANEL_RECEIPT_SCHEMA,
        "schema_version": SOURCE_PANEL_RECEIPT_VERSION,
        "execution_lock_id": execution_lock_id,
        "implementation_revision": revision,
        "plan_id": plan["plan_id"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "prediction_batch_file_sha256": _v5._sha256_file(batch_path),  # noqa: SLF001
        "prediction_record_count": 100,
        "source_prediction_seal_file_sha256": dict(sorted(seal_file_digests.items())),
        "information_boundary": dict(SOURCE_PLAN_BOUNDARY),
    }
    receipt = {**receipt_identity, "receipt_id": content_id(receipt_identity)}
    require_exact_fields(receipt, expected=_RECEIPT_FIELDS, name="source receipt")
    _v5._publish_or_validate_json(  # noqa: SLF001
        receipt,
        output / "source-prediction-receipt.json",
        label="v5.2 source prediction receipt",
    )
    return receipt


__all__ = [
    "CAMERA_RECOVERY_ARTIFACT_NAMES",
    "CAMERA_REUSE_ARTIFACT_NAMES",
    "SOURCE_PLAN_BOUNDARY",
    "SOURCE_PLAN_SCHEMA",
    "SOURCE_PLAN_SEMANTICS",
    "SOURCE_PLAN_VERSION",
    "build_deform360_joint_sparse_source_prediction_plan_v5_2",
    "publish_deform360_joint_sparse_source_prediction_panel_v5_2",
    "validate_deform360_joint_sparse_source_prediction_plan_v5_2",
    "validate_deform360_joint_sparse_source_prediction_receipt_v5_2",
]
