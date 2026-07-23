"""Target-free case seals and the complete-cohort prediction barrier.

This module is the pre-outcome publication boundary for the prospective
Deform360 adaptive-covariance confirmation.  Phase one atomically publishes
one exact H2-locked case directory containing only caller-supplied prediction
arrays and target-free diagnostics.  Phase two publishes a cohort barrier only
after every exact locked case directory and every bound content hash validates.

No function in this module accepts a hidden future, scoring array, metric, or
evaluation callback.  The only files it opens are the H2 lock and artifacts
created by this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    validate_confirmation_cohort_lock,
)
from .deform360_adaptive_covariance_rbf import (
    ADAPTIVE_COVARIANCE_PROTOCOL_ID,
)


SCHEMA_VERSION = 1
CASE_SEAL_KIND = "Deform360AdaptiveCovarianceConfirmationCaseSealV1"
CASE_DIAGNOSTIC_KIND = "Deform360AdaptiveCovarianceConfirmationTargetFreeDiagnosticV1"
COHORT_BARRIER_KIND = "Deform360AdaptiveCovarianceConfirmationPredictionBarrierV1"
ARRAY_ARCHIVE_FILENAME = "target_free_predictions.npz"
DIAGNOSTIC_FILENAME = "target_free_diagnostics.json"
CASE_MANIFEST_FILENAME = "case_prediction_seal.json"

ARRAY_ROLES = (
    "physical_prior_m",
    "persistence_m",
    "adaptive_prediction_m",
    "fixed_4_rbf_prediction_m",
    "fixed_8_rbf_prediction_m",
    "selected_raw_prediction_m",
)
CAMERA_BUDGETS = (4, 8)
FRAME_COUNT = 76
CENTER_COUNT = 16
UPDATE_FRAMES = (19, 38, 57)
UPDATE_STOPS = (38, 57, 76)
ALLOWED_ROUTES = (
    "4_view_rbf",
    "8_view_rbf",
    "physical_prior_fallback",
)
RETAINED_FAILURE_CODES = (
    "automatic_twin_backend_failure",
    "prediction_runtime_failure",
    "resource_exhaustion",
)
TARGET_FREE_BOUNDARY: Mapping[str, Any] = {
    "sealer_target_or_outcome_argument_accepted": False,
    "sealer_target_or_outcome_path_opened": False,
    "metric_or_score_computed": False,
    "prediction_content_only": True,
    "all_case_predictions_must_seal_before_barrier": True,
}

_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")
_FULL_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_EVALUATION_KEY_TOKENS = (
    "target",
    "outcome",
    "ground_truth",
    "groundtruth",
    "metric",
    "score",
    "oracle",
    "evaluation",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_full_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and _FULL_SHA1.fullmatch(value) is not None
        and value != "0" * 40
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _FULL_SHA256.fullmatch(value) is not None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        encoded = _canonical_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not finite canonical JSON") from error
    return json.loads(encoded, object_pairs_hook=_strict_json_object)


def _reject_evaluation_fields(
    value: Any,
    *,
    label: str,
    path: tuple[str, ...] = (),
    allowed_false_keys: frozenset[str] = frozenset(),
) -> None:
    """Reject hidden evaluation payloads embedded in generic diagnostics."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(isinstance(key, str), f"{label} contains a non-string key")
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            forbidden = any(
                token in normalized for token in _FORBIDDEN_EVALUATION_KEY_TOKENS
            )
            if forbidden:
                _require(
                    key in allowed_false_keys and child is False,
                    f"{label} contains forbidden evaluation field: "
                    f"{'.'.join(path + (key,))}",
                )
            _reject_evaluation_fields(
                child,
                label=label,
                path=path + (key,),
                allowed_false_keys=allowed_false_keys,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_evaluation_fields(
                child,
                label=label,
                path=path + (str(index),),
                allowed_false_keys=allowed_false_keys,
            )


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one JSON artifact while excluding its declared self-digest."""

    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash an array with explicit dtype and shape framing."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    dtype = array.dtype.str.encode("ascii")
    shape = np.asarray(array.shape, dtype=">i8").tobytes()
    digest.update(len(dtype).to_bytes(8, "big"))
    digest.update(dtype)
    digest.update(len(shape).to_bytes(8, "big"))
    digest.update(shape)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _stable_regular_file_bytes(path: str | Path) -> tuple[Path, bytes, os.stat_result]:
    source = Path(path).absolute()
    before = os.lstat(source)
    _require(not stat.S_ISLNK(before.st_mode), f"symlink is forbidden: {source}")
    _require(stat.S_ISREG(before.st_mode), f"not a regular file: {source}")
    _require(
        source.resolve(strict=True) == source,
        f"file has a symlinked or noncanonical ancestor: {source}",
    )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"file changed while opening: {source}",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        current = os.lstat(source)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        _require(
            identity
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            == (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ),
            f"file changed while reading: {source}",
        )
        _require(len(payload) == opened.st_size, f"short file read: {source}")
        return source, payload, after
    finally:
        os.close(descriptor)


def _load_strict_json_file(
    path: str | Path, *, label: str
) -> tuple[Path, dict[str, Any], bytes]:
    source, payload, _ = _stable_regular_file_bytes(path)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return source, value, payload


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _canonical_absent_path(path: str | Path, *, label: str) -> Path:
    given = Path(path).absolute()
    _require(
        not given.exists() and not given.is_symlink(),
        f"{label} already exists: {given}",
    )
    given.parent.mkdir(parents=True, exist_ok=True)
    parent = given.parent.resolve(strict=True)
    _require(parent == given.parent, f"{label} parent is noncanonical or symlinked")
    return parent / given.name


def _load_lock_binding(
    lock_path: str | Path,
    h2_commit: str,
    *,
    expected_h1: str | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _require(
        _is_full_sha1(h2_commit),
        "H2 must be a full non-null lowercase 40-hex commit SHA",
    )
    source, payload, raw = _load_strict_json_file(lock_path, label="H2 cohort lock")
    validation = validate_confirmation_cohort_lock(
        payload,
        expected_implementation_commit_h1=expected_h1,
    )
    h1 = validation["implementation_commit_h1"]
    _require(h2_commit != h1, "H2 must differ from implementation commit H1")
    binding = {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_sha256": payload["artifact_sha256"],
        "implementation_commit_h1": h1,
        "cohort_lock_commit_h2": h2_commit,
    }
    return payload, binding, source


def _lock_case_identity(lock: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    _require(
        isinstance(case_id, str) and case_id in lock["selected_case_ids"],
        "case is outside the exact H2-locked cohort",
    )
    matches: list[dict[str, Any]] = []
    for stratum, object_records in lock["cohort"].items():
        for object_record in object_records:
            for episode in object_record["episodes"]:
                if episode["case_id"] == case_id:
                    matches.append(
                        {
                            "case_id": case_id,
                            "stratum": stratum,
                            "object_id": object_record["object_id"],
                            "episode_id": episode["episode_id"],
                        }
                    )
    _require(len(matches) == 1, "locked case identity is ambiguous")
    return matches[0]


def _snapshot_prediction_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    _require(
        isinstance(arrays, Mapping) and set(arrays) == set(ARRAY_ROLES),
        "prediction arrays must contain the exact six frozen roles",
    )
    snapshots: dict[str, np.ndarray] = {}
    reference_shape: tuple[int, ...] | None = None
    reference_dtype: np.dtype[Any] | None = None
    for role in ARRAY_ROLES:
        source = np.asarray(arrays[role])
        _require(
            source.ndim == 3
            and source.shape[0] == FRAME_COUNT
            and source.shape[1] > CENTER_COUNT
            and source.shape[2] == 3,
            f"{role} must have shape (76, N, 3) with N > 16",
        )
        _require(
            np.issubdtype(source.dtype, np.floating)
            and source.dtype.hasobject is False,
            f"{role} must have a non-object floating dtype",
        )
        _require(np.all(np.isfinite(source)), f"{role} contains a non-finite value")
        if reference_shape is None:
            reference_shape = source.shape
            reference_dtype = source.dtype
        _require(source.shape == reference_shape, f"{role} shape changed")
        _require(source.dtype == reference_dtype, f"{role} dtype changed")
        snapshot = np.array(source, copy=True, order="C")
        snapshot.setflags(write=False)
        snapshots[role] = snapshot
    frame_zero = snapshots["physical_prior_m"][0]
    for role in ARRAY_ROLES[1:]:
        _require(
            array_sha256(snapshots[role][0]) == array_sha256(frame_zero),
            f"{role} frame zero differs from the physical material identity",
        )
    return snapshots


def _normalize_selected_cameras(
    selected_cameras_by_budget: Mapping[int, Sequence[str]],
) -> dict[str, list[str]]:
    _require(
        isinstance(selected_cameras_by_budget, Mapping)
        and set(selected_cameras_by_budget) == set(CAMERA_BUDGETS),
        "selected-camera budgets must be exactly integer keys 4 and 8",
    )
    normalized: dict[int, tuple[str, ...]] = {}
    for budget in CAMERA_BUDGETS:
        cameras = tuple(selected_cameras_by_budget[budget])
        _require(
            len(cameras) == budget
            and len(set(cameras)) == budget
            and all(isinstance(camera, str) and camera for camera in cameras),
            f"{budget}-view selected cameras are invalid",
        )
        normalized[budget] = cameras
    _require(
        normalized[8][:4] == normalized[4],
        "four-view cameras must be the exact ordered prefix of eight-view cameras",
    )
    return {str(budget): list(normalized[budget]) for budget in CAMERA_BUDGETS}


def _validate_covariance_routing(
    diagnostic: Mapping[str, Any],
    selected_cameras: Mapping[str, list[str]],
    *,
    point_count: int,
    center_ids: Sequence[int],
    arrays: Mapping[str, np.ndarray] | None = None,
    retained_technical_failure: bool = False,
) -> dict[str, Any]:
    value = _json_copy(diagnostic, label="covariance/routing diagnostic")
    _require(isinstance(value, dict), "covariance/routing diagnostic must be an object")
    _reject_evaluation_fields(value, label="covariance/routing diagnostic")
    _require(
        value.get("protocol_id") == ADAPTIVE_COVARIANCE_PROTOCOL_ID,
        "covariance/routing diagnostic protocol changed",
    )
    fallback = value.get("fallback")
    fallback_trajectory = (
        "persistence" if retained_technical_failure else "physical_prior"
    )
    _require(
        isinstance(fallback, dict)
        and fallback
        == {
            "trajectory": fallback_trajectory,
            "rbf_state_update": False,
            "bit_exact": True,
        },
        "physical fallback contract changed",
    )
    updates = value.get("updates")
    _require(
        isinstance(updates, list) and len(updates) == len(UPDATE_FRAMES),
        "routing must contain exactly three frozen updates",
    )
    frames: list[int] = []
    centers = set(center_ids)
    for expected_frame, expected_stop, update in zip(
        UPDATE_FRAMES,
        UPDATE_STOPS,
        updates,
        strict=True,
    ):
        _require(isinstance(update, dict), "routing update must be an object")
        frame = update.get("frame")
        stop = update.get("stop_frame_exclusive")
        route = update.get("route")
        _require(
            frame == expected_frame,
            "routing update frames must be exactly 19, 38, 57",
        )
        _require(
            stop == expected_stop,
            "routing interval stops must be exactly 38, 57, 76",
        )
        _require(route in ALLOWED_ROUTES, "routing disposition changed")
        if retained_technical_failure:
            _require(
                route == "physical_prior_fallback",
                "retained technical failure must use physical fallback",
            )
        frames.append(frame)
        expected_budget = {
            "4_view_rbf": 4,
            "8_view_rbf": 8,
            "physical_prior_fallback": None,
        }[route]
        _require(
            update.get("selected_camera_budget") == expected_budget,
            "route and selected camera budget differ",
        )
        attempted_budgets = (4,) if route == "4_view_rbf" else CAMERA_BUDGETS
        budget_diagnostics = update.get("budget_diagnostics")
        _require(
            isinstance(budget_diagnostics, dict)
            and set(budget_diagnostics)
            == {str(budget) for budget in attempted_budgets},
            "attempted covariance budgets changed",
        )
        for budget in attempted_budgets:
            record = budget_diagnostics[str(budget)]
            _require(
                isinstance(record, dict)
                and type(record.get("valid_covariance_center_count")) is int
                and 0 <= record["valid_covariance_center_count"] <= CENTER_COUNT
                and type(record.get("reliable")) is bool,
                "covariance routing record is malformed",
            )
            valid_center_ids = record.get("valid_covariance_center_ids")
            _require(
                isinstance(valid_center_ids, list)
                and len(valid_center_ids) == record["valid_covariance_center_count"]
                and all(
                    type(center_id) is int
                    and 0 <= center_id < point_count
                    and center_id in centers
                    for center_id in valid_center_ids
                )
                and len(set(valid_center_ids)) == len(valid_center_ids),
                "valid covariance center IDs are invalid",
            )
            dispersion = record.get("normalized_covariance_dispersion")
            _require(
                dispersion is None
                or (
                    type(dispersion) in (int, float)
                    and np.isfinite(dispersion)
                    and dispersion >= 0.0
                ),
                "normalized covariance dispersion is invalid",
            )
        expected_reliability = {
            "4_view_rbf": {"4": True},
            "8_view_rbf": {"4": False, "8": True},
            "physical_prior_fallback": {"4": False, "8": False},
        }[route]
        _require(
            {key: budget_diagnostics[key]["reliable"] for key in budget_diagnostics}
            == expected_reliability,
            "route disagrees with covariance reliability",
        )
        activated = selected_cameras["4" if route == "4_view_rbf" else "8"]
        tracked = update.get("tracked_cameras")
        _require(
            isinstance(tracked, list)
            and tracked == activated
            and update.get("tracked_camera_count") == len(activated),
            "routing diagnostic tracked cameras differ from the nested plan",
        )
        correction_expected = route != "physical_prior_fallback"
        _require(
            update.get("rbf_correction_applied") is correction_expected
            and update.get("state_updated") is correction_expected,
            "routing correction/state disposition changed",
        )
        if route == "physical_prior_fallback":
            selected_fallback = (
                "persistence" if retained_technical_failure else "physical_prior"
            )
            _require(
                update.get("selected_backbone") == selected_fallback,
                "physical fallback selected the wrong frozen backbone",
            )
            if retained_technical_failure:
                _require(
                    update.get("camera_streams_charged_as_attempted") is True
                    and update.get("dynamic_observation_available") is False
                    and update.get("tracker_inference_executed") is False,
                    "retained failure camera-attempt disposition changed",
                )
            if arrays is not None:
                interval = slice(frame + 1, stop)
                baseline = arrays[
                    (
                        "persistence_m"
                        if retained_technical_failure
                        else "physical_prior_m"
                    )
                ][interval]
                _require(
                    np.array_equal(
                        arrays["adaptive_prediction_m"][interval],
                        baseline,
                    )
                    and np.array_equal(
                        arrays["selected_raw_prediction_m"][interval],
                        baseline,
                    ),
                    "fallback interval is not bit-exact",
                )
        else:
            _require(
                update.get("selected_backbone") in {"physical_prior", "persistence"},
                "accepted route selected an invalid backbone",
            )
    _require(
        tuple(frames) == UPDATE_FRAMES,
        "routing update frames changed",
    )
    if retained_technical_failure and arrays is not None:
        persistence = arrays["persistence_m"]
        _require(
            all(np.array_equal(arrays[role], persistence) for role in ARRAY_ROLES),
            "retained technical failure roles are not all bit-exact persistence",
        )
    return value


def _validate_technical_disposition(
    technical_disposition: Mapping[str, Any],
    *,
    point_count: int,
) -> dict[str, Any]:
    value = _json_copy(technical_disposition, label="technical disposition")
    _require(isinstance(value, dict), "technical disposition must be an object")
    _reject_evaluation_fields(
        value,
        label="technical disposition",
        allowed_false_keys=frozenset({"disposition_based_on_target_or_outcome"}),
    )
    _require(
        value.get("status") in {"prediction_complete", "retained_technical_failure"},
        "technical disposition status is missing",
    )
    _require(
        value.get("case_retained") is True,
        "technical disposition may not exclude the locked case",
    )
    _require(
        value.get("disposition_based_on_target_or_outcome") is False,
        "technical disposition crossed the target/outcome boundary",
    )
    center_ids = value.get("center_ids")
    _require(
        isinstance(center_ids, list)
        and len(center_ids) == CENTER_COUNT
        and len(set(center_ids)) == CENTER_COUNT
        and all(
            type(center_id) is int and 0 <= center_id < point_count
            for center_id in center_ids
        ),
        "technical disposition center_ids must be 16 unique in-range integers",
    )
    if value["status"] == "retained_technical_failure":
        _require(
            value.get("failure_code") in RETAINED_FAILURE_CODES
            and value.get("fallback_label") == "persistence_only",
            "retained technical failure metadata is incomplete",
        )
    return value


def _diagnostic_payload(
    *,
    identity: Mapping[str, Any],
    selected_cameras: Mapping[str, list[str]],
    covariance_routing: Mapping[str, Any],
    technical_disposition: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CASE_DIAGNOSTIC_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_identity": dict(identity),
        "nested_selected_cameras": dict(selected_cameras),
        "covariance_routing": dict(covariance_routing),
        "technical_disposition": dict(technical_disposition),
        "information_boundary": dict(TARGET_FREE_BOUNDARY),
    }
    payload["artifact_sha256"] = artifact_sha256(payload)
    return payload


def _array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        role: {
            "dtype": arrays[role].dtype.str,
            "shape": list(arrays[role].shape),
            "array_sha256": array_sha256(arrays[role]),
        }
        for role in ARRAY_ROLES
    }


def _write_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_file(path, serialized)


def _publish_directory_noreplace(staging: Path, destination: Path) -> None:
    claim = destination.parent / f".{destination.name}.publish-claim"
    descriptor: int | None = None
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _require(
            not destination.exists() and not destination.is_symlink(),
            f"case seal output appeared during publication: {destination}",
        )
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
    except FileExistsError as error:
        raise ValueError(
            f"case seal publication is already in progress: {claim}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        claim.unlink(missing_ok=True)


def _case_manifest_payload(
    *,
    identity: Mapping[str, Any],
    lock_binding: Mapping[str, Any],
    archive_path: Path,
    array_records: Mapping[str, Any],
    diagnostic_path: Path,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    archive_bytes = _stable_regular_file_bytes(archive_path)[1]
    diagnostic_bytes = _stable_regular_file_bytes(diagnostic_path)[1]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": CASE_SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_identity": dict(identity),
        "lock_binding": dict(lock_binding),
        "content": {
            "prediction_archive": {
                "filename": ARRAY_ARCHIVE_FILENAME,
                "size_bytes": len(archive_bytes),
                "file_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                "arrays": dict(array_records),
            },
            "target_free_diagnostic": {
                "filename": DIAGNOSTIC_FILENAME,
                "size_bytes": len(diagnostic_bytes),
                "file_sha256": hashlib.sha256(diagnostic_bytes).hexdigest(),
                "artifact_sha256": diagnostic["artifact_sha256"],
            },
        },
        "information_boundary": dict(TARGET_FREE_BOUNDARY),
    }
    payload["artifact_sha256"] = artifact_sha256(payload)
    return payload


def _validate_archive_bytes(
    payload: bytes,
    records: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(ARRAY_ROLES),
                "prediction archive roles changed",
            )
            arrays = {role: np.asarray(stored[role]).copy() for role in ARRAY_ROLES}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("prediction archive is invalid") from error
    snapshots = _snapshot_prediction_arrays(arrays)
    _require(
        isinstance(records, Mapping) and set(records) == set(ARRAY_ROLES),
        "manifest array records changed",
    )
    for role in ARRAY_ROLES:
        expected = {
            "dtype": snapshots[role].dtype.str,
            "shape": list(snapshots[role].shape),
            "array_sha256": array_sha256(snapshots[role]),
        }
        _require(records[role] == expected, f"{role} content hash changed")
    return snapshots


def _validate_case_with_lock(
    case_dir: str | Path,
    lock: Mapping[str, Any],
    lock_binding: Mapping[str, Any],
    *,
    expected_case_id: str,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    root = Path(case_dir).absolute()
    _require(root.resolve(strict=True) == root, "case seal root is noncanonical")
    _require(root.is_dir() and not root.is_symlink(), "case seal root is invalid")
    if require_directory_name:
        _require(root.name == expected_case_id, "case seal directory name changed")
    entries = tuple(sorted(path.name for path in root.iterdir()))
    _require(
        entries
        == tuple(
            sorted(
                (
                    ARRAY_ARCHIVE_FILENAME,
                    DIAGNOSTIC_FILENAME,
                    CASE_MANIFEST_FILENAME,
                )
            )
        ),
        "case seal directory contains missing or extra artifacts",
    )
    identity = _lock_case_identity(lock, expected_case_id)
    _, manifest, manifest_raw = _load_strict_json_file(
        root / CASE_MANIFEST_FILENAME,
        label="case prediction seal",
    )
    _require(
        manifest.get("artifact_sha256") == artifact_sha256(manifest),
        "case prediction seal self-hash changed",
    )
    _require(manifest.get("case_identity") == identity, "case identity changed")
    _require(manifest.get("lock_binding") == lock_binding, "case lock binding changed")
    _require(
        manifest.get("information_boundary") == TARGET_FREE_BOUNDARY,
        "case information boundary changed",
    )
    content = manifest.get("content")
    _require(
        isinstance(content, Mapping)
        and set(content) == {"prediction_archive", "target_free_diagnostic"},
        "case content set changed",
    )

    archive_record = content["prediction_archive"]
    _require(
        isinstance(archive_record, Mapping)
        and set(archive_record) == {"filename", "size_bytes", "file_sha256", "arrays"}
        and archive_record.get("filename") == ARRAY_ARCHIVE_FILENAME,
        "prediction archive binding changed",
    )
    archive_path, archive_raw, _ = _stable_regular_file_bytes(
        root / ARRAY_ARCHIVE_FILENAME
    )
    _require(archive_path.parent == root, "prediction archive path escaped case root")
    _require(
        archive_record.get("size_bytes") == len(archive_raw)
        and archive_record.get("file_sha256")
        == hashlib.sha256(archive_raw).hexdigest(),
        "prediction archive content hash changed",
    )
    arrays = _validate_archive_bytes(archive_raw, archive_record.get("arrays", {}))
    point_count = arrays["physical_prior_m"].shape[1]

    diagnostic_record = content["target_free_diagnostic"]
    _require(
        isinstance(diagnostic_record, Mapping)
        and set(diagnostic_record)
        == {
            "filename",
            "size_bytes",
            "file_sha256",
            "artifact_sha256",
        }
        and diagnostic_record.get("filename") == DIAGNOSTIC_FILENAME,
        "diagnostic binding changed",
    )
    diagnostic_path, diagnostic, diagnostic_raw = _load_strict_json_file(
        root / DIAGNOSTIC_FILENAME,
        label="target-free diagnostic",
    )
    _require(diagnostic_path.parent == root, "diagnostic path escaped case root")
    _require(
        diagnostic_record.get("size_bytes") == len(diagnostic_raw)
        and diagnostic_record.get("file_sha256")
        == hashlib.sha256(diagnostic_raw).hexdigest(),
        "target-free diagnostic content hash changed",
    )
    _require(
        diagnostic.get("artifact_sha256")
        == artifact_sha256(diagnostic)
        == diagnostic_record.get("artifact_sha256"),
        "target-free diagnostic self-hash changed",
    )
    expected_diagnostic_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "case_identity",
        "nested_selected_cameras",
        "covariance_routing",
        "technical_disposition",
        "information_boundary",
        "artifact_sha256",
    }
    _require(
        set(diagnostic) == expected_diagnostic_keys
        and diagnostic.get("schema_version") == SCHEMA_VERSION
        and diagnostic.get("artifact_kind") == CASE_DIAGNOSTIC_KIND
        and diagnostic.get("protocol_id") == PROTOCOL_ID
        and diagnostic.get("case_identity") == identity
        and diagnostic.get("information_boundary") == TARGET_FREE_BOUNDARY,
        "target-free diagnostic schema changed",
    )
    cameras_raw = diagnostic["nested_selected_cameras"]
    _require(
        isinstance(cameras_raw, Mapping) and set(cameras_raw) == {"4", "8"},
        "nested camera diagnostic changed",
    )
    cameras = _normalize_selected_cameras(
        {
            4: cameras_raw["4"],
            8: cameras_raw["8"],
        }
    )
    _require(cameras == cameras_raw, "nested camera diagnostic is noncanonical")
    disposition = _validate_technical_disposition(
        diagnostic["technical_disposition"],
        point_count=point_count,
    )
    _validate_covariance_routing(
        diagnostic["covariance_routing"],
        cameras,
        point_count=point_count,
        center_ids=disposition["center_ids"],
        arrays=arrays,
        retained_technical_failure=(
            disposition["status"] == "retained_technical_failure"
        ),
    )

    expected_manifest = dict(manifest)
    expected_manifest.pop("artifact_sha256", None)
    _require(
        set(expected_manifest)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "case_identity",
            "lock_binding",
            "content",
            "information_boundary",
        }
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_kind") == CASE_SEAL_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID,
        "case prediction seal schema changed",
    )
    return {
        "case_id": expected_case_id,
        "manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_artifact_sha256": manifest["artifact_sha256"],
        "prediction_archive_sha256": archive_record["file_sha256"],
        "diagnostic_file_sha256": diagnostic_record["file_sha256"],
        "diagnostic_artifact_sha256": diagnostic_record["artifact_sha256"],
    }


def seal_confirmation_case(
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    output_dir: str | Path,
    prediction_arrays: Mapping[str, np.ndarray],
    selected_cameras_by_budget: Mapping[int, Sequence[str]],
    covariance_routing_diagnostic: Mapping[str, Any],
    technical_disposition: Mapping[str, Any],
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Atomically seal one exact locked case without opening an evaluation."""

    lock, lock_binding, lock_source = _load_lock_binding(
        lock_path,
        h2_commit,
        expected_h1=expected_h1,
    )
    identity = _lock_case_identity(lock, case_id)
    output = _canonical_absent_path(output_dir, label="case seal output")
    _require(output.name == case_id, "case seal directory must use the exact case ID")
    _require(
        not _paths_overlap(output, lock_source),
        "case seal output overlaps the H2 lock",
    )
    arrays = _snapshot_prediction_arrays(prediction_arrays)
    selected_cameras = _normalize_selected_cameras(selected_cameras_by_budget)
    disposition = _validate_technical_disposition(
        technical_disposition,
        point_count=arrays["physical_prior_m"].shape[1],
    )
    routing = _validate_covariance_routing(
        covariance_routing_diagnostic,
        selected_cameras,
        point_count=arrays["physical_prior_m"].shape[1],
        center_ids=disposition["center_ids"],
        arrays=arrays,
        retained_technical_failure=(
            disposition["status"] == "retained_technical_failure"
        ),
    )
    diagnostic = _diagnostic_payload(
        identity=identity,
        selected_cameras=selected_cameras,
        covariance_routing=routing,
        technical_disposition=disposition,
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        archive_path = staging / ARRAY_ARCHIVE_FILENAME
        with archive_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{role: arrays[role] for role in ARRAY_ROLES},
            )
            handle.flush()
            os.fsync(handle.fileno())
        diagnostic_path = staging / DIAGNOSTIC_FILENAME
        _write_json(diagnostic_path, diagnostic)
        manifest = _case_manifest_payload(
            identity=identity,
            lock_binding=lock_binding,
            archive_path=archive_path,
            array_records=_array_records(arrays),
            diagnostic_path=diagnostic_path,
            diagnostic=diagnostic,
        )
        _write_json(staging / CASE_MANIFEST_FILENAME, manifest)
        _fsync_directory(staging)
        _validate_case_with_lock(
            staging,
            lock,
            lock_binding,
            expected_case_id=case_id,
            require_directory_name=False,
        )
        _, current_binding, current_source = _load_lock_binding(
            lock_path,
            h2_commit,
            expected_h1=expected_h1,
        )
        _require(
            current_binding == lock_binding and current_source == lock_source,
            "H2 lock changed before publication",
        )
        _publish_directory_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_confirmation_case_seal(
        output,
        lock_path,
        h2_commit,
        expected_case_id=case_id,
        expected_h1=expected_h1,
    )


def validate_confirmation_case_seal(
    case_dir: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    *,
    expected_case_id: str | None = None,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Replay a case seal and every array/diagnostic content binding."""

    lock, lock_binding, lock_source = _load_lock_binding(
        lock_path,
        h2_commit,
        expected_h1=expected_h1,
    )
    root = Path(case_dir).absolute()
    case_id = expected_case_id or root.name
    _require(
        not _paths_overlap(root, lock_source),
        "case seal root overlaps the H2 lock",
    )
    return _validate_case_with_lock(
        root,
        lock,
        lock_binding,
        expected_case_id=case_id,
    )


def _collect_case_records(
    lock: Mapping[str, Any],
    lock_binding: Mapping[str, Any],
    lock_source: Path,
    case_seal_dirs: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    expected_cases = tuple(lock["selected_case_ids"])
    _require(
        isinstance(case_seal_dirs, Mapping)
        and set(case_seal_dirs) == set(expected_cases),
        "cohort barrier requires every exact locked case and no extras",
    )
    normalized_roots = {
        case_id: Path(case_seal_dirs[case_id]).absolute() for case_id in expected_cases
    }
    roots = tuple(normalized_roots.values())
    _require(
        len(set(roots)) == len(roots),
        "cohort case seal roots are duplicated",
    )
    for index, left in enumerate(roots):
        _require(
            not _paths_overlap(left, lock_source),
            "cohort case seal overlaps the H2 lock",
        )
        for right in roots[index + 1 :]:
            _require(
                not _paths_overlap(left, right),
                "cohort case seal roots overlap",
            )
    return [
        _validate_case_with_lock(
            normalized_roots[case_id],
            lock,
            lock_binding,
            expected_case_id=case_id,
        )
        for case_id in expected_cases
    ]


def _barrier_payload(
    *,
    lock: Mapping[str, Any],
    lock_binding: Mapping[str, Any],
    case_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": COHORT_BARRIER_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "complete-target-free-cohort-prediction-barrier",
        "lock_binding": dict(lock_binding),
        "exact_case_ids": list(lock["selected_case_ids"]),
        "case_count": len(lock["selected_case_ids"]),
        "ordered_case_seals": [dict(record) for record in case_records],
        "information_boundary": dict(TARGET_FREE_BOUNDARY),
    }
    payload["artifact_sha256"] = artifact_sha256(payload)
    return payload


def _write_new_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"barrier output already exists: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def create_confirmation_prediction_barrier(
    output_path: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    case_seal_dirs: Mapping[str, str | Path],
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Publish the barrier only after the complete exact cohort validates."""

    lock, lock_binding, lock_source = _load_lock_binding(
        lock_path,
        h2_commit,
        expected_h1=expected_h1,
    )
    output = _canonical_absent_path(output_path, label="cohort barrier output")
    _require(
        not _paths_overlap(output, lock_source),
        "cohort barrier output overlaps the H2 lock",
    )
    for case_id, case_dir in case_seal_dirs.items():
        _require(
            not _paths_overlap(output, Path(case_dir).absolute()),
            f"cohort barrier output overlaps case seal {case_id}",
        )
    first_records = _collect_case_records(
        lock,
        lock_binding,
        lock_source,
        case_seal_dirs,
    )
    payload = _barrier_payload(
        lock=lock,
        lock_binding=lock_binding,
        case_records=first_records,
    )

    # A second complete pass immediately before publication closes the ordinary
    # validate-then-write window.  Any changed manifest or content hash aborts.
    current_lock, current_binding, current_source = _load_lock_binding(
        lock_path,
        h2_commit,
        expected_h1=expected_h1,
    )
    _require(
        current_binding == lock_binding
        and current_lock == lock
        and current_source == lock_source,
        "H2 lock changed before cohort barrier publication",
    )
    second_records = _collect_case_records(
        current_lock,
        current_binding,
        current_source,
        case_seal_dirs,
    )
    _require(
        second_records == first_records,
        "one or more case seals changed before cohort barrier publication",
    )
    _write_new_json_atomic(output, payload)
    return validate_confirmation_prediction_barrier(
        output,
        lock_path,
        h2_commit,
        case_seal_dirs,
        expected_h1=expected_h1,
    )


def validate_confirmation_prediction_barrier(
    barrier_path: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    case_seal_dirs: Mapping[str, str | Path],
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Replay the complete barrier against the current H2 lock and case seals."""

    lock, lock_binding, lock_source = _load_lock_binding(
        lock_path,
        h2_commit,
        expected_h1=expected_h1,
    )
    source, barrier, _ = _load_strict_json_file(
        barrier_path,
        label="cohort prediction barrier",
    )
    _require(
        not _paths_overlap(source, lock_source),
        "cohort barrier overlaps the H2 lock",
    )
    _require(
        barrier.get("artifact_sha256") == artifact_sha256(barrier),
        "cohort prediction barrier self-hash changed",
    )
    expected_cases = list(lock["selected_case_ids"])
    _require(
        barrier.get("exact_case_ids") == expected_cases
        and barrier.get("case_count") == len(expected_cases),
        "cohort prediction barrier case set changed",
    )
    records = barrier.get("ordered_case_seals")
    _require(
        isinstance(records, list)
        and all(isinstance(record, Mapping) for record in records),
        "cohort prediction barrier records are malformed",
    )
    _require(
        [record.get("case_id") for record in records] == expected_cases,
        "cohort prediction barrier case order changed",
    )
    _require(
        isinstance(case_seal_dirs, Mapping)
        and set(case_seal_dirs) == set(expected_cases),
        "barrier validation requires every exact current case seal path",
    )
    for case_id, case_dir in case_seal_dirs.items():
        _require(
            not _paths_overlap(source, Path(case_dir).absolute()),
            f"cohort barrier overlaps case seal {case_id}",
        )
    current_records = _collect_case_records(
        lock,
        lock_binding,
        lock_source,
        case_seal_dirs,
    )
    expected = _barrier_payload(
        lock=lock,
        lock_binding=lock_binding,
        case_records=current_records,
    )
    _require(barrier == expected, "cohort prediction barrier or case content changed")
    return barrier


__all__ = [
    "ARRAY_ARCHIVE_FILENAME",
    "ARRAY_ROLES",
    "CAMERA_BUDGETS",
    "CASE_DIAGNOSTIC_KIND",
    "CASE_MANIFEST_FILENAME",
    "CASE_SEAL_KIND",
    "COHORT_BARRIER_KIND",
    "DIAGNOSTIC_FILENAME",
    "TARGET_FREE_BOUNDARY",
    "array_sha256",
    "artifact_sha256",
    "create_confirmation_prediction_barrier",
    "seal_confirmation_case",
    "validate_confirmation_case_seal",
    "validate_confirmation_prediction_barrier",
]
