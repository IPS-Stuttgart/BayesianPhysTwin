"""Permit-gated target/query split for the Deform360 held v8 protocol.

The official reconstruction is materialized once, behind a target-
reconstruction permit, into two deliberately asymmetric artifacts:

* a full, future-bearing official target which is opened only after a later
  future-score permit; and
* an independent frame-zero query artifact containing exactly identity IDs
  and coordinates.  It is created through
  :mod:`deform360_held_v8_query_artifacts` and does not bind the future-bearing
  container.

The permit implementations live in the prospective v8 protocol module.  This
module accepts narrow injected consumers so importing it cannot accidentally
fall back to a v7 capability.  Each consumer must validate and consume its
single-use capability for the declared case before returning.

No source-to-target assignment, identity transport, target-dependent query,
or score computation is performed here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from typing import Any, Literal, Protocol

import numpy as np

from . import deform360_held_v8_query_artifacts as query_artifacts


PROTOCOL_ID = "deform360-held-online-belief-v8"
SCHEMA_VERSION = 1
FRAME_COUNT = 76
CENTER_COUNT = 16

OFFICIAL_TARGET_KIND = "Deform360HeldV8OfficialTarget"
TARGET_ARRAY_NAMES = frozenset(
    {
        "identity_ids",
        "object_points",
        "object_visibilities",
        "object_motions_valid",
    }
)
TARGET_RECONSTRUCTION_OPERATION = "create-official-target-v1"
FUTURE_SCORE_OPERATION = "read-official-target-for-score-v1"

_SEALED_FILE_MODE = 0o400
_FILE_RECORD_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_ARRAY_RECORD_FIELDS = frozenset({"shape", "dtype", "sha256"})
_SHA256_LENGTH = 64
_ROLE_VALUES = frozenset({"calibration", "confirmation"})
_CANONICAL_FLOAT32 = np.dtype("<f4")
_CANONICAL_INT64 = np.dtype("<i8")


class PermitConsumer(Protocol):
    """Injected single-use permit validator owned by the v8 protocol.

    Implementations must fail closed unless ``permit`` authorizes exactly the
    declared case and operation.  Successful return consumes the capability.
    The returned mapping is immutable audit evidence embedded in the artifact
    or retained by the scoring adapter.
    """

    def __call__(
        self,
        permit: object,
        *,
        case_name: str,
        operation: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CanonicalOfficialTarget:
    """Canonical arrays written to the v8 full target archive."""

    identity_ids: np.ndarray
    object_points: np.ndarray
    object_visibilities: np.ndarray
    object_motions_valid: np.ndarray
    identity_rule: str
    reconstruction_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class DirectScoringInputs:
    """Validated keyword inputs for v8 direct-official-identity scoring."""

    case_name: str
    object_id: str
    primary_prediction_m: np.ndarray
    selected_raw_backbone_m: np.ndarray
    queried_identity_ids: np.ndarray
    target_identity_ids: np.ndarray
    official_frame_zero_m: np.ndarray
    target_points_m: np.ndarray
    object_visibilities: np.ndarray
    object_motions_valid: np.ndarray
    shared_support_mask: np.ndarray
    center_exclusion_mask: np.ndarray
    frame_indices: np.ndarray
    source_node_count: int | None
    permit_evidence: Mapping[str, Any]

    def scoring_kwargs(self) -> dict[str, Any]:
        """Return the exact arguments accepted by the pure v8 scorer."""

        return {
            "case_name": self.case_name,
            "object_id": self.object_id,
            "primary_prediction_m": self.primary_prediction_m,
            "selected_raw_backbone_m": self.selected_raw_backbone_m,
            "queried_identity_ids": self.queried_identity_ids,
            "target_identity_ids": self.target_identity_ids,
            "official_frame_zero_m": self.official_frame_zero_m,
            "target_points_m": self.target_points_m,
            "object_visibilities": self.object_visibilities,
            "object_motions_valid": self.object_motions_valid,
            "shared_support_mask": self.shared_support_mask,
            "center_exclusion_mask": self.center_exclusion_mask,
            "frame_indices": self.frame_indices,
            "source_node_count": self.source_node_count,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("artifact evidence is not canonical JSON") from error


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = deepcopy(dict(value))
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_canonical_directory(path: Path) -> None:
    observed = os.lstat(path)
    _require(not stat.S_ISLNK(observed.st_mode), f"{path} is a symlink")
    _require(stat.S_ISDIR(observed.st_mode), f"{path} is not a directory")
    _require(path.resolve() == path, f"{path} has a symlinked ancestor")


def _prepare_destination_parent(destination: Path) -> None:
    existing = destination.parent
    while not os.path.lexists(existing):
        parent = existing.parent
        _require(parent != existing, "destination has no existing ancestor")
        existing = parent
    _require_canonical_directory(existing)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_canonical_directory(destination.parent)


def _preflight_new_paths(paths: tuple[str | Path, ...]) -> tuple[Path, ...]:
    canonical = tuple(_canonical_path(path) for path in paths)
    _require(len(set(canonical)) == len(canonical), "artifact paths must be distinct")
    for path in canonical:
        _prepare_destination_parent(path)
        _require(not os.path.lexists(path), f"destination already exists: {path}")
    return canonical


def _open_regular_file_snapshot(
    path: str | Path,
) -> tuple[Path, int, os.stat_result]:
    source = _canonical_path(path)
    before = os.lstat(source)
    _require(not stat.S_ISLNK(before.st_mode), f"{source} is a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{source} is not a regular file")
    _require(source.resolve() == source, f"{source} has a symlinked ancestor")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{source} changed while opening",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return source, descriptor, opened


def _read_file_bytes(path: str | Path) -> tuple[Path, bytes, os.stat_result]:
    source, descriptor, opened = _open_regular_file_snapshot(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        current = os.lstat(source)
        identity = (opened.st_dev, opened.st_ino)
        _require(
            stat.S_ISREG(current.st_mode)
            and (after.st_dev, after.st_ino) == identity
            and (current.st_dev, current.st_ino) == identity
            and after.st_size == opened.st_size
            and after.st_mtime_ns == opened.st_mtime_ns
            and after.st_ctime_ns == opened.st_ctime_ns,
            f"{source} changed while reading",
        )
    finally:
        os.close(descriptor)
    return source, payload, after


def _bound_file(path: str | Path) -> dict[str, Any]:
    source, payload, observed = _read_file_bytes(path)
    return {
        "path": str(source),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": observed.st_size,
    }


def _require_exact_file_mode(path: str | Path, *, role: str) -> None:
    source = _canonical_path(path)
    observed = os.lstat(source)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == _SEALED_FILE_MODE,
        f"{role} must be a regular non-symlink file with mode 0400",
    )


def _validate_bound_file(record: object, *, role: str) -> Path:
    _require(
        isinstance(record, Mapping) and set(record) == _FILE_RECORD_FIELDS,
        f"{role} file record fields changed",
    )
    path = record.get("path")
    _require(isinstance(path, str) and path, f"{role} path is missing")
    _require_exact_file_mode(path, role=role)
    observed = _bound_file(path)
    _require(observed == dict(record), f"{role} file binding changed")
    return Path(observed["path"])


def _load_json(path: str | Path) -> dict[str, Any]:
    source, payload, _ = _read_file_bytes(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} is not canonical JSON input") from error
    _require(isinstance(value, dict), f"{source} must contain a JSON object")
    return value


def _write_new_bytes(path: str | Path, payload: bytes) -> Path:
    destination = _canonical_path(path)
    _prepare_destination_parent(destination)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        _SEALED_FILE_MODE,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination, _SEALED_FILE_MODE, follow_symlinks=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _write_new_bytes(path, payload)


def _write_new_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    destination = _canonical_path(path)
    _prepare_destination_parent(destination)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        _SEALED_FILE_MODE,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination, _SEALED_FILE_MODE, follow_symlinks=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _sha256_array(array),
    }


def _array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: _array_record(arrays[name]) for name in sorted(arrays)}


def _load_npz_arrays(path: str | Path) -> tuple[Path, dict[str, np.ndarray]]:
    source, payload, _ = _read_file_bytes(path)
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{source} is not a valid non-pickle NPZ archive") from error
    return source, arrays


def _validate_array_records(
    records: object,
    arrays: Mapping[str, np.ndarray],
    *,
    role: str,
) -> None:
    _require(
        isinstance(records, Mapping) and set(records) == set(arrays),
        f"{role} array-record set changed",
    )
    for name, array in arrays.items():
        record = records.get(name)
        _require(
            isinstance(record, Mapping) and set(record) == _ARRAY_RECORD_FIELDS,
            f"{role} {name} array-record fields changed",
        )
        _require(
            dict(record) == _array_record(array),
            f"{role} {name} array binding changed",
        )


def _bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes()
        == np.ascontiguousarray(right_array).tobytes()
    )


def _immutable_copy(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _validate_lock(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    _require_exact_file_mode(path, role="held v8 lock")
    record = _bound_file(path)
    if expected_sha256 is not None:
        _require(_valid_sha256(expected_sha256), "lock SHA-256 is invalid")
        _require(record["sha256"] == expected_sha256, "lock SHA-256 changed")
    lock = _load_json(path)
    _require(lock.get("protocol_id") == PROTOCOL_ID, "lock protocol changed")
    if "artifact_sha256" in lock:
        _require(
            lock.get("artifact_sha256") == _artifact_sha256(lock),
            "lock content checksum changed",
        )
    return record


def _json_mapping(value: object, *, role: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{role} must return audit evidence")
    result = deepcopy(dict(value))
    _canonical_bytes(result)
    return result


def _extract_reconstruction_value(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        _require(name in value, f"official reconstruction lacks {name}")
        return value[name]
    _require(hasattr(value, name), f"official reconstruction lacks {name}")
    return getattr(value, name)


def _optional_reconstruction_value(
    value: object,
    name: str,
    default: object,
) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def canonicalize_official_reconstruction(value: object) -> CanonicalOfficialTarget:
    """Canonicalize an existing official reconstruction without using colour.

    The caller must invoke this only after consuming a target-reconstruction
    permit.  Floating coordinates are materialized as little/native float32;
    identities are canonical int64.  If the upstream reconstruction has no
    identity array, its stable point-axis order is named explicitly by
    ``arange(M)``.
    """

    points_raw = np.asarray(_extract_reconstruction_value(value, "object_points"))
    visible_raw = np.asarray(
        _extract_reconstruction_value(value, "object_visibilities")
    )
    valid_raw = np.asarray(_extract_reconstruction_value(value, "object_motions_valid"))
    _require(
        np.issubdtype(points_raw.dtype, np.floating)
        and points_raw.ndim == 3
        and points_raw.shape[0] == FRAME_COUNT
        and points_raw.shape[2] == 3,
        "official object_points must have floating shape (76, M, 3)",
    )
    identity_count = points_raw.shape[1]
    _require(
        identity_count >= CENTER_COUNT,
        "official reconstruction has fewer than 16 identities",
    )
    _require(
        visible_raw.dtype == np.dtype(bool)
        and valid_raw.dtype == np.dtype(bool)
        and visible_raw.shape == valid_raw.shape == points_raw.shape[:2],
        "official visibility and validity must have bool shape (76, M)",
    )
    points = np.ascontiguousarray(points_raw, dtype=_CANONICAL_FLOAT32)
    _require(
        np.all(np.isfinite(points[0])),
        "official frame-zero coordinates must be finite",
    )
    identities_raw = _optional_reconstruction_value(value, "identity_ids", None)
    if identities_raw is None:
        identities = np.arange(identity_count, dtype=_CANONICAL_INT64)
        identity_rule = "implicit-point-axis-order-arange-v1"
    else:
        raw = np.asarray(identities_raw)
        _require(
            raw.ndim == 1
            and raw.shape == (identity_count,)
            and np.issubdtype(raw.dtype, np.integer),
            "official identity_ids must have integer shape (M,)",
        )
        _require(
            np.all(raw >= np.iinfo(np.int64).min)
            and np.all(raw <= np.iinfo(np.int64).max),
            "official identity_ids exceed int64",
        )
        identities = np.ascontiguousarray(raw, dtype=_CANONICAL_INT64)
        _require(
            np.all(identities[1:] > identities[:-1]),
            "official identity_ids must be strictly increasing",
        )
        identity_rule = "explicit-strictly-increasing-int64-v1"
    provenance = _json_mapping(
        _optional_reconstruction_value(value, "provenance", {}),
        role="official reconstruction provenance",
    )
    return CanonicalOfficialTarget(
        identity_ids=identities.copy(),
        object_points=points.copy(),
        object_visibilities=np.ascontiguousarray(visible_raw, dtype=bool),
        object_motions_valid=np.ascontiguousarray(valid_raw, dtype=bool),
        identity_rule=identity_rule,
        reconstruction_provenance=provenance,
    )


def _target_arrays(target: CanonicalOfficialTarget) -> dict[str, np.ndarray]:
    arrays = {
        "identity_ids": np.asarray(target.identity_ids),
        "object_points": np.asarray(target.object_points),
        "object_visibilities": np.asarray(target.object_visibilities),
        "object_motions_valid": np.asarray(target.object_motions_valid),
    }
    _validate_target_arrays(arrays)
    return arrays


def _validate_target_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    _require(
        set(arrays) == TARGET_ARRAY_NAMES,
        "official target archive array set changed",
    )
    identities = np.asarray(arrays["identity_ids"])
    points = np.asarray(arrays["object_points"])
    visible = np.asarray(arrays["object_visibilities"])
    valid = np.asarray(arrays["object_motions_valid"])
    _require(
        identities.dtype == _CANONICAL_INT64
        and identities.ndim == 1
        and len(identities) >= CENTER_COUNT
        and np.all(identities[1:] > identities[:-1]),
        "official target identity_ids must be strictly increasing int64",
    )
    _require(
        points.dtype == _CANONICAL_FLOAT32
        and points.shape == (FRAME_COUNT, len(identities), 3),
        "official target object_points must have float32 shape (76, M, 3)",
    )
    _require(
        visible.dtype == valid.dtype == np.dtype(bool)
        and visible.shape == valid.shape == (FRAME_COUNT, len(identities)),
        "official target masks must have bool shape (76, M)",
    )
    _require(
        np.all(np.isfinite(points[0])),
        "official target frame zero must be finite",
    )


def _cleanup_new_files(paths: tuple[Path, ...]) -> None:
    for path in reversed(paths):
        if os.path.lexists(path):
            observed = os.lstat(path)
            if stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
                os.chmod(path, 0o600, follow_symlinks=False)
                path.unlink(missing_ok=True)


def write_official_target_and_frame_zero_query_artifacts(
    target_archive_path: str | Path,
    target_manifest_path: str | Path,
    official_query_archive_path: str | Path,
    official_query_manifest_path: str | Path,
    *,
    lock_path: str | Path,
    lock_sha256: str,
    case_name: str,
    role: Literal["calibration", "confirmation"],
    target_reconstruction_permit: object,
    consume_target_reconstruction_permit: PermitConsumer,
    reconstruction_loader: Callable[[], object],
) -> dict[str, Any]:
    """Seal a full target and independent x0 query after one v8 permit.

    No reconstruction value is requested before the injected protocol callback
    successfully validates and consumes the case-specific permit.
    """

    _require(isinstance(case_name, str) and case_name, "case_name is missing")
    _require(role in _ROLE_VALUES, "role must be calibration or confirmation")
    _require(callable(reconstruction_loader), "reconstruction_loader is not callable")
    _require(
        callable(consume_target_reconstruction_permit),
        "target-reconstruction permit consumer is not callable",
    )
    destinations = _preflight_new_paths(
        (
            target_archive_path,
            target_manifest_path,
            official_query_archive_path,
            official_query_manifest_path,
        )
    )
    target_archive, target_manifest, query_archive, query_manifest = destinations
    lock_record = _validate_lock(lock_path, expected_sha256=lock_sha256)

    # This call is intentionally the last operation before opening the full
    # official reconstruction.  The protocol owns validation and single use.
    permit_evidence = _json_mapping(
        consume_target_reconstruction_permit(
            target_reconstruction_permit,
            case_name=case_name,
            operation=TARGET_RECONSTRUCTION_OPERATION,
        ),
        role="target-reconstruction permit consumer",
    )
    reconstruction = reconstruction_loader()
    target = canonicalize_official_reconstruction(reconstruction)
    arrays = _target_arrays(target)
    x0_arrays = {
        "identity_ids": arrays["identity_ids"].copy(),
        "positions_m": arrays["object_points"][0].copy(),
    }

    try:
        _write_new_npz(target_archive, arrays)
        query_artifacts.write_official_frame_zero_query_artifact(
            query_archive,
            query_manifest,
            lock_path,
            lock_sha256=lock_sha256,
            case_name=case_name,
            query_arrays=x0_arrays,
        )
        query_value = query_artifacts.validate_official_frame_zero_query_artifact(
            query_manifest,
            lock_path,
            expected_case_name=case_name,
        )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": OFFICIAL_TARGET_KIND,
            "protocol_id": PROTOCOL_ID,
            "case_name": case_name,
            "role": role,
            "lock": lock_record,
            "target_reconstruction_permit_evidence": permit_evidence,
            "identity_rule": target.identity_rule,
            "archive": _bound_file(target_archive),
            "array_records": _array_records(arrays),
            "frame_zero_array_records": {
                "identity_ids": _array_record(arrays["identity_ids"]),
                "positions_m": _array_record(arrays["object_points"][0]),
            },
            "official_query_manifest": _bound_file(query_manifest),
            "official_query_manifest_artifact_sha256": query_value["artifact_sha256"],
            "reconstruction_provenance": target.reconstruction_provenance,
            "canonicalization": {
                "frame_count": FRAME_COUNT,
                "coordinate_dtype": "<f4",
                "identity_dtype": "<i8",
                "visibility_dtype": "|b1",
                "validity_dtype": "|b1",
                "implicit_identity_rule": "arange(M, dtype=int64)",
                "colour_read_or_stored": False,
            },
            "information_boundary": {
                "target_reconstruction_permit_consumed_before_full_target_read": True,
                "full_target_created_after_reconstruction_permit": True,
                "frame_zero_query_copied_from_canonical_full_target": True,
                "frame_zero_query_contains_future_coordinates": False,
                "frame_zero_query_contains_visibility_or_validity": False,
                "frame_zero_query_contains_colour": False,
                "frame_zero_query_binds_future_container": False,
                "source_to_target_assignment_performed": False,
                "identity_transport_performed": False,
                "score_computed": False,
            },
        }
        manifest["artifact_sha256"] = _artifact_sha256(manifest)
        _write_new_json(target_manifest, manifest)
        return validate_official_target_artifact(
            target_manifest,
            lock_path=lock_path,
            expected_case_name=case_name,
            expected_role=role,
        )
    except BaseException:
        # All four paths were proved absent by the exclusive preflight, so any
        # one which now exists belongs to this failed transaction (including a
        # query writer which failed during its own post-write validation).
        _cleanup_new_files(destinations)
        raise


def validate_official_target_artifact(
    manifest_path: str | Path,
    *,
    lock_path: str | Path,
    expected_case_name: str | None = None,
    expected_role: Literal["calibration", "confirmation"] | None = None,
) -> dict[str, Any]:
    """Validate one full target; call only behind an appropriate permit."""

    _require_exact_file_mode(manifest_path, role="official target manifest")
    manifest = _load_json(manifest_path)
    _require(
        set(manifest)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "case_name",
            "role",
            "lock",
            "target_reconstruction_permit_evidence",
            "identity_rule",
            "archive",
            "array_records",
            "frame_zero_array_records",
            "official_query_manifest",
            "official_query_manifest_artifact_sha256",
            "reconstruction_provenance",
            "canonicalization",
            "information_boundary",
            "artifact_sha256",
        },
        "official target manifest fields changed",
    )
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_kind") == OFFICIAL_TARGET_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID,
        "official target schema, kind, or protocol changed",
    )
    _require(
        manifest.get("artifact_sha256") == _artifact_sha256(manifest),
        "official target content checksum changed",
    )
    case_name = manifest.get("case_name")
    role = manifest.get("role")
    _require(isinstance(case_name, str) and case_name, "official target case missing")
    _require(role in _ROLE_VALUES, "official target role changed")
    if expected_case_name is not None:
        _require(case_name == expected_case_name, "official target binds another case")
    if expected_role is not None:
        _require(role == expected_role, "official target binds another role")
    lock_record = _validate_lock(lock_path)
    _require(manifest.get("lock") == lock_record, "official target binds another lock")
    _json_mapping(
        manifest.get("target_reconstruction_permit_evidence"),
        role="target-reconstruction permit evidence",
    )
    _json_mapping(
        manifest.get("reconstruction_provenance"),
        role="reconstruction provenance",
    )
    _require(
        manifest.get("identity_rule")
        in {
            "implicit-point-axis-order-arange-v1",
            "explicit-strictly-increasing-int64-v1",
        },
        "official target identity rule changed",
    )
    archive = _validate_bound_file(manifest.get("archive"), role="official target")
    _, arrays = _load_npz_arrays(archive)
    _validate_target_arrays(arrays)
    _validate_array_records(
        manifest.get("array_records"), arrays, role="official target"
    )
    if manifest["identity_rule"] == "implicit-point-axis-order-arange-v1":
        _require(
            np.array_equal(
                arrays["identity_ids"],
                np.arange(len(arrays["identity_ids"]), dtype=_CANONICAL_INT64),
            ),
            "implicit official identities are not canonical arange(M)",
        )
    expected_x0_records = {
        "identity_ids": _array_record(arrays["identity_ids"]),
        "positions_m": _array_record(arrays["object_points"][0]),
    }
    _require(
        manifest.get("frame_zero_array_records") == expected_x0_records,
        "official target frame-zero array binding changed",
    )
    query_manifest_path = _validate_bound_file(
        manifest.get("official_query_manifest"),
        role="official frame-zero query manifest",
    )
    query_manifest = query_artifacts.validate_official_frame_zero_query_artifact(
        query_manifest_path,
        lock_path,
        expected_case_name=case_name,
    )
    _require(
        manifest.get("official_query_manifest_artifact_sha256")
        == query_manifest["artifact_sha256"],
        "official target binds another frame-zero query",
    )
    query_archive = _validate_bound_file(
        query_manifest["archive"], role="official frame-zero query"
    )
    _, query_arrays = _load_npz_arrays(query_archive)
    _require(
        set(query_arrays) == {"identity_ids", "positions_m"},
        "official frame-zero query contains future or mask arrays",
    )
    _require(
        _bit_equal(query_arrays["identity_ids"], arrays["identity_ids"]),
        "official target and frame-zero query identity bytes differ",
    )
    _require(
        _bit_equal(query_arrays["positions_m"], arrays["object_points"][0]),
        "official target and frame-zero query x0 bytes differ",
    )
    _require(
        manifest.get("canonicalization")
        == {
            "frame_count": FRAME_COUNT,
            "coordinate_dtype": "<f4",
            "identity_dtype": "<i8",
            "visibility_dtype": "|b1",
            "validity_dtype": "|b1",
            "implicit_identity_rule": "arange(M, dtype=int64)",
            "colour_read_or_stored": False,
        },
        "official target canonicalization changed",
    )
    _require(
        manifest.get("information_boundary")
        == {
            "target_reconstruction_permit_consumed_before_full_target_read": True,
            "full_target_created_after_reconstruction_permit": True,
            "frame_zero_query_copied_from_canonical_full_target": True,
            "frame_zero_query_contains_future_coordinates": False,
            "frame_zero_query_contains_visibility_or_validity": False,
            "frame_zero_query_contains_colour": False,
            "frame_zero_query_binds_future_container": False,
            "source_to_target_assignment_performed": False,
            "identity_transport_performed": False,
            "score_computed": False,
        },
        "official target information boundary changed",
    )
    return manifest


def _load_validated_target_arrays(
    manifest_path: str | Path,
    *,
    lock_path: str | Path,
    expected_case_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = validate_official_target_artifact(
        manifest_path,
        lock_path=lock_path,
        expected_case_name=expected_case_name,
    )
    archive = _validate_bound_file(manifest["archive"], role="official target")
    _, arrays = _load_npz_arrays(archive)
    _validate_target_arrays(arrays)
    return manifest, arrays


def _load_validated_queried_prediction_arrays(
    seal_path: str | Path,
    *,
    lock_path: str | Path,
    expected_case_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seal = query_artifacts.validate_queried_prediction_artifact(
        seal_path,
        lock_path=lock_path,
        expected_case_name=expected_case_name,
    )
    archive = _validate_bound_file(seal["archive"], role="queried prediction")
    _, arrays = _load_npz_arrays(archive)
    return seal, arrays


def _case_object_id(case_name: str) -> str:
    _require("-ep" in case_name, "case_name lacks an episode suffix")
    object_id, episode = case_name.rsplit("-ep", maxsplit=1)
    _require(object_id and episode.isdigit(), "case_name episode suffix is invalid")
    return object_id


def _assemble_direct_scoring_inputs(
    *,
    case_name: str,
    queried: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    source_node_count: int | None,
    permit_evidence: Mapping[str, Any],
) -> DirectScoringInputs:
    _validate_target_arrays(target)
    required_query = {
        "primary_prediction_m",
        "selected_raw_backbone_m",
        "identity_ids",
        "positions_m",
        "shared_support_mask",
        "center_exclusion_mask",
        "frame_indices",
    }
    _require(
        required_query.issubset(queried),
        "queried prediction lacks direct-scoring arrays",
    )
    identities = np.asarray(queried["identity_ids"])
    positions = np.asarray(queried["positions_m"])
    _require(
        _bit_equal(identities, target["identity_ids"]),
        "queried and target identity bytes differ",
    )
    _require(
        _bit_equal(positions, target["object_points"][0]),
        "queried and target frame-zero bytes differ",
    )
    if source_node_count is not None:
        _require(
            isinstance(source_node_count, int)
            and not isinstance(source_node_count, bool)
            and source_node_count > 0,
            "source_node_count must be a positive integer when supplied",
        )
    evidence = _json_mapping(permit_evidence, role="future-score permit consumer")
    values = DirectScoringInputs(
        case_name=case_name,
        object_id=_case_object_id(case_name),
        primary_prediction_m=_immutable_copy(queried["primary_prediction_m"]),
        selected_raw_backbone_m=_immutable_copy(queried["selected_raw_backbone_m"]),
        queried_identity_ids=_immutable_copy(identities),
        target_identity_ids=_immutable_copy(target["identity_ids"]),
        official_frame_zero_m=_immutable_copy(positions),
        target_points_m=_immutable_copy(target["object_points"]),
        object_visibilities=_immutable_copy(target["object_visibilities"]),
        object_motions_valid=_immutable_copy(target["object_motions_valid"]),
        shared_support_mask=_immutable_copy(queried["shared_support_mask"]),
        center_exclusion_mask=_immutable_copy(queried["center_exclusion_mask"]),
        frame_indices=_immutable_copy(queried["frame_indices"]),
        source_node_count=source_node_count,
        permit_evidence=evidence,
    )
    # Cardinality of the source anchors is deliberately not part of this
    # identity/x0 check.  The pure scorer reports M<N, M=N, or M>N as metadata.
    return values


def load_direct_scoring_inputs_after_future_score_permit(
    *,
    case_name: str,
    queried_prediction_seal_path: str | Path,
    target_manifest_path: str | Path,
    lock_path: str | Path,
    future_score_permit: object,
    consume_future_score_permit: PermitConsumer,
    source_node_count: int | None = None,
) -> DirectScoringInputs:
    """Load exact scorer inputs only after consuming a future-score permit.

    The sealed queried prediction is safe to validate before that capability:
    its own validator proves it depends only on the official x0 artifact.  The
    target manifest path is not opened or hashed until after the separate
    future-score capability has been consumed.
    """

    _require(isinstance(case_name, str) and case_name, "case_name is missing")
    _require(
        callable(consume_future_score_permit),
        "future-score permit consumer is not callable",
    )
    queried_seal, queried = _load_validated_queried_prediction_arrays(
        queried_prediction_seal_path,
        lock_path=lock_path,
        expected_case_name=case_name,
    )
    permit_evidence = _json_mapping(
        consume_future_score_permit(
            future_score_permit,
            case_name=case_name,
            operation=FUTURE_SCORE_OPERATION,
        ),
        role="future-score permit consumer",
    )
    target_manifest, target = _load_validated_target_arrays(
        target_manifest_path,
        lock_path=lock_path,
        expected_case_name=case_name,
    )
    _require(
        target_manifest["official_query_manifest"]
        == queried_seal["official_query_manifest"],
        "target and queried prediction bind different x0 query manifests",
    )
    _require(
        target_manifest["official_query_manifest_artifact_sha256"]
        == queried_seal["official_query_manifest_artifact_sha256"],
        "target and queried prediction x0 query checksums differ",
    )
    return _assemble_direct_scoring_inputs(
        case_name=case_name,
        queried=queried,
        target=target,
        source_node_count=source_node_count,
        permit_evidence=permit_evidence,
    )


__all__ = [
    "CENTER_COUNT",
    "CanonicalOfficialTarget",
    "DirectScoringInputs",
    "FRAME_COUNT",
    "FUTURE_SCORE_OPERATION",
    "OFFICIAL_TARGET_KIND",
    "PROTOCOL_ID",
    "PermitConsumer",
    "TARGET_ARRAY_NAMES",
    "TARGET_RECONSTRUCTION_OPERATION",
    "canonicalize_official_reconstruction",
    "load_direct_scoring_inputs_after_future_score_permit",
    "validate_official_target_artifact",
    "write_official_target_and_frame_zero_query_artifacts",
]
