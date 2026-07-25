"""Fail-closed query artifacts for the Deform360 held v8 protocol.

The module separates three operations which must occur on opposite sides of
the target-information boundary:

* before outcomes, bind a sealed nodal prediction to one frozen displacement
  field;
* after the cohort barrier, extract and seal only official frame-zero query
  identities and coordinates;
* query both frozen prediction arms and seal the result before any future
  target coordinate, visibility, validity, colour, or score is opened.

There is intentionally no outcome/scorer import or API in this module.  The
only official data accepted by the query operation are the two arrays in the
frame-zero query artifact.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np

from . import deform360_frozen_query_field as frozen_query_field
from .deform360_frozen_query_field import (
    FrameZeroQuerySet,
    FrozenFieldConfig,
    build_radius_union_center_exclusion,
    build_frozen_nodal_field,
    query_frozen_nodal_field,
)


PROTOCOL_ID = "deform360-held-online-belief-v8.2"
SCHEMA_VERSION = 1
FRAME_COUNT = 76
CENTER_COUNT = 16

OFFICIAL_QUERY_KIND = "Deform360HeldV8OfficialFrameZeroQuery"
FROZEN_FIELD_KIND = "Deform360HeldV8PreOutcomeFrozenField"
QUERIED_PREDICTION_KIND = "Deform360HeldV8QueriedPredictionSeal"

OFFICIAL_QUERY_ARRAY_NAMES = frozenset({"identity_ids", "positions_m"})
FROZEN_SOURCE_ARRAY_NAMES = (
    "primary_prediction_m",
    "selected_raw_backbone_m",
    "frame_zero_points_m",
    "center_ids",
)
QUERIED_PREDICTION_ARRAY_NAMES = frozenset(
    {
        "primary_prediction_m",
        "selected_raw_backbone_m",
        "identity_ids",
        "positions_m",
        "frame_indices",
        "shared_support_mask",
        "exact_anchor_mask",
        "neighbor_anchor_ids",
        "neighbor_weights",
        "neighbor_distances_m",
        "nearest_anchor_distance_m",
        "kth_anchor_distance_m",
        "center_ids",
        "center_nearest_query_identity_ids",
        "center_nearest_query_indices",
        "center_nearest_query_distance_m",
        "center_within_radius_mask",
        "center_exclusion_mask",
    }
)

FIELD_OPERATOR_ID = "gaussian-knn-normalized-v1"
FIELD_SEMANTICS = "total-displacement-from-frame-zero-v1"
GAUSSIAN_NEIGHBOR_COUNT = 4
GAUSSIAN_LENGTH_SCALE_FRACTION = 0.05
SUPPORT_RADIUS_FRACTION = 0.50
ROBUST_SCALE_QUANTILES = (0.05, 0.95)
ROBUST_SCALE_QUANTILE_METHOD = "linear"
MINIMUM_METRIC_SCALE_M = 1e-12
CENTER_EXCLUSION_MAXIMUM_DISTANCE_M = 0.015
CENTER_EXCLUSION_CONTRACT = {
    "operator_id": frozen_query_field.RADIUS_UNION_CENTER_EXCLUSION_OPERATOR_ID,
    "coordinate_source": "frozen-assimilation-and-official-query-x0-only",
    "coordinate_input_dtype": "<f4",
    "distance_compute_dtype": "<f8",
    "distance_metric": "euclidean",
    "distance_computation": (
        "cast-float32-x0-coordinates-to-float64-before-euclidean-norm"
    ),
    "maximum_distance_m": CENTER_EXCLUSION_MAXIMUM_DISTANCE_M,
    "inclusion_predicate": "distance_m <= maximum_distance_m",
    "union_semantics": "set-union-over-all-assimilation-centers",
    "excluded_query_cardinality": "variable-zero-to-official-query-count",
    "unmatched_center_policy": "exclude-no-query",
    "per_center_nearest_query_is_audit_only": True,
    "per_center_nearest_query_tie_break": (
        frozen_query_field.RADIUS_UNION_NEAREST_QUERY_TIE_BREAK_RULE
    ),
    "query_batch_order_invariant": True,
    "future_coordinates_or_masks_used": False,
    "cohort_coverage_gate_imposed_here": False,
}
CENTER_EXCLUSION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        CENTER_EXCLUSION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
UNSUPPORTED_QUERY_POLICY = "emit-prediction-and-mask-v1"
EXACT_ANCHOR_RULE = "bit-exact-nodal-value"
TIE_BREAK_RULE = "distance-then-anchor-id"
COORDINATE_FRAME = "Deform360 world frame"
LENGTH_UNIT = "metre"
DEVELOPMENT_PROTOCOL_ID = "deform360-open27-frozen-query-field-v1-development"
DEVELOPMENT_CANDIDATE_ID = "gaussian-knn-normalized-v1-k04-length05pct"

_FILE_RECORD_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_ARRAY_RECORD_FIELDS = frozenset({"shape", "dtype", "sha256"})
_SHA256_LENGTH = 64
_SEALED_FILE_MODE = 0o400


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
    """Create missing parents only below a canonical, non-symlink directory."""

    existing = destination.parent
    while not os.path.lexists(existing):
        parent = existing.parent
        _require(parent != existing, "destination has no existing ancestor")
        existing = parent
    _require_canonical_directory(existing)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_canonical_directory(destination.parent)


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


def _require_unchanged_open_file(
    source: Path,
    descriptor: int,
    opened: os.stat_result,
) -> os.stat_result:
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
    return after


def _read_file_bytes(path: str | Path) -> tuple[Path, bytes, os.stat_result]:
    source, descriptor, opened = _open_regular_file_snapshot(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = _require_unchanged_open_file(source, descriptor, opened)
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


def _require_exact_file_mode(
    path: str | Path,
    *,
    role: str,
    expected_mode: int = _SEALED_FILE_MODE,
) -> None:
    source = _canonical_path(path)
    observed = os.lstat(source)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == expected_mode,
        f"{role} must be a regular non-symlink file with mode {expected_mode:04o}",
    )


def _validate_bound_file(
    record: object,
    *,
    role: str,
    required_mode: int | None = None,
) -> Path:
    _require(
        isinstance(record, Mapping) and set(record) == _FILE_RECORD_FIELDS,
        f"{role} file record fields changed",
    )
    path = record.get("path")
    _require(isinstance(path, str) and path, f"{role} path is missing")
    observed = _bound_file(path)
    _require(observed == dict(record), f"{role} file binding changed")
    if required_mode is not None:
        _require_exact_file_mode(
            observed["path"], role=role, expected_mode=required_mode
        )
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
    from io import BytesIO

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


def _validate_lock(
    path: str | Path, expected_sha256: str | None = None
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


def _validate_expected_source_binding(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_path: Path,
    role: str,
) -> dict[str, Any]:
    _require(_valid_sha256(expected_sha256), f"{role} SHA-256 is invalid")
    record = _bound_file(path)
    _require(
        Path(record["path"]) == expected_path,
        f"{role} path is not the loaded implementation source",
    )
    _require(record["sha256"] == expected_sha256, f"{role} SHA-256 changed")
    return record


def _validate_official_query_arrays(
    arrays: Mapping[str, np.ndarray],
) -> FrameZeroQuerySet:
    _require(
        set(arrays) == OFFICIAL_QUERY_ARRAY_NAMES,
        "official query archive must contain only identity_ids and positions_m",
    )
    identities = np.asarray(arrays["identity_ids"])
    positions = np.asarray(arrays["positions_m"])
    _require(
        identities.dtype == np.dtype(np.int64) and identities.ndim == 1,
        "official query identity_ids must have dtype int64 and shape (M,)",
    )
    _require(
        positions.dtype == np.dtype(np.float32)
        and positions.ndim == 2
        and positions.shape == (len(identities), 3),
        "official query positions_m must have dtype float32 and shape (M, 3)",
    )
    _require(
        len(identities) >= CENTER_COUNT,
        "official query set has fewer than 16 identities",
    )
    _require(
        np.all(np.diff(identities) > 0),
        "official query identity order must be strictly increasing",
    )
    return FrameZeroQuerySet(identity_ids=identities, positions_m=positions)


def write_official_query_artifact(
    archive_path: str | Path,
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    lock_sha256: str,
    case_name: str,
    query_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Exclusively write an official frame-zero-only query NPZ and manifest."""

    _require(isinstance(case_name, str) and case_name, "case_name is missing")
    queries = _validate_official_query_arrays(query_arrays)
    arrays = {
        "identity_ids": queries.identity_ids,
        "positions_m": queries.positions_m,
    }
    lock_record = _validate_lock(lock_path, lock_sha256)
    _write_new_npz(archive_path, arrays)
    archive_record = _bound_file(archive_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": OFFICIAL_QUERY_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "lock": lock_record,
        "archive": archive_record,
        "array_records": _array_records(arrays),
        "frame_indices_read": [0],
        "information_boundary": {
            "official_arrays_present": sorted(OFFICIAL_QUERY_ARRAY_NAMES),
            "frame_zero_coordinates_only": True,
            "future_coordinates_present_or_read": False,
            "visibility_or_validity_present_or_read": False,
            "mask_present_or_read": False,
            "colour_present_or_read": False,
            "future_source_container_bound_or_hashed": False,
            "score_present_or_read": False,
        },
    }
    manifest["artifact_sha256"] = _artifact_sha256(manifest)
    try:
        _write_new_json(manifest_path, manifest)
    except BaseException:
        os.chmod(_canonical_path(archive_path), 0o600, follow_symlinks=False)
        _canonical_path(archive_path).unlink(missing_ok=True)
        raise
    return validate_official_query_artifact(
        manifest_path,
        lock_path,
        expected_case_name=case_name,
    )


def validate_official_query_artifact(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str | None = None,
) -> dict[str, Any]:
    """Revalidate the exact two-array frame-zero query artifact."""

    _require_exact_file_mode(manifest_path, role="official query manifest")
    manifest = _load_json(manifest_path)
    _require(
        set(manifest)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "case_name",
            "lock",
            "archive",
            "array_records",
            "frame_indices_read",
            "information_boundary",
            "artifact_sha256",
        },
        "official query manifest fields changed",
    )
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_kind") == OFFICIAL_QUERY_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID,
        "official query schema, kind, or protocol changed",
    )
    case_name = manifest.get("case_name")
    _require(isinstance(case_name, str) and case_name, "official query case missing")
    if expected_case_name is not None:
        _require(case_name == expected_case_name, "official query binds another case")
    lock_record = _validate_lock(lock_path)
    _require(manifest.get("lock") == lock_record, "official query binds another lock")
    archive = _validate_bound_file(
        manifest.get("archive"),
        role="official query",
        required_mode=_SEALED_FILE_MODE,
    )
    _, arrays = _load_npz_arrays(archive)
    _validate_official_query_arrays(arrays)
    _validate_array_records(
        manifest.get("array_records"), arrays, role="official query"
    )
    _require(
        manifest.get("frame_indices_read") == [0],
        "official query reads a nonzero frame",
    )
    boundary = manifest.get("information_boundary")
    _require(
        boundary
        == {
            "official_arrays_present": sorted(OFFICIAL_QUERY_ARRAY_NAMES),
            "frame_zero_coordinates_only": True,
            "future_coordinates_present_or_read": False,
            "visibility_or_validity_present_or_read": False,
            "mask_present_or_read": False,
            "colour_present_or_read": False,
            "future_source_container_bound_or_hashed": False,
            "score_present_or_read": False,
        },
        "official query information boundary changed",
    )
    _require(
        manifest.get("artifact_sha256") == _artifact_sha256(manifest),
        "official query content checksum changed",
    )
    return manifest


def _validate_development_decision(path: str | Path) -> dict[str, Any]:
    decision = _load_json(path)
    selection = decision.get("selection")
    _require(
        decision.get("protocol_id") == DEVELOPMENT_PROTOCOL_ID
        and isinstance(selection, Mapping),
        "open27 development decision protocol or selection changed",
    )
    expected_config = {
        "candidate_id": DEVELOPMENT_CANDIDATE_ID,
        "operator_id": FIELD_OPERATOR_ID,
        "neighbor_count": GAUSSIAN_NEIGHBOR_COUNT,
        "length_scale_fraction": GAUSSIAN_LENGTH_SCALE_FRACTION,
        "support_radius_fraction": SUPPORT_RADIUS_FRACTION,
    }
    _require(
        selection.get("status")
        == "locked using only non-held open-development evidence"
        and selection.get("selected_candidate_id") == DEVELOPMENT_CANDIDATE_ID
        and selection.get("selected_config") == expected_config
        and selection.get("future_target_scores_used_for_selection") is False
        and selection.get("future_target_masks_used_for_selection") is False,
        "open27 development decision does not select the frozen v8 field",
    )
    return decision


def _validate_online_seal_binding(
    seal_path: str | Path,
    archive_path: str | Path,
    lock_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_exact_file_mode(seal_path, role="online prediction seal")
    _require_exact_file_mode(archive_path, role="online prediction archive")
    seal_record = _bound_file(seal_path)
    seal = _load_json(seal_path)
    _require(seal.get("protocol_id") == PROTOCOL_ID, "online seal protocol changed")
    _require(
        seal.get("lock") == dict(lock_record),
        "online prediction seal binds another lock",
    )
    online = seal.get("online_artifacts")
    _require(
        isinstance(online, Mapping) and "online_prediction_archive" in online,
        "online prediction seal lacks its prediction archive",
    )
    archive_record = _bound_file(archive_path)
    _require(
        online.get("online_prediction_archive") == archive_record,
        "online prediction archive differs from its seal",
    )
    _validate_bound_file(
        online.get("online_prediction_archive"),
        role="sealed online prediction",
        required_mode=_SEALED_FILE_MODE,
    )
    _require(
        seal.get("artifact_sha256") == _artifact_sha256(seal),
        "online prediction seal content checksum changed",
    )
    boundary = seal.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("outcome_created") is False
        and boundary.get("outcome_read") is False
        and boundary.get("all_frozen_predictions_hashed_before_outcome") is True,
        "online prediction was not sealed before outcome access",
    )
    return seal_record, seal


def _validate_frozen_source_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], float, float, float]:
    _require(
        all(name in arrays for name in FROZEN_SOURCE_ARRAY_NAMES),
        "online prediction lacks a frozen field source array",
    )
    selected = {name: np.asarray(arrays[name]) for name in FROZEN_SOURCE_ARRAY_NAMES}
    primary = selected["primary_prediction_m"]
    comparator = selected["selected_raw_backbone_m"]
    frame_zero = selected["frame_zero_points_m"]
    centers = selected["center_ids"]
    _require(
        frame_zero.dtype == np.dtype(np.float32)
        and frame_zero.ndim == 2
        and frame_zero.shape[1] == 3
        and len(frame_zero) >= GAUSSIAN_NEIGHBOR_COUNT
        and np.all(np.isfinite(frame_zero)),
        "frame_zero_points_m must have finite float32 shape (N, 3)",
    )
    expected = (FRAME_COUNT, len(frame_zero), 3)
    _require(
        primary.dtype == comparator.dtype == np.dtype(np.float32)
        and primary.shape == comparator.shape == expected
        and np.all(np.isfinite(primary))
        and np.all(np.isfinite(comparator)),
        "both frozen source trajectories must have finite float32 shape (76, N, 3)",
    )
    _require(
        np.array_equal(primary[0], frame_zero)
        and np.array_equal(comparator[0], frame_zero),
        "both frozen source trajectories must equal frame_zero_points_m at frame 0",
    )
    _require(
        centers.dtype == np.dtype(np.int64)
        and centers.shape == (CENTER_COUNT,)
        and len(np.unique(centers)) == CENTER_COUNT
        and np.all((0 <= centers) & (centers < len(frame_zero))),
        "center_ids must contain exactly 16 unique valid int64 anchor indices",
    )
    bounds = np.quantile(
        frame_zero.astype(np.float64),
        np.asarray(ROBUST_SCALE_QUANTILES, dtype=np.float64),
        axis=0,
        method=ROBUST_SCALE_QUANTILE_METHOD,
    )
    raw_scale = float(np.linalg.norm(bounds[1] - bounds[0]))
    _require(np.isfinite(raw_scale), "robust source scale is not finite")
    effective_scale = max(raw_scale, MINIMUM_METRIC_SCALE_M)
    length_scale = max(
        GAUSSIAN_LENGTH_SCALE_FRACTION * effective_scale,
        MINIMUM_METRIC_SCALE_M,
    )
    support = max(
        SUPPORT_RADIUS_FRACTION * effective_scale,
        MINIMUM_METRIC_SCALE_M,
    )
    return selected, effective_scale, length_scale, support


def _field_contract(
    object_scale_m: float,
    length_scale_m: float,
    support_radius_m: float,
) -> dict[str, Any]:
    return {
        "field_semantics": FIELD_SEMANTICS,
        "operator_id": FIELD_OPERATOR_ID,
        "gaussian_neighbor_count": GAUSSIAN_NEIGHBOR_COUNT,
        "gaussian_length_scale_fraction": GAUSSIAN_LENGTH_SCALE_FRACTION,
        "gaussian_length_scale_m": length_scale_m,
        "support_radius_fraction": SUPPORT_RADIUS_FRACTION,
        "maximum_support_distance_m": support_radius_m,
        "minimum_metric_scale_m": MINIMUM_METRIC_SCALE_M,
        "robust_object_scale": {
            "rule": "Euclidean diagonal of coordinate-wise frame-zero quantile bbox",
            "lower_quantile": ROBUST_SCALE_QUANTILES[0],
            "upper_quantile": ROBUST_SCALE_QUANTILES[1],
            "quantile_method": ROBUST_SCALE_QUANTILE_METHOD,
            "geometry": "sealed source frame_zero_points_m only",
            "effective_scale_m": object_scale_m,
        },
        "unsupported_query_policy": UNSUPPORTED_QUERY_POLICY,
        "unsupported_output_semantics": (
            "emit both predictions but permanently set shared_support_mask false"
        ),
        "exact_anchor_rule": EXACT_ANCHOR_RULE,
        "tie_break_rule": TIE_BREAK_RULE,
        "query_batch_order_and_cardinality_invariant": True,
        "arm_specific_masks_permitted": False,
        "input_coordinate_dtype": "<f4",
        "output_coordinate_dtype": "<f4",
        "distance_and_weight_dtype": "<f8",
        "coordinate_frame": COORDINATE_FRAME,
        "length_unit": LENGTH_UNIT,
        "frame_indices_dtype": "<i8",
        "frame_indices": list(range(FRAME_COUNT)),
        "center_count": CENTER_COUNT,
        "center_exclusion": {
            **deepcopy(CENTER_EXCLUSION_CONTRACT),
            "contract_sha256": CENTER_EXCLUSION_CONTRACT_SHA256,
        },
    }


def write_preoutcome_frozen_field_manifest(
    manifest_path: str | Path,
    *,
    lock_path: str | Path,
    lock_sha256: str,
    online_prediction_archive_path: str | Path,
    online_prediction_seal_path: str | Path,
    field_source_path: str | Path,
    field_source_sha256: str,
    artifact_module_source_path: str | Path,
    artifact_module_source_sha256: str,
    development_decision_path: str | Path,
    development_decision_sha256: str,
    case_name: str,
) -> dict[str, Any]:
    """Bind a sealed online prediction to the fixed v8 field before outcomes."""

    _require(isinstance(case_name, str) and case_name, "case_name is missing")
    lock_record = _validate_lock(lock_path, lock_sha256)
    seal_record, seal = _validate_online_seal_binding(
        online_prediction_seal_path,
        online_prediction_archive_path,
        lock_record,
    )
    _require(seal.get("case_name") == case_name, "online seal binds another case")
    expected_field_source = Path(frozen_query_field.__file__).resolve()
    expected_artifact_source = Path(__file__).resolve()
    field_source = _validate_expected_source_binding(
        field_source_path,
        field_source_sha256,
        expected_path=expected_field_source,
        role="frozen field source",
    )
    artifact_source = _validate_expected_source_binding(
        artifact_module_source_path,
        artifact_module_source_sha256,
        expected_path=expected_artifact_source,
        role="v8 query artifact source",
    )
    _require(
        _valid_sha256(development_decision_sha256),
        "development decision SHA-256 is invalid",
    )
    decision_record = _bound_file(development_decision_path)
    _require(
        decision_record["sha256"] == development_decision_sha256,
        "development decision SHA-256 changed",
    )
    _validate_development_decision(development_decision_path)

    archive_record = _bound_file(online_prediction_archive_path)
    _, online_arrays = _load_npz_arrays(online_prediction_archive_path)
    arrays, scale, length_scale, support = _validate_frozen_source_arrays(online_arrays)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": FROZEN_FIELD_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "lock": lock_record,
        "online_prediction_seal": seal_record,
        "online_prediction_seal_artifact_sha256": seal["artifact_sha256"],
        "online_prediction_archive": archive_record,
        "source_array_records": _array_records(arrays),
        "implementation_bindings": {
            "frozen_field_source": field_source,
            "v8_query_artifact_source": artifact_source,
            "open27_development_decision": decision_record,
        },
        "development_decision_artifact": {
            "protocol_id": DEVELOPMENT_PROTOCOL_ID,
            "selected_candidate_id": DEVELOPMENT_CANDIDATE_ID,
        },
        "field_contract": _field_contract(scale, length_scale, support),
        "information_boundary": {
            "created_pre_outcome": True,
            "sealed_online_prediction_only": True,
            "official_query_coordinates_read": False,
            "future_target_coordinates_read": False,
            "future_visibility_or_validity_read": False,
            "future_mask_or_colour_read": False,
            "score_read": False,
        },
    }
    manifest["artifact_sha256"] = _artifact_sha256(manifest)
    _write_new_json(manifest_path, manifest)
    return validate_preoutcome_frozen_field_manifest(
        manifest_path,
        lock_path=lock_path,
        expected_case_name=case_name,
    )


def validate_preoutcome_frozen_field_manifest(
    manifest_path: str | Path,
    *,
    lock_path: str | Path,
    expected_case_name: str | None = None,
) -> dict[str, Any]:
    """Rehash the lock, code, decision, seal, archive, and selected arrays."""

    _require_exact_file_mode(manifest_path, role="pre-outcome frozen field manifest")
    manifest = _load_json(manifest_path)
    _require(
        set(manifest)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "case_name",
            "lock",
            "online_prediction_seal",
            "online_prediction_seal_artifact_sha256",
            "online_prediction_archive",
            "source_array_records",
            "implementation_bindings",
            "development_decision_artifact",
            "field_contract",
            "information_boundary",
            "artifact_sha256",
        },
        "frozen field manifest fields changed",
    )
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_kind") == FROZEN_FIELD_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID,
        "frozen field schema, kind, or protocol changed",
    )
    case_name = manifest.get("case_name")
    _require(isinstance(case_name, str) and case_name, "frozen field case missing")
    if expected_case_name is not None:
        _require(case_name == expected_case_name, "frozen field binds another case")
    lock_record = _validate_lock(lock_path)
    _require(manifest.get("lock") == lock_record, "frozen field binds another lock")

    archive = _validate_bound_file(
        manifest.get("online_prediction_archive"),
        role="online prediction",
        required_mode=_SEALED_FILE_MODE,
    )
    seal_path = _validate_bound_file(
        manifest.get("online_prediction_seal"),
        role="online prediction seal",
        required_mode=_SEALED_FILE_MODE,
    )
    _, seal = _validate_online_seal_binding(seal_path, archive, lock_record)
    _require(
        seal.get("case_name") == case_name
        and manifest.get("online_prediction_seal_artifact_sha256")
        == seal.get("artifact_sha256"),
        "frozen field online prediction seal binding changed",
    )

    bindings = manifest.get("implementation_bindings")
    _require(
        isinstance(bindings, Mapping)
        and set(bindings)
        == {
            "frozen_field_source",
            "v8_query_artifact_source",
            "open27_development_decision",
        },
        "frozen field implementation binding set changed",
    )
    field_source = _validate_bound_file(
        bindings["frozen_field_source"], role="frozen field source"
    )
    artifact_source = _validate_bound_file(
        bindings["v8_query_artifact_source"], role="v8 query artifact source"
    )
    decision_path = _validate_bound_file(
        bindings["open27_development_decision"], role="open27 development decision"
    )
    _require(
        field_source == Path(frozen_query_field.__file__).resolve()
        and artifact_source == Path(__file__).resolve(),
        "frozen field implementation source path changed",
    )
    _validate_development_decision(decision_path)
    _require(
        manifest.get("development_decision_artifact")
        == {
            "protocol_id": DEVELOPMENT_PROTOCOL_ID,
            "selected_candidate_id": DEVELOPMENT_CANDIDATE_ID,
        },
        "development decision mirror changed",
    )

    _, online_arrays = _load_npz_arrays(archive)
    arrays, scale, length_scale, support = _validate_frozen_source_arrays(online_arrays)
    _validate_array_records(
        manifest.get("source_array_records"), arrays, role="frozen source"
    )
    _require(
        manifest.get("field_contract") == _field_contract(scale, length_scale, support),
        "frozen field contract changed",
    )
    _require(
        manifest.get("information_boundary")
        == {
            "created_pre_outcome": True,
            "sealed_online_prediction_only": True,
            "official_query_coordinates_read": False,
            "future_target_coordinates_read": False,
            "future_visibility_or_validity_read": False,
            "future_mask_or_colour_read": False,
            "score_read": False,
        },
        "frozen field information boundary changed",
    )
    _require(
        manifest.get("artifact_sha256") == _artifact_sha256(manifest),
        "frozen field content checksum changed",
    )
    return manifest


def _load_validated_query_set(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str,
) -> FrameZeroQuerySet:
    manifest = validate_official_query_artifact(
        manifest_path,
        lock_path,
        expected_case_name=expected_case_name,
    )
    archive = _validate_bound_file(
        manifest["archive"],
        role="official query",
        required_mode=_SEALED_FILE_MODE,
    )
    _, arrays = _load_npz_arrays(archive)
    return _validate_official_query_arrays(arrays)


def _load_validated_frozen_field(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str | None = None,
):
    manifest = validate_preoutcome_frozen_field_manifest(
        manifest_path,
        lock_path=lock_path,
        expected_case_name=expected_case_name,
    )
    archive = _validate_bound_file(
        manifest["online_prediction_archive"],
        role="online prediction",
        required_mode=_SEALED_FILE_MODE,
    )
    _, online_arrays = _load_npz_arrays(archive)
    arrays, _, _, _ = _validate_frozen_source_arrays(online_arrays)
    contract = manifest["field_contract"]
    config = FrozenFieldConfig(
        operator_id=FIELD_OPERATOR_ID,
        maximum_support_distance_m=float(contract["maximum_support_distance_m"]),
        unsupported_query_policy=UNSUPPORTED_QUERY_POLICY,
        gaussian_neighbor_count=GAUSSIAN_NEIGHBOR_COUNT,
        gaussian_length_scale_m=float(contract["gaussian_length_scale_m"]),
        tie_break_rule=TIE_BREAK_RULE,
        exact_anchor_rule=EXACT_ANCHOR_RULE,
    )
    field = build_frozen_nodal_field(
        arrays["frame_zero_points_m"],
        arrays["primary_prediction_m"],
        arrays["selected_raw_backbone_m"],
        arrays["center_ids"],
        config=config,
    )
    return manifest, field


def _queried_arrays(field, queries: FrameZeroQuerySet) -> dict[str, np.ndarray]:
    result = query_frozen_nodal_field(field, queries)
    exclusion = build_radius_union_center_exclusion(
        field.geometry,
        queries,
        maximum_distance_m=CENTER_EXCLUSION_MAXIMUM_DISTANCE_M,
    )
    neighbor_positions = field.geometry.anchor_positions_m[
        result.neighbor_anchor_ids
    ].astype(np.float64)
    neighbor_distance = np.linalg.norm(
        neighbor_positions - queries.positions_m[:, None].astype(np.float64),
        axis=2,
    )
    arrays = {
        "primary_prediction_m": result.primary_prediction_m,
        "selected_raw_backbone_m": result.comparator_prediction_m,
        "identity_ids": queries.identity_ids,
        "positions_m": queries.positions_m,
        "frame_indices": np.arange(FRAME_COUNT, dtype=np.int64),
        "shared_support_mask": result.supported_identity_mask,
        "exact_anchor_mask": result.exact_anchor_mask,
        "neighbor_anchor_ids": result.neighbor_anchor_ids,
        "neighbor_weights": result.neighbor_weights,
        "neighbor_distances_m": neighbor_distance,
        "nearest_anchor_distance_m": result.nearest_anchor_distance_m,
        "kth_anchor_distance_m": result.kth_anchor_distance_m,
        "center_ids": exclusion.assimilation_anchor_ids,
        "center_nearest_query_identity_ids": exclusion.nearest_query_identity_ids,
        "center_nearest_query_indices": exclusion.nearest_query_indices,
        "center_nearest_query_distance_m": exclusion.nearest_query_distance_m,
        "center_within_radius_mask": exclusion.center_within_radius_mask,
        "center_exclusion_mask": exclusion.excluded_query_mask,
    }
    _validate_queried_arrays(arrays, field=field, queries=queries)
    return arrays


def _validate_queried_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    field,
    queries: FrameZeroQuerySet,
) -> None:
    _require(
        set(arrays) == QUERIED_PREDICTION_ARRAY_NAMES,
        "queried prediction archive fields changed or contain arm-specific masks",
    )
    count = len(queries.identity_ids)
    expected_trajectory_shape = (FRAME_COUNT, count, 3)
    _require(
        np.asarray(arrays["primary_prediction_m"]).dtype == np.dtype(np.float32)
        and np.asarray(arrays["selected_raw_backbone_m"]).dtype == np.dtype(np.float32)
        and np.asarray(arrays["primary_prediction_m"]).shape
        == np.asarray(arrays["selected_raw_backbone_m"]).shape
        == expected_trajectory_shape,
        "queried arms must share float32 shape (76, M, 3)",
    )
    _require(
        np.array_equal(arrays["identity_ids"], queries.identity_ids)
        and np.array_equal(arrays["positions_m"], queries.positions_m),
        "queried x0 identity order or positions changed",
    )
    _require(
        np.array_equal(arrays["frame_indices"], np.arange(FRAME_COUNT, dtype=np.int64)),
        "queried logical frame_indices changed",
    )
    _require(
        np.array_equal(arrays["primary_prediction_m"][0], queries.positions_m)
        and np.array_equal(arrays["selected_raw_backbone_m"][0], queries.positions_m),
        "queried predictions do not preserve x0 exactly",
    )
    support = np.asarray(arrays["shared_support_mask"])
    _require(
        support.dtype == np.dtype(bool) and support.shape == (count,),
        "shared_support_mask must have shape (M,) and bool dtype",
    )
    _require(
        np.asarray(arrays["exact_anchor_mask"]).dtype == np.dtype(bool)
        and np.asarray(arrays["exact_anchor_mask"]).shape == (count,),
        "exact_anchor_mask must have shape (M,) and bool dtype",
    )
    _require(
        np.asarray(arrays["neighbor_anchor_ids"]).dtype == np.dtype(np.int64)
        and np.asarray(arrays["neighbor_anchor_ids"]).shape
        == (count, GAUSSIAN_NEIGHBOR_COUNT)
        and np.asarray(arrays["neighbor_weights"]).dtype == np.dtype(np.float64)
        and np.asarray(arrays["neighbor_weights"]).shape
        == (count, GAUSSIAN_NEIGHBOR_COUNT)
        and np.asarray(arrays["neighbor_distances_m"]).dtype == np.dtype(np.float64)
        and np.asarray(arrays["neighbor_distances_m"]).shape
        == (count, GAUSSIAN_NEIGHBOR_COUNT),
        "queried neighbor diagnostics changed",
    )
    for name in ("nearest_anchor_distance_m", "kth_anchor_distance_m"):
        value = np.asarray(arrays[name])
        _require(
            value.dtype == np.dtype(np.float64) and value.shape == (count,),
            f"{name} must have float64 shape (M,)",
        )
    _require(
        np.asarray(arrays["center_ids"]).dtype == np.dtype(np.int64)
        and np.asarray(arrays["center_ids"]).shape == (CENTER_COUNT,)
        and np.asarray(arrays["center_nearest_query_identity_ids"]).dtype
        == np.dtype(np.int64)
        and np.asarray(arrays["center_nearest_query_identity_ids"]).shape
        == (CENTER_COUNT,)
        and np.asarray(arrays["center_nearest_query_indices"]).dtype
        == np.dtype(np.int64)
        and np.asarray(arrays["center_nearest_query_indices"]).shape == (CENTER_COUNT,)
        and np.asarray(arrays["center_nearest_query_distance_m"]).dtype
        == np.dtype(np.float64)
        and np.asarray(arrays["center_nearest_query_distance_m"]).shape
        == (CENTER_COUNT,)
        and np.asarray(arrays["center_within_radius_mask"]).dtype == np.dtype(bool)
        and np.asarray(arrays["center_within_radius_mask"]).shape == (CENTER_COUNT,)
        and np.asarray(arrays["center_exclusion_mask"]).dtype == np.dtype(bool)
        and np.asarray(arrays["center_exclusion_mask"]).shape == (count,),
        "center exclusion audit or mask shape changed",
    )
    exclusion = build_radius_union_center_exclusion(
        field.geometry,
        queries,
        maximum_distance_m=CENTER_EXCLUSION_MAXIMUM_DISTANCE_M,
    )
    _require(
        np.array_equal(
            arrays["center_nearest_query_identity_ids"],
            exclusion.nearest_query_identity_ids,
        )
        and np.array_equal(
            arrays["center_nearest_query_indices"], exclusion.nearest_query_indices
        )
        and np.array_equal(
            arrays["center_nearest_query_distance_m"],
            exclusion.nearest_query_distance_m,
        )
        and np.array_equal(
            arrays["center_within_radius_mask"],
            exclusion.center_within_radius_mask,
        )
        and np.array_equal(
            arrays["center_exclusion_mask"], exclusion.excluded_query_mask
        ),
        "center exclusion differs from the x0-only radius union",
    )
    _require(
        np.all(np.isfinite(arrays["primary_prediction_m"]))
        and np.all(np.isfinite(arrays["selected_raw_backbone_m"]))
        and np.all(np.isfinite(arrays["neighbor_weights"]))
        and np.all(np.isfinite(arrays["neighbor_distances_m"])),
        "queried predictions or diagnostics are not finite",
    )
    _require(
        np.array_equal(arrays["center_ids"], field.geometry.assimilation_anchor_ids),
        "queried center IDs differ from the frozen field",
    )


def write_queried_prediction_artifact(
    archive_path: str | Path,
    seal_path: str | Path,
    *,
    lock_path: str | Path,
    lock_sha256: str,
    frozen_field_manifest_path: str | Path,
    official_query_manifest_path: str | Path,
) -> dict[str, Any]:
    """Query both arms from x0 only and exclusively seal the prediction."""

    _validate_lock(lock_path, lock_sha256)
    field_manifest, field = _load_validated_frozen_field(
        frozen_field_manifest_path,
        lock_path,
    )
    case_name = field_manifest["case_name"]
    queries = _load_validated_query_set(
        official_query_manifest_path,
        lock_path,
        expected_case_name=case_name,
    )
    query_manifest = _load_json(official_query_manifest_path)
    arrays = _queried_arrays(field, queries)
    _write_new_npz(archive_path, arrays)
    seal: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": QUERIED_PREDICTION_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "lock": _bound_file(lock_path),
        "frozen_field_manifest": _bound_file(frozen_field_manifest_path),
        "frozen_field_manifest_artifact_sha256": field_manifest["artifact_sha256"],
        "official_query_manifest": _bound_file(official_query_manifest_path),
        "official_query_manifest_artifact_sha256": query_manifest["artifact_sha256"],
        "archive": _bound_file(archive_path),
        "array_records": _array_records(arrays),
        "field_contract": deepcopy(field_manifest["field_contract"]),
        "mask_contract": {
            "single_shared_support_mask_for_both_arms": True,
            "arm_specific_masks_permitted": False,
            "unsupported_predictions_emitted": True,
            "unsupported_queries_permanently_masked_false": True,
            "center_exclusion_mask_geometry_only": True,
            "center_exclusion_rule": "exclude-all-x0-queries-within-radius-v2",
            "center_exclusion_contract_sha256": (CENTER_EXCLUSION_CONTRACT_SHA256),
            "unmatched_assimilation_centers_allowed": True,
            "cohort_coverage_gate_imposed_here": False,
        },
        "information_boundary": {
            "official_query_arrays_read": sorted(OFFICIAL_QUERY_ARRAY_NAMES),
            "official_query_frame_zero_only": True,
            "future_target_coordinates_read": False,
            "future_visibility_or_validity_read": False,
            "future_mask_or_colour_read": False,
            "target_path_accepted": False,
            "scorer_or_outcome_module_imported": False,
            "queried_prediction_written_before_future_target_access": True,
            "score_read_or_written": False,
        },
    }
    seal["artifact_sha256"] = _artifact_sha256(seal)
    try:
        _write_new_json(seal_path, seal)
    except BaseException:
        os.chmod(_canonical_path(archive_path), 0o600, follow_symlinks=False)
        _canonical_path(archive_path).unlink(missing_ok=True)
        raise
    return validate_queried_prediction_artifact(
        seal_path,
        lock_path=lock_path,
        expected_case_name=case_name,
    )


def validate_queried_prediction_artifact(
    seal_path: str | Path,
    *,
    lock_path: str | Path,
    expected_case_name: str | None = None,
) -> dict[str, Any]:
    """Recompute the x0-only query and require bit-exact sealed arrays."""

    _require_exact_file_mode(seal_path, role="queried prediction seal")
    seal = _load_json(seal_path)
    _require(
        set(seal)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "case_name",
            "lock",
            "frozen_field_manifest",
            "frozen_field_manifest_artifact_sha256",
            "official_query_manifest",
            "official_query_manifest_artifact_sha256",
            "archive",
            "array_records",
            "field_contract",
            "mask_contract",
            "information_boundary",
            "artifact_sha256",
        },
        "queried prediction seal fields changed",
    )
    _require(
        seal.get("schema_version") == SCHEMA_VERSION
        and seal.get("artifact_kind") == QUERIED_PREDICTION_KIND
        and seal.get("protocol_id") == PROTOCOL_ID,
        "queried prediction schema, kind, or protocol changed",
    )
    case_name = seal.get("case_name")
    _require(isinstance(case_name, str) and case_name, "queried case missing")
    if expected_case_name is not None:
        _require(case_name == expected_case_name, "queried seal binds another case")
    lock_record = _validate_lock(lock_path)
    _require(seal.get("lock") == lock_record, "queried seal binds another lock")

    field_manifest_path = _validate_bound_file(
        seal.get("frozen_field_manifest"),
        role="frozen field manifest",
        required_mode=_SEALED_FILE_MODE,
    )
    field_manifest, field = _load_validated_frozen_field(
        field_manifest_path,
        lock_path,
        expected_case_name=case_name,
    )
    _require(
        seal.get("frozen_field_manifest_artifact_sha256")
        == field_manifest["artifact_sha256"],
        "queried seal frozen field binding changed",
    )
    query_manifest_path = _validate_bound_file(
        seal.get("official_query_manifest"),
        role="official query manifest",
        required_mode=_SEALED_FILE_MODE,
    )
    query_manifest = validate_official_query_artifact(
        query_manifest_path,
        lock_path,
        expected_case_name=case_name,
    )
    _require(
        seal.get("official_query_manifest_artifact_sha256")
        == query_manifest["artifact_sha256"],
        "queried seal official query binding changed",
    )
    queries = _load_validated_query_set(
        query_manifest_path,
        lock_path,
        expected_case_name=case_name,
    )
    expected_arrays = _queried_arrays(field, queries)
    archive = _validate_bound_file(
        seal.get("archive"),
        role="queried prediction",
        required_mode=_SEALED_FILE_MODE,
    )
    _, observed_arrays = _load_npz_arrays(archive)
    _validate_queried_arrays(observed_arrays, field=field, queries=queries)
    _validate_array_records(
        seal.get("array_records"), observed_arrays, role="queried prediction"
    )
    for name in sorted(QUERIED_PREDICTION_ARRAY_NAMES):
        _require(
            np.array_equal(observed_arrays[name], expected_arrays[name]),
            f"queried prediction {name} differs from the frozen x0-only query",
        )
    _require(
        seal.get("field_contract") == field_manifest["field_contract"],
        "queried field contract changed",
    )
    _require(
        seal.get("mask_contract")
        == {
            "single_shared_support_mask_for_both_arms": True,
            "arm_specific_masks_permitted": False,
            "unsupported_predictions_emitted": True,
            "unsupported_queries_permanently_masked_false": True,
            "center_exclusion_mask_geometry_only": True,
            "center_exclusion_rule": "exclude-all-x0-queries-within-radius-v2",
            "center_exclusion_contract_sha256": (CENTER_EXCLUSION_CONTRACT_SHA256),
            "unmatched_assimilation_centers_allowed": True,
            "cohort_coverage_gate_imposed_here": False,
        },
        "queried mask contract changed",
    )
    _require(
        seal.get("information_boundary")
        == {
            "official_query_arrays_read": sorted(OFFICIAL_QUERY_ARRAY_NAMES),
            "official_query_frame_zero_only": True,
            "future_target_coordinates_read": False,
            "future_visibility_or_validity_read": False,
            "future_mask_or_colour_read": False,
            "target_path_accepted": False,
            "scorer_or_outcome_module_imported": False,
            "queried_prediction_written_before_future_target_access": True,
            "score_read_or_written": False,
        },
        "queried prediction information boundary changed",
    )
    _require(
        seal.get("artifact_sha256") == _artifact_sha256(seal),
        "queried prediction content checksum changed",
    )
    return seal


# Explicit aliases make the x0-only nature visible to protocol operators.
write_official_frame_zero_query_artifact = write_official_query_artifact
validate_official_frame_zero_query_artifact = validate_official_query_artifact


__all__ = [
    "CENTER_COUNT",
    "CENTER_EXCLUSION_CONTRACT",
    "CENTER_EXCLUSION_CONTRACT_SHA256",
    "CENTER_EXCLUSION_MAXIMUM_DISTANCE_M",
    "FIELD_OPERATOR_ID",
    "FRAME_COUNT",
    "FROZEN_FIELD_KIND",
    "GAUSSIAN_LENGTH_SCALE_FRACTION",
    "GAUSSIAN_NEIGHBOR_COUNT",
    "OFFICIAL_QUERY_ARRAY_NAMES",
    "OFFICIAL_QUERY_KIND",
    "PROTOCOL_ID",
    "QUERIED_PREDICTION_ARRAY_NAMES",
    "QUERIED_PREDICTION_KIND",
    "SUPPORT_RADIUS_FRACTION",
    "UNSUPPORTED_QUERY_POLICY",
    "validate_official_frame_zero_query_artifact",
    "validate_official_query_artifact",
    "validate_preoutcome_frozen_field_manifest",
    "validate_queried_prediction_artifact",
    "write_official_frame_zero_query_artifact",
    "write_official_query_artifact",
    "write_preoutcome_frozen_field_manifest",
    "write_queried_prediction_artifact",
]
