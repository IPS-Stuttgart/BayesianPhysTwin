"""Target-blind public Deform360 tactile and robot prefix materialization.

This module reduces released Deform360 tactile grids and robot poses to the
metric contact rows consumed by the calibration factor materializer. It does
not read a physical prediction, a state innovation, or any confirmation
payload. Consequently, the emitted prior reliability cannot depend on the
PhysTwin residual.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import canonical_relative_posix_path, plain_json
from ._portable_contracts import (
    content_id,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .deform360_bias_aware_prospective_physical import _gripper_taxel_points
from .deform360_calibration_visual_execution_admission import (
    _load_stable_json_object,
    validate_deform360_prepared_source_inventory,
)

DEFORM360_TACTILE_AXIS_MAP_SCHEMA: Final = (
    "bayesian-phystwin.deform360-tactile-axis-map"
)
DEFORM360_TACTILE_AXIS_MAP_VERSION: Final = 1
DEFORM360_TACTILE_AXIS_MAP_SEMANTICS: Final = (
    "source-only-tactile-group-to-released-robot-axis-v1"
)
DEFORM360_PUBLIC_CONTACT_PREFIX_SCHEMA: Final = (
    "bayesian-phystwin.deform360-public-contact-prefix"
)
DEFORM360_PUBLIC_CONTACT_PREFIX_VERSION: Final = 1
DEFORM360_PUBLIC_CONTACT_PREFIX_SEMANTICS: Final = (
    "released-prefix-tactile-weighted-taxel-geometry-v1"
)
DEFORM360_PUBLIC_CONTACT_PREFIX_CLAIM_BOUNDARY: Final = (
    "Calibration-only reduction of released Deform360 tactile and robot-prefix "
    "measurements. The artifact does not establish contact accuracy, physical "
    "state accuracy, visual-provider competence, calibration, confirmation "
    "benefit, deployment safety, or state of the art."
)
DEFORM360_TACTILE_AXIS_MAP_CLAIM_BOUNDARY: Final = (
    "Source-only identity mapping between released tactile stream groups and "
    "released robot axes. The artifact is not a contact estimate, outcome, "
    "human approval, or confirmation authorization."
)

TACTILE_ROWS_USED: Final = 12
TACTILE_COLUMNS: Final = 32
TAXELS_PER_GRIPPER: Final = 2 * TACTILE_ROWS_USED * TACTILE_COLUMNS
DEFORM360_CONTACT_PATIENCE_FRAMES: Final = 5
CONTACT_THRESHOLD: Final = 0.0

_AXIS_MAP_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "object_id",
        "episode_id",
        "prepared_source_inventory_id",
        "group_to_robot_axis",
        "selection_evidence_id",
        "selection_semantics",
        "information_boundary",
        "claim_boundary",
        "artifact_id",
    }
)
_PREFIX_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "status",
        "support_negative_reason",
        "object_id",
        "episode_id",
        "stratum",
        "prepared_source_inventory_id",
        "prepared_source_inventory_file_sha256",
        "tactile_axis_map_id",
        "tactile_axis_map_file_sha256",
        "processing_revision",
        "prefix_raw_frame_range_half_open",
        "causal_frame_stop",
        "bimanual",
        "robot_axis_count",
        "mapped_groups",
        "supported_robot_axes",
        "missing_contact_robot_axes",
        "row_count",
        "contact_episode_count",
        "contact_detection",
        "reliability_policy",
        "files",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
        "materialization_id",
    }
)
_AXIS_MAP_BOUNDARY = {
    "calibration_source_mapping_metadata_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "human_approval_required": False,
}
_PREFIX_BOUNDARY = {
    "calibration_tactile_prefix_opened": True,
    "calibration_robot_prefix_opened": True,
    "physical_prediction_opened": False,
    "state_innovation_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
}
_DATA_FILE_NAMES = (
    "contact-episode-ids.json",
    "frame-ids.npy",
    "sensor-names.json",
    "source-artifacts.json",
    "source-reliability.npy",
    "tactile-response.npy",
    "taxel-world-positions-m.npy",
)
_MAX_SOURCE_ARRAY_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _literal_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symlinks: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an ordinary directory: {path}")
    return resolved


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symlinks: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be an ordinary file: {path}")
    return resolved


def _verified_inventory_file(
    root: Path,
    record: object,
    *,
    name: str,
) -> tuple[bytes, str, str]:
    value = _mapping(record, name=f"{name} inventory record")
    relative = canonical_relative_posix_path(value.get("path"), name=f"{name} path")
    expected_digest = sha256_digest(value.get("sha256"), name=f"{name} SHA-256")
    expected_bytes = _literal_integer(
        value.get("byte_count"),
        name=f"{name} byte_count",
        minimum=1,
    )
    _require(
        expected_bytes <= _MAX_SOURCE_ARRAY_BYTES,
        f"{name} exceeds {_MAX_SOURCE_ARRAY_BYTES} bytes",
    )
    path = _ordinary_file(root / relative, name=name)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the processed root") from error
    flags = (
        os.O_RDONLY
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open {name}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{name} is not a regular file")
        blocks: list[bytes] = []
        digest_state = hashlib.sha256()
        byte_count = 0
        while True:
            block = os.read(descriptor, _READ_CHUNK_BYTES)
            if not block:
                break
            blocks.append(block)
            digest_state.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"cannot read {name}") from error
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or byte_count != after.st_size
    ):
        raise ValueError(f"{name} changed while it was read")
    digest = digest_state.hexdigest()
    _require(byte_count == expected_bytes, f"{name} byte count changed")
    _require(digest == expected_digest, f"{name} SHA-256 changed")
    return b"".join(blocks), relative, digest


def _axis_map_identity(
    *,
    object_id: str,
    episode_id: int,
    prepared_source_inventory_id: str,
    group_to_robot_axis: Mapping[str, int],
    selection_evidence_id: str,
    selection_semantics: str,
) -> dict[str, Any]:
    normalized: dict[str, int] = {}
    for group, axis in group_to_robot_axis.items():
        name = _literal_string(group, name="tactile group")
        if name.endswith("_left") or name.endswith("_right"):
            raise ValueError("axis-map keys must be gripper groups, not sensor sides")
        normalized[name] = _literal_integer(axis, name=f"axis for {name}")
    _require(bool(normalized), "group_to_robot_axis must not be empty")
    _require(
        list(normalized) == sorted(normalized),
        "group_to_robot_axis keys must be sorted",
    )
    _require(
        len(set(normalized.values())) == len(normalized),
        "group_to_robot_axis must be one-to-one",
    )
    return {
        "schema": DEFORM360_TACTILE_AXIS_MAP_SCHEMA,
        "schema_version": DEFORM360_TACTILE_AXIS_MAP_VERSION,
        "semantics": DEFORM360_TACTILE_AXIS_MAP_SEMANTICS,
        "object_id": _literal_string(object_id, name="object_id"),
        "episode_id": _literal_integer(episode_id, name="episode_id"),
        "prepared_source_inventory_id": sha256_digest(
            prepared_source_inventory_id,
            name="prepared_source_inventory_id",
        ),
        "group_to_robot_axis": normalized,
        "selection_evidence_id": sha256_digest(
            selection_evidence_id,
            name="selection_evidence_id",
        ),
        "selection_semantics": _literal_string(
            selection_semantics,
            name="selection_semantics",
        ),
        "information_boundary": dict(_AXIS_MAP_BOUNDARY),
        "claim_boundary": DEFORM360_TACTILE_AXIS_MAP_CLAIM_BOUNDARY,
    }


def build_deform360_tactile_axis_map(
    *,
    object_id: str,
    episode_id: int,
    prepared_source_inventory_id: str,
    group_to_robot_axis: Mapping[str, int],
    selection_evidence_id: str,
    selection_semantics: str = "locked-source-only-calibration-v1",
) -> dict[str, Any]:
    """Build one content-addressed source-only tactile-axis map."""

    identity = _axis_map_identity(
        object_id=object_id,
        episode_id=episode_id,
        prepared_source_inventory_id=prepared_source_inventory_id,
        group_to_robot_axis=group_to_robot_axis,
        selection_evidence_id=selection_evidence_id,
        selection_semantics=selection_semantics,
    )
    return {**identity, "artifact_id": content_id(identity)}


def validate_deform360_tactile_axis_map(value: object) -> dict[str, Any]:
    """Validate and normalize one tactile-axis map."""

    mapping = dict(_mapping(value, name="tactile-axis map"))
    require_exact_fields(mapping, expected=_AXIS_MAP_FIELDS, name="tactile-axis map")
    identity = _axis_map_identity(
        object_id=mapping["object_id"],
        episode_id=mapping["episode_id"],
        prepared_source_inventory_id=mapping["prepared_source_inventory_id"],
        group_to_robot_axis=_mapping(
            mapping["group_to_robot_axis"],
            name="group_to_robot_axis",
        ),
        selection_evidence_id=mapping["selection_evidence_id"],
        selection_semantics=mapping["selection_semantics"],
    )
    if mapping["schema"] != identity["schema"]:
        raise ValueError("tactile-axis map schema changed")
    if mapping["schema_version"] != identity["schema_version"]:
        raise ValueError("tactile-axis map version changed")
    if mapping["semantics"] != identity["semantics"]:
        raise ValueError("tactile-axis map semantics changed")
    if mapping["information_boundary"] != identity["information_boundary"]:
        raise ValueError("tactile-axis map information boundary changed")
    if mapping["claim_boundary"] != identity["claim_boundary"]:
        raise ValueError("tactile-axis map claim boundary changed")
    declared = sha256_digest(mapping["artifact_id"], name="axis-map artifact_id")
    if declared != content_id(identity):
        raise ValueError("tactile-axis map artifact_id does not match content")
    return {**identity, "artifact_id": declared}


def save_deform360_tactile_axis_map(
    value: object,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Validate and atomically publish a tactile-axis map."""

    write_atomic_json(
        validate_deform360_tactile_axis_map(value),
        path,
        overwrite=overwrite,
    )


def load_deform360_tactile_axis_map(path: str | Path) -> dict[str, Any]:
    """Load one strict tactile-axis map."""

    source = _ordinary_file(path, name="tactile-axis map")
    value, _digest, _byte_count = _load_stable_json_object(
        source,
        label="tactile-axis map",
    )
    return validate_deform360_tactile_axis_map(value)


def _gripper_group(sensor_name: str) -> tuple[str, str]:
    for side in ("left", "right"):
        suffix = f"_{side}"
        if sensor_name.endswith(suffix):
            return sensor_name[: -len(suffix)], side
    raise ValueError(f"tactile sensor lacks a _left/_right side: {sensor_name}")


def _official_contact_window(active: np.ndarray) -> np.ndarray:
    signal = np.asarray(active, dtype=bool)
    _require(signal.ndim == 1, "contact signal must be one-dimensional")
    output = np.zeros_like(signal)
    start: int | None = None
    end: int | None = None
    missing = 0
    for frame, is_active in enumerate(signal):
        if start is None:
            if bool(is_active):
                start = frame
            continue
        if bool(is_active):
            missing = 0
        else:
            missing += 1
            if missing > DEFORM360_CONTACT_PATIENCE_FRAMES:
                end = frame - missing
                break
    if start is None:
        return output
    if end is None:
        end = len(signal) - 1
    output[start : end + 1] = True
    return output


def _load_robot_prefix(
    payload: bytes,
    *,
    prefix_start: int,
    prefix_stop: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
            required = {"actions", "T_worlds", "openings", "bimanual"}
            _require(required <= set(stored.files), "robot archive is incomplete")
            actions = np.asarray(stored["actions"], dtype=np.float64)
            poses = np.asarray(stored["T_worlds"], dtype=np.float64)
            openings = np.asarray(stored["openings"], dtype=np.float64)
            bimanual_value = np.asarray(stored["bimanual"])
    except (OSError, ValueError) as error:
        raise ValueError("cannot load robot archive snapshot") from error
    _require(
        bimanual_value.shape == () and bimanual_value.dtype == np.dtype(np.bool_),
        "robot bimanual flag must be a Boolean scalar",
    )
    bimanual = bool(bimanual_value.item())
    frame_count = len(actions)
    expected_action_tail = (2, 5, 3) if bimanual else (5, 3)
    expected_pose_tail = (2, 4, 4) if bimanual else (4, 4)
    expected_opening_tail = (2,) if bimanual else ()
    _require(
        actions.shape == (frame_count, *expected_action_tail),
        "robot action shape changed",
    )
    _require(
        poses.shape == (frame_count, *expected_pose_tail),
        "robot pose shape changed",
    )
    _require(
        openings.shape == (frame_count, *expected_opening_tail),
        "robot opening shape changed",
    )
    _require(
        0 <= prefix_start < prefix_stop <= frame_count,
        "prefix exceeds robot trajectory",
    )
    selected_poses = poses[prefix_start:prefix_stop]
    selected_openings = openings[prefix_start:prefix_stop]
    _require(
        bool(np.all(np.isfinite(selected_poses)))
        and bool(np.all(np.isfinite(selected_openings))),
        "robot prefix contains non-finite values",
    )
    _require(
        bool(np.all(selected_openings >= 0.0)),
        "robot openings must be nonnegative",
    )
    flat_poses = selected_poses.reshape(-1, 4, 4)
    _require(
        np.allclose(
            flat_poses[:, 3, :],
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=1e-8,
        ),
        "robot poses are not homogeneous transforms",
    )
    return selected_poses, selected_openings, bimanual


def _load_tactile_prefix(
    payload: bytes,
    *,
    prefix_start: int,
    prefix_stop: int,
) -> np.ndarray:
    try:
        stored = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("cannot load tactile stream snapshot") from error
    _require(isinstance(stored, np.ndarray), "tactile stream must be one array")
    _require(
        stored.ndim == 3
        and stored.shape[1:] == (16, TACTILE_COLUMNS)
        and prefix_stop <= len(stored),
        "tactile stream shape changed",
    )
    selected = np.asarray(stored[prefix_start:prefix_stop], dtype=np.float64)
    _require(
        bool(np.all(np.isfinite(selected))),
        "tactile prefix contains non-finite values",
    )
    return selected[:, :TACTILE_ROWS_USED, :]


def _interleave_tactile_sides(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    _require(left.shape == right.shape, "left/right tactile shapes differ")
    count = len(left)
    result: np.ndarray = np.empty(
        (count, TAXELS_PER_GRIPPER),
        dtype=np.float64,
    )
    result[:, 0::2] = left.reshape(count, -1)
    result[:, 1::2] = right.reshape(count, -1)
    return result


def _inventory_object(
    inventory: Mapping[str, Any],
    *,
    object_id: str,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in _sequence(inventory["objects"], name="inventory objects")
        if isinstance(item, Mapping) and item.get("object_id") == object_id
    ]
    _require(len(matches) == 1, f"inventory does not contain object {object_id!r}")
    return cast(Mapping[str, Any], matches[0])


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(plain_json(value), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _materialization_identity(
    *,
    status: str,
    support_negative_reason: str | None,
    object_row: Mapping[str, Any],
    inventory: Mapping[str, Any],
    inventory_file_sha256: str,
    axis_map: Mapping[str, Any],
    axis_map_file_sha256: str,
    bimanual: bool,
    prefix_range: list[int],
    mapped_groups: list[dict[str, Any]],
    supported_axes: list[int],
    row_count: int,
    contact_episode_count: int,
    files: Mapping[str, str],
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    axis_count = 2 if bimanual else 1
    return {
        "schema": DEFORM360_PUBLIC_CONTACT_PREFIX_SCHEMA,
        "schema_version": DEFORM360_PUBLIC_CONTACT_PREFIX_VERSION,
        "semantics": DEFORM360_PUBLIC_CONTACT_PREFIX_SEMANTICS,
        "status": status,
        "support_negative_reason": support_negative_reason,
        "object_id": object_row["object_id"],
        "episode_id": object_row["episode_id"],
        "stratum": object_row["stratum"],
        "prepared_source_inventory_id": inventory["inventory_id"],
        "prepared_source_inventory_file_sha256": inventory_file_sha256,
        "tactile_axis_map_id": axis_map["artifact_id"],
        "tactile_axis_map_file_sha256": axis_map_file_sha256,
        "processing_revision": inventory["processing_revision"],
        "prefix_raw_frame_range_half_open": prefix_range,
        "causal_frame_stop": prefix_range[1],
        "bimanual": bimanual,
        "robot_axis_count": axis_count,
        "mapped_groups": mapped_groups,
        "supported_robot_axes": supported_axes,
        "missing_contact_robot_axes": sorted(
            set(range(axis_count)) - set(supported_axes)
        ),
        "row_count": row_count,
        "contact_episode_count": contact_episode_count,
        "contact_detection": {
            "active_taxel_threshold": CONTACT_THRESHOLD,
            "minimum_active_taxels": 2,
            "patience_frames": DEFORM360_CONTACT_PATIENCE_FRAMES,
            "first_event_only_per_group": True,
            "inactive_patience_rows_emitted": False,
            "tactile_rows_used": TACTILE_ROWS_USED,
            "tactile_columns": TACTILE_COLUMNS,
        },
        "reliability_policy": {
            "source_reliability_value": 1.0,
            "depends_on_tactile_magnitude": False,
            "depends_on_physical_prediction": False,
            "depends_on_state_innovation": False,
            "innovation_likelihood_application_count": 0,
        },
        "files": dict(files),
        "source_artifacts": dict(source_artifacts),
        "information_boundary": dict(_PREFIX_BOUNDARY),
        "claim_boundary": DEFORM360_PUBLIC_CONTACT_PREFIX_CLAIM_BOUNDARY,
    }


def materialize_deform360_public_contact_prefix(
    *,
    prepared_source_inventory_path: str | Path,
    processed_root: str | Path,
    tactile_axis_map_path: str | Path,
    object_id: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Publish one exact public prefix artifact without replacement.

    Exit policy belongs to the CLI: this function returns a valid
    ``support-negative`` artifact when no mapped tactile group has a usable
    first contact event in the frozen prefix.
    """

    inventory_path = _ordinary_file(
        prepared_source_inventory_path,
        name="prepared-source inventory",
    )
    inventory_value, inventory_file_sha256, _inventory_bytes = _load_stable_json_object(
        inventory_path,
        label="prepared-source inventory",
    )
    inventory = validate_deform360_prepared_source_inventory(inventory_value)
    object_name = _literal_string(object_id, name="object_id")
    object_row = _inventory_object(inventory, object_id=object_name)
    axis_map_path = _ordinary_file(tactile_axis_map_path, name="tactile-axis map")
    axis_map_value, axis_map_file_sha256, _axis_map_bytes = _load_stable_json_object(
        axis_map_path,
        label="tactile-axis map",
    )
    axis_map = validate_deform360_tactile_axis_map(axis_map_value)
    _require(axis_map["object_id"] == object_name, "axis-map object differs")
    _require(
        axis_map["episode_id"] == object_row["episode_id"], "axis-map episode differs"
    )
    _require(
        axis_map["prepared_source_inventory_id"] == inventory["inventory_id"],
        "axis map is not bound to this prepared-source inventory",
    )

    root = _ordinary_directory(processed_root, name="processed root")
    episode_files = _mapping(object_row["episode_files"], name="episode_files")
    robot_payload, robot_relative, robot_sha256 = _verified_inventory_file(
        root,
        episode_files.get("robot"),
        name=f"{object_name} robot",
    )
    action_window = _mapping(object_row["action_window"], name="action_window")
    prefix_values = _sequence(
        action_window.get("prefix_raw_frame_range_half_open"),
        name="prefix frame range",
    )
    _require(len(prefix_values) == 2, "prefix frame range must have two bounds")
    prefix_start = _literal_integer(prefix_values[0], name="prefix start")
    prefix_stop = _literal_integer(prefix_values[1], name="prefix stop", minimum=1)
    _require(prefix_start < prefix_stop, "prefix frame range is empty")
    poses, openings, bimanual = _load_robot_prefix(
        robot_payload,
        prefix_start=prefix_start,
        prefix_stop=prefix_stop,
    )
    axis_count = 2 if bimanual else 1
    group_to_axis = cast(Mapping[str, int], axis_map["group_to_robot_axis"])
    _require(
        sorted(group_to_axis.values()) == list(range(axis_count)),
        "axis map must cover every released robot axis exactly once",
    )

    tactile_records: dict[str, Mapping[str, Any]] = {}
    for record_value in _sequence(object_row["tactile"], name="tactile records"):
        record = _mapping(record_value, name="tactile record")
        sensor = _literal_string(record.get("sensor"), name="tactile sensor")
        _require(sensor not in tactile_records, "inventory repeats a tactile sensor")
        tactile_records[sensor] = record

    grouped: dict[str, dict[str, tuple[np.ndarray, str, str]]] = {}
    source_artifacts = {
        "prepared-source-inventory.json": inventory_file_sha256,
        "tactile-axis-map.json": axis_map_file_sha256,
        robot_relative: robot_sha256,
    }
    for sensor, record in sorted(tactile_records.items()):
        try:
            group, side = _gripper_group(sensor)
        except ValueError:
            if sensor in group_to_axis:
                raise
            continue
        if group not in group_to_axis:
            continue
        tactile_payload, relative, tactile_sha256 = _verified_inventory_file(
            root,
            record,
            name=f"{object_name} tactile {sensor}",
        )
        values = _load_tactile_prefix(
            tactile_payload,
            prefix_start=prefix_start,
            prefix_stop=prefix_stop,
        )
        sides = grouped.setdefault(group, {})
        _require(side not in sides, f"tactile group {group} repeats side {side}")
        sides[side] = (values, sensor, relative)
        source_artifacts[relative] = tactile_sha256

    _require(set(grouped) == set(group_to_axis), "axis-map tactile groups are missing")
    for group, sides in grouped.items():
        _require(
            set(sides) == {"left", "right"},
            f"tactile group {group} must contain exactly left and right sensors",
        )

    rows: list[tuple[int, int, str, str, np.ndarray, np.ndarray]] = []
    mapped_groups: list[dict[str, Any]] = []
    for group, axis in sorted(
        group_to_axis.items(), key=lambda item: (item[1], item[0])
    ):
        sides = grouped[group]
        left = sides["left"][0]
        right = sides["right"][0]
        response = _interleave_tactile_sides(left, right)
        active = np.count_nonzero(response > CONTACT_THRESHOLD, axis=1) >= 2
        window = _official_contact_window(active)
        selected_local = np.flatnonzero(window & active)
        event_id: str | None = None
        if len(selected_local):
            window_frames = np.flatnonzero(window)
            event_id = (
                f"deform360-prefix-contact:{group}:"
                f"{prefix_start + int(window_frames[0])}:"
                f"{prefix_start + int(window_frames[-1])}"
            )
            for local_frame in selected_local:
                frame_id = prefix_start + int(local_frame)
                pose = poses[local_frame, axis] if bimanual else poses[local_frame]
                opening = (
                    openings[local_frame, axis] if bimanual else openings[local_frame]
                )
                positions = _gripper_taxel_points(float(opening), pose)
                rows.append(
                    (
                        frame_id,
                        int(axis),
                        group,
                        event_id,
                        response[local_frame].copy(),
                        np.asarray(positions, dtype=np.float64),
                    )
                )
        mapped_groups.append(
            {
                "group": group,
                "robot_axis": int(axis),
                "left_sensor": sides["left"][1],
                "right_sensor": sides["right"][1],
                "event_detected": event_id is not None,
                "active_row_count": int(len(selected_local)),
                "contact_episode_id": event_id,
            }
        )

    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    frame_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    sensors = [row[2] for row in rows]
    episodes = [row[3] for row in rows]
    response_array = (
        np.stack([row[4] for row in rows]).astype(np.float64, copy=False)
        if rows
        else np.empty((0, TAXELS_PER_GRIPPER), dtype=np.float64)
    )
    position_array = (
        np.stack([row[5] for row in rows]).astype(np.float64, copy=False)
        if rows
        else np.empty((0, TAXELS_PER_GRIPPER, 3), dtype=np.float64)
    )
    reliability: np.ndarray = np.ones(len(rows), dtype=np.float64)
    supported_axes = sorted({row[1] for row in rows})
    status = "materialized" if rows else "support-negative"
    negative_reason = None if rows else "no-mapped-first-contact-event-in-frozen-prefix"

    target = Path(output_directory).absolute()
    target_parent = target.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    _ordinary_directory(target_parent, name="output parent")
    lock = target_parent / f".{target.name}.lock"
    temporary = target_parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    lock_descriptor: int | None = None
    lock_created = False
    try:
        temporary.mkdir(mode=0o700)
        np.save(temporary / "frame-ids.npy", frame_ids, allow_pickle=False)
        _write_json(temporary / "sensor-names.json", sensors)
        _write_json(temporary / "contact-episode-ids.json", episodes)
        np.save(
            temporary / "tactile-response.npy",
            response_array,
            allow_pickle=False,
        )
        np.save(
            temporary / "taxel-world-positions-m.npy",
            position_array,
            allow_pickle=False,
        )
        np.save(
            temporary / "source-reliability.npy",
            reliability,
            allow_pickle=False,
        )
        source_artifacts = dict(sorted(source_artifacts.items()))
        _write_json(temporary / "source-artifacts.json", source_artifacts)
        files = {name: _sha256_file(temporary / name) for name in _DATA_FILE_NAMES}
        identity = _materialization_identity(
            status=status,
            support_negative_reason=negative_reason,
            object_row=object_row,
            inventory=inventory,
            inventory_file_sha256=inventory_file_sha256,
            axis_map=axis_map,
            axis_map_file_sha256=axis_map_file_sha256,
            bimanual=bimanual,
            prefix_range=[prefix_start, prefix_stop],
            mapped_groups=mapped_groups,
            supported_axes=supported_axes,
            row_count=len(rows),
            contact_episode_count=len(set(episodes)),
            files=files,
            source_artifacts=source_artifacts,
        )
        manifest = {**identity, "materialization_id": content_id(identity)}
        _write_json(temporary / "contact-prefix.json", manifest)
        checksum_names = (*_DATA_FILE_NAMES, "contact-prefix.json")
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256_file(temporary / name)}  {name}\n"
                for name in sorted(checksum_names)
            ),
            encoding="ascii",
        )
        validate_deform360_public_contact_prefix(temporary)
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        lock_created = True
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(lock_descriptor)
        lock_descriptor = None
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if lock_created:
            lock.unlink(missing_ok=True)
    return manifest


def validate_deform360_public_contact_prefix(
    directory: str | Path,
) -> dict[str, Any]:
    """Validate one published or staged public contact-prefix artifact."""

    root = _ordinary_directory(directory, name="contact-prefix directory")
    expected_names = set(_DATA_FILE_NAMES) | {"contact-prefix.json", "SHA256SUMS"}
    actual_names = {path.name for path in root.iterdir()}
    _require(actual_names == expected_names, "contact-prefix file roster changed")
    manifest_path = _ordinary_file(root / "contact-prefix.json", name="contact prefix")
    manifest_value, _manifest_sha256, _manifest_bytes = _load_stable_json_object(
        manifest_path,
        label="contact prefix",
    )
    manifest = dict(
        _mapping(
            manifest_value,
            name="contact prefix",
        )
    )
    require_exact_fields(manifest, expected=_PREFIX_FIELDS, name="contact prefix")
    _require(
        manifest["schema"] == DEFORM360_PUBLIC_CONTACT_PREFIX_SCHEMA,
        "contact-prefix schema changed",
    )
    _require(
        manifest["schema_version"] == DEFORM360_PUBLIC_CONTACT_PREFIX_VERSION,
        "contact-prefix version changed",
    )
    _require(
        manifest["semantics"] == DEFORM360_PUBLIC_CONTACT_PREFIX_SEMANTICS,
        "contact-prefix semantics changed",
    )
    status = _literal_string(manifest["status"], name="status")
    _require(status in {"materialized", "support-negative"}, "status changed")
    row_count = _literal_integer(manifest["row_count"], name="row_count")
    _require(
        (status == "materialized" and row_count > 0)
        or (status == "support-negative" and row_count == 0),
        "contact-prefix status and row count disagree",
    )
    _require(
        manifest["information_boundary"] == _PREFIX_BOUNDARY,
        "contact-prefix information boundary changed",
    )
    _require(
        manifest["claim_boundary"] == DEFORM360_PUBLIC_CONTACT_PREFIX_CLAIM_BOUNDARY,
        "contact-prefix claim boundary changed",
    )
    _require(
        manifest["reliability_policy"]
        == {
            "source_reliability_value": 1.0,
            "depends_on_tactile_magnitude": False,
            "depends_on_physical_prediction": False,
            "depends_on_state_innovation": False,
            "innovation_likelihood_application_count": 0,
        },
        "contact-prefix reliability policy changed",
    )
    if status == "materialized":
        _require(
            manifest["support_negative_reason"] is None,
            "materialized contact prefix has a negative reason",
        )
    else:
        _require(
            manifest["support_negative_reason"]
            == "no-mapped-first-contact-event-in-frozen-prefix",
            "support-negative reason changed",
        )
    prefix_range = _sequence(
        manifest["prefix_raw_frame_range_half_open"],
        name="prefix frame range",
    )
    _require(len(prefix_range) == 2, "prefix frame range must have two bounds")
    prefix_start = _literal_integer(prefix_range[0], name="prefix start")
    prefix_stop = _literal_integer(prefix_range[1], name="prefix stop", minimum=1)
    _require(prefix_start < prefix_stop, "prefix frame range is empty")
    _require(
        manifest["causal_frame_stop"] == prefix_stop,
        "causal frame stop differs from prefix stop",
    )
    bimanual = manifest["bimanual"]
    _require(type(bimanual) is bool, "bimanual must be Boolean")
    axis_count = _literal_integer(
        manifest["robot_axis_count"],
        name="robot_axis_count",
        minimum=1,
    )
    _require(axis_count == (2 if bimanual else 1), "robot axis count changed")
    mapped_groups = _sequence(manifest["mapped_groups"], name="mapped groups")
    _require(len(mapped_groups) == axis_count, "mapped groups do not cover robot axes")
    mapped_names: list[str] = []
    mapped_axes: list[int] = []
    mapped_episode_ids: set[str] = set()
    for item_value in mapped_groups:
        item = _mapping(item_value, name="mapped group")
        require_exact_fields(
            item,
            expected=frozenset(
                {
                    "group",
                    "robot_axis",
                    "left_sensor",
                    "right_sensor",
                    "event_detected",
                    "active_row_count",
                    "contact_episode_id",
                }
            ),
            name="mapped group",
        )
        group = _literal_string(item["group"], name="mapped group name")
        axis = _literal_integer(item["robot_axis"], name="mapped robot axis")
        _require(axis < axis_count, "mapped robot axis is out of range")
        _require(
            item["left_sensor"] == f"{group}_left"
            and item["right_sensor"] == f"{group}_right",
            "mapped sensor sides changed",
        )
        detected = item["event_detected"]
        _require(type(detected) is bool, "event_detected must be Boolean")
        active_rows = _literal_integer(
            item["active_row_count"],
            name="active_row_count",
        )
        episode_id = item["contact_episode_id"]
        _require(
            bool(
                (
                    detected
                    and active_rows > 0
                    and type(episode_id) is str
                    and episode_id
                )
                or (not detected and active_rows == 0 and episode_id is None)
            ),
            "mapped event fields disagree",
        )
        if type(episode_id) is str:
            mapped_episode_ids.add(episode_id)
        mapped_names.append(group)
        mapped_axes.append(axis)
    _require(mapped_axes == list(range(axis_count)), "mapped group order changed")
    _require(len(set(mapped_names)) == axis_count, "mapped group names repeat")
    supported_axes = list(
        _sequence(manifest["supported_robot_axes"], name="supported robot axes")
    )
    missing_axes = list(
        _sequence(
            manifest["missing_contact_robot_axes"],
            name="missing contact robot axes",
        )
    )
    _require(
        all(type(axis) is int for axis in (*supported_axes, *missing_axes)),
        "support axis lists must contain integers",
    )
    _require(
        supported_axes == sorted(set(supported_axes))
        and missing_axes == sorted(set(missing_axes))
        and sorted((*supported_axes, *missing_axes)) == list(range(axis_count)),
        "support axis partition changed",
    )
    files = _mapping(manifest["files"], name="contact-prefix files")
    _require(set(files) == set(_DATA_FILE_NAMES), "contact-prefix data roster changed")
    for name in _DATA_FILE_NAMES:
        expected = sha256_digest(files[name], name=f"{name} digest")
        _require(_sha256_file(root / name) == expected, f"{name} digest changed")
    frames = np.load(root / "frame-ids.npy", allow_pickle=False)
    response = np.load(root / "tactile-response.npy", allow_pickle=False)
    positions = np.load(root / "taxel-world-positions-m.npy", allow_pickle=False)
    reliability = np.load(root / "source-reliability.npy", allow_pickle=False)
    sensors = json.loads((root / "sensor-names.json").read_text(encoding="utf-8"))
    episodes = json.loads(
        (root / "contact-episode-ids.json").read_text(encoding="utf-8")
    )
    _require(frames.shape == (row_count,), "frame-id shape changed")
    _require(frames.dtype == np.dtype(np.int64), "frame-id dtype changed")
    _require(
        response.shape == (row_count, TAXELS_PER_GRIPPER)
        and positions.shape == (row_count, TAXELS_PER_GRIPPER, 3)
        and reliability.shape == (row_count,),
        "contact-prefix array shape changed",
    )
    _require(
        bool(np.all(np.isfinite(response)))
        and bool(np.all(np.isfinite(positions)))
        and bool(np.all(reliability == 1.0)),
        "contact-prefix values or neutral reliability changed",
    )
    _require(
        isinstance(sensors, list)
        and isinstance(episodes, list)
        and len(sensors) == len(episodes) == row_count
        and all(type(value) is str and value for value in (*sensors, *episodes)),
        "contact-prefix identity rows changed",
    )
    _require(
        bool(np.all((frames >= prefix_start) & (frames < prefix_stop))),
        "contact-prefix frame IDs leave the frozen prefix",
    )
    _require(
        row_count < 2 or bool(np.all(frames[1:] >= frames[:-1])),
        "contact-prefix frame IDs are not chronological",
    )
    _require(
        row_count == 0
        or bool(np.all(np.count_nonzero(response > CONTACT_THRESHOLD, axis=1) >= 2)),
        "contact-prefix row has fewer than two active taxels",
    )
    _require(set(sensors) <= set(mapped_names), "contact-prefix sensor group changed")
    _require(
        set(episodes) == mapped_episode_ids,
        "contact-prefix episode identities differ from mapped groups",
    )
    _require(
        manifest["contact_episode_count"] == len(mapped_episode_ids),
        "contact episode count changed",
    )
    source_artifacts = _mapping(
        manifest["source_artifacts"],
        name="source artifacts",
    )
    stored_source_artifacts = json.loads(
        (root / "source-artifacts.json").read_text(encoding="utf-8")
    )
    _require(
        stored_source_artifacts == source_artifacts,
        "source-artifacts file differs from manifest",
    )
    for source_path, source_digest in source_artifacts.items():
        canonical_relative_posix_path(source_path, name="source artifact path")
        sha256_digest(source_digest, name=f"source artifact {source_path}")
    identity = {
        key: value for key, value in manifest.items() if key != "materialization_id"
    }
    declared = sha256_digest(
        manifest["materialization_id"],
        name="materialization_id",
    )
    _require(
        declared == content_id(identity), "materialization_id does not match content"
    )

    checksum_lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    expected_lines = [
        f"{_sha256_file(root / name)}  {name}"
        for name in sorted((*_DATA_FILE_NAMES, "contact-prefix.json"))
    ]
    _require(checksum_lines == expected_lines, "SHA256SUMS changed")
    return manifest


__all__ = [
    "DEFORM360_CONTACT_PATIENCE_FRAMES",
    "DEFORM360_PUBLIC_CONTACT_PREFIX_CLAIM_BOUNDARY",
    "DEFORM360_PUBLIC_CONTACT_PREFIX_SCHEMA",
    "DEFORM360_PUBLIC_CONTACT_PREFIX_SEMANTICS",
    "DEFORM360_PUBLIC_CONTACT_PREFIX_VERSION",
    "DEFORM360_TACTILE_AXIS_MAP_CLAIM_BOUNDARY",
    "DEFORM360_TACTILE_AXIS_MAP_SCHEMA",
    "DEFORM360_TACTILE_AXIS_MAP_SEMANTICS",
    "DEFORM360_TACTILE_AXIS_MAP_VERSION",
    "TAXELS_PER_GRIPPER",
    "build_deform360_tactile_axis_map",
    "load_deform360_tactile_axis_map",
    "materialize_deform360_public_contact_prefix",
    "save_deform360_tactile_axis_map",
    "validate_deform360_public_contact_prefix",
    "validate_deform360_tactile_axis_map",
]
