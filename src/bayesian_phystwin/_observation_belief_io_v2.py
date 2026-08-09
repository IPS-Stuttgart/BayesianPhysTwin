"""Strict container preflight helpers for observation-belief I/O v2."""

from __future__ import annotations

import json
import math
import os
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, cast

import numpy as np

_DESCRIPTOR_FIELDS: Final = frozenset(
    {
        "schema_name",
        "schema_version",
        "case_id",
        "stream_id",
        "causal_frame_stop",
        "view_names",
        "window_names",
        "factor_names",
        "source_repository",
        "source_revision",
        "source_artifact_sha256",
        "metadata",
        "artifact_id",
    }
)
_ARRAY_DTYPES: Final = {
    "declared_frame_ids": np.dtype(np.int64),
    "mean_xyz_m": np.dtype(np.float64),
    "frame_ids": np.dtype(np.int64),
    "entity_ids": np.dtype(np.int64),
    "view_indices": np.dtype(np.int64),
    "window_indices": np.dtype(np.int64),
    "correlation_group_ids": np.dtype(np.int64),
    "factor_group_ids": np.dtype(np.int64),
    "prior_reliability": np.dtype(np.float64),
    "association_probability": np.dtype(np.float64),
    "local_covariance_m2": np.dtype(np.float64),
    "low_rank_factor_m": np.dtype(np.float64),
    "group_ids": np.dtype(np.int64),
    "group_prior_nominal_probability": np.dtype(np.float64),
    "group_composite_weight": np.dtype(np.float64),
}
_REQUIRED_ZIP_MEMBERS: Final = frozenset(
    {"descriptor_json.npy", *(f"{name}.npy" for name in _ARRAY_DTYPES)}
)
_ALLOWED_COMPRESSIONS: Final = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


@dataclass(frozen=True)
class ObservationBeliefIOLimitsV2:
    """Fail-closed resource budgets for one serialized observation belief."""

    maximum_archive_bytes: int = 512 * 1024 * 1024
    maximum_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    maximum_member_bytes: int = 1024 * 1024 * 1024
    maximum_descriptor_bytes: int = 4 * 1024 * 1024
    maximum_npy_header_bytes: int = 64 * 1024
    maximum_observation_count: int = 5_000_000
    maximum_declared_frame_count: int = 5_000_000
    maximum_group_count: int = 5_000_000
    maximum_factor_rank: int = 4096

    def __post_init__(self) -> None:
        values = {
            name: getattr(self, name)
            for name in (
                "maximum_archive_bytes",
                "maximum_uncompressed_bytes",
                "maximum_member_bytes",
                "maximum_descriptor_bytes",
                "maximum_npy_header_bytes",
                "maximum_observation_count",
                "maximum_declared_frame_count",
                "maximum_group_count",
                "maximum_factor_rank",
            )
        }
        for name, raw in values.items():
            if isinstance(raw, (bool, np.bool_)) or not isinstance(
                raw, (int, np.integer)
            ):
                raise TypeError(f"{name} must be a genuine integer")
            value = int(raw)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.maximum_member_bytes > self.maximum_uncompressed_bytes:
            raise ValueError(
                "maximum_member_bytes cannot exceed maximum_uncompressed_bytes"
            )
        if self.maximum_descriptor_bytes > self.maximum_member_bytes:
            raise ValueError(
                "maximum_descriptor_bytes cannot exceed maximum_member_bytes"
            )


@dataclass(frozen=True)
class NpyMemberV2:
    """Bounded structural information parsed from one NPY member."""

    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    payload_bytes: int


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"descriptor_json contains non-finite constant {value!r}")


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"descriptor_json contains duplicate key {key!r}")
        result[key] = value
    return result


def strict_descriptor_json(text: str) -> Mapping[str, Any]:
    """Parse duplicate-free finite JSON and require one descriptor mapping."""

    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("descriptor_json is not valid JSON") from error
    except TypeError as error:
        raise ValueError("descriptor_json is not strict finite JSON") from error
    except ValueError as error:
        raise ValueError(str(error)) from error
    if not isinstance(value, Mapping):
        raise ValueError("observation descriptor must be a JSON object")
    return cast(Mapping[str, Any], value)


def validate_sha256(value: object, *, name: str) -> str:
    """Return one strict lowercase SHA-256 digest."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def ordinary_input_stream(path: Path) -> tuple[BinaryIO, os.stat_result]:
    """Open one ordinary regular file without following its final symlink."""

    if not hasattr(os, "O_NOFOLLOW"):
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError("observation artifact must not be a symbolic link")
        except FileNotFoundError as error:
            raise ValueError("observation artifact does not exist") from error
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            "cannot open observation artifact as an ordinary file"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("observation artifact must be a regular file")
        return os.fdopen(descriptor, "rb"), metadata
    except Exception:
        os.close(descriptor)
        raise


def _read_npy_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    limits: ObservationBeliefIOLimitsV2,
) -> NpyMemberV2:
    with archive.open(info, "r") as stream:
        try:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(
                    stream,
                    max_header_size=limits.maximum_npy_header_bytes,
                )  # type: ignore[call-arg]
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(
                    stream,
                    max_header_size=limits.maximum_npy_header_bytes,
                )  # type: ignore[call-arg]
            else:
                raise ValueError(f"unsupported NPY format version {version!r}")
        except (EOFError, TypeError, ValueError) as error:
            raise ValueError(f"invalid NPY header in {info.filename}") from error
        normalized_shape = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in normalized_shape):
            raise ValueError(f"negative array dimension in {info.filename}")
        normalized_dtype = np.dtype(dtype)
        if normalized_dtype.hasobject:
            raise ValueError(f"object dtype is forbidden in {info.filename}")
        payload_bytes = math.prod(normalized_shape) * normalized_dtype.itemsize
        if payload_bytes > limits.maximum_member_bytes:
            raise ValueError(f"{info.filename} exceeds its decoded member budget")
        if stream.tell() + payload_bytes != info.file_size:
            raise ValueError(f"{info.filename} size disagrees with its NPY header")
        return NpyMemberV2(normalized_shape, normalized_dtype, payload_bytes)


def _require_shape(
    headers: Mapping[str, NpyMemberV2],
    name: str,
    shape: tuple[int, ...],
) -> None:
    if headers[f"{name}.npy"].shape != shape:
        raise ValueError(f"{name} must declare shape {shape}")


def preflight_archive(
    stream: BinaryIO,
    *,
    limits: ObservationBeliefIOLimitsV2,
) -> Mapping[str, NpyMemberV2]:
    """Validate ZIP and NPY structure before NumPy allocates arrays."""

    try:
        with zipfile.ZipFile(stream, "r") as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError("observation artifact contains duplicate ZIP members")
            if set(names) != _REQUIRED_ZIP_MEMBERS:
                missing = sorted(_REQUIRED_ZIP_MEMBERS - set(names))
                extra = sorted(set(names) - _REQUIRED_ZIP_MEMBERS)
                raise ValueError(
                    "observation artifact members changed; "
                    f"missing={missing}, extra={extra}"
                )
            total = 0
            headers: dict[str, NpyMemberV2] = {}
            for entry in entries:
                if entry.is_dir() or entry.flag_bits & 0x1:
                    raise ValueError("directory or encrypted ZIP member is forbidden")
                if entry.compress_type not in _ALLOWED_COMPRESSIONS:
                    raise ValueError("unsupported ZIP compression method")
                if entry.file_size < 1 or entry.file_size > limits.maximum_member_bytes:
                    raise ValueError(f"{entry.filename} exceeds its member budget")
                total += entry.file_size
                if total > limits.maximum_uncompressed_bytes:
                    raise ValueError(
                        "observation artifact exceeds the uncompressed byte budget"
                    )
                headers[entry.filename] = _read_npy_header(
                    archive,
                    entry,
                    limits=limits,
                )
    except zipfile.BadZipFile as error:
        raise ValueError("observation artifact is not a valid ZIP archive") from error

    descriptor = headers["descriptor_json.npy"]
    if descriptor.shape != () or descriptor.dtype.kind not in {"S", "U"}:
        raise ValueError("descriptor_json must be one scalar byte or Unicode string")
    if descriptor.payload_bytes > limits.maximum_descriptor_bytes:
        raise ValueError("descriptor_json exceeds its decoded byte budget")
    for name, expected_dtype in _ARRAY_DTYPES.items():
        if headers[f"{name}.npy"].dtype != expected_dtype:
            raise ValueError(f"{name} must have exact dtype {expected_dtype.name}")

    mean_shape = headers["mean_xyz_m.npy"].shape
    if len(mean_shape) != 2 or mean_shape[1:] != (3,):
        raise ValueError("mean_xyz_m must declare shape (N, 3)")
    count = mean_shape[0]
    if count < 1 or count > limits.maximum_observation_count:
        raise ValueError("observation count exceeds the configured budget")
    factor_shape = headers["low_rank_factor_m.npy"].shape
    if len(factor_shape) != 3 or factor_shape[:2] != (count, 3):
        raise ValueError("low_rank_factor_m must declare shape (N, 3, R)")
    if factor_shape[2] > limits.maximum_factor_rank:
        raise ValueError("low-rank factor rank exceeds the configured budget")
    for name in (
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "correlation_group_ids",
        "factor_group_ids",
        "prior_reliability",
        "association_probability",
    ):
        _require_shape(headers, name, (count,))
    _require_shape(headers, "local_covariance_m2", (count, 3, 3))

    declared_shape = headers["declared_frame_ids.npy"].shape
    if len(declared_shape) != 1:
        raise ValueError("declared_frame_ids must declare one dimension")
    if declared_shape[0] > limits.maximum_declared_frame_count:
        raise ValueError("declared frame count exceeds the configured budget")
    group_shape = headers["group_ids.npy"].shape
    if len(group_shape) != 1:
        raise ValueError("group_ids must declare one dimension")
    group_count = group_shape[0]
    if group_count < 1 or group_count > limits.maximum_group_count:
        raise ValueError("group count exceeds the configured budget")
    _require_shape(headers, "group_prior_nominal_probability", (group_count,))
    _require_shape(headers, "group_composite_weight", (group_count,))
    return headers


def snapshot_unchanged(before: os.stat_result, after: os.stat_result) -> bool:
    """Compare the stable file identity and mutation-relevant metadata."""

    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


__all__ = [
    "ObservationBeliefIOLimitsV2",
]
