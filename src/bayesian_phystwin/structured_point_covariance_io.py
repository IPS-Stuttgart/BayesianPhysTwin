"""Strict portable I/O for structured point-covariance artifacts.

``StructuredPointCovarianceV1`` already preserves conditional point-local
covariance and labeled coherent low-rank roots. This module adds a bounded,
content-verified NPZ boundary so that decomposition can cross repository and
process boundaries without collapsing to dense or marginal covariance.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from .structured_point_covariance import (
    SHARED_COVARIANCE_COMPONENTS,
    STRUCTURED_POINT_COVARIANCE_SCHEMA,
    STRUCTURED_POINT_COVARIANCE_SEMANTICS,
    STRUCTURED_POINT_COVARIANCE_VERSION,
    StructuredPointCovarianceV1,
)

STRUCTURED_POINT_COVARIANCE_ARCHIVE_SCHEMA: Final = (
    "bayesian_phystwin.structured_point_covariance_archive"
)
STRUCTURED_POINT_COVARIANCE_ARCHIVE_VERSION: Final = 1
STRUCTURED_POINT_COVARIANCE_ARCHIVE_SEMANTICS: Final = (
    "bounded-strict-npz-no-clobber-publication-v1"
)
STRUCTURED_POINT_COVARIANCE_ARCHIVE_CLAIM_BOUNDARY: Final = (
    "This archive preserves covariance decomposition, units, coordinate frame, "
    "lineage, and content identity. It does not establish covariance calibration, "
    "physical-mechanism identification, downstream benefit, deployment safety, "
    "or state of the art."
)

_DESCRIPTOR_MEMBER = "descriptor_json.npy"
_LOCAL_MEMBER = "local_covariance_m2.npy"
_FACTOR_PREFIX = "shared_factor__"
_ARCHIVE_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "covariance_descriptor",
        "artifact_id",
    }
)
_COVARIANCE_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "point_ids",
        "coordinate_frame",
        "source_artifact_id",
        "calibration_artifact_id",
        "local_covariance_m2",
        "shared_factors_m",
        "metadata",
    }
)


@dataclass(frozen=True, slots=True)
class StructuredPointCovarianceIOLimitsV1:
    """Resource limits applied before and during structured-covariance loading."""

    maximum_archive_bytes: int = 256 * 1024 * 1024
    maximum_uncompressed_bytes: int = 1024 * 1024 * 1024
    maximum_descriptor_bytes: int = 1024 * 1024
    maximum_points: int = 1_000_000
    maximum_total_shared_rank: int = 4096
    maximum_compression_ratio: float = 10_000.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_archive_bytes",
            "maximum_uncompressed_bytes",
            "maximum_descriptor_bytes",
            "maximum_points",
            "maximum_total_shared_rank",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        ratio = self.maximum_compression_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError("maximum_compression_ratio must be a finite number")
        normalized = float(ratio)
        if not np.isfinite(normalized) or normalized < 1.0:
            raise ValueError(
                "maximum_compression_ratio must be finite and at least one"
            )
        object.__setattr__(self, "maximum_compression_ratio", normalized)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _strict_json_object(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError("descriptor_json is not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise ValueError("descriptor_json root must be an object")
    return parsed


def _exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _genuine_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _lower_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _ordinary_input(path: Path) -> tuple[Any, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(
            f"cannot inspect structured covariance archive {path}"
        ) from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("structured covariance archive must be an ordinary file")
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open structured covariance archive {path}") from error
    stream = os.fdopen(descriptor, "rb")
    after_open = os.fstat(stream.fileno())
    if not stat.S_ISREG(after_open.st_mode):
        stream.close()
        raise ValueError("structured covariance archive must remain an ordinary file")
    if (before.st_dev, before.st_ino) != (after_open.st_dev, after_open.st_ino):
        stream.close()
        raise ValueError("structured covariance archive changed before opening")
    return stream, after_open


def _snapshot_unchanged(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )


def _preflight_zip(
    stream: Any,
    *,
    limits: StructuredPointCovarianceIOLimitsV1,
) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("structured covariance archive is not a valid NPZ") from error
    if not members:
        raise ValueError("structured covariance archive is empty")
    names = [member.filename for member in members]
    if len(set(names)) != len(names):
        raise ValueError("structured covariance archive contains duplicate members")
    if any(member.is_dir() or member.flag_bits & 0x1 for member in members):
        raise ValueError("structured covariance archive contains an invalid member")
    if any("/" in name or "\\" in name or name.startswith(".") for name in names):
        raise ValueError("structured covariance archive member names are not canonical")
    if _DESCRIPTOR_MEMBER not in names or _LOCAL_MEMBER not in names:
        raise ValueError("structured covariance archive lacks required members")
    total_uncompressed = sum(member.file_size for member in members)
    total_compressed = sum(max(member.compress_size, 1) for member in members)
    if total_uncompressed > limits.maximum_uncompressed_bytes:
        raise ValueError(
            "structured covariance archive exceeds uncompressed byte budget"
        )
    if total_uncompressed / total_compressed > limits.maximum_compression_ratio:
        raise ValueError(
            "structured covariance archive exceeds compression-ratio budget"
        )
    return tuple(names)


def _descriptor_from_member(
    archive: Any,
    *,
    limits: StructuredPointCovarianceIOLimitsV1,
) -> Mapping[str, Any]:
    descriptor_member = np.asarray(archive["descriptor_json"])
    if descriptor_member.shape != ():
        raise ValueError("descriptor_json must be a scalar array")
    raw = descriptor_member.item()
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("descriptor_json is not valid UTF-8") from error
    if type(raw) is not str:
        raise ValueError("descriptor_json must contain a string")
    if len(raw.encode("utf-8")) > limits.maximum_descriptor_bytes:
        raise ValueError("descriptor_json exceeds its byte budget")
    return _strict_json_object(raw)


def _component_names(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    factors = descriptor.get("shared_factors_m")
    if not isinstance(factors, Mapping):
        raise ValueError("shared_factors_m descriptor must be an object")
    names = tuple(factors)
    if names != tuple(sorted(names)):
        raise ValueError("shared covariance component names must be sorted")
    if len(set(names)) != len(names):
        raise ValueError("shared covariance component names must be unique")
    allowed = set(SHARED_COVARIANCE_COMPONENTS)
    if any(type(name) is not str or name not in allowed for name in names):
        raise ValueError("shared covariance component name is unsupported")
    return names


def _archive_descriptor(covariance: StructuredPointCovarianceV1) -> dict[str, object]:
    return {
        "schema": STRUCTURED_POINT_COVARIANCE_ARCHIVE_SCHEMA,
        "schema_version": STRUCTURED_POINT_COVARIANCE_ARCHIVE_VERSION,
        "semantics": STRUCTURED_POINT_COVARIANCE_ARCHIVE_SEMANTICS,
        "covariance_descriptor": covariance.descriptor(),
        "artifact_id": covariance.artifact_id,
    }


def load_structured_point_covariance(
    path: str | Path,
    *,
    limits: StructuredPointCovarianceIOLimitsV1 | None = None,
) -> StructuredPointCovarianceV1:
    """Load a bounded archive and verify its complete covariance content identity."""

    settings = limits or StructuredPointCovarianceIOLimitsV1()
    if not isinstance(settings, StructuredPointCovarianceIOLimitsV1):
        raise TypeError("limits must be a StructuredPointCovarianceIOLimitsV1")
    source = Path(path)
    stream, before = _ordinary_input(source)
    with stream:
        if before.st_size < 1 or before.st_size > settings.maximum_archive_bytes:
            raise ValueError(
                "structured covariance archive exceeds archive byte budget"
            )
        member_names = _preflight_zip(stream, limits=settings)
        stream.seek(0)
        try:
            with np.load(stream, allow_pickle=False) as archive:
                descriptor = _descriptor_from_member(archive, limits=settings)
                _exact_fields(
                    descriptor,
                    expected=_ARCHIVE_DESCRIPTOR_FIELDS,
                    name="archive descriptor",
                )
                if descriptor["schema"] != STRUCTURED_POINT_COVARIANCE_ARCHIVE_SCHEMA:
                    raise ValueError("structured covariance archive schema changed")
                if (
                    _genuine_integer(
                        descriptor["schema_version"],
                        name="archive schema_version",
                    )
                    != STRUCTURED_POINT_COVARIANCE_ARCHIVE_VERSION
                ):
                    raise ValueError("structured covariance archive version changed")
                if (
                    descriptor["semantics"]
                    != STRUCTURED_POINT_COVARIANCE_ARCHIVE_SEMANTICS
                ):
                    raise ValueError("structured covariance archive semantics changed")
                covariance_descriptor = descriptor["covariance_descriptor"]
                if not isinstance(covariance_descriptor, Mapping):
                    raise ValueError("covariance_descriptor must be an object")
                _exact_fields(
                    covariance_descriptor,
                    expected=_COVARIANCE_DESCRIPTOR_FIELDS,
                    name="covariance descriptor",
                )
                if (
                    covariance_descriptor["schema"]
                    != STRUCTURED_POINT_COVARIANCE_SCHEMA
                ):
                    raise ValueError("structured covariance schema changed")
                if (
                    _genuine_integer(
                        covariance_descriptor["schema_version"],
                        name="covariance schema_version",
                    )
                    != STRUCTURED_POINT_COVARIANCE_VERSION
                ):
                    raise ValueError("structured covariance version changed")
                if (
                    covariance_descriptor["semantics"]
                    != STRUCTURED_POINT_COVARIANCE_SEMANTICS
                ):
                    raise ValueError("structured covariance semantics changed")
                names = _component_names(covariance_descriptor)
                expected_members = {
                    _DESCRIPTOR_MEMBER,
                    _LOCAL_MEMBER,
                    *{f"{_FACTOR_PREFIX}{name}.npy" for name in names},
                }
                if set(member_names) != expected_members:
                    raise ValueError("structured covariance archive member set changed")

                local = np.asarray(archive["local_covariance_m2"])
                if local.dtype != np.dtype(np.float64):
                    raise ValueError(
                        "local_covariance_m2 must have exact dtype float64"
                    )
                if local.ndim != 3 or local.shape[1:] != (3, 3):
                    raise ValueError(
                        "local_covariance_m2 must have shape (point, 3, 3)"
                    )
                if not 0 < local.shape[0] <= settings.maximum_points:
                    raise ValueError("structured covariance point count exceeds budget")
                shared: dict[str, np.ndarray] = {}
                total_rank = 0
                for name in names:
                    factor = np.asarray(archive[f"{_FACTOR_PREFIX}{name}"])
                    if factor.dtype != np.dtype(np.float64):
                        raise ValueError(
                            f"shared factor {name} must have exact dtype float64"
                        )
                    if factor.ndim != 3 or factor.shape[:2] != (local.shape[0], 3):
                        raise ValueError(
                            f"shared factor {name} must have shape (point, 3, rank)"
                        )
                    if factor.shape[2] < 1:
                        raise ValueError(
                            f"shared factor {name} must have positive rank"
                        )
                    total_rank += int(factor.shape[2])
                    if total_rank > settings.maximum_total_shared_rank:
                        raise ValueError(
                            "structured covariance shared rank exceeds budget"
                        )
                    shared[name] = factor
        except (
            OSError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError("cannot decode structured covariance archive") from error
        after = os.fstat(stream.fileno())
        if not _snapshot_unchanged(before, after):
            raise ValueError("structured covariance archive changed while being read")

    artifact_id = _lower_sha256(descriptor["artifact_id"], name="artifact_id")
    point_ids = covariance_descriptor["point_ids"]
    if not isinstance(point_ids, list) or any(
        type(value) is not str for value in point_ids
    ):
        raise ValueError("point_ids must be a JSON string array")
    metadata = covariance_descriptor["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("structured covariance metadata must be an object")
    calibration_id = covariance_descriptor["calibration_artifact_id"]
    if calibration_id is not None and type(calibration_id) is not str:
        raise ValueError("calibration_artifact_id must be a string or null")
    result = StructuredPointCovarianceV1(
        point_ids=tuple(point_ids),
        local_covariance_m2=local,
        shared_factors_m=shared,
        coordinate_frame=cast(str, covariance_descriptor["coordinate_frame"]),
        source_artifact_id=cast(str, covariance_descriptor["source_artifact_id"]),
        calibration_artifact_id=cast(str | None, calibration_id),
        metadata=cast(Mapping[str, Any], metadata),
        artifact_id=artifact_id,
    )
    if result.descriptor() != dict(covariance_descriptor):
        raise ValueError(
            "structured covariance descriptor changed during reconstruction"
        )
    return result


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def write_structured_point_covariance(
    path: str | Path,
    covariance: StructuredPointCovarianceV1,
    *,
    overwrite: bool = False,
    limits: StructuredPointCovarianceIOLimitsV1 | None = None,
) -> None:
    """Publish one verified archive atomically, without clobbering by default."""

    if not isinstance(covariance, StructuredPointCovarianceV1):
        raise TypeError("covariance must be a StructuredPointCovarianceV1")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    settings = limits or StructuredPointCovarianceIOLimitsV1()
    if not isinstance(settings, StructuredPointCovarianceIOLimitsV1):
        raise TypeError("limits must be a StructuredPointCovarianceIOLimitsV1")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir():
        raise ValueError("structured covariance target must not be a directory")

    descriptor = _archive_descriptor(covariance)
    payload: dict[str, object] = {
        "descriptor_json": np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "local_covariance_m2": covariance.local_covariance_m2,
    }
    payload.update(
        {
            f"{_FACTOR_PREFIX}{name}": factor
            for name, factor in covariance.shared_factors_m.items()
        }
    )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w+b") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        verified = load_structured_point_covariance(temporary, limits=settings)
        if verified.artifact_id != covariance.artifact_id:
            raise ValueError("serialized structured covariance changed identity")
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError:
                raise FileExistsError(target) from None
            temporary.unlink()
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "STRUCTURED_POINT_COVARIANCE_ARCHIVE_CLAIM_BOUNDARY",
    "STRUCTURED_POINT_COVARIANCE_ARCHIVE_SCHEMA",
    "STRUCTURED_POINT_COVARIANCE_ARCHIVE_SEMANTICS",
    "STRUCTURED_POINT_COVARIANCE_ARCHIVE_VERSION",
    "StructuredPointCovarianceIOLimitsV1",
    "load_structured_point_covariance",
    "write_structured_point_covariance",
]
