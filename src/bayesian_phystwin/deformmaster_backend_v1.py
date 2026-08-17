"""Causal DeformMaster rollout ingestion for Bayesian-PhysTwin.

The public DeformMaster release exposes an inference runtime, but its dataset
loader does not define an observation/future boundary.  This module therefore
adapts only producer-attested, prefix-causal surface rollouts.  It deliberately
does not import DeformMaster or claim that the current public loader satisfies
the contract.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .physical_rollout_v1 import (
    load_physical_rollout_archive,
    validate_physical_rollout_arrays,
    write_deterministic_npz,
)

DEFORMMASTER_RUNTIME_SCHEMA: Final = "bayesian-phystwin.deformmaster-causal-runtime"
DEFORMMASTER_TRAINING_SCHEMA: Final = "bayesian-phystwin.deformmaster-training-data"
DEFORMMASTER_ARTIFACT_SCHEMA: Final = "bayesian-phystwin.deformmaster-backend"
DEFORMMASTER_SCHEMA_VERSION: Final = 1
DEFORMMASTER_BACKEND_KIND: Final = "deformmaster-mpm-neural-v1"
DEFORMMASTER_SOURCE_REPOSITORY: Final = "CAN-Lee/DeformMaster"
DEFORMMASTER_RAW_ARRAY_NAMES: Final = frozenset(
    {
        "driven_surface_positions_m",
        "zero_action_surface_positions_m",
        "action_support",
        "frame_zero_points_m",
    }
)

PHYSICAL_ARCHIVE_FILENAME: Final = "physical-prediction.npz"
RAW_ARCHIVE_FILENAME: Final = "deformmaster-surface-rollout.npz"
RUNTIME_FILENAME: Final = "deformmaster-runtime.json"
CONFIGURATION_FILENAME: Final = "deformmaster-config.yaml"
TRAINING_MANIFEST_FILENAME: Final = "deformmaster-training-data.json"
ARTIFACT_FILENAME: Final = "deformmaster-backend.json"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"

DEFORMMASTER_CLAIM_BOUNDARY: Final = (
    "A custody-checked adapter for a producer-attested, prefix-causal "
    "DeformMaster surface rollout. It does not certify the public all-frame "
    "dataset loader, reproduce a published benchmark, establish target "
    "transfer, calibrate a predictive distribution, or support a "
    "state-of-the-art claim."
)

_FILE_IDENTITY_FIELDS: Final = frozenset({"sha256", "byte_count"})
_LOCAL_FILE_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})
_CHECKPOINT_RECORD_FIELDS: Final = frozenset(
    {"sha256", "byte_count", "external_verified_at_materialization"}
)
_TRAINING_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "source_repository",
        "source_revision",
        "checkpoint_sha256",
        "training_object_ids",
        "manifest_id",
    }
)
_RUNTIME_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_kind",
        "source_repository",
        "source_revision",
        "producer_repository",
        "producer_revision",
        "case_id",
        "target_object_id",
        "training_object_ids",
        "target_object_excluded",
        "checkpoint",
        "configuration",
        "training_manifest",
        "coordinate_frame",
        "position_units",
        "time_units",
        "time_step_s",
        "frame_count",
        "particle_count",
        "observation_frame_range_half_open",
        "forecast_frame_range_half_open",
        "router_input_frame_range_half_open",
        "initialization_input_frame_range_half_open",
        "offset_input_frame_range_half_open",
        "controller_action_frame_range_half_open",
        "material_identity_preserved",
        "information_boundary",
        "raw_rollout_sha256",
        "runtime_id",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "future_object_tracks_read",
        "future_rgb_read",
        "future_depth_read",
        "future_outcomes_read",
        "known_future_controller_action_used",
        "checkpoint_frozen_before_target",
        "prediction_hashed_before_future_scoring",
    }
)
_EXPECTED_BOUNDARY: Final = {
    "future_object_tracks_read": False,
    "future_rgb_read": False,
    "future_depth_read": False,
    "future_outcomes_read": False,
    "known_future_controller_action_used": True,
    "checkpoint_frozen_before_target": True,
    "prediction_hashed_before_future_scoring": True,
}
_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "backend_kind",
        "runtime_id",
        "inputs",
        "output",
        "mapping",
        "information_boundary",
        "claim_boundary",
        "artifact_id",
    }
)
_ARTIFACT_INPUT_FIELDS: Final = frozenset(
    {
        "raw_rollout",
        "runtime_manifest",
        "configuration",
        "training_manifest",
        "checkpoint",
    }
)
_MAPPING_FIELDS: Final = frozenset(
    {
        "frame_count",
        "particle_count",
        "material_identity_preserved",
        "coordinate_frame",
        "position_units",
        "observation_frame_range_half_open",
        "forecast_frame_range_half_open",
    }
)

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _half_open_range(value: object, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain two integer frame indices")
    start, stop = cast(int, value[0]), cast(int, value[1])
    if not 0 <= start < stop:
        raise ValueError(f"{name} must be a nonempty half-open range")
    return start, stop


def _file_identity(path: str | Path, *, name: str) -> dict[str, object]:
    source = _ordinary_file(path, name=name)
    return {"sha256": file_sha256(source), "byte_count": source.stat().st_size}


def _normalize_file_identity(value: object, *, name: str) -> dict[str, object]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_IDENTITY_FIELDS, name=name)
    digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    byte_count = _positive_integer(record.get("byte_count"), name=f"{name}.byte_count")
    return {"sha256": digest, "byte_count": byte_count}


def _verify_file_identity(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    source = _ordinary_file(path, name=name)
    _require(
        source.stat().st_size == expected["byte_count"], f"{name} byte count changed"
    )
    _require(file_sha256(source) == expected["sha256"], f"{name} SHA-256 changed")
    return source


def validate_deformmaster_training_manifest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate target-exclusion provenance for one checkpoint."""

    require_exact_fields(value, expected=_TRAINING_FIELDS, name="training manifest")
    _require(
        value.get("schema") == DEFORMMASTER_TRAINING_SCHEMA, "training schema changed"
    )
    _require(
        value.get("schema_version") == DEFORMMASTER_SCHEMA_VERSION,
        "training schema version changed",
    )
    _require(
        value.get("source_repository") == DEFORMMASTER_SOURCE_REPOSITORY,
        "training source repository changed",
    )
    exact_revision(value.get("source_revision"), name="training source_revision")
    sha256_digest(value.get("checkpoint_sha256"), name="training checkpoint_sha256")
    raw_ids = value.get("training_object_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("training_object_ids must be a JSON array")
    object_ids = canonical_sorted_strings(raw_ids, name="training_object_ids")
    _require(
        list(object_ids) == raw_ids, "training_object_ids must be sorted and unique"
    )
    identity = {key: item for key, item in value.items() if key != "manifest_id"}
    _require(
        value.get("manifest_id") == content_id(identity),
        "training manifest identity changed",
    )
    return cast(dict[str, Any], plain_json(value))


def load_deformmaster_surface_rollout(
    path: str | Path,
) -> tuple[Path, dict[str, npt.NDArray[Any]]]:
    """Load one fixed-material-identity DeformMaster surface rollout."""

    source = _ordinary_file(path, name="raw rollout")
    try:
        with np.load(source, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(DEFORMMASTER_RAW_ARRAY_NAMES),
                "raw DeformMaster array roster changed",
            )
            arrays = {
                name: np.ascontiguousarray(np.asarray(stored[name])).copy()
                for name in stored.files
            }
    except (OSError, ValueError) as error:
        raise ValueError("cannot load raw DeformMaster rollout") from error

    driven = arrays["driven_surface_positions_m"]
    zero = arrays["zero_action_surface_positions_m"]
    support = arrays["action_support"]
    frame_zero = arrays["frame_zero_points_m"]
    _require(
        driven.ndim == 3
        and driven.shape[0] >= 2
        and driven.shape[1] >= 1
        and driven.shape[2] == 3,
        "DeformMaster surface positions must have shape (T,N,3)",
    )
    _require(zero.shape == driven.shape, "driven and zero-action shapes differ")
    _require(frame_zero.shape == driven.shape[1:], "frame-zero shape changed")
    _require(support.shape == (driven.shape[1],), "action_support shape changed")
    _require(
        all(np.issubdtype(array.dtype, np.floating) for array in arrays.values()),
        "DeformMaster rollout arrays must be floating point",
    )
    _require(
        len({array.dtype.str for array in arrays.values()}) == 1,
        "DeformMaster rollout dtypes differ",
    )
    _require(
        all(np.all(np.isfinite(array)) for array in arrays.values()),
        "DeformMaster rollout contains non-finite values",
    )
    _require(
        np.array_equal(driven[0], frame_zero) and np.array_equal(zero[0], frame_zero),
        "DeformMaster rollout changed frame-zero material identity",
    )
    _require(
        np.all((support >= 0.0) & (support <= 1.0)),
        "action_support is outside [0,1]",
    )
    return source, arrays


def validate_deformmaster_runtime_manifest(
    value: Mapping[str, Any],
    *,
    raw_rollout_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    configuration_path: str | Path | None = None,
    training_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the explicit causal and training boundary of one rollout."""

    require_exact_fields(value, expected=_RUNTIME_FIELDS, name="runtime manifest")
    _require(
        value.get("schema") == DEFORMMASTER_RUNTIME_SCHEMA, "runtime schema changed"
    )
    _require(
        value.get("schema_version") == DEFORMMASTER_SCHEMA_VERSION,
        "runtime schema version changed",
    )
    _require(
        value.get("backend_kind") == DEFORMMASTER_BACKEND_KIND, "backend kind changed"
    )
    _require(
        value.get("source_repository") == DEFORMMASTER_SOURCE_REPOSITORY,
        "source repository changed",
    )
    source_revision = exact_revision(
        value.get("source_revision"), name="source_revision"
    )
    repository_name(value.get("producer_repository"), name="producer_repository")
    exact_revision(value.get("producer_revision"), name="producer_revision")
    nonempty_string(value.get("case_id"), name="case_id")
    target_object = nonempty_string(
        value.get("target_object_id"), name="target_object_id"
    )

    raw_ids = value.get("training_object_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("training_object_ids must be a JSON array")
    training_ids = canonical_sorted_strings(raw_ids, name="training_object_ids")
    _require(
        list(training_ids) == raw_ids, "training_object_ids must be sorted and unique"
    )
    _require(
        value.get("target_object_excluded") is True,
        "target object exclusion is not attested",
    )
    _require(
        target_object not in training_ids,
        "target object occurs in checkpoint training data",
    )

    checkpoint = _normalize_file_identity(value.get("checkpoint"), name="checkpoint")
    configuration = _normalize_file_identity(
        value.get("configuration"), name="configuration"
    )
    training_identity = _normalize_file_identity(
        value.get("training_manifest"), name="training_manifest"
    )
    if checkpoint_path is not None:
        _verify_file_identity(checkpoint_path, checkpoint, name="checkpoint")
    if configuration_path is not None:
        _verify_file_identity(configuration_path, configuration, name="configuration")
    if training_manifest_path is not None:
        training_source = _verify_file_identity(
            training_manifest_path,
            training_identity,
            name="training manifest",
        )
        training = validate_deformmaster_training_manifest(
            load_strict_json_object(
                training_source, label="DeformMaster training manifest"
            )
        )
        _require(
            training["source_revision"] == source_revision,
            "training source revision changed",
        )
        _require(
            training["checkpoint_sha256"] == checkpoint["sha256"],
            "training checkpoint changed",
        )
        _require(
            training["training_object_ids"] == list(training_ids),
            "training object roster changed",
        )

    _require(
        value.get("coordinate_frame") == "right-handed-z-up-world-v1",
        "coordinate frame changed",
    )
    _require(value.get("position_units") == "m", "position units must be metres")
    _require(value.get("time_units") == "s", "time units must be seconds")
    _finite_positive(value.get("time_step_s"), name="time_step_s")
    frame_count = _positive_integer(value.get("frame_count"), name="frame_count")
    _require(frame_count >= 2, "frame_count must be at least two")
    _positive_integer(value.get("particle_count"), name="particle_count")

    observation = _half_open_range(
        value.get("observation_frame_range_half_open"),
        name="observation_frame_range_half_open",
    )
    forecast = _half_open_range(
        value.get("forecast_frame_range_half_open"),
        name="forecast_frame_range_half_open",
    )
    _require(observation[0] == 0, "observation range must start at frame zero")
    _require(
        forecast == (observation[1], frame_count),
        "forecast range must begin after the prefix and cover the remaining rollout",
    )
    for field in (
        "router_input_frame_range_half_open",
        "initialization_input_frame_range_half_open",
        "offset_input_frame_range_half_open",
    ):
        admitted = _half_open_range(value.get(field), name=field)
        _require(
            observation[0] <= admitted[0] < admitted[1] <= observation[1],
            f"{field} crosses the observation boundary",
        )
    controller = _half_open_range(
        value.get("controller_action_frame_range_half_open"),
        name="controller_action_frame_range_half_open",
    )
    _require(
        controller == (0, frame_count), "controller action range must cover the rollout"
    )
    _require(
        value.get("material_identity_preserved") is True,
        "material identity is not preserved",
    )

    boundary = _mapping(value.get("information_boundary"), name="information_boundary")
    require_exact_fields(
        boundary, expected=_BOUNDARY_FIELDS, name="information_boundary"
    )
    _require(
        dict(boundary) == _EXPECTED_BOUNDARY, "runtime information boundary changed"
    )
    raw_digest = sha256_digest(
        value.get("raw_rollout_sha256"), name="raw_rollout_sha256"
    )
    identity = {key: item for key, item in value.items() if key != "runtime_id"}
    _require(
        value.get("runtime_id") == content_id(identity), "runtime identity changed"
    )

    if raw_rollout_path is not None:
        raw_source, arrays = load_deformmaster_surface_rollout(raw_rollout_path)
        _require(file_sha256(raw_source) == raw_digest, "raw rollout SHA-256 changed")
        driven = arrays["driven_surface_positions_m"]
        _require(value["frame_count"] == driven.shape[0], "runtime frame count differs")
        _require(
            value["particle_count"] == driven.shape[1], "runtime particle count differs"
        )
    return cast(dict[str, Any], plain_json(value))


def seal_deformmaster_runtime_manifest(
    *,
    raw_rollout_path: str | Path,
    checkpoint_path: str | Path,
    configuration_path: str | Path,
    training_manifest_path: str | Path,
    output_path: str | Path,
    source_revision: str,
    producer_repository: str,
    producer_revision: str,
    case_id: str,
    target_object_id: str,
    prefix_end_frame_exclusive: int,
    time_step_s: float,
) -> dict[str, Any]:
    """Seal one producer attestation before future scoring."""

    raw_source, arrays = load_deformmaster_surface_rollout(raw_rollout_path)
    checkpoint_source = _ordinary_file(checkpoint_path, name="checkpoint")
    configuration_source = _ordinary_file(configuration_path, name="configuration")
    training_source = _ordinary_file(training_manifest_path, name="training manifest")
    training = validate_deformmaster_training_manifest(
        load_strict_json_object(training_source, label="DeformMaster training manifest")
    )
    normalized_source_revision = exact_revision(source_revision, name="source_revision")
    _require(
        training["source_revision"] == normalized_source_revision,
        "training source revision changed",
    )
    checkpoint = _file_identity(checkpoint_source, name="checkpoint")
    _require(
        training["checkpoint_sha256"] == checkpoint["sha256"],
        "training checkpoint changed",
    )
    training_ids = cast(list[str], training["training_object_ids"])
    target = nonempty_string(target_object_id, name="target_object_id")
    _require(
        target not in training_ids, "target object occurs in checkpoint training data"
    )
    frame_count, particle_count = arrays["driven_surface_positions_m"].shape[:2]
    if (
        isinstance(prefix_end_frame_exclusive, bool)
        or not isinstance(prefix_end_frame_exclusive, int)
        or not 1 <= prefix_end_frame_exclusive < frame_count
    ):
        raise ValueError("prefix_end_frame_exclusive must leave a nonempty future")
    identity = {
        "schema": DEFORMMASTER_RUNTIME_SCHEMA,
        "schema_version": DEFORMMASTER_SCHEMA_VERSION,
        "backend_kind": DEFORMMASTER_BACKEND_KIND,
        "source_repository": DEFORMMASTER_SOURCE_REPOSITORY,
        "source_revision": normalized_source_revision,
        "producer_repository": repository_name(
            producer_repository, name="producer_repository"
        ),
        "producer_revision": exact_revision(
            producer_revision, name="producer_revision"
        ),
        "case_id": nonempty_string(case_id, name="case_id"),
        "target_object_id": target,
        "training_object_ids": training_ids,
        "target_object_excluded": True,
        "checkpoint": checkpoint,
        "configuration": _file_identity(configuration_source, name="configuration"),
        "training_manifest": _file_identity(training_source, name="training manifest"),
        "coordinate_frame": "right-handed-z-up-world-v1",
        "position_units": "m",
        "time_units": "s",
        "time_step_s": _finite_positive(time_step_s, name="time_step_s"),
        "frame_count": int(frame_count),
        "particle_count": int(particle_count),
        "observation_frame_range_half_open": [0, prefix_end_frame_exclusive],
        "forecast_frame_range_half_open": [
            prefix_end_frame_exclusive,
            int(frame_count),
        ],
        "router_input_frame_range_half_open": [0, prefix_end_frame_exclusive],
        "initialization_input_frame_range_half_open": [0, 1],
        "offset_input_frame_range_half_open": [0, 1],
        "controller_action_frame_range_half_open": [0, int(frame_count)],
        "material_identity_preserved": True,
        "information_boundary": dict(_EXPECTED_BOUNDARY),
        "raw_rollout_sha256": file_sha256(raw_source),
    }
    runtime = {**identity, "runtime_id": content_id(identity)}
    target_path = Path(output_path).absolute()
    _require(not target_path.exists(), "runtime output already exists")
    write_atomic_json(runtime, target_path, overwrite=False)
    return validate_deformmaster_runtime_manifest(
        runtime,
        raw_rollout_path=raw_source,
        checkpoint_path=checkpoint_source,
        configuration_path=configuration_source,
        training_manifest_path=training_source,
    )


def physical_rollout_from_deformmaster(
    arrays: Mapping[str, npt.NDArray[Any]],
) -> dict[str, FloatArray]:
    """Map a fixed-identity surface rollout into the portable contract."""

    driven = np.ascontiguousarray(arrays["driven_surface_positions_m"])
    zero = np.ascontiguousarray(arrays["zero_action_surface_positions_m"])
    support = np.ascontiguousarray(arrays["action_support"])
    frame_zero = np.ascontiguousarray(arrays["frame_zero_points_m"])
    persistence = np.ascontiguousarray(
        np.repeat(frame_zero[None], driven.shape[0], axis=0)
    )
    return validate_physical_rollout_arrays(
        {
            "prediction_m": driven.copy(),
            "persistence_m": persistence,
            "driven_readout_m": driven.copy(),
            "zero_action_readout_m": zero.copy(),
            "action_support": support.copy(),
            "frame_zero_points_m": frame_zero.copy(),
        }
    )


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _validate_local_file_record(
    value: object,
    *,
    root: Path,
    expected_path: str,
    name: str,
) -> Path:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_LOCAL_FILE_FIELDS, name=name)
    _require(record.get("path") == expected_path, f"{name} path changed")
    identity = _normalize_file_identity(
        {"sha256": record.get("sha256"), "byte_count": record.get("byte_count")},
        name=name,
    )
    return _verify_file_identity(root / expected_path, identity, name=name)


def materialize_deformmaster_backend(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    checkpoint_path: str | Path,
    configuration_path: str | Path,
    training_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish one causal DeformMaster candidate bundle."""

    raw_source, arrays = load_deformmaster_surface_rollout(raw_rollout_path)
    runtime_source = _ordinary_file(runtime_manifest_path, name="runtime manifest")
    checkpoint_source = _ordinary_file(checkpoint_path, name="checkpoint")
    configuration_source = _ordinary_file(configuration_path, name="configuration")
    training_source = _ordinary_file(training_manifest_path, name="training manifest")
    runtime = validate_deformmaster_runtime_manifest(
        load_strict_json_object(runtime_source, label="DeformMaster runtime manifest"),
        raw_rollout_path=raw_source,
        checkpoint_path=checkpoint_source,
        configuration_path=configuration_source,
        training_manifest_path=training_source,
    )
    physical = physical_rollout_from_deformmaster(arrays)

    output = Path(output_dir).absolute()
    _require(not output.exists(), "output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        provenance = staging / "provenance"
        provenance.mkdir()
        raw_target = provenance / RAW_ARCHIVE_FILENAME
        runtime_target = provenance / RUNTIME_FILENAME
        configuration_target = provenance / CONFIGURATION_FILENAME
        training_target = provenance / TRAINING_MANIFEST_FILENAME
        shutil.copyfile(raw_source, raw_target)
        shutil.copyfile(runtime_source, runtime_target)
        shutil.copyfile(configuration_source, configuration_target)
        shutil.copyfile(training_source, training_target)
        physical_target = staging / PHYSICAL_ARCHIVE_FILENAME
        write_deterministic_npz(physical_target, physical)

        checkpoint_record = {
            **cast(dict[str, object], runtime["checkpoint"]),
            "external_verified_at_materialization": True,
        }
        inputs = {
            "raw_rollout": _file_record(raw_target, relative_to=staging),
            "runtime_manifest": _file_record(runtime_target, relative_to=staging),
            "configuration": _file_record(configuration_target, relative_to=staging),
            "training_manifest": _file_record(training_target, relative_to=staging),
            "checkpoint": checkpoint_record,
        }
        mapping = {
            "frame_count": runtime["frame_count"],
            "particle_count": runtime["particle_count"],
            "material_identity_preserved": True,
            "coordinate_frame": runtime["coordinate_frame"],
            "position_units": "m",
            "observation_frame_range_half_open": runtime[
                "observation_frame_range_half_open"
            ],
            "forecast_frame_range_half_open": runtime["forecast_frame_range_half_open"],
        }
        identity = {
            "schema": DEFORMMASTER_ARTIFACT_SCHEMA,
            "schema_version": DEFORMMASTER_SCHEMA_VERSION,
            "backend_kind": DEFORMMASTER_BACKEND_KIND,
            "runtime_id": runtime["runtime_id"],
            "inputs": inputs,
            "output": _file_record(physical_target, relative_to=staging),
            "mapping": mapping,
            "information_boundary": runtime["information_boundary"],
            "claim_boundary": DEFORMMASTER_CLAIM_BOUNDARY,
        }
        artifact = {**identity, "artifact_id": content_id(identity)}
        artifact_path = staging / ARTIFACT_FILENAME
        write_atomic_json(artifact, artifact_path, overwrite=False)
        checksum_paths = [
            artifact_path,
            physical_target,
            raw_target,
            runtime_target,
            configuration_target,
            training_target,
        ]
        checksums = "".join(
            f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}\n"
            for path in sorted(checksum_paths, key=lambda item: item.as_posix())
        )
        (staging / CHECKSUMS_FILENAME).write_text(checksums, encoding="ascii")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_deformmaster_backend(output)


def validate_deformmaster_backend(output_dir: str | Path) -> dict[str, Any]:
    """Validate bundle custody and rederive the portable rollout exactly."""

    requested = Path(output_dir).absolute()
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        "backend bundle is not an ordinary non-symlink directory",
    )
    root = requested.resolve(strict=True)
    expected_roster = {
        ARTIFACT_FILENAME,
        CHECKSUMS_FILENAME,
        PHYSICAL_ARCHIVE_FILENAME,
        f"provenance/{RAW_ARCHIVE_FILENAME}",
        f"provenance/{RUNTIME_FILENAME}",
        f"provenance/{CONFIGURATION_FILENAME}",
        f"provenance/{TRAINING_MANIFEST_FILENAME}",
    }
    actual_roster = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(actual_roster == expected_roster, "backend bundle file roster changed")
    artifact = load_strict_json_object(
        root / ARTIFACT_FILENAME, label="DeformMaster artifact"
    )
    require_exact_fields(artifact, expected=_ARTIFACT_FIELDS, name="artifact")
    _require(
        artifact.get("schema") == DEFORMMASTER_ARTIFACT_SCHEMA,
        "artifact schema changed",
    )
    _require(
        artifact.get("schema_version") == DEFORMMASTER_SCHEMA_VERSION,
        "artifact schema version changed",
    )
    _require(
        artifact.get("backend_kind") == DEFORMMASTER_BACKEND_KIND,
        "artifact backend changed",
    )
    inputs = _mapping(artifact.get("inputs"), name="inputs")
    require_exact_fields(inputs, expected=_ARTIFACT_INPUT_FIELDS, name="inputs")
    raw_path = _validate_local_file_record(
        inputs.get("raw_rollout"),
        root=root,
        expected_path=f"provenance/{RAW_ARCHIVE_FILENAME}",
        name="raw_rollout",
    )
    runtime_path = _validate_local_file_record(
        inputs.get("runtime_manifest"),
        root=root,
        expected_path=f"provenance/{RUNTIME_FILENAME}",
        name="runtime_manifest",
    )
    configuration_path = _validate_local_file_record(
        inputs.get("configuration"),
        root=root,
        expected_path=f"provenance/{CONFIGURATION_FILENAME}",
        name="configuration",
    )
    training_path = _validate_local_file_record(
        inputs.get("training_manifest"),
        root=root,
        expected_path=f"provenance/{TRAINING_MANIFEST_FILENAME}",
        name="training_manifest",
    )
    checkpoint = _mapping(inputs.get("checkpoint"), name="checkpoint")
    require_exact_fields(
        checkpoint, expected=_CHECKPOINT_RECORD_FIELDS, name="checkpoint"
    )
    checkpoint_identity = _normalize_file_identity(
        {
            "sha256": checkpoint.get("sha256"),
            "byte_count": checkpoint.get("byte_count"),
        },
        name="checkpoint",
    )
    _require(
        checkpoint.get("external_verified_at_materialization") is True,
        "checkpoint was not verified at materialization",
    )
    runtime = validate_deformmaster_runtime_manifest(
        load_strict_json_object(runtime_path, label="DeformMaster runtime manifest"),
        raw_rollout_path=raw_path,
        configuration_path=configuration_path,
        training_manifest_path=training_path,
    )
    _require(
        runtime["checkpoint"] == checkpoint_identity, "checkpoint identity changed"
    )
    _require(artifact.get("runtime_id") == runtime["runtime_id"], "runtime ID changed")

    output_path = _validate_local_file_record(
        artifact.get("output"),
        root=root,
        expected_path=PHYSICAL_ARCHIVE_FILENAME,
        name="output",
    )
    _, raw = load_deformmaster_surface_rollout(raw_path)
    expected_physical = physical_rollout_from_deformmaster(raw)
    actual_physical = load_physical_rollout_archive(
        output_path, expected_frame_count=runtime["frame_count"]
    )
    for name, expected in expected_physical.items():
        _require(np.array_equal(actual_physical[name], expected), f"{name} changed")

    mapping = _mapping(artifact.get("mapping"), name="mapping")
    require_exact_fields(mapping, expected=_MAPPING_FIELDS, name="mapping")
    expected_mapping = {
        "frame_count": runtime["frame_count"],
        "particle_count": runtime["particle_count"],
        "material_identity_preserved": True,
        "coordinate_frame": runtime["coordinate_frame"],
        "position_units": "m",
        "observation_frame_range_half_open": runtime[
            "observation_frame_range_half_open"
        ],
        "forecast_frame_range_half_open": runtime["forecast_frame_range_half_open"],
    }
    _require(dict(mapping) == expected_mapping, "artifact mapping changed")
    _require(
        artifact.get("information_boundary") == runtime["information_boundary"],
        "artifact information boundary changed",
    )
    _require(
        artifact.get("claim_boundary") == DEFORMMASTER_CLAIM_BOUNDARY,
        "artifact claim boundary changed",
    )
    identity = {key: item for key, item in artifact.items() if key != "artifact_id"}
    _require(
        artifact.get("artifact_id") == content_id(identity), "artifact identity changed"
    )

    expected_checksums = "".join(
        f"{file_sha256(root / path)}  {path}\n"
        for path in sorted(expected_roster - {CHECKSUMS_FILENAME})
    )
    actual_checksums = (root / CHECKSUMS_FILENAME).read_text(encoding="ascii")
    _require(actual_checksums == expected_checksums, "bundle checksums changed")
    return cast(dict[str, Any], plain_json(artifact))
