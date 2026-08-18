"""First-class producer boundary for the official MatPhys-to-Warp path.

The official MatPhys recipe predicts spring and contact parameters, while the
official PhysTwin Warp simulator produces trajectories.  This module turns a
strict fixed-identity replay export into the two physical archives required by
the guarded MatPhys backend.  It also keeps published per-case fitting separate
from causal, target-excluded transfer.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .material_trajectory_backend_v1 import array_sha256, file_sha256
from .matphys_backend_v1 import (
    MATPHYS_SOURCE_REPOSITORY,
    PHYSTWIN_SOURCE_REPOSITORY,
    build_matphys_backend_proposal,
    validate_matphys_backend_proposal,
)
from .physical_rollout_v1 import (
    load_physical_rollout_archive,
    validate_physical_rollout_arrays,
    write_deterministic_npz,
)

MATPHYS_OFFICIAL_PRODUCER_SCHEMA: Final = "bayesian-phystwin.matphys-official-producer"
MATPHYS_OFFICIAL_PRODUCER_VERSION: Final = 1
MATPHYS_OFFICIAL_PRODUCER_PROTOCOL: Final = "official-matphys-fixed-identity-export-v1"
MATPHYS_OFFICIAL_BACKEND_KIND: Final = "official-matphys-full-pipeline-phystwin-warp-v1"
MATPHYS_OFFICIAL_PARAMETERIZATION: Final = (
    "per-edge-spring-contact-and-damping-parameters-v1"
)
MATPHYS_OFFICIAL_ROLLOUT_BACKEND: Final = "official-phystwin-warp"
MATPHYS_PUBLISHED_PARITY_MODE: Final = "published-per-case-parity-v1"
MATPHYS_CAUSAL_PREFIX_MODE: Final = "causal-prefix-transfer-v1"
MATPHYS_OFFICIAL_MODES: Final = frozenset(
    {MATPHYS_PUBLISHED_PARITY_MODE, MATPHYS_CAUSAL_PREFIX_MODE}
)
MATPHYS_OFFICIAL_PIPELINE_COMPONENTS: Final = (
    "collision-and-damping-parameters",
    "dino-feature-lift",
    "gpt-physics-prior",
    "material-distribution",
    "part-segmentation",
    "per-edge-spring-field",
    "video-material-encoder",
)

REPLAY_INPUT_FILENAME: Final = "matphys-official-replay-input.npz"
ARTIFACT_FILENAME: Final = "matphys-official-producer.json"
CANDIDATE_ARCHIVE_FILENAME: Final = "matphys-candidate-physical.npz"
IDENTITY_ARCHIVE_FILENAME: Final = "matphys-identity-replay-physical.npz"
CAUSAL_PROPOSAL_FILENAME: Final = "matphys-proposal.json"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"

MATPHYS_OFFICIAL_CLAIM_BOUNDARY: Final = (
    "This bundle proves custody from an official MatPhys parameter export to "
    "fixed-identity PhysTwin/Warp replay arrays. Published per-case parity mode "
    "is a benchmark control and is not eligible for causal or deployment "
    "claims. Causal-prefix mode requires target-excluded checkpoint training "
    "and still needs the independent guarded prefix gate. Neither mode alone "
    "establishes accuracy, calibration, transfer, safety, or state of the art."
)

_REPLAY_ARRAY_NAMES: Final = frozenset(
    {
        "candidate_driven_state_m",
        "candidate_zero_action_state_m",
        "identity_driven_state_m",
        "identity_zero_action_state_m",
        "material_query_indices",
        "action_support",
        "frame_indices",
    }
)
_STATE_ARRAY_NAMES: Final = (
    "candidate_driven_state_m",
    "candidate_zero_action_state_m",
    "identity_driven_state_m",
    "identity_zero_action_state_m",
)
_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "producer_protocol",
        "mode",
        "backend_kind",
        "parameterization",
        "rollout_backend",
        "coordinate_frame",
        "position_unit",
        "source_repository",
        "source_revision",
        "simulator_repository",
        "simulator_revision",
        "case_id",
        "target_object_id",
        "checkpoint_training_object_ids",
        "target_fit_frame_range_half_open",
        "future_frame_start",
        "proposal_strength",
        "pipeline_components",
        "pipeline_component_artifacts",
        "checkpoint",
        "spring_field",
        "candidate_parameters",
        "identity_parameters",
        "replay_input",
        "replay_summary",
        "source_artifacts",
        "outputs",
        "information_boundary",
        "claim_boundary",
        "artifact_id",
    }
)
_FILE_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})
_SPRING_FIELD_FIELDS: Final = frozenset({"path", "sha256", "byte_count", "count"})
_REPLAY_SUMMARY_FIELDS: Final = frozenset(
    {
        "frame_count",
        "state_count",
        "query_count",
        "dtype",
        "frame_indices",
        "frame_indices_sha256",
        "material_query_indices_sha256",
    }
)
_OUTPUT_FIELDS: Final = frozenset(
    {"candidate_archive", "identity_replay_archive", "causal_proposal"}
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "target_prefix_used_for_parameter_fit",
        "target_object_used_for_checkpoint_training",
        "target_future_observations_used",
        "future_outcomes_opened",
        "known_future_robot_action_used",
        "causal_backend_eligible",
        "published_benchmark_control_only",
    }
)

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntegerArray: TypeAlias = npt.NDArray[np.integer[Any]]
MatPhysOfficialMode: TypeAlias = Literal[
    "published-per-case-parity-v1", "causal-prefix-transfer-v1"
]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[object], value)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_strength(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("proposal_strength must be a finite number in (0,1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("proposal_strength must be a finite number in (0,1]")
    return result


def _frame_range(value: object, *, name: str) -> tuple[int, int]:
    raw = _sequence(value, name=name)
    if len(raw) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw
    ):
        raise ValueError(f"{name} must contain exactly two integer indices")
    start, stop = cast(int, raw[0]), cast(int, raw[1])
    if not 0 <= start < stop:
        raise ValueError(f"{name} must be a nonempty nonnegative half-open range")
    return start, stop


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def _file_record(path: str | Path, *, name: str) -> dict[str, object]:
    source = _ordinary_file(path, name=name)
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "byte_count": source.stat().st_size,
    }


def _normalize_file_record(
    value: object,
    *,
    name: str,
    verify_file: bool,
) -> dict[str, object]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    path = nonempty_string(record.get("path"), name=f"{name}.path")
    digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    byte_count = _positive_integer(record.get("byte_count"), name=f"{name}.byte_count")
    if verify_file:
        source = _ordinary_file(path, name=name)
        _require(file_sha256(source) == digest, f"{name} SHA-256 changed")
        _require(source.stat().st_size == byte_count, f"{name} byte count changed")
        path = str(source)
    return {"path": path, "sha256": digest, "byte_count": byte_count}


def _normalized_object_ids(value: object) -> tuple[str, ...]:
    raw = _sequence(value, name="checkpoint_training_object_ids")
    values = tuple(
        nonempty_string(item, name="checkpoint_training_object_ids entry")
        for item in raw
    )
    _require(bool(values), "checkpoint_training_object_ids must not be empty")
    _require(
        values == tuple(sorted(set(values))),
        "checkpoint_training_object_ids must be sorted and unique",
    )
    return values


def _component_artifacts(value: object) -> dict[str, str]:
    record = _mapping(value, name="pipeline_component_artifacts")
    _require(
        set(record) == set(MATPHYS_OFFICIAL_PIPELINE_COMPONENTS),
        "official MatPhys pipeline component roster changed",
    )
    return {
        name: sha256_digest(record.get(name), name=f"component {name}")
        for name in MATPHYS_OFFICIAL_PIPELINE_COMPONENTS
    }


def validate_matphys_official_replay_arrays(
    values: Mapping[str, npt.NDArray[Any]],
) -> dict[str, npt.NDArray[Any]]:
    """Validate one official MatPhys replay export without coercing identities."""

    _require(
        set(values) == set(_REPLAY_ARRAY_NAMES),
        "official MatPhys replay array roster changed",
    )
    arrays = {
        name: np.ascontiguousarray(np.asarray(values[name])).copy()
        for name in _REPLAY_ARRAY_NAMES
    }
    states = [arrays[name] for name in _STATE_ARRAY_NAMES]
    reference = states[0]
    _require(
        reference.ndim == 3
        and reference.shape[0] >= 2
        and reference.shape[1] >= 1
        and reference.shape[2] == 3,
        "official MatPhys states must have shape (T,S,3)",
    )
    _require(
        all(value.shape == reference.shape for value in states),
        "official MatPhys state shapes differ",
    )
    _require(
        all(np.issubdtype(value.dtype, np.floating) for value in states),
        "official MatPhys states must be floating point",
    )
    _require(
        len({value.dtype.str for value in states}) == 1,
        "official MatPhys state dtypes differ",
    )
    _require(
        all(np.all(np.isfinite(value)) for value in states),
        "official MatPhys states contain non-finite values",
    )
    _require(
        all(np.array_equal(reference[0], value[0]) for value in states[1:]),
        "official MatPhys replay arms changed frame-zero state identity",
    )

    frame_indices = arrays["frame_indices"]
    _require(
        frame_indices.ndim == 1
        and frame_indices.shape == (reference.shape[0],)
        and np.issubdtype(frame_indices.dtype, np.integer)
        and not np.issubdtype(frame_indices.dtype, np.bool_),
        "frame_indices must be an integer vector matching T",
    )
    _require(
        np.all(frame_indices >= 0) and np.all(np.diff(frame_indices) > 0),
        "frame_indices must be nonnegative and strictly increasing",
    )

    query_indices = arrays["material_query_indices"]
    _require(
        query_indices.ndim == 1
        and len(query_indices) >= 1
        and np.issubdtype(query_indices.dtype, np.integer)
        and not np.issubdtype(query_indices.dtype, np.bool_),
        "material_query_indices must be a nonempty integer vector",
    )
    _require(
        len(np.unique(query_indices)) == len(query_indices)
        and np.all((query_indices >= 0) & (query_indices < reference.shape[1])),
        "material_query_indices must be unique and in range",
    )

    support = arrays["action_support"]
    _require(
        support.shape == (len(query_indices),)
        and np.issubdtype(support.dtype, np.floating)
        and support.dtype == reference.dtype,
        "action_support must match query count and state dtype",
    )
    _require(
        np.all(np.isfinite(support)) and np.all((support >= 0.0) & (support <= 1.0)),
        "action_support must be finite and in [0,1]",
    )
    return arrays


def write_matphys_official_replay_input(
    path: str | Path,
    values: Mapping[str, npt.NDArray[Any]],
) -> Path:
    """Write a deterministic no-pickle MatPhys replay input archive once."""

    arrays = validate_matphys_official_replay_arrays(values)
    return cast(Path, write_deterministic_npz(path, arrays))


def load_matphys_official_replay_input(
    path: str | Path,
) -> tuple[Path, dict[str, npt.NDArray[Any]]]:
    """Load and validate one no-pickle official MatPhys replay input."""

    source = _ordinary_file(path, name="official MatPhys replay input")
    try:
        with np.load(source, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load official MatPhys replay input") from error
    return source, validate_matphys_official_replay_arrays(arrays)


def _physical_archives(
    replay: Mapping[str, npt.NDArray[Any]],
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    indices = np.asarray(replay["material_query_indices"], dtype=np.int64)
    support = cast(FloatArray, replay["action_support"])

    def build(driven_name: str, zero_name: str) -> dict[str, FloatArray]:
        driven = cast(FloatArray, replay[driven_name][:, indices, :])
        zero = cast(FloatArray, replay[zero_name][:, indices, :])
        frame_zero = np.ascontiguousarray(driven[0]).copy()
        persistence = np.repeat(frame_zero[None], driven.shape[0], axis=0)
        return cast(
            dict[str, FloatArray],
            validate_physical_rollout_arrays(
                {
                    "prediction_m": driven,
                    "persistence_m": persistence,
                    "driven_readout_m": driven,
                    "zero_action_readout_m": zero,
                    "action_support": support,
                    "frame_zero_points_m": frame_zero,
                }
            ),
        )

    return (
        build("candidate_driven_state_m", "candidate_zero_action_state_m"),
        build("identity_driven_state_m", "identity_zero_action_state_m"),
    )


def _replay_summary(
    replay: Mapping[str, npt.NDArray[Any]],
) -> dict[str, object]:
    state = replay["candidate_driven_state_m"]
    return {
        "frame_count": int(state.shape[0]),
        "state_count": int(state.shape[1]),
        "query_count": int(len(replay["material_query_indices"])),
        "dtype": state.dtype.str,
        "frame_indices": [int(value) for value in replay["frame_indices"]],
        "frame_indices_sha256": array_sha256(replay["frame_indices"]),
        "material_query_indices_sha256": array_sha256(replay["material_query_indices"]),
    }


def _output_record(path: Path, *, relative_path: str) -> dict[str, object]:
    return {
        "path": relative_path,
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _write_checksums(root: Path, roster: Sequence[str]) -> None:
    lines = [f"{file_sha256(root / name)}  {name}\n" for name in sorted(roster)]
    target = root / CHECKSUMS_FILENAME
    with target.open("x", encoding="ascii", newline="\n") as stream:
        stream.writelines(lines)
        stream.flush()
        os.fsync(stream.fileno())


def _new_output_directory(path: str | Path) -> tuple[Path, Path]:
    output = Path(path).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not any(parent.is_symlink() for parent in output.parents),
        "output path must not traverse a symlink",
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    return output, staging


def materialize_matphys_official_producer(
    *,
    replay_input_path: str | Path,
    checkpoint_path: str | Path,
    spring_field_path: str | Path,
    candidate_parameter_path: str | Path,
    identity_parameter_path: str | Path,
    output_dir: str | Path,
    mode: MatPhysOfficialMode,
    source_revision: str,
    simulator_revision: str,
    case_id: str,
    target_object_id: str,
    checkpoint_training_object_ids: Sequence[str],
    target_fit_frame_range_half_open: tuple[int, int],
    future_frame_start: int,
    proposal_strength: float,
    pipeline_component_artifacts: Mapping[str, str],
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Publish official MatPhys replay outputs under an explicit claim regime."""

    replay_path, replay = load_matphys_official_replay_input(replay_input_path)
    mode_value = nonempty_string(mode, name="mode")
    _require(mode_value in MATPHYS_OFFICIAL_MODES, "unknown official MatPhys mode")
    source_revision_value = exact_revision(source_revision, name="source_revision")
    simulator_revision_value = exact_revision(
        simulator_revision, name="simulator_revision"
    )
    case = nonempty_string(case_id, name="case_id")
    target = nonempty_string(target_object_id, name="target_object_id")
    training = _normalized_object_ids(checkpoint_training_object_ids)
    if mode_value == MATPHYS_CAUSAL_PREFIX_MODE:
        _require(
            target not in training,
            "causal MatPhys mode requires target-excluded checkpoint training",
        )
    fit_start, fit_stop = _frame_range(
        target_fit_frame_range_half_open,
        name="target_fit_frame_range_half_open",
    )
    future_start = _positive_integer(future_frame_start, name="future_frame_start")
    _require(fit_stop <= future_start, "MatPhys target fitting crosses future boundary")
    if mode_value == MATPHYS_CAUSAL_PREFIX_MODE:
        _require(
            fit_stop < future_start,
            "causal MatPhys mode requires a disjoint validation prefix",
        )
    replay_frames = [int(value) for value in replay["frame_indices"]]
    _require(
        fit_start >= replay_frames[0]
        and fit_start in replay_frames
        and future_start in replay_frames,
        "MatPhys fit and future boundaries are not represented by replay frames",
    )
    strength = _finite_strength(proposal_strength)
    component_artifacts = _component_artifacts(pipeline_component_artifacts)
    source_records = source_artifact_mapping(source_artifacts, name="source_artifacts")
    checkpoint = _file_record(checkpoint_path, name="MatPhys checkpoint")
    candidate_parameters = _file_record(
        candidate_parameter_path, name="MatPhys candidate parameters"
    )
    identity_parameters = _file_record(
        identity_parameter_path, name="MatPhys identity parameters"
    )
    spring_path = _ordinary_file(spring_field_path, name="MatPhys spring field")
    try:
        spring = np.asarray(np.load(spring_path, allow_pickle=False))
    except (OSError, ValueError) as error:
        raise ValueError("cannot load MatPhys spring field") from error
    _require(
        spring.ndim == 1
        and np.issubdtype(spring.dtype, np.floating)
        and len(spring) >= 1
        and np.all(np.isfinite(spring))
        and np.all(spring > 0.0),
        "MatPhys spring field must be a finite positive floating vector",
    )
    spring_record = {
        **_file_record(spring_path, name="MatPhys spring field"),
        "count": int(len(spring)),
    }
    candidate, identity_replay = _physical_archives(replay)
    output, staging = _new_output_directory(output_dir)
    try:
        candidate_path = write_deterministic_npz(
            staging / CANDIDATE_ARCHIVE_FILENAME, candidate
        )
        identity_path = write_deterministic_npz(
            staging / IDENTITY_ARCHIVE_FILENAME, identity_replay
        )
        proposal_record: dict[str, object] | None = None
        if mode_value == MATPHYS_CAUSAL_PREFIX_MODE:
            proposal = build_matphys_backend_proposal(
                source_revision=source_revision_value,
                simulator_revision=simulator_revision_value,
                target_object_id=target,
                training_object_ids=training,
                target_evidence_end_frame_exclusive=fit_stop,
                proposal_strength=strength,
                checkpoint_path=checkpoint_path,
                spring_field_path=spring_field_path,
                source_artifacts=cast(Mapping[str, str], source_records),
            )
            proposal_path = staging / CAUSAL_PROPOSAL_FILENAME
            write_atomic_json(proposal, proposal_path, overwrite=False)
            proposal_record = _output_record(
                proposal_path, relative_path=CAUSAL_PROPOSAL_FILENAME
            )

        boundary = {
            "target_prefix_used_for_parameter_fit": True,
            "target_object_used_for_checkpoint_training": target in training,
            "target_future_observations_used": False,
            "future_outcomes_opened": False,
            "known_future_robot_action_used": True,
            "causal_backend_eligible": mode_value == MATPHYS_CAUSAL_PREFIX_MODE,
            "published_benchmark_control_only": (
                mode_value == MATPHYS_PUBLISHED_PARITY_MODE
            ),
        }
        identity: dict[str, Any] = {
            "schema": MATPHYS_OFFICIAL_PRODUCER_SCHEMA,
            "schema_version": MATPHYS_OFFICIAL_PRODUCER_VERSION,
            "producer_protocol": MATPHYS_OFFICIAL_PRODUCER_PROTOCOL,
            "mode": mode_value,
            "backend_kind": MATPHYS_OFFICIAL_BACKEND_KIND,
            "parameterization": MATPHYS_OFFICIAL_PARAMETERIZATION,
            "rollout_backend": MATPHYS_OFFICIAL_ROLLOUT_BACKEND,
            "coordinate_frame": "right-handed-z-up-world-v1",
            "position_unit": "m",
            "source_repository": MATPHYS_SOURCE_REPOSITORY,
            "source_revision": source_revision_value,
            "simulator_repository": PHYSTWIN_SOURCE_REPOSITORY,
            "simulator_revision": simulator_revision_value,
            "case_id": case,
            "target_object_id": target,
            "checkpoint_training_object_ids": list(training),
            "target_fit_frame_range_half_open": [fit_start, fit_stop],
            "future_frame_start": future_start,
            "proposal_strength": strength,
            "pipeline_components": list(MATPHYS_OFFICIAL_PIPELINE_COMPONENTS),
            "pipeline_component_artifacts": component_artifacts,
            "checkpoint": checkpoint,
            "spring_field": spring_record,
            "candidate_parameters": candidate_parameters,
            "identity_parameters": identity_parameters,
            "replay_input": _file_record(
                replay_path, name="official MatPhys replay input"
            ),
            "replay_summary": _replay_summary(replay),
            "source_artifacts": dict(source_records),
            "outputs": {
                "candidate_archive": _output_record(
                    candidate_path, relative_path=CANDIDATE_ARCHIVE_FILENAME
                ),
                "identity_replay_archive": _output_record(
                    identity_path, relative_path=IDENTITY_ARCHIVE_FILENAME
                ),
                "causal_proposal": proposal_record,
            },
            "information_boundary": boundary,
            "claim_boundary": MATPHYS_OFFICIAL_CLAIM_BOUNDARY,
        }
        artifact = {**identity, "artifact_id": content_id(identity)}
        artifact = validate_matphys_official_producer_record(
            artifact, verify_sources=True
        )
        write_atomic_json(artifact, staging / ARTIFACT_FILENAME, overwrite=False)
        roster = [
            ARTIFACT_FILENAME,
            CANDIDATE_ARCHIVE_FILENAME,
            IDENTITY_ARCHIVE_FILENAME,
        ]
        if proposal_record is not None:
            roster.append(CAUSAL_PROPOSAL_FILENAME)
        _write_checksums(staging, roster)
        validate_matphys_official_producer_artifact(staging, verify_sources=True)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_matphys_official_producer_artifact(output, verify_sources=True)


def validate_matphys_official_producer_record(
    value: object,
    *,
    verify_sources: bool,
) -> dict[str, Any]:
    """Validate the content-addressed producer record itself."""

    artifact = _mapping(value, name="official MatPhys producer artifact")
    require_exact_fields(
        artifact, expected=_ARTIFACT_FIELDS, name="official MatPhys producer artifact"
    )
    _require(
        artifact.get("schema") == MATPHYS_OFFICIAL_PRODUCER_SCHEMA
        and artifact.get("schema_version") == MATPHYS_OFFICIAL_PRODUCER_VERSION,
        "official MatPhys producer schema changed",
    )
    _require(
        artifact.get("producer_protocol") == MATPHYS_OFFICIAL_PRODUCER_PROTOCOL
        and artifact.get("backend_kind") == MATPHYS_OFFICIAL_BACKEND_KIND
        and artifact.get("parameterization") == MATPHYS_OFFICIAL_PARAMETERIZATION
        and artifact.get("rollout_backend") == MATPHYS_OFFICIAL_ROLLOUT_BACKEND,
        "official MatPhys producer semantics changed",
    )
    mode = nonempty_string(artifact.get("mode"), name="mode")
    _require(mode in MATPHYS_OFFICIAL_MODES, "unknown official MatPhys mode")
    target = nonempty_string(artifact.get("target_object_id"), name="target_object_id")
    training = _normalized_object_ids(artifact.get("checkpoint_training_object_ids"))
    if mode == MATPHYS_CAUSAL_PREFIX_MODE:
        _require(
            target not in training,
            "causal MatPhys mode requires target-excluded checkpoint training",
        )
    fit_start, fit_stop = _frame_range(
        artifact.get("target_fit_frame_range_half_open"),
        name="target_fit_frame_range_half_open",
    )
    future_start = _positive_integer(
        artifact.get("future_frame_start"), name="future_frame_start"
    )
    _require(fit_stop <= future_start, "MatPhys target fitting crosses future boundary")
    if mode == MATPHYS_CAUSAL_PREFIX_MODE:
        _require(
            fit_stop < future_start,
            "causal MatPhys mode requires a disjoint validation prefix",
        )
    components = tuple(
        nonempty_string(item, name="pipeline_components entry")
        for item in _sequence(
            artifact.get("pipeline_components"), name="pipeline_components"
        )
    )
    _require(
        components == MATPHYS_OFFICIAL_PIPELINE_COMPONENTS,
        "official MatPhys pipeline components changed",
    )
    component_artifacts = _component_artifacts(
        artifact.get("pipeline_component_artifacts")
    )
    checkpoint = _normalize_file_record(
        artifact.get("checkpoint"),
        name="MatPhys checkpoint",
        verify_file=verify_sources,
    )
    candidate_parameters = _normalize_file_record(
        artifact.get("candidate_parameters"),
        name="MatPhys candidate parameters",
        verify_file=verify_sources,
    )
    identity_parameters = _normalize_file_record(
        artifact.get("identity_parameters"),
        name="MatPhys identity parameters",
        verify_file=verify_sources,
    )
    spring_raw = _mapping(artifact.get("spring_field"), name="spring_field")
    require_exact_fields(spring_raw, expected=_SPRING_FIELD_FIELDS, name="spring_field")
    spring = {
        **_normalize_file_record(
            {
                "path": spring_raw.get("path"),
                "sha256": spring_raw.get("sha256"),
                "byte_count": spring_raw.get("byte_count"),
            },
            name="MatPhys spring field",
            verify_file=verify_sources,
        ),
        "count": _positive_integer(spring_raw.get("count"), name="spring_field.count"),
    }
    if verify_sources:
        values = np.asarray(np.load(str(spring["path"]), allow_pickle=False))
        _require(
            values.ndim == 1
            and len(values) == spring["count"]
            and np.issubdtype(values.dtype, np.floating)
            and np.all(np.isfinite(values))
            and np.all(values > 0.0),
            "MatPhys spring field content changed",
        )
    replay_input = _normalize_file_record(
        artifact.get("replay_input"),
        name="official MatPhys replay input",
        verify_file=verify_sources,
    )
    summary_raw = _mapping(artifact.get("replay_summary"), name="replay_summary")
    require_exact_fields(
        summary_raw, expected=_REPLAY_SUMMARY_FIELDS, name="replay_summary"
    )
    replay_summary = {
        "frame_count": _positive_integer(
            summary_raw.get("frame_count"), name="replay_summary.frame_count"
        ),
        "state_count": _positive_integer(
            summary_raw.get("state_count"), name="replay_summary.state_count"
        ),
        "query_count": _positive_integer(
            summary_raw.get("query_count"), name="replay_summary.query_count"
        ),
        "dtype": nonempty_string(summary_raw.get("dtype"), name="replay_summary.dtype"),
        "frame_indices": [
            int(value)
            for value in _sequence(
                summary_raw.get("frame_indices"), name="replay_summary.frame_indices"
            )
            if not isinstance(value, bool) and isinstance(value, int)
        ],
        "frame_indices_sha256": sha256_digest(
            summary_raw.get("frame_indices_sha256"),
            name="replay_summary.frame_indices_sha256",
        ),
        "material_query_indices_sha256": sha256_digest(
            summary_raw.get("material_query_indices_sha256"),
            name="replay_summary.material_query_indices_sha256",
        ),
    }
    raw_frame_indices = _sequence(
        summary_raw.get("frame_indices"), name="replay_summary.frame_indices"
    )
    _require(
        len(replay_summary["frame_indices"]) == len(raw_frame_indices)
        and len(raw_frame_indices) == replay_summary["frame_count"],
        "replay_summary.frame_indices must contain one integer per frame",
    )
    frame_indices = cast(list[int], replay_summary["frame_indices"])
    _require(
        bool(frame_indices)
        and frame_indices[0] >= 0
        and all(
            right > left
            for left, right in zip(frame_indices, frame_indices[1:], strict=False)
        ),
        "replay_summary.frame_indices must be nonnegative and strictly increasing",
    )
    _require(
        fit_start >= frame_indices[0]
        and fit_start in frame_indices
        and future_start in frame_indices,
        "MatPhys fit and future boundaries are not represented by replay frames",
    )
    artifacts = source_artifact_mapping(
        _mapping(artifact.get("source_artifacts"), name="source_artifacts"),
        name="source_artifacts",
    )
    outputs_raw = _mapping(artifact.get("outputs"), name="outputs")
    require_exact_fields(outputs_raw, expected=_OUTPUT_FIELDS, name="outputs")
    outputs: dict[str, object] = {
        "candidate_archive": _normalize_file_record(
            outputs_raw.get("candidate_archive"),
            name="candidate archive",
            verify_file=False,
        ),
        "identity_replay_archive": _normalize_file_record(
            outputs_raw.get("identity_replay_archive"),
            name="identity replay archive",
            verify_file=False,
        ),
    }
    proposal_value = outputs_raw.get("causal_proposal")
    if proposal_value is None:
        _require(
            mode == MATPHYS_PUBLISHED_PARITY_MODE,
            "causal MatPhys mode is missing its guarded proposal",
        )
        outputs["causal_proposal"] = None
    else:
        _require(
            mode == MATPHYS_CAUSAL_PREFIX_MODE,
            "published parity mode cannot emit a causal proposal",
        )
        outputs["causal_proposal"] = _normalize_file_record(
            proposal_value, name="causal proposal", verify_file=False
        )
    boundary_raw = _mapping(
        artifact.get("information_boundary"), name="information_boundary"
    )
    require_exact_fields(
        boundary_raw, expected=_BOUNDARY_FIELDS, name="information_boundary"
    )
    expected_boundary = {
        "target_prefix_used_for_parameter_fit": True,
        "target_object_used_for_checkpoint_training": target in training,
        "target_future_observations_used": False,
        "future_outcomes_opened": False,
        "known_future_robot_action_used": True,
        "causal_backend_eligible": mode == MATPHYS_CAUSAL_PREFIX_MODE,
        "published_benchmark_control_only": mode == MATPHYS_PUBLISHED_PARITY_MODE,
    }
    _require(dict(boundary_raw) == expected_boundary, "information boundary changed")
    identity: dict[str, Any] = {
        "schema": MATPHYS_OFFICIAL_PRODUCER_SCHEMA,
        "schema_version": MATPHYS_OFFICIAL_PRODUCER_VERSION,
        "producer_protocol": MATPHYS_OFFICIAL_PRODUCER_PROTOCOL,
        "mode": mode,
        "backend_kind": MATPHYS_OFFICIAL_BACKEND_KIND,
        "parameterization": MATPHYS_OFFICIAL_PARAMETERIZATION,
        "rollout_backend": MATPHYS_OFFICIAL_ROLLOUT_BACKEND,
        "coordinate_frame": nonempty_string(
            artifact.get("coordinate_frame"), name="coordinate_frame"
        ),
        "position_unit": nonempty_string(
            artifact.get("position_unit"), name="position_unit"
        ),
        "source_repository": repository_name(
            artifact.get("source_repository"), name="source_repository"
        ),
        "source_revision": exact_revision(
            artifact.get("source_revision"), name="source_revision"
        ),
        "simulator_repository": repository_name(
            artifact.get("simulator_repository"), name="simulator_repository"
        ),
        "simulator_revision": exact_revision(
            artifact.get("simulator_revision"), name="simulator_revision"
        ),
        "case_id": nonempty_string(artifact.get("case_id"), name="case_id"),
        "target_object_id": target,
        "checkpoint_training_object_ids": list(training),
        "target_fit_frame_range_half_open": [fit_start, fit_stop],
        "future_frame_start": future_start,
        "proposal_strength": _finite_strength(artifact.get("proposal_strength")),
        "pipeline_components": list(components),
        "pipeline_component_artifacts": component_artifacts,
        "checkpoint": checkpoint,
        "spring_field": spring,
        "candidate_parameters": candidate_parameters,
        "identity_parameters": identity_parameters,
        "replay_input": replay_input,
        "replay_summary": replay_summary,
        "source_artifacts": dict(artifacts),
        "outputs": outputs,
        "information_boundary": expected_boundary,
        "claim_boundary": nonempty_string(
            artifact.get("claim_boundary"), name="claim_boundary"
        ),
    }
    _require(
        identity["coordinate_frame"] == "right-handed-z-up-world-v1",
        "coordinate frame changed",
    )
    _require(identity["position_unit"] == "m", "position unit changed")
    _require(
        identity["source_repository"] == MATPHYS_SOURCE_REPOSITORY,
        "MatPhys repository changed",
    )
    _require(
        identity["simulator_repository"] == PHYSTWIN_SOURCE_REPOSITORY,
        "PhysTwin repository changed",
    )
    _require(
        identity["claim_boundary"] == MATPHYS_OFFICIAL_CLAIM_BOUNDARY,
        "claim boundary changed",
    )
    normalized = {**identity, "artifact_id": content_id(identity)}
    _require(
        artifact.get("artifact_id") == normalized["artifact_id"],
        "official MatPhys producer content identity changed",
    )
    return cast(dict[str, Any], plain_json(normalized))


def _validate_output_record(
    value: object,
    *,
    root: Path,
    expected_path: str,
    name: str,
) -> Path:
    record = _normalize_file_record(value, name=name, verify_file=False)
    _require(record["path"] == expected_path, f"{name} path changed")
    path = _ordinary_file(root / expected_path, name=name)
    _require(file_sha256(path) == record["sha256"], f"{name} SHA-256 changed")
    _require(path.stat().st_size == record["byte_count"], f"{name} byte count changed")
    return path


def validate_matphys_official_producer_artifact(
    output_dir: str | Path,
    *,
    verify_sources: bool = False,
) -> dict[str, Any]:
    """Validate one published producer bundle and optionally rederive outputs."""

    requested = Path(output_dir).absolute()
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        "official MatPhys producer bundle is not an ordinary directory",
    )
    root = requested.resolve(strict=True)
    artifact = validate_matphys_official_producer_record(
        load_strict_json_object(root / ARTIFACT_FILENAME, label="MatPhys producer"),
        verify_sources=verify_sources,
    )
    outputs = cast(Mapping[str, Any], artifact["outputs"])
    expected_roster = {
        ARTIFACT_FILENAME,
        CHECKSUMS_FILENAME,
        CANDIDATE_ARCHIVE_FILENAME,
        IDENTITY_ARCHIVE_FILENAME,
    }
    if outputs["causal_proposal"] is not None:
        expected_roster.add(CAUSAL_PROPOSAL_FILENAME)
    actual_roster = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(actual_roster == expected_roster, "producer bundle file roster changed")
    _require(
        not any(path.is_symlink() for path in root.rglob("*")),
        "producer bundle contains a symlink",
    )
    candidate_path = _validate_output_record(
        outputs["candidate_archive"],
        root=root,
        expected_path=CANDIDATE_ARCHIVE_FILENAME,
        name="candidate archive",
    )
    identity_path = _validate_output_record(
        outputs["identity_replay_archive"],
        root=root,
        expected_path=IDENTITY_ARCHIVE_FILENAME,
        name="identity replay archive",
    )
    candidate = load_physical_rollout_archive(candidate_path)
    identity_replay = load_physical_rollout_archive(identity_path)
    for name in candidate:
        _require(
            candidate[name].shape == identity_replay[name].shape
            and candidate[name].dtype == identity_replay[name].dtype,
            f"MatPhys producer output contract differs for {name}",
        )
    for name in ("frame_zero_points_m", "persistence_m", "action_support"):
        _require(
            np.array_equal(candidate[name], identity_replay[name]),
            f"MatPhys producer changed {name} across replay arms",
        )
    proposal_record = outputs["causal_proposal"]
    if proposal_record is not None:
        proposal_path = _validate_output_record(
            proposal_record,
            root=root,
            expected_path=CAUSAL_PROPOSAL_FILENAME,
            name="causal proposal",
        )
        proposal = validate_matphys_backend_proposal(
            load_strict_json_object(proposal_path, label="causal MatPhys proposal"),
            verify_files=verify_sources,
        )
        _require(
            proposal["source_revision"] == artifact["source_revision"]
            and proposal["simulator_revision"] == artifact["simulator_revision"]
            and proposal["target_object_id"] == artifact["target_object_id"]
            and proposal["training_object_ids"]
            == artifact["checkpoint_training_object_ids"]
            and proposal["target_evidence_end_frame_exclusive"]
            == artifact["target_fit_frame_range_half_open"][1],
            "causal MatPhys proposal disagrees with producer provenance",
        )
        proposal_checkpoint = cast(Mapping[str, Any], proposal["checkpoint"])
        artifact_checkpoint = cast(Mapping[str, Any], artifact["checkpoint"])
        proposal_spring = cast(Mapping[str, Any], proposal["spring_field"])
        artifact_spring = cast(Mapping[str, Any], artifact["spring_field"])
        _require(
            proposal_checkpoint["path"] == artifact_checkpoint["path"]
            and proposal_checkpoint["sha256"] == artifact_checkpoint["sha256"]
            and proposal_spring["path"] == artifact_spring["path"]
            and proposal_spring["sha256"] == artifact_spring["sha256"]
            and proposal_spring["count"] == artifact_spring["count"]
            and proposal["proposal_strength"] == artifact["proposal_strength"]
            and proposal["source_artifacts"] == artifact["source_artifacts"],
            "causal MatPhys proposal input identity changed",
        )
    checksum_roster = sorted(expected_roster - {CHECKSUMS_FILENAME})
    expected_checksums = "".join(
        f"{file_sha256(root / name)}  {name}\n" for name in checksum_roster
    )
    checksums = _ordinary_file(root / CHECKSUMS_FILENAME, name="checksum manifest")
    _require(
        checksums.read_text(encoding="ascii") == expected_checksums,
        "producer checksum manifest changed",
    )
    if verify_sources:
        replay_path, replay = load_matphys_official_replay_input(
            cast(Mapping[str, Any], artifact["replay_input"])["path"]
        )
        _require(
            file_sha256(replay_path)
            == cast(Mapping[str, Any], artifact["replay_input"])["sha256"],
            "official MatPhys replay input SHA-256 changed",
        )
        _require(
            _replay_summary(replay) == artifact["replay_summary"],
            "replay summary changed",
        )
        expected_candidate, expected_identity = _physical_archives(replay)
        for name in candidate:
            _require(
                np.array_equal(candidate[name], expected_candidate[name])
                and np.array_equal(identity_replay[name], expected_identity[name]),
                f"MatPhys physical output no longer derives from replay input: {name}",
            )
    return artifact


__all__ = [
    "ARTIFACT_FILENAME",
    "CANDIDATE_ARCHIVE_FILENAME",
    "CAUSAL_PROPOSAL_FILENAME",
    "IDENTITY_ARCHIVE_FILENAME",
    "MATPHYS_CAUSAL_PREFIX_MODE",
    "MATPHYS_OFFICIAL_PIPELINE_COMPONENTS",
    "MATPHYS_PUBLISHED_PARITY_MODE",
    "MatPhysOfficialMode",
    "load_matphys_official_replay_input",
    "materialize_matphys_official_producer",
    "validate_matphys_official_producer_artifact",
    "validate_matphys_official_producer_record",
    "validate_matphys_official_replay_arrays",
    "write_matphys_official_replay_input",
]
