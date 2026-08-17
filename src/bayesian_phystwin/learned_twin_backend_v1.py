"""Portable, claim-bounded intake for learned deformable-twin rollouts.

The public release state of learned physical twins varies substantially.  This
module records that state explicitly and provides one strict intake for a
producer-generated :mod:`physical_rollout_v1` archive.  It does not import an
upstream model, infer that a paper-only method is executable, or turn a
portable export into an official reproduction claim.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np

from ._canonical_contracts import canonical_relative_posix_path, plain_json
from ._portable_contracts import (
    canonical_sorted_strings,
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
from .physical_rollout_v1 import (
    PHYSICAL_ROLLOUT_ARRAY_NAMES,
    load_physical_rollout_archive,
    write_deterministic_npz,
)

LEARNED_TWIN_REGISTRY_SCHEMA: Final = "bayesian-phystwin.learned-twin-registry"
LEARNED_TWIN_ARTIFACT_SCHEMA: Final = "bayesian-phystwin.learned-twin-backend"
LEARNED_TWIN_SCHEMA_VERSION: Final = 1
LEARNED_TWIN_REGISTRY_SNAPSHOT: Final = "2026-08-18"
LEARNED_TWIN_INTERFACE_KIND: Final = "portable-fixed-identity-rollout-intake-v1"
PORTABLE_ROLLOUT_CONTRACT: Final = "physical_rollout_v1"
CANONICAL_COORDINATE_FRAME: Final = "right-handed-z-up-world-v1"
POSITION_UNITS: Final = "m"

ARTIFACT_FILENAME: Final = "learned-twin-backend.json"
PHYSICAL_ARCHIVE_FILENAME: Final = "physical-prediction.npz"
SOURCE_ARCHIVE_FILENAME: Final = "source-physical-rollout.npz"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"

LearnedTwinMode = Literal["causal-source-v1", "published-parity-v1"]
LEARNED_TWIN_MODES: Final = frozenset(
    cast(tuple[LearnedTwinMode, ...], ("causal-source-v1", "published-parity-v1"))
)

LEARNED_TWIN_CLAIM_BOUNDARY: Final = (
    "A custody-checked portable intake of a producer-generated fixed-identity "
    "rollout. It is not evidence that BayesianPhysTwin executed the named "
    "upstream implementation, reproduced its paper, matched its benchmark, "
    "improved a target, calibrated uncertainty, or achieved state of the art. "
    "Only causal-source-v1 artifacts satisfy the recorded causal input split."
)


@dataclass(frozen=True, slots=True)
class LearnedTwinProfile:
    """One frozen public-availability and adapter-support record."""

    profile_id: str
    method_name: str
    paper_url: str
    upstream_repository: str | None
    upstream_snapshot_revision: str | None
    public_release_status: str
    model_family: str
    native_adapter_status: str
    public_runtime_executable: bool
    portable_intake_supported: bool = True
    portable_output_contract: str = PORTABLE_ROLLOUT_CONTRACT
    evidence_stage: str = "registered-adapter"
    source_value_qualified: bool = False
    recommended_for_claim_bearing_evaluation: bool = False

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], plain_json(asdict(self)))


_PROFILES: Final[tuple[LearnedTwinProfile, ...]] = (
    LearnedTwinProfile(
        profile_id="matphys-v1",
        method_name="MatPhys",
        paper_url="https://arxiv.org/abs/2605.19386",
        upstream_repository="Yrainy0615/MatPhys",
        upstream_snapshot_revision="c16b858dfb79bf21024ead24b45a710600de7b4f",
        public_release_status="executable-source-public",
        model_family="learned-spring-field-plus-warp-rollout",
        native_adapter_status="guarded-spring-proposal-v1",
        public_runtime_executable=True,
    ),
    LearnedTwinProfile(
        profile_id="neuspring-v1",
        method_name="NeuSpring",
        paper_url="https://arxiv.org/abs/2511.08310",
        upstream_repository="GhiXu/NeuSpring",
        upstream_snapshot_revision="51d94f67ed1e2557fca29c1e86b418506e3d51ca",
        public_release_status="repository-metadata-only",
        model_family="neural-spring-field",
        native_adapter_status="blocked-on-public-runtime-and-checkpoint",
        public_runtime_executable=False,
    ),
    LearnedTwinProfile(
        profile_id="physpring-v1",
        method_name="PhySPRING",
        paper_url="https://arxiv.org/abs/2605.07687",
        upstream_repository=None,
        upstream_snapshot_revision=None,
        public_release_status="paper-only",
        model_family="physics-guided-spring-representation",
        native_adapter_status="blocked-on-public-runtime-and-checkpoint",
        public_runtime_executable=False,
    ),
    LearnedTwinProfile(
        profile_id="physworld-v1",
        method_name="PhysWorld",
        paper_url="https://arxiv.org/abs/2510.21447",
        upstream_repository="AlanYoung123/PhysWorld",
        upstream_snapshot_revision="157a309e4f58634b0265cae7c1f4fc04b07394c0",
        public_release_status="repository-metadata-only",
        model_family="learned-physical-world-model",
        native_adapter_status="blocked-on-public-runtime-and-checkpoint",
        public_runtime_executable=False,
    ),
    LearnedTwinProfile(
        profile_id="egophys-v1",
        method_name="EgoPhys",
        paper_url="https://arxiv.org/abs/2606.16202",
        upstream_repository="hjhyunjinkim/EgoPhys",
        upstream_snapshot_revision=None,
        public_release_status="announced-code-and-data-unavailable",
        model_family="egocentric-physical-world-model",
        native_adapter_status="blocked-on-public-runtime-data-and-checkpoint",
        public_runtime_executable=False,
    ),
)
LEARNED_TWIN_PROFILES: Final = {profile.profile_id: profile for profile in _PROFILES}

_PROFILE_FIELDS: Final = frozenset(asdict(_PROFILES[0]))
_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "interface_kind",
        "registry_snapshot",
        "profile",
        "mode",
        "case_id",
        "target_object_id",
        "training_object_ids",
        "inputs",
        "output",
        "coordinate_contract",
        "frame_contract",
        "information_boundary",
        "claim_boundary",
        "artifact_id",
    }
)
_INPUT_FIELDS: Final = frozenset({"source_rollout", "model_artifacts", "producer"})
_LOCAL_FILE_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})
_EXTERNAL_FILE_FIELDS: Final = frozenset(
    {"logical_path", "external_path", "sha256", "byte_count"}
)
_PRODUCER_FIELDS: Final = frozenset({"repository", "revision", "source_artifacts"})
_OUTPUT_FIELDS: Final = frozenset(
    {
        "path",
        "sha256",
        "byte_count",
        "contract",
        "source_array_identity_verified",
        "material_identity_fixed",
    }
)
_COORDINATE_FIELDS: Final = frozenset({"coordinate_frame", "position_units"})
_FRAME_FIELDS: Final = frozenset(
    {"evidence_frame_range_half_open", "rollout_frame_range_half_open"}
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "causal_forecast_eligible",
        "target_object_excluded_from_training",
        "target_future_observations_used",
        "future_outcomes_opened_before_sealing",
        "known_future_controller_action_used",
        "prediction_hashed_before_future_scoring",
        "official_method_reproduction_claimed",
        "published_benchmark_parity_claimed",
    }
)


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


def _frame_range(value: object, *, name: str) -> tuple[int, int]:
    values = _sequence(value, name=name)
    if len(values) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in values
    ):
        raise ValueError(f"{name} must contain two integer frame indices")
    start, stop = cast(int, values[0]), cast(int, values[1])
    if not 0 <= start < stop:
        raise ValueError(f"{name} must be a nonempty half-open range")
    return start, stop


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _external_file_records(
    values: Mapping[str, str | Path],
) -> list[dict[str, object]]:
    _require(isinstance(values, Mapping) and bool(values), "model artifacts are empty")
    records: list[dict[str, object]] = []
    for logical_path, value in sorted(values.items()):
        if type(logical_path) is not str or logical_path.strip() != logical_path:
            raise ValueError("model artifact keys must be canonical relative paths")
        logical = canonical_relative_posix_path(
            logical_path, name="model artifact logical path"
        )
        source = _ordinary_file(value, name=f"model artifact {logical}")
        records.append(
            {
                "logical_path": logical,
                "external_path": str(source),
                "sha256": _sha256_file(source),
                "byte_count": source.stat().st_size,
            }
        )
    return records


def _profile_from_record(value: object) -> LearnedTwinProfile:
    record = _mapping(value, name="profile")
    require_exact_fields(record, expected=_PROFILE_FIELDS, name="profile")
    profile_id = nonempty_string(record.get("profile_id"), name="profile.profile_id")
    expected = LEARNED_TWIN_PROFILES.get(profile_id)
    _require(expected is not None, "unknown learned-twin profile")
    selected = cast(LearnedTwinProfile, expected)
    _require(dict(record) == selected.to_dict(), "learned-twin profile changed")
    return selected


def describe_learned_twin_profiles() -> dict[str, object]:
    """Return the frozen availability and support matrix."""

    identity: dict[str, object] = {
        "schema": LEARNED_TWIN_REGISTRY_SCHEMA,
        "schema_version": LEARNED_TWIN_SCHEMA_VERSION,
        "snapshot_date": LEARNED_TWIN_REGISTRY_SNAPSHOT,
        "portable_interface": LEARNED_TWIN_INTERFACE_KIND,
        "profiles": [profile.to_dict() for profile in _PROFILES],
        "claim_boundary": LEARNED_TWIN_CLAIM_BOUNDARY,
    }
    return {**identity, "registry_id": content_id(identity)}


def _normalize_mode(value: object) -> LearnedTwinMode:
    mode = nonempty_string(value, name="mode")
    if mode not in LEARNED_TWIN_MODES:
        raise ValueError(f"unknown learned-twin mode: {mode}")
    return cast(LearnedTwinMode, mode)


def _normalize_local_file(
    value: object,
    *,
    root: Path,
    expected_path: str,
    name: str,
) -> Path:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_LOCAL_FILE_FIELDS, name=name)
    _require(record.get("path") == expected_path, f"{name} path changed")
    digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    byte_count = _positive_integer(record.get("byte_count"), name=f"{name}.byte_count")
    path = _ordinary_file(root / expected_path, name=name)
    _require(path.stat().st_size == byte_count, f"{name} byte count changed")
    _require(_sha256_file(path) == digest, f"{name} digest changed")
    return path


def _normalize_external_files(
    value: object,
    *,
    verify_sources: bool,
) -> list[dict[str, object]]:
    raw = _sequence(value, name="model_artifacts")
    _require(bool(raw), "model artifacts are empty")
    records: list[dict[str, object]] = []
    logical_paths: list[str] = []
    for index, item in enumerate(raw):
        name = f"model_artifacts[{index}]"
        record = _mapping(item, name=name)
        require_exact_fields(record, expected=_EXTERNAL_FILE_FIELDS, name=name)
        logical = canonical_relative_posix_path(
            nonempty_string(record.get("logical_path"), name=f"{name}.logical_path"),
            name=f"{name}.logical_path",
        )
        external_path = nonempty_string(
            record.get("external_path"), name=f"{name}.external_path"
        )
        digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
        byte_count = _positive_integer(
            record.get("byte_count"), name=f"{name}.byte_count"
        )
        normalized = {
            "logical_path": logical,
            "external_path": external_path,
            "sha256": digest,
            "byte_count": byte_count,
        }
        if verify_sources:
            source = _ordinary_file(external_path, name=name)
            _require(source.stat().st_size == byte_count, f"{name} byte count changed")
            _require(_sha256_file(source) == digest, f"{name} digest changed")
        logical_paths.append(logical)
        records.append(normalized)
    _require(
        logical_paths == sorted(set(logical_paths)),
        "model artifacts must be sorted and unique by logical path",
    )
    return records


def _normalize_producer(value: object) -> dict[str, object]:
    producer = _mapping(value, name="producer")
    require_exact_fields(producer, expected=_PRODUCER_FIELDS, name="producer")
    repository = repository_name(producer.get("repository"), name="producer.repository")
    revision = exact_revision(producer.get("revision"), name="producer.revision")
    artifacts = source_artifact_mapping(
        _mapping(producer.get("source_artifacts"), name="producer.source_artifacts"),
        name="producer.source_artifacts",
    )
    return {
        "repository": repository,
        "revision": revision,
        "source_artifacts": dict(artifacts),
    }


def _expected_information_boundary(
    *,
    mode: LearnedTwinMode,
    target_excluded: bool,
    target_future_observations_used: bool,
    known_future_action_used: bool,
) -> dict[str, bool]:
    causal = (
        mode == "causal-source-v1"
        and target_excluded
        and not target_future_observations_used
    )
    return {
        "causal_forecast_eligible": causal,
        "target_object_excluded_from_training": target_excluded,
        "target_future_observations_used": target_future_observations_used,
        "future_outcomes_opened_before_sealing": False,
        "known_future_controller_action_used": known_future_action_used,
        "prediction_hashed_before_future_scoring": True,
        "official_method_reproduction_claimed": False,
        "published_benchmark_parity_claimed": False,
    }


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_FILENAME
    )
    lines = "".join(
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
    )
    (root / CHECKSUMS_FILENAME).write_text(lines, encoding="ascii")


def materialize_learned_twin_backend(
    *,
    source_rollout_path: str | Path,
    model_artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
    profile_id: str,
    mode: LearnedTwinMode,
    producer_repository: str,
    producer_revision: str,
    producer_source_artifacts: Mapping[str, str],
    case_id: str,
    target_object_id: str,
    training_object_ids: Sequence[str],
    evidence_frame_range_half_open: tuple[int, int],
    rollout_frame_range_half_open: tuple[int, int],
    target_future_observations_used: bool = False,
    known_future_controller_action_used: bool = True,
) -> dict[str, Any]:
    """Publish one portable learned-twin backend bundle exactly once."""

    profile = LEARNED_TWIN_PROFILES.get(profile_id)
    _require(profile is not None, "unknown learned-twin profile")
    selected_mode = _normalize_mode(mode)
    case = nonempty_string(case_id, name="case_id")
    target = nonempty_string(target_object_id, name="target_object_id")
    training = canonical_sorted_strings(
        training_object_ids,
        name="training_object_ids",
        allow_empty=True,
    )
    target_excluded = target not in training
    evidence_range = _frame_range(
        evidence_frame_range_half_open,
        name="evidence_frame_range_half_open",
    )
    rollout_range = _frame_range(
        rollout_frame_range_half_open,
        name="rollout_frame_range_half_open",
    )
    _require(
        type(target_future_observations_used) is bool,
        "target_future_observations_used must be Boolean",
    )
    _require(
        type(known_future_controller_action_used) is bool,
        "known_future_controller_action_used must be Boolean",
    )
    if selected_mode == "causal-source-v1":
        _require(target_excluded, "causal mode requires target-object exclusion")
        _require(
            evidence_range[1] <= rollout_range[0],
            "causal evidence must end before the rollout begins",
        )
        _require(
            not target_future_observations_used,
            "causal mode forbids target future observations",
        )

    producer = _normalize_producer(
        {
            "repository": producer_repository,
            "revision": producer_revision,
            "source_artifacts": producer_source_artifacts,
        }
    )
    source = _ordinary_file(source_rollout_path, name="source rollout")
    arrays = load_physical_rollout_archive(source)
    _require(
        rollout_range[1] - rollout_range[0] == arrays["prediction_m"].shape[0],
        "rollout frame range does not match the physical archive",
    )
    external_models = _external_file_records(model_artifacts)

    output = Path(output_dir).absolute()
    _require(not output.exists(), "output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not output.parent.is_symlink()
        and not any(parent.is_symlink() for parent in output.parent.parents),
        "output parent must not traverse symlinks",
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    ).resolve(strict=True)
    try:
        provenance = staging / "provenance"
        provenance.mkdir()
        source_copy = provenance / SOURCE_ARCHIVE_FILENAME
        shutil.copyfile(source, source_copy)
        physical = staging / PHYSICAL_ARCHIVE_FILENAME
        write_deterministic_npz(physical, arrays)

        information_boundary = _expected_information_boundary(
            mode=selected_mode,
            target_excluded=target_excluded,
            target_future_observations_used=target_future_observations_used,
            known_future_action_used=known_future_controller_action_used,
        )
        identity: dict[str, object] = {
            "schema": LEARNED_TWIN_ARTIFACT_SCHEMA,
            "schema_version": LEARNED_TWIN_SCHEMA_VERSION,
            "interface_kind": LEARNED_TWIN_INTERFACE_KIND,
            "registry_snapshot": LEARNED_TWIN_REGISTRY_SNAPSHOT,
            "profile": cast(LearnedTwinProfile, profile).to_dict(),
            "mode": selected_mode,
            "case_id": case,
            "target_object_id": target,
            "training_object_ids": list(training),
            "inputs": {
                "source_rollout": _file_record(source_copy, relative_to=staging),
                "model_artifacts": external_models,
                "producer": producer,
            },
            "output": {
                **_file_record(physical, relative_to=staging),
                "contract": PORTABLE_ROLLOUT_CONTRACT,
                "source_array_identity_verified": True,
                "material_identity_fixed": True,
            },
            "coordinate_contract": {
                "coordinate_frame": CANONICAL_COORDINATE_FRAME,
                "position_units": POSITION_UNITS,
            },
            "frame_contract": {
                "evidence_frame_range_half_open": list(evidence_range),
                "rollout_frame_range_half_open": list(rollout_range),
            },
            "information_boundary": information_boundary,
            "claim_boundary": LEARNED_TWIN_CLAIM_BOUNDARY,
        }
        artifact = {**identity, "artifact_id": content_id(identity)}
        write_atomic_json(
            artifact,
            staging / ARTIFACT_FILENAME,
            overwrite=False,
        )
        _write_checksums(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_learned_twin_backend(output)


def validate_learned_twin_backend(
    output_dir: str | Path,
    *,
    verify_sources: bool = False,
) -> dict[str, Any]:
    """Validate custody, semantics, and exact portable-array identity."""

    _require(type(verify_sources) is bool, "verify_sources must be Boolean")
    requested = Path(output_dir).absolute()
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        "learned-twin bundle must be an ordinary non-symlink directory",
    )
    root = requested.resolve(strict=True)
    _require(
        not any(path.is_symlink() for path in root.rglob("*")),
        "learned-twin bundle must not contain symlinks",
    )
    expected_roster = {
        ARTIFACT_FILENAME,
        CHECKSUMS_FILENAME,
        PHYSICAL_ARCHIVE_FILENAME,
        f"provenance/{SOURCE_ARCHIVE_FILENAME}",
    }
    actual_roster = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(actual_roster == expected_roster, "learned-twin bundle roster changed")

    artifact = load_strict_json_object(
        root / ARTIFACT_FILENAME, label="learned-twin artifact"
    )
    require_exact_fields(artifact, expected=_ARTIFACT_FIELDS, name="artifact")
    _require(
        artifact.get("schema") == LEARNED_TWIN_ARTIFACT_SCHEMA,
        "artifact schema changed",
    )
    _require(
        artifact.get("schema_version") == LEARNED_TWIN_SCHEMA_VERSION,
        "artifact schema version changed",
    )
    _require(
        artifact.get("interface_kind") == LEARNED_TWIN_INTERFACE_KIND,
        "learned-twin interface changed",
    )
    _require(
        artifact.get("registry_snapshot") == LEARNED_TWIN_REGISTRY_SNAPSHOT,
        "learned-twin registry snapshot changed",
    )
    _profile_from_record(artifact.get("profile"))
    mode = _normalize_mode(artifact.get("mode"))
    nonempty_string(artifact.get("case_id"), name="case_id")
    target = nonempty_string(artifact.get("target_object_id"), name="target_object_id")
    raw_training = _sequence(artifact.get("training_object_ids"), name="training")
    training_values = [
        nonempty_string(value, name="training_object_ids entry")
        for value in raw_training
    ]
    training = canonical_sorted_strings(
        training_values,
        name="training_object_ids",
        allow_empty=True,
    )
    _require(list(training) == list(raw_training), "training object order changed")
    target_excluded = target not in training

    inputs = _mapping(artifact.get("inputs"), name="inputs")
    require_exact_fields(inputs, expected=_INPUT_FIELDS, name="inputs")
    source = _normalize_local_file(
        inputs.get("source_rollout"),
        root=root,
        expected_path=f"provenance/{SOURCE_ARCHIVE_FILENAME}",
        name="source_rollout",
    )
    _normalize_external_files(
        inputs.get("model_artifacts"), verify_sources=verify_sources
    )
    _normalize_producer(inputs.get("producer"))

    output = _mapping(artifact.get("output"), name="output")
    require_exact_fields(output, expected=_OUTPUT_FIELDS, name="output")
    physical = _normalize_local_file(
        {key: output.get(key) for key in _LOCAL_FILE_FIELDS},
        root=root,
        expected_path=PHYSICAL_ARCHIVE_FILENAME,
        name="output",
    )
    _require(output.get("contract") == PORTABLE_ROLLOUT_CONTRACT, "contract changed")
    _require(
        output.get("source_array_identity_verified") is True,
        "source array identity was not verified",
    )
    _require(
        output.get("material_identity_fixed") is True,
        "material identity is not fixed",
    )
    source_arrays = load_physical_rollout_archive(source)
    output_arrays = load_physical_rollout_archive(physical)
    for name in sorted(PHYSICAL_ROLLOUT_ARRAY_NAMES):
        left, right = source_arrays[name], output_arrays[name]
        _require(left.dtype == right.dtype, f"{name} dtype changed")
        _require(left.shape == right.shape, f"{name} shape changed")
        _require(right.flags.c_contiguous, f"{name} is not C contiguous")
        _require(
            left.tobytes(order="C") == right.tobytes(order="C"),
            f"{name} values changed",
        )

    coordinate = _mapping(artifact.get("coordinate_contract"), name="coordinate")
    require_exact_fields(coordinate, expected=_COORDINATE_FIELDS, name="coordinate")
    _require(
        dict(coordinate)
        == {
            "coordinate_frame": CANONICAL_COORDINATE_FRAME,
            "position_units": POSITION_UNITS,
        },
        "coordinate contract changed",
    )
    frames = _mapping(artifact.get("frame_contract"), name="frame_contract")
    require_exact_fields(frames, expected=_FRAME_FIELDS, name="frame_contract")
    evidence_range = _frame_range(
        frames.get("evidence_frame_range_half_open"),
        name="evidence_frame_range_half_open",
    )
    rollout_range = _frame_range(
        frames.get("rollout_frame_range_half_open"),
        name="rollout_frame_range_half_open",
    )
    _require(
        rollout_range[1] - rollout_range[0] == output_arrays["prediction_m"].shape[0],
        "rollout frame range changed",
    )

    boundary = _mapping(artifact.get("information_boundary"), name="boundary")
    require_exact_fields(boundary, expected=_BOUNDARY_FIELDS, name="boundary")
    future_observations = boundary.get("target_future_observations_used")
    known_action = boundary.get("known_future_controller_action_used")
    _require(type(future_observations) is bool, "future-observation flag changed")
    _require(type(known_action) is bool, "known-action flag changed")
    expected_boundary = _expected_information_boundary(
        mode=mode,
        target_excluded=target_excluded,
        target_future_observations_used=cast(bool, future_observations),
        known_future_action_used=cast(bool, known_action),
    )
    _require(dict(boundary) == expected_boundary, "information boundary changed")
    if mode == "causal-source-v1":
        _require(target_excluded, "causal target is present in training")
        _require(
            evidence_range[1] <= rollout_range[0],
            "causal evidence overlaps the rollout",
        )
        _require(not future_observations, "causal artifact uses future observations")
    _require(
        artifact.get("claim_boundary") == LEARNED_TWIN_CLAIM_BOUNDARY,
        "claim boundary changed",
    )
    identity = {key: value for key, value in artifact.items() if key != "artifact_id"}
    _require(
        artifact.get("artifact_id") == content_id(identity),
        "learned-twin artifact identity changed",
    )

    expected_checksums = "".join(
        f"{_sha256_file(root / path)}  {path}\n"
        for path in sorted(expected_roster - {CHECKSUMS_FILENAME})
    )
    actual_checksums = (root / CHECKSUMS_FILENAME).read_text(encoding="ascii")
    _require(actual_checksums == expected_checksums, "bundle checksums changed")
    return cast(dict[str, Any], plain_json(artifact))


__all__ = [
    "LEARNED_TWIN_ARTIFACT_SCHEMA",
    "LEARNED_TWIN_CLAIM_BOUNDARY",
    "LEARNED_TWIN_PROFILES",
    "LEARNED_TWIN_REGISTRY_SCHEMA",
    "LEARNED_TWIN_SCHEMA_VERSION",
    "LearnedTwinMode",
    "LearnedTwinProfile",
    "describe_learned_twin_profiles",
    "materialize_learned_twin_backend",
    "validate_learned_twin_backend",
]
