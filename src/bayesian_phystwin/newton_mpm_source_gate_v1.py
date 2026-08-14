"""Source-only preparation and scoring for the Newton MPM backend gate."""

from __future__ import annotations

import pickle
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .newton_mpm_backend_v1 import file_sha256
from .physical_rollout_v1 import (
    load_physical_rollout_archive,
    write_deterministic_npz,
)

PROTOCOL_SCHEMA = "bayesian-phystwin.newton-mpm-source-gate-protocol"
PROTOCOL_VERSION = 1
SOURCE_INPUT_SCHEMA = "bayesian-phystwin.newton-mpm-source-input-v1"
SOURCE_CUSTODY_SCHEMA = "bayesian-phystwin.newton-mpm-source-custody-v1"
GRID_SCHEMA = "bayesian-phystwin.newton-mpm-source-grid-v1"
PREFIX_RESULT_SCHEMA = "bayesian-phystwin.newton-mpm-source-prefix-result-v1"
FUTURE_RESULT_SCHEMA = "bayesian-phystwin.newton-mpm-source-future-result-v1"

SOURCE_INPUT_FILENAME = "source-inputs.npz"
PREFIX_OUTCOME_FILENAME = "prefix-outcomes.npz"
FUTURE_OUTCOME_FILENAME = "future-outcomes.npz"
SOURCE_CUSTODY_FILENAME = "source-custody.json"
GRID_MANIFEST_FILENAME = "newton-grid.json"
PREFIX_RESULT_FILENAME = "prefix-result.json"
FUTURE_RESULT_FILENAME = "future-result.json"
SELECTED_PHYSICAL_FILENAME = "selected-physical-prediction.npz"

GRID_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "source_inputs_sha256",
        "runtime",
        "implementation",
        "information_boundary",
        "candidates",
        "successful_candidate_count",
        "technical_failure_count",
        "final_ensemble_spread_m",
        "grid_id",
    }
)
IMPLEMENTATION_SOURCE_PATHS = frozenset(
    {
        "src/bayesian_phystwin/_newton_mpm_source_runtime.py",
        "src/bayesian_phystwin/newton_mpm_source_gate_v1.py",
        "src/bayesian_phystwin/cli/newton_mpm_backend.py",
    }
)
GRID_SUCCESS_FIELDS = frozenset(
    {
        "candidate_index",
        "young_modulus_pa",
        "damping",
        "status",
        "physical_archive",
        "physical_archive_sha256",
        "replay_coordinate_rmse_m",
        "maximum_zero_action_drift_m",
        "maximum_action_response_m",
    }
)
GRID_FAILURE_FIELDS = frozenset(
    {
        "candidate_index",
        "young_modulus_pa",
        "damping",
        "status",
        "error_type",
        "error_message",
    }
)
PREFIX_RESULT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "source_custody_sha256",
        "grid_manifest_sha256",
        "information_boundary",
        "comparators",
        "candidates",
        "successful_candidate_count",
        "required_successful_candidate_count",
        "selected_candidate_index",
        "final_ensemble_spread_m",
        "validation_checks",
        "validation_gate_passed",
        "selection",
        "selected_physical_sha256",
        "selected_physical_is_byte_exact_source",
        "future_scoring_authorized",
        "result_id",
    }
)
SOURCE_CUSTODY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "case_id",
        "information_boundary",
        "mapping",
        "sources",
        "artifacts",
        "custody_id",
    }
)
VALIDATION_CHECK_FIELDS = frozenset(
    {
        "complete_candidate_denominator",
        "finite_ensemble_spread",
        "minimum_ensemble_spread",
        "maximum_ensemble_spread",
        "balanced_validation_improvement",
        "identity_nonregression_vs_incumbent",
        "chamfer_nonregression_vs_incumbent",
        "zero_action_drift",
        "deterministic_replay",
    }
)

SOURCE_INPUT_ARRAYS = frozenset(
    {
        "frame_zero_points_m",
        "controller_points_m",
        "attachment_indices",
        "attachment_weights",
        "action_support",
    }
)
OUTCOME_ARRAYS = frozenset({"object_points_m", "valid_mask", "frame_indices"})

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class SourceProtocol:
    """Validated view of the frozen one-case source protocol."""

    path: Path
    value: Mapping[str, Any]
    sha256: str

    @property
    def protocol_id(self) -> str:
        return cast(str, self.value["protocol_id"])

    @property
    def frame_count(self) -> int:
        return int(cast(Mapping[str, Any], self.value["geometry"])["frame_count"])

    @property
    def observed_count(self) -> int:
        return int(
            cast(Mapping[str, Any], self.value["geometry"])["observed_identity_count"]
        )

    @property
    def material_count(self) -> int:
        return int(
            cast(Mapping[str, Any], self.value["geometry"])["material_particle_count"]
        )

    @property
    def fit_range(self) -> tuple[int, int]:
        boundary = cast(Mapping[str, Any], self.value["information_boundary"])
        return _frame_range(boundary["fit_object_frames_half_open"], name="fit")

    @property
    def validation_range(self) -> tuple[int, int]:
        boundary = cast(Mapping[str, Any], self.value["information_boundary"])
        return _frame_range(
            boundary["validation_object_frames_half_open"],
            name="validation",
        )

    @property
    def future_range(self) -> tuple[int, int]:
        boundary = cast(Mapping[str, Any], self.value["information_boundary"])
        return _frame_range(
            boundary["future_object_frames_half_open"],
            name="future",
        )


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a finite number")
    return result


def _canonical_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty relative POSIX path")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    return value


def _frame_range(value: object, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} frame range must contain two integers")
    start, stop = int(value[0]), int(value[1])
    if start < 0 or stop <= start:
        raise ValueError(f"{name} frame range is empty or negative")
    return start, stop


def load_source_protocol(path: str | Path) -> SourceProtocol:
    """Load the exact one-case protocol while rejecting malformed grids."""

    source = _ordinary_file(path, name="source protocol")
    value = load_strict_json_object(source, label="source protocol")
    _require(value.get("schema") == PROTOCOL_SCHEMA, "protocol schema changed")
    _require(
        value.get("schema_version") == PROTOCOL_VERSION,
        "protocol schema version changed",
    )
    _require(
        value.get("cohort_role") == "already-open-development-source",
        "protocol cohort role changed",
    )
    boundary = _mapping(value.get("information_boundary"), name="information_boundary")
    _require(
        boundary.get("target_or_held_out_artifact_access_allowed") is False,
        "target access must remain forbidden",
    )
    geometry = _mapping(value.get("geometry"), name="geometry")
    frame_count = _positive_integer(geometry.get("frame_count"), name="frame_count")
    material_count = _positive_integer(
        geometry.get("material_particle_count"),
        name="material_particle_count",
    )
    observed_count = _positive_integer(
        geometry.get("observed_identity_count"),
        name="observed_identity_count",
    )
    _require(
        observed_count <= material_count, "observed identity count exceeds material"
    )
    ranges = (
        _frame_range(boundary.get("fit_object_frames_half_open"), name="fit"),
        _frame_range(
            boundary.get("validation_object_frames_half_open"),
            name="validation",
        ),
        _frame_range(boundary.get("future_object_frames_half_open"), name="future"),
    )
    _require(
        ranges[0][1] == ranges[1][0]
        and ranges[1][1] == ranges[2][0]
        and ranges[2][1] == frame_count,
        "protocol frame partitions are not contiguous and exhaustive",
    )
    source_files = _mapping(value.get("source_files"), name="source_files")
    _require(
        set(source_files)
        == {
            "final_data",
            "optimal_params",
            "incumbent_physical",
            "matphys_physical",
            "matphys_replay_result",
        },
        "source file roster changed",
    )
    for name, record_value in source_files.items():
        record = _mapping(record_value, name=f"source_files.{name}")
        require_exact_fields(
            record,
            expected=frozenset({"sha256"}),
            name=f"source_files.{name}",
        )
        sha256_digest(record.get("sha256"), name=f"source_files.{name}.sha256")
    grid = value.get("parameter_grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("parameter_grid must be a nonempty list")
    identities: set[tuple[float, float]] = set()
    for index, candidate_value in enumerate(grid):
        candidate = _mapping(candidate_value, name=f"parameter_grid[{index}]")
        require_exact_fields(
            candidate,
            expected=frozenset({"young_modulus_pa", "damping"}),
            name=f"parameter_grid[{index}]",
        )
        young = _finite_number(
            candidate.get("young_modulus_pa"),
            name=f"parameter_grid[{index}].young_modulus_pa",
            positive=True,
        )
        damping = _finite_number(
            candidate.get("damping"),
            name=f"parameter_grid[{index}].damping",
        )
        _require(damping >= 0.0, "candidate damping must be nonnegative")
        identity = (young, damping)
        _require(identity not in identities, "parameter_grid contains duplicates")
        identities.add(identity)
    selection = _mapping(value.get("selection"), name="selection")
    _require(
        _positive_integer(
            selection.get("required_successful_candidates"),
            name="required_successful_candidates",
        )
        == len(grid),
        "successful-candidate denominator differs from the grid",
    )
    return SourceProtocol(path=source, value=value, sha256=file_sha256(source))


def _pinned_file(
    protocol: SourceProtocol,
    path: str | Path,
    *,
    key: str,
) -> Path:
    source = _ordinary_file(path, name=key)
    source_files = cast(Mapping[str, Any], protocol.value["source_files"])
    record = cast(Mapping[str, Any], source_files[key])
    expected = cast(str, record["sha256"])
    _require(file_sha256(source) == expected, f"{key} SHA-256 changed")
    return source


def _load_pinned_pickle(path: Path, *, name: str) -> Any:
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except (OSError, pickle.PickleError) as error:
        raise ValueError(f"cannot load pinned {name}") from error


def _float_array(
    value: object,
    *,
    name: str,
    ndim: int,
) -> npt.NDArray[np.floating[Any]]:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating {ndim}D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _controller_attachments(
    points: FloatArray,
    controllers: FloatArray,
    *,
    radius_m: float,
    maximum: int,
    minimum_distance_m: float,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], int]:
    per_node: dict[int, list[tuple[int, float]]] = {}
    edge_count = 0
    points64 = points.astype(np.float64)
    for controller_index, controller in enumerate(controllers.astype(np.float64)):
        delta = points64 - controller
        distance = np.linalg.norm(delta, axis=1)
        candidates = np.flatnonzero(distance <= radius_m)
        order = np.lexsort((candidates, distance[candidates]))[:maximum]
        for node_value in candidates[order]:
            node = int(node_value)
            inverse_distance = 1.0 / max(float(distance[node]), minimum_distance_m)
            per_node.setdefault(node, []).append((controller_index, inverse_distance))
            edge_count += 1
    if not per_node:
        raise ValueError("controller mapping produced no material attachments")
    indices = np.asarray(sorted(per_node), dtype=np.int32)
    weights: npt.NDArray[np.float32] = np.zeros(
        (len(indices), len(controllers)),
        dtype=np.float32,
    )
    for row, node in enumerate(indices):
        entries = per_node[node]
        normalizer = sum(weight for _, weight in entries)
        for controller_index, weight in entries:
            weights[row, controller_index] = np.float32(weight / normalizer)
    _require(
        np.allclose(np.sum(weights, axis=1), 1.0, atol=1.0e-6, rtol=0.0),
        "controller attachment weights do not sum to one",
    )
    return indices, weights, edge_count


def _write_source_manifest(
    protocol: SourceProtocol,
    output: Path,
    *,
    inputs_path: Path,
    prefix_path: Path,
    future_path: Path,
    source_paths: Mapping[str, Path],
    edge_count: int,
    attachment_count: int,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": SOURCE_CUSTODY_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "case_id": protocol.value["case_id"],
        "information_boundary": {
            "custody_preparer_read_full_already_open_source_payload": True,
            "prediction_input_contains_future_object_observations": False,
            "prediction_input_contains_known_full_controller_action": True,
            "prefix_and_future_outcomes_are_separate_files": True,
            "target_or_held_out_artifact_read": False,
        },
        "mapping": {
            "controller_edge_count": edge_count,
            "attached_material_particle_count": attachment_count,
        },
        "sources": {
            name: {"sha256": file_sha256(path), "byte_count": path.stat().st_size}
            for name, path in sorted(source_paths.items())
        },
        "artifacts": {
            SOURCE_INPUT_FILENAME: {
                "sha256": file_sha256(inputs_path),
                "byte_count": inputs_path.stat().st_size,
            },
            PREFIX_OUTCOME_FILENAME: {
                "sha256": file_sha256(prefix_path),
                "byte_count": prefix_path.stat().st_size,
            },
            FUTURE_OUTCOME_FILENAME: {
                "sha256": file_sha256(future_path),
                "byte_count": future_path.stat().st_size,
            },
        },
    }
    manifest = {**identity, "custody_id": content_id(identity)}
    write_atomic_json(manifest, output / SOURCE_CUSTODY_FILENAME, overwrite=False)
    return manifest


def load_source_custody(
    path: str | Path,
    *,
    protocol: SourceProtocol,
) -> Mapping[str, Any]:
    """Validate the content-addressed source split without opening outcomes."""

    source = _ordinary_file(path, name="source custody")
    value = load_strict_json_object(source, label="source custody")
    require_exact_fields(value, expected=SOURCE_CUSTODY_FIELDS, name="source custody")
    _require(value.get("schema") == SOURCE_CUSTODY_SCHEMA, "custody schema changed")
    _require(value.get("schema_version") == 1, "custody version changed")
    _require(
        value.get("protocol_id") == protocol.protocol_id,
        "custody protocol ID changed",
    )
    _require(
        value.get("protocol_sha256") == protocol.sha256,
        "custody protocol changed",
    )
    _require(value.get("case_id") == protocol.value["case_id"], "custody case changed")
    boundary = _mapping(value.get("information_boundary"), name="custody boundary")
    require_exact_fields(
        boundary,
        expected=frozenset(
            {
                "custody_preparer_read_full_already_open_source_payload",
                "prediction_input_contains_future_object_observations",
                "prediction_input_contains_known_full_controller_action",
                "prefix_and_future_outcomes_are_separate_files",
                "target_or_held_out_artifact_read",
            }
        ),
        name="custody boundary",
    )
    _require(
        boundary.get("custody_preparer_read_full_already_open_source_payload") is True
        and boundary.get("prediction_input_contains_future_object_observations")
        is False
        and boundary.get("prediction_input_contains_known_full_controller_action")
        is True
        and boundary.get("prefix_and_future_outcomes_are_separate_files") is True
        and boundary.get("target_or_held_out_artifact_read") is False,
        "source custody crossed its information boundary",
    )
    mapping = _mapping(value.get("mapping"), name="custody mapping")
    require_exact_fields(
        mapping,
        expected=frozenset(
            {"controller_edge_count", "attached_material_particle_count"}
        ),
        name="custody mapping",
    )
    contact = cast(Mapping[str, Any], protocol.value["contact_mapping"])
    _require(
        mapping.get("controller_edge_count")
        == int(contact["expected_controller_edge_count"])
        and mapping.get("attached_material_particle_count")
        == int(contact["expected_attached_material_particle_count"]),
        "custody contact mapping changed",
    )
    sources = _mapping(value.get("sources"), name="custody sources")
    source_files = cast(Mapping[str, Any], protocol.value["source_files"])
    require_exact_fields(
        sources,
        expected=frozenset(source_files),
        name="custody sources",
    )
    for name, record_value in sources.items():
        record = _mapping(record_value, name=f"custody sources.{name}")
        require_exact_fields(
            record,
            expected=frozenset({"sha256", "byte_count"}),
            name=f"custody sources.{name}",
        )
        _require(
            record.get("sha256")
            == cast(Mapping[str, Any], source_files[name])["sha256"],
            f"custody source {name} differs from the protocol",
        )
        _positive_integer(
            record.get("byte_count"),
            name=f"custody sources.{name}.byte_count",
        )
    artifacts = _mapping(value.get("artifacts"), name="custody artifacts")
    require_exact_fields(
        artifacts,
        expected=frozenset(
            {SOURCE_INPUT_FILENAME, PREFIX_OUTCOME_FILENAME, FUTURE_OUTCOME_FILENAME}
        ),
        name="custody artifacts",
    )
    for name, record_value in artifacts.items():
        record = _mapping(record_value, name=f"custody artifacts.{name}")
        require_exact_fields(
            record,
            expected=frozenset({"sha256", "byte_count"}),
            name=f"custody artifacts.{name}",
        )
        sha256_digest(
            record.get("sha256"),
            name=f"custody artifacts.{name}.sha256",
        )
        _positive_integer(
            record.get("byte_count"),
            name=f"custody artifacts.{name}.byte_count",
        )
    identity = dict(value)
    custody_id = identity.pop("custody_id")
    _require(custody_id == content_id(identity), "source custody content ID changed")
    return value


def prepare_source_case(
    *,
    protocol_path: str | Path,
    final_data_path: str | Path,
    optimal_params_path: str | Path,
    incumbent_physical_path: str | Path,
    matphys_physical_path: str | Path,
    matphys_replay_result_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Split one pinned source payload into prediction, prefix, and future files."""

    protocol = load_source_protocol(protocol_path)
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "final_data": _pinned_file(protocol, final_data_path, key="final_data"),
        "optimal_params": _pinned_file(
            protocol,
            optimal_params_path,
            key="optimal_params",
        ),
        "incumbent_physical": _pinned_file(
            protocol,
            incumbent_physical_path,
            key="incumbent_physical",
        ),
        "matphys_physical": _pinned_file(
            protocol,
            matphys_physical_path,
            key="matphys_physical",
        ),
        "matphys_replay_result": _pinned_file(
            protocol,
            matphys_replay_result_path,
            key="matphys_replay_result",
        ),
    }
    final_data_raw = _load_pinned_pickle(paths["final_data"], name="final_data")
    optimal_raw = _load_pinned_pickle(paths["optimal_params"], name="optimal_params")
    if not isinstance(final_data_raw, Mapping) or not isinstance(optimal_raw, Mapping):
        raise ValueError("pinned source pickles must contain mappings")
    final_data = cast(Mapping[str, Any], final_data_raw)
    optimal = cast(Mapping[str, Any], optimal_raw)
    required = {
        "object_points",
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
        "surface_points",
        "interior_points",
    }
    _require(required <= set(final_data), "final_data source fields are incomplete")
    object_points = _float_array(
        final_data["object_points"], name="object_points", ndim=3
    )
    controllers = _float_array(
        final_data["controller_points"], name="controller_points", ndim=3
    )
    surface = _float_array(final_data["surface_points"], name="surface_points", ndim=2)
    interior = _float_array(
        final_data["interior_points"], name="interior_points", ndim=2
    )
    visible = np.asarray(final_data["object_visibilities"])
    motion_valid = np.asarray(final_data["object_motions_valid"])
    geometry = cast(Mapping[str, Any], protocol.value["geometry"])
    expected_object = int(geometry["observed_identity_count"])
    expected_surface = int(geometry["surface_point_count"])
    expected_interior = int(geometry["interior_point_count"])
    expected_controller = int(geometry["controller_point_count"])
    _require(
        object_points.shape == (protocol.frame_count, expected_object, 3),
        "object point shape changed",
    )
    _require(surface.shape == (expected_surface, 3), "surface point shape changed")
    _require(interior.shape == (expected_interior, 3), "interior point shape changed")
    _require(
        controllers.shape == (protocol.frame_count, expected_controller, 3),
        "controller trajectory shape changed",
    )
    _require(
        visible.shape == motion_valid.shape == object_points.shape[:2],
        "source validity shape changed",
    )
    _require(
        visible.dtype == np.bool_ and motion_valid.dtype == np.bool_,
        "validity must be boolean",
    )
    structure = np.concatenate((object_points[0], surface, interior), axis=0).astype(
        np.float32
    )
    _require(
        structure.shape == (protocol.material_count, 3),
        "material structure shape changed",
    )
    incumbent = load_physical_rollout_archive(
        paths["incumbent_physical"],
        expected_frame_count=protocol.frame_count,
    )
    matphys = load_physical_rollout_archive(
        paths["matphys_physical"],
        expected_frame_count=protocol.frame_count,
    )
    for name, archive in (("incumbent", incumbent), ("matphys", matphys)):
        _require(
            archive["prediction_m"].shape
            == (protocol.frame_count, protocol.material_count, 3),
            f"{name} physical shape changed",
        )
        _require(
            np.array_equal(archive["frame_zero_points_m"], structure),
            f"{name} frame zero differs from the source geometry",
        )
    _require(
        np.array_equal(incumbent["action_support"], matphys["action_support"]),
        "comparator action support changed",
    )
    contact = cast(Mapping[str, Any], protocol.value["contact_mapping"])
    radius = float(contact["controller_radius_m"])
    maximum = int(contact["controller_max_neighbours"])
    optimal_radius = _finite_number(
        optimal.get("controller_radius"),
        name="optimal controller radius",
        positive=True,
    )
    optimal_maximum = _positive_integer(
        optimal.get("controller_max_neighbours"),
        name="optimal controller maximum neighbours",
    )
    _require(
        optimal_radius == radius and optimal_maximum == maximum,
        "pinned optimal contact settings differ from the protocol",
    )
    attachments, weights, edge_count = _controller_attachments(
        structure,
        controllers[0],
        radius_m=radius,
        maximum=maximum,
        minimum_distance_m=float(contact["minimum_distance_m"]),
    )
    _require(
        edge_count == int(contact["expected_controller_edge_count"]),
        "controller edge count changed",
    )
    _require(
        len(attachments) == int(contact["expected_attached_material_particle_count"]),
        "attached material-particle count changed",
    )
    inputs_path = output / SOURCE_INPUT_FILENAME
    write_deterministic_npz(
        inputs_path,
        {
            "frame_zero_points_m": structure,
            "controller_points_m": controllers.astype(np.float32),
            "attachment_indices": attachments,
            "attachment_weights": weights,
            "action_support": incumbent["action_support"],
        },
    )
    valid = np.logical_and(visible, motion_valid)
    prefix_stop = protocol.validation_range[1]
    future_start, future_stop = protocol.future_range
    prefix_path = output / PREFIX_OUTCOME_FILENAME
    write_deterministic_npz(
        prefix_path,
        {
            "object_points_m": object_points[:prefix_stop].astype(np.float32),
            "valid_mask": valid[:prefix_stop],
            "frame_indices": np.arange(prefix_stop, dtype=np.int32),
        },
    )
    future_path = output / FUTURE_OUTCOME_FILENAME
    write_deterministic_npz(
        future_path,
        {
            "object_points_m": object_points[future_start:future_stop].astype(
                np.float32
            ),
            "valid_mask": valid[future_start:future_stop],
            "frame_indices": np.arange(future_start, future_stop, dtype=np.int32),
        },
    )
    return _write_source_manifest(
        protocol,
        output,
        inputs_path=inputs_path,
        prefix_path=prefix_path,
        future_path=future_path,
        source_paths=paths,
        edge_count=edge_count,
        attachment_count=len(attachments),
    )


def load_source_inputs(
    path: str | Path,
    *,
    protocol: SourceProtocol,
) -> dict[str, npt.NDArray[Any]]:
    """Load the prediction-only geometry/action artifact."""

    source = _ordinary_file(path, name="source inputs")
    try:
        with np.load(source, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load source inputs") from error
    _require(set(arrays) == SOURCE_INPUT_ARRAYS, "source input array roster changed")
    points = arrays["frame_zero_points_m"]
    controllers = arrays["controller_points_m"]
    indices = arrays["attachment_indices"]
    weights = arrays["attachment_weights"]
    support = arrays["action_support"]
    _require(
        points.shape == (protocol.material_count, 3)
        and points.dtype == np.float32
        and np.all(np.isfinite(points)),
        "source frame-zero geometry changed",
    )
    controller_count = int(
        cast(Mapping[str, Any], protocol.value["geometry"])["controller_point_count"]
    )
    _require(
        controllers.shape == (protocol.frame_count, controller_count, 3)
        and controllers.dtype == np.float32
        and np.all(np.isfinite(controllers)),
        "source controller trajectory changed",
    )
    _require(
        indices.ndim == 1
        and indices.dtype == np.int32
        and len(indices) > 0
        and np.array_equal(indices, np.unique(indices))
        and int(indices[0]) >= 0
        and int(indices[-1]) < protocol.material_count,
        "source attachment indices changed",
    )
    _require(
        weights.shape == (len(indices), controller_count)
        and weights.dtype == np.float32
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1.0e-6, rtol=0.0),
        "source attachment weights changed",
    )
    _require(
        support.shape == (protocol.material_count,)
        and support.dtype == np.float32
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "source action support changed",
    )
    return arrays


def _load_outcomes(
    path: str | Path,
    *,
    expected_indices: npt.NDArray[np.int32],
    observed_count: int,
) -> dict[str, npt.NDArray[Any]]:
    source = _ordinary_file(path, name="source outcomes")
    try:
        with np.load(source, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load source outcomes") from error
    _require(set(arrays) == OUTCOME_ARRAYS, "source outcome roster changed")
    points = arrays["object_points_m"]
    valid = arrays["valid_mask"]
    indices = arrays["frame_indices"]
    _require(
        points.shape == (len(expected_indices), observed_count, 3)
        and points.dtype == np.float32
        and np.all(np.isfinite(points)),
        "source outcome points changed",
    )
    _require(
        valid.shape == points.shape[:2] and valid.dtype == np.bool_,
        "source outcome validity changed",
    )
    _require(
        indices.dtype == np.int32 and np.array_equal(indices, expected_indices),
        "source outcome frame indices changed",
    )
    return arrays


def _coordinate_rmse(
    prediction: FloatArray,
    outcome: FloatArray,
    valid: npt.NDArray[np.bool_],
) -> float:
    frame_scores: list[float] = []
    for frame in range(len(prediction)):
        values = prediction[frame, valid[frame]] - outcome[frame, valid[frame]]
        if len(values):
            frame_scores.append(
                float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
            )
    if not frame_scores:
        raise ValueError("metric split has no valid identities")
    return float(np.mean(frame_scores))


def _symmetric_chamfer(
    prediction: FloatArray,
    outcome: FloatArray,
    valid: npt.NDArray[np.bool_],
) -> float:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - optional graph dependency
        raise RuntimeError(
            "source scoring requires bayesian-phystwin[graph]"
        ) from error
    frame_scores: list[float] = []
    for frame in range(len(prediction)):
        observed = outcome[frame, valid[frame]].astype(np.float64)
        predicted = prediction[frame, valid[frame]].astype(np.float64)
        if not len(observed):
            continue
        forward = cKDTree(observed).query(predicted, k=1)[0]
        reverse = cKDTree(predicted).query(observed, k=1)[0]
        frame_scores.append(float(0.5 * (np.mean(forward) + np.mean(reverse))))
    if not frame_scores:
        raise ValueError("metric split has no Chamfer-supporting frame")
    return float(np.mean(frame_scores))


def _split_metrics(
    prediction: FloatArray,
    outcome: FloatArray,
    valid: npt.NDArray[np.bool_],
    *,
    local_start: int,
    local_stop: int,
) -> dict[str, float]:
    selected_prediction = prediction[local_start:local_stop]
    selected_outcome = outcome[local_start:local_stop]
    selected_valid = valid[local_start:local_stop]
    return {
        "identity_coordinate_rmse_m": _coordinate_rmse(
            selected_prediction,
            selected_outcome,
            selected_valid,
        ),
        "symmetric_chamfer_m": _symmetric_chamfer(
            selected_prediction,
            selected_outcome,
            selected_valid,
        ),
    }


def _physical_prediction(
    path: str | Path,
    *,
    protocol: SourceProtocol,
) -> dict[str, npt.NDArray[Any]]:
    arrays = load_physical_rollout_archive(
        _ordinary_file(path, name="physical prediction"),
        expected_frame_count=protocol.frame_count,
    )
    _require(
        arrays["prediction_m"].shape
        == (protocol.frame_count, protocol.material_count, 3),
        "physical prediction shape changed",
    )
    return arrays


def load_grid_manifest(
    path: str | Path,
    *,
    protocol: SourceProtocol,
) -> Mapping[str, Any]:
    """Validate the sealed, outcome-blind Newton prediction grid."""

    source = _ordinary_file(path, name="Newton grid manifest")
    value = load_strict_json_object(source, label="Newton grid manifest")
    require_exact_fields(value, expected=GRID_FIELDS, name="Newton grid manifest")
    _require(value.get("schema") == GRID_SCHEMA, "Newton grid schema changed")
    _require(value.get("schema_version") == 1, "Newton grid version changed")
    _require(value.get("protocol_id") == protocol.protocol_id, "protocol ID changed")
    _require(
        value.get("protocol_sha256") == protocol.sha256, "protocol binding changed"
    )
    sha256_digest(value.get("source_inputs_sha256"), name="source_inputs_sha256")
    runtime = _mapping(value.get("runtime"), name="grid runtime")
    require_exact_fields(
        runtime,
        expected=frozenset(
            {
                "engine_version",
                "warp_version",
                "numpy_version",
                "scipy_version",
                "python_version",
                "device",
                "device_name",
            }
        ),
        name="grid runtime",
    )
    simulation = cast(Mapping[str, Any], protocol.value["simulation"])
    _require(
        runtime.get("engine_version") == simulation["engine_version"]
        and runtime.get("warp_version") == simulation["warp_version"],
        "grid runtime versions differ from the frozen protocol",
    )
    _require(
        runtime.get("numpy_version") == simulation["numpy_version"]
        and runtime.get("scipy_version") == simulation["scipy_version"],
        "grid numerical-library versions differ from the frozen protocol",
    )
    implementation = _mapping(value.get("implementation"), name="implementation")
    require_exact_fields(
        implementation,
        expected=frozenset({"git_head", "git_worktree_clean", "source_files"}),
        name="implementation",
    )
    git_head = implementation.get("git_head")
    _require(
        isinstance(git_head, str)
        and len(git_head) == 40
        and all(character in "0123456789abcdef" for character in git_head),
        "implementation Git HEAD is not a full lowercase commit ID",
    )
    _require(
        implementation.get("git_worktree_clean") is True,
        "prediction implementation was not run from a clean worktree",
    )
    source_files = _mapping(
        implementation.get("source_files"),
        name="implementation source files",
    )
    require_exact_fields(
        source_files,
        expected=IMPLEMENTATION_SOURCE_PATHS,
        name="implementation source files",
    )
    for name, digest in source_files.items():
        sha256_digest(digest, name=f"implementation source files.{name}")
    boundary = _mapping(value.get("information_boundary"), name="grid boundary")
    require_exact_fields(
        boundary,
        expected=frozenset(
            {
                "frame_zero_geometry_read",
                "known_full_controller_action_read",
                "object_outcome_artifact_read",
                "target_or_held_out_artifact_read",
            }
        ),
        name="grid boundary",
    )
    _require(
        boundary.get("frame_zero_geometry_read") is True
        and boundary.get("known_full_controller_action_read") is True
        and boundary.get("object_outcome_artifact_read") is False
        and boundary.get("target_or_held_out_artifact_read") is False,
        "prediction grid crossed its information boundary",
    )
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("grid candidates must be a list")
    expected_grid = cast(list[Mapping[str, Any]], protocol.value["parameter_grid"])
    _require(
        len(candidates) == len(expected_grid), "grid candidate denominator changed"
    )
    successful_count = 0
    for index, (candidate_value, expected_value) in enumerate(
        zip(candidates, expected_grid, strict=True)
    ):
        candidate = _mapping(candidate_value, name=f"grid candidates[{index}]")
        _require(candidate.get("candidate_index") == index, "candidate order changed")
        young = _finite_number(
            candidate.get("young_modulus_pa"),
            name=f"grid candidates[{index}].young_modulus_pa",
            positive=True,
        )
        damping = _finite_number(
            candidate.get("damping"),
            name=f"grid candidates[{index}].damping",
        )
        _require(
            young == float(expected_value["young_modulus_pa"])
            and damping == float(expected_value["damping"]),
            "grid candidate parameters differ from the frozen parameter grid",
        )
        status = candidate.get("status")
        if status == "success":
            require_exact_fields(
                candidate,
                expected=GRID_SUCCESS_FIELDS,
                name=f"grid candidates[{index}]",
            )
            _canonical_relative_path(
                candidate.get("physical_archive"),
                name=f"grid candidates[{index}].physical_archive",
            )
            sha256_digest(
                candidate.get("physical_archive_sha256"),
                name=f"grid candidates[{index}].physical_archive_sha256",
            )
            for field in (
                "replay_coordinate_rmse_m",
                "maximum_zero_action_drift_m",
                "maximum_action_response_m",
            ):
                metric = _finite_number(
                    candidate.get(field),
                    name=f"grid candidates[{index}].{field}",
                )
                _require(metric >= 0.0, f"grid candidates[{index}].{field} is negative")
            successful_count += 1
        elif status == "technical_failure":
            require_exact_fields(
                candidate,
                expected=GRID_FAILURE_FIELDS,
                name=f"grid candidates[{index}]",
            )
            _require(
                isinstance(candidate.get("error_type"), str)
                and bool(candidate.get("error_type"))
                and isinstance(candidate.get("error_message"), str),
                "technical-failure record is malformed",
            )
        else:
            raise ValueError("grid candidate status changed")
    _require(
        value.get("successful_candidate_count") == successful_count
        and value.get("technical_failure_count") == len(candidates) - successful_count,
        "grid status counts changed",
    )
    spread = _finite_number(
        value.get("final_ensemble_spread_m"),
        name="final_ensemble_spread_m",
    )
    _require(spread >= 0.0, "final ensemble spread is negative")
    identity = dict(value)
    grid_id = identity.pop("grid_id")
    _require(grid_id == content_id(identity), "Newton grid content ID changed")
    return value


def _candidate_records(
    grid: Mapping[str, Any],
    *,
    grid_root: Path,
    protocol: SourceProtocol,
) -> list[dict[str, Any]]:
    values = cast(list[Mapping[str, Any]], grid["candidates"])
    records: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        candidate = dict(value)
        _require(candidate.get("candidate_index") == index, "candidate order changed")
        if candidate.get("status") != "success":
            records.append(candidate)
            continue
        relative = _canonical_relative_path(
            candidate.get("physical_archive"),
            name="candidate physical archive",
        )
        archive = _ordinary_file(
            grid_root / relative, name="candidate physical archive"
        )
        _require(
            file_sha256(archive) == candidate.get("physical_archive_sha256"),
            "candidate physical SHA-256 changed",
        )
        _physical_prediction(archive, protocol=protocol)
        candidate["resolved_physical_archive"] = str(archive)
        records.append(candidate)
    return records


def _score_prediction_splits(
    arrays: Mapping[str, npt.NDArray[Any]],
    outcomes: Mapping[str, npt.NDArray[Any]],
    *,
    protocol: SourceProtocol,
) -> dict[str, dict[str, float]]:
    prediction = np.asarray(arrays["prediction_m"])[:, : protocol.observed_count]
    outcome = np.asarray(outcomes["object_points_m"])
    valid = np.asarray(outcomes["valid_mask"], dtype=bool)
    fit_start, fit_stop = protocol.fit_range
    validation_start, validation_stop = protocol.validation_range
    return {
        "fit": _split_metrics(
            prediction,
            outcome,
            valid,
            local_start=fit_start,
            local_stop=fit_stop,
        ),
        "validation": _split_metrics(
            prediction,
            outcome,
            valid,
            local_start=validation_start,
            local_stop=validation_stop,
        ),
    }


def _balanced_ratio(
    metrics: Mapping[str, float],
    persistence: Mapping[str, float],
) -> float:
    return float(
        0.5
        * (
            metrics["identity_coordinate_rmse_m"]
            / persistence["identity_coordinate_rmse_m"]
            + metrics["symmetric_chamfer_m"] / persistence["symmetric_chamfer_m"]
        )
    )


def _validated_metric_block(
    value: object,
    *,
    name: str,
    include_ratio: bool,
) -> Mapping[str, float]:
    block = _mapping(value, name=name)
    expected = {
        "identity_coordinate_rmse_m",
        "symmetric_chamfer_m",
    }
    if include_ratio:
        expected.add("balanced_ratio_vs_persistence")
    require_exact_fields(block, expected=frozenset(expected), name=name)
    normalized: dict[str, float] = {}
    for field in sorted(expected):
        metric = _finite_number(block.get(field), name=f"{name}.{field}")
        _require(metric >= 0.0, f"{name}.{field} is negative")
        normalized[field] = metric
    return normalized


def _validated_split_metrics(
    value: object,
    *,
    name: str,
    include_ratio: bool,
) -> Mapping[str, Mapping[str, float]]:
    splits = _mapping(value, name=name)
    require_exact_fields(
        splits,
        expected=frozenset({"fit", "validation"}),
        name=name,
    )
    return {
        split: _validated_metric_block(
            splits.get(split),
            name=f"{name}.{split}",
            include_ratio=include_ratio,
        )
        for split in ("fit", "validation")
    }


def _public_candidate_records(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in candidate.items()
            if key != "resolved_physical_archive"
        }
        for candidate in candidates
    ]


def _validate_prefix_result(
    prefix: Mapping[str, Any],
    *,
    protocol: SourceProtocol,
    custody_path: Path,
    grid_path: Path,
    grid: Mapping[str, Any],
    selected_path: Path,
    incumbent_path: Path,
) -> None:
    """Re-derive every frozen authorization check before future access."""

    require_exact_fields(prefix, expected=PREFIX_RESULT_FIELDS, name="prefix result")
    _require(
        prefix.get("schema") == PREFIX_RESULT_SCHEMA, "prefix result schema changed"
    )
    _require(prefix.get("schema_version") == 1, "prefix result version changed")
    _require(
        prefix.get("protocol_id") == protocol.protocol_id, "prefix protocol ID changed"
    )
    _require(
        prefix.get("protocol_sha256") == protocol.sha256, "prefix protocol changed"
    )
    _require(
        prefix.get("source_custody_sha256") == file_sha256(custody_path),
        "prefix source custody changed",
    )
    _require(
        prefix.get("grid_manifest_sha256") == file_sha256(grid_path),
        "prefix grid manifest changed",
    )
    identity = dict(prefix)
    result_id = identity.pop("result_id")
    _require(result_id == content_id(identity), "prefix result content ID changed")

    boundary = _mapping(prefix.get("information_boundary"), name="prefix boundary")
    require_exact_fields(
        boundary,
        expected=frozenset(
            {
                "prefix_outcomes_read",
                "future_outcomes_read",
                "target_or_held_out_artifact_read",
            }
        ),
        name="prefix boundary",
    )
    _require(
        boundary.get("prefix_outcomes_read") is True
        and boundary.get("future_outcomes_read") is False
        and boundary.get("target_or_held_out_artifact_read") is False,
        "prefix result crossed its information boundary",
    )

    comparators = _mapping(prefix.get("comparators"), name="prefix comparators")
    require_exact_fields(
        comparators,
        expected=frozenset({"persistence", "incumbent", "matphys"}),
        name="prefix comparators",
    )
    comparator_metrics = {
        name: _validated_split_metrics(
            comparators.get(name),
            name=f"prefix comparators.{name}",
            include_ratio=False,
        )
        for name in ("persistence", "incumbent", "matphys")
    }
    for split in ("fit", "validation"):
        for field in ("identity_coordinate_rmse_m", "symmetric_chamfer_m"):
            _require(
                comparator_metrics["persistence"][split][field] > 0.0,
                "persistence metric cannot define a finite comparison ratio",
            )

    candidate_values = prefix.get("candidates")
    if not isinstance(candidate_values, list):
        raise ValueError("prefix candidates must be a list")
    grid_candidates = cast(list[Mapping[str, Any]], grid["candidates"])
    _require(
        len(candidate_values) == len(grid_candidates),
        "prefix candidate denominator changed",
    )
    successful: list[Mapping[str, Any]] = []
    for index, (candidate_value, grid_candidate) in enumerate(
        zip(candidate_values, grid_candidates, strict=True)
    ):
        candidate = _mapping(candidate_value, name=f"prefix candidates[{index}]")
        if grid_candidate["status"] == "success":
            require_exact_fields(
                candidate,
                expected=GRID_SUCCESS_FIELDS | {"metrics"},
                name=f"prefix candidates[{index}]",
            )
            for field in GRID_SUCCESS_FIELDS:
                _require(
                    candidate.get(field) == grid_candidate.get(field),
                    f"prefix candidate {index} differs from its sealed grid record",
                )
            metrics = _validated_split_metrics(
                candidate.get("metrics"),
                name=f"prefix candidates[{index}].metrics",
                include_ratio=True,
            )
            for split in ("fit", "validation"):
                expected_ratio = _balanced_ratio(
                    metrics[split],
                    comparator_metrics["persistence"][split],
                )
                _require(
                    np.isclose(
                        metrics[split]["balanced_ratio_vs_persistence"],
                        expected_ratio,
                        atol=1.0e-15,
                        rtol=0.0,
                    ),
                    "stored balanced ratio differs from the frozen metric definition",
                )
            successful.append(candidate)
        else:
            require_exact_fields(
                candidate,
                expected=GRID_FAILURE_FIELDS,
                name=f"prefix candidates[{index}]",
            )
            _require(
                dict(candidate) == dict(grid_candidate),
                f"prefix failure {index} differs from its sealed grid record",
            )

    selection = cast(Mapping[str, Any], protocol.value["selection"])
    required = int(selection["required_successful_candidates"])
    _require(
        prefix.get("successful_candidate_count") == len(successful)
        and prefix.get("required_successful_candidate_count") == required,
        "prefix candidate counts changed",
    )
    selected: Mapping[str, Any] | None = None
    if len(successful) == required:
        selected = min(
            successful,
            key=lambda result: (
                float(
                    cast(Mapping[str, Any], result["metrics"])["fit"][
                        "balanced_ratio_vs_persistence"
                    ]
                ),
                float(result["young_modulus_pa"]),
                float(result["damping"]),
            ),
        )
    expected_selected_index = (
        int(selected["candidate_index"]) if selected is not None else None
    )
    _require(
        prefix.get("selected_candidate_index") == expected_selected_index,
        "prefix selected candidate changed",
    )
    spread = _finite_number(
        prefix.get("final_ensemble_spread_m"),
        name="prefix final_ensemble_spread_m",
    )
    _require(
        spread == float(grid["final_ensemble_spread_m"]),
        "prefix ensemble spread differs from the sealed grid",
    )
    gates = cast(Mapping[str, Any], selection["validation_gates"])
    derived: dict[str, bool] = {
        "complete_candidate_denominator": len(successful) == required,
        "finite_ensemble_spread": bool(np.isfinite(spread)),
        "minimum_ensemble_spread": spread
        >= float(gates["minimum_final_ensemble_spread_m"]),
        "maximum_ensemble_spread": spread
        <= float(gates["maximum_final_ensemble_spread_m"]),
    }
    if selected is None:
        derived.update(
            {
                "balanced_validation_improvement": False,
                "identity_nonregression_vs_incumbent": False,
                "chamfer_nonregression_vs_incumbent": False,
                "zero_action_drift": False,
                "deterministic_replay": False,
            }
        )
    else:
        metrics = _validated_split_metrics(
            selected["metrics"],
            name="selected prefix metrics",
            include_ratio=True,
        )
        validation = metrics["validation"]
        incumbent_validation = comparator_metrics["incumbent"]["validation"]
        _require(
            incumbent_validation["identity_coordinate_rmse_m"] > 0.0
            and incumbent_validation["symmetric_chamfer_m"] > 0.0,
            "incumbent validation metric cannot define a ratio",
        )
        derived.update(
            {
                "balanced_validation_improvement": validation[
                    "balanced_ratio_vs_persistence"
                ]
                <= float(gates["maximum_balanced_ratio_vs_persistence"]),
                "identity_nonregression_vs_incumbent": validation[
                    "identity_coordinate_rmse_m"
                ]
                / incumbent_validation["identity_coordinate_rmse_m"]
                <= float(gates["maximum_identity_rmse_ratio_vs_incumbent"]),
                "chamfer_nonregression_vs_incumbent": validation["symmetric_chamfer_m"]
                / incumbent_validation["symmetric_chamfer_m"]
                <= float(gates["maximum_chamfer_ratio_vs_incumbent"]),
                "zero_action_drift": float(selected["maximum_zero_action_drift_m"])
                <= float(gates["maximum_zero_action_drift_m"]),
                "deterministic_replay": float(selected["replay_coordinate_rmse_m"])
                <= float(gates["maximum_replay_coordinate_rmse_m"]),
            }
        )
    checks = _mapping(prefix.get("validation_checks"), name="validation checks")
    require_exact_fields(
        checks,
        expected=VALIDATION_CHECK_FIELDS,
        name="validation checks",
    )
    _require(dict(checks) == derived, "prefix validation checks were not re-derived")
    gate_passed = all(derived.values())
    _require(
        prefix.get("validation_gate_passed") is gate_passed
        and prefix.get("future_scoring_authorized") is gate_passed,
        "prefix future authorization differs from the frozen gate",
    )
    expected_selection = "newton_mpm" if gate_passed else "exact_incumbent_fallback"
    _require(prefix.get("selection") == expected_selection, "prefix selection changed")
    _require(
        prefix.get("selected_physical_is_byte_exact_source") is True,
        "prefix selected archive lacks byte-exact provenance",
    )
    sha256_digest(
        prefix.get("selected_physical_sha256"),
        name="selected_physical_sha256",
    )
    _require(
        file_sha256(selected_path) == prefix.get("selected_physical_sha256"),
        "selected physical prediction changed",
    )
    expected_source = (
        _ordinary_file(
            grid_path.parent
            / _canonical_relative_path(
                selected["physical_archive"],
                name="selected grid physical archive",
            ),
            name="selected grid physical archive",
        )
        if gate_passed and selected is not None
        else incumbent_path
    )
    _require(
        selected_path.read_bytes() == expected_source.read_bytes(),
        "selected physical archive is not the exact gated source",
    )


def score_prefix_gate(
    *,
    protocol_path: str | Path,
    source_bundle_dir: str | Path,
    grid_manifest_path: str | Path,
    incumbent_physical_path: str | Path,
    matphys_physical_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Score fit/validation only, choose one MPM arm, and apply exact fallback."""

    protocol = load_source_protocol(protocol_path)
    source_root = Path(source_bundle_dir).absolute()
    custody_path = _ordinary_file(
        source_root / SOURCE_CUSTODY_FILENAME, name="source custody"
    )
    custody = load_source_custody(custody_path, protocol=protocol)
    artifacts = _mapping(custody.get("artifacts"), name="custody artifacts")
    prefix_path = _ordinary_file(
        source_root / PREFIX_OUTCOME_FILENAME, name="prefix outcomes"
    )
    prefix_record = _mapping(
        artifacts.get(PREFIX_OUTCOME_FILENAME), name="prefix record"
    )
    _require(
        file_sha256(prefix_path) == prefix_record.get("sha256"),
        "prefix outcomes changed",
    )
    prefix_stop = protocol.validation_range[1]
    outcomes = _load_outcomes(
        prefix_path,
        expected_indices=np.arange(prefix_stop, dtype=np.int32),
        observed_count=protocol.observed_count,
    )
    incumbent_path = _pinned_file(
        protocol,
        incumbent_physical_path,
        key="incumbent_physical",
    )
    matphys_path = _pinned_file(
        protocol,
        matphys_physical_path,
        key="matphys_physical",
    )
    incumbent = _physical_prediction(incumbent_path, protocol=protocol)
    matphys = _physical_prediction(matphys_path, protocol=protocol)
    persistence = dict(incumbent)
    persistence["prediction_m"] = incumbent["persistence_m"]
    comparator_arrays = {
        "persistence": persistence,
        "incumbent": incumbent,
        "matphys": matphys,
    }
    comparator_metrics = {
        name: _score_prediction_splits(arrays, outcomes, protocol=protocol)
        for name, arrays in comparator_arrays.items()
    }
    grid_path = _ordinary_file(grid_manifest_path, name="Newton grid manifest")
    grid = load_grid_manifest(grid_path, protocol=protocol)
    source_input_record = _mapping(
        artifacts.get(SOURCE_INPUT_FILENAME),
        name="source input record",
    )
    _require(
        grid.get("source_inputs_sha256") == source_input_record.get("sha256"),
        "prediction grid is not bound to the prepared source input",
    )
    candidates = _candidate_records(
        grid,
        grid_root=grid_path.parent,
        protocol=protocol,
    )
    candidate_results: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    for candidate in candidates:
        result = dict(candidate)
        if result.get("status") == "success":
            arrays = _physical_prediction(
                cast(str, result["resolved_physical_archive"]),
                protocol=protocol,
            )
            metrics = _score_prediction_splits(arrays, outcomes, protocol=protocol)
            metrics["fit"]["balanced_ratio_vs_persistence"] = _balanced_ratio(
                metrics["fit"],
                comparator_metrics["persistence"]["fit"],
            )
            metrics["validation"]["balanced_ratio_vs_persistence"] = _balanced_ratio(
                metrics["validation"],
                comparator_metrics["persistence"]["validation"],
            )
            result["metrics"] = metrics
            successful.append(result)
        candidate_results.append(result)
    selection = cast(Mapping[str, Any], protocol.value["selection"])
    required = int(selection["required_successful_candidates"])
    selected: dict[str, Any] | None = None
    if len(successful) == required:
        selected = min(
            successful,
            key=lambda result: (
                float(
                    cast(Mapping[str, Any], result["metrics"])["fit"][
                        "balanced_ratio_vs_persistence"
                    ]
                ),
                float(result["young_modulus_pa"]),
                float(result["damping"]),
            ),
        )
    ensemble_spread = float(grid.get("final_ensemble_spread_m", float("nan")))
    gates = cast(Mapping[str, Any], selection["validation_gates"])
    checks: dict[str, bool] = {
        "complete_candidate_denominator": len(successful) == required,
        "finite_ensemble_spread": bool(np.isfinite(ensemble_spread)),
        "minimum_ensemble_spread": ensemble_spread
        >= float(gates["minimum_final_ensemble_spread_m"]),
        "maximum_ensemble_spread": ensemble_spread
        <= float(gates["maximum_final_ensemble_spread_m"]),
    }
    if selected is None:
        checks.update(
            {
                "balanced_validation_improvement": False,
                "identity_nonregression_vs_incumbent": False,
                "chamfer_nonregression_vs_incumbent": False,
                "zero_action_drift": False,
                "deterministic_replay": False,
            }
        )
    else:
        selected_metrics = cast(
            Mapping[str, Mapping[str, float]],
            selected["metrics"],
        )
        validation = selected_metrics["validation"]
        incumbent_validation = comparator_metrics["incumbent"]["validation"]
        checks.update(
            {
                "balanced_validation_improvement": validation[
                    "balanced_ratio_vs_persistence"
                ]
                <= float(gates["maximum_balanced_ratio_vs_persistence"]),
                "identity_nonregression_vs_incumbent": validation[
                    "identity_coordinate_rmse_m"
                ]
                / incumbent_validation["identity_coordinate_rmse_m"]
                <= float(gates["maximum_identity_rmse_ratio_vs_incumbent"]),
                "chamfer_nonregression_vs_incumbent": validation["symmetric_chamfer_m"]
                / incumbent_validation["symmetric_chamfer_m"]
                <= float(gates["maximum_chamfer_ratio_vs_incumbent"]),
                "zero_action_drift": float(selected["maximum_zero_action_drift_m"])
                <= float(gates["maximum_zero_action_drift_m"]),
                "deterministic_replay": float(selected["replay_coordinate_rmse_m"])
                <= float(gates["maximum_replay_coordinate_rmse_m"]),
            }
        )
    gate_passed = all(checks.values())
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected_source = (
        Path(cast(str, selected["resolved_physical_archive"]))
        if gate_passed and selected is not None
        else incumbent_path
    )
    selected_output = output / SELECTED_PHYSICAL_FILENAME
    shutil.copyfile(selected_source, selected_output)
    _require(
        selected_output.read_bytes() == selected_source.read_bytes(),
        "selected physical archive is not byte exact",
    )
    identity: dict[str, Any] = {
        "schema": PREFIX_RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "source_custody_sha256": file_sha256(custody_path),
        "grid_manifest_sha256": file_sha256(grid_path),
        "information_boundary": {
            "prefix_outcomes_read": True,
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
        "comparators": comparator_metrics,
        "candidates": _public_candidate_records(candidate_results),
        "successful_candidate_count": len(successful),
        "required_successful_candidate_count": required,
        "selected_candidate_index": (
            int(selected["candidate_index"]) if selected is not None else None
        ),
        "final_ensemble_spread_m": ensemble_spread,
        "validation_checks": checks,
        "validation_gate_passed": gate_passed,
        "selection": "newton_mpm" if gate_passed else "exact_incumbent_fallback",
        "selected_physical_sha256": file_sha256(selected_output),
        "selected_physical_is_byte_exact_source": True,
        "future_scoring_authorized": gate_passed,
    }
    result = {**identity, "result_id": content_id(identity)}
    write_atomic_json(result, output / PREFIX_RESULT_FILENAME, overwrite=False)
    return result


def score_future_if_authorized(
    *,
    protocol_path: str | Path,
    source_bundle_dir: str | Path,
    prefix_result_dir: str | Path,
    grid_manifest_path: str | Path,
    incumbent_physical_path: str | Path,
    matphys_physical_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Open and score the source future only after the frozen validation gate."""

    protocol = load_source_protocol(protocol_path)
    prefix_root = Path(prefix_result_dir).absolute()
    prefix_path = _ordinary_file(
        prefix_root / PREFIX_RESULT_FILENAME, name="prefix result"
    )
    prefix = load_strict_json_object(prefix_path, label="prefix result")
    source_root = Path(source_bundle_dir).absolute()
    custody_path = _ordinary_file(
        source_root / SOURCE_CUSTODY_FILENAME,
        name="source custody",
    )
    custody = load_source_custody(custody_path, protocol=protocol)
    grid_path = _ordinary_file(grid_manifest_path, name="Newton grid manifest")
    grid = load_grid_manifest(grid_path, protocol=protocol)
    incumbent_path = _pinned_file(
        protocol,
        incumbent_physical_path,
        key="incumbent_physical",
    )
    matphys_path = _pinned_file(
        protocol,
        matphys_physical_path,
        key="matphys_physical",
    )
    selected_path = _ordinary_file(
        prefix_root / SELECTED_PHYSICAL_FILENAME,
        name="selected physical prediction",
    )
    _validate_prefix_result(
        prefix,
        protocol=protocol,
        custody_path=custody_path,
        grid_path=grid_path,
        grid=grid,
        selected_path=selected_path,
        incumbent_path=incumbent_path,
    )
    if prefix.get("future_scoring_authorized") is not True:
        identity: dict[str, Any] = {
            "schema": FUTURE_RESULT_SCHEMA,
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "prefix_result_sha256": file_sha256(prefix_path),
            "status": "future-not-opened-validation-gate-failed",
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        }
        result = {**identity, "result_id": content_id(identity)}
        write_atomic_json(result, output_path, overwrite=False)
        return result

    artifacts = _mapping(custody.get("artifacts"), name="custody artifacts")
    future_path = _ordinary_file(
        source_root / FUTURE_OUTCOME_FILENAME, name="future outcomes"
    )
    future_record = _mapping(
        artifacts.get(FUTURE_OUTCOME_FILENAME), name="future record"
    )
    _require(
        file_sha256(future_path) == future_record.get("sha256"),
        "future outcomes changed",
    )
    future_start, future_stop = protocol.future_range
    outcomes = _load_outcomes(
        future_path,
        expected_indices=np.arange(future_start, future_stop, dtype=np.int32),
        observed_count=protocol.observed_count,
    )
    arrays = {
        "selected": _physical_prediction(selected_path, protocol=protocol),
        "incumbent": _physical_prediction(incumbent_path, protocol=protocol),
        "matphys": _physical_prediction(matphys_path, protocol=protocol),
    }
    persistence = dict(arrays["incumbent"])
    persistence["prediction_m"] = arrays["incumbent"]["persistence_m"]
    arrays["persistence"] = persistence
    local_stop = future_stop - future_start
    metrics: dict[str, Any] = {}
    for name, physical in arrays.items():
        prediction = np.asarray(physical["prediction_m"])[
            future_start:future_stop, : protocol.observed_count
        ]
        metrics[name] = _split_metrics(
            prediction,
            np.asarray(outcomes["object_points_m"]),
            np.asarray(outcomes["valid_mask"], dtype=bool),
            local_start=0,
            local_stop=local_stop,
        )
    identity = {
        "schema": FUTURE_RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "prefix_result_sha256": file_sha256(prefix_path),
        "status": "source-future-scored-after-passing-gate",
        "future_outcomes_read": True,
        "target_or_held_out_artifact_read": False,
        "metrics": metrics,
    }
    result = {**identity, "result_id": content_id(identity)}
    write_atomic_json(result, output_path, overwrite=False)
    return result


__all__ = [
    "FUTURE_RESULT_FILENAME",
    "GRID_MANIFEST_FILENAME",
    "PREFIX_RESULT_FILENAME",
    "SELECTED_PHYSICAL_FILENAME",
    "SOURCE_CUSTODY_FILENAME",
    "SOURCE_INPUT_FILENAME",
    "SourceProtocol",
    "load_grid_manifest",
    "load_source_custody",
    "load_source_inputs",
    "load_source_protocol",
    "prepare_source_case",
    "score_future_if_authorized",
    "score_prefix_gate",
]
