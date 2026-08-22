"""Frozen source-only qualification for the keyed-Dirichlet SOFA FEM v2 arm."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._portable_contracts import content_id, load_strict_json_object, write_atomic_json
from .jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
    attachment_targets_m,
    rigid_contact_projection_v1,
    rigid_transform_v1,
)
from .material_backend_qualification_v1 import (
    MaterialBackendQualificationV1,
    save_material_backend_qualification_v1,
)
from .material_backend_v1 import BackendTransportV1, resolve_material_backend_profile
from .native_tet_fem_source_v1 import prepare_native_tet_source_geometry_v1
from .physical_rollout_v1 import write_deterministic_npz
from .sofa_fem_kinematic_source_v2 import (
    ATTACHMENT_MODEL,
    BACKEND_VARIANT,
    CONTINUATION_POLICY,
    SofaKinematicSourceReplayV2,
    run_sofa_fem_kinematic_source_replay_v2,
)
from .sofa_fem_source_v1 import (
    CONSTITUTIVE_MODEL,
    SOFA_ARCHIVE_FILENAME,
    SOFA_ARCHIVE_SHA256,
    SOFA_REPOSITORY,
    SOFA_REVISION,
    SOFA_VERSION,
    NativeSofaFemModulesV1,
    load_native_sofa_fem_modules_v1,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

PROTOCOL_SCHEMA: Final = "bayesian-phystwin.sofa-fem-source-physics-protocol-v2"
RESULT_SCHEMA: Final = "bayesian-phystwin.sofa-fem-source-physics-result-v2"
RESULT_FILENAME: Final = "sofa-fem-source-physics-result-v2.json"
QUALIFICATION_FILENAME: Final = "material-backend-qualification.json"
GROUP_ARCHIVE_FILENAME: Final = "sofa-fem-source-physics-trajectories-v2.npz"
FALLBACK_FILENAME: Final = "exact-incumbent-fallback.npz"

_PROTOCOL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_label",
        "claim_boundary",
        "backend",
        "source_groups",
        "simulation",
        "gates",
        "information_boundary",
    }
)
_BACKEND_FIELDS: Final = frozenset(
    {
        "canonical_profile_id",
        "producer_profile_id",
        "transport",
        "backend_variant",
        "engine_repository",
        "engine_revision",
        "engine_version",
        "distribution_archive_filename",
        "distribution_archive_sha256",
        "native_smoke_artifact_sha256",
        "native_smoke_id",
        "runtime_id",
    }
)
_GROUP_FIELDS: Final = frozenset(
    {
        "group_id",
        "source_inputs_relative_path",
        "source_inputs_sha256",
        "prepared_archive_relative_path",
        "prepared_archive_sha256",
        "incumbent_relative_path",
        "incumbent_sha256",
        "frame_count",
        "material_node_count",
        "controller_point_count",
        "attached_node_count",
        "tetrahedron_count",
        "expected_contact_patch_sizes",
    }
)
_SIMULATION_FIELDS: Final = frozenset(
    {
        "backend",
        "precision",
        "seed",
        "fps",
        "qualification_frame_count",
        "base_interval_substeps",
        "refined_interval_substeps",
        "element_type",
        "constitutive_model",
        "young_modulus_pa",
        "young_modulus_probe_low_pa",
        "young_modulus_probe_high_pa",
        "poisson_ratio",
        "density_kg_m3",
        "rayleigh_stiffness",
        "rayleigh_mass",
        "base_mesh_max_edge_m",
        "minimum_tetrahedron_shape_ratio",
        "contact_cluster_radius_m",
        "attachment_model",
        "continuation_policy",
        "rigid_transform_rotation_axis",
        "rigid_transform_angle_rad",
        "rigid_transform_translation_m",
        "hard_minimum_deformation_determinant",
    }
)
_GATE_FIELDS: Final = frozenset(
    {
        "maximum_zero_action_drift_m",
        "maximum_rigid_equivariance_error_m",
        "maximum_time_step_refinement_absolute_error_m",
        "maximum_time_step_refinement_relative_error",
        "maximum_source_query_parity_rmse_m",
        "minimum_action_response_m",
        "minimum_material_sensitivity_m",
        "maximum_material_sensitivity_m",
        "maximum_attachment_error_m",
        "maximum_node_displacement_m",
        "minimum_deformation_determinant",
        "maximum_deformation_determinant",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "frame_zero_geometry_allowed",
        "known_full_controller_trajectory_allowed",
        "incumbent_bytes_allowed_for_exact_fallback",
        "incumbent_prediction_arrays_allowed",
        "source_object_outcomes_allowed",
        "target_or_held_out_artifact_access_allowed",
        "future_scoring_authorized",
        "no_replacement",
    }
)
_SOURCE_INPUT_ARRAYS: Final = frozenset(
    {
        "frame_zero_points_m",
        "controller_points_m",
        "attachment_indices",
        "attachment_weights",
        "action_support",
    }
)
_PREPARED_ARRAYS: Final = frozenset(
    {
        "points",
        "cells",
        "attachment_indices",
        "projected_targets",
        "rotations",
        "translations",
        "patch_flat",
        "patch_offsets",
        "patch_ranks",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical nonempty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _canonical_string(value, name=name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _git_revision(value: object, *, name: str) -> str:
    text = _canonical_string(value, name=name)
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a full lowercase Git revision")
    return text


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _vector3(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element list")
    return cast(
        tuple[float, float, float],
        tuple(
            _finite(item, name=f"{name}[{index}]") for index, item in enumerate(value)
        ),
    )


def _positive_int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty integer list")
    return tuple(_positive_int(item, name=f"{name}[]") for item in value)


def _canonical_relative_path(value: object, *, name: str) -> PurePosixPath:
    text = _canonical_string(value, name=name)
    path = PurePosixPath(text)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require("\\" not in text, f"{name} must use POSIX separators")
    _require(
        all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} is not canonical",
    )
    _require(path.as_posix() == text, f"{name} is not canonical")
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SofaSourceGroupV2:
    group_id: str
    source_inputs_relative_path: PurePosixPath
    source_inputs_sha256: str
    prepared_archive_relative_path: PurePosixPath
    prepared_archive_sha256: str
    incumbent_relative_path: PurePosixPath
    incumbent_sha256: str
    frame_count: int
    material_node_count: int
    controller_point_count: int
    attached_node_count: int
    tetrahedron_count: int
    expected_contact_patch_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SofaSourcePhysicsProtocolV2:
    value: Mapping[str, Any]
    protocol_sha256: str
    canonical_profile_id: str
    producer_profile_id: str
    transport: BackendTransportV1
    runtime_id: str
    source_groups: tuple[SofaSourceGroupV2, ...]

    @property
    def backend(self) -> Mapping[str, Any]:
        return _mapping(self.value["backend"], name="backend")

    @property
    def simulation(self) -> Mapping[str, Any]:
        return _mapping(self.value["simulation"], name="simulation")

    @property
    def gates(self) -> Mapping[str, Any]:
        return _mapping(self.value["gates"], name="gates")


@dataclass(frozen=True, slots=True)
class PreparedSofaSourceV2:
    points_m: FloatArray
    cells: npt.NDArray[np.int32]
    attachment_indices: IntArray
    contact: RigidContactProjectionV1


def load_sofa_source_physics_protocol_v2(
    path: str | Path,
) -> SofaSourcePhysicsProtocolV2:
    source = Path(path)
    value = load_strict_json_object(source, label="SOFA source-physics protocol v2")
    _exact_fields(value, _PROTOCOL_FIELDS, "protocol")
    _require(value["schema"] == PROTOCOL_SCHEMA, "protocol schema changed")
    _require(value["schema_version"] == 2, "protocol version changed")
    _canonical_string(value["protocol_label"], name="protocol_label")
    _canonical_string(value["claim_boundary"], name="claim_boundary")

    backend = _mapping(value["backend"], name="backend")
    _exact_fields(backend, _BACKEND_FIELDS, "backend")
    canonical_profile_id = _canonical_string(
        backend["canonical_profile_id"], name="canonical_profile_id"
    )
    producer_profile_id = _canonical_string(
        backend["producer_profile_id"], name="producer_profile_id"
    )
    resolved = resolve_material_backend_profile(producer_profile_id)
    _require(
        resolved.profile_id == canonical_profile_id,
        "backend profile family changed",
    )
    _require(backend["transport"] == resolved.transport, "backend transport changed")
    _require(backend["backend_variant"] == BACKEND_VARIANT, "backend variant changed")
    _require(backend["engine_repository"] == SOFA_REPOSITORY, "repository changed")
    _require(
        _git_revision(backend["engine_revision"], name="engine_revision")
        == SOFA_REVISION,
        "SOFA revision changed",
    )
    _require(backend["engine_version"] == SOFA_VERSION, "SOFA version changed")
    _require(
        backend["distribution_archive_filename"] == SOFA_ARCHIVE_FILENAME,
        "SOFA archive filename changed",
    )
    _require(
        _sha256(
            backend["distribution_archive_sha256"],
            name="distribution_archive_sha256",
        )
        == SOFA_ARCHIVE_SHA256,
        "SOFA archive digest changed",
    )
    _sha256(
        backend["native_smoke_artifact_sha256"],
        name="native_smoke_artifact_sha256",
    )
    _sha256(backend["native_smoke_id"], name="native_smoke_id")
    runtime_id = _sha256(backend["runtime_id"], name="runtime_id")

    raw_groups = value["source_groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) < 2:
        raise ValueError("at least two source groups are required")
    groups: list[SofaSourceGroupV2] = []
    for index, raw in enumerate(raw_groups):
        group = _mapping(raw, name=f"source_groups[{index}]")
        _exact_fields(group, _GROUP_FIELDS, f"source_groups[{index}]")
        patch_sizes = _positive_int_tuple(
            group["expected_contact_patch_sizes"],
            name="expected_contact_patch_sizes",
        )
        attached_count = _positive_int(
            group["attached_node_count"], name="attached_node_count"
        )
        _require(
            sum(patch_sizes) == attached_count,
            "contact patches do not cover attached nodes",
        )
        groups.append(
            SofaSourceGroupV2(
                group_id=_canonical_string(group["group_id"], name="group_id"),
                source_inputs_relative_path=_canonical_relative_path(
                    group["source_inputs_relative_path"],
                    name="source_inputs_relative_path",
                ),
                source_inputs_sha256=_sha256(
                    group["source_inputs_sha256"], name="source_inputs_sha256"
                ),
                prepared_archive_relative_path=_canonical_relative_path(
                    group["prepared_archive_relative_path"],
                    name="prepared_archive_relative_path",
                ),
                prepared_archive_sha256=_sha256(
                    group["prepared_archive_sha256"],
                    name="prepared_archive_sha256",
                ),
                incumbent_relative_path=_canonical_relative_path(
                    group["incumbent_relative_path"], name="incumbent_relative_path"
                ),
                incumbent_sha256=_sha256(
                    group["incumbent_sha256"], name="incumbent_sha256"
                ),
                frame_count=_positive_int(group["frame_count"], name="frame_count"),
                material_node_count=_positive_int(
                    group["material_node_count"], name="material_node_count"
                ),
                controller_point_count=_positive_int(
                    group["controller_point_count"], name="controller_point_count"
                ),
                attached_node_count=attached_count,
                tetrahedron_count=_positive_int(
                    group["tetrahedron_count"], name="tetrahedron_count"
                ),
                expected_contact_patch_sizes=patch_sizes,
            )
        )
    _require(
        len({group.group_id for group in groups}) == len(groups),
        "source group IDs must be unique",
    )

    simulation = _mapping(value["simulation"], name="simulation")
    _exact_fields(simulation, _SIMULATION_FIELDS, "simulation")
    _require(simulation["backend"] == "cpu", "qualification backend changed")
    _require(simulation["precision"] == "64", "qualification precision changed")
    _nonnegative_int(simulation["seed"], name="seed")
    for name in (
        "fps",
        "young_modulus_pa",
        "young_modulus_probe_low_pa",
        "young_modulus_probe_high_pa",
        "density_kg_m3",
        "base_mesh_max_edge_m",
        "minimum_tetrahedron_shape_ratio",
        "contact_cluster_radius_m",
        "rigid_transform_angle_rad",
        "hard_minimum_deformation_determinant",
    ):
        _finite(simulation[name], name=name, positive=True)
    for name in ("rayleigh_stiffness", "rayleigh_mass"):
        _require(
            _finite(simulation[name], name=name) >= 0.0,
            f"{name} must be nonnegative",
        )
    qualification_frames = _positive_int(
        simulation["qualification_frame_count"],
        name="qualification_frame_count",
    )
    base_substeps = _positive_int(
        simulation["base_interval_substeps"], name="base_interval_substeps"
    )
    refined_substeps = _positive_int(
        simulation["refined_interval_substeps"], name="refined_interval_substeps"
    )
    _require(qualification_frames == 2, "qualification horizon changed")
    _require(
        refined_substeps == 2 * base_substeps,
        "refinement must exactly halve the integrator time step",
    )
    _require(simulation["element_type"] == "TET4", "element type changed")
    _require(
        simulation["constitutive_model"] == CONSTITUTIVE_MODEL,
        "constitutive model changed",
    )
    poisson = _finite(simulation["poisson_ratio"], name="poisson_ratio")
    _require(-1.0 < poisson < 0.5, "poisson_ratio is invalid")
    _require(
        float(simulation["young_modulus_probe_low_pa"])
        < float(simulation["young_modulus_pa"])
        < float(simulation["young_modulus_probe_high_pa"]),
        "Young's modulus probe order changed",
    )
    _require(
        simulation["attachment_model"] == ATTACHMENT_MODEL,
        "attachment model changed",
    )
    _require(
        simulation["continuation_policy"] == CONTINUATION_POLICY,
        "continuation policy changed",
    )
    axis = np.asarray(
        _vector3(
            simulation["rigid_transform_rotation_axis"],
            name="rigid_transform_rotation_axis",
        )
    )
    _require(np.linalg.norm(axis) > 0.0, "rigid transform axis is zero")
    _vector3(
        simulation["rigid_transform_translation_m"],
        name="rigid_transform_translation_m",
    )

    gates = _mapping(value["gates"], name="gates")
    _exact_fields(gates, _GATE_FIELDS, "gates")
    for name in _GATE_FIELDS:
        _finite(gates[name], name=name, positive=True)
    _require(
        float(gates["minimum_material_sensitivity_m"])
        < float(gates["maximum_material_sensitivity_m"]),
        "material sensitivity interval is empty",
    )
    _require(
        float(gates["minimum_deformation_determinant"])
        < float(gates["maximum_deformation_determinant"]),
        "deformation determinant interval is empty",
    )
    _require(
        float(simulation["hard_minimum_deformation_determinant"])
        <= float(gates["minimum_deformation_determinant"]),
        "hard determinant floor exceeds the reporting gate",
    )

    boundary = _mapping(value["information_boundary"], name="information_boundary")
    _exact_fields(boundary, _BOUNDARY_FIELDS, "information_boundary")
    _require(
        boundary
        == {
            "frame_zero_geometry_allowed": True,
            "known_full_controller_trajectory_allowed": True,
            "incumbent_bytes_allowed_for_exact_fallback": True,
            "incumbent_prediction_arrays_allowed": False,
            "source_object_outcomes_allowed": False,
            "target_or_held_out_artifact_access_allowed": False,
            "future_scoring_authorized": False,
            "no_replacement": True,
        },
        "information boundary changed",
    )
    return SofaSourcePhysicsProtocolV2(
        value=value,
        protocol_sha256=file_sha256(source),
        canonical_profile_id=canonical_profile_id,
        producer_profile_id=producer_profile_id,
        transport=resolved.transport,
        runtime_id=runtime_id,
        source_groups=tuple(groups),
    )


def load_sofa_source_inputs_v2(
    path: str | Path,
    *,
    group: SofaSourceGroupV2,
) -> dict[str, npt.NDArray[Any]]:
    source = Path(path)
    _require(
        source.is_file() and not source.is_symlink(),
        "source inputs must be an ordinary file",
    )
    _require(
        file_sha256(source) == group.source_inputs_sha256,
        "source input SHA-256 changed",
    )
    with np.load(source, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    _require(frozenset(arrays) == _SOURCE_INPUT_ARRAYS, "source input roster changed")
    points = arrays["frame_zero_points_m"]
    controller = arrays["controller_points_m"]
    indices = arrays["attachment_indices"]
    weights = arrays["attachment_weights"]
    support = arrays["action_support"]
    _require(
        points.shape == (group.material_node_count, 3)
        and points.dtype == np.float32
        and np.all(np.isfinite(points)),
        "frame-zero source geometry changed",
    )
    _require(
        controller.shape == (group.frame_count, group.controller_point_count, 3)
        and controller.dtype == np.float32
        and np.all(np.isfinite(controller)),
        "controller source trajectory changed",
    )
    _require(
        indices.shape == (group.attached_node_count,)
        and indices.dtype == np.int32
        and len(np.unique(indices)) == len(indices)
        and np.all((indices >= 0) & (indices < group.material_node_count)),
        "attachment indices changed",
    )
    _require(
        weights.shape == (group.attached_node_count, group.controller_point_count)
        and weights.dtype == np.float32
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1.0e-6, rtol=0.0),
        "attachment weights changed",
    )
    _require(
        support.shape == (group.material_node_count,)
        and support.dtype == np.float32
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "action support changed",
    )
    return arrays


def load_prepared_sofa_source_v2(
    path: str | Path,
    *,
    group: SofaSourceGroupV2,
    source_inputs: Mapping[str, npt.NDArray[Any]],
    qualification_frame_count: int,
) -> PreparedSofaSourceV2:
    source = Path(path)
    _require(
        source.is_file() and not source.is_symlink(),
        "prepared source archive must be an ordinary file",
    )
    _require(
        file_sha256(source) == group.prepared_archive_sha256,
        "prepared source archive SHA-256 changed",
    )
    with np.load(source, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    _require(frozenset(arrays) == _PREPARED_ARRAYS, "prepared array roster changed")

    points = arrays["points"]
    cells = arrays["cells"]
    indices = arrays["attachment_indices"]
    projected = arrays["projected_targets"]
    rotations = arrays["rotations"]
    translations = arrays["translations"]
    patch_flat = arrays["patch_flat"]
    patch_offsets = arrays["patch_offsets"]
    patch_ranks = arrays["patch_ranks"]
    patch_count = len(group.expected_contact_patch_sizes)
    _require(
        points.shape == (group.material_node_count, 3)
        and points.dtype == np.float64
        and np.all(np.isfinite(points)),
        "prepared points changed",
    )
    _require(
        cells.shape == (group.tetrahedron_count, 4) and cells.dtype == np.int32,
        "prepared tetrahedra changed",
    )
    _require(
        indices.shape == (group.attached_node_count,) and indices.dtype == np.int64,
        "prepared attachment identities changed",
    )
    _require(
        projected.shape == (qualification_frame_count, group.attached_node_count, 3)
        and projected.dtype == np.float64
        and np.all(np.isfinite(projected)),
        "prepared projected targets changed",
    )
    _require(
        rotations.shape == (qualification_frame_count, patch_count, 3, 3)
        and rotations.dtype == np.float64
        and np.all(np.isfinite(rotations)),
        "prepared rotations changed",
    )
    _require(
        translations.shape == (qualification_frame_count, patch_count, 3)
        and translations.dtype == np.float64
        and np.all(np.isfinite(translations)),
        "prepared translations changed",
    )
    _require(
        patch_flat.shape == (group.attached_node_count,)
        and patch_flat.dtype == np.int64
        and np.array_equal(np.sort(patch_flat), np.arange(group.attached_node_count)),
        "prepared contact patch roster changed",
    )
    _require(
        patch_offsets.shape == (patch_count + 1,)
        and patch_offsets.dtype == np.int64
        and patch_offsets[0] == 0
        and patch_offsets[-1] == group.attached_node_count
        and np.all(np.diff(patch_offsets) > 0),
        "prepared contact patch offsets changed",
    )
    _require(
        tuple(int(value) for value in np.diff(patch_offsets))
        == group.expected_contact_patch_sizes,
        "prepared contact patch sizes changed",
    )
    _require(
        patch_ranks.shape == (patch_count,)
        and patch_ranks.dtype == np.int64
        and np.array_equal(patch_ranks, np.full(patch_count, 3, dtype=np.int64)),
        "prepared contact patch ranks changed",
    )
    _require(
        np.array_equal(
            points,
            np.asarray(source_inputs["frame_zero_points_m"], dtype=np.float64),
        )
        and np.array_equal(
            indices,
            np.asarray(source_inputs["attachment_indices"], dtype=np.int64),
        ),
        "prepared source identities diverged from source inputs",
    )
    patches = tuple(
        np.ascontiguousarray(
            patch_flat[int(patch_offsets[index]) : int(patch_offsets[index + 1])],
            dtype=np.int64,
        )
        for index in range(patch_count)
    )
    raw_targets = attachment_targets_m(
        points,
        np.asarray(source_inputs["controller_points_m"])[:qualification_frame_count],
        indices,
        source_inputs["attachment_weights"],
    )
    rebuilt = rigid_contact_projection_v1(points, indices, raw_targets, patches)
    _require(
        np.allclose(
            rebuilt.projected_targets_m,
            projected,
            atol=1.0e-14,
            rtol=0.0,
        )
        and np.allclose(rebuilt.rotations, rotations, atol=1.0e-14, rtol=0.0)
        and np.allclose(
            rebuilt.translations_m,
            translations,
            atol=1.0e-14,
            rtol=0.0,
        )
        and rebuilt.patch_ranks == tuple(int(value) for value in patch_ranks),
        "prepared rigid contact projection does not replay from source inputs",
    )
    contact = RigidContactProjectionV1(
        projected_targets_m=np.ascontiguousarray(projected),
        rotations=np.ascontiguousarray(rotations),
        translations_m=np.ascontiguousarray(translations),
        patch_local_indices=patches,
        patch_ranks=tuple(int(value) for value in patch_ranks),
    )
    prepare_native_tet_source_geometry_v1(
        points_m=points,
        cells=cells,
        attachment_indices=indices,
        contact=contact,
    )
    return PreparedSofaSourceV2(
        points_m=np.ascontiguousarray(points),
        cells=np.ascontiguousarray(cells),
        attachment_indices=np.ascontiguousarray(indices),
        contact=contact,
    )


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    source_paths = (
        "configs/sota/sofa_fem_zebra_source_physics_v2.json",
        "src/bayesian_phystwin/sofa_fem_source_v1.py",
        "src/bayesian_phystwin/sofa_fem_kinematic_source_v2.py",
        "src/bayesian_phystwin/sofa_fem_source_qualification_v2.py",
        "scripts/remote/run_sofa_fem_source_qualification_v2.py",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git_revision(head, name="git_head")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(status == "", "source qualification requires a clean Git worktree")
    return {
        "git_head": head,
        "git_worktree_clean": True,
        "source_files": {
            relative: file_sha256(repo_root / relative) for relative in source_paths
        },
    }


def _rmse(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(difference))))


def _native_replay(
    *,
    native: NativeSofaFemModulesV1,
    prepared: PreparedSofaSourceV2,
    contact: RigidContactProjectionV1,
    points_m: FloatArray,
    driven: bool,
    interval_substeps: int,
    young_modulus_pa: float,
    simulation: Mapping[str, Any],
) -> SofaKinematicSourceReplayV2:
    return run_sofa_fem_kinematic_source_replay_v2(
        native=native,
        points_m=points_m,
        cells=prepared.cells,
        attachment_indices=prepared.attachment_indices,
        contact=contact,
        driven=driven,
        integrator_time_step_s=(1.0 / (float(simulation["fps"]) * interval_substeps)),
        interval_substeps=interval_substeps,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=float(simulation["poisson_ratio"]),
        density_kg_m3=float(simulation["density_kg_m3"]),
        rayleigh_stiffness=float(simulation["rayleigh_stiffness"]),
        rayleigh_mass=float(simulation["rayleigh_mass"]),
        hard_minimum_deformation_determinant=float(
            simulation["hard_minimum_deformation_determinant"]
        ),
    )


def _rigidly_transformed_source(
    prepared: PreparedSofaSourceV2,
    *,
    rotation: FloatArray,
    translation_m: FloatArray,
) -> tuple[FloatArray, RigidContactProjectionV1]:
    points = np.ascontiguousarray(prepared.points_m @ rotation.T + translation_m)
    targets = np.ascontiguousarray(
        prepared.contact.projected_targets_m @ rotation.T + translation_m
    )
    contact = rigid_contact_projection_v1(
        points,
        prepared.attachment_indices,
        targets,
        prepared.contact.patch_local_indices,
    )
    return points, contact


def _topology_matches(
    replay: SofaKinematicSourceReplayV2,
    *,
    group: SofaSourceGroupV2,
    frame_count: int,
) -> bool:
    return bool(
        replay.positions_m.shape == (frame_count, group.material_node_count, 3)
        and replay.deformation_determinants.shape
        == (frame_count, group.tetrahedron_count)
        and replay.material_vertex_count == group.material_node_count
        and replay.tetrahedron_count == group.tetrahedron_count
        and replay.attachment_count == group.attached_node_count
        and replay.native_step_count >= frame_count - 1
    )


def run_sofa_fem_source_qualification_v2(
    *,
    protocol_path: str | Path,
    group_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    repo_root: str | Path,
    distribution_archive: str | Path,
    sofa_root: str | Path,
) -> dict[str, Any]:
    """Run the frozen target-blind SOFA v2 qualification exactly once."""

    protocol = load_sofa_source_physics_protocol_v2(protocol_path)
    expected_groups = {group.group_id for group in protocol.source_groups}
    if set(group_roots) != expected_groups:
        raise ValueError("group roots must match the complete frozen source roster")
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = _git_provenance(Path(repo_root).absolute())
    native = load_native_sofa_fem_modules_v1(
        distribution_archive=distribution_archive,
        sofa_root=sofa_root,
    )

    simulation = protocol.simulation
    qualification_frames = int(simulation["qualification_frame_count"])
    prepared_groups: list[
        tuple[SofaSourceGroupV2, Path, PreparedSofaSourceV2, Path]
    ] = []
    for group in protocol.source_groups:
        root = Path(group_roots[group.group_id]).absolute()
        source_path = root / group.source_inputs_relative_path.as_posix()
        prepared_path = root / group.prepared_archive_relative_path.as_posix()
        incumbent_path = root / group.incumbent_relative_path.as_posix()
        arrays = load_sofa_source_inputs_v2(source_path, group=group)
        prepared = load_prepared_sofa_source_v2(
            prepared_path,
            group=group,
            source_inputs=arrays,
            qualification_frame_count=qualification_frames,
        )
        _require(
            incumbent_path.is_file() and not incumbent_path.is_symlink(),
            "incumbent fallback must be an ordinary file",
        )
        _require(
            file_sha256(incumbent_path) == group.incumbent_sha256,
            "incumbent fallback SHA-256 changed",
        )
        prepared_groups.append((group, prepared_path, prepared, incumbent_path))

    gates = protocol.gates
    rotation = rigid_transform_v1(
        simulation["rigid_transform_rotation_axis"],
        float(simulation["rigid_transform_angle_rad"]),
    )
    translation = np.asarray(
        simulation["rigid_transform_translation_m"], dtype=np.float64
    )
    base_substeps = int(simulation["base_interval_substeps"])
    refined_substeps = int(simulation["refined_interval_substeps"])
    base_young = float(simulation["young_modulus_pa"])
    group_results: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    all_deterministic = True
    all_topology = True
    all_units = True
    all_fallback = True
    sanity_violations = 0
    maximum_zero_drift = 0.0
    maximum_equivariance = 0.0
    maximum_refinement = 0.0
    maximum_parity = 0.0

    for group, prepared_path, prepared, incumbent_path in prepared_groups:
        common = {
            "native": native,
            "prepared": prepared,
            "contact": prepared.contact,
            "points_m": prepared.points_m,
            "simulation": simulation,
        }
        base = _native_replay(
            **common,
            driven=True,
            interval_substeps=base_substeps,
            young_modulus_pa=base_young,
        )
        repeat = _native_replay(
            **common,
            driven=True,
            interval_substeps=base_substeps,
            young_modulus_pa=base_young,
        )
        zero = _native_replay(
            **common,
            driven=False,
            interval_substeps=base_substeps,
            young_modulus_pa=base_young,
        )
        refined = _native_replay(
            **common,
            driven=True,
            interval_substeps=refined_substeps,
            young_modulus_pa=base_young,
        )
        low = _native_replay(
            **common,
            driven=True,
            interval_substeps=base_substeps,
            young_modulus_pa=float(simulation["young_modulus_probe_low_pa"]),
        )
        high = _native_replay(
            **common,
            driven=True,
            interval_substeps=base_substeps,
            young_modulus_pa=float(simulation["young_modulus_probe_high_pa"]),
        )
        transformed_points, transformed_contact = _rigidly_transformed_source(
            prepared,
            rotation=rotation,
            translation_m=translation,
        )
        transformed = _native_replay(
            native=native,
            prepared=prepared,
            contact=transformed_contact,
            points_m=transformed_points,
            driven=True,
            interval_substeps=base_substeps,
            young_modulus_pa=base_young,
            simulation=simulation,
        )
        replays = (base, repeat, zero, refined, low, high, transformed)
        deterministic = bool(
            np.array_equal(base.positions_m, repeat.positions_m)
            and np.array_equal(
                base.deformation_determinants,
                repeat.deformation_determinants,
            )
            and base.scene_sha256 == repeat.scene_sha256
            and base.schedule_sha256 == repeat.schedule_sha256
        )
        topology = all(
            _topology_matches(
                replay,
                group=group,
                frame_count=qualification_frames,
            )
            for replay in replays
        )
        zero_drift = float(
            np.max(
                np.linalg.norm(
                    zero.positions_m - prepared.points_m[None],
                    axis=2,
                )
            )
        )
        transformed_in_source = np.ascontiguousarray(
            (transformed.positions_m - translation) @ rotation
        )
        equivariance = float(
            np.max(np.linalg.norm(transformed_in_source - base.positions_m, axis=2))
        )
        response = _rmse(base.positions_m[-1], prepared.points_m)
        refinement_absolute = _rmse(base.positions_m[-1], refined.positions_m[-1])
        refinement_relative = refinement_absolute / max(
            _rmse(refined.positions_m[-1], prepared.points_m), 1.0e-15
        )
        material_sensitivity = _rmse(low.positions_m[-1], high.positions_m[-1])
        parity = _rmse(base.positions_m[0], prepared.points_m)
        maximum_attachment = max(
            replay.maximum_attachment_error_m for replay in replays
        )
        maximum_displacement = max(
            float(
                np.max(
                    np.linalg.norm(
                        replay.positions_m
                        - (
                            transformed_points[None]
                            if replay is transformed
                            else prepared.points_m[None]
                        ),
                        axis=2,
                    )
                )
            )
            for replay in replays
        )
        determinants = np.concatenate(
            tuple(replay.deformation_determinants.reshape(-1) for replay in replays)
        )
        finite = bool(
            all(np.all(np.isfinite(replay.positions_m)) for replay in replays)
            and np.all(np.isfinite(determinants))
        )
        group_sanity = {
            "finite": finite,
            "action_response": response >= float(gates["minimum_action_response_m"]),
            "material_sensitivity_lower": material_sensitivity
            >= float(gates["minimum_material_sensitivity_m"]),
            "material_sensitivity_upper": material_sensitivity
            <= float(gates["maximum_material_sensitivity_m"]),
            "time_step_refinement_absolute": refinement_absolute
            <= float(gates["maximum_time_step_refinement_absolute_error_m"]),
            "attachment_projection": maximum_attachment
            <= float(gates["maximum_attachment_error_m"]),
            "node_displacement": maximum_displacement
            <= float(gates["maximum_node_displacement_m"]),
            "deformation_determinant_lower": float(np.min(determinants))
            >= float(gates["minimum_deformation_determinant"]),
            "deformation_determinant_upper": float(np.max(determinants))
            <= float(gates["maximum_deformation_determinant"]),
        }
        sanity_violations += sum(not value for value in group_sanity.values())
        units_valid = bool(
            base.positions_m.shape
            == (qualification_frames, group.material_node_count, 3)
            and base.positions_m.dtype == np.float64
            and parity <= float(gates["maximum_source_query_parity_rmse_m"])
        )

        group_output = output / group.group_id
        group_output.mkdir()
        archive_path = write_deterministic_npz(
            group_output / GROUP_ARCHIVE_FILENAME,
            {
                "base_driven_m": base.positions_m,
                "base_repeat_m": repeat.positions_m,
                "zero_action_m": zero.positions_m,
                "refined_driven_m": refined.positions_m,
                "low_modulus_driven_m": low.positions_m,
                "high_modulus_driven_m": high.positions_m,
                "rigid_transformed_driven_m": transformed.positions_m,
                "base_deformation_determinant": base.deformation_determinants,
                "refined_deformation_determinant": (refined.deformation_determinants),
            },
        )
        fallback_path = group_output / FALLBACK_FILENAME
        shutil.copyfile(incumbent_path, fallback_path)
        fallback_exact = file_sha256(fallback_path) == group.incumbent_sha256

        all_deterministic = all_deterministic and deterministic
        all_topology = all_topology and topology
        all_units = all_units and units_valid
        all_fallback = all_fallback and fallback_exact
        maximum_zero_drift = max(maximum_zero_drift, zero_drift)
        maximum_equivariance = max(maximum_equivariance, equivariance)
        maximum_refinement = max(maximum_refinement, refinement_relative)
        maximum_parity = max(maximum_parity, parity)
        record = {
            "group_id": group.group_id,
            "source_inputs_sha256": group.source_inputs_sha256,
            "prepared_archive_sha256": file_sha256(prepared_path),
            "incumbent_sha256": group.incumbent_sha256,
            "trajectory_archive_sha256": file_sha256(archive_path),
            "fallback_sha256": file_sha256(fallback_path),
            "deterministic_replay_valid": deterministic,
            "topology_identity_preserved": topology,
            "units_coordinate_entity_order_valid": units_valid,
            "exact_fallback_verified": fallback_exact,
            "maximum_zero_action_drift_m": zero_drift,
            "maximum_rigid_equivariance_error_m": equivariance,
            "time_step_refinement_absolute_error_m": refinement_absolute,
            "time_step_refinement_relative_error": refinement_relative,
            "source_query_parity_rmse_m": parity,
            "action_response_rmse_m": response,
            "material_sensitivity_rmse_m": material_sensitivity,
            "maximum_attachment_error_m": maximum_attachment,
            "maximum_node_displacement_m": maximum_displacement,
            "minimum_deformation_determinant": float(np.min(determinants)),
            "maximum_deformation_determinant": float(np.max(determinants)),
            "base_scene_sha256": base.scene_sha256,
            "base_schedule_sha256": base.schedule_sha256,
            "refined_scene_sha256": refined.scene_sha256,
            "refined_schedule_sha256": refined.schedule_sha256,
            "physical_sanity_checks": group_sanity,
        }
        group_results.append(record)
        evidence_records.append(
            {
                "group_id": group.group_id,
                "source_inputs_sha256": group.source_inputs_sha256,
                "prepared_archive_sha256": group.prepared_archive_sha256,
                "incumbent_sha256": group.incumbent_sha256,
                "trajectory_archive_sha256": record["trajectory_archive_sha256"],
            }
        )

    source_evidence_id = content_id({"source_groups": evidence_records})
    incumbent_runtime_id = content_id(
        {
            "incumbent_archives": [
                {"group_id": group.group_id, "sha256": group.incumbent_sha256}
                for group in protocol.source_groups
            ]
        }
    )
    qualification = MaterialBackendQualificationV1(
        canonical_profile_id=protocol.canonical_profile_id,
        producer_profile_id=protocol.producer_profile_id,
        transport=protocol.transport,
        runtime_id=protocol.runtime_id,
        qualification_protocol_id=protocol.protocol_sha256,
        source_evidence_id=source_evidence_id,
        source_group_ids=tuple(group.group_id for group in protocol.source_groups),
        incumbent_runtime_id=incumbent_runtime_id,
        units_coordinate_entity_order_valid=all_units,
        deterministic_replay_valid=all_deterministic,
        maximum_zero_action_drift_m=maximum_zero_drift,
        allowed_zero_action_drift_m=float(gates["maximum_zero_action_drift_m"]),
        maximum_rigid_equivariance_error_m=maximum_equivariance,
        allowed_rigid_equivariance_error_m=float(
            gates["maximum_rigid_equivariance_error_m"]
        ),
        time_step_refinement_relative_error=maximum_refinement,
        allowed_time_step_refinement_relative_error=float(
            gates["maximum_time_step_refinement_relative_error"]
        ),
        topology_identity_preserved=all_topology,
        physical_sanity_violations=sanity_violations,
        gradient_claimed=False,
        maximum_jacobian_relative_error=None,
        allowed_jacobian_relative_error=None,
        source_query_parity_rmse_m=maximum_parity,
        allowed_source_query_parity_rmse_m=float(
            gates["maximum_source_query_parity_rmse_m"]
        ),
        exact_fallback_verified=all_fallback,
        protocol_frozen_before_source_outcomes=True,
        target_outcomes_used=False,
        metadata={
            "evidence_role": "already-open-source-physics-only",
            "backend_variant": BACKEND_VARIANT,
            "engine_revision": protocol.backend["engine_revision"],
            "engine_version": protocol.backend["engine_version"],
            "native_smoke_id": protocol.backend["native_smoke_id"],
            "attachment_model": ATTACHMENT_MODEL,
            "continuation_policy": CONTINUATION_POLICY,
            "material_sensitivity_is_a_physical_sanity_gate": True,
            "incumbent_prediction_arrays_read": False,
            "gradient_claim": "none",
        },
    )
    save_material_backend_qualification_v1(
        qualification,
        output / QUALIFICATION_FILENAME,
    )
    identity: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 2,
        "claim_boundary": protocol.value["claim_boundary"],
        "protocol_sha256": protocol.protocol_sha256,
        "runtime_id": protocol.runtime_id,
        "runtime": {
            "distribution_archive_sha256": SOFA_ARCHIVE_SHA256,
            "engine_revision": SOFA_REVISION,
            "engine_version": SOFA_VERSION,
            "installed_records": native.installed_records,
        },
        "implementation": provenance,
        "source_groups": group_results,
        "qualification_artifact_id": qualification.artifact_id,
        "qualified": qualification.qualified,
        "failure_reasons": list(qualification.failure_reasons),
        "source_value_scoring_authorized": qualification.qualified,
        "information_boundary": {
            "source_inputs_read": True,
            "prepared_source_archives_read": True,
            "incumbent_bytes_read_for_exact_fallback": True,
            "incumbent_prediction_arrays_read": False,
            "source_object_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
    }
    result = {**identity, "result_id": content_id(identity)}
    write_atomic_json(result, output / RESULT_FILENAME, overwrite=False)
    return result


__all__ = [
    "FALLBACK_FILENAME",
    "GROUP_ARCHIVE_FILENAME",
    "PROTOCOL_SCHEMA",
    "QUALIFICATION_FILENAME",
    "RESULT_FILENAME",
    "PreparedSofaSourceV2",
    "SofaSourceGroupV2",
    "SofaSourcePhysicsProtocolV2",
    "file_sha256",
    "load_prepared_sofa_source_v2",
    "load_sofa_source_inputs_v2",
    "load_sofa_source_physics_protocol_v2",
    "run_sofa_fem_source_qualification_v2",
]
