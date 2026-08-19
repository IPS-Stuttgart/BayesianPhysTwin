"""Frozen source-physics qualification for the optional JAX-FEM backend."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._portable_contracts import content_id, load_strict_json_object, write_atomic_json
from .material_backend_qualification_v1 import (
    MaterialBackendQualificationV1,
    save_material_backend_qualification_v1,
)
from .material_backend_v1 import BackendTransportV1, resolve_material_backend_profile
from .physical_rollout_v1 import load_physical_rollout_archive, write_deterministic_npz

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntegerArray: TypeAlias = npt.NDArray[np.integer[Any]]

PROTOCOL_SCHEMA: Final = "bayesian-phystwin.jax-fem-source-physics-protocol"
RESULT_SCHEMA: Final = "bayesian-phystwin.jax-fem-source-physics-result"
RESULT_FILENAME: Final = "jax-fem-source-physics-result.json"
QUALIFICATION_FILENAME: Final = "material-backend-qualification.json"
GROUP_ARCHIVE_FILENAME: Final = "jax-fem-source-physics-trajectories.npz"
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
        "engine_repository",
        "engine_revision",
        "engine_version",
        "native_smoke_artifact_sha256",
        "native_smoke_id",
        "runtime_id",
        "installed_source_sha256",
        "runtime_versions",
    }
)
_RUNTIME_VERSION_FIELDS: Final = frozenset(
    {"python", "jax", "jax_fem", "numpy", "scipy"}
)
_GROUP_FIELDS: Final = frozenset(
    {
        "group_id",
        "source_inputs_relative_path",
        "source_inputs_sha256",
        "incumbent_relative_path",
        "incumbent_sha256",
        "frame_count",
        "material_node_count",
        "controller_point_count",
        "attached_node_count",
        "expected_contact_patch_sizes",
        "expected_base_cell_count",
        "expected_coarse_cell_count",
    }
)
_SIMULATION_FIELDS: Final = frozenset(
    {
        "backend",
        "precision",
        "seed",
        "qualification_frame_count",
        "base_frame_indices",
        "refined_frame_indices",
        "base_mesh_max_edge_m",
        "coarse_mesh_max_edge_m",
        "minimum_tetrahedron_shape_ratio",
        "contact_cluster_radius_m",
        "element_type",
        "constitutive_model",
        "young_modulus_pa",
        "young_modulus_probe_low_pa",
        "young_modulus_probe_high_pa",
        "low_poisson_ratio",
        "base_poisson_ratio",
        "high_poisson_ratio",
        "solver",
        "rigid_transform_rotation_axis",
        "rigid_transform_angle_rad",
        "rigid_transform_translation_m",
        "mesh_policy",
        "contact_boundary_policy",
        "load_step_refinement_semantics",
    }
)
_GATE_FIELDS: Final = frozenset(
    {
        "maximum_zero_action_drift_m",
        "maximum_rigid_equivariance_error_m",
        "maximum_time_step_refinement_relative_error",
        "maximum_mesh_connectivity_sensitivity_relative_error",
        "maximum_source_query_parity_rmse_m",
        "minimum_action_response_m",
        "minimum_poisson_sensitivity_m",
        "maximum_poisson_sensitivity_m",
        "maximum_young_modulus_invariance_error_m",
        "maximum_contact_projection_error_m",
        "maximum_node_displacement_m",
        "minimum_deformation_determinant",
        "maximum_deformation_determinant",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "frame_zero_geometry_allowed",
        "known_full_controller_trajectory_allowed",
        "incumbent_prediction_allowed_for_query_parity_and_exact_fallback",
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


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


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


def _integer_list(
    value: object,
    *,
    name: str,
    minimum_length: int = 1,
    sorted_unique: bool = True,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) < minimum_length:
        raise ValueError(f"{name} must be a nonempty integer list")
    result = tuple(_nonnegative_int(item, name=f"{name}[]") for item in value)
    if sorted_unique:
        _require(
            tuple(sorted(set(result))) == result,
            f"{name} must be sorted and unique",
        )
    return result


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
class JaxFemSourceGroupV1:
    group_id: str
    source_inputs_relative_path: PurePosixPath
    source_inputs_sha256: str
    incumbent_relative_path: PurePosixPath
    incumbent_sha256: str
    frame_count: int
    material_node_count: int
    controller_point_count: int
    attached_node_count: int
    expected_contact_patch_sizes: tuple[int, ...]
    expected_base_cell_count: int
    expected_coarse_cell_count: int


@dataclass(frozen=True, slots=True)
class JaxFemSourcePhysicsProtocolV1:
    value: Mapping[str, Any]
    protocol_sha256: str
    canonical_profile_id: str
    producer_profile_id: str
    transport: BackendTransportV1
    runtime_id: str
    source_groups: tuple[JaxFemSourceGroupV1, ...]

    @property
    def backend(self) -> Mapping[str, Any]:
        return _mapping(self.value["backend"], name="backend")

    @property
    def simulation(self) -> Mapping[str, Any]:
        return _mapping(self.value["simulation"], name="simulation")

    @property
    def gates(self) -> Mapping[str, Any]:
        return _mapping(self.value["gates"], name="gates")


def load_jax_fem_source_physics_protocol_v1(
    path: str | Path,
) -> JaxFemSourcePhysicsProtocolV1:
    source = Path(path)
    value = load_strict_json_object(source, label="JAX-FEM source-physics protocol")
    _exact_fields(value, _PROTOCOL_FIELDS, "protocol")
    _require(value["schema"] == PROTOCOL_SCHEMA, "protocol schema changed")
    _require(value["schema_version"] == 1, "protocol version changed")
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
    _require(
        backend["engine_repository"] == "https://github.com/deepmodeling/jax-fem",
        "engine repository changed",
    )
    _git_revision(backend["engine_revision"], name="engine_revision")
    _canonical_string(backend["engine_version"], name="engine_version")
    _sha256(
        backend["native_smoke_artifact_sha256"],
        name="native_smoke_artifact_sha256",
    )
    _sha256(backend["native_smoke_id"], name="native_smoke_id")
    runtime_id = _sha256(backend["runtime_id"], name="runtime_id")
    sources = _mapping(
        backend["installed_source_sha256"], name="installed_source_sha256"
    )
    _require(
        frozenset(sources)
        == frozenset(
            {
                "jax_fem/generate_mesh.py",
                "jax_fem/problem.py",
                "jax_fem/solver.py",
            }
        ),
        "installed source roster changed",
    )
    for key, digest in sources.items():
        _sha256(digest, name=f"installed_source_sha256[{key}]")
    versions = _mapping(backend["runtime_versions"], name="runtime_versions")
    _exact_fields(versions, _RUNTIME_VERSION_FIELDS, "runtime_versions")
    for key, version in versions.items():
        _canonical_string(version, name=f"runtime_versions[{key}]")

    raw_groups = value["source_groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) < 2:
        raise ValueError("at least two source groups are required")
    groups: list[JaxFemSourceGroupV1] = []
    for index, raw in enumerate(raw_groups):
        group = _mapping(raw, name=f"source_groups[{index}]")
        _exact_fields(group, _GROUP_FIELDS, f"source_groups[{index}]")
        patch_sizes = _integer_list(
            group["expected_contact_patch_sizes"],
            name="expected_contact_patch_sizes",
            minimum_length=1,
            sorted_unique=False,
        )
        _require(
            all(size > 0 for size in patch_sizes), "contact patches must be nonempty"
        )
        attached_count = _positive_int(
            group["attached_node_count"], name="attached_node_count"
        )
        _require(
            sum(patch_sizes) == attached_count,
            "contact patch sizes do not cover attached nodes",
        )
        groups.append(
            JaxFemSourceGroupV1(
                group_id=_canonical_string(group["group_id"], name="group_id"),
                source_inputs_relative_path=_canonical_relative_path(
                    group["source_inputs_relative_path"],
                    name="source_inputs_relative_path",
                ),
                source_inputs_sha256=_sha256(
                    group["source_inputs_sha256"], name="source_inputs_sha256"
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
                expected_contact_patch_sizes=patch_sizes,
                expected_base_cell_count=_positive_int(
                    group["expected_base_cell_count"],
                    name="expected_base_cell_count",
                ),
                expected_coarse_cell_count=_positive_int(
                    group["expected_coarse_cell_count"],
                    name="expected_coarse_cell_count",
                ),
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
    qualification_frames = _positive_int(
        simulation["qualification_frame_count"],
        name="qualification_frame_count",
    )
    base_frames = _integer_list(
        simulation["base_frame_indices"], name="base_frame_indices"
    )
    refined_frames = _integer_list(
        simulation["refined_frame_indices"], name="refined_frame_indices"
    )
    _require(base_frames[0] == 0, "base frame roster must begin at frame zero")
    _require(
        refined_frames == tuple(range(qualification_frames)), "refined frames changed"
    )
    _require(
        set(base_frames).issubset(refined_frames), "base frames are not refined frames"
    )
    for name in (
        "base_mesh_max_edge_m",
        "coarse_mesh_max_edge_m",
        "minimum_tetrahedron_shape_ratio",
        "contact_cluster_radius_m",
        "young_modulus_pa",
        "young_modulus_probe_low_pa",
        "young_modulus_probe_high_pa",
        "rigid_transform_angle_rad",
    ):
        _finite(simulation[name], name=name, positive=True)
    _require(
        float(simulation["coarse_mesh_max_edge_m"])
        > float(simulation["base_mesh_max_edge_m"]),
        "coarse mesh edge limit must exceed the base limit",
    )
    _require(simulation["element_type"] == "TET4", "element type changed")
    _require(
        simulation["constitutive_model"] == "small-strain-isotropic-linear-elasticity",
        "constitutive model changed",
    )
    _require(simulation["solver"] == "scipy-spsolve", "solver changed")
    poisson_values = [
        _finite(simulation[name], name=name)
        for name in (
            "low_poisson_ratio",
            "base_poisson_ratio",
            "high_poisson_ratio",
        )
    ]
    _require(
        -1.0 < poisson_values[0] < poisson_values[1] < poisson_values[2] < 0.5,
        "Poisson ratio roster changed",
    )
    axis = np.asarray(
        _vector3(
            simulation["rigid_transform_rotation_axis"],
            name="rigid_transform_rotation_axis",
        )
    )
    _require(np.linalg.norm(axis) > 0.0, "rotation axis is zero")
    _vector3(
        simulation["rigid_transform_translation_m"],
        name="rigid_transform_translation_m",
    )
    _require(
        simulation["mesh_policy"]
        == "deterministic-delaunay-qbb-qc-qz-q12-edge-shape-filter-v1",
        "mesh policy changed",
    )
    _require(
        simulation["contact_boundary_policy"]
        == "weighted-controller-targets-rigid-se3-connected-patches-v1",
        "contact boundary policy changed",
    )
    _require(
        simulation["load_step_refinement_semantics"]
        == "independent-quasistatic-shared-source-frame-parity-v1",
        "load-step refinement semantics changed",
    )

    gates = _mapping(value["gates"], name="gates")
    _exact_fields(gates, _GATE_FIELDS, "gates")
    for name in _GATE_FIELDS:
        _finite(gates[name], name=name, positive=True)
    _require(
        float(gates["minimum_poisson_sensitivity_m"])
        < float(gates["maximum_poisson_sensitivity_m"]),
        "Poisson sensitivity interval is empty",
    )
    _require(
        float(gates["minimum_deformation_determinant"])
        < float(gates["maximum_deformation_determinant"]),
        "deformation determinant interval is empty",
    )

    boundary = _mapping(value["information_boundary"], name="information_boundary")
    _exact_fields(boundary, _BOUNDARY_FIELDS, "information_boundary")
    _require(
        boundary
        == {
            "frame_zero_geometry_allowed": True,
            "known_full_controller_trajectory_allowed": True,
            "incumbent_prediction_allowed_for_query_parity_and_exact_fallback": True,
            "source_object_outcomes_allowed": False,
            "target_or_held_out_artifact_access_allowed": False,
            "future_scoring_authorized": False,
            "no_replacement": True,
        },
        "information boundary changed",
    )
    return JaxFemSourcePhysicsProtocolV1(
        value=value,
        protocol_sha256=file_sha256(source),
        canonical_profile_id=canonical_profile_id,
        producer_profile_id=producer_profile_id,
        transport=resolved.transport,
        runtime_id=runtime_id,
        source_groups=tuple(groups),
    )


def load_jax_fem_source_inputs_v1(
    path: str | Path,
    *,
    group: JaxFemSourceGroupV1,
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


def attachment_targets_m(
    frame_zero_points_m: npt.ArrayLike,
    controller_points_m: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    attachment_weights: npt.ArrayLike,
) -> FloatArray:
    points = np.asarray(frame_zero_points_m, dtype=np.float64)
    controller = np.asarray(controller_points_m, dtype=np.float64)
    indices = np.asarray(attachment_indices, dtype=np.int64)
    weights = np.asarray(attachment_weights, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 3, "points must have shape (N,3)")
    _require(
        controller.ndim == 3 and controller.shape[2] == 3,
        "controller must have shape (T,C,3)",
    )
    _require(indices.ndim == 1 and len(indices) >= 1, "attachment indices changed")
    _require(
        weights.shape == (len(indices), controller.shape[1]),
        "attachment weights changed",
    )
    displacement = controller - controller[:1]
    weighted = np.einsum("ac,tcd->tad", weights, displacement, optimize=True)
    return cast(FloatArray, np.ascontiguousarray(points[indices][None] + weighted))


def build_tetrahedral_cells_v1(
    points_m: npt.ArrayLike,
    *,
    maximum_edge_m: float,
    minimum_shape_ratio: float,
) -> npt.NDArray[np.int32]:
    from scipy.spatial import Delaunay

    points = np.asarray(points_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[0] >= 4 and points.shape[1] == 3,
        "tetrahedral source points must have shape (N,3)",
    )
    _require(np.all(np.isfinite(points)), "tetrahedral source points are not finite")
    maximum_edge = _finite(maximum_edge_m, name="maximum_edge_m", positive=True)
    minimum_shape = _finite(
        minimum_shape_ratio, name="minimum_shape_ratio", positive=True
    )
    raw = np.asarray(
        Delaunay(points, qhull_options="Qbb Qc Qz Q12").simplices,
        dtype=np.int32,
    )
    vertices = points[raw]
    edges = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 1],
            vertices[:, 3] - vertices[:, 1],
            vertices[:, 3] - vertices[:, 2],
        ),
        axis=1,
    )
    maximum_edges = np.max(np.linalg.norm(edges, axis=2), axis=1)
    signed = np.linalg.det(
        np.stack(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ),
            axis=2,
        )
    )
    volumes = np.abs(signed) / 6.0
    shape_ratio = volumes / np.maximum(maximum_edges**3, np.finfo(np.float64).tiny)
    keep = (maximum_edges <= maximum_edge) & (shape_ratio >= minimum_shape)
    cells = np.ascontiguousarray(raw[keep], dtype=np.int32)
    signed = signed[keep]
    _require(len(cells) >= 1, "tetrahedral mesh is empty")
    negative = signed < 0.0
    temporary = cells[negative, 1].copy()
    cells[negative, 1] = cells[negative, 2]
    cells[negative, 2] = temporary
    return cells


def mesh_component_count_v1(cells: npt.ArrayLike, *, node_count: int) -> int:
    tetrahedra = np.asarray(cells, dtype=np.int64)
    _require(tetrahedra.ndim == 2 and tetrahedra.shape[1] == 4, "cells changed")
    parent: npt.NDArray[np.int64] = np.arange(node_count, dtype=np.int64)

    def find(value: int) -> int:
        current = value
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        return current

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    for cell in tetrahedra:
        first = int(cell[0])
        for value in cell[1:]:
            union(first, int(value))
    used = np.unique(tetrahedra)
    return len({find(int(value)) for value in used})


def contact_patch_local_indices_v1(
    points_m: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    *,
    radius_m: float,
) -> tuple[npt.NDArray[np.int64], ...]:
    from scipy.spatial import cKDTree

    points = np.asarray(points_m, dtype=np.float64)
    indices = np.asarray(attachment_indices, dtype=np.int64)
    radius = _finite(radius_m, name="radius_m", positive=True)
    attached = points[indices]
    parent: npt.NDArray[np.int64] = np.arange(len(indices), dtype=np.int64)

    def find(value: int) -> int:
        current = value
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        return current

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    for left, right in sorted(cKDTree(attached).query_pairs(radius)):
        union(int(left), int(right))
    groups: dict[int, list[int]] = {}
    for local_index in range(len(indices)):
        groups.setdefault(find(local_index), []).append(local_index)
    patches = tuple(
        np.ascontiguousarray(values, dtype=np.int64)
        for values in sorted(
            groups.values(),
            key=lambda values: int(np.min(indices[np.asarray(values)])),
        )
    )
    _require(sum(map(len, patches)) == len(indices), "contact patches lost nodes")
    return patches


@dataclass(frozen=True, slots=True)
class RigidContactProjectionV1:
    projected_targets_m: FloatArray
    rotations: FloatArray
    translations_m: FloatArray
    patch_local_indices: tuple[npt.NDArray[np.int64], ...]
    patch_ranks: tuple[int, ...]


def rigid_contact_projection_v1(
    points_m: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    raw_targets_m: npt.ArrayLike,
    patch_local_indices: Sequence[npt.ArrayLike],
) -> RigidContactProjectionV1:
    points = np.asarray(points_m, dtype=np.float64)
    indices = np.asarray(attachment_indices, dtype=np.int64)
    targets = np.asarray(raw_targets_m, dtype=np.float64)
    patches = tuple(np.asarray(patch, dtype=np.int64) for patch in patch_local_indices)
    _require(targets.shape[1:] == (len(indices), 3), "raw attachment targets changed")
    projected = np.empty_like(targets)
    rotations: npt.NDArray[np.float64] = np.empty(
        (len(targets), len(patches), 3, 3), dtype=np.float64
    )
    translations: npt.NDArray[np.float64] = np.empty(
        (len(targets), len(patches), 3), dtype=np.float64
    )
    ranks: list[int] = []
    for patch in patches:
        reference = points[indices[patch]]
        ranks.append(int(np.linalg.matrix_rank(reference - np.mean(reference, axis=0))))
    _require(all(rank == 3 for rank in ranks), "contact patch is not full rank")
    for frame in range(len(targets)):
        for patch_index, patch in enumerate(patches):
            source = points[indices[patch]]
            destination = targets[frame, patch]
            if frame == 0:
                rotation = np.eye(3, dtype=np.float64)
                translation: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)
            else:
                source_center = np.mean(source, axis=0)
                destination_center = np.mean(destination, axis=0)
                left, _, right_t = np.linalg.svd(
                    (source - source_center).T @ (destination - destination_center),
                    full_matrices=False,
                )
                rotation = right_t.T @ left.T
                if np.linalg.det(rotation) < 0.0:
                    right_t[-1] *= -1.0
                    rotation = right_t.T @ left.T
                translation = destination_center - rotation @ source_center
            rotations[frame, patch_index] = rotation
            translations[frame, patch_index] = translation
            projected[frame, patch] = source @ rotation.T + translation
    return RigidContactProjectionV1(
        projected_targets_m=cast(FloatArray, np.ascontiguousarray(projected)),
        rotations=cast(FloatArray, np.ascontiguousarray(rotations)),
        translations_m=cast(FloatArray, np.ascontiguousarray(translations)),
        patch_local_indices=tuple(
            np.ascontiguousarray(patch, dtype=np.int64) for patch in patches
        ),
        patch_ranks=tuple(ranks),
    )


def deformation_determinants_v1(
    reference_points_m: npt.ArrayLike,
    cells: npt.ArrayLike,
    deformed_points_m: npt.ArrayLike,
) -> FloatArray:
    reference = np.asarray(reference_points_m, dtype=np.float64)
    tetrahedra = np.asarray(cells, dtype=np.int64)
    deformed = np.asarray(deformed_points_m, dtype=np.float64)
    _require(
        deformed.ndim == 3 and deformed.shape[1:] == reference.shape,
        "deformed trajectory shape changed",
    )
    reference_cells = reference[tetrahedra]
    reference_edges = np.stack(
        (
            reference_cells[:, 1] - reference_cells[:, 0],
            reference_cells[:, 2] - reference_cells[:, 0],
            reference_cells[:, 3] - reference_cells[:, 0],
        ),
        axis=2,
    )
    inverse_reference = np.linalg.inv(reference_edges)
    result: list[npt.NDArray[Any]] = []
    for frame in deformed:
        cells_frame = frame[tetrahedra]
        current_edges = np.stack(
            (
                cells_frame[:, 1] - cells_frame[:, 0],
                cells_frame[:, 2] - cells_frame[:, 0],
                cells_frame[:, 3] - cells_frame[:, 0],
            ),
            axis=2,
        )
        result.append(np.linalg.det(current_edges @ inverse_reference))
    return cast(FloatArray, np.ascontiguousarray(np.stack(result)))


def rigid_transform_v1(
    axis: npt.ArrayLike,
    angle_rad: float,
) -> FloatArray:
    vector = np.asarray(axis, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    x_value, y_value, z_value = vector
    skew = np.asarray(
        [[0.0, -z_value, y_value], [z_value, 0.0, -x_value], [-y_value, x_value, 0.0]],
        dtype=np.float64,
    )
    angle = float(angle_rad)
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    return cast(FloatArray, np.ascontiguousarray(rotation))


@dataclass(frozen=True, slots=True)
class _NativeModules:
    jax: Any
    jnp: Any
    Mesh: Any
    Problem: Any
    solver: Any


def _load_native_modules(protocol: JaxFemSourcePhysicsProtocolV1) -> _NativeModules:
    backend = protocol.backend
    versions = _mapping(backend["runtime_versions"], name="runtime_versions")
    _require(platform.python_version() == versions["python"], "Python version changed")
    _require(
        importlib.metadata.version("jax-fem") == versions["jax_fem"],
        "JAX-FEM version changed",
    )
    _require(
        importlib.metadata.version("numpy") == versions["numpy"],
        "NumPy version changed",
    )
    _require(
        importlib.metadata.version("scipy") == versions["scipy"],
        "SciPy version changed",
    )
    jax = importlib.import_module("jax")
    _require(jax.__version__ == versions["jax"], "JAX version changed")
    jax.config.update("jax_enable_x64", True)
    jnp = importlib.import_module("jax.numpy")
    jax_fem = importlib.import_module("jax_fem")
    package_file = getattr(jax_fem, "__file__", None)
    _require(package_file is not None, "JAX-FEM package path is unavailable")
    package_root = Path(cast(str, package_file)).resolve().parent
    sources = _mapping(
        backend["installed_source_sha256"], name="installed_source_sha256"
    )
    for relative, expected in sources.items():
        path = package_root.parent / relative
        _require(
            path.is_file() and not path.is_symlink(), "JAX-FEM source is unavailable"
        )
        _require(file_sha256(path) == expected, f"JAX-FEM source changed: {relative}")
    devices = jax.devices()
    _require(
        len(devices) >= 1 and devices[0].platform == "cpu", "JAX CPU runtime changed"
    )
    generate_mesh = importlib.import_module("jax_fem.generate_mesh")
    problem = importlib.import_module("jax_fem.problem")
    solver_module = importlib.import_module("jax_fem.solver")
    return _NativeModules(
        jax=jax,
        jnp=jnp,
        Mesh=generate_mesh.Mesh,
        Problem=problem.Problem,
        solver=solver_module.solver,
    )


@dataclass(frozen=True, slots=True)
class _NativeReplay:
    frame_indices: tuple[int, ...]
    positions_m: FloatArray
    deformation_determinants: FloatArray


def _location_factory(jnp: Any, node_ids: Any) -> Any:
    def location(point: Any, index: Any) -> Any:
        del point
        return jnp.any(index == node_ids)

    return location


def _value_factory(
    rotation: Any,
    translation: Any,
    component: int,
) -> Any:
    def value(point: Any) -> Any:
        return (rotation @ point + translation - point)[component]

    return value


def _run_native_replay(
    *,
    native: _NativeModules,
    points_m: FloatArray,
    cells: npt.NDArray[np.int32],
    attachment_indices: npt.NDArray[np.int64],
    contact: RigidContactProjectionV1,
    frame_indices: tuple[int, ...],
    young_modulus_pa: float,
    poisson_ratio: float,
    driven: bool,
) -> _NativeReplay:  # pragma: no cover - exercised by the frozen native run
    jnp = native.jnp
    mesh = native.Mesh(jnp.asarray(points_m), jnp.asarray(cells), ele_type="TET4")
    positions: list[npt.NDArray[Any]] = []
    young = float(young_modulus_pa)
    poisson = float(poisson_ratio)

    def get_tensor_map(problem: Any) -> Any:
        def stress(displacement_gradient: Any) -> Any:
            shear = young / (2.0 * (1.0 + poisson))
            lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
            strain = 0.5 * (displacement_gradient + displacement_gradient.T)
            return (
                lame * jnp.trace(strain) * jnp.eye(problem.dim) + 2.0 * shear * strain
            )

        return stress

    linear_elasticity = cast(
        type[Any],
        type(
            "LinearElasticity",
            (native.Problem,),
            {"get_tensor_map": get_tensor_map},
        ),
    )

    for frame_index in frame_indices:
        contact_frame = frame_index if driven else 0
        locations: list[Any] = []
        components: list[int] = []
        values: list[Any] = []
        for patch_index, local_indices in enumerate(contact.patch_local_indices):
            node_ids = jnp.asarray(attachment_indices[local_indices])
            rotation = jnp.asarray(contact.rotations[contact_frame, patch_index])
            translation = jnp.asarray(
                contact.translations_m[contact_frame, patch_index]
            )
            for component in range(3):
                locations.append(_location_factory(jnp, node_ids))
                components.append(component)
                values.append(_value_factory(rotation, translation, component))
        problem = linear_elasticity(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type="TET4",
            dirichlet_bc_info=[locations, components, values],
        )
        solution = native.solver(problem, solver_options={"spsolve_solver": {}})
        _require(
            isinstance(solution, (list, tuple)) and len(solution) == 1,
            "JAX-FEM returned an unexpected solution structure",
        )
        displacement = np.ascontiguousarray(np.asarray(solution[0]), dtype=np.float64)
        _require(displacement.shape == points_m.shape, "JAX-FEM node roster changed")
        _require(
            np.all(np.isfinite(displacement)), "JAX-FEM displacement is not finite"
        )
        positions.append(points_m + displacement)
    trajectory = cast(FloatArray, np.ascontiguousarray(np.stack(positions)))
    determinants = deformation_determinants_v1(points_m, cells, trajectory)
    return _NativeReplay(
        frame_indices=frame_indices,
        positions_m=trajectory,
        deformation_determinants=determinants,
    )


def _git_provenance(
    repo_root: Path,
    *,
    source_paths: tuple[str, ...] = (
        "src/bayesian_phystwin/jax_fem_source_qualification_v1.py",
        "scripts/remote/run_jax_fem_source_qualification_v1.py",
    ),
) -> dict[str, Any]:
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


def _select_shared_frames(
    replay: _NativeReplay,
    wanted: tuple[int, ...],
) -> FloatArray:
    lookup = {frame: index for index, frame in enumerate(replay.frame_indices)}
    return cast(
        FloatArray,
        np.ascontiguousarray(
            np.stack([replay.positions_m[lookup[frame]] for frame in wanted])
        ),
    )


def run_jax_fem_source_qualification_v1(
    *,
    protocol_path: str | Path,
    group_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    protocol = load_jax_fem_source_physics_protocol_v1(protocol_path)
    if set(group_roots) != {group.group_id for group in protocol.source_groups}:
        raise ValueError("group roots must match the complete frozen source roster")
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = _git_provenance(Path(repo_root).absolute())
    native = _load_native_modules(protocol)

    simulation = protocol.simulation
    gates = protocol.gates
    base_frames = cast(tuple[int, ...], tuple(simulation["base_frame_indices"]))
    refined_frames = cast(tuple[int, ...], tuple(simulation["refined_frame_indices"]))
    group_results: list[dict[str, Any]] = []
    source_evidence_records: list[dict[str, Any]] = []
    all_deterministic = True
    all_topology = True
    all_fallback = True
    all_units = True
    sanity_violations = 0
    maximum_zero_drift = 0.0
    maximum_equivariance = 0.0
    maximum_refinement = 0.0
    maximum_parity = 0.0

    rotation = rigid_transform_v1(
        simulation["rigid_transform_rotation_axis"],
        float(simulation["rigid_transform_angle_rad"]),
    )
    translation = np.asarray(
        simulation["rigid_transform_translation_m"], dtype=np.float64
    )

    for group in protocol.source_groups:
        root = Path(group_roots[group.group_id]).absolute()
        source_path = root / group.source_inputs_relative_path.as_posix()
        incumbent_path = root / group.incumbent_relative_path.as_posix()
        _require(
            incumbent_path.is_file() and not incumbent_path.is_symlink(),
            "incumbent must be an ordinary file",
        )
        _require(
            file_sha256(incumbent_path) == group.incumbent_sha256,
            "incumbent SHA-256 changed",
        )
        arrays = load_jax_fem_source_inputs_v1(source_path, group=group)
        incumbent = load_physical_rollout_archive(
            incumbent_path, expected_frame_count=group.frame_count
        )
        _require(
            incumbent["prediction_m"].shape
            == (group.frame_count, group.material_node_count, 3),
            "incumbent physical shape changed",
        )
        points = np.asarray(arrays["frame_zero_points_m"], dtype=np.float64)
        controller = np.asarray(
            arrays["controller_points_m"][
                : int(simulation["qualification_frame_count"])
            ],
            dtype=np.float64,
        )
        indices = np.asarray(arrays["attachment_indices"], dtype=np.int64)
        weights = np.asarray(arrays["attachment_weights"], dtype=np.float64)
        raw_targets = attachment_targets_m(points, controller, indices, weights)
        patches = contact_patch_local_indices_v1(
            points,
            indices,
            radius_m=float(simulation["contact_cluster_radius_m"]),
        )
        contact = rigid_contact_projection_v1(points, indices, raw_targets, patches)
        patch_sizes = tuple(len(patch) for patch in patches)

        base_cells = build_tetrahedral_cells_v1(
            points,
            maximum_edge_m=float(simulation["base_mesh_max_edge_m"]),
            minimum_shape_ratio=float(simulation["minimum_tetrahedron_shape_ratio"]),
        )
        coarse_cells = build_tetrahedral_cells_v1(
            points,
            maximum_edge_m=float(simulation["coarse_mesh_max_edge_m"]),
            minimum_shape_ratio=float(simulation["minimum_tetrahedron_shape_ratio"]),
        )
        base_topology = bool(
            len(base_cells) == group.expected_base_cell_count
            and len(np.unique(base_cells)) == group.material_node_count
            and mesh_component_count_v1(base_cells, node_count=len(points)) == 1
        )
        coarse_topology = bool(
            len(coarse_cells) == group.expected_coarse_cell_count
            and len(np.unique(coarse_cells)) == group.material_node_count
            and mesh_component_count_v1(coarse_cells, node_count=len(points)) == 1
        )
        contact_topology = bool(
            patch_sizes == group.expected_contact_patch_sizes
            and contact.patch_ranks == tuple(3 for _ in patches)
        )

        common = {
            "native": native,
            "points_m": points,
            "attachment_indices": indices,
            "contact": contact,
            "young_modulus_pa": float(simulation["young_modulus_pa"]),
        }
        base = _run_native_replay(
            **common,
            cells=base_cells,
            frame_indices=base_frames,
            poisson_ratio=float(simulation["base_poisson_ratio"]),
            driven=True,
        )
        repeat = _run_native_replay(
            **common,
            cells=base_cells,
            frame_indices=base_frames,
            poisson_ratio=float(simulation["base_poisson_ratio"]),
            driven=True,
        )
        zero = _run_native_replay(
            **common,
            cells=base_cells,
            frame_indices=base_frames,
            poisson_ratio=float(simulation["base_poisson_ratio"]),
            driven=False,
        )
        refined = _run_native_replay(
            **common,
            cells=base_cells,
            frame_indices=refined_frames,
            poisson_ratio=float(simulation["base_poisson_ratio"]),
            driven=True,
        )
        coarse = _run_native_replay(
            **common,
            cells=coarse_cells,
            frame_indices=base_frames,
            poisson_ratio=float(simulation["base_poisson_ratio"]),
            driven=True,
        )
        low_poisson = _run_native_replay(
            **common,
            cells=base_cells,
            frame_indices=base_frames,
            poisson_ratio=float(simulation["low_poisson_ratio"]),
            driven=True,
        )
        high_poisson = _run_native_replay(
            **common,
            cells=base_cells,
            frame_indices=base_frames,
            poisson_ratio=float(simulation["high_poisson_ratio"]),
            driven=True,
        )
        terminal_frame = (base_frames[-1],)
        low_young = _run_native_replay(
            **{
                **common,
                "young_modulus_pa": float(simulation["young_modulus_probe_low_pa"]),
            },
            cells=base_cells,
            frame_indices=terminal_frame,
            poisson_ratio=float(simulation["base_poisson_ratio"]),
            driven=True,
        )
        high_young = _run_native_replay(
            **{
                **common,
                "young_modulus_pa": float(simulation["young_modulus_probe_high_pa"]),
            },
            cells=base_cells,
            frame_indices=terminal_frame,
            poisson_ratio=float(simulation["base_poisson_ratio"]),
            driven=True,
        )

        transformed_points = np.ascontiguousarray(points @ rotation.T + translation)
        transformed_controller = np.ascontiguousarray(
            controller @ rotation.T + translation
        )
        transformed_targets = attachment_targets_m(
            transformed_points,
            transformed_controller,
            indices,
            weights,
        )
        transformed_contact = rigid_contact_projection_v1(
            transformed_points,
            indices,
            transformed_targets,
            patches,
        )
        transformed = _run_native_replay(
            native=native,
            points_m=transformed_points,
            cells=base_cells,
            attachment_indices=indices,
            contact=transformed_contact,
            frame_indices=base_frames,
            young_modulus_pa=float(simulation["young_modulus_pa"]),
            poisson_ratio=float(simulation["base_poisson_ratio"]),
            driven=True,
        )

        deterministic = bool(
            np.array_equal(base.positions_m, repeat.positions_m)
            and np.array_equal(
                base.deformation_determinants, repeat.deformation_determinants
            )
        )
        zero_drift = float(
            np.max(np.linalg.norm(zero.positions_m - points[None], axis=2))
        )
        inverse_transformed = (transformed.positions_m - translation) @ rotation
        equivariance = float(
            np.max(np.linalg.norm(inverse_transformed - base.positions_m, axis=2))
        )
        refined_shared = _select_shared_frames(refined, base_frames)
        response = _rmse(base.positions_m[-1], points)
        refinement = _rmse(base.positions_m, refined_shared) / max(response, 1.0e-15)
        mesh_sensitivity = _rmse(base.positions_m[-1], coarse.positions_m[-1]) / max(
            response, 1.0e-15
        )
        poisson_sensitivity = _rmse(
            low_poisson.positions_m[-1], high_poisson.positions_m[-1]
        )
        young_invariance = _rmse(low_young.positions_m[-1], high_young.positions_m[-1])
        parity = _rmse(base.positions_m[0], incumbent["prediction_m"][0])
        maximum_displacement = float(
            np.max(np.linalg.norm(base.positions_m - points[None], axis=2))
        )
        contact_error = float(
            np.max(np.linalg.norm(contact.projected_targets_m - raw_targets, axis=2))
        )
        determinants = np.concatenate(
            tuple(
                replay.deformation_determinants.reshape(-1)
                for replay in (
                    base,
                    zero,
                    transformed,
                    refined,
                    coarse,
                    low_poisson,
                    high_poisson,
                    low_young,
                    high_young,
                )
            )
        )
        finite = bool(
            all(
                np.all(np.isfinite(replay.positions_m))
                for replay in (
                    base,
                    repeat,
                    zero,
                    transformed,
                    refined,
                    coarse,
                    low_poisson,
                    high_poisson,
                    low_young,
                    high_young,
                )
            )
            and np.all(np.isfinite(determinants))
        )
        group_sanity = {
            "finite": finite,
            "action_response": response >= float(gates["minimum_action_response_m"]),
            "poisson_sensitivity_lower": poisson_sensitivity
            >= float(gates["minimum_poisson_sensitivity_m"]),
            "poisson_sensitivity_upper": poisson_sensitivity
            <= float(gates["maximum_poisson_sensitivity_m"]),
            "young_modulus_unidentifiable_under_dirichlet_loading": young_invariance
            <= float(gates["maximum_young_modulus_invariance_error_m"]),
            "mesh_connectivity_sensitivity": mesh_sensitivity
            <= float(gates["maximum_mesh_connectivity_sensitivity_relative_error"]),
            "contact_projection": contact_error
            <= float(gates["maximum_contact_projection_error_m"]),
            "node_displacement": maximum_displacement
            <= float(gates["maximum_node_displacement_m"]),
            "deformation_determinant_lower": float(np.min(determinants))
            >= float(gates["minimum_deformation_determinant"]),
            "deformation_determinant_upper": float(np.max(determinants))
            <= float(gates["maximum_deformation_determinant"]),
        }
        sanity_violations += sum(not value for value in group_sanity.values())
        topology = base_topology and coarse_topology and contact_topology
        units_valid = bool(
            base.positions_m.shape == (len(base_frames), group.material_node_count, 3)
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
                "rigid_transformed_driven_m": transformed.positions_m,
                "refined_driven_m": refined.positions_m,
                "coarse_mesh_driven_m": coarse.positions_m,
                "low_poisson_driven_m": low_poisson.positions_m,
                "high_poisson_driven_m": high_poisson.positions_m,
                "low_young_terminal_m": low_young.positions_m,
                "high_young_terminal_m": high_young.positions_m,
                "base_cells": base_cells,
                "coarse_cells": coarse_cells,
                "base_deformation_determinant": base.deformation_determinants,
                "contact_projected_targets_m": contact.projected_targets_m,
            },
        )
        fallback_path = group_output / FALLBACK_FILENAME
        shutil.copyfile(incumbent_path, fallback_path)
        fallback_exact = fallback_path.read_bytes() == incumbent_path.read_bytes()
        all_fallback = all_fallback and fallback_exact
        all_deterministic = all_deterministic and deterministic
        all_topology = all_topology and topology
        all_units = all_units and units_valid
        maximum_zero_drift = max(maximum_zero_drift, zero_drift)
        maximum_equivariance = max(maximum_equivariance, equivariance)
        maximum_refinement = max(maximum_refinement, refinement)
        maximum_parity = max(maximum_parity, parity)
        record = {
            "group_id": group.group_id,
            "source_inputs_sha256": group.source_inputs_sha256,
            "incumbent_sha256": group.incumbent_sha256,
            "trajectory_archive_sha256": file_sha256(archive_path),
            "fallback_sha256": file_sha256(fallback_path),
            "deterministic_replay_valid": deterministic,
            "topology_identity_preserved": topology,
            "units_coordinate_entity_order_valid": units_valid,
            "exact_fallback_verified": fallback_exact,
            "base_cell_count": len(base_cells),
            "coarse_cell_count": len(coarse_cells),
            "contact_patch_sizes": list(patch_sizes),
            "maximum_zero_action_drift_m": zero_drift,
            "maximum_rigid_equivariance_error_m": equivariance,
            "time_step_refinement_relative_error": refinement,
            "mesh_connectivity_sensitivity_relative_error": mesh_sensitivity,
            "source_query_parity_rmse_m": parity,
            "action_response_rmse_m": response,
            "poisson_sensitivity_rmse_m": poisson_sensitivity,
            "young_modulus_invariance_rmse_m": young_invariance,
            "maximum_contact_projection_error_m": contact_error,
            "maximum_node_displacement_m": maximum_displacement,
            "minimum_deformation_determinant": float(np.min(determinants)),
            "maximum_deformation_determinant": float(np.max(determinants)),
            "physical_sanity_checks": group_sanity,
        }
        group_results.append(record)
        source_evidence_records.append(
            {
                "group_id": group.group_id,
                "source_inputs_sha256": group.source_inputs_sha256,
                "incumbent_sha256": group.incumbent_sha256,
                "trajectory_archive_sha256": record["trajectory_archive_sha256"],
            }
        )

    source_evidence_id = content_id({"source_groups": source_evidence_records})
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
            "engine_revision": protocol.backend["engine_revision"],
            "engine_version": protocol.backend["engine_version"],
            "native_smoke_id": protocol.backend["native_smoke_id"],
            "constitutive_model": simulation["constitutive_model"],
            "contact_boundary_policy": simulation["contact_boundary_policy"],
            "mesh_policy": simulation["mesh_policy"],
            "rigid_equivariance_probe": "fixed-se3-coordinate-transform",
            "time_step_refinement_interpretation": simulation[
                "load_step_refinement_semantics"
            ],
            "young_modulus_identifiability": (
                "structurally-unidentifiable-under-displacement-only-"
                "quasistatic-dirichlet-loading"
            ),
            "parameter_ensemble_axis": "poisson-ratio-only",
            "gradient_claim": "none",
        },
    )
    save_material_backend_qualification_v1(
        qualification, output / QUALIFICATION_FILENAME
    )
    identity: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "claim_boundary": protocol.value["claim_boundary"],
        "protocol_sha256": protocol.protocol_sha256,
        "runtime_id": protocol.runtime_id,
        "implementation": provenance,
        "source_groups": group_results,
        "qualification_artifact_id": qualification.artifact_id,
        "qualified": qualification.qualified,
        "failure_reasons": list(qualification.failure_reasons),
        "source_value_scoring_authorized": qualification.qualified,
        "information_boundary": {
            "source_inputs_read": True,
            "incumbent_predictions_read": True,
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
    "JaxFemSourceGroupV1",
    "JaxFemSourcePhysicsProtocolV1",
    "QUALIFICATION_FILENAME",
    "RESULT_FILENAME",
    "RigidContactProjectionV1",
    "attachment_targets_m",
    "build_tetrahedral_cells_v1",
    "contact_patch_local_indices_v1",
    "deformation_determinants_v1",
    "file_sha256",
    "load_jax_fem_source_inputs_v1",
    "load_jax_fem_source_physics_protocol_v1",
    "mesh_component_count_v1",
    "rigid_contact_projection_v1",
    "rigid_transform_v1",
    "run_jax_fem_source_qualification_v1",
]
