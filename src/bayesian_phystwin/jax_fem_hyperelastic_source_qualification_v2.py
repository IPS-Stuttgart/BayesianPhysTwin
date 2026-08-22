"""Frozen source-only qualification for the JAX-FEM hyperelastic v2 arm."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._portable_contracts import content_id, load_strict_json_object, write_atomic_json
from .jax_fem_hyperelastic_v2 import (
    BACKEND_VARIANT,
    CONSTITUTIVE_MODEL,
    CONTINUATION_POLICY,
    NONLINEAR_SOLVER,
    HyperelasticReplayV2,
    load_native_jax_fem_modules_v2,
    normalized_objectivity_errors_v2,
    run_hyperelastic_replay_v2,
)
from .jax_fem_source_qualification_v1 import (
    FALLBACK_FILENAME,
    JaxFemSourcePhysicsProtocolV1,
    attachment_targets_m,
    build_tetrahedral_cells_v1,
    contact_patch_local_indices_v1,
    file_sha256,
    load_jax_fem_source_inputs_v1,
    load_jax_fem_source_physics_protocol_v1,
    mesh_component_count_v1,
    rigid_contact_projection_v1,
    rigid_transform_v1,
)
from .material_backend_qualification_v1 import (
    MaterialBackendQualificationV1,
    save_material_backend_qualification_v1,
)
from .physical_rollout_v1 import load_physical_rollout_archive, write_deterministic_npz

FloatArray: TypeAlias = npt.NDArray[np.float64]

PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin.jax-fem-hyperelastic-source-physics-protocol-v2"
)
RESULT_SCHEMA: Final = "bayesian-phystwin.jax-fem-hyperelastic-source-physics-result-v2"
RESULT_FILENAME: Final = "jax-fem-hyperelastic-source-physics-result-v2.json"
QUALIFICATION_FILENAME: Final = "material-backend-qualification.json"
GROUP_ARCHIVE_FILENAME: Final = (
    "jax-fem-hyperelastic-source-physics-trajectories-v2.npz"
)

_PROTOCOL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_label",
        "claim_boundary",
        "base_protocol",
        "backend",
        "simulation",
        "gates",
        "information_boundary",
    }
)
_BASE_PROTOCOL_FIELDS: Final = frozenset({"relative_path", "sha256"})
_BACKEND_FIELDS: Final = frozenset(
    {
        "canonical_profile_id",
        "producer_profile_id",
        "transport",
        "backend_variant",
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
_SOURCE_FILES: Final = frozenset(
    {
        "jax_fem/__init__.py",
        "jax_fem/basis.py",
        "jax_fem/fe.py",
        "jax_fem/generate_mesh.py",
        "jax_fem/problem.py",
        "jax_fem/solver.py",
    }
)
_RUNTIME_VERSION_FIELDS: Final = frozenset(
    {
        "python",
        "jax",
        "jaxlib",
        "jax_fem",
        "numpy",
        "scipy",
        "petsc4py",
        "gmsh",
        "meshio",
    }
)
_SIMULATION_FIELDS: Final = frozenset(
    {
        "backend",
        "precision",
        "seed",
        "qualification_frame_count",
        "sampled_frame_indices",
        "base_interval_substeps",
        "refined_interval_substeps",
        "element_type",
        "constitutive_model",
        "young_modulus_pa",
        "young_modulus_probe_low_pa",
        "young_modulus_probe_high_pa",
        "low_poisson_ratio",
        "base_poisson_ratio",
        "high_poisson_ratio",
        "solver",
        "newton_absolute_tolerance",
        "newton_relative_tolerance",
        "line_search",
        "hard_minimum_deformation_determinant",
        "rotation_interpolation",
        "continuation_policy",
    }
)
_GATE_FIELDS: Final = frozenset(
    {
        "maximum_zero_action_drift_m",
        "maximum_rigid_equivariance_error_m",
        "maximum_continuation_refinement_relative_error",
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
        "maximum_normalized_rest_stress_error",
        "maximum_normalized_rigid_rotation_stress_error",
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


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical nonempty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _canonical_string(value, name=name)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _git_revision(value: object, *, name: str) -> str:
    result = _canonical_string(value, name=name)
    if len(result) != 40 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a full lowercase Git revision")
    return result


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


def _canonical_relative_path(value: object, *, name: str) -> PurePosixPath:
    text = _canonical_string(value, name=name)
    path = PurePosixPath(text)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require("\\" not in text, f"{name} must use POSIX separators")
    _require(
        path.as_posix() == text
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} is not canonical",
    )
    return path


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty integer list")
    result = tuple(_nonnegative_int(item, name=f"{name}[]") for item in value)
    _require(tuple(sorted(set(result))) == result, f"{name} must be sorted and unique")
    return result


def runtime_descriptor_v2(
    backend: Mapping[str, Any],
    simulation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the domain-separated exact runtime descriptor."""

    return {
        "schema": "bayesian-phystwin.jax-fem-hyperelastic-runtime-v2",
        "canonical_profile_id": backend["canonical_profile_id"],
        "backend_variant": backend["backend_variant"],
        "engine_repository": backend["engine_repository"],
        "engine_revision": backend["engine_revision"],
        "engine_version": backend["engine_version"],
        "installed_source_sha256": backend["installed_source_sha256"],
        "runtime_versions": backend["runtime_versions"],
        "constitutive_model": simulation["constitutive_model"],
        "nonlinear_solver": simulation["solver"],
        "continuation": simulation["continuation_policy"],
    }


@dataclass(frozen=True, slots=True)
class JaxFemHyperelasticSourceProtocolV2:
    value: Mapping[str, Any]
    protocol_path: Path
    protocol_sha256: str
    base_protocol: JaxFemSourcePhysicsProtocolV1
    runtime_id: str

    @property
    def backend(self) -> Mapping[str, Any]:
        return _mapping(self.value["backend"], name="backend")

    @property
    def simulation(self) -> Mapping[str, Any]:
        return _mapping(self.value["simulation"], name="simulation")

    @property
    def gates(self) -> Mapping[str, Any]:
        return _mapping(self.value["gates"], name="gates")


def load_jax_fem_hyperelastic_source_protocol_v2(
    path: str | Path,
) -> JaxFemHyperelasticSourceProtocolV2:
    source = Path(path).absolute()
    value = load_strict_json_object(
        source,
        label="JAX-FEM hyperelastic source-physics protocol v2",
    )
    _exact_fields(value, _PROTOCOL_FIELDS, "protocol")
    _require(value["schema"] == PROTOCOL_SCHEMA, "protocol schema changed")
    _require(value["schema_version"] == 2, "protocol version changed")
    _canonical_string(value["protocol_label"], name="protocol_label")
    _canonical_string(value["claim_boundary"], name="claim_boundary")

    base = _mapping(value["base_protocol"], name="base_protocol")
    _exact_fields(base, _BASE_PROTOCOL_FIELDS, "base_protocol")
    relative = _canonical_relative_path(
        base["relative_path"], name="base_protocol.relative_path"
    )
    base_path = source.parent / relative.as_posix()
    _require(
        base_path.is_file() and not base_path.is_symlink(),
        "base protocol is unavailable",
    )
    _require(
        file_sha256(base_path) == _sha256(base["sha256"], name="base_protocol.sha256"),
        "base protocol SHA-256 changed",
    )
    base_protocol = load_jax_fem_source_physics_protocol_v1(base_path)

    backend = _mapping(value["backend"], name="backend")
    _exact_fields(backend, _BACKEND_FIELDS, "backend")
    _require(
        backend["canonical_profile_id"] == base_protocol.canonical_profile_id
        and backend["producer_profile_id"] == base_protocol.producer_profile_id
        and backend["transport"] == base_protocol.transport,
        "v2 backend family diverged from the frozen base protocol",
    )
    _require(backend["backend_variant"] == BACKEND_VARIANT, "backend variant changed")
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
    sources = _mapping(
        backend["installed_source_sha256"], name="installed_source_sha256"
    )
    _require(frozenset(sources) == _SOURCE_FILES, "installed source roster changed")
    for name, digest in sources.items():
        _sha256(digest, name=f"installed_source_sha256[{name}]")
    versions = _mapping(backend["runtime_versions"], name="runtime_versions")
    _exact_fields(versions, _RUNTIME_VERSION_FIELDS, "runtime_versions")
    for name, version in versions.items():
        _canonical_string(version, name=f"runtime_versions[{name}]")

    simulation = _mapping(value["simulation"], name="simulation")
    _exact_fields(simulation, _SIMULATION_FIELDS, "simulation")
    _require(simulation["backend"] == "cpu", "simulation backend changed")
    _require(simulation["precision"] == "64", "simulation precision changed")
    _nonnegative_int(simulation["seed"], name="seed")
    frame_count = _positive_int(
        simulation["qualification_frame_count"],
        name="qualification_frame_count",
    )
    sampled = _integer_tuple(
        simulation["sampled_frame_indices"], name="sampled_frame_indices"
    )
    _require(sampled[0] == 0 and sampled[-1] < frame_count, "sampled frames changed")
    base_steps = _positive_int(
        simulation["base_interval_substeps"], name="base_interval_substeps"
    )
    refined_steps = _positive_int(
        simulation["refined_interval_substeps"], name="refined_interval_substeps"
    )
    _require(base_steps == 1 and refined_steps == 2, "continuation roster changed")
    _require(simulation["element_type"] == "TET4", "element type changed")
    _require(
        simulation["constitutive_model"] == CONSTITUTIVE_MODEL,
        "constitutive model changed",
    )
    _require(simulation["solver"] == NONLINEAR_SOLVER, "nonlinear solver changed")
    _require(simulation["line_search"] is True, "line search must remain enabled")
    _require(
        simulation["rotation_interpolation"]
        == "linear-blend-polar-projection-to-SO3-v2",
        "rotation interpolation changed",
    )
    _require(
        simulation["continuation_policy"] == CONTINUATION_POLICY,
        "continuation policy changed",
    )
    for name in (
        "young_modulus_pa",
        "young_modulus_probe_low_pa",
        "young_modulus_probe_high_pa",
        "newton_absolute_tolerance",
        "newton_relative_tolerance",
        "hard_minimum_deformation_determinant",
    ):
        _finite(simulation[name], name=name, positive=True)
    poisson = tuple(
        _finite(simulation[name], name=name)
        for name in ("low_poisson_ratio", "base_poisson_ratio", "high_poisson_ratio")
    )
    _require(0.0 < poisson[0] < poisson[1] < poisson[2] < 0.5, "Poisson roster changed")
    _require(
        frame_count == int(base_protocol.simulation["qualification_frame_count"]),
        "qualification frame count diverged from the frozen base protocol",
    )

    gates = _mapping(value["gates"], name="gates")
    _exact_fields(gates, _GATE_FIELDS, "gates")
    for name, gate in gates.items():
        _finite(gate, name=name, positive=True)
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
    _require(
        float(simulation["hard_minimum_deformation_determinant"])
        < float(gates["minimum_deformation_determinant"]),
        "hard determinant floor must be below the scientific gate",
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
    runtime_id = _sha256(backend["runtime_id"], name="runtime_id")
    _require(
        runtime_id == content_id(runtime_descriptor_v2(backend, simulation)),
        "runtime_id does not match the v2 runtime descriptor",
    )
    return JaxFemHyperelasticSourceProtocolV2(
        value=value,
        protocol_path=source,
        protocol_sha256=file_sha256(source),
        base_protocol=base_protocol,
        runtime_id=runtime_id,
    )


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    source_paths = (
        "src/bayesian_phystwin/jax_fem_hyperelastic_v2.py",
        "src/bayesian_phystwin/jax_fem_hyperelastic_source_qualification_v2.py",
        "scripts/remote/run_jax_fem_hyperelastic_source_qualification_v2.py",
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
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def _sample(replay: HyperelasticReplayV2, indices: tuple[int, ...]) -> FloatArray:
    return np.ascontiguousarray(replay.positions_m[np.asarray(indices)])


_run_native_replay_v2 = run_hyperelastic_replay_v2


def run_jax_fem_hyperelastic_source_qualification_v2(
    *,
    protocol_path: str | Path,
    group_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Run the frozen target-blind v2 source qualification exactly once."""

    protocol = load_jax_fem_hyperelastic_source_protocol_v2(protocol_path)
    base_protocol = protocol.base_protocol
    if set(group_roots) != {group.group_id for group in base_protocol.source_groups}:
        raise ValueError("group roots must match the complete frozen source roster")
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = _git_provenance(Path(repo_root).absolute())
    backend = protocol.backend
    simulation = protocol.simulation
    gates = protocol.gates
    native = load_native_jax_fem_modules_v2(
        runtime_versions=cast(Mapping[str, str], backend["runtime_versions"]),
        installed_source_sha256=cast(
            Mapping[str, str], backend["installed_source_sha256"]
        ),
    )
    rest_stress_error, rotation_stress_error = normalized_objectivity_errors_v2(
        native,
        young_modulus_pa=float(simulation["young_modulus_pa"]),
        poisson_ratio=float(simulation["base_poisson_ratio"]),
    )

    base_simulation = base_protocol.simulation
    frame_count = int(simulation["qualification_frame_count"])
    sampled_frames = cast(tuple[int, ...], tuple(simulation["sampled_frame_indices"]))
    group_results: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
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
        base_simulation["rigid_transform_rotation_axis"],
        float(base_simulation["rigid_transform_angle_rad"]),
    )
    translation = np.asarray(
        base_simulation["rigid_transform_translation_m"], dtype=np.float64
    )

    for group in base_protocol.source_groups:
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
            incumbent_path,
            expected_frame_count=group.frame_count,
        )
        points = np.asarray(arrays["frame_zero_points_m"], dtype=np.float64)
        controller = np.asarray(
            arrays["controller_points_m"][:frame_count], dtype=np.float64
        )
        indices = np.asarray(arrays["attachment_indices"], dtype=np.int64)
        weights = np.asarray(arrays["attachment_weights"], dtype=np.float64)
        raw_targets = attachment_targets_m(points, controller, indices, weights)
        patches = contact_patch_local_indices_v1(
            points,
            indices,
            radius_m=float(base_simulation["contact_cluster_radius_m"]),
        )
        contact = rigid_contact_projection_v1(points, indices, raw_targets, patches)
        patch_sizes = tuple(len(patch) for patch in patches)
        base_cells = build_tetrahedral_cells_v1(
            points,
            maximum_edge_m=float(base_simulation["base_mesh_max_edge_m"]),
            minimum_shape_ratio=float(
                base_simulation["minimum_tetrahedron_shape_ratio"]
            ),
        )
        coarse_cells = build_tetrahedral_cells_v1(
            points,
            maximum_edge_m=float(base_simulation["coarse_mesh_max_edge_m"]),
            minimum_shape_ratio=float(
                base_simulation["minimum_tetrahedron_shape_ratio"]
            ),
        )
        topology = bool(
            len(base_cells) == group.expected_base_cell_count
            and len(coarse_cells) == group.expected_coarse_cell_count
            and len(np.unique(base_cells)) == group.material_node_count
            and len(np.unique(coarse_cells)) == group.material_node_count
            and mesh_component_count_v1(base_cells, node_count=len(points)) == 1
            and mesh_component_count_v1(coarse_cells, node_count=len(points)) == 1
            and patch_sizes == group.expected_contact_patch_sizes
            and contact.patch_ranks == tuple(3 for _ in patches)
        )

        common: dict[str, Any] = {
            "native": native,
            "points_m": points,
            "attachment_indices": indices,
            "contact": contact,
            "newton_absolute_tolerance": float(simulation["newton_absolute_tolerance"]),
            "newton_relative_tolerance": float(simulation["newton_relative_tolerance"]),
            "hard_minimum_deformation_determinant": float(
                simulation["hard_minimum_deformation_determinant"]
            ),
        }

        def replay(
            *,
            cells: npt.ArrayLike = base_cells,
            substeps: int = int(simulation["base_interval_substeps"]),
            young: float = float(simulation["young_modulus_pa"]),
            poisson: float = float(simulation["base_poisson_ratio"]),
            driven: bool = True,
            overrides: Mapping[str, Any] | None = None,
            common_args: Mapping[str, Any] = common,
        ) -> HyperelasticReplayV2:
            arguments = {**common_args, **dict(overrides or {})}
            return _run_native_replay_v2(
                **arguments,
                cells=cells,
                interval_substeps=substeps,
                young_modulus_pa=young,
                poisson_ratio=poisson,
                driven=driven,
            )

        base = replay()
        repeat = replay()
        zero = replay(driven=False)
        refined = replay(substeps=int(simulation["refined_interval_substeps"]))
        coarse = replay(cells=coarse_cells)
        low_poisson = replay(poisson=float(simulation["low_poisson_ratio"]))
        high_poisson = replay(poisson=float(simulation["high_poisson_ratio"]))
        low_young = replay(young=float(simulation["young_modulus_probe_low_pa"]))
        high_young = replay(young=float(simulation["young_modulus_probe_high_pa"]))

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
        transformed = replay(
            overrides={
                "points_m": transformed_points,
                "contact": transformed_contact,
            }
        )

        deterministic = bool(
            np.array_equal(base.positions_m, repeat.positions_m)
            and np.array_equal(
                base.deformation_determinants,
                repeat.deformation_determinants,
            )
        )
        zero_drift = float(
            np.max(np.linalg.norm(zero.positions_m - points[None], axis=2))
        )
        inverse_transformed = (transformed.positions_m - translation) @ rotation
        equivariance = float(
            np.max(np.linalg.norm(inverse_transformed - base.positions_m, axis=2))
        )
        response = _rmse(base.positions_m[-1], points)
        refinement = _rmse(base.positions_m, refined.positions_m) / max(
            response, 1.0e-15
        )
        mesh_sensitivity = _rmse(base.positions_m[-1], coarse.positions_m[-1]) / max(
            response, 1.0e-15
        )
        poisson_sensitivity = _rmse(
            low_poisson.positions_m[-1], high_poisson.positions_m[-1]
        )
        young_invariance = _rmse(low_young.positions_m[-1], high_young.positions_m[-1])
        parity = _rmse(base.positions_m[0], incumbent["prediction_m"][0])
        contact_error = float(
            np.max(np.linalg.norm(contact.projected_targets_m - raw_targets, axis=2))
        )
        maximum_displacement = float(
            np.max(np.linalg.norm(base.positions_m - points[None], axis=2))
        )
        determinants = np.concatenate(
            [
                item.deformation_determinants.reshape(-1)
                for item in (
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
            ]
        )
        finite = bool(
            all(
                np.all(np.isfinite(item.positions_m))
                for item in (
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
        sanity = {
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
            "rest_stress_objectivity": rest_stress_error
            <= float(gates["maximum_normalized_rest_stress_error"]),
            "rigid_rotation_stress_objectivity": rotation_stress_error
            <= float(gates["maximum_normalized_rigid_rotation_stress_error"]),
        }
        sanity_violations += sum(not value for value in sanity.values())
        units_valid = bool(
            base.positions_m.shape == (frame_count, group.material_node_count, 3)
            and base.positions_m.dtype == np.float64
            and parity <= float(gates["maximum_source_query_parity_rmse_m"])
        )

        group_output = output / group.group_id
        group_output.mkdir()
        archive_path = write_deterministic_npz(
            group_output / GROUP_ARCHIVE_FILENAME,
            {
                "sampled_frame_indices": np.asarray(sampled_frames, dtype=np.int32),
                "base_driven_m": _sample(base, sampled_frames),
                "base_repeat_m": _sample(repeat, sampled_frames),
                "zero_action_m": _sample(zero, sampled_frames),
                "rigid_transformed_driven_m": _sample(transformed, sampled_frames),
                "refined_driven_m": _sample(refined, sampled_frames),
                "coarse_mesh_driven_m": _sample(coarse, sampled_frames),
                "low_poisson_driven_m": _sample(low_poisson, sampled_frames),
                "high_poisson_driven_m": _sample(high_poisson, sampled_frames),
                "low_young_driven_m": _sample(low_young, sampled_frames),
                "high_young_driven_m": _sample(high_young, sampled_frames),
                "base_cells": base_cells,
                "coarse_cells": coarse_cells,
                "base_deformation_determinant": base.deformation_determinants,
                "contact_projected_targets_m": contact.projected_targets_m,
            },
        )
        fallback_path = group_output / FALLBACK_FILENAME
        shutil.copyfile(incumbent_path, fallback_path)
        fallback_exact = fallback_path.read_bytes() == incumbent_path.read_bytes()
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
            "base_native_solve_count": base.native_solve_count,
            "refined_native_solve_count": refined.native_solve_count,
            "maximum_zero_action_drift_m": zero_drift,
            "maximum_rigid_equivariance_error_m": equivariance,
            "continuation_refinement_relative_error": refinement,
            "mesh_connectivity_sensitivity_relative_error": mesh_sensitivity,
            "source_query_parity_rmse_m": parity,
            "action_response_rmse_m": response,
            "poisson_sensitivity_rmse_m": poisson_sensitivity,
            "young_modulus_invariance_rmse_m": young_invariance,
            "maximum_contact_projection_error_m": contact_error,
            "maximum_node_displacement_m": maximum_displacement,
            "minimum_deformation_determinant": float(np.min(determinants)),
            "maximum_deformation_determinant": float(np.max(determinants)),
            "minimum_continuation_deformation_determinant": min(
                item.minimum_continuation_deformation_determinant
                for item in (
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
            ),
            "physical_sanity_checks": sanity,
        }
        group_results.append(record)
        evidence_records.append(
            {
                "group_id": group.group_id,
                "source_inputs_sha256": group.source_inputs_sha256,
                "incumbent_sha256": group.incumbent_sha256,
                "trajectory_archive_sha256": record["trajectory_archive_sha256"],
            }
        )
        all_deterministic = all_deterministic and deterministic
        all_topology = all_topology and topology
        all_fallback = all_fallback and fallback_exact
        all_units = all_units and units_valid
        maximum_zero_drift = max(maximum_zero_drift, zero_drift)
        maximum_equivariance = max(maximum_equivariance, equivariance)
        maximum_refinement = max(maximum_refinement, refinement)
        maximum_parity = max(maximum_parity, parity)

    source_evidence_id = content_id(
        {
            "schema": "bayesian-phystwin.jax-fem-hyperelastic-source-evidence-v2",
            "base_protocol_sha256": base_protocol.protocol_sha256,
            "v2_protocol_sha256": protocol.protocol_sha256,
            "source_groups": evidence_records,
        }
    )
    incumbent_runtime_id = content_id(
        {
            "incumbent_archives": [
                {"group_id": group.group_id, "sha256": group.incumbent_sha256}
                for group in base_protocol.source_groups
            ]
        }
    )
    qualification = MaterialBackendQualificationV1(
        canonical_profile_id=base_protocol.canonical_profile_id,
        producer_profile_id=base_protocol.producer_profile_id,
        transport=base_protocol.transport,
        runtime_id=protocol.runtime_id,
        qualification_protocol_id=protocol.protocol_sha256,
        source_evidence_id=source_evidence_id,
        source_group_ids=tuple(group.group_id for group in base_protocol.source_groups),
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
            gates["maximum_continuation_refinement_relative_error"]
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
            "evidence_role": "already-open-source-inputs-only",
            "backend_variant": BACKEND_VARIANT,
            "base_protocol_sha256": base_protocol.protocol_sha256,
            "engine_revision": backend["engine_revision"],
            "engine_version": backend["engine_version"],
            "native_smoke_id": backend["native_smoke_id"],
            "constitutive_model": simulation["constitutive_model"],
            "contact_boundary_policy": base_simulation["contact_boundary_policy"],
            "mesh_policy": base_simulation["mesh_policy"],
            "nonlinear_solver": simulation["solver"],
            "continuation_policy": simulation["continuation_policy"],
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
        "base_protocol_sha256": base_protocol.protocol_sha256,
        "protocol_sha256": protocol.protocol_sha256,
        "runtime_id": protocol.runtime_id,
        "implementation": provenance,
        "normalized_rest_stress_error": rest_stress_error,
        "normalized_rigid_rotation_stress_error": rotation_stress_error,
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
    "GROUP_ARCHIVE_FILENAME",
    "JaxFemHyperelasticSourceProtocolV2",
    "PROTOCOL_SCHEMA",
    "QUALIFICATION_FILENAME",
    "RESULT_FILENAME",
    "RESULT_SCHEMA",
    "load_jax_fem_hyperelastic_source_protocol_v2",
    "run_jax_fem_hyperelastic_source_qualification_v2",
    "runtime_descriptor_v2",
]
