"""Outcome-gated source-value experiment for the qualified JAX-FEM runtime."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ._portable_contracts import content_id, load_strict_json_object, write_atomic_json
from .jax_fem_source_qualification_v1 import (
    _git_provenance,
    _load_native_modules,
    _run_native_replay,
    attachment_targets_m,
    build_tetrahedral_cells_v1,
    contact_patch_local_indices_v1,
    file_sha256,
    load_jax_fem_source_inputs_v1,
    load_jax_fem_source_physics_protocol_v1,
    rigid_contact_projection_v1,
)
from .material_backend_qualification_v1 import (
    load_material_backend_qualification_v1,
    require_qualified_material_backend_runtime,
)
from .newton_mpm_source_gate_v1 import _coordinate_rmse, _symmetric_chamfer
from .physical_rollout_v1 import load_physical_rollout_archive, write_deterministic_npz

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

PROTOCOL_SCHEMA: Final = "bayesian-phystwin.jax-fem-source-value-protocol"
GRID_SCHEMA: Final = "bayesian-phystwin.jax-fem-source-value-grid"
PREFIX_SCHEMA: Final = "bayesian-phystwin.jax-fem-source-value-prefix-result"
FUTURE_SCHEMA: Final = "bayesian-phystwin.jax-fem-source-value-future-result"
PRE_PREFIX_SCHEMA: Final = "bayesian-phystwin.jax-fem-source-value-pre-prefix-result"
GRID_FILENAME: Final = "jax-fem-source-value-grid.json"
PREFIX_FILENAME: Final = "jax-fem-source-value-prefix-result.json"
FUTURE_FILENAME: Final = "jax-fem-source-value-future-result.json"
PRE_PREFIX_FILENAME: Final = "jax-fem-source-value-pre-prefix-result.json"
SELECTED_FILENAME: Final = "selected-physical-prediction.npz"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields changed")


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical nonempty string")
    return value


def _digest(value: object, *, name: str) -> str:
    text = _string(value, name=name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _relative(value: object, *, name: str) -> PurePosixPath:
    text = _string(value, name=name)
    path = PurePosixPath(text)
    _require(
        not path.is_absolute() and "\\" not in text, f"{name} must be POSIX-relative"
    )
    _require(
        path.as_posix() == text
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} is not canonical",
    )
    return path


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    _require(
        np.isfinite(result) and (not positive or result > 0.0), f"{name} must be finite"
    )
    return result


@dataclass(frozen=True, slots=True)
class JaxFemValueGroupV1:
    group_id: str
    source_inputs_relative_path: PurePosixPath
    source_inputs_sha256: str
    prefix_outcomes_relative_path: PurePosixPath
    prefix_outcomes_sha256: str
    future_outcomes_relative_path: PurePosixPath
    future_outcomes_sha256: str
    incumbent_relative_path: PurePosixPath
    incumbent_sha256: str
    matphys_sha256: str
    frame_count: int
    material_particle_count: int
    controller_point_count: int
    attached_particle_count: int
    base_cell_count: int
    contact_patch_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class JaxFemSourceValueProtocolV1:
    value: Mapping[str, Any]
    sha256: str
    runtime_id: str
    qualification_protocol_sha256: str
    qualification_result_sha256: str
    qualification_artifact_sha256: str
    qualification_artifact_id: str
    groups: tuple[JaxFemValueGroupV1, ...]
    poisson_ratios: tuple[float, ...]
    young_modulus_pa: float
    weights: tuple[float, ...]

    @property
    def candidate(self) -> Mapping[str, Any]:
        return _mapping(self.value["candidate"], name="candidate")

    @property
    def gates(self) -> Mapping[str, Any]:
        return _mapping(self.value["validation_gates"], name="validation_gates")


def load_jax_fem_source_value_protocol_v1(
    path: str | Path,
) -> JaxFemSourceValueProtocolV1:
    source = Path(path)
    value = load_strict_json_object(source, label="JaxFem source-value protocol")
    _exact_fields(
        value,
        {
            "schema",
            "schema_version",
            "protocol_label",
            "claim_boundary",
            "qualification",
            "source_groups",
            "candidate",
            "validation_gates",
            "information_boundary",
        },
        name="protocol",
    )
    _require(
        value["schema"] == PROTOCOL_SCHEMA and value["schema_version"] == 1,
        "protocol identity changed",
    )
    qualification = _mapping(value["qualification"], name="qualification")
    _exact_fields(
        qualification,
        {
            "runtime_id",
            "source_physics_protocol_sha256",
            "source_physics_result_sha256",
            "qualification_artifact_sha256",
            "qualification_artifact_id",
        },
        name="qualification",
    )
    runtime_id = _digest(qualification["runtime_id"], name="runtime_id")
    physics_protocol_sha = _digest(
        qualification["source_physics_protocol_sha256"],
        name="source physics protocol",
    )
    result_sha = _digest(
        qualification["source_physics_result_sha256"], name="source physics result"
    )
    artifact_sha = _digest(
        qualification["qualification_artifact_sha256"], name="qualification artifact"
    )
    artifact_id = _digest(
        qualification["qualification_artifact_id"], name="qualification artifact ID"
    )

    raw_groups = value["source_groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) != 2:
        raise ValueError("source-value protocol requires exactly two groups")
    group_fields = {
        "group_id",
        "source_inputs_relative_path",
        "source_inputs_sha256",
        "prefix_outcomes_relative_path",
        "prefix_outcomes_sha256",
        "future_outcomes_relative_path",
        "future_outcomes_sha256",
        "incumbent_relative_path",
        "incumbent_sha256",
        "matphys_sha256",
        "frame_count",
        "material_particle_count",
        "controller_point_count",
        "attached_particle_count",
        "base_cell_count",
        "contact_patch_sizes",
    }
    groups: list[JaxFemValueGroupV1] = []
    for raw in raw_groups:
        group = _mapping(raw, name="source group")
        _exact_fields(group, group_fields, name="source group")
        frame_count = group["frame_count"]
        particle_count = group["material_particle_count"]
        controller_count = group["controller_point_count"]
        attached_count = group["attached_particle_count"]
        base_cell_count = group["base_cell_count"]
        raw_patch_sizes = group["contact_patch_sizes"]
        _require(type(frame_count) is int and frame_count >= 2, "frame_count changed")
        _require(
            type(particle_count) is int and particle_count >= 1,
            "particle count changed",
        )
        _require(
            type(controller_count) is int and controller_count >= 1,
            "controller count changed",
        )
        _require(
            type(attached_count) is int and attached_count >= 1,
            "attached count changed",
        )
        _require(
            type(base_cell_count) is int and base_cell_count >= 1,
            "base cell count changed",
        )
        _require(
            isinstance(raw_patch_sizes, list)
            and len(raw_patch_sizes) >= 1
            and all(type(size) is int and size >= 1 for size in raw_patch_sizes)
            and sum(raw_patch_sizes) == attached_count,
            "contact patch sizes changed",
        )
        groups.append(
            JaxFemValueGroupV1(
                group_id=_string(group["group_id"], name="group_id"),
                source_inputs_relative_path=_relative(
                    group["source_inputs_relative_path"], name="source inputs path"
                ),
                source_inputs_sha256=_digest(
                    group["source_inputs_sha256"], name="source inputs SHA-256"
                ),
                prefix_outcomes_relative_path=_relative(
                    group["prefix_outcomes_relative_path"], name="prefix outcomes path"
                ),
                prefix_outcomes_sha256=_digest(
                    group["prefix_outcomes_sha256"], name="prefix outcomes SHA-256"
                ),
                future_outcomes_relative_path=_relative(
                    group["future_outcomes_relative_path"], name="future outcomes path"
                ),
                future_outcomes_sha256=_digest(
                    group["future_outcomes_sha256"], name="future outcomes SHA-256"
                ),
                incumbent_relative_path=_relative(
                    group["incumbent_relative_path"], name="incumbent path"
                ),
                incumbent_sha256=_digest(
                    group["incumbent_sha256"], name="incumbent SHA-256"
                ),
                matphys_sha256=_digest(group["matphys_sha256"], name="MatPhys SHA-256"),
                frame_count=frame_count,
                material_particle_count=particle_count,
                controller_point_count=controller_count,
                attached_particle_count=attached_count,
                base_cell_count=base_cell_count,
                contact_patch_sizes=tuple(raw_patch_sizes),
            )
        )
    _require(len({group.group_id for group in groups}) == 2, "source group IDs changed")

    candidate = _mapping(value["candidate"], name="candidate")
    _exact_fields(
        candidate,
        {
            "poisson_ratio",
            "young_modulus_pa",
            "weights",
            "point_estimate",
            "distribution_score",
            "fit_fraction",
        },
        name="candidate",
    )
    raw_poisson = candidate["poisson_ratio"]
    raw_weights = candidate["weights"]
    if (
        not isinstance(raw_poisson, list)
        or not isinstance(raw_weights, list)
        or len(raw_poisson) != 3
        or len(raw_weights) != 3
    ):
        raise ValueError("candidate ensemble must contain exactly three members")
    poisson_ratios = tuple(
        _finite(value, name="Poisson ratio") for value in raw_poisson
    )
    young_modulus = _finite(
        candidate["young_modulus_pa"], name="young modulus", positive=True
    )
    weights = tuple(
        _finite(value, name="weight", positive=True) for value in raw_weights
    )
    _require(
        len(set(poisson_ratios)) == 3
        and -1.0 < poisson_ratios[0] < poisson_ratios[1] < poisson_ratios[2] < 0.5
        and np.isclose(sum(weights), 1.0),
        "candidate ensemble changed",
    )
    _require(
        candidate["point_estimate"] == "equal-weight-ensemble-mean",
        "point estimate changed",
    )
    _require(
        candidate["distribution_score"] == "equal-event-3d-marginal-energy-score",
        "distribution score changed",
    )
    _require(
        np.isclose(_finite(candidate["fit_fraction"], name="fit_fraction"), 2.0 / 3.0),
        "fit fraction changed",
    )

    gates = _mapping(value["validation_gates"], name="validation gates")
    gate_fields = {
        "maximum_equal_group_balanced_point_ratio_vs_persistence",
        "maximum_worst_group_balanced_point_ratio_vs_persistence",
        "maximum_equal_group_energy_ratio_vs_persistence",
        "maximum_equal_group_identity_ratio_vs_incumbent",
        "maximum_equal_group_chamfer_ratio_vs_incumbent",
        "minimum_final_ensemble_spread_m",
        "maximum_final_ensemble_spread_m",
        "maximum_full_horizon_contact_projection_error_m",
        "maximum_full_horizon_node_displacement_m",
        "minimum_full_horizon_deformation_determinant",
        "maximum_full_horizon_deformation_determinant",
        "required_successful_candidate_count",
    }
    _exact_fields(gates, gate_fields, name="validation gates")
    for name in gate_fields - {"required_successful_candidate_count"}:
        _finite(gates[name], name=name, positive=True)
    _require(
        gates["required_successful_candidate_count"] == 3,
        "candidate denominator changed",
    )
    boundary = _mapping(value["information_boundary"], name="information boundary")
    _require(
        boundary
        == {
            "prediction_uses_frame_zero_geometry_and_known_action_only": True,
            "prefix_outcomes_open_only_after_all_predictions_sealed": True,
            "future_outcomes_open_only_after_validation_gate": True,
            "target_or_held_out_artifact_access_allowed": False,
            "failure_policy": "byte-exact-incumbent-per-source-group",
            "no_replacement": True,
        },
        "information boundary changed",
    )
    return JaxFemSourceValueProtocolV1(
        value=value,
        sha256=file_sha256(source),
        runtime_id=runtime_id,
        qualification_protocol_sha256=physics_protocol_sha,
        qualification_result_sha256=result_sha,
        qualification_artifact_sha256=artifact_sha,
        qualification_artifact_id=artifact_id,
        groups=tuple(groups),
        poisson_ratios=poisson_ratios,
        young_modulus_pa=young_modulus,
        weights=weights,
    )


def _physics_simulation(physics_protocol_path: str | Path) -> Mapping[str, Any]:
    value = load_strict_json_object(
        physics_protocol_path, label="source-physics protocol"
    )
    return _mapping(value["simulation"], name="source-physics simulation")


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file() and not source.is_symlink(), f"{name} must be an ordinary file"
    )
    return source


def _ratio(numerator: float, denominator: float, *, name: str) -> float:
    _require(np.isfinite(numerator) and numerator >= 0.0, f"{name} numerator changed")
    _require(
        np.isfinite(denominator) and denominator > 0.0, f"{name} denominator changed"
    )
    return float(numerator / denominator)


def _physical_arrays(
    positions_m: FloatArray,
    *,
    frame_zero_m: FloatArray,
    action_support: FloatArray,
) -> dict[str, npt.NDArray[Any]]:
    prediction = np.ascontiguousarray(positions_m, dtype=np.float32)
    frame_zero = np.ascontiguousarray(frame_zero_m, dtype=np.float32)
    persistence = np.broadcast_to(frame_zero[None], prediction.shape).copy()
    return {
        "action_support": np.ascontiguousarray(action_support, dtype=np.float32),
        "driven_readout_m": prediction,
        "frame_zero_points_m": frame_zero,
        "persistence_m": persistence,
        "prediction_m": prediction,
        "zero_action_readout_m": persistence.copy(),
    }


def generate_jax_fem_source_value_predictions_v1(
    *,
    protocol_path: str | Path,
    physics_protocol_path: str | Path,
    physics_result_path: str | Path,
    qualification_path: str | Path,
    group_roots: Mapping[str, str | Path],
    matphys_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    protocol = load_jax_fem_source_value_protocol_v1(protocol_path)
    _require(
        file_sha256(physics_protocol_path) == protocol.qualification_protocol_sha256,
        "source-physics protocol changed",
    )
    _require(
        file_sha256(physics_result_path) == protocol.qualification_result_sha256,
        "source-physics result changed",
    )
    _require(
        file_sha256(qualification_path) == protocol.qualification_artifact_sha256,
        "qualification artifact changed",
    )
    qualification = load_material_backend_qualification_v1(qualification_path)
    _require(
        qualification.artifact_id == protocol.qualification_artifact_id,
        "qualification artifact ID changed",
    )
    require_qualified_material_backend_runtime(
        profile_id="jax-fem-quasistatic-v1",
        producer_profile_id="jax-fem-quasistatic-v1",
        runtime_id=protocol.runtime_id,
        qualification=qualification,
    )
    expected = {group.group_id for group in protocol.groups}
    _require(
        set(group_roots) == expected and set(matphys_paths) == expected,
        "complete source roots are required",
    )
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = _git_provenance(
        Path(repo_root).absolute(),
        source_paths=(
            "src/bayesian_phystwin/jax_fem_source_qualification_v1.py",
            "src/bayesian_phystwin/jax_fem_source_value_v1.py",
            "scripts/remote/run_jax_fem_source_value_v1.py",
        ),
    )

    physics_protocol = load_jax_fem_source_physics_protocol_v1(physics_protocol_path)
    _require(
        physics_protocol.runtime_id == protocol.runtime_id,
        "source-value runtime differs from source physics",
    )
    physics_groups = {group.group_id: group for group in physics_protocol.source_groups}
    _require(set(physics_groups) == expected, "source-physics group roster changed")
    simulation = physics_protocol.simulation
    _require(
        protocol.poisson_ratios
        == (
            float(simulation["low_poisson_ratio"]),
            float(simulation["base_poisson_ratio"]),
            float(simulation["high_poisson_ratio"]),
        )
        and protocol.young_modulus_pa == float(simulation["young_modulus_pa"]),
        "source-value ensemble differs from the qualified physics parameters",
    )
    native = _load_native_modules(physics_protocol)

    records: list[dict[str, Any]] = []
    for group in protocol.groups:
        root = Path(group_roots[group.group_id]).absolute()
        source_path = root / group.source_inputs_relative_path.as_posix()
        incumbent_path = root / group.incumbent_relative_path.as_posix()
        matphys_path = Path(matphys_paths[group.group_id]).absolute()
        _require(
            file_sha256(incumbent_path) == group.incumbent_sha256, "incumbent changed"
        )
        _require(file_sha256(matphys_path) == group.matphys_sha256, "MatPhys changed")
        physics_group = physics_groups[group.group_id]
        _require(
            physics_group.source_inputs_relative_path
            == group.source_inputs_relative_path
            and physics_group.source_inputs_sha256 == group.source_inputs_sha256
            and physics_group.incumbent_relative_path == group.incumbent_relative_path
            and physics_group.incumbent_sha256 == group.incumbent_sha256
            and physics_group.frame_count == group.frame_count
            and physics_group.material_node_count == group.material_particle_count
            and physics_group.controller_point_count == group.controller_point_count
            and physics_group.attached_node_count == group.attached_particle_count
            and physics_group.expected_base_cell_count == group.base_cell_count
            and physics_group.expected_contact_patch_sizes == group.contact_patch_sizes,
            "source-value group differs from the qualified source-physics group",
        )
        arrays = load_jax_fem_source_inputs_v1(source_path, group=physics_group)
        points = np.asarray(arrays["frame_zero_points_m"], dtype=np.float64)
        controller = np.asarray(arrays["controller_points_m"], dtype=np.float64)
        indices = np.asarray(arrays["attachment_indices"], dtype=np.int64)
        raw_targets = attachment_targets_m(
            points, controller, indices, arrays["attachment_weights"]
        )
        patches = contact_patch_local_indices_v1(
            points,
            indices,
            radius_m=float(simulation["contact_cluster_radius_m"]),
        )
        _require(
            tuple(len(patch) for patch in patches)
            == physics_group.expected_contact_patch_sizes,
            "contact patch topology changed",
        )
        contact = rigid_contact_projection_v1(
            points,
            indices,
            raw_targets,
            patches,
        )
        cells = build_tetrahedral_cells_v1(
            points,
            maximum_edge_m=float(simulation["base_mesh_max_edge_m"]),
            minimum_shape_ratio=float(simulation["minimum_tetrahedron_shape_ratio"]),
        )
        _require(
            len(cells) == physics_group.expected_base_cell_count,
            "source-value tetrahedral topology changed",
        )
        maximum_contact_error = float(
            np.max(np.linalg.norm(contact.projected_targets_m - raw_targets, axis=2))
        )
        group_dir = output / group.group_id
        group_dir.mkdir()
        member_arrays: list[dict[str, npt.NDArray[Any]]] = []
        members: list[dict[str, Any]] = []
        for index, poisson_ratio in enumerate(protocol.poisson_ratios):
            replay = _run_native_replay(
                native=native,
                points_m=points,
                cells=cells,
                attachment_indices=indices,
                contact=contact,
                frame_indices=tuple(range(group.frame_count)),
                young_modulus_pa=protocol.young_modulus_pa,
                poisson_ratio=poisson_ratio,
                driven=True,
            )
            physical = _physical_arrays(
                replay.positions_m,
                frame_zero_m=points,
                action_support=np.asarray(arrays["action_support"]),
            )
            member_path = write_deterministic_npz(
                group_dir / f"member-{index:02d}.npz", physical
            )
            member_arrays.append(physical)
            members.append(
                {
                    "candidate_index": index,
                    "poisson_ratio": poisson_ratio,
                    "young_modulus_pa": protocol.young_modulus_pa,
                    "weight": protocol.weights[index],
                    "physical_archive": f"{group.group_id}/{member_path.name}",
                    "physical_archive_sha256": file_sha256(member_path),
                    "minimum_deformation_determinant": float(
                        np.min(replay.deformation_determinants)
                    ),
                    "maximum_deformation_determinant": float(
                        np.max(replay.deformation_determinants)
                    ),
                    "maximum_node_displacement_m": float(
                        np.max(
                            np.linalg.norm(replay.positions_m - points[None], axis=2)
                        )
                    ),
                    "status": "success",
                }
            )
        stack = np.stack(
            [
                np.asarray(value["prediction_m"], dtype=np.float64)
                for value in member_arrays
            ]
        )
        mean = np.tensordot(np.asarray(protocol.weights), stack, axes=(0, 0))
        mean_physical = _physical_arrays(
            cast(FloatArray, mean),
            frame_zero_m=points,
            action_support=np.asarray(arrays["action_support"]),
        )
        mean_path = write_deterministic_npz(
            group_dir / "ensemble-mean.npz", mean_physical
        )
        final_spread = float(np.sqrt(np.mean(np.var(stack[:, -1], axis=0, ddof=0))))
        records.append(
            {
                "group_id": group.group_id,
                "source_inputs_sha256": group.source_inputs_sha256,
                "incumbent_sha256": group.incumbent_sha256,
                "matphys_sha256": group.matphys_sha256,
                "base_cell_count": len(cells),
                "contact_patch_sizes": [len(patch) for patch in patches],
                "maximum_contact_projection_error_m": maximum_contact_error,
                "members": members,
                "ensemble_mean_archive": f"{group.group_id}/{mean_path.name}",
                "ensemble_mean_sha256": file_sha256(mean_path),
                "final_ensemble_spread_m": final_spread,
            }
        )
    identity: dict[str, Any] = {
        "schema": GRID_SCHEMA,
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "qualification_artifact_id": protocol.qualification_artifact_id,
        "implementation": provenance,
        "groups": records,
        "successful_candidate_count_per_group": len(protocol.poisson_ratios),
        "information_boundary": {
            "source_inputs_read": True,
            "incumbent_and_matphys_predictions_read": True,
            "prefix_outcomes_read": False,
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
    }
    grid = {**identity, "grid_id": content_id(identity)}
    write_atomic_json(grid, output / GRID_FILENAME, overwrite=False)
    return grid


def _load_outcomes(
    path: Path,
    *,
    digest: str,
    frame_count: int,
) -> dict[str, npt.NDArray[Any]]:
    source = _ordinary_file(path, name="source outcome")
    _require(file_sha256(source) == digest, "outcome SHA-256 changed")
    with np.load(source, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    _require(
        set(arrays) == {"object_points_m", "valid_mask", "frame_indices"},
        "outcome roster changed",
    )
    points = arrays["object_points_m"]
    valid = arrays["valid_mask"]
    indices = arrays["frame_indices"]
    _require(
        points.ndim == 3
        and points.shape[0] >= 2
        and points.shape[1] >= 1
        and points.shape[2] == 3
        and points.dtype == np.float32,
        "outcome points changed",
    )
    _require(
        valid.shape == points.shape[:2] and valid.dtype == np.bool_,
        "outcome validity changed",
    )
    _require(
        indices.shape == (len(points),) and indices.dtype == np.int32,
        "outcome frame indices changed",
    )
    _require(
        np.all(np.diff(indices) > 0)
        and int(indices[0]) >= 0
        and int(indices[-1]) < frame_count,
        "outcome frame order changed",
    )
    _require(
        np.any(valid) and np.all(np.isfinite(points[valid])), "outcome support changed"
    )
    return arrays


def marginal_energy_score_v1(
    samples_m: npt.ArrayLike,
    outcomes_m: npt.ArrayLike,
    valid_mask: npt.ArrayLike,
) -> float:
    samples = np.asarray(samples_m, dtype=np.float64)
    outcomes = np.asarray(outcomes_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    _require(
        samples.ndim == 4 and samples.shape[1:] == outcomes.shape,
        "energy samples changed",
    )
    _require(valid.shape == outcomes.shape[:2], "energy validity changed")
    frame_scores: list[float] = []
    for frame in range(len(outcomes)):
        if not np.any(valid[frame]):
            continue
        current = samples[:, frame, valid[frame]]
        truth = outcomes[frame, valid[frame]]
        first = np.mean(np.linalg.norm(current - truth[None], axis=2), axis=0)
        pairwise = np.linalg.norm(current[:, None] - current[None, :], axis=3)
        second = 0.5 * np.mean(pairwise, axis=(0, 1))
        frame_scores.append(float(np.mean(first - second)))
    if not frame_scores:
        raise ValueError("energy score has no supported frame")
    return float(np.mean(frame_scores))


def _metric_block(
    point_prediction: FloatArray,
    samples: FloatArray,
    outcome: FloatArray,
    valid: BoolArray,
) -> dict[str, float]:
    return {
        "identity_coordinate_rmse_m": _coordinate_rmse(
            point_prediction, outcome, valid
        ),
        "symmetric_chamfer_m": _symmetric_chamfer(point_prediction, outcome, valid),
        "marginal_energy_score_m": marginal_energy_score_v1(samples, outcome, valid),
    }


def _load_grid(
    path: Path,
    *,
    protocol: JaxFemSourceValueProtocolV1,
    enforce_physical_gate: bool = True,
) -> Mapping[str, Any]:
    source = _ordinary_file(path, name="JaxFem source-value grid")
    value = load_strict_json_object(source, label="JaxFem source-value grid")
    _exact_fields(
        value,
        {
            "schema",
            "schema_version",
            "protocol_sha256",
            "qualification_artifact_id",
            "implementation",
            "groups",
            "successful_candidate_count_per_group",
            "information_boundary",
            "grid_id",
        },
        name="grid",
    )
    _require(
        value["schema"] == GRID_SCHEMA and value["schema_version"] == 1,
        "grid identity changed",
    )
    _require(value["protocol_sha256"] == protocol.sha256, "grid protocol changed")
    _require(
        value["qualification_artifact_id"] == protocol.qualification_artifact_id,
        "grid qualification changed",
    )
    identity = dict(value)
    grid_id = identity.pop("grid_id", None)
    _require(grid_id == content_id(identity), "grid content ID changed")
    implementation = _mapping(value["implementation"], name="grid implementation")
    _exact_fields(
        implementation,
        {"git_head", "git_worktree_clean", "source_files"},
        name="grid implementation",
    )
    git_head = _string(implementation["git_head"], name="grid Git revision")
    _require(
        len(git_head) == 40
        and all(character in "0123456789abcdef" for character in git_head),
        "grid Git revision changed",
    )
    _require(
        implementation["git_worktree_clean"] is True, "grid worktree was not clean"
    )
    source_files = _mapping(implementation["source_files"], name="grid source files")
    protocol_label = _string(protocol.value["protocol_label"], name="protocol label")
    if protocol_label == "jax-fem-zebra-source-value-v1":
        expected_source_files = {
            "src/bayesian_phystwin/jax_fem_source_qualification_v1.py",
            "src/bayesian_phystwin/jax_fem_source_value_v1.py",
            "scripts/remote/run_jax_fem_source_value_v1.py",
        }
    elif protocol_label == "jax-fem-zebra-source-value-v2":
        expected_source_files = {
            "src/bayesian_phystwin/jax_fem_source_qualification_v1.py",
            "src/bayesian_phystwin/jax_fem_hyperelastic_v2.py",
            ("src/bayesian_phystwin/jax_fem_hyperelastic_source_qualification_v2.py"),
            "src/bayesian_phystwin/jax_fem_source_value_v1.py",
            "src/bayesian_phystwin/jax_fem_hyperelastic_source_value_v2.py",
            "scripts/remote/run_jax_fem_hyperelastic_source_value_v2.py",
        }
    else:
        raise ValueError("grid protocol label is not registered")
    _require(
        set(source_files) == expected_source_files,
        "grid source-file roster changed",
    )
    for relative, digest in source_files.items():
        _relative(relative, name="grid source path")
        _digest(digest, name="grid source SHA-256")
    boundary = _mapping(value["information_boundary"], name="grid boundary")
    _require(
        boundary
        == {
            "source_inputs_read": True,
            "incumbent_and_matphys_predictions_read": True,
            "prefix_outcomes_read": False,
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
        "grid crossed outcome boundary",
    )
    groups = value["groups"]
    _require(
        isinstance(groups, list) and len(groups) == len(protocol.groups),
        "grid group denominator changed",
    )
    _require(
        value["successful_candidate_count_per_group"]
        == protocol.gates["required_successful_candidate_count"],
        "grid candidate denominator changed",
    )
    grid_root = source.parent
    group_fields = {
        "group_id",
        "source_inputs_sha256",
        "incumbent_sha256",
        "matphys_sha256",
        "base_cell_count",
        "contact_patch_sizes",
        "maximum_contact_projection_error_m",
        "members",
        "ensemble_mean_archive",
        "ensemble_mean_sha256",
        "final_ensemble_spread_m",
    }
    member_fields = {
        "candidate_index",
        "poisson_ratio",
        "young_modulus_pa",
        "weight",
        "physical_archive",
        "physical_archive_sha256",
        "minimum_deformation_determinant",
        "maximum_deformation_determinant",
        "maximum_node_displacement_m",
        "status",
    }
    for expected_group, raw_record in zip(protocol.groups, groups, strict=True):
        record = _mapping(raw_record, name="grid group")
        _exact_fields(record, group_fields, name="grid group")
        _require(
            record["group_id"] == expected_group.group_id, "grid group order changed"
        )
        _require(
            record["source_inputs_sha256"] == expected_group.source_inputs_sha256
            and record["incumbent_sha256"] == expected_group.incumbent_sha256
            and record["matphys_sha256"] == expected_group.matphys_sha256,
            "grid source binding changed",
        )
        _require(
            record["base_cell_count"] == expected_group.base_cell_count
            and record["contact_patch_sizes"]
            == list(expected_group.contact_patch_sizes),
            "grid topology binding changed",
        )
        contact_error = _finite(
            record["maximum_contact_projection_error_m"],
            name="maximum contact projection error",
        )
        if enforce_physical_gate:
            _require(
                contact_error
                <= float(
                    protocol.gates["maximum_full_horizon_contact_projection_error_m"]
                ),
                "full-horizon contact projection gate failed",
            )
        members = record["members"]
        _require(
            isinstance(members, list) and len(members) == len(protocol.poisson_ratios),
            "grid member denominator changed",
        )
        predictions: list[FloatArray] = []
        reference_arrays: Mapping[str, FloatArray] | None = None
        for index, raw_member in enumerate(members):
            member = _mapping(raw_member, name="grid member")
            _exact_fields(member, member_fields, name="grid member")
            _require(
                member["candidate_index"] == index
                and member["status"] == "success"
                and member["poisson_ratio"] == protocol.poisson_ratios[index]
                and member["young_modulus_pa"] == protocol.young_modulus_pa
                and member["weight"] == protocol.weights[index],
                "grid member differs from frozen ensemble",
            )
            minimum_determinant = _finite(
                member["minimum_deformation_determinant"],
                name="minimum deformation determinant",
            )
            maximum_determinant = _finite(
                member["maximum_deformation_determinant"],
                name="maximum deformation determinant",
            )
            maximum_displacement = _finite(
                member["maximum_node_displacement_m"],
                name="maximum node displacement",
            )
            if enforce_physical_gate:
                _require(
                    minimum_determinant
                    >= float(
                        protocol.gates["minimum_full_horizon_deformation_determinant"]
                    )
                    and maximum_determinant
                    <= float(
                        protocol.gates["maximum_full_horizon_deformation_determinant"]
                    )
                    and maximum_displacement
                    <= float(
                        protocol.gates["maximum_full_horizon_node_displacement_m"]
                    ),
                    "full-horizon physical sanity gate failed",
                )
            archive_relative = _relative(
                member["physical_archive"],
                name="member archive",
            )
            archive = _ordinary_file(
                grid_root / archive_relative.as_posix(),
                name="member archive",
            )
            _require(
                file_sha256(archive)
                == _digest(member["physical_archive_sha256"], name="member SHA-256"),
                "member archive changed",
            )
            arrays = load_physical_rollout_archive(
                archive,
                expected_frame_count=expected_group.frame_count,
            )
            if reference_arrays is None:
                reference_arrays = arrays
            else:
                for field in (
                    "frame_zero_points_m",
                    "persistence_m",
                    "zero_action_readout_m",
                    "action_support",
                ):
                    _require(
                        np.array_equal(arrays[field], reference_arrays[field]),
                        "ensemble member identity changed",
                    )
            predictions.append(cast(FloatArray, arrays["prediction_m"]))
        mean_relative = _relative(
            record["ensemble_mean_archive"], name="ensemble mean archive"
        )
        mean_archive = _ordinary_file(
            grid_root / mean_relative.as_posix(),
            name="ensemble mean archive",
        )
        _require(
            file_sha256(mean_archive)
            == _digest(record["ensemble_mean_sha256"], name="ensemble mean SHA-256"),
            "ensemble mean archive changed",
        )
        mean_arrays = load_physical_rollout_archive(
            mean_archive,
            expected_frame_count=expected_group.frame_count,
        )
        stack = np.stack(predictions)
        expected_mean = np.ascontiguousarray(
            np.tensordot(
                np.asarray(protocol.weights), stack.astype(np.float64), axes=(0, 0)
            ),
            dtype=np.float32,
        )
        _require(
            np.array_equal(mean_arrays["prediction_m"], expected_mean),
            "ensemble mean was not re-derived from frozen members",
        )
        spread = _finite(
            record["final_ensemble_spread_m"],
            name="final ensemble spread",
        )
        expected_spread = float(
            np.sqrt(np.mean(np.var(stack[:, -1].astype(np.float64), axis=0, ddof=0)))
        )
        _require(
            np.isclose(spread, expected_spread, rtol=1e-12, atol=0.0),
            "final ensemble spread changed",
        )
    return cast(Mapping[str, Any], value)


def _grid_physical_checks(
    grid: Mapping[str, Any], *, protocol: JaxFemSourceValueProtocolV1
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    gates = protocol.gates
    details: list[dict[str, Any]] = []
    contact_ok = True
    displacement_ok = True
    determinant_ok = True
    for raw_group in cast(list[Mapping[str, Any]], grid["groups"]):
        contact_error = float(raw_group["maximum_contact_projection_error_m"])
        group_contact_ok = contact_error <= float(
            gates["maximum_full_horizon_contact_projection_error_m"]
        )
        members: list[dict[str, Any]] = []
        for raw_member in cast(list[Mapping[str, Any]], raw_group["members"]):
            minimum_determinant = float(raw_member["minimum_deformation_determinant"])
            maximum_determinant = float(raw_member["maximum_deformation_determinant"])
            maximum_displacement = float(raw_member["maximum_node_displacement_m"])
            member_determinant_ok = minimum_determinant >= float(
                gates["minimum_full_horizon_deformation_determinant"]
            ) and maximum_determinant <= float(
                gates["maximum_full_horizon_deformation_determinant"]
            )
            member_displacement_ok = maximum_displacement <= float(
                gates["maximum_full_horizon_node_displacement_m"]
            )
            determinant_ok = determinant_ok and member_determinant_ok
            displacement_ok = displacement_ok and member_displacement_ok
            members.append(
                {
                    "candidate_index": raw_member["candidate_index"],
                    "poisson_ratio": raw_member["poisson_ratio"],
                    "minimum_deformation_determinant": minimum_determinant,
                    "maximum_deformation_determinant": maximum_determinant,
                    "maximum_node_displacement_m": maximum_displacement,
                    "deformation_determinant_gate_passed": member_determinant_ok,
                    "node_displacement_gate_passed": member_displacement_ok,
                }
            )
        contact_ok = contact_ok and group_contact_ok
        details.append(
            {
                "group_id": raw_group["group_id"],
                "maximum_contact_projection_error_m": contact_error,
                "contact_projection_gate_passed": group_contact_ok,
                "members": members,
            }
        )
    checks = {
        "full_horizon_contact_projection": contact_ok,
        "full_horizon_deformation_determinants": determinant_ok,
        "full_horizon_node_displacement": displacement_ok,
    }
    return checks, details


def finalize_jax_fem_source_value_pre_prefix_v1(
    *,
    protocol_path: str | Path,
    group_roots: Mapping[str, str | Path],
    grid_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Apply the outcome-blind physical gate before any prefix may be opened."""

    protocol = load_jax_fem_source_value_protocol_v1(protocol_path)
    expected = {group.group_id for group in protocol.groups}
    _require(set(group_roots) == expected, "complete source roots are required")
    grid_root = Path(grid_dir).absolute()
    grid_path = grid_root / GRID_FILENAME
    grid = _load_grid(
        grid_path,
        protocol=protocol,
        enforce_physical_gate=False,
    )
    checks, details = _grid_physical_checks(grid, protocol=protocol)
    passed = all(checks.values())
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fallback_records: list[dict[str, Any]] = []
    if not passed:
        for group in protocol.groups:
            source = _ordinary_file(
                Path(group_roots[group.group_id]).absolute()
                / group.incumbent_relative_path.as_posix(),
                name="pre-prefix incumbent fallback",
            )
            _require(
                file_sha256(source) == group.incumbent_sha256,
                "pre-prefix incumbent changed",
            )
            target_dir = output / group.group_id
            target_dir.mkdir()
            target = target_dir / SELECTED_FILENAME
            shutil.copyfile(source, target)
            _require(target.read_bytes() == source.read_bytes(), "fallback changed")
            fallback_records.append(
                {
                    "group_id": group.group_id,
                    "selection": "exact_incumbent_fallback",
                    "selected_sha256": file_sha256(target),
                    "source_sha256": file_sha256(source),
                    "byte_exact_source": True,
                }
            )
    identity: dict[str, Any] = {
        "schema": PRE_PREFIX_SCHEMA,
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "grid_sha256": file_sha256(grid_path),
        "physical_checks": checks,
        "physical_details": details,
        "physical_gate_passed": passed,
        "prefix_scoring_authorized": passed,
        "selected_predictions": fallback_records,
        "status": (
            "prefix-scoring-authorized"
            if passed
            else "pre-prefix-physical-gate-failed-exact-fallback"
        ),
        "information_boundary": {
            "prefix_outcomes_read": False,
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
    }
    result = {**identity, "result_id": content_id(identity)}
    write_atomic_json(result, output / PRE_PREFIX_FILENAME, overwrite=False)
    return result


def _validation_ratios(metrics: Mapping[str, Any]) -> dict[str, float]:
    validation = _mapping(metrics["validation"], name="validation metrics")
    jax_fem = _mapping(validation["jax_fem"], name="JaxFem validation metrics")
    persistence = _mapping(
        validation["persistence"],
        name="persistence validation metrics",
    )
    incumbent = _mapping(
        validation["incumbent"],
        name="incumbent validation metrics",
    )
    return {
        "balanced_point_ratio_vs_persistence": 0.5
        * (
            _ratio(
                float(jax_fem["identity_coordinate_rmse_m"]),
                float(persistence["identity_coordinate_rmse_m"]),
                name="identity versus persistence",
            )
            + _ratio(
                float(jax_fem["symmetric_chamfer_m"]),
                float(persistence["symmetric_chamfer_m"]),
                name="Chamfer versus persistence",
            )
        ),
        "energy_ratio_vs_persistence": _ratio(
            float(jax_fem["marginal_energy_score_m"]),
            float(persistence["marginal_energy_score_m"]),
            name="energy versus persistence",
        ),
        "identity_ratio_vs_incumbent": _ratio(
            float(jax_fem["identity_coordinate_rmse_m"]),
            float(incumbent["identity_coordinate_rmse_m"]),
            name="identity versus incumbent",
        ),
        "chamfer_ratio_vs_incumbent": _ratio(
            float(jax_fem["symmetric_chamfer_m"]),
            float(incumbent["symmetric_chamfer_m"]),
            name="Chamfer versus incumbent",
        ),
    }


def _validation_checks(
    *,
    protocol: JaxFemSourceValueProtocolV1,
    group_metrics: list[dict[str, Any]],
    successful_candidate_count: object,
) -> dict[str, bool]:
    group_ratios = [
        cast(Mapping[str, float], group["validation_ratios"]) for group in group_metrics
    ]
    spreads = [float(group["final_ensemble_spread_m"]) for group in group_metrics]
    gates = protocol.gates
    return {
        "complete_candidate_denominator": successful_candidate_count
        == gates["required_successful_candidate_count"],
        "equal_group_balanced_point_improvement": float(
            np.mean(
                [value["balanced_point_ratio_vs_persistence"] for value in group_ratios]
            )
        )
        <= float(gates["maximum_equal_group_balanced_point_ratio_vs_persistence"]),
        "worst_group_balanced_point_nonregression": max(
            value["balanced_point_ratio_vs_persistence"] for value in group_ratios
        )
        <= float(gates["maximum_worst_group_balanced_point_ratio_vs_persistence"]),
        "equal_group_energy_improvement": float(
            np.mean([value["energy_ratio_vs_persistence"] for value in group_ratios])
        )
        <= float(gates["maximum_equal_group_energy_ratio_vs_persistence"]),
        "equal_group_identity_nonregression_vs_incumbent": float(
            np.mean([value["identity_ratio_vs_incumbent"] for value in group_ratios])
        )
        <= float(gates["maximum_equal_group_identity_ratio_vs_incumbent"]),
        "equal_group_chamfer_nonregression_vs_incumbent": float(
            np.mean([value["chamfer_ratio_vs_incumbent"] for value in group_ratios])
        )
        <= float(gates["maximum_equal_group_chamfer_ratio_vs_incumbent"]),
        "minimum_ensemble_spread": min(spreads)
        >= float(gates["minimum_final_ensemble_spread_m"]),
        "maximum_ensemble_spread": max(spreads)
        <= float(gates["maximum_final_ensemble_spread_m"]),
    }


def _validated_metric_block(value: object, *, name: str) -> dict[str, float]:
    block = _mapping(value, name=name)
    fields = {
        "identity_coordinate_rmse_m",
        "symmetric_chamfer_m",
        "marginal_energy_score_m",
    }
    _exact_fields(block, fields, name=name)
    result: dict[str, float] = {}
    for field in sorted(fields):
        metric = _finite(block[field], name=f"{name}.{field}")
        _require(metric >= 0.0, f"{name}.{field} is negative")
        result[field] = metric
    return result


def _validated_group_metrics(
    value: object,
    *,
    expected_group: JaxFemValueGroupV1,
) -> dict[str, Any]:
    group = _mapping(value, name="prefix group metrics")
    _exact_fields(
        group,
        {
            "group_id",
            "prefix_outcome_sha256",
            "prefix_frame_indices",
            "validation_start_local_index",
            "metrics",
            "validation_ratios",
            "final_ensemble_spread_m",
        },
        name="prefix group metrics",
    )
    _require(group["group_id"] == expected_group.group_id, "prefix group order changed")
    _require(
        group["prefix_outcome_sha256"] == expected_group.prefix_outcomes_sha256,
        "prefix outcome binding changed",
    )
    raw_indices = group["prefix_frame_indices"]
    _require(
        isinstance(raw_indices, list)
        and len(raw_indices) >= 2
        and all(type(index) is int for index in raw_indices),
        "prefix frame indices changed",
    )
    indices = cast(list[int], raw_indices)
    _require(
        all(left < right for left, right in zip(indices, indices[1:], strict=False))
        and indices[0] >= 0
        and indices[-1] < expected_group.frame_count,
        "prefix frame indices changed",
    )
    split = group["validation_start_local_index"]
    _require(type(split) is int and 0 < split < len(indices), "prefix split changed")
    raw_metrics = _mapping(group["metrics"], name="prefix metrics")
    _exact_fields(raw_metrics, {"fit", "validation"}, name="prefix metrics")
    normalized_metrics: dict[str, Any] = {}
    for split_name in ("fit", "validation"):
        raw_split = _mapping(raw_metrics[split_name], name=f"prefix {split_name}")
        _exact_fields(
            raw_split,
            {"jax_fem", "persistence", "incumbent", "matphys"},
            name=f"prefix {split_name}",
        )
        normalized_metrics[split_name] = {
            comparator: _validated_metric_block(
                raw_split[comparator],
                name=f"prefix {split_name}.{comparator}",
            )
            for comparator in ("jax_fem", "persistence", "incumbent", "matphys")
        }
    raw_ratios = _mapping(group["validation_ratios"], name="validation ratios")
    expected_ratios = _validation_ratios(normalized_metrics)
    _exact_fields(raw_ratios, set(expected_ratios), name="validation ratios")
    for name, expected in expected_ratios.items():
        observed = _finite(raw_ratios[name], name=f"validation ratio {name}")
        _require(
            np.isclose(observed, expected, rtol=1e-12, atol=0.0),
            f"validation ratio {name} was not re-derived",
        )
    return {
        "group_id": expected_group.group_id,
        "prefix_outcome_sha256": expected_group.prefix_outcomes_sha256,
        "prefix_frame_indices": indices,
        "validation_start_local_index": split,
        "metrics": normalized_metrics,
        "validation_ratios": expected_ratios,
        "final_ensemble_spread_m": _finite(
            group["final_ensemble_spread_m"],
            name="prefix final ensemble spread",
        ),
    }


def score_jax_fem_source_value_prefix_v1(
    *,
    protocol_path: str | Path,
    group_roots: Mapping[str, str | Path],
    matphys_paths: Mapping[str, str | Path],
    grid_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    protocol = load_jax_fem_source_value_protocol_v1(protocol_path)
    expected = {group.group_id for group in protocol.groups}
    _require(
        set(group_roots) == expected and set(matphys_paths) == expected,
        "complete source roots are required",
    )
    grid_root = Path(grid_dir).absolute()
    grid_path = grid_root / GRID_FILENAME
    grid = _load_grid(grid_path, protocol=protocol)
    grid_records = {
        record["group_id"]: record
        for record in cast(list[Mapping[str, Any]], grid["groups"])
    }
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    group_metrics: list[dict[str, Any]] = []
    selected_sources: dict[str, Path] = {}
    for group in protocol.groups:
        root = Path(group_roots[group.group_id]).absolute()
        outcome_arrays = _load_outcomes(
            root / group.prefix_outcomes_relative_path.as_posix(),
            digest=group.prefix_outcomes_sha256,
            frame_count=group.frame_count,
        )
        outcome = np.asarray(outcome_arrays["object_points_m"])
        valid = np.asarray(outcome_arrays["valid_mask"])
        frame_indices = np.asarray(outcome_arrays["frame_indices"], dtype=np.int64)
        observed_count = outcome.shape[1]
        _require(
            observed_count <= group.material_particle_count, "observed count changed"
        )
        record = grid_records[group.group_id]
        members = cast(list[Mapping[str, Any]], record["members"])
        member_predictions: list[FloatArray] = []
        for member in members:
            path = grid_root / str(member["physical_archive"])
            _require(
                file_sha256(path) == member["physical_archive_sha256"],
                "member archive changed",
            )
            physical = load_physical_rollout_archive(
                path, expected_frame_count=group.frame_count
            )
            member_predictions.append(
                cast(
                    FloatArray,
                    np.asarray(physical["prediction_m"])[
                        frame_indices, :observed_count
                    ],
                )
            )
        samples = cast(FloatArray, np.stack(member_predictions))
        mean_path = grid_root / str(record["ensemble_mean_archive"])
        _require(
            file_sha256(mean_path) == record["ensemble_mean_sha256"],
            "mean archive changed",
        )
        mean_physical = load_physical_rollout_archive(
            mean_path, expected_frame_count=group.frame_count
        )
        mean_prediction = cast(
            FloatArray,
            np.asarray(mean_physical["prediction_m"])[frame_indices, :observed_count],
        )
        incumbent_path = root / group.incumbent_relative_path.as_posix()
        matphys_path = Path(matphys_paths[group.group_id]).absolute()
        _require(
            file_sha256(incumbent_path) == group.incumbent_sha256, "incumbent changed"
        )
        _require(file_sha256(matphys_path) == group.matphys_sha256, "MatPhys changed")
        incumbent = load_physical_rollout_archive(
            incumbent_path, expected_frame_count=group.frame_count
        )
        matphys = load_physical_rollout_archive(
            matphys_path, expected_frame_count=group.frame_count
        )
        persistence = np.asarray(incumbent["persistence_m"])[
            frame_indices, :observed_count
        ]
        incumbent_prediction = np.asarray(incumbent["prediction_m"])[
            frame_indices, :observed_count
        ]
        matphys_prediction = np.asarray(matphys["prediction_m"])[
            frame_indices, :observed_count
        ]
        split = max(
            1,
            min(
                len(outcome) - 1,
                int(np.floor(len(outcome) * float(protocol.candidate["fit_fraction"]))),
            ),
        )
        blocks: dict[str, Any] = {}
        for split_name, selected in {
            "fit": slice(0, split),
            "validation": slice(split, len(outcome)),
        }.items():
            current_outcome = cast(FloatArray, outcome[selected])
            current_valid = cast(BoolArray, valid[selected])
            blocks[split_name] = {
                "jax_fem": _metric_block(
                    mean_prediction[selected],
                    samples[:, selected],
                    current_outcome,
                    current_valid,
                ),
                "persistence": _metric_block(
                    cast(FloatArray, persistence[selected]),
                    cast(FloatArray, persistence[selected][None]),
                    current_outcome,
                    current_valid,
                ),
                "incumbent": _metric_block(
                    cast(FloatArray, incumbent_prediction[selected]),
                    cast(FloatArray, incumbent_prediction[selected][None]),
                    current_outcome,
                    current_valid,
                ),
                "matphys": _metric_block(
                    cast(FloatArray, matphys_prediction[selected]),
                    cast(FloatArray, matphys_prediction[selected][None]),
                    current_outcome,
                    current_valid,
                ),
            }
        validation = blocks["validation"]
        _mapping(validation, name="validation metrics")
        current_ratios = _validation_ratios(blocks)
        group_metrics.append(
            {
                "group_id": group.group_id,
                "prefix_outcome_sha256": group.prefix_outcomes_sha256,
                "prefix_frame_indices": frame_indices.tolist(),
                "validation_start_local_index": split,
                "metrics": blocks,
                "validation_ratios": current_ratios,
                "final_ensemble_spread_m": record["final_ensemble_spread_m"],
            }
        )
        selected_sources[group.group_id] = mean_path

    checks = _validation_checks(
        protocol=protocol,
        group_metrics=group_metrics,
        successful_candidate_count=grid["successful_candidate_count_per_group"],
    )
    passed = all(checks.values())
    selected_records: list[dict[str, Any]] = []
    for group in protocol.groups:
        root = Path(group_roots[group.group_id]).absolute()
        incumbent_path = root / group.incumbent_relative_path.as_posix()
        source = selected_sources[group.group_id] if passed else incumbent_path
        target_dir = output / group.group_id
        target_dir.mkdir()
        target = target_dir / SELECTED_FILENAME
        shutil.copyfile(source, target)
        _require(
            target.read_bytes() == source.read_bytes(),
            "selected archive is not byte exact",
        )
        selected_records.append(
            {
                "group_id": group.group_id,
                "selection": "jax_fem_equal_ensemble_mean"
                if passed
                else "exact_incumbent_fallback",
                "selected_sha256": file_sha256(target),
                "source_sha256": file_sha256(source),
                "byte_exact_source": True,
            }
        )
    identity: dict[str, Any] = {
        "schema": PREFIX_SCHEMA,
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "grid_sha256": file_sha256(grid_path),
        "group_metrics": group_metrics,
        "validation_checks": checks,
        "validation_gate_passed": passed,
        "future_scoring_authorized": passed,
        "selected_predictions": selected_records,
        "information_boundary": {
            "prefix_outcomes_read": True,
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
    }
    result = {**identity, "result_id": content_id(identity)}
    write_atomic_json(result, output / PREFIX_FILENAME, overwrite=False)
    return result


def _validate_prefix_result(
    prefix: Mapping[str, Any],
    *,
    protocol: JaxFemSourceValueProtocolV1,
    prefix_root: Path,
    grid_root: Path,
    group_roots: Mapping[str, str | Path],
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    _exact_fields(
        prefix,
        {
            "schema",
            "schema_version",
            "protocol_sha256",
            "grid_sha256",
            "group_metrics",
            "validation_checks",
            "validation_gate_passed",
            "future_scoring_authorized",
            "selected_predictions",
            "information_boundary",
            "result_id",
        },
        name="prefix result",
    )
    _require(
        prefix["schema"] == PREFIX_SCHEMA and prefix["schema_version"] == 1,
        "prefix result identity changed",
    )
    _require(prefix["protocol_sha256"] == protocol.sha256, "prefix protocol changed")
    identity = dict(prefix)
    result_id = identity.pop("result_id")
    _require(result_id == content_id(identity), "prefix result content ID changed")
    boundary = _mapping(prefix["information_boundary"], name="prefix boundary")
    _require(
        boundary
        == {
            "prefix_outcomes_read": True,
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        },
        "prefix result crossed its information boundary",
    )
    grid_path = grid_root / GRID_FILENAME
    _require(file_sha256(grid_path) == prefix["grid_sha256"], "prefix grid changed")
    grid = _load_grid(grid_path, protocol=protocol)

    raw_group_metrics = prefix["group_metrics"]
    _require(
        isinstance(raw_group_metrics, list)
        and len(raw_group_metrics) == len(protocol.groups),
        "prefix group denominator changed",
    )
    normalized = [
        _validated_group_metrics(raw, expected_group=group)
        for raw, group in zip(raw_group_metrics, protocol.groups, strict=True)
    ]
    derived_checks = _validation_checks(
        protocol=protocol,
        group_metrics=normalized,
        successful_candidate_count=grid["successful_candidate_count_per_group"],
    )
    raw_checks = _mapping(prefix["validation_checks"], name="validation checks")
    _exact_fields(raw_checks, set(derived_checks), name="validation checks")
    _require(
        dict(raw_checks) == derived_checks, "validation checks were not re-derived"
    )
    gate_passed = all(derived_checks.values())
    _require(
        prefix["validation_gate_passed"] is gate_passed
        and prefix["future_scoring_authorized"] is gate_passed,
        "future authorization differs from the frozen gate",
    )

    expected_groups = {group.group_id for group in protocol.groups}
    _require(set(group_roots) == expected_groups, "complete source roots are required")
    raw_selected = prefix["selected_predictions"]
    _require(
        isinstance(raw_selected, list) and len(raw_selected) == len(protocol.groups),
        "selected prediction denominator changed",
    )
    grid_records = {
        record["group_id"]: record
        for record in cast(list[Mapping[str, Any]], grid["groups"])
    }
    for raw, group in zip(raw_selected, protocol.groups, strict=True):
        selected = _mapping(raw, name="selected prediction")
        _exact_fields(
            selected,
            {
                "group_id",
                "selection",
                "selected_sha256",
                "source_sha256",
                "byte_exact_source",
            },
            name="selected prediction",
        )
        _require(selected["group_id"] == group.group_id, "selected group order changed")
        expected_selection = (
            "jax_fem_equal_ensemble_mean" if gate_passed else "exact_incumbent_fallback"
        )
        _require(
            selected["selection"] == expected_selection, "selected mechanism changed"
        )
        selected_path = _ordinary_file(
            prefix_root / group.group_id / SELECTED_FILENAME,
            name="selected physical prediction",
        )
        if gate_passed:
            record = grid_records[group.group_id]
            relative = _relative(
                record["ensemble_mean_archive"],
                name="selected ensemble mean archive",
            )
            expected_source = _ordinary_file(
                grid_root / relative.as_posix(),
                name="selected ensemble mean archive",
            )
        else:
            expected_source = _ordinary_file(
                Path(group_roots[group.group_id]).absolute()
                / group.incumbent_relative_path.as_posix(),
                name="selected incumbent fallback",
            )
        expected_sha = file_sha256(expected_source)
        _require(
            selected["byte_exact_source"] is True
            and selected["source_sha256"] == expected_sha
            and selected["selected_sha256"] == file_sha256(selected_path)
            and selected_path.read_bytes() == expected_source.read_bytes(),
            "selected physical prediction is not the exact gated source",
        )
    return grid, normalized


def score_jax_fem_source_value_future_v1(
    *,
    protocol_path: str | Path,
    group_roots: Mapping[str, str | Path],
    matphys_paths: Mapping[str, str | Path],
    prefix_dir: str | Path,
    grid_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    protocol = load_jax_fem_source_value_protocol_v1(protocol_path)
    expected = {group.group_id for group in protocol.groups}
    _require(
        set(group_roots) == expected and set(matphys_paths) == expected,
        "complete source roots are required",
    )
    prefix_root = Path(prefix_dir).absolute()
    grid_root = Path(grid_dir).absolute()
    prefix_path = _ordinary_file(prefix_root / PREFIX_FILENAME, name="prefix result")
    prefix = load_strict_json_object(
        prefix_path, label="JaxFem source-value prefix result"
    )
    grid, normalized_prefix = _validate_prefix_result(
        prefix,
        protocol=protocol,
        prefix_root=prefix_root,
        grid_root=grid_root,
        group_roots=group_roots,
    )
    if prefix["future_scoring_authorized"] is not True:
        result_identity: dict[str, Any] = {
            "schema": FUTURE_SCHEMA,
            "schema_version": 1,
            "protocol_sha256": protocol.sha256,
            "prefix_result_sha256": file_sha256(prefix_path),
            "status": "future-not-opened-validation-gate-failed",
            "future_outcomes_read": False,
            "target_or_held_out_artifact_read": False,
        }
        result = {**result_identity, "result_id": content_id(result_identity)}
        write_atomic_json(result, output_path, overwrite=False)
        return result

    grid_records = {
        record["group_id"]: record
        for record in cast(list[Mapping[str, Any]], grid["groups"])
    }
    prefix_metrics = {group["group_id"]: group for group in normalized_prefix}
    future_groups: list[dict[str, Any]] = []
    for group in protocol.groups:
        root = Path(group_roots[group.group_id]).absolute()
        outcome_arrays = _load_outcomes(
            root / group.future_outcomes_relative_path.as_posix(),
            digest=group.future_outcomes_sha256,
            frame_count=group.frame_count,
        )
        outcome = cast(FloatArray, np.asarray(outcome_arrays["object_points_m"]))
        valid = cast(BoolArray, np.asarray(outcome_arrays["valid_mask"]))
        frame_indices = np.asarray(outcome_arrays["frame_indices"], dtype=np.int64)
        _require(
            int(frame_indices[0])
            > int(prefix_metrics[group.group_id]["prefix_frame_indices"][-1]),
            "future outcomes overlap the prefix",
        )
        observed_count = outcome.shape[1]
        _require(
            observed_count <= group.material_particle_count, "observed count changed"
        )
        record = grid_records[group.group_id]
        members = cast(list[Mapping[str, Any]], record["members"])
        member_predictions: list[FloatArray] = []
        for member in members:
            relative = _relative(
                member["physical_archive"], name="future member archive"
            )
            archive = _ordinary_file(
                grid_root / relative.as_posix(),
                name="future member archive",
            )
            physical = load_physical_rollout_archive(
                archive,
                expected_frame_count=group.frame_count,
            )
            member_predictions.append(
                cast(
                    FloatArray,
                    np.asarray(physical["prediction_m"])[
                        frame_indices,
                        :observed_count,
                    ],
                )
            )
        samples = cast(FloatArray, np.stack(member_predictions))
        selected_path = _ordinary_file(
            prefix_root / group.group_id / SELECTED_FILENAME,
            name="future selected physical prediction",
        )
        incumbent_path = _ordinary_file(
            root / group.incumbent_relative_path.as_posix(),
            name="future incumbent",
        )
        matphys_path = _ordinary_file(
            matphys_paths[group.group_id],
            name="future MatPhys comparator",
        )
        _require(
            file_sha256(incumbent_path) == group.incumbent_sha256, "incumbent changed"
        )
        _require(file_sha256(matphys_path) == group.matphys_sha256, "MatPhys changed")
        physical_arrays = {
            "selected": load_physical_rollout_archive(
                selected_path,
                expected_frame_count=group.frame_count,
            ),
            "incumbent": load_physical_rollout_archive(
                incumbent_path,
                expected_frame_count=group.frame_count,
            ),
            "matphys": load_physical_rollout_archive(
                matphys_path,
                expected_frame_count=group.frame_count,
            ),
        }
        persistence_prediction = cast(
            FloatArray,
            np.asarray(physical_arrays["incumbent"]["persistence_m"])[
                frame_indices,
                :observed_count,
            ],
        )
        metrics: dict[str, Any] = {
            "selected": _metric_block(
                cast(
                    FloatArray,
                    np.asarray(physical_arrays["selected"]["prediction_m"])[
                        frame_indices,
                        :observed_count,
                    ],
                ),
                samples,
                outcome,
                valid,
            ),
            "persistence": _metric_block(
                persistence_prediction,
                cast(FloatArray, persistence_prediction[None]),
                outcome,
                valid,
            ),
        }
        for name in ("incumbent", "matphys"):
            prediction = cast(
                FloatArray,
                np.asarray(physical_arrays[name]["prediction_m"])[
                    frame_indices,
                    :observed_count,
                ],
            )
            metrics[name] = _metric_block(
                prediction,
                cast(FloatArray, prediction[None]),
                outcome,
                valid,
            )
        selected_metrics = _mapping(metrics["selected"], name="selected future metrics")
        persistence_metrics = _mapping(
            metrics["persistence"],
            name="persistence future metrics",
        )
        incumbent_metrics = _mapping(
            metrics["incumbent"],
            name="incumbent future metrics",
        )
        future_groups.append(
            {
                "group_id": group.group_id,
                "future_outcome_sha256": group.future_outcomes_sha256,
                "future_frame_indices": frame_indices.tolist(),
                "metrics": metrics,
                "ratios": {
                    "balanced_point_ratio_vs_persistence": 0.5
                    * (
                        _ratio(
                            float(selected_metrics["identity_coordinate_rmse_m"]),
                            float(persistence_metrics["identity_coordinate_rmse_m"]),
                            name="future identity versus persistence",
                        )
                        + _ratio(
                            float(selected_metrics["symmetric_chamfer_m"]),
                            float(persistence_metrics["symmetric_chamfer_m"]),
                            name="future Chamfer versus persistence",
                        )
                    ),
                    "energy_ratio_vs_persistence": _ratio(
                        float(selected_metrics["marginal_energy_score_m"]),
                        float(persistence_metrics["marginal_energy_score_m"]),
                        name="future energy versus persistence",
                    ),
                    "identity_ratio_vs_incumbent": _ratio(
                        float(selected_metrics["identity_coordinate_rmse_m"]),
                        float(incumbent_metrics["identity_coordinate_rmse_m"]),
                        name="future identity versus incumbent",
                    ),
                    "chamfer_ratio_vs_incumbent": _ratio(
                        float(selected_metrics["symmetric_chamfer_m"]),
                        float(incumbent_metrics["symmetric_chamfer_m"]),
                        name="future Chamfer versus incumbent",
                    ),
                },
            }
        )
    ratio_names = (
        "balanced_point_ratio_vs_persistence",
        "energy_ratio_vs_persistence",
        "identity_ratio_vs_incumbent",
        "chamfer_ratio_vs_incumbent",
    )
    equal_group_ratios = {
        name: float(
            np.mean(
                [
                    cast(Mapping[str, float], group["ratios"])[name]
                    for group in future_groups
                ]
            )
        )
        for name in ratio_names
    }
    result_identity = {
        "schema": FUTURE_SCHEMA,
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "prefix_result_sha256": file_sha256(prefix_path),
        "status": "source-future-scored-after-passing-gate",
        "future_outcomes_read": True,
        "target_or_held_out_artifact_read": False,
        "groups": future_groups,
        "equal_group_ratios": equal_group_ratios,
    }
    result = {**result_identity, "result_id": content_id(result_identity)}
    write_atomic_json(result, output_path, overwrite=False)
    return result


__all__ = [
    "FUTURE_FILENAME",
    "GRID_FILENAME",
    "PRE_PREFIX_FILENAME",
    "PREFIX_FILENAME",
    "JaxFemSourceValueProtocolV1",
    "finalize_jax_fem_source_value_pre_prefix_v1",
    "generate_jax_fem_source_value_predictions_v1",
    "load_jax_fem_source_value_protocol_v1",
    "marginal_energy_score_v1",
    "score_jax_fem_source_value_future_v1",
    "score_jax_fem_source_value_prefix_v1",
]
