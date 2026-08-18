"""Frozen source-physics qualification for the optional Genesis MPM backend."""

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
from .material_backend_qualification_v1 import (
    MaterialBackendQualificationV1,
    save_material_backend_qualification_v1,
)
from .material_backend_v1 import BackendTransportV1, resolve_material_backend_profile
from .physical_rollout_v1 import load_physical_rollout_archive, write_deterministic_npz

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

PROTOCOL_SCHEMA: Final = "bayesian-phystwin.genesis-mpm-source-physics-protocol"
RESULT_SCHEMA: Final = "bayesian-phystwin.genesis-mpm-source-physics-result"
RESULT_FILENAME: Final = "genesis-mpm-source-physics-result.json"
QUALIFICATION_FILENAME: Final = "material-backend-qualification.json"
GROUP_ARCHIVE_FILENAME: Final = "genesis-source-physics-trajectories.npz"
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
    }
)
_GROUP_FIELDS: Final = frozenset(
    {
        "group_id",
        "source_inputs_relative_path",
        "source_inputs_sha256",
        "incumbent_relative_path",
        "incumbent_sha256",
        "frame_count",
        "material_particle_count",
        "controller_point_count",
        "attached_particle_count",
    }
)
_SIMULATION_FIELDS: Final = frozenset(
    {
        "backend",
        "precision",
        "seed",
        "fps",
        "qualification_frame_count",
        "base_substeps",
        "refined_substeps",
        "grid_density",
        "particle_size_m",
        "domain_padding_m",
        "gravity_m_s2",
        "constitutive_model",
        "base_young_modulus_pa",
        "soft_young_modulus_pa",
        "stiff_young_modulus_pa",
        "poisson_ratio",
        "density_kg_m3",
        "rigid_translation_m",
        "nowhere_activation_policy",
    }
)
_GATE_FIELDS: Final = frozenset(
    {
        "maximum_zero_action_drift_m",
        "maximum_rigid_translation_equivariance_error_m",
        "maximum_time_step_refinement_relative_error",
        "maximum_source_query_parity_rmse_m",
        "minimum_action_response_m",
        "minimum_parameter_sensitivity_m",
        "maximum_parameter_sensitivity_m",
        "maximum_particle_step_m",
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


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
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
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _git_revision(value: object, *, name: str) -> str:
    text = _canonical_string(value, name=name)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a full lowercase Git revision")
    return text


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
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
    return tuple(_finite(item, name=f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _canonical_relative_path(value: object, *, name: str) -> PurePosixPath:
    text = _canonical_string(value, name=name)
    path = PurePosixPath(text)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require("\\" not in text, f"{name} must use POSIX separators")
    _require(all(part not in {"", ".", ".."} for part in path.parts), f"{name} is not canonical")
    _require(path.as_posix() == text, f"{name} is not canonical")
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GenesisSourceGroupV1:
    group_id: str
    source_inputs_relative_path: PurePosixPath
    source_inputs_sha256: str
    incumbent_relative_path: PurePosixPath
    incumbent_sha256: str
    frame_count: int
    material_particle_count: int
    controller_point_count: int
    attached_particle_count: int


@dataclass(frozen=True, slots=True)
class GenesisSourcePhysicsProtocolV1:
    value: Mapping[str, Any]
    protocol_sha256: str
    canonical_profile_id: str
    producer_profile_id: str
    transport: BackendTransportV1
    runtime_id: str
    source_groups: tuple[GenesisSourceGroupV1, ...]

    @property
    def simulation(self) -> Mapping[str, Any]:
        return _mapping(self.value["simulation"], name="simulation")

    @property
    def gates(self) -> Mapping[str, Any]:
        return _mapping(self.value["gates"], name="gates")


def load_genesis_source_physics_protocol_v1(
    path: str | Path,
) -> GenesisSourcePhysicsProtocolV1:
    source = Path(path)
    value = load_strict_json_object(source, label="Genesis source-physics protocol")
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
    _require(resolved.profile_id == canonical_profile_id, "backend profile family changed")
    _require(backend["transport"] == resolved.transport, "backend transport changed")
    _canonical_string(backend["engine_repository"], name="engine_repository")
    _git_revision(backend["engine_revision"], name="engine_revision")
    _canonical_string(backend["engine_version"], name="engine_version")
    _sha256(backend["native_smoke_artifact_sha256"], name="native_smoke_artifact_sha256")
    _sha256(backend["native_smoke_id"], name="native_smoke_id")
    runtime_id = _sha256(backend["runtime_id"], name="runtime_id")

    raw_groups = value["source_groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) < 2:
        raise ValueError("at least two source groups are required")
    groups: list[GenesisSourceGroupV1] = []
    for index, raw in enumerate(raw_groups):
        group = _mapping(raw, name=f"source_groups[{index}]")
        _exact_fields(group, _GROUP_FIELDS, f"source_groups[{index}]")
        groups.append(
            GenesisSourceGroupV1(
                group_id=_canonical_string(group["group_id"], name="group_id"),
                source_inputs_relative_path=_canonical_relative_path(
                    group["source_inputs_relative_path"], name="source_inputs_relative_path"
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
                material_particle_count=_positive_int(
                    group["material_particle_count"], name="material_particle_count"
                ),
                controller_point_count=_positive_int(
                    group["controller_point_count"], name="controller_point_count"
                ),
                attached_particle_count=_positive_int(
                    group["attached_particle_count"], name="attached_particle_count"
                ),
            )
        )
    _require(len({group.group_id for group in groups}) == len(groups), "source group IDs must be unique")

    simulation = _mapping(value["simulation"], name="simulation")
    _exact_fields(simulation, _SIMULATION_FIELDS, "simulation")
    _require(simulation["backend"] == "cpu", "qualification backend changed")
    _require(simulation["precision"] == "64", "qualification precision changed")
    _require(type(simulation["seed"]) is int and simulation["seed"] >= 0, "seed changed")
    for name in (
        "fps",
        "particle_size_m",
        "domain_padding_m",
        "base_young_modulus_pa",
        "soft_young_modulus_pa",
        "stiff_young_modulus_pa",
        "density_kg_m3",
    ):
        _finite(simulation[name], name=name, positive=True)
    for name in ("qualification_frame_count", "base_substeps", "refined_substeps", "grid_density"):
        _positive_int(simulation[name], name=name)
    _require(
        int(simulation["refined_substeps"]) > int(simulation["base_substeps"]),
        "refined_substeps must exceed base_substeps",
    )
    _vector3(simulation["gravity_m_s2"], name="gravity_m_s2")
    _vector3(simulation["rigid_translation_m"], name="rigid_translation_m")
    _require(simulation["constitutive_model"] == "neohooken", "constitutive model changed")
    _require(
        simulation["nowhere_activation_policy"]
        == "set-particles-active-then-mark-forward-active-v1",
        "Nowhere activation policy changed",
    )
    poisson = _finite(simulation["poisson_ratio"], name="poisson_ratio")
    _require(-1.0 < poisson < 0.5, "poisson_ratio is invalid")

    gates = _mapping(value["gates"], name="gates")
    _exact_fields(gates, _GATE_FIELDS, "gates")
    for name in _GATE_FIELDS:
        _finite(gates[name], name=name, positive=True)
    _require(
        float(gates["minimum_parameter_sensitivity_m"])
        < float(gates["maximum_parameter_sensitivity_m"]),
        "parameter sensitivity interval is empty",
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
    return GenesisSourcePhysicsProtocolV1(
        value=value,
        protocol_sha256=file_sha256(source),
        canonical_profile_id=canonical_profile_id,
        producer_profile_id=producer_profile_id,
        transport=resolved.transport,
        runtime_id=runtime_id,
        source_groups=tuple(groups),
    )


def load_genesis_source_inputs_v1(
    path: str | Path,
    *,
    group: GenesisSourceGroupV1,
) -> dict[str, npt.NDArray[Any]]:
    source = Path(path)
    _require(source.is_file() and not source.is_symlink(), "source inputs must be an ordinary file")
    _require(file_sha256(source) == group.source_inputs_sha256, "source input SHA-256 changed")
    with np.load(source, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    _require(frozenset(arrays) == _SOURCE_INPUT_ARRAYS, "source input array roster changed")
    points = arrays["frame_zero_points_m"]
    controller = arrays["controller_points_m"]
    indices = arrays["attachment_indices"]
    weights = arrays["attachment_weights"]
    support = arrays["action_support"]
    _require(
        points.shape == (group.material_particle_count, 3)
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
        indices.shape == (group.attached_particle_count,)
        and indices.dtype == np.int32
        and len(np.unique(indices)) == len(indices)
        and np.all((indices >= 0) & (indices < group.material_particle_count)),
        "attachment indices changed",
    )
    _require(
        weights.shape == (group.attached_particle_count, group.controller_point_count)
        and weights.dtype == np.float32
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1.0e-6, rtol=0.0),
        "attachment weights changed",
    )
    _require(
        support.shape == (group.material_particle_count,)
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
    _require(controller.ndim == 3 and controller.shape[2] == 3, "controller must have shape (T,C,3)")
    _require(indices.ndim == 1 and len(indices) >= 1, "attachment indices must be one-dimensional")
    _require(weights.shape == (len(indices), controller.shape[1]), "attachment weights changed")
    displacement = controller - controller[:1]
    weighted = np.einsum("ac,tcd->tad", weights, displacement, optimize=True)
    return cast(FloatArray, np.ascontiguousarray(points[indices][None, :, :] + weighted))


def _host(value: object) -> npt.NDArray[Any]:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy = getattr(current, "numpy", None)
    if callable(numpy):
        current = numpy()
    return np.ascontiguousarray(np.asarray(current)).copy()


def _domain_bounds(
    points_m: FloatArray,
    targets_m: FloatArray,
    *,
    padding_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    stacked = np.concatenate((points_m, targets_m.reshape(-1, 3)), axis=0)
    lower = np.min(stacked, axis=0) - padding_m
    upper = np.max(stacked, axis=0) + padding_m
    minimum_span = 0.1
    span = upper - lower
    deficit = np.maximum(minimum_span - span, 0.0)
    lower -= 0.5 * deficit
    upper += 0.5 * deficit
    lower_tuple = cast(tuple[float, float, float], tuple(float(value) for value in lower))
    upper_tuple = cast(tuple[float, float, float], tuple(float(value) for value in upper))
    return lower_tuple, upper_tuple


@dataclass(frozen=True, slots=True)
class _NativeReplay:
    positions_m: FloatArray
    active: BoolArray
    deformation_determinants: FloatArray


def _run_native_replay(
    *,
    gs: Any,
    torch: Any,
    points_m: FloatArray,
    targets_m: FloatArray,
    attachment_indices: npt.NDArray[np.int64],
    young_modulus_pa: float,
    substeps: int,
    simulation: Mapping[str, Any],
    driven: bool,
    translation_m: FloatArray | None = None,
) -> _NativeReplay:
    shift = np.zeros(3, dtype=np.float64) if translation_m is None else translation_m
    shifted_points = np.ascontiguousarray(points_m + shift)
    shifted_targets = np.ascontiguousarray(targets_m + shift[None, None, :])
    lower, upper = _domain_bounds(
        shifted_points,
        shifted_targets,
        padding_m=float(simulation["domain_padding_m"]),
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / float(simulation["fps"]),
            substeps=substeps,
            gravity=tuple(simulation["gravity_m_s2"]),
            requires_grad=False,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=float(simulation["grid_density"]),
            particle_size=float(simulation["particle_size_m"]),
            lower_bound=lower,
            upper_bound=upper,
        ),
        show_viewer=False,
    )
    entity = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=young_modulus_pa,
            nu=float(simulation["poisson_ratio"]),
            rho=float(simulation["density_kg_m3"]),
            model=str(simulation["constitutive_model"]),
        ),
        morph=gs.morphs.Nowhere(n_particles=len(shifted_points)),
        surface=gs.surfaces.Default(vis_mode="particle"),
    )
    scene.build()
    all_indices = torch.arange(len(shifted_points), dtype=torch.int64)
    entity.set_particles_active(torch.ones(len(shifted_points), dtype=torch.bool), all_indices)
    # Genesis 1.3.3 documents ``active`` as non-informative for Nowhere entities,
    # but its public set_free() guard still reads it. The emitter path likewise
    # activates particles directly. Establish the forward guard only after the
    # complete persistent roster has been activated in the solver.
    entity.active = True
    entity.set_particles_pos(torch.as_tensor(shifted_points, dtype=torch.float64), all_indices)
    entity.set_particles_vel(torch.zeros((len(shifted_points), 3), dtype=torch.float64), all_indices)
    free = torch.ones(len(shifted_points), dtype=torch.bool)
    free[torch.as_tensor(attachment_indices, dtype=torch.int64)] = False
    entity.set_free(free)

    frame_count = len(shifted_targets)
    positions: list[npt.NDArray[Any]] = []
    active: list[npt.NDArray[Any]] = []
    determinants: list[npt.NDArray[Any]] = []

    def record() -> None:
        state = entity.get_state()
        pos = _host(state.pos)[0]
        mask = _host(state.active)[0].astype(np.bool_, copy=False)
        deformation = _host(state.F)[0]
        positions.append(np.ascontiguousarray(pos).copy())
        active.append(np.ascontiguousarray(mask).copy())
        determinants.append(np.ascontiguousarray(np.linalg.det(deformation)).copy())

    record()
    frame_dt = 1.0 / float(simulation["fps"])
    attached_tensor = torch.as_tensor(attachment_indices, dtype=torch.int64)
    for frame in range(1, frame_count):
        target = shifted_targets[frame] if driven else shifted_targets[0]
        previous = shifted_targets[frame - 1] if driven else shifted_targets[0]
        velocity = (target - previous) / frame_dt
        entity.set_particles_pos(torch.as_tensor(target, dtype=torch.float64), attached_tensor)
        entity.set_particles_vel(torch.as_tensor(velocity, dtype=torch.float64), attached_tensor)
        scene.step()
        record()
    return _NativeReplay(
        positions_m=cast(FloatArray, np.ascontiguousarray(np.stack(positions))),
        active=np.ascontiguousarray(np.stack(active), dtype=np.bool_),
        deformation_determinants=cast(
            FloatArray,
            np.ascontiguousarray(np.stack(determinants)),
        ),
    )


def _git_provenance(repo_root: Path) -> dict[str, Any]:
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
    source_paths = (
        "src/bayesian_phystwin/genesis_mpm_source_qualification_v1.py",
        "scripts/remote/run_genesis_mpm_source_qualification_v1.py",
    )
    return {
        "git_head": head,
        "git_worktree_clean": True,
        "source_files": {
            relative: file_sha256(repo_root / relative) for relative in source_paths
        },
    }


def _rmse(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def run_genesis_mpm_source_qualification_v1(
    *,
    protocol_path: str | Path,
    group_roots: Mapping[str, str | Path],
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    protocol = load_genesis_source_physics_protocol_v1(protocol_path)
    if set(group_roots) != {group.group_id for group in protocol.source_groups}:
        raise ValueError("group roots must match the complete frozen source roster")
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = _git_provenance(Path(repo_root).absolute())

    try:
        import genesis as gs
        import torch
    except ImportError as error:  # pragma: no cover - optional native runtime
        raise RuntimeError("native Genesis source qualification requires Genesis and torch") from error

    simulation = protocol.simulation
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gs.init(
        backend=gs.cpu,
        precision=str(simulation["precision"]),
        seed=int(simulation["seed"]),
        logging_level="warning",
    )

    gates = protocol.gates
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

    for group in protocol.source_groups:
        root = Path(group_roots[group.group_id]).absolute()
        source_path = root / group.source_inputs_relative_path.as_posix()
        incumbent_path = root / group.incumbent_relative_path.as_posix()
        _require(incumbent_path.is_file() and not incumbent_path.is_symlink(), "incumbent must be an ordinary file")
        _require(file_sha256(incumbent_path) == group.incumbent_sha256, "incumbent SHA-256 changed")
        arrays = load_genesis_source_inputs_v1(source_path, group=group)
        incumbent = load_physical_rollout_archive(
            incumbent_path,
            expected_frame_count=group.frame_count,
        )
        _require(
            incumbent["prediction_m"].shape == (group.frame_count, group.material_particle_count, 3),
            "incumbent physical shape changed",
        )
        qualification_frames = int(simulation["qualification_frame_count"])
        _require(qualification_frames <= group.frame_count, "qualification horizon exceeds source action")
        points = np.asarray(arrays["frame_zero_points_m"], dtype=np.float64)
        controller = np.asarray(arrays["controller_points_m"][:qualification_frames], dtype=np.float64)
        indices = np.asarray(arrays["attachment_indices"], dtype=np.int64)
        targets = attachment_targets_m(points, controller, indices, arrays["attachment_weights"])
        common = {
            "gs": gs,
            "torch": torch,
            "points_m": points,
            "targets_m": targets,
            "attachment_indices": indices,
            "simulation": simulation,
        }
        base = _run_native_replay(
            **common,
            young_modulus_pa=float(simulation["base_young_modulus_pa"]),
            substeps=int(simulation["base_substeps"]),
            driven=True,
        )
        repeat = _run_native_replay(
            **common,
            young_modulus_pa=float(simulation["base_young_modulus_pa"]),
            substeps=int(simulation["base_substeps"]),
            driven=True,
        )
        zero = _run_native_replay(
            **common,
            young_modulus_pa=float(simulation["base_young_modulus_pa"]),
            substeps=int(simulation["base_substeps"]),
            driven=False,
        )
        translated = _run_native_replay(
            **common,
            young_modulus_pa=float(simulation["base_young_modulus_pa"]),
            substeps=int(simulation["base_substeps"]),
            driven=True,
            translation_m=np.asarray(simulation["rigid_translation_m"], dtype=np.float64),
        )
        refined = _run_native_replay(
            **common,
            young_modulus_pa=float(simulation["base_young_modulus_pa"]),
            substeps=int(simulation["refined_substeps"]),
            driven=True,
        )
        soft = _run_native_replay(
            **common,
            young_modulus_pa=float(simulation["soft_young_modulus_pa"]),
            substeps=int(simulation["base_substeps"]),
            driven=True,
        )
        stiff = _run_native_replay(
            **common,
            young_modulus_pa=float(simulation["stiff_young_modulus_pa"]),
            substeps=int(simulation["base_substeps"]),
            driven=True,
        )

        deterministic = bool(
            np.array_equal(base.positions_m, repeat.positions_m)
            and np.array_equal(base.active, repeat.active)
            and np.array_equal(base.deformation_determinants, repeat.deformation_determinants)
        )
        active_expected = np.ones_like(base.active, dtype=np.bool_)
        topology = all(
            replay.positions_m.shape == base.positions_m.shape
            and replay.active.shape == active_expected.shape
            and np.array_equal(replay.active, active_expected)
            for replay in (base, repeat, zero, translated, refined, soft, stiff)
        )
        zero_drift = float(np.max(np.linalg.norm(zero.positions_m - points[None, :, :], axis=2)))
        shift = np.asarray(simulation["rigid_translation_m"], dtype=np.float64)
        equivariance = float(
            np.max(np.linalg.norm(translated.positions_m - shift[None, None, :] - base.positions_m, axis=2))
        )
        response = _rmse(base.positions_m[-1], points)
        refinement = _rmse(base.positions_m[-1], refined.positions_m[-1]) / max(
            _rmse(refined.positions_m[-1], points), 1.0e-15
        )
        parameter_sensitivity = _rmse(soft.positions_m[-1], stiff.positions_m[-1])
        parity = _rmse(base.positions_m[0], incumbent["prediction_m"][0])
        maximum_step = float(
            np.max(np.linalg.norm(np.diff(base.positions_m, axis=0), axis=2))
        )
        determinants = np.concatenate(
            tuple(replay.deformation_determinants.reshape(-1) for replay in (base, zero, translated, refined, soft, stiff))
        )
        finite = bool(
            all(np.all(np.isfinite(replay.positions_m)) for replay in (base, repeat, zero, translated, refined, soft, stiff))
            and np.all(np.isfinite(determinants))
        )
        group_sanity = {
            "finite": finite,
            "action_response": response >= float(gates["minimum_action_response_m"]),
            "parameter_sensitivity_lower": parameter_sensitivity >= float(gates["minimum_parameter_sensitivity_m"]),
            "parameter_sensitivity_upper": parameter_sensitivity <= float(gates["maximum_parameter_sensitivity_m"]),
            "particle_step": maximum_step <= float(gates["maximum_particle_step_m"]),
            "deformation_determinant_lower": float(np.min(determinants)) >= float(gates["minimum_deformation_determinant"]),
            "deformation_determinant_upper": float(np.max(determinants)) <= float(gates["maximum_deformation_determinant"]),
        }
        sanity_violations += sum(not value for value in group_sanity.values())
        units_valid = bool(
            base.positions_m.shape == (qualification_frames, group.material_particle_count, 3)
            and base.positions_m.dtype.kind == "f"
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
                "translated_driven_m": translated.positions_m,
                "refined_driven_m": refined.positions_m,
                "soft_driven_m": soft.positions_m,
                "stiff_driven_m": stiff.positions_m,
                "base_active": base.active,
                "base_deformation_determinant": base.deformation_determinants,
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
            "maximum_zero_action_drift_m": zero_drift,
            "maximum_rigid_translation_equivariance_error_m": equivariance,
            "time_step_refinement_relative_error": refinement,
            "source_query_parity_rmse_m": parity,
            "action_response_rmse_m": response,
            "parameter_sensitivity_rmse_m": parameter_sensitivity,
            "maximum_particle_step_m": maximum_step,
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
            gates["maximum_rigid_translation_equivariance_error_m"]
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
            "engine_revision": cast(Mapping[str, Any], protocol.value["backend"])["engine_revision"],
            "engine_version": cast(Mapping[str, Any], protocol.value["backend"])["engine_version"],
            "native_smoke_id": cast(Mapping[str, Any], protocol.value["backend"])["native_smoke_id"],
            "parameter_sensitivity_is_a_physical_sanity_gate": True,
            "rigid_equivariance_probe": "translation-only",
            "nowhere_activation_policy": simulation["nowhere_activation_policy"],
            "gradient_claim": "none",
        },
    )
    save_material_backend_qualification_v1(
        qualification,
        output / QUALIFICATION_FILENAME,
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
    "GenesisSourceGroupV1",
    "GenesisSourcePhysicsProtocolV1",
    "QUALIFICATION_FILENAME",
    "RESULT_FILENAME",
    "attachment_targets_m",
    "file_sha256",
    "load_genesis_source_inputs_v1",
    "load_genesis_source_physics_protocol_v1",
    "run_genesis_mpm_source_qualification_v1",
]
