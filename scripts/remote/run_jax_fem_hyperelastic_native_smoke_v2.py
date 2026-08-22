#!/usr/bin/env python3
"""Run the pinned native stable-Neo-Hookean JAX-FEM v2 smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin.jax_fem_producer_v1 import produce_jax_fem_backend
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive

JAX_FEM_REVISION = "82c6993c16704e38611f9cb91a5b70f1c690daee"
JAX_FEM_VERSION = "0.0.12"
JAX_FEM_REPOSITORY = "https://github.com/deepmodeling/jax-fem"
JAX_FEM_SOURCE_BLOBS = {
    "__init__.py": "eed02f352137abe508490d6b9b08b4807d1c94ed",
    "basis.py": "a7b7b04445baad3a25d3c43401e63745797b0141",
    "fe.py": "771030879228331a036608950c6e14e54e6f21c3",
    "generate_mesh.py": "bd564c8f4a049ae28bc3592e21d9547a5f509629",
    "problem.py": "8a20d24fc2e98aa33d4bd76e543f00c471740551",
    "solver.py": "f0f64cb629e202f2d179710b745ea4d682f1ace2",
}
RUNTIME_VERSIONS = {
    "python": "3.12.13",
    "jax": "0.4.38",
    "jaxlib": "0.4.38",
    "jax-fem": JAX_FEM_VERSION,
    "numpy": "2.2.6",
    "scipy": "1.15.2",
    "petsc4py": "3.23.7",
    "gmsh": "4.13.1",
    "meshio": "5.3.5",
}
SMOKE_SCHEMA = "bayesian-phystwin.jax-fem-hyperelastic-native-smoke-v2"
PORTABLE_MEMBERS = (
    "SHA256SUMS",
    "lagrangian-backend.json",
    "physical-prediction.npz",
    "provenance/lagrangian-rollout.npz",
    "provenance/lagrangian-runtime.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def _content_id(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _repository_revision(script_path: Path) -> str:
    repository_root = script_path.parents[2]
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
        raise RuntimeError("unable to resolve an exact BayesianPhysTwin revision")
    return revision


def _lame_parameters(
    young_modulus_pa: float,
    poisson_ratio: float,
) -> tuple[float, float, float]:
    shear = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    first_lame = (
        young_modulus_pa
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    stable_alpha = 1.0 + shear / first_lame
    return shear, first_lame, stable_alpha


def _mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.array(
        [
            [i * 0.02, j * 0.01, k * 0.01]
            for i in range(3)
            for j in range(3)
            for k in range(3)
        ],
        dtype=np.float64,
    )

    def node(i: int, j: int, k: int) -> int:
        return i * 9 + j * 3 + k

    hexahedra: list[list[int]] = []
    tetrahedra: list[list[int]] = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = node(i, j, k)
                b = node(i + 1, j, k)
                c = node(i, j + 1, k)
                d = node(i + 1, j + 1, k)
                e = node(i, j, k + 1)
                f = node(i + 1, j, k + 1)
                g = node(i, j + 1, k + 1)
                h = node(i + 1, j + 1, k + 1)
                hexahedra.append([a, b, d, c, e, f, h, g])
                tetrahedra.extend(
                    (
                        [a, b, d, h],
                        [a, d, c, h],
                        [a, c, g, h],
                        [a, g, e, h],
                        [a, e, f, h],
                        [a, f, b, h],
                    )
                )
    cells = np.asarray(hexahedra, dtype=np.int32)
    tets = np.asarray(tetrahedra, dtype=np.int32)
    determinants = _deformation_determinants(points, points, tets)
    if (
        points.shape != (27, 3)
        or cells.shape != (8, 8)
        or tets.shape != (48, 4)
        or not np.array_equal(determinants, np.ones(48))
    ):
        raise RuntimeError("JAX-FEM v2 synthetic mesh construction changed")
    return points, cells, tets


def _deformation_determinants(
    reference: np.ndarray,
    positions: np.ndarray,
    tetrahedra: np.ndarray,
) -> np.ndarray:
    reference_cells = reference[tetrahedra]
    current_cells = positions[tetrahedra]
    reference_edges = np.stack(
        (
            reference_cells[:, 1] - reference_cells[:, 0],
            reference_cells[:, 2] - reference_cells[:, 0],
            reference_cells[:, 3] - reference_cells[:, 0],
        ),
        axis=2,
    )
    current_edges = np.stack(
        (
            current_cells[:, 1] - current_cells[:, 0],
            current_cells[:, 2] - current_cells[:, 0],
            current_cells[:, 3] - current_cells[:, 0],
        ),
        axis=2,
    )
    gradients = np.linalg.solve(
        np.swapaxes(reference_edges, 1, 2),
        np.swapaxes(current_edges, 1, 2),
    )
    return np.linalg.det(np.swapaxes(gradients, 1, 2))


@dataclass(frozen=True, slots=True)
class _NativeModules:
    jax: Any
    jnp: Any
    Mesh: Any
    Problem: Any
    solver: Any
    package_root: Path
    source_records: Mapping[str, Mapping[str, str]]


def _load_native_modules() -> _NativeModules:
    observed_versions = {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in RUNTIME_VERSIONS
            if name != "python"
        },
    }
    if observed_versions != RUNTIME_VERSIONS:
        raise RuntimeError(
            "JAX-FEM v2 runtime versions changed: "
            f"expected {RUNTIME_VERSIONS}, found {observed_versions}"
        )

    jax = importlib.import_module("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = importlib.import_module("jax.numpy")
    jax_fem = importlib.import_module("jax_fem")
    generate_mesh = importlib.import_module("jax_fem.generate_mesh")
    problem_module = importlib.import_module("jax_fem.problem")
    solver_module = importlib.import_module("jax_fem.solver")

    package_file = getattr(jax_fem, "__file__", None)
    if not package_file:
        raise RuntimeError("jax_fem package does not expose __file__")
    package_root = Path(package_file).resolve().parent
    source_records: dict[str, Mapping[str, str]] = {}
    for name, expected_blob in JAX_FEM_SOURCE_BLOBS.items():
        path = package_root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"JAX-FEM source is not an ordinary file: {path}")
        observed_blob = _git_blob_sha1(path)
        if observed_blob != expected_blob:
            raise RuntimeError(
                f"JAX-FEM source mismatch for {name}: expected Git blob "
                f"{expected_blob}, found {observed_blob}"
            )
        source_records[f"jax_fem/{name}"] = {
            "git_blob_sha1": observed_blob,
            "sha256": _sha256_file(path),
        }
    return _NativeModules(
        jax=jax,
        jnp=jnp,
        Mesh=generate_mesh.Mesh,
        Problem=problem_module.Problem,
        solver=solver_module.solver,
        package_root=package_root,
        source_records=source_records,
    )


def _stable_neo_hookean_energy(
    native: _NativeModules,
    *,
    young_modulus_pa: float,
    poisson_ratio: float,
) -> Any:
    jnp = native.jnp
    shear, first_lame, alpha = _lame_parameters(
        young_modulus_pa,
        poisson_ratio,
    )

    def energy(deformation_gradient: Any) -> Any:
        first_invariant = jnp.trace(deformation_gradient.T @ deformation_gradient)
        determinant = jnp.linalg.det(deformation_gradient)
        return (
            0.5 * shear * (first_invariant - 3.0)
            + 0.5 * first_lame * (determinant - alpha) ** 2
        )

    return energy


def _right_face_displacement(
    point: Any,
    *,
    jnp: Any,
    load_fraction: float,
    twist_angle_rad: float,
    axial_displacement_m: float,
) -> Any:
    angle = load_fraction * twist_angle_rad
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    relative_y = point[1] - 0.01
    relative_z = point[2] - 0.01
    target = jnp.array(
        [
            point[0] + load_fraction * axial_displacement_m,
            0.01 + cosine * relative_y - sine * relative_z,
            0.01 + sine * relative_y + cosine * relative_z,
        ]
    )
    return target - point


class _NativeReplay:
    def __init__(
        self,
        native: _NativeModules,
        *,
        young_modulus_pa: float,
        poisson_ratio: float,
        twist_angle_rad: float,
        axial_displacement_m: float,
        continuation_substeps: int,
        minimum_deformation_determinant: float,
    ) -> None:
        points, cells, tetrahedra = _mesh()
        self.native = native
        self.reference = native.jnp.asarray(points)
        self.mesh = native.Mesh(
            self.reference,
            native.jnp.asarray(cells),
            ele_type="HEX8",
        )
        self.tetrahedra = tetrahedra
        self.young_modulus_pa = young_modulus_pa
        self.poisson_ratio = poisson_ratio
        self.twist_angle_rad = twist_angle_rad
        self.axial_displacement_m = axial_displacement_m
        self.continuation_substeps = continuation_substeps
        self.minimum_allowed_determinant = minimum_deformation_determinant
        self.target_fraction = 0.0
        self.current_fraction = 0.0
        self.solution = np.zeros_like(points)
        self.solve_count = 0
        self.determinant_history: list[float] = []

    def get_reference_points_m(self) -> object:
        return self.reference

    def set_load_fraction(self, fraction: float) -> None:
        value = float(fraction)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("load fraction must be finite and in [0,1]")
        if value + 1e-15 < self.current_fraction:
            raise ValueError("JAX-FEM v2 continuation load cannot move backward")
        self.target_fraction = value

    def _solve_fraction(self, fraction: float) -> None:
        native = self.native
        jnp = native.jnp
        energy = _stable_neo_hookean_energy(
            native,
            young_modulus_pa=self.young_modulus_pa,
            poisson_ratio=self.poisson_ratio,
        )

        def get_tensor_map(problem: Any) -> Any:
            stress = native.jax.grad(energy)

            def first_piola(displacement_gradient: Any) -> Any:
                return stress(displacement_gradient + jnp.eye(problem.dim))

            return first_piola

        stable_neo_hookean = cast(
            type[Any],
            type(
                "StableNeoHookeanV2",
                (native.Problem,),
                {"get_tensor_map": get_tensor_map},
            ),
        )

        def left(point: Any) -> Any:
            return jnp.isclose(point[0], 0.0, atol=1e-9)

        def right(point: Any) -> Any:
            return jnp.isclose(point[0], 0.04, atol=1e-9)

        def zero(_: Any) -> float:
            return 0.0

        def prescribed(component: int) -> Any:
            def value(point: Any) -> Any:
                return _right_face_displacement(
                    point,
                    jnp=jnp,
                    load_fraction=fraction,
                    twist_angle_rad=self.twist_angle_rad,
                    axial_displacement_m=self.axial_displacement_m,
                )[component]

            return value

        problem = stable_neo_hookean(
            mesh=self.mesh,
            vec=3,
            dim=3,
            ele_type="HEX8",
            dirichlet_bc_info=[
                [left, left, left, right, right, right],
                [0, 1, 2, 0, 1, 2],
                [zero, zero, zero, prescribed(0), prescribed(1), prescribed(2)],
            ],
        )
        solution = native.solver(
            problem,
            solver_options={
                "newton": {
                    "tol": 1e-8,
                    "rel_tol": 1e-10,
                    "line_search_flag": True,
                    "initial_guess": [jnp.asarray(self.solution)],
                    "linear": {"spsolve_solver": {}},
                }
            },
        )
        if not isinstance(solution, (list, tuple)) or len(solution) != 1:
            raise RuntimeError(
                "JAX-FEM v2 solver returned an unexpected solution structure"
            )
        displacement = np.ascontiguousarray(np.asarray(solution[0]), dtype=np.float64)
        if displacement.shape != self.solution.shape or not np.all(
            np.isfinite(displacement)
        ):
            raise RuntimeError("JAX-FEM v2 displacement state is invalid")
        determinants = _deformation_determinants(
            np.asarray(self.reference),
            np.asarray(self.reference) + displacement,
            self.tetrahedra,
        )
        minimum = float(np.min(determinants))
        if (
            not np.all(np.isfinite(determinants))
            or minimum < self.minimum_allowed_determinant
        ):
            raise RuntimeError(
                "JAX-FEM v2 continuation violated its orientation threshold"
            )
        self.solution = displacement
        self.solve_count += 1
        self.determinant_history.append(minimum)

    def solve(self) -> object:
        delta = self.target_fraction - self.current_fraction
        steps = self.continuation_substeps if delta > 1e-15 else 1
        start = self.current_fraction
        for index in range(1, steps + 1):
            fraction = start + delta * index / steps
            self._solve_fraction(fraction)
        self.current_fraction = self.target_fraction
        return self.native.jnp.asarray(self.solution)


def _portable_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for member in PORTABLE_MEMBERS:
        path = directory / member
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing portable member: {member}")
        hashes[member] = _sha256_file(path)
    return hashes


def _run_once(
    native: _NativeModules,
    output_dir: Path,
    *,
    frame_count: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    twist_angle_rad: float,
    axial_displacement_m: float,
    continuation_substeps: int,
    minimum_deformation_determinant: float,
    script_path: Path,
) -> Mapping[str, Any]:
    replays: list[_NativeReplay] = []

    def replay_factory() -> _NativeReplay:
        replay = _NativeReplay(
            native,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
            twist_angle_rad=twist_angle_rad,
            axial_displacement_m=axial_displacement_m,
            continuation_substeps=continuation_substeps,
            minimum_deformation_determinant=minimum_deformation_determinant,
        )
        replays.append(replay)
        return replay

    def driven_control(index: int, replay: Any) -> None:
        replay.set_load_fraction((index + 1) / (frame_count - 1))

    def zero_control(_: int, replay: Any) -> None:
        replay.set_load_fraction(0.0)

    source_artifacts = {
        "scripts/remote/run_jax_fem_hyperelastic_native_smoke_v2.py": (
            _sha256_file(script_path)
        ),
        **{
            f"native/{path}": record["sha256"]
            for path, record in native.source_records.items()
        },
    }
    devices = native.jax.devices()
    if not devices:
        raise RuntimeError("JAX reports no execution device")
    primary = devices[0]
    device = (
        f"{getattr(primary, 'platform', 'unknown')}:"
        f"{getattr(primary, 'id', 'unknown')}:"
        f"{getattr(primary, 'device_kind', 'unknown')}"
    )
    artifact = produce_jax_fem_backend(
        output_dir=output_dir,
        replay_factory=replay_factory,
        driven_control=driven_control,
        zero_action_control=zero_control,
        frame_count=frame_count,
        material_query_indices=np.arange(27, dtype=np.int64),
        action_support=np.repeat(
            np.array([0.0, 0.5, 1.0], dtype=np.float64),
            9,
        ),
        engine_revision=JAX_FEM_REVISION,
        engine_version=JAX_FEM_VERSION,
        source_artifacts=source_artifacts,
        device=device,
        load_step_size=1.0 / (frame_count - 1),
        element_type="HEX8",
        constitutive_model=("stable-Neo-Hookean-Smith-2018-finite-deformation-v2"),
        nonlinear_solver=(
            "jax-fem-Newton-scipy-spsolve-line-search-warm-start-continuation"
        ),
        source_kind="synthetic",
    )
    arrays = load_physical_rollout_archive(
        output_dir / "physical-prediction.npz",
        expected_frame_count=frame_count,
    )
    zero_delta = arrays["zero_action_readout_m"] - arrays["frame_zero_points_m"][None]
    response = arrays["driven_readout_m"] - arrays["zero_action_readout_m"]
    final = np.asarray(arrays["driven_readout_m"][-1], dtype=np.float64)
    reference = np.asarray(arrays["frame_zero_points_m"], dtype=np.float64)
    _, _, tetrahedra = _mesh()
    determinants = _deformation_determinants(reference, final, tetrahedra)
    return {
        "artifact_id": artifact["artifact_id"],
        "maximum_zero_action_drift_m": float(
            np.max(np.linalg.norm(zero_delta, axis=-1))
        ),
        "maximum_driven_minus_zero_response_m": float(
            np.max(np.linalg.norm(response, axis=-1))
        ),
        "minimum_final_deformation_determinant": float(np.min(determinants)),
        "maximum_final_deformation_determinant": float(np.max(determinants)),
        "driven_native_solve_count": replays[0].solve_count,
        "zero_native_solve_count": replays[1].solve_count,
        "minimum_continuation_deformation_determinant": float(
            min(min(replay.determinant_history) for replay in replays)
        ),
        "portable_sha256": _portable_hashes(output_dir),
    }


def _objectivity_error(
    native: _NativeModules,
    *,
    young_modulus_pa: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    jnp = native.jnp
    energy = _stable_neo_hookean_energy(
        native,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
    )
    stress = native.jax.grad(energy)
    angle = 0.73
    rotation = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rest = np.asarray(stress(jnp.eye(3)), dtype=np.float64)
    rotated = np.asarray(stress(rotation), dtype=np.float64)
    scale = max(young_modulus_pa, 1.0)
    return float(np.max(np.abs(rest)) / scale), float(np.max(np.abs(rotated)) / scale)


def run_smoke(
    output_dir: str | Path,
    *,
    frame_count: int = 7,
    young_modulus_pa: float = 100_000.0,
    poisson_ratio: float = 0.35,
    twist_angle_rad: float = np.pi / 3.0,
    axial_displacement_m: float = 0.005,
    continuation_substeps: int = 2,
    minimum_deformation_determinant: float = 0.35,
) -> Mapping[str, Any]:
    if type(frame_count) is not int or frame_count < 3:
        raise ValueError("frame_count must be an integer >= 3")
    if type(continuation_substeps) is not int or continuation_substeps < 1:
        raise ValueError("continuation_substeps must be a positive integer")
    for name, value in (
        ("young_modulus_pa", young_modulus_pa),
        ("twist_angle_rad", twist_angle_rad),
        ("axial_displacement_m", axial_displacement_m),
    ):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(poisson_ratio) or not 0.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (0,0.5)")
    if (
        not np.isfinite(minimum_deformation_determinant)
        or not 0.0 < minimum_deformation_determinant < 1.0
    ):
        raise ValueError("minimum_deformation_determinant must lie in (0,1)")

    root = Path(output_dir).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    native = _load_native_modules()
    script_path = Path(__file__).resolve()
    producer_revision = _repository_revision(script_path)
    rest_error, rotation_error = _objectivity_error(
        native,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
    )
    objectivity_tolerance = 1e-10
    if rest_error > objectivity_tolerance or rotation_error > objectivity_tolerance:
        raise RuntimeError("stable Neo-Hookean stress failed the objectivity gate")

    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.staging.",
        dir=root.parent,
    ) as temporary:
        staging = Path(temporary) / root.name
        staging.mkdir()

        def run_once(name: str) -> Mapping[str, Any]:
            return _run_once(
                native,
                staging / name,
                frame_count=frame_count,
                young_modulus_pa=young_modulus_pa,
                poisson_ratio=poisson_ratio,
                twist_angle_rad=twist_angle_rad,
                axial_displacement_m=axial_displacement_m,
                continuation_substeps=continuation_substeps,
                minimum_deformation_determinant=minimum_deformation_determinant,
                script_path=script_path,
            )

        run_a = run_once("run-a")
        run_b = run_once("run-b")
        deterministic = run_a["portable_sha256"] == run_b["portable_sha256"]
        if not deterministic:
            raise RuntimeError("JAX-FEM v2 replay is not byte-deterministic")
        zero_drift = float(run_a["maximum_zero_action_drift_m"])
        response = float(run_a["maximum_driven_minus_zero_response_m"])
        minimum_determinant = float(
            run_a["minimum_continuation_deformation_determinant"]
        )
        if zero_drift > 1e-10:
            raise RuntimeError("JAX-FEM v2 zero-action drift exceeds the smoke bound")
        if response <= 0.005:
            raise RuntimeError("JAX-FEM v2 driven arm did not reach finite deformation")
        if minimum_determinant < minimum_deformation_determinant:
            raise RuntimeError("JAX-FEM v2 deformation determinant gate failed")

        devices = [
            {
                "platform": str(getattr(device, "platform", "unknown")),
                "id": str(getattr(device, "id", "unknown")),
                "device_kind": str(getattr(device, "device_kind", "unknown")),
            }
            for device in native.jax.devices()
        ]
        descriptor: dict[str, Any] = {
            "schema": SMOKE_SCHEMA,
            "claim_boundary": (
                "Synthetic native finite-deformation, stable-Neo-Hookean, "
                "continuation, objectivity, orientation, and provenance smoke only; "
                "no source-value, fresh-object, calibration, or downstream claim."
            ),
            "backend_profile": "jax-fem-quasistatic-v1",
            "backend_variant": "jax-fem-stable-neo-hookean-v2",
            "engine": {
                "repository": JAX_FEM_REPOSITORY,
                "revision": JAX_FEM_REVISION,
                "version": JAX_FEM_VERSION,
                "source_records": native.source_records,
            },
            "producer_revision": producer_revision,
            "runtime": {
                "versions": RUNTIME_VERSIONS,
                "jax_enable_x64": bool(native.jax.config.jax_enable_x64),
                "devices": devices,
            },
            "problem": {
                "element_type": "HEX8",
                "point_count": 27,
                "cell_count": 8,
                "tetrahedral_orientation_probe_count": 48,
                "frame_count": frame_count,
                "young_modulus_pa": young_modulus_pa,
                "poisson_ratio": poisson_ratio,
                "twist_angle_rad": twist_angle_rad,
                "axial_displacement_m": axial_displacement_m,
                "continuation_substeps": continuation_substeps,
                "constitutive_model": (
                    "Smith-2018 stable Neo-Hookean finite deformation"
                ),
            },
            "checks": {
                "installed_source_matches_pinned_git_blobs": True,
                "portable_replay_byte_deterministic": deterministic,
                "maximum_zero_action_drift_m": zero_drift,
                "maximum_driven_minus_zero_response_m": response,
                "minimum_continuation_deformation_determinant": minimum_determinant,
                "minimum_required_deformation_determinant": (
                    minimum_deformation_determinant
                ),
                "normalized_rest_stress_error": rest_error,
                "normalized_rigid_rotation_stress_error": rotation_error,
                "objectivity_tolerance": objectivity_tolerance,
            },
            "run_a": run_a,
            "run_b": run_b,
            "future_outcomes_read": False,
            "dataset_payload_read": False,
        }
        descriptor["smoke_id"] = _content_id(descriptor)
        result_path = staging / "jax-fem-hyperelastic-native-smoke-v2.json"
        result_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, root)
    return descriptor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-count", type=int, default=7)
    parser.add_argument("--young-modulus-pa", type=float, default=100_000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.35)
    parser.add_argument("--twist-angle-rad", type=float, default=np.pi / 3.0)
    parser.add_argument("--axial-displacement-m", type=float, default=0.005)
    parser.add_argument("--continuation-substeps", type=int, default=2)
    parser.add_argument("--minimum-deformation-determinant", type=float, default=0.35)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_smoke(
        args.output_dir,
        frame_count=args.frame_count,
        young_modulus_pa=args.young_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        twist_angle_rad=args.twist_angle_rad,
        axial_displacement_m=args.axial_displacement_m,
        continuation_substeps=args.continuation_substeps,
        minimum_deformation_determinant=args.minimum_deformation_determinant,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
