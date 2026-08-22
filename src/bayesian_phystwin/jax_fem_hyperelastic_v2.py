"""Pinned finite-deformation JAX-FEM v2 runtime helpers.

This module contains the reusable native solver used by the source-only
qualification and, if that gate passes, the later source-value prediction.
It never loads observations or scoring artifacts.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .jax_fem_source_qualification_v1 import (
    RigidContactProjectionV1,
    deformation_determinants_v1,
    file_sha256,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntegerArray: TypeAlias = npt.NDArray[np.integer[Any]]

RUNTIME_SCHEMA = "bayesian-phystwin.jax-fem-hyperelastic-runtime-v2"
BACKEND_VARIANT = "jax-fem-stable-neo-hookean-v2"
CONSTITUTIVE_MODEL = "Smith-2018-stable-Neo-Hookean-finite-deformation"
NONLINEAR_SOLVER = "JAX-FEM-Newton-SciPy-spsolve-line-search-warm-start"
CONTINUATION_POLICY = "one-source-frame-step-base-two-substep-refinement-v2"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


@dataclass(frozen=True, slots=True)
class NativeJaxFemModulesV2:
    """Exact native modules admitted by the frozen runtime descriptor."""

    jax: Any
    jnp: Any
    Mesh: Any
    Problem: Any
    solver: Any
    package_root: Path


def load_native_jax_fem_modules_v2(
    *,
    runtime_versions: Mapping[str, str],
    installed_source_sha256: Mapping[str, str],
) -> NativeJaxFemModulesV2:
    """Load and hash-check the exact CPU JAX-FEM runtime."""

    expected_version_keys = frozenset(
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
    _require(
        frozenset(runtime_versions) == expected_version_keys,
        "JAX-FEM v2 runtime version roster changed",
    )
    observed_versions = {
        "python": platform.python_version(),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "jax_fem": importlib.metadata.version("jax-fem"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "petsc4py": importlib.metadata.version("petsc4py"),
        "gmsh": importlib.metadata.version("gmsh"),
        "meshio": importlib.metadata.version("meshio"),
    }
    _require(
        observed_versions == dict(runtime_versions),
        "JAX-FEM v2 runtime versions changed",
    )

    jax = importlib.import_module("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = importlib.import_module("jax.numpy")
    jax_fem = importlib.import_module("jax_fem")
    package_file = getattr(jax_fem, "__file__", None)
    _require(package_file is not None, "JAX-FEM package path is unavailable")
    package_root = Path(cast(str, package_file)).resolve().parent
    for relative, expected in installed_source_sha256.items():
        path = package_root.parent / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"JAX-FEM source is unavailable: {relative}",
        )
        _require(file_sha256(path) == expected, f"JAX-FEM source changed: {relative}")
    devices = jax.devices()
    _require(
        len(devices) >= 1 and all(device.platform == "cpu" for device in devices),
        "JAX-FEM v2 qualification requires the frozen CPU runtime",
    )
    generate_mesh = importlib.import_module("jax_fem.generate_mesh")
    problem = importlib.import_module("jax_fem.problem")
    solver = importlib.import_module("jax_fem.solver")
    return NativeJaxFemModulesV2(
        jax=jax,
        jnp=jnp,
        Mesh=generate_mesh.Mesh,
        Problem=problem.Problem,
        solver=solver.solver,
        package_root=package_root,
    )


def lame_parameters_v2(
    young_modulus_pa: float,
    poisson_ratio: float,
) -> tuple[float, float, float]:
    """Return shear, first Lame, and stable-Neo-Hookean alpha."""

    young = _finite(young_modulus_pa, name="young_modulus_pa", positive=True)
    poisson = _finite(poisson_ratio, name="poisson_ratio")
    _require(0.0 < poisson < 0.5, "poisson_ratio must lie in (0,0.5)")
    shear = young / (2.0 * (1.0 + poisson))
    first_lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    alpha = 1.0 + shear / first_lame
    return shear, first_lame, alpha


def stable_neo_hookean_energy_v2(
    native: NativeJaxFemModulesV2,
    *,
    young_modulus_pa: float,
    poisson_ratio: float,
) -> Any:
    """Construct the frozen stable-Neo-Hookean energy density."""

    shear, first_lame, alpha = lame_parameters_v2(
        young_modulus_pa,
        poisson_ratio,
    )
    jnp = native.jnp

    def energy(deformation_gradient: Any) -> Any:
        first_invariant = jnp.trace(deformation_gradient.T @ deformation_gradient)
        determinant = jnp.linalg.det(deformation_gradient)
        return (
            0.5 * shear * (first_invariant - 3.0)
            + 0.5 * first_lame * (determinant - alpha) ** 2
        )

    return energy


def normalized_objectivity_errors_v2(
    native: NativeJaxFemModulesV2,
    *,
    young_modulus_pa: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    """Probe zero stress at rest and under a rigid rotation."""

    energy = stable_neo_hookean_energy_v2(
        native,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
    )
    stress = native.jax.grad(energy)
    jnp = native.jnp
    angle = 0.73
    rotation = jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = max(float(young_modulus_pa), 1.0)
    rest = np.asarray(stress(jnp.eye(3)), dtype=np.float64)
    rotated = np.asarray(stress(rotation), dtype=np.float64)
    return (
        float(np.max(np.abs(rest)) / scale),
        float(np.max(np.abs(rotated)) / scale),
    )


def interpolate_rotation_v2(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    fraction: float,
) -> FloatArray:
    """Interpolate two rotations and project deterministically onto SO(3)."""

    start = np.asarray(left, dtype=np.float64)
    end = np.asarray(right, dtype=np.float64)
    alpha = _finite(fraction, name="fraction")
    _require(start.shape == (3, 3) and end.shape == (3, 3), "rotations changed")
    _require(0.0 <= alpha <= 1.0, "fraction must lie in [0,1]")
    if alpha == 0.0:
        return np.ascontiguousarray(start)
    if alpha == 1.0:
        return np.ascontiguousarray(end)
    left_singular, _, right_singular = np.linalg.svd(
        (1.0 - alpha) * start + alpha * end,
        full_matrices=False,
    )
    rotation = left_singular @ right_singular
    if np.linalg.det(rotation) < 0.0:
        left_singular[:, -1] *= -1.0
        rotation = left_singular @ right_singular
    _require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12, rtol=0.0)
        and np.linalg.det(rotation) > 0.0,
        "interpolated contact rotation left SO(3)",
    )
    return np.ascontiguousarray(rotation)


@dataclass(frozen=True, slots=True)
class HyperelasticReplayV2:
    """Native source-frame trajectory and continuation diagnostics."""

    positions_m: FloatArray
    deformation_determinants: FloatArray
    minimum_continuation_deformation_determinant: float
    native_solve_count: int


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


def run_hyperelastic_replay_v2(
    *,
    native: NativeJaxFemModulesV2,
    points_m: npt.ArrayLike,
    cells: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    contact: RigidContactProjectionV1,
    young_modulus_pa: float,
    poisson_ratio: float,
    interval_substeps: int,
    driven: bool,
    newton_absolute_tolerance: float,
    newton_relative_tolerance: float,
    hard_minimum_deformation_determinant: float,
) -> HyperelasticReplayV2:  # pragma: no cover - exercised by frozen native runs
    """Solve one complete source prefix with fixed warm-start continuation."""

    points = np.ascontiguousarray(np.asarray(points_m, dtype=np.float64))
    tetrahedra = np.ascontiguousarray(np.asarray(cells, dtype=np.int32))
    indices = np.ascontiguousarray(np.asarray(attachment_indices, dtype=np.int64))
    _require(points.ndim == 2 and points.shape[1] == 3, "points changed")
    _require(
        tetrahedra.ndim == 2 and tetrahedra.shape[1] == 4,
        "TET4 cells changed",
    )
    _require(indices.ndim == 1 and len(indices) >= 1, "attachments changed")
    _require(
        type(interval_substeps) is int and interval_substeps >= 1, "invalid substeps"
    )
    absolute_tolerance = _finite(
        newton_absolute_tolerance,
        name="newton_absolute_tolerance",
        positive=True,
    )
    relative_tolerance = _finite(
        newton_relative_tolerance,
        name="newton_relative_tolerance",
        positive=True,
    )
    hard_minimum = _finite(
        hard_minimum_deformation_determinant,
        name="hard_minimum_deformation_determinant",
        positive=True,
    )
    frame_count = int(contact.rotations.shape[0])
    _require(frame_count >= 2, "contact trajectory is too short")
    _require(
        contact.translations_m.shape[:2] == contact.rotations.shape[:2],
        "contact transform roster changed",
    )

    jnp = native.jnp
    mesh = native.Mesh(jnp.asarray(points), jnp.asarray(tetrahedra), ele_type="TET4")
    energy = stable_neo_hookean_energy_v2(
        native,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
    )

    def get_tensor_map(problem: Any) -> Any:
        stress = native.jax.grad(energy)

        def first_piola(displacement_gradient: Any) -> Any:
            return stress(displacement_gradient + jnp.eye(problem.dim))

        return first_piola

    problem_type = cast(
        type[Any],
        type(
            "StableNeoHookeanSourceV2",
            (native.Problem,),
            {"get_tensor_map": get_tensor_map},
        ),
    )
    solution = np.zeros_like(points)
    positions: list[FloatArray] = []
    frame_determinants: list[FloatArray] = []
    continuation_minima: list[float] = []
    solve_count = 0

    def solve_transform(rotations: FloatArray, translations: FloatArray) -> None:
        nonlocal solution, solve_count
        locations: list[Any] = []
        components: list[int] = []
        values: list[Any] = []
        for patch_index, local_indices in enumerate(contact.patch_local_indices):
            node_ids = jnp.asarray(indices[local_indices])
            rotation = jnp.asarray(rotations[patch_index])
            translation = jnp.asarray(translations[patch_index])
            for component in range(3):
                locations.append(_location_factory(jnp, node_ids))
                components.append(component)
                values.append(_value_factory(rotation, translation, component))
        problem = problem_type(
            mesh=mesh,
            vec=3,
            dim=3,
            ele_type="TET4",
            dirichlet_bc_info=[locations, components, values],
        )
        native_solution = native.solver(
            problem,
            solver_options={
                "newton": {
                    "tol": absolute_tolerance,
                    "rel_tol": relative_tolerance,
                    "line_search_flag": True,
                    "initial_guess": [jnp.asarray(solution)],
                    "linear": {"spsolve_solver": {}},
                }
            },
        )
        _require(
            isinstance(native_solution, (list, tuple)) and len(native_solution) == 1,
            "JAX-FEM v2 returned an unexpected solution structure",
        )
        candidate = np.ascontiguousarray(
            np.asarray(native_solution[0]),
            dtype=np.float64,
        )
        _require(
            candidate.shape == points.shape and np.all(np.isfinite(candidate)),
            "JAX-FEM v2 displacement state is invalid",
        )
        determinants = deformation_determinants_v1(
            points,
            tetrahedra,
            (points + candidate)[None],
        )[0]
        minimum = float(np.min(determinants))
        _require(
            np.all(np.isfinite(determinants)) and minimum >= hard_minimum,
            "JAX-FEM v2 continuation violated its hard orientation threshold",
        )
        solution = candidate
        solve_count += 1
        continuation_minima.append(minimum)

    identity_rotations = np.repeat(
        np.eye(3, dtype=np.float64)[None],
        len(contact.patch_local_indices),
        axis=0,
    )
    zero_translations: FloatArray = np.zeros(
        (len(contact.patch_local_indices), 3),
        dtype=np.float64,
    )
    for frame_index in range(frame_count):
        if driven:
            previous_index = max(frame_index - 1, 0)
            previous_rotations = np.asarray(
                contact.rotations[previous_index], dtype=np.float64
            )
            previous_translations = np.asarray(
                contact.translations_m[previous_index], dtype=np.float64
            )
            target_rotations = np.asarray(
                contact.rotations[frame_index], dtype=np.float64
            )
            target_translations = np.asarray(
                contact.translations_m[frame_index], dtype=np.float64
            )
        else:
            previous_rotations = target_rotations = identity_rotations
            previous_translations = target_translations = zero_translations
        steps = 1 if frame_index == 0 else interval_substeps
        for substep in range(1, steps + 1):
            fraction = substep / steps
            rotations = np.stack(
                [
                    interpolate_rotation_v2(left, right, fraction)
                    for left, right in zip(
                        previous_rotations,
                        target_rotations,
                        strict=True,
                    )
                ]
            )
            translations = (
                1.0 - fraction
            ) * previous_translations + fraction * target_translations
            solve_transform(
                np.ascontiguousarray(rotations),
                np.ascontiguousarray(translations),
            )
        position = np.ascontiguousarray(points + solution)
        positions.append(position)
        frame_determinants.append(
            deformation_determinants_v1(points, tetrahedra, position[None])[0]
        )

    return HyperelasticReplayV2(
        positions_m=np.ascontiguousarray(np.stack(positions)),
        deformation_determinants=np.ascontiguousarray(np.stack(frame_determinants)),
        minimum_continuation_deformation_determinant=min(continuation_minima),
        native_solve_count=solve_count,
    )


__all__ = [
    "BACKEND_VARIANT",
    "CONSTITUTIVE_MODEL",
    "CONTINUATION_POLICY",
    "HyperelasticReplayV2",
    "NONLINEAR_SOLVER",
    "NativeJaxFemModulesV2",
    "RUNTIME_SCHEMA",
    "interpolate_rotation_v2",
    "lame_parameters_v2",
    "load_native_jax_fem_modules_v2",
    "normalized_objectivity_errors_v2",
    "run_hyperelastic_replay_v2",
    "stable_neo_hookean_energy_v2",
]
