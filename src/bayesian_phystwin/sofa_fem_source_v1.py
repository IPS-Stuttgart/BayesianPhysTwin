"""Exact-runtime SOFA hyperelastic FEM replay for registered source meshes.

This adapter uses SOFA's native stable-Neo-Hookean tetrahedral force field and
its native pairwise projective attachment constraint. The target mechanical
state carries the registered rigid-patch trajectory; no post-step correction
or spring attachment is used.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from .jax_fem_source_qualification_v1 import RigidContactProjectionV1
from .native_tet_fem_source_v1 import (
    attached_targets_from_transform_v1,
    contact_transform_at_fraction_v1,
    prepare_native_tet_source_geometry_v1,
    replay_deformation_determinants_v1,
    tetrahedral_nodal_masses_v1,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]

SOFA_REVISION = "7c18e95d5c5f2839079892c69e7d89a313c79603"
SOFA_VERSION = "26.06.00"
SOFA_REPORTED_VERSION = "v26.06"
SOFA_REPOSITORY = "https://github.com/sofa-framework/sofa"
SOFA_ARCHIVE_FILENAME = "SOFA_v26.06.00_Linux_Python3.10.zip"
SOFA_ARCHIVE_SHA256 = "129211fd01781bdd5ba3f28f1c3617a2f3792a71b62dc609cf866eec4ac745e2"
SOFA_INSTALLED_FILE_SHA256 = {
    "git-info.txt": (
        "fd798c058c651d201ba552f7bfbd77cf2ad63d095dd650c0f39d3b40e0e0ed3b"
    ),
    "lib/libSofa.Component.AnimationLoop.so.26.06.00": (
        "5dc44e8039684d3f112066e056c7f9e993ad10adcc50244569beba27022d2ea9"
    ),
    "lib/libSofa.Component.Constraint.Projective.so.26.06.00": (
        "1b5bc1b6e60dfd74f892a2acda34bba69fb7cd3370f631a040bd368c0ff6f828"
    ),
    "lib/libSofa.Component.LinearSolver.Direct.so.26.06.00": (
        "7a94c419c39d18236c24281c707e8e4db6c3e032c43e942b63514293926663d4"
    ),
    "lib/libSofa.Component.Mass.so.26.06.00": (
        "180e04093c8f8096e3e2c19ae986886bc2af2f23adeefbd11fd8aff47b0210d9"
    ),
    "lib/libSofa.Component.ODESolver.Backward.so.26.06.00": (
        "8bd1d79204e3dab5d2e02b1f7fecf040c99b51c212ff98b54ccad247a2415c12"
    ),
    "lib/libSofa.Component.SolidMechanics.FEM.HyperElastic.so.26.06.00": (
        "aed8df7c26bbf49f0bcf11e890ede9ddbff4afb6e8765ff7ea6a1a96a7bbaae9"
    ),
    "lib/libSofa.Component.StateContainer.so.26.06.00": (
        "aa1e846c2a2978a2d0d6b1ddaafd90b3bff9b186bda3c60d9c6d2cce42e672c5"
    ),
    "lib/libSofa.Component.Topology.Container.Dynamic.so.26.06.00": (
        "1348185d3a522e8f7139229a6d72ce7263baeae4599e5308650ae2cd522708bb"
    ),
    "plugins/SofaPython3/lib/libSofaPython3.so.1.0": (
        "b8b2aaf5e43082a217cd839ccc56b7b19a929d19c80777c33fa2b099696742bf"
    ),
    (
        "plugins/SofaPython3/lib/python3/site-packages/Sofa/"
        "Core.cpython-310-x86_64-linux-gnu.so"
    ): "3f41ddef8cbfa98806473cff828a2e650fc7916aca14f722cec3c0a2c45ced6f",
    (
        "plugins/SofaPython3/lib/python3/site-packages/Sofa/"
        "Simulation.cpython-310-x86_64-linux-gnu.so"
    ): "54483e9f24c0b5adc7be4c522b4e8ed8311e6887b537542f12d503049854dac6",
}
SOFA_REQUIRED_PLUGINS = (
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.HyperElastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
)
RUNTIME_SCHEMA = "bayesian-phystwin.sofa-hyperelastic-source-runtime-v1"
BACKEND_VARIANT = "sofa-stable-neo-hookean-fem-v1"
CONSTITUTIVE_MODEL = "SOFA-native-Smith-2018-stable-Neo-Hookean"
ATTACHMENT_MODEL = "AttachProjectiveConstraint-moving-Dirichlet-v1"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_environment_path(name: str, expected: Path) -> None:
    raw = os.environ.get(name)
    _require(raw is not None, f"{name} must bind the frozen SOFA distribution")
    assert raw is not None
    if name in {"LD_LIBRARY_PATH", "SOFA_PLUGIN_PATH"}:
        paths = {Path(value).resolve() for value in raw.split(os.pathsep) if value}
        _require(
            expected.resolve() in paths,
            f"{name} does not include the frozen SOFA path",
        )
    else:
        _require(
            Path(raw).resolve() == expected.resolve(),
            f"{name} differs from the frozen SOFA root",
        )


@dataclass(frozen=True, slots=True)
class NativeSofaFemModulesV1:
    """Exact SOFA modules admitted by the frozen distribution identity."""

    sofa: Any
    sofa_runtime: Any
    root: Path
    installed_records: Mapping[str, Mapping[str, str]]


def load_native_sofa_fem_modules_v1(
    *,
    distribution_archive: str | Path,
    sofa_root: str | Path,
) -> NativeSofaFemModulesV1:
    """Import SOFA only after archive, ABI, environment, and files match."""

    archive = Path(distribution_archive).absolute()
    root = Path(sofa_root).absolute()
    _require(
        archive.is_file() and not archive.is_symlink(),
        "SOFA distribution archive must be an ordinary file",
    )
    _require(
        archive.name == SOFA_ARCHIVE_FILENAME,
        "SOFA archive filename differs from the frozen runtime",
    )
    _require(
        _sha256_file(archive) == SOFA_ARCHIVE_SHA256,
        "SOFA archive SHA-256 differs from the frozen runtime",
    )
    _require(
        root.is_dir() and not root.is_symlink(),
        "SOFA root must be an ordinary extracted directory",
    )
    _require(
        platform.python_implementation() == "CPython"
        and platform.python_version_tuple()[:2] == ("3", "10"),
        "SOFA source replay requires the frozen CPython 3.10 ABI",
    )
    _required_environment_path("SOFA_ROOT", root)
    _required_environment_path("SOFA_PLUGIN_PATH", root / "plugins")
    _required_environment_path("LD_LIBRARY_PATH", root / "lib")
    _required_environment_path(
        "LD_LIBRARY_PATH",
        root / "plugins" / "SofaPython3" / "lib",
    )

    records: dict[str, Mapping[str, str]] = {}
    for relative_path, expected in SOFA_INSTALLED_FILE_SHA256.items():
        path = root / relative_path
        _require(
            path.is_file() and not path.is_symlink(),
            f"SOFA runtime member is unavailable: {relative_path}",
        )
        observed = _sha256_file(path)
        _require(observed == expected, f"SOFA runtime member changed: {relative_path}")
        records[relative_path] = {"sha256": observed}
    _require(
        SOFA_REVISION in (root / "git-info.txt").read_text(encoding="utf-8"),
        "SOFA git-info does not bind the frozen revision",
    )
    sofa = importlib.import_module("Sofa")
    sofa_runtime = importlib.import_module("SofaRuntime")
    for module in (sofa, sofa_runtime):
        module_path = Path(str(getattr(module, "__file__", ""))).resolve()
        _require(
            root in module_path.parents,
            "SOFA Python module was imported outside the frozen root",
        )
    _require(
        str(sofa.GetVersion()) == SOFA_REPORTED_VERSION,
        "SOFA reported version differs from the frozen runtime",
    )
    for plugin in SOFA_REQUIRED_PLUGINS:
        sofa_runtime.importPlugin(plugin)
    return NativeSofaFemModulesV1(
        sofa=sofa,
        sofa_runtime=sofa_runtime,
        root=root,
        installed_records=records,
    )


def stable_neo_hookean_lame_parameters_v1(
    young_modulus_pa: float,
    poisson_ratio: float,
) -> tuple[float, float]:
    """Return SOFA StableNeoHookean's shear and first Lame parameters."""

    young = _finite(young_modulus_pa, name="young_modulus_pa", positive=True)
    poisson = _finite(poisson_ratio, name="poisson_ratio")
    _require(0.0 < poisson < 0.5, "poisson_ratio must lie in (0,0.5)")
    shear = young / (2.0 * (1.0 + poisson))
    first_lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return shear, first_lame


@dataclass(frozen=True, slots=True)
class SofaFemSourceReplayV1:
    """Native trajectory plus fixed-reference and attachment diagnostics."""

    positions_m: FloatArray
    deformation_determinants: FloatArray
    minimum_continuation_deformation_determinant: float
    maximum_attachment_error_m: float
    native_step_count: int
    scene_sha256: str
    material_vertex_count: int
    tetrahedron_count: int
    attachment_count: int
    total_reference_mass_kg: float


def _scene_identity(
    *,
    points: FloatArray,
    cells: npt.NDArray[np.int64],
    attachment_indices: npt.NDArray[np.int64],
    parameters: Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    for array in (points, cells, attachment_indices):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.tobytes(order="C"))
    for key, value in sorted(parameters.items()):
        digest.update(key.encode())
        digest.update(repr(value).encode())
    return digest.hexdigest()


def run_sofa_fem_source_replay_v1(
    *,
    native: NativeSofaFemModulesV1,
    points_m: npt.ArrayLike,
    cells: npt.ArrayLike,
    attachment_indices: npt.ArrayLike,
    contact: RigidContactProjectionV1,
    driven: bool,
    integrator_time_step_s: float,
    interval_substeps: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    density_kg_m3: float,
    rayleigh_stiffness: float,
    rayleigh_mass: float,
    hard_minimum_deformation_determinant: float,
) -> SofaFemSourceReplayV1:  # pragma: no cover - exact native runtime
    """Run one complete source replay through native SOFA hyperelastic FEM."""

    time_step = _finite(
        integrator_time_step_s,
        name="integrator_time_step_s",
        positive=True,
    )
    _require(
        type(interval_substeps) is int and interval_substeps >= 1,
        "interval_substeps must be a positive integer",
    )
    density = _finite(density_kg_m3, name="density_kg_m3", positive=True)
    stiffness_damping = _finite(rayleigh_stiffness, name="rayleigh_stiffness")
    mass_damping = _finite(rayleigh_mass, name="rayleigh_mass")
    _require(
        stiffness_damping >= 0.0 and mass_damping >= 0.0,
        "Rayleigh damping values must be non-negative",
    )
    determinant_floor = _finite(
        hard_minimum_deformation_determinant,
        name="hard_minimum_deformation_determinant",
        positive=True,
    )
    geometry = prepare_native_tet_source_geometry_v1(
        points_m=points_m,
        cells=cells,
        attachment_indices=attachment_indices,
        contact=contact,
    )
    shear, first_lame = stable_neo_hookean_lame_parameters_v1(
        young_modulus_pa,
        poisson_ratio,
    )
    _, total_mass = tetrahedral_nodal_masses_v1(
        geometry,
        density_kg_m3=density,
    )
    parameters: dict[str, object] = {
        "attachment_model": ATTACHMENT_MODEL,
        "constitutive_model": CONSTITUTIVE_MODEL,
        "density_kg_m3": density,
        "first_lame_pa": first_lame,
        "integrator_time_step_s": time_step,
        "rayleigh_mass": mass_damping,
        "rayleigh_stiffness": stiffness_damping,
        "shear_modulus_pa": shear,
    }
    scene_sha256 = _scene_identity(
        points=geometry.points_m,
        cells=geometry.cells,
        attachment_indices=geometry.attachment_indices,
        parameters=parameters,
    )

    sofa = native.sofa
    root = sofa.Core.Node("root")
    initialized = False
    try:
        root.dt = time_step
        root.gravity = [0.0, 0.0, 0.0]
        root.addObject("DefaultAnimationLoop")
        initial_targets = geometry.points_m[geometry.attachment_indices]
        target_node = root.addChild("target")
        target = target_node.addObject(
            "MechanicalObject",
            name="dofs",
            template="Vec3d",
            position=initial_targets.tolist(),
            velocity=np.zeros_like(initial_targets).tolist(),
        )
        body = root.addChild("body")
        body.addObject(
            "EulerImplicitSolver",
            rayleighStiffness=stiffness_damping,
            rayleighMass=mass_damping,
        )
        linear_solver = body.addObject(
            "SparseLDLSolver",
            template="CompressedRowSparseMatrixMat3x3d",
        )
        topology = body.addObject(
            "TetrahedronSetTopologyContainer",
            name="topology",
            position=geometry.points_m.tolist(),
            tetrahedra=geometry.cells.tolist(),
        )
        mechanical = body.addObject(
            "MechanicalObject",
            name="dofs",
            template="Vec3d",
            position=geometry.points_m.tolist(),
        )
        mass = body.addObject(
            "MeshMatrixMass",
            name="mass",
            massDensity=density,
        )
        force_field = body.addObject(
            "TetrahedronHyperelasticityFEMForceField",
            name="stable_neo_hookean_fem",
            materialName="StableNeoHookean",
            ParameterSet=f"{shear:.17g} {first_lame:.17g}",
        )
        attachment = root.addObject(
            "AttachProjectiveConstraint",
            name="moving_dirichlet_attachment",
            object1="@target/dofs",
            object2="@body/dofs",
            indices1=list(range(len(geometry.attachment_indices))),
            indices2=geometry.attachment_indices.tolist(),
            twoWay=False,
            positionFactor=2.0,
            velocityFactor=2.0,
            responseFactor=1.0,
            constraintFactor=[1.0] * len(geometry.attachment_indices),
        )
        sofa.Simulation.init(root)
        initialized = True
        for component in (
            linear_solver,
            mass,
            force_field,
            attachment,
        ):
            state = str(component.componentState.value)
            _require(
                state == "Valid",
                f"SOFA source component {component.getName()} initialized {state}",
            )
        _require(
            len(np.asarray(topology.tetrahedra.value)) == len(geometry.cells)
            and np.asarray(mechanical.position.value).shape == geometry.points_m.shape
            and np.asarray(target.position.value).shape == initial_targets.shape,
            "SOFA source topology or mechanical-state roster changed",
        )

        frame_count = int(contact.rotations.shape[0])
        positions: FloatArray = np.empty(
            (frame_count, len(geometry.points_m), 3),
            dtype=np.float64,
        )
        positions[0] = np.asarray(mechanical.position.value, dtype=np.float64)
        _require(
            np.allclose(positions[0], geometry.points_m, atol=1e-12, rtol=0.0),
            "SOFA initial material state changed",
        )
        minimum_determinant = 1.0
        maximum_attachment_error = 0.0
        native_step_count = 0
        previous_targets = initial_targets.copy()
        for frame in range(1, frame_count):
            for substep in range(1, interval_substeps + 1):
                rotations, translations = contact_transform_at_fraction_v1(
                    contact,
                    previous_frame=frame - 1,
                    target_frame=frame,
                    fraction=substep / interval_substeps,
                    driven=driven,
                )
                targets = attached_targets_from_transform_v1(
                    geometry,
                    contact,
                    rotations=rotations,
                    translations_m=translations,
                )
                target.position.value = targets.tolist()
                target.velocity.value = (
                    (targets - previous_targets) / time_step
                ).tolist()
                previous_targets = targets
                previous_time = float(root.time.value)
                sofa.Simulation.animate(root, time_step)
                native_step_count += 1
                current = np.asarray(mechanical.position.value, dtype=np.float64).copy()
                _require(
                    float(root.time.value) > previous_time,
                    "SOFA failed to advance source-scene time",
                )
                _require(
                    np.all(np.isfinite(current))
                    and np.all(
                        np.isfinite(
                            np.asarray(mechanical.velocity.value, dtype=np.float64)
                        )
                    ),
                    "SOFA source replay produced non-finite state",
                )
                determinants = replay_deformation_determinants_v1(
                    geometry,
                    current[None],
                )[0]
                step_minimum = float(np.min(determinants))
                minimum_determinant = min(minimum_determinant, step_minimum)
                _require(
                    step_minimum >= determinant_floor,
                    "SOFA source replay violated its hard orientation threshold "
                    f"at frame={frame}, substep={substep}, "
                    f"minimum_determinant={step_minimum:.17g}, "
                    f"required_minimum={determinant_floor:.17g}",
                )
                attachment_error = float(
                    np.max(
                        np.linalg.norm(
                            current[geometry.attachment_indices] - targets,
                            axis=1,
                        )
                    )
                )
                maximum_attachment_error = max(
                    maximum_attachment_error,
                    attachment_error,
                )
            positions[frame] = current

        deformation_determinants = replay_deformation_determinants_v1(
            geometry,
            positions,
        )
        return SofaFemSourceReplayV1(
            positions_m=np.ascontiguousarray(positions),
            deformation_determinants=np.ascontiguousarray(deformation_determinants),
            minimum_continuation_deformation_determinant=minimum_determinant,
            maximum_attachment_error_m=maximum_attachment_error,
            native_step_count=native_step_count,
            scene_sha256=scene_sha256,
            material_vertex_count=len(geometry.points_m),
            tetrahedron_count=len(geometry.cells),
            attachment_count=len(geometry.attachment_indices),
            total_reference_mass_kg=total_mass,
        )
    finally:
        if initialized:
            sofa.Simulation.unload(root)


__all__ = [
    "ATTACHMENT_MODEL",
    "BACKEND_VARIANT",
    "CONSTITUTIVE_MODEL",
    "NativeSofaFemModulesV1",
    "SOFA_ARCHIVE_FILENAME",
    "SOFA_ARCHIVE_SHA256",
    "SOFA_INSTALLED_FILE_SHA256",
    "SOFA_REQUIRED_PLUGINS",
    "SOFA_REVISION",
    "SOFA_VERSION",
    "SofaFemSourceReplayV1",
    "load_native_sofa_fem_modules_v1",
    "run_sofa_fem_source_replay_v1",
    "stable_neo_hookean_lame_parameters_v1",
]
