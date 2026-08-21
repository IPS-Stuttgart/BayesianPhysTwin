#!/usr/bin/env python3
"""Run the pinned synthetic native SOFA Neo-Hookean FEM smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    SofaMechanicalObjectReplayV1,
)
from bayesian_phystwin.material_trajectory_producer_v1 import (
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive

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
    "lib/libSofa.Component.MechanicalLoad.so.26.06.00": (
        "1865a71a187b6783e3171461f788095f451cec0b75acc8a4c2b3b6581e510bf2"
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
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.HyperElastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
)
SMOKE_SCHEMA = "bayesian-phystwin.sofa-fem-native-smoke-v1"
PORTABLE_MEMBERS = (
    "SHA256SUMS",
    "material-trajectory-backend.json",
    "physical-prediction.npz",
    "provenance/material-trajectory-rollout.npz",
    "provenance/material-runtime.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _required_environment_path(name: str, expected: Path) -> None:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} must bind the frozen SOFA distribution")
    if name in {"LD_LIBRARY_PATH", "SOFA_PLUGIN_PATH"}:
        paths = {Path(value).resolve() for value in raw.split(os.pathsep) if value}
        if expected.resolve() not in paths:
            raise RuntimeError(f"{name} does not include the frozen SOFA path")
    elif Path(raw).resolve() != expected.resolve():
        raise RuntimeError(f"{name} differs from the frozen SOFA root")


@dataclass(frozen=True, slots=True)
class _NativeModules:
    sofa: Any
    sofa_runtime: Any
    root: Path
    installed_records: Mapping[str, Mapping[str, str]]


def _load_native_modules(archive_path: Path, sofa_root: Path) -> _NativeModules:
    archive = archive_path.absolute()
    root = sofa_root.absolute()
    if not archive.is_file() or archive.is_symlink():
        raise RuntimeError("SOFA distribution archive must be an ordinary file")
    if archive.name != SOFA_ARCHIVE_FILENAME:
        raise RuntimeError("SOFA archive filename differs from the frozen runtime")
    if _sha256_file(archive) != SOFA_ARCHIVE_SHA256:
        raise RuntimeError("SOFA archive SHA-256 differs from the frozen runtime")
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("SOFA root must be an ordinary extracted directory")
    if platform.python_implementation() != "CPython" or platform.python_version_tuple()[
        :2
    ] != ("3", "10"):
        raise RuntimeError("SOFA smoke requires the frozen CPython 3.10 ABI")
    _required_environment_path("SOFA_ROOT", root)
    _required_environment_path("SOFA_PLUGIN_PATH", root / "plugins")
    _required_environment_path("LD_LIBRARY_PATH", root / "lib")
    _required_environment_path(
        "LD_LIBRARY_PATH", root / "plugins" / "SofaPython3" / "lib"
    )

    records: dict[str, Mapping[str, str]] = {}
    for relative_path, expected_sha256 in SOFA_INSTALLED_FILE_SHA256.items():
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"SOFA runtime member is unavailable: {relative_path}")
        observed = _sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"SOFA runtime member changed: {relative_path}; "
                f"expected {expected_sha256}, found {observed}"
            )
        records[relative_path] = {"sha256": observed}
    git_info = (root / "git-info.txt").read_text(encoding="utf-8")
    if SOFA_REVISION not in git_info:
        raise RuntimeError("SOFA git-info does not bind the frozen revision")

    sofa = importlib.import_module("Sofa")
    sofa_runtime = importlib.import_module("SofaRuntime")
    for module in (sofa, sofa_runtime):
        module_path = Path(str(getattr(module, "__file__", ""))).resolve()
        if root not in module_path.parents:
            raise RuntimeError(
                "SOFA Python module was imported outside the frozen root"
            )
    if str(sofa.GetVersion()) != SOFA_REPORTED_VERSION:
        raise RuntimeError("SOFA reported version differs from the frozen runtime")
    for plugin in SOFA_REQUIRED_PLUGINS:
        sofa_runtime.importPlugin(plugin)
    return _NativeModules(
        sofa=sofa,
        sofa_runtime=sofa_runtime,
        root=root,
        installed_records=records,
    )


def _mesh() -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [
            [i * 0.1, j * 0.05, k * 0.05]
            for i in range(3)
            for j in range(3)
            for k in range(3)
        ],
        dtype=np.float64,
    )

    def node(i: int, j: int, k: int) -> int:
        return i * 9 + j * 3 + k

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
    cells = np.asarray(tetrahedra, dtype=np.int64)
    determinants = np.array(
        [np.linalg.det((points[cell[1:]] - points[cell[0]]).T) for cell in cells]
    )
    if len(points) != 27 or len(cells) != 48 or not np.all(determinants > 0.0):
        raise RuntimeError("SOFA synthetic tetrahedral mesh construction changed")
    return points, cells


def _lame_parameters(
    young_modulus_pa: float, poisson_ratio: float
) -> tuple[float, float]:
    shear_modulus = young_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    first_lame = (
        young_modulus_pa
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    return shear_modulus, first_lame


@dataclass(slots=True)
class _ForceControlledReplay:
    replay: SofaMechanicalObjectReplayV1
    force_field: Any
    root_node: Any
    tetrahedra: np.ndarray
    reference_points_m: np.ndarray
    force_n: float = 0.0

    @property
    def context(self) -> object:
        return self

    def synchronize(self) -> object:
        return self.replay.synchronize()

    def get_material_positions_m(self) -> object:
        return self.replay.get_material_positions_m()

    def step(self) -> object:
        self.force_field.totalForce.value = [self.force_n, 0.0, 0.0]
        return self.replay.step()


def _build_replay(
    native: _NativeModules,
    *,
    output_time_step_s: float,
    integrator_time_step_s: float,
    integrator_substeps: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    total_mass_kg: float,
) -> _ForceControlledReplay:
    sofa = native.sofa
    points, tetrahedra = _mesh()
    left = np.flatnonzero(np.isclose(points[:, 0], 0.0, atol=1e-12))
    right = np.flatnonzero(np.isclose(points[:, 0], 0.2, atol=1e-12))
    if len(left) != 9 or len(right) != 9:
        raise RuntimeError("SOFA fixed/driven face identity changed")
    shear_modulus, first_lame = _lame_parameters(young_modulus_pa, poisson_ratio)

    root = sofa.Core.Node("root")
    root.dt = integrator_time_step_s
    root.gravity = [0.0, 0.0, 0.0]
    root.addObject("DefaultAnimationLoop")
    body = root.addChild("body")
    body.addObject(
        "EulerImplicitSolver",
        rayleighStiffness=0.1,
        rayleighMass=0.1,
    )
    body.addObject(
        "SparseLDLSolver",
        template="CompressedRowSparseMatrixMat3x3d",
    )
    body.addObject(
        "TetrahedronSetTopologyContainer",
        name="topology",
        position=points.tolist(),
        tetrahedra=tetrahedra.tolist(),
    )
    mechanical = body.addObject(
        "MechanicalObject",
        name="dofs",
        template="Vec3d",
        position=points.tolist(),
    )
    body.addObject("UniformMass", totalMass=total_mass_kg)
    fem = body.addObject(
        "TetrahedronHyperelasticityFEMForceField",
        name="neo_hookean_fem",
        materialName="NeoHookean",
        ParameterSet=f"{shear_modulus:.17g} {first_lame:.17g}",
    )
    body.addObject("FixedProjectiveConstraint", indices=left.tolist())
    force = body.addObject(
        "ConstantForceField",
        name="right_face_force",
        indices=right.tolist(),
        totalForce=[0.0, 0.0, 0.0],
    )
    sofa.Simulation.init(root)
    if (
        str(fem.componentState.value) != "Valid"
        or str(force.componentState.value) != "Valid"
    ):
        sofa.Simulation.unload(root)
        raise RuntimeError("SOFA native FEM scene did not initialize valid components")

    def animate(observed_root: object, observed_dt: float) -> None:
        if observed_root is not root or not np.isclose(
            observed_dt,
            output_time_step_s,
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError("SOFA replay root or output time step changed")
        for _ in range(integrator_substeps):
            sofa.Simulation.animate(root, integrator_time_step_s)
        state = np.asarray(mechanical.position.value)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("SOFA Neo-Hookean FEM produced non-finite state")

    return _ForceControlledReplay(
        replay=SofaMechanicalObjectReplayV1(
            mechanical_object=mechanical,
            root_node=root,
            animate_callback=animate,
            time_step_s=output_time_step_s,
        ),
        force_field=force,
        root_node=root,
        tetrahedra=tetrahedra,
        reference_points_m=points,
    )


def _deformation_determinants(
    reference: np.ndarray,
    positions: np.ndarray,
    tetrahedra: np.ndarray,
) -> np.ndarray:
    values = []
    for cell in tetrahedra:
        reference_gradient = (reference[cell[1:]] - reference[cell[0]]).T
        current_gradient = (positions[cell[1:]] - positions[cell[0]]).T
        values.append(
            np.linalg.det(current_gradient) / np.linalg.det(reference_gradient)
        )
    return np.asarray(values, dtype=np.float64)


def _portable_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for member in PORTABLE_MEMBERS:
        path = directory / member
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing portable member: {member}")
        hashes[member] = _sha256_file(path)
    return hashes


def _topology_descriptor() -> dict[str, object]:
    return {
        "grid_count": [3, 3, 3],
        "grid_spacing_m": [0.1, 0.05, 0.05],
        "node_count": 27,
        "tetrahedron_count": 48,
        "fixed_face": "minimum-x-nine-nodes",
        "driven_face": "maximum-x-nine-nodes",
        "tetrahedralization": "six-positive-tetrahedra-per-hexahedral-cell-v1",
    }


def _run_once(
    native: _NativeModules,
    output_dir: Path,
    *,
    frame_count: int,
    output_time_step_s: float,
    integrator_time_step_s: float,
    integrator_substeps: int,
    total_force_n: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    total_mass_kg: float,
    script_path: Path,
    adapter_path: Path,
    producer_revision: str,
) -> Mapping[str, Any]:
    replays: list[_ForceControlledReplay] = []

    def replay_factory() -> _ForceControlledReplay:
        replay = _build_replay(
            native,
            output_time_step_s=output_time_step_s,
            integrator_time_step_s=integrator_time_step_s,
            integrator_substeps=integrator_substeps,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
            total_mass_kg=total_mass_kg,
        )
        replays.append(replay)
        return replay

    def driven_control(_: int, replay: Any) -> None:
        replay.force_n = total_force_n

    def zero_control(_: int, replay: Any) -> None:
        replay.force_n = 0.0

    topology = _topology_descriptor()
    source_artifacts = {
        "scripts/remote/run_sofa_fem_native_smoke.py": _sha256_file(script_path),
        "src/bayesian_phystwin/material_trajectory_engine_replays_v1.py": (
            _sha256_file(adapter_path)
        ),
        **{
            f"native/{path}": record["sha256"]
            for path, record in native.installed_records.items()
        },
    }
    try:
        artifact = produce_material_trajectory_backend(
            output_dir=output_dir,
            backend_kind="sofa-fem-v1",
            replay_factory=replay_factory,
            driven_control=driven_control,
            zero_action_control=zero_control,
            frame_count=frame_count,
            material_query_indices=np.arange(27, dtype=np.int64),
            action_support=np.repeat(np.array([0.0, 0.5, 1.0], dtype=np.float64), 9),
            engine_revision=SOFA_REVISION,
            engine_version=SOFA_VERSION,
            producer_repository="IPS-Stuttgart/BayesianPhysTwin",
            producer_revision=producer_revision,
            producer_version="sofa-fem-native-smoke-v1",
            producer_artifacts=source_artifacts,
            topology_sha256=_content_id(topology),
            device="cpu",
            device_name=platform.processor() or platform.machine(),
            time_step_s=output_time_step_s,
            scene_id="synthetic-pinned-neo-hookean-tetrahedral-block-v1",
            model_kind="three-dimensional-tetrahedral-fem",
            constitutive_model="SOFA NeoHookean hyperelasticity",
            integrator="EulerImplicitSolver",
            solver="SparseLDLSolver CompressedRowSparseMatrixMat3x3d",
            substeps=integrator_substeps,
            engine_parameters={
                **topology,
                "integrator_time_step_s": integrator_time_step_s,
                "output_time_step_s": output_time_step_s,
                "gravity_m_s2": [0.0, 0.0, 0.0],
                "right_face_total_force_n": total_force_n,
                "young_modulus_pa": young_modulus_pa,
                "poisson_ratio": poisson_ratio,
                "total_mass_kg": total_mass_kg,
                "material_name": "NeoHookean",
                "rayleigh_stiffness": 0.1,
                "rayleigh_mass": 0.1,
            },
        )
        arrays = load_physical_rollout_archive(
            output_dir / "physical-prediction.npz",
            expected_frame_count=frame_count,
        )
        zero_delta = (
            arrays["zero_action_readout_m"] - arrays["frame_zero_points_m"][None]
        )
        response = arrays["driven_readout_m"] - arrays["zero_action_readout_m"]
        _, cells = _mesh()
        determinants = _deformation_determinants(
            np.asarray(arrays["frame_zero_points_m"], dtype=np.float64),
            np.asarray(arrays["driven_readout_m"][-1], dtype=np.float64),
            cells,
        )
        return {
            "artifact_id": artifact["artifact_id"],
            "runtime_id": artifact["runtime_id"],
            "maximum_zero_action_drift_m": float(
                np.max(np.linalg.norm(zero_delta, axis=-1))
            ),
            "maximum_driven_minus_zero_response_m": float(
                np.max(np.linalg.norm(response, axis=-1))
            ),
            "minimum_final_deformation_determinant": float(np.min(determinants)),
            "maximum_final_deformation_determinant": float(np.max(determinants)),
            "portable_sha256": _portable_hashes(output_dir),
        }
    finally:
        for replay in replays:
            native.sofa.Simulation.unload(replay.root_node)


def _stiffness_probe(
    native: _NativeModules,
    *,
    frame_count: int,
    output_time_step_s: float,
    integrator_time_step_s: float,
    integrator_substeps: int,
    total_force_n: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    total_mass_kg: float,
) -> float:
    replay = _build_replay(
        native,
        output_time_step_s=output_time_step_s,
        integrator_time_step_s=integrator_time_step_s,
        integrator_substeps=integrator_substeps,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
        total_mass_kg=total_mass_kg,
    )
    try:
        initial = np.asarray(replay.get_material_positions_m()).copy()
        replay.force_n = total_force_n
        for _ in range(frame_count - 1):
            replay.step()
        response = np.asarray(replay.get_material_positions_m()) - initial
        return float(np.max(response[:, 0]))
    finally:
        native.sofa.Simulation.unload(replay.root_node)


def run_smoke(
    output_dir: str | Path,
    *,
    distribution_archive: str | Path,
    sofa_root: str | Path,
    frame_count: int = 5,
    output_time_step_s: float = 0.025,
    integrator_time_step_s: float = 0.001,
    total_force_n: float = 1.0,
    young_modulus_pa: float = 1000.0,
    poisson_ratio: float = 0.3,
    total_mass_kg: float = 1.0,
) -> Mapping[str, Any]:
    if type(frame_count) is not int or frame_count < 2:
        raise ValueError("frame_count must be an integer >= 2")
    for name, value in (
        ("output_time_step_s", output_time_step_s),
        ("integrator_time_step_s", integrator_time_step_s),
        ("total_force_n", total_force_n),
        ("young_modulus_pa", young_modulus_pa),
        ("total_mass_kg", total_mass_kg),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1,0.5)")
    ratio = output_time_step_s / integrator_time_step_s
    integrator_substeps = int(round(ratio))
    if integrator_substeps < 1 or not np.isclose(
        integrator_substeps * integrator_time_step_s,
        output_time_step_s,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("output time step must be an exact integrator-step multiple")

    root = Path(output_dir).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    native = _load_native_modules(
        Path(distribution_archive),
        Path(sofa_root),
    )
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]
    adapter_path = (
        repository_root
        / "src"
        / "bayesian_phystwin"
        / "material_trajectory_engine_replays_v1.py"
    )
    if not adapter_path.is_file() or adapter_path.is_symlink():
        raise RuntimeError("SOFA replay adapter source is unavailable")
    producer_revision = _repository_revision(script_path)

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
                output_time_step_s=output_time_step_s,
                integrator_time_step_s=integrator_time_step_s,
                integrator_substeps=integrator_substeps,
                total_force_n=total_force_n,
                young_modulus_pa=young_modulus_pa,
                poisson_ratio=poisson_ratio,
                total_mass_kg=total_mass_kg,
                script_path=script_path,
                adapter_path=adapter_path,
                producer_revision=producer_revision,
            )

        run_a = run_once("run-a")
        run_b = run_once("run-b")
        deterministic = run_a["portable_sha256"] == run_b["portable_sha256"]
        if not deterministic:
            raise RuntimeError("native SOFA FEM replay is not byte-deterministic")

        response = float(run_a["maximum_driven_minus_zero_response_m"])
        zero_drift = float(run_a["maximum_zero_action_drift_m"])
        minimum_determinant = float(run_a["minimum_final_deformation_determinant"])
        maximum_determinant = float(run_a["maximum_final_deformation_determinant"])
        minimum_response = 1e-4
        if response <= minimum_response:
            raise RuntimeError("SOFA FEM driven arm did not produce a response")
        if zero_drift > 1e-12:
            raise RuntimeError("SOFA FEM zero-action drift exceeds the smoke bound")
        if minimum_determinant <= 0.5 or maximum_determinant >= 2.0:
            raise RuntimeError("SOFA FEM synthetic scene violated deformation bounds")
        low_young = young_modulus_pa / 2.0
        high_young = young_modulus_pa * 2.0
        low_response = _stiffness_probe(
            native,
            frame_count=frame_count,
            output_time_step_s=output_time_step_s,
            integrator_time_step_s=integrator_time_step_s,
            integrator_substeps=integrator_substeps,
            total_force_n=total_force_n,
            young_modulus_pa=low_young,
            poisson_ratio=poisson_ratio,
            total_mass_kg=total_mass_kg,
        )
        high_response = _stiffness_probe(
            native,
            frame_count=frame_count,
            output_time_step_s=output_time_step_s,
            integrator_time_step_s=integrator_time_step_s,
            integrator_substeps=integrator_substeps,
            total_force_n=total_force_n,
            young_modulus_pa=high_young,
            poisson_ratio=poisson_ratio,
            total_mass_kg=total_mass_kg,
        )
        if not low_response > high_response * 1.25:
            raise RuntimeError("SOFA FEM response lacks Young-modulus sensitivity")

        descriptor: dict[str, Any] = {
            "schema": SMOKE_SCHEMA,
            "claim_boundary": (
                "Synthetic native-execution, tetrahedral Neo-Hookean FEM, "
                "constitutive-sensitivity, and provenance smoke only; no source-value, "
                "fresh-object, calibration, or Causal4D benefit claim."
            ),
            "backend_profile": "sofa-fem-v1",
            "engine": {
                "repository": SOFA_REPOSITORY,
                "revision": SOFA_REVISION,
                "version": SOFA_VERSION,
                "archive_filename": SOFA_ARCHIVE_FILENAME,
                "archive_sha256": SOFA_ARCHIVE_SHA256,
                "installed_records": native.installed_records,
                "required_plugins": list(SOFA_REQUIRED_PLUGINS),
            },
            "producer_revision": producer_revision,
            "runtime": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy_version": np.__version__,
                "device": "cpu",
                "sofa_root": str(native.root),
            },
            "problem": {
                **_topology_descriptor(),
                "frame_count": frame_count,
                "output_time_step_s": output_time_step_s,
                "integrator_time_step_s": integrator_time_step_s,
                "integrator_substeps": integrator_substeps,
                "right_face_total_force_n": total_force_n,
                "young_modulus_pa": young_modulus_pa,
                "poisson_ratio": poisson_ratio,
                "total_mass_kg": total_mass_kg,
                "gravity_m_s2": [0.0, 0.0, 0.0],
                "constitutive_model": "NeoHookean",
            },
            "checks": {
                "archive_matches_pinned_sha256": True,
                "installed_runtime_matches_pinned_sha256": True,
                "portable_replay_byte_deterministic": deterministic,
                "maximum_zero_action_drift_m": zero_drift,
                "maximum_driven_minus_zero_response_m": response,
                "minimum_required_response_m": minimum_response,
                "minimum_final_deformation_determinant": minimum_determinant,
                "maximum_final_deformation_determinant": maximum_determinant,
                "low_young_modulus_pa": low_young,
                "low_young_response_m": low_response,
                "high_young_modulus_pa": high_young,
                "high_young_response_m": high_response,
                "minimum_low_over_high_response_ratio": 1.25,
                "observed_low_over_high_response_ratio": low_response / high_response,
            },
            "run_a": run_a,
            "run_b": run_b,
            "future_outcomes_read": False,
            "dataset_payload_read": False,
        }
        descriptor["smoke_id"] = _content_id(descriptor)
        result_path = staging / "sofa-fem-native-smoke.json"
        result_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, root)
    return descriptor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--distribution-archive", required=True)
    parser.add_argument("--sofa-root", required=True)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--output-time-step-s", type=float, default=0.025)
    parser.add_argument("--integrator-time-step-s", type=float, default=0.001)
    parser.add_argument("--total-force-n", type=float, default=1.0)
    parser.add_argument("--young-modulus-pa", type=float, default=1000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--total-mass-kg", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_smoke(
        args.output_dir,
        distribution_archive=args.distribution_archive,
        sofa_root=args.sofa_root,
        frame_count=args.frame_count,
        output_time_step_s=args.output_time_step_s,
        integrator_time_step_s=args.integrator_time_step_s,
        total_force_n=args.total_force_n,
        young_modulus_pa=args.young_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        total_mass_kg=args.total_mass_kg,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
