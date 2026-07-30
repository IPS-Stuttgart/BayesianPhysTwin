"""Target-free Codim-IPC cloth-backbone qualification helpers."""

from __future__ import annotations

import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rgbench_libuipc import FlingPinController


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class CodimIPCClothParameters:
    """Metric material and numerical settings for a Codim-IPC rollout."""

    timestep_s: float
    youngs_modulus_pa: float
    poisson_ratio: float
    volume_density_kg_m3: float
    thickness_m: float
    bending_stiffness_multiplier: float
    newton_tolerance: float
    contact_thickness_m: float
    collision_enabled: bool

    def __post_init__(self) -> None:
        positive = {
            "timestep_s": self.timestep_s,
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "volume_density_kg_m3": self.volume_density_kg_m3,
            "thickness_m": self.thickness_m,
            "bending_stiffness_multiplier": self.bending_stiffness_multiplier,
            "newton_tolerance": self.newton_tolerance,
            "contact_thickness_m": self.contact_thickness_m,
        }
        for name, value in positive.items():
            _require(math.isfinite(value) and value > 0.0, f"{name} must be positive")
        _require(
            math.isfinite(self.poisson_ratio) and -1.0 < self.poisson_ratio < 0.5,
            "poisson_ratio must be in (-1, 0.5)",
        )
        _require(
            isinstance(self.collision_enabled, bool),
            "collision_enabled must be boolean",
        )


@dataclass(frozen=True)
class CodimIPCRollout:
    """Numerical output of one isolated Codim-IPC replay."""

    final_vertices_m: np.ndarray
    maximum_pin_target_error_m: float
    step_count: int
    total_newton_iterations: int

    def __post_init__(self) -> None:
        vertices = np.asarray(self.final_vertices_m, dtype=np.float64)
        _require(
            vertices.ndim == 2 and vertices.shape[1] == 3,
            "final_vertices_m must have shape (N, 3)",
        )
        _require(np.all(np.isfinite(vertices)), "final vertices must be finite")
        _require(
            math.isfinite(self.maximum_pin_target_error_m)
            and self.maximum_pin_target_error_m >= 0.0,
            "pin target error is invalid",
        )
        _require(self.step_count >= 1, "step_count must be positive")
        _require(
            self.total_newton_iterations >= 0,
            "total_newton_iterations must be nonnegative",
        )
        vertices = np.array(vertices, dtype=np.float64, copy=True)
        vertices.setflags(write=False)
        object.__setattr__(self, "final_vertices_m", vertices)


def write_obj_triangles(
    path: str | Path,
    vertices_m: np.ndarray,
    triangles: np.ndarray,
) -> None:
    """Write one metric triangle mesh without changing its vertex contract."""

    destination = Path(path)
    vertices = np.asarray(vertices_m, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3, "invalid vertices")
    _require(faces.ndim == 2 and faces.shape[1] == 3, "invalid triangles")
    _require(len(vertices) >= 3 and np.all(np.isfinite(vertices)), "invalid vertices")
    _require(
        len(faces) >= 1
        and np.min(faces) >= 0
        and np.max(faces) < len(vertices),
        "triangle index is out of bounds",
    )
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        *(f"v {x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in vertices),
        *(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces),
    ]
    destination.write_text("".join(lines), encoding="ascii")


def _load_runtime(
    *,
    module_root: Path,
    python_root: Path,
    expected_linear_solver_backend: str | None,
) -> tuple[Any, Any]:
    for root in (module_root, python_root):
        _require(root.is_dir(), f"Codim-IPC runtime path does not exist: {root}")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    try:
        jgsl = importlib.import_module("JGSL")
        drivers = importlib.import_module("Drivers")
    except (ImportError, OSError) as error:
        raise RuntimeError("Codim-IPC runtime import failed") from error
    required = (
        "Init_Dirichlet_Nodes",
        "Set_Dirichlet_Targets",
        "Node_Positions",
        "Dirichlet_Max_Error",
    )
    missing = [name for name in required if not hasattr(jgsl.FEM, name)]
    if missing:
        raise RuntimeError(f"Codim-IPC RGBench patch is missing: {missing}")
    if expected_linear_solver_backend is not None:
        actual_backend = getattr(jgsl, "linear_solver_backend", None)
        if actual_backend != expected_linear_solver_backend:
            raise RuntimeError(
                "Codim-IPC linear solver changed: "
                f"expected {expected_linear_solver_backend}, got {actual_backend}"
            )
    return jgsl, drivers


def _vector3(jgsl: Any, values: np.ndarray) -> Any:
    return jgsl.Vector3d(float(values[0]), float(values[1]), float(values[2]))


def _target_vector(jgsl: Any, targets_m: np.ndarray) -> Any:
    targets = np.asarray(targets_m, dtype=np.float64)
    _require(targets.ndim == 2 and targets.shape[1] == 3, "invalid pin targets")
    _require(np.all(np.isfinite(targets)), "pin targets must be finite")
    return jgsl.StdVectorVector3d([_vector3(jgsl, row) for row in targets])


def _node_array(node_vectors: Any) -> np.ndarray:
    return np.asarray(
        [[vector[0], vector[1], vector[2]] for vector in node_vectors],
        dtype=np.float64,
    )


def run_codim_ipc_fling(
    *,
    vertices_m: np.ndarray,
    triangles: np.ndarray,
    controller: FlingPinController,
    parameters: CodimIPCClothParameters,
    duration_s: float,
    workspace: str | Path,
    module_root: str | Path,
    python_root: str | Path,
    expected_linear_solver_backend: str | None = None,
) -> CodimIPCRollout:
    """Run a single-thread-compatible Codim-IPC replay with exact moving pins."""

    vertices = np.asarray(vertices_m, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3, "invalid vertices")
    _require(faces.ndim == 2 and faces.shape[1] == 3, "invalid triangles")
    _require(
        len(vertices) >= 128 and np.all(np.isfinite(vertices)),
        "mesh is not backend-admissible",
    )
    _require(
        len(faces) >= 1
        and np.min(faces) >= 0
        and np.max(faces) < len(vertices),
        "triangle index is out of bounds",
    )
    _require(max(controller.pin_indices) < len(vertices), "pin index is out of bounds")
    _require(math.isfinite(duration_s) and duration_s > 0.0, "duration is invalid")
    step_ratio = duration_s / parameters.timestep_s
    step_count = int(round(step_ratio))
    _require(
        step_count >= 1 and math.isclose(step_ratio, step_count, abs_tol=1e-12),
        "duration must be an integer multiple of timestep_s",
    )
    destination = Path(workspace).resolve()
    _require(not destination.exists(), f"workspace already exists: {destination}")
    destination.mkdir(parents=True)
    mesh_path = destination / "metric_cloth.obj"
    write_obj_triangles(mesh_path, vertices, faces)

    old_cwd = Path.cwd()
    old_stdout = sys.stdout
    old_argv = sys.argv
    try:
        os.chdir(destination)
        sys.argv = ["codim_ipc_rgbbench"]
        jgsl, drivers = _load_runtime(
            module_root=Path(module_root).resolve(),
            python_root=Path(python_root).resolve(),
            expected_linear_solver_backend=expected_linear_solver_backend,
        )
        simulation = drivers.FEMDiscreteShellBase("double", 3)
        simulation.output_folder = str(destination / "simulator_output") + "/"
        Path(simulation.output_folder).mkdir(parents=True, exist_ok=True)
        simulation.add_shell_3D(
            str(mesh_path),
            jgsl.Vector3d(0.0, 0.0, 0.0),
            jgsl.Vector3d(0.0, 0.0, 0.0),
            jgsl.Vector3d(0.0, 1.0, 0.0),
            0.0,
        )
        pin_ids = jgsl.StdVectorXi([int(index) for index in controller.pin_indices])
        jgsl.FEM.Init_Dirichlet_Nodes(simulation.X, pin_ids, simulation.DBC)
        simulation.gravity = jgsl.Vector3d(0.0, 0.0, -9.81)
        simulation.dt = float(parameters.timestep_s)
        simulation.frame_dt = float(parameters.timestep_s)
        simulation.withCollision = bool(parameters.collision_enabled)
        simulation.PNTol = float(parameters.newton_tolerance)
        simulation.initialize(
            float(parameters.volume_density_kg_m3),
            float(parameters.youngs_modulus_pa),
            float(parameters.poisson_ratio),
            float(parameters.thickness_m),
            0,
        )
        simulation.bendingStiffMult = float(
            parameters.bending_stiffness_multiplier
        )
        simulation.initialize_OIPC(
            float(parameters.contact_thickness_m),
            0.0,
        )
        maximum_pin_error = 0.0
        for step_index in range(step_count):
            time_s = float((step_index + 1) * parameters.timestep_s)
            targets = _target_vector(jgsl, controller.targets_at(time_s))
            jgsl.FEM.Set_Dirichlet_Targets(targets, simulation.DBC)
            simulation.advance_one_time_step(float(parameters.timestep_s))
            maximum_pin_error = max(
                maximum_pin_error,
                float(jgsl.FEM.Dirichlet_Max_Error(simulation.X, simulation.DBC)),
            )
        final = _node_array(jgsl.FEM.Node_Positions(simulation.X))
        total_iterations = int(simulation.PNIterCount)
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        os.chdir(old_cwd)
    _require(final.shape == vertices.shape, "Codim-IPC changed the vertex contract")
    return CodimIPCRollout(
        final_vertices_m=final,
        maximum_pin_target_error_m=maximum_pin_error,
        step_count=step_count,
        total_newton_iterations=total_iterations,
    )
