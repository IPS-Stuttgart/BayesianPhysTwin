"""Target-free ARCSim cloth-backbone qualification helpers for RGBench."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rgbench_libuipc import FlingPinController


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ARCSimClothParameters:
    """Metric thin-shell and numerical settings for one ARCSim rollout."""

    timestep_s: float
    youngs_modulus_pa: float
    poisson_ratio: float
    volume_density_kg_m3: float
    thickness_m: float
    damping_s: float
    handle_stiffness: float
    gravity_m_s2: tuple[float, float, float]
    kinematic_handles: bool = False

    def __post_init__(self) -> None:
        positive = {
            "timestep_s": self.timestep_s,
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "volume_density_kg_m3": self.volume_density_kg_m3,
            "thickness_m": self.thickness_m,
            "handle_stiffness": self.handle_stiffness,
        }
        for name, value in positive.items():
            _require(math.isfinite(value) and value > 0.0, f"{name} must be positive")
        _require(
            math.isfinite(self.poisson_ratio) and -1.0 < self.poisson_ratio < 0.5,
            "poisson_ratio must be in (-1, 0.5)",
        )
        _require(
            math.isfinite(self.damping_s) and self.damping_s >= 0.0,
            "damping_s must be nonnegative",
        )
        gravity = np.asarray(self.gravity_m_s2, dtype=np.float64)
        _require(
            gravity.shape == (3,) and np.all(np.isfinite(gravity)), "invalid gravity"
        )
        _require(
            isinstance(self.kinematic_handles, bool),
            "kinematic_handles must be boolean",
        )

    @property
    def areal_density_kg_m2(self) -> float:
        return float(self.volume_density_kg_m3 * self.thickness_m)

    @property
    def membrane_coefficients_n_m(self) -> tuple[float, float, float, float]:
        extensional = (
            self.youngs_modulus_pa * self.thickness_m / (1.0 - self.poisson_ratio**2)
        )
        return (
            float(extensional),
            float(self.poisson_ratio * extensional),
            float(extensional),
            float(
                2.0
                * self.youngs_modulus_pa
                * self.thickness_m
                / (1.0 + self.poisson_ratio)
            ),
        )

    @property
    def bending_stiffness_n_m(self) -> float:
        return float(
            self.youngs_modulus_pa
            * self.thickness_m**3
            / (12.0 * (1.0 - self.poisson_ratio**2))
        )


@dataclass(frozen=True)
class ARCSimRollout:
    """Numerical output of one isolated ARCSim replay."""

    final_vertices_m: np.ndarray
    maximum_pin_target_error_m: float
    step_count: int
    elapsed_s: float

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
            math.isfinite(self.elapsed_s) and self.elapsed_s >= 0.0,
            "elapsed_s is invalid",
        )
        vertices = np.array(vertices, dtype=np.float64, copy=True)
        vertices.setflags(write=False)
        object.__setattr__(self, "final_vertices_m", vertices)


def write_arcsim_isotropic_material(
    path: str | Path,
    parameters: ARCSimClothParameters,
) -> None:
    """Write ARCSim's tabulated format for an isotropic linear thin shell."""

    destination = Path(path)
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    membrane = [0.5 * value for value in parameters.membrane_coefficients_n_m]
    payload = {
        "density": parameters.areal_density_kg_m2,
        "stretching": [membrane for _ in range(6)],
        "bending": [
            [parameters.bending_stiffness_n_m for _ in range(5)] for _ in range(3)
        ],
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _motion_points(
    controller: FlingPinController,
    *,
    pin_offset: int,
    duration_s: float,
    step_count: int,
) -> list[dict[str, object]]:
    initial = controller.initial_positions_m[pin_offset]
    points: list[dict[str, object]] = []
    for step_index in range(step_count + 1):
        time_s = duration_s * step_index / step_count
        target = controller.targets_at(time_s)[pin_offset]
        translation = (target - initial).tolist()
        points.append(
            {
                "time": float(time_s),
                "transform": {"translate": [float(value) for value in translation]},
            }
        )
    return points


def write_arcsim_scene(
    path: str | Path,
    *,
    mesh_path: str | Path,
    material_path: str | Path,
    controller: FlingPinController,
    parameters: ARCSimClothParameters,
    duration_s: float,
    initial_pose_xyz_wxyz: tuple[float, float, float, float, float, float, float],
) -> int:
    """Write a fixed-topology, action-driven ARCSim scene."""

    destination = Path(path)
    mesh = Path(mesh_path).resolve()
    material = Path(material_path).resolve()
    _require(mesh.is_file(), f"mesh does not exist: {mesh}")
    _require(material.is_file(), f"material does not exist: {material}")
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    _require(math.isfinite(duration_s) and duration_s > 0.0, "duration is invalid")
    ratio = duration_s / parameters.timestep_s
    step_count = int(round(ratio))
    _require(
        step_count >= 1 and math.isclose(ratio, step_count, abs_tol=1e-12),
        "duration must be an integer multiple of timestep_s",
    )
    pose = np.asarray(initial_pose_xyz_wxyz, dtype=np.float64)
    _require(pose.shape == (7,) and np.all(np.isfinite(pose)), "invalid pose")
    _require(
        np.allclose(
            pose[3:],
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            rtol=0.0,
            atol=0.0,
        ),
        "v8 ARCSim adapter supports the frozen identity rotation only",
    )
    handles: list[dict[str, object]] = [
        {"nodes": [int(controller.pin_indices[0])], "motion": 0},
        {"nodes": [int(controller.pin_indices[1])], "motion": 1},
    ]
    if parameters.kinematic_handles:
        for handle in handles:
            handle["kinematic"] = True

    payload = {
        "frame_time": float(parameters.timestep_s),
        "frame_steps": 1,
        "end_time": float(duration_s),
        "cloths": [
            {
                "mesh": str(mesh),
                "transform": {"translate": pose[:3].tolist()},
                "materials": [
                    {
                        "data": str(material),
                        "damping": float(parameters.damping_s),
                    }
                ],
            }
        ],
        "motions": [
            _motion_points(
                controller,
                pin_offset=0,
                duration_s=duration_s,
                step_count=step_count,
            ),
            _motion_points(
                controller,
                pin_offset=1,
                duration_s=duration_s,
                step_count=step_count,
            ),
        ],
        "handles": handles,
        "gravity": [float(value) for value in parameters.gravity_m_s2],
        "disable": [
            "proximity",
            "strainlimiting",
            "collision",
            "remeshing",
            "separation",
            "popfilter",
            "plasticity",
        ],
        "magic": {"handle_stiffness": float(parameters.handle_stiffness)},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return step_count


def load_arcsim_vertices(path: str | Path) -> np.ndarray:
    """Read world-space node positions from one ARCSim OBJ output."""

    source = Path(path)
    _require(source.is_file(), f"ARCSim output does not exist: {source}")
    vertices: list[list[float]] = []
    with source.open("r", encoding="ascii") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            fields = line.split()
            _require(len(fields) == 4, f"malformed ARCSim vertex row in {source}")
            vertices.append([float(value) for value in fields[1:]])
    array = np.asarray(vertices, dtype=np.float64)
    _require(
        array.ndim == 2
        and array.shape[1:] == (3,)
        and len(array) >= 3
        and np.all(np.isfinite(array)),
        f"invalid ARCSim vertices in {source}",
    )
    return array


def run_arcsim_fling(
    *,
    source_mesh_path: str | Path,
    initial_vertices_m: np.ndarray,
    controller: FlingPinController,
    parameters: ARCSimClothParameters,
    duration_s: float,
    initial_pose_xyz_wxyz: tuple[float, float, float, float, float, float, float],
    workspace: str | Path,
    arcsim_root: str | Path,
    timeout_s: float,
) -> ARCSimRollout:
    """Run one fixed-topology, single-thread ARCSim replay."""

    initial = np.asarray(initial_vertices_m, dtype=np.float64)
    _require(initial.ndim == 2 and initial.shape[1] == 3, "invalid initial vertices")
    _require(len(initial) >= 128 and np.all(np.isfinite(initial)), "invalid mesh")
    _require(max(controller.pin_indices) < len(initial), "pin index is out of bounds")
    _require(math.isfinite(timeout_s) and timeout_s > 0.0, "timeout_s is invalid")
    destination = Path(workspace).resolve()
    root = Path(arcsim_root).resolve()
    executable = root / "bin" / "arcsim"
    _require(not destination.exists(), f"workspace already exists: {destination}")
    _require(executable.is_file(), f"ARCSim executable is missing: {executable}")
    destination.mkdir(parents=True)
    material_path = destination / "isotropic_material.json"
    scene_path = destination / "scene.json"
    output_path = destination / "output"
    log_path = destination / "arcsim.log"
    output_path.mkdir()
    write_arcsim_isotropic_material(material_path, parameters)
    step_count = write_arcsim_scene(
        scene_path,
        mesh_path=source_mesh_path,
        material_path=material_path,
        controller=controller,
        parameters=parameters,
        duration_s=duration_s,
        initial_pose_xyz_wxyz=initial_pose_xyz_wxyz,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
        }
    )
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                str(executable),
                "simulateoffline",
                str(scene_path),
                str(output_path),
            ],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=float(timeout_s),
        )
    elapsed_s = time.monotonic() - start
    if completed.returncode != 0:
        raise RuntimeError(f"ARCSim exited with {completed.returncode}; see {log_path}")

    maximum_pin_error = 0.0
    final: np.ndarray | None = None
    pin_indices = np.asarray(controller.pin_indices, dtype=np.int64)
    for step_index in range(step_count + 1):
        frame_path = output_path / f"{step_index:04d}_00.obj"
        vertices = load_arcsim_vertices(frame_path)
        _require(vertices.shape == initial.shape, "ARCSim changed the vertex contract")
        targets = controller.targets_at(step_index * parameters.timestep_s)
        maximum_pin_error = max(
            maximum_pin_error,
            float(np.max(np.linalg.norm(vertices[pin_indices] - targets, axis=1))),
        )
        final = vertices
    _require(final is not None, "ARCSim emitted no final state")
    return ARCSimRollout(
        final_vertices_m=final,
        maximum_pin_target_error_m=maximum_pin_error,
        step_count=step_count,
        elapsed_s=elapsed_s,
    )
