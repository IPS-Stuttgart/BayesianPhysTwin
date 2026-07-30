"""Source-only LibuIPC cloth backbone helpers for RGBench."""

from __future__ import annotations

import csv
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly_float_array(value: object, *, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    tail_matches = (
        True
        if not shape_tail
        else tuple(array.shape[-len(shape_tail) :]) == shape_tail
    )
    _require(
        array.ndim >= len(shape_tail) and tail_matches,
        f"array must end with shape {shape_tail}",
    )
    _require(np.all(np.isfinite(array)), "array must be finite")
    array = np.array(array, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PositionTrajectory:
    """Timestamped Cartesian actuator positions in the RGBench world frame."""

    times_s: np.ndarray
    positions_m: np.ndarray

    def __post_init__(self) -> None:
        times = _readonly_float_array(self.times_s, shape_tail=())
        positions = _readonly_float_array(self.positions_m, shape_tail=(3,))
        _require(times.ndim == 1, "times_s must have shape (T,)")
        _require(positions.ndim == 2, "positions_m must have shape (T, 3)")
        _require(len(times) == len(positions) >= 2, "trajectory needs at least 2 rows")
        _require(np.all(np.diff(times) > 0.0), "trajectory times must increase")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "positions_m", positions)

    def position_at(self, absolute_time_s: float) -> np.ndarray:
        """Linearly interpolate position, clamping outside the recorded interval."""

        time = float(absolute_time_s)
        _require(math.isfinite(time), "query time must be finite")
        if time <= self.times_s[0]:
            return self.positions_m[0].copy()
        if time >= self.times_s[-1]:
            return self.positions_m[-1].copy()
        upper = int(np.searchsorted(self.times_s, time, side="right"))
        lower = upper - 1
        alpha = (time - self.times_s[lower]) / (
            self.times_s[upper] - self.times_s[lower]
        )
        return (
            (1.0 - alpha) * self.positions_m[lower]
            + alpha * self.positions_m[upper]
        )


def load_rgbbench_position_trajectory(
    path: str | Path,
    *,
    base_translation_m: tuple[float, float, float],
) -> PositionTrajectory:
    """Load only the recorded time and end-effector position columns."""

    source = Path(path)
    _require(source.is_file(), f"trajectory does not exist: {source}")
    times: list[float] = []
    positions: list[list[float]] = []
    offset = np.asarray(base_translation_m, dtype=np.float64)
    _require(offset.shape == (3,) and np.all(np.isfinite(offset)), "invalid offset")
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"time", "pos_x", "pos_y", "pos_z"}
        _require(
            reader.fieldnames is not None
            and required <= set(reader.fieldnames),
            f"{source} is missing required columns",
        )
        for row in reader:
            times.append(float(row["time"]))
            positions.append(
                (
                    np.asarray(
                        [float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])],
                        dtype=np.float64,
                    )
                    + offset
                ).tolist()
            )
    return PositionTrajectory(np.asarray(times), np.asarray(positions))


def transform_vertices_wxyz(
    vertices_m: np.ndarray,
    pose_xyz_wxyz: tuple[float, float, float, float, float, float, float],
) -> np.ndarray:
    """Apply an RGBench ``xyz + wxyz`` pose to mesh vertices."""

    vertices = np.asarray(vertices_m, dtype=np.float64)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3, "invalid vertices")
    _require(np.all(np.isfinite(vertices)), "vertices must be finite")
    pose = np.asarray(pose_xyz_wxyz, dtype=np.float64)
    _require(pose.shape == (7,) and np.all(np.isfinite(pose)), "invalid pose")
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    _require(norm > 1e-12, "pose quaternion must be nonzero")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transformed = vertices @ rotation.T + pose[:3]
    transformed.setflags(write=False)
    return transformed


def triangle_mesh_area_m2(
    vertices_m: np.ndarray,
    triangles: np.ndarray,
) -> float:
    """Return total triangle area for a metric surface mesh."""

    vertices = np.asarray(vertices_m, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3, "invalid vertices")
    _require(faces.ndim == 2 and faces.shape[1] == 3, "invalid triangles")
    _require(
        len(faces) >= 1
        and np.min(faces) >= 0
        and np.max(faces) < len(vertices),
        "triangle index is out of bounds",
    )
    edges_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edges_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    area = float(0.5 * np.linalg.norm(np.cross(edges_a, edges_b), axis=1).sum())
    _require(math.isfinite(area) and area > 0.0, "mesh area must be positive")
    return area


@dataclass(frozen=True)
class FlingPinController:
    """Match RGBench's fixed-point fling preparation and playback semantics."""

    pin_indices: tuple[int, int]
    initial_positions_m: np.ndarray
    left: PositionTrajectory
    right: PositionTrajectory
    prepare_time_s: float
    wait_time_s: float

    def __post_init__(self) -> None:
        pins = tuple(int(index) for index in self.pin_indices)
        initial = _readonly_float_array(self.initial_positions_m, shape_tail=(3,))
        _require(len(pins) == 2 and min(pins) >= 0, "two pin indices are required")
        _require(initial.shape == (2, 3), "initial_positions_m must have shape (2, 3)")
        _require(
            math.isfinite(self.prepare_time_s) and self.prepare_time_s >= 0.0,
            "prepare_time_s must be nonnegative",
        )
        _require(
            math.isfinite(self.wait_time_s) and self.wait_time_s >= 0.0,
            "wait_time_s must be nonnegative",
        )
        object.__setattr__(self, "pin_indices", pins)
        object.__setattr__(self, "initial_positions_m", initial)

    @property
    def master_start_time_s(self) -> float:
        return float(min(self.left.times_s[0], self.right.times_s[0]))

    def targets_at(self, simulation_time_s: float) -> np.ndarray:
        """Return the two constrained vertex targets at simulation time."""

        time = float(simulation_time_s)
        _require(math.isfinite(time) and time >= 0.0, "simulation time is invalid")
        first = np.vstack((self.left.positions_m[0], self.right.positions_m[0]))
        if time < self.prepare_time_s:
            alpha = time / self.prepare_time_s if self.prepare_time_s > 0.0 else 1.0
            return self.initial_positions_m + alpha * (
                first - self.initial_positions_m
            )
        if time < self.prepare_time_s + self.wait_time_s:
            return first.copy()
        absolute_time = (
            self.master_start_time_s
            + time
            - self.prepare_time_s
            - self.wait_time_s
        )
        return np.vstack(
            (
                self.left.position_at(absolute_time),
                self.right.position_at(absolute_time),
            )
        )


@dataclass(frozen=True)
class LibuIPCClothParameters:
    """Physical and numerical settings for one source-only cloth rollout."""

    timestep_s: float
    youngs_modulus_pa: float
    poisson_ratio: float
    volume_density_kg_m3: float
    thickness_m: float
    bending_stiffness: float
    friction_coefficient: float
    contact_distance_m: float
    contact_resistance: float
    constraint_strength_ratio: float

    def __post_init__(self) -> None:
        positive = {
            "timestep_s": self.timestep_s,
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "volume_density_kg_m3": self.volume_density_kg_m3,
            "thickness_m": self.thickness_m,
            "bending_stiffness": self.bending_stiffness,
            "contact_distance_m": self.contact_distance_m,
            "contact_resistance": self.contact_resistance,
            "constraint_strength_ratio": self.constraint_strength_ratio,
        }
        for name, value in positive.items():
            _require(math.isfinite(value) and value > 0.0, f"{name} must be positive")
        _require(
            math.isfinite(self.poisson_ratio) and -1.0 < self.poisson_ratio < 0.5,
            "poisson_ratio must be in (-1, 0.5)",
        )
        _require(
            math.isfinite(self.friction_coefficient)
            and self.friction_coefficient >= 0.0,
            "friction_coefficient must be nonnegative",
        )


@dataclass(frozen=True)
class ReplayEnsembleSummary:
    """Target-free numerical spread of independent simulator replays."""

    mean_vertices_m: np.ndarray
    variance_m2: np.ndarray
    maximum_pairwise_rmse_m: float
    maximum_pairwise_coordinate_difference_m: float

    def __post_init__(self) -> None:
        mean = _readonly_float_array(self.mean_vertices_m, shape_tail=(3,))
        variance = _readonly_float_array(self.variance_m2, shape_tail=(3,))
        _require(mean.ndim == 2, "mean_vertices_m must have shape (N, 3)")
        _require(variance.shape == mean.shape, "variance shape changed")
        _require(np.all(variance >= 0.0), "replay variance must be nonnegative")
        _require(
            math.isfinite(self.maximum_pairwise_rmse_m)
            and self.maximum_pairwise_rmse_m >= 0.0,
            "maximum pairwise RMSE is invalid",
        )
        _require(
            math.isfinite(self.maximum_pairwise_coordinate_difference_m)
            and self.maximum_pairwise_coordinate_difference_m >= 0.0,
            "maximum pairwise coordinate difference is invalid",
        )
        object.__setattr__(self, "mean_vertices_m", mean)
        object.__setattr__(self, "variance_m2", variance)


def summarize_independent_replays(
    replay_vertices_m: list[np.ndarray] | tuple[np.ndarray, ...],
) -> ReplayEnsembleSummary:
    """Summarize replay spread without consulting an observed outcome."""

    _require(len(replay_vertices_m) >= 2, "at least two replays are required")
    arrays = [
        np.asarray(replay, dtype=np.float64) for replay in replay_vertices_m
    ]
    reference_shape = arrays[0].shape
    _require(
        len(reference_shape) == 2 and reference_shape[1] == 3,
        "replay vertices must have shape (N, 3)",
    )
    _require(
        all(array.shape == reference_shape for array in arrays),
        "replay vertex contracts differ",
    )
    _require(
        all(np.all(np.isfinite(array)) for array in arrays),
        "replay vertices must be finite",
    )
    maximum_rmse = 0.0
    maximum_coordinate = 0.0
    for left_index, left in enumerate(arrays[:-1]):
        for right in arrays[left_index + 1 :]:
            difference = left - right
            maximum_rmse = max(
                maximum_rmse,
                float(np.sqrt(np.mean(np.square(difference)))),
            )
            maximum_coordinate = max(
                maximum_coordinate,
                float(np.max(np.abs(difference))),
            )
    stacked = np.stack(arrays)
    return ReplayEnsembleSummary(
        mean_vertices_m=np.mean(stacked, axis=0),
        variance_m2=np.var(stacked, axis=0, ddof=1),
        maximum_pairwise_rmse_m=maximum_rmse,
        maximum_pairwise_coordinate_difference_m=maximum_coordinate,
    )


def _load_uipc() -> Any:
    try:
        return importlib.import_module("uipc")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "LibuIPC rollout requires the optional pyuipc runtime"
        ) from error


def libuipc_vector_values(values: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3)`` vectors to pyuipc's ``(..., 3, 1)`` layout."""

    array = np.asarray(values, dtype=np.float64)
    _require(
        array.ndim >= 1 and array.shape[-1] == 3,
        "LibuIPC vector values must end with dimension 3",
    )
    _require(np.all(np.isfinite(array)), "LibuIPC vector values must be finite")
    return np.ascontiguousarray(array[..., np.newaxis])


def run_libuipc_fling(
    *,
    vertices_m: np.ndarray,
    triangles: np.ndarray,
    controller: FlingPinController,
    parameters: LibuIPCClothParameters,
    duration_s: float,
    workspace: str | Path,
) -> np.ndarray:
    """Run a deterministic, fixed-point LibuIPC cloth rollout."""

    # pyuipc requests writable contiguous buffers even though it does not mutate
    # the caller's source artifacts.
    vertices = np.array(vertices_m, dtype=np.float64, order="C", copy=True)
    faces = np.array(triangles, dtype=np.int32, order="C", copy=True)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3, "invalid vertices")
    _require(faces.ndim == 2 and faces.shape[1] == 3, "invalid triangles")
    _require(
        len(vertices) >= 128
        and len(faces) >= 1
        and np.all(np.isfinite(vertices)),
        "mesh is not backend-admissible",
    )
    _require(
        np.min(faces) >= 0 and np.max(faces) < len(vertices),
        "triangle index is out of bounds",
    )
    _require(
        max(controller.pin_indices) < len(vertices),
        "pin index is out of bounds",
    )
    _require(math.isfinite(duration_s) and duration_s > 0.0, "duration is invalid")
    destination = Path(workspace)
    _require(not destination.exists(), f"workspace already exists: {destination}")
    destination.mkdir(parents=True)

    uipc = _load_uipc()
    uipc.Logger.set_level(uipc.Logger.Level.Warn)
    config = uipc.Scene.default_config()
    config["dt"] = float(parameters.timestep_s)
    config["gravity"] = [[0.0], [0.0], [-9.81]]
    config["contact"]["enable"] = 1
    config["contact"]["d_hat"] = float(parameters.contact_distance_m)
    config["contact"]["friction"]["enable"] = int(
        parameters.friction_coefficient > 0.0
    )
    config["line_search"]["report_energy"] = 0

    scene = uipc.Scene(config)
    scene.contact_tabular().default_model(
        float(parameters.friction_coefficient),
        float(parameters.contact_resistance),
    )
    default_contact = scene.contact_tabular().default_element()

    mesh = uipc.geometry.trimesh(vertices, faces)
    uipc.geometry.label_surface(mesh)
    shell = uipc.constitution.StrainLimitingBaraffWitkinShell()
    moduli = uipc.constitution.ElasticModuli2D.youngs_poisson(
        float(parameters.youngs_modulus_pa),
        float(parameters.poisson_ratio),
    )
    shell.apply_to(
        mesh,
        moduli,
        float(parameters.volume_density_kg_m3),
        float(parameters.thickness_m),
    )
    bending = uipc.constitution.DiscreteShellBending()
    bending.apply_to(mesh, float(parameters.bending_stiffness))
    position_constraint = uipc.constitution.SoftPositionConstraint()
    position_constraint.apply_to(
        mesh,
        float(parameters.constraint_strength_ratio),
    )
    default_contact.apply_to(mesh)

    cloth = scene.objects().create("rgbbench_cloth")
    cloth_slot, _ = cloth.geometries().create(mesh)
    pin_indices = np.asarray(controller.pin_indices, dtype=np.int64)
    animation_errors: list[str] = []
    animation_calls = 0

    def animate(info: Any) -> None:
        nonlocal animation_calls
        try:
            geometry = info.geo_slots()[0].geometry()
            constrained = geometry.vertices().find(uipc.builtin.is_constrained)
            aims = geometry.vertices().find(uipc.builtin.aim_position)
            constrained_view = uipc.view(constrained)
            aim_view = uipc.view(aims)
            constrained_view[pin_indices] = 1
            aim_view[pin_indices] = libuipc_vector_values(
                controller.targets_at(float(info.dt()) * float(info.frame()))
            )
            animation_calls += 1
        except Exception as error:
            animation_errors.append(f"{type(error).__name__}: {error}")
            raise

    scene.animator().insert(cloth, animate)
    ground_object = scene.objects().create("ground")
    ground_object.geometries().create(
        uipc.geometry.ground(0.0, np.asarray([0.0, 0.0, 1.0]))
    )

    engine = uipc.Engine("cuda", str(destination))
    world = uipc.World(engine)
    world.init(scene)
    _require(bool(world.is_valid()), "LibuIPC world initialization failed")
    _require(not animation_errors, f"LibuIPC animation failed: {animation_errors}")
    step_count = int(math.ceil(duration_s / parameters.timestep_s))
    for _ in range(step_count):
        world.advance()
        _require(
            not animation_errors,
            f"LibuIPC animation failed: {animation_errors}",
        )
        _require(bool(world.is_valid()), "LibuIPC world became invalid")
        world.retrieve()
    _require(
        animation_calls >= step_count,
        "LibuIPC did not invoke the position-constraint animation",
    )

    final_geometry = cloth_slot.geometry()
    final = np.asarray(final_geometry.positions().view(), dtype=np.float64).reshape(
        -1,
        3,
    )
    _require(final.shape == vertices.shape, "LibuIPC changed the vertex contract")
    _require(np.all(np.isfinite(final)), "LibuIPC produced non-finite vertices")
    return np.array(final, dtype=np.float64, copy=True)
