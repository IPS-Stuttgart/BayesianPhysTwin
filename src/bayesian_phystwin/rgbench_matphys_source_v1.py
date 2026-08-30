"""Target-closed RGBench inputs for the native MatPhys source study."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from .phystwin_graph import (
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from .rgbench_matphys_protocol_v1 import (
    AmendedRGBenchMatPhysProtocolV1,
    RGBenchCellV1,
)

PCD_FILE_PATTERN: Final = re.compile(
    r"^pointcloud_(?P<timestamp>[0-9]+(?:\.[0-9]+)?)_segmented\.pcd$"
)
PCD_FIELDS: Final = ("x", "y", "z", "rgb")
PCD_SIZES: Final = (4, 4, 4, 4)
PCD_TYPES: Final = ("F", "F", "F", "F")
PCD_COUNTS: Final = (1, 1, 1, 1)
POSE_COLUMNS: Final = (
    "time",
    "pos_x",
    "pos_y",
    "pos_z",
)
LEFT_BASE_OFFSET_M: Final = np.asarray((0.0, 0.25, 0.0), dtype=np.float64)
RIGHT_BASE_OFFSET_M: Final = np.asarray((0.0, -0.25, 0.0), dtype=np.float64)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _ordinary_file(path: Path, *, label: str) -> Path:
    _require(
        path.is_file() and not path.is_symlink(), f"{label} is not an ordinary file"
    )
    return path.resolve(strict=True)


@dataclass(frozen=True, slots=True)
class RGBenchPoseTrajectoryV1:
    """One timestamped end-effector position trajectory in world coordinates."""

    times_s: npt.NDArray[np.float64]
    positions_m: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RGBenchSourceEpisodeIndexV1:
    """Source-only episode paths and causal control signals."""

    cell: RGBenchCellV1
    episode_dir: Path
    pcd_paths: tuple[Path, ...]
    frame_times_s: npt.NDArray[np.float64]
    world_to_camera: npt.NDArray[np.float64]
    controller_points_m: npt.NDArray[np.float64]
    camera_delay_s: float


def amended_source_cell_v1(
    protocol: AmendedRGBenchMatPhysProtocolV1,
    *,
    garment_id: str,
    action: str,
    sample_id: str,
) -> RGBenchCellV1:
    """Return an amended source cell without resolving any target path."""

    identity = (garment_id, action, sample_id)
    for cell in protocol.source_cells:
        if cell.identity[:3] == identity:
            return cell
    for cell in protocol.target_cells:
        if cell.identity[:3] == identity:
            raise PermissionError("reserved RGBench target cell access is forbidden")
    raise ValueError("cell is absent from the amended RGBench protocol")


def resolve_amended_source_episode_dir_v1(
    protocol: AmendedRGBenchMatPhysProtocolV1,
    dataset_root: str | Path,
    *,
    garment_id: str,
    action: str,
    sample_id: str,
) -> tuple[RGBenchCellV1, Path]:
    """Resolve one exact source directory without enumerating the dataset root."""

    cell = amended_source_cell_v1(
        protocol,
        garment_id=garment_id,
        action=action,
        sample_id=sample_id,
    )
    root = Path(dataset_root).resolve(strict=True)
    _require(root.is_dir(), "RGBench dataset root is not a directory")
    candidate = root.joinpath(*Path(cell.data_subfolder).parts)
    _require(not candidate.is_symlink(), "RGBench source episode must not be a symlink")
    episode = candidate.resolve(strict=True)
    _require(episode.is_dir(), "RGBench source episode is not a directory")
    try:
        episode.relative_to(root)
    except ValueError as error:
        raise ValueError("RGBench source episode escapes the dataset root") from error
    return cell, episode


def pcd_timestamp_s_v1(path: str | Path) -> float:
    """Parse the absolute timestamp from one RGBench segmented-cloud filename."""

    match = PCD_FILE_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError("RGBench segmented PCD filename is invalid")
    timestamp = float(match.group("timestamp"))
    _require(np.isfinite(timestamp) and timestamp > 0.0, "PCD timestamp is invalid")
    return timestamp


def _pcd_header(stream) -> tuple[dict[str, tuple[str, ...]], int]:
    header: dict[str, tuple[str, ...]] = {}
    byte_count = 0
    for _ in range(64):
        raw = stream.readline()
        byte_count += len(raw)
        if not raw:
            raise ValueError("PCD header ended before DATA")
        try:
            line = raw.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("PCD header is not ASCII") from error
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        _require(key not in header, f"PCD header repeats {key}")
        header[key] = tuple(values)
        if key == "DATA":
            break
    else:
        raise ValueError("PCD header is too long")
    return header, byte_count


def read_binary_xyzrgb_pcd_v1(path: str | Path) -> npt.NDArray[np.float32]:
    """Read the strict binary XYZRGB layout released by RGBench."""

    source = _ordinary_file(Path(path), label="RGBench PCD")
    with source.open("rb") as stream:
        header, header_bytes = _pcd_header(stream)
        _require(header.get("VERSION") == ("0.7",), "PCD version changed")
        _require(header.get("FIELDS") == PCD_FIELDS, "PCD fields changed")
        _require(
            tuple(int(value) for value in header.get("SIZE", ())) == PCD_SIZES,
            "PCD field sizes changed",
        )
        _require(header.get("TYPE") == PCD_TYPES, "PCD field types changed")
        _require(
            tuple(int(value) for value in header.get("COUNT", ())) == PCD_COUNTS,
            "PCD field counts changed",
        )
        _require(header.get("HEIGHT") == ("1",), "PCD must be unorganized")
        _require(header.get("DATA") == ("binary",), "PCD data mode changed")
        try:
            width = int(header["WIDTH"][0])
            point_count = int(header["POINTS"][0])
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError("PCD point count is invalid") from error
        _require(width == point_count and point_count > 0, "PCD point count changed")
        payload = stream.read()
    _require(
        source.stat().st_size == header_bytes + len(payload),
        "PCD byte accounting changed",
    )
    _require(len(payload) == 16 * point_count, "PCD payload length changed")
    records = np.frombuffer(
        payload,
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]),
        count=point_count,
    )
    points = np.column_stack((records["x"], records["y"], records["z"]))
    _require(points.shape == (point_count, 3), "PCD coordinates have invalid shape")
    _require(np.all(np.isfinite(points)), "PCD coordinates must be finite")
    return np.asarray(points, dtype=np.float32)


def load_world_to_camera_v1(path: str | Path) -> npt.NDArray[np.float64]:
    """Load and validate the RGBench camera extrinsic matrix."""

    source = _ordinary_file(Path(path), label="RGBench calibration")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RGBench calibration is not valid JSON") from error
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (4, 4), "world-to-camera matrix must be 4 by 4")
    _require(np.all(np.isfinite(matrix)), "world-to-camera matrix must be finite")
    _require(
        np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12),
        "world-to-camera homogeneous row changed",
    )
    _require(
        abs(float(np.linalg.det(matrix[:3, :3]))) > 1e-8, "camera rotation is singular"
    )
    return matrix


def camera_points_to_world_v1(
    points_camera_m: np.ndarray,
    world_to_camera: np.ndarray,
) -> npt.NDArray[np.float32]:
    """Apply RGBench's inverse extrinsic convention to camera-frame points."""

    points = np.asarray(points_camera_m, dtype=np.float64)
    transform = np.asarray(world_to_camera, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 3, "points must have shape (N,3)")
    _require(len(points) > 0 and np.all(np.isfinite(points)), "points must be finite")
    _require(transform.shape == (4, 4), "world-to-camera matrix must be 4 by 4")
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    condition_number = float(np.linalg.cond(transform))
    _require(
        np.isfinite(condition_number) and condition_number <= 1e8,
        "world-to-camera matrix is ill-conditioned",
    )
    world = np.linalg.solve(transform, homogeneous.T).T
    residual = float(np.linalg.norm(world @ transform.T - homogeneous, ord=np.inf))
    residual_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * condition_number
        * max(1.0, float(np.linalg.norm(homogeneous, ord=np.inf)))
    )
    _require(residual <= residual_tolerance, "camera transform solve is inaccurate")
    _require(
        np.allclose(world[:, 3], 1.0, rtol=0.0, atol=1e-8),
        "camera transform produced invalid homogeneous coordinates",
    )
    result = world[:, :3]
    _require(np.all(np.isfinite(result)), "world coordinates must be finite")
    return np.asarray(result, dtype=np.float32)


def read_pose_trajectory_csv_v1(
    path: str | Path,
    *,
    base_offset_m: np.ndarray,
) -> RGBenchPoseTrajectoryV1:
    """Read the released end-pose columns without pandas or robot dependencies."""

    source = _ordinary_file(Path(path), label="RGBench pose trajectory")
    offset = np.asarray(base_offset_m, dtype=np.float64)
    _require(
        offset.shape == (3,) and np.all(np.isfinite(offset)), "base offset is invalid"
    )
    times: list[float] = []
    positions: list[tuple[float, float, float]] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        _require(
            all(column in fields for column in POSE_COLUMNS),
            "RGBench pose columns changed",
        )
        for row in reader:
            try:
                timestamp = float(row["time"])
                point = (
                    float(row["pos_x"]),
                    float(row["pos_y"]),
                    float(row["pos_z"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("RGBench pose row is invalid") from error
            times.append(timestamp)
            positions.append(point)
    time_array = np.asarray(times, dtype=np.float64)
    point_array = np.asarray(positions, dtype=np.float64)
    _require(len(time_array) >= 2, "RGBench pose trajectory is too short")
    _require(point_array.shape == (len(time_array), 3), "RGBench pose shape changed")
    _require(
        np.all(np.isfinite(time_array)) and np.all(np.isfinite(point_array)),
        "RGBench pose trajectory must be finite",
    )
    _require(np.all(np.diff(time_array) > 0.0), "RGBench pose times must increase")
    return RGBenchPoseTrajectoryV1(
        times_s=time_array,
        positions_m=point_array + offset,
    )


def interpolate_pose_positions_v1(
    trajectory: RGBenchPoseTrajectoryV1,
    query_times_s: np.ndarray,
) -> npt.NDArray[np.float32]:
    """Linearly interpolate positions while rejecting temporal extrapolation."""

    query = np.asarray(query_times_s, dtype=np.float64).reshape(-1)
    _require(len(query) > 0 and np.all(np.isfinite(query)), "query times are invalid")
    _require(np.all(np.diff(query) > 0.0), "query times must increase")
    tolerance = 1e-9
    _require(
        query[0] >= trajectory.times_s[0] - tolerance
        and query[-1] <= trajectory.times_s[-1] + tolerance,
        "controller interpolation would extrapolate",
    )
    values = np.column_stack(
        [
            np.interp(query, trajectory.times_s, trajectory.positions_m[:, axis])
            for axis in range(3)
        ]
    )
    return np.asarray(values, dtype=np.float32)


def load_rgbench_source_episode_index_v1(
    protocol: AmendedRGBenchMatPhysProtocolV1,
    dataset_root: str | Path,
    *,
    garment_id: str,
    action: str,
    sample_id: str,
    camera_delay_s: float,
) -> RGBenchSourceEpisodeIndexV1:
    """Index one registered source episode and align its recorded controls."""

    _require(
        np.isfinite(camera_delay_s) and 0.0 <= camera_delay_s <= 1.0,
        "camera delay is invalid",
    )
    cell, episode = resolve_amended_source_episode_dir_v1(
        protocol,
        dataset_root,
        garment_id=garment_id,
        action=action,
        sample_id=sample_id,
    )
    pcd_dir = episode / "segment_pcds"
    _require(
        pcd_dir.is_dir() and not pcd_dir.is_symlink(),
        "segment PCD directory is invalid",
    )
    paths: list[tuple[float, Path]] = []
    for candidate in pcd_dir.iterdir():
        if PCD_FILE_PATTERN.fullmatch(candidate.name) is None:
            continue
        ordinary = _ordinary_file(candidate, label="RGBench PCD")
        paths.append((pcd_timestamp_s_v1(ordinary), ordinary))
    paths.sort(key=lambda item: (item[0], item[1].name))
    _require(len(paths) >= 3, "RGBench source episode has too few PCD frames")
    times = np.asarray([item[0] for item in paths], dtype=np.float64)
    _require(np.all(np.diff(times) > 0.0), "RGBench PCD timestamps must increase")
    world_to_camera = load_world_to_camera_v1(
        episode / "calibration" / "world_to_camera_transform.json"
    )
    left = read_pose_trajectory_csv_v1(
        episode / "joints" / "left_arm_joint_states_and_end_pose.csv",
        base_offset_m=LEFT_BASE_OFFSET_M,
    )
    right = read_pose_trajectory_csv_v1(
        episode / "joints" / "right_arm_joint_states_and_end_pose.csv",
        base_offset_m=RIGHT_BASE_OFFSET_M,
    )
    aligned = times + float(camera_delay_s)
    controller = np.stack(
        (
            interpolate_pose_positions_v1(left, aligned),
            interpolate_pose_positions_v1(right, aligned),
        ),
        axis=1,
    )
    return RGBenchSourceEpisodeIndexV1(
        cell=cell,
        episode_dir=episode,
        pcd_paths=tuple(item[1] for item in paths),
        frame_times_s=times,
        world_to_camera=world_to_camera,
        controller_points_m=controller,
        camera_delay_s=float(camera_delay_s),
    )


def load_episode_world_points_v1(
    episode: RGBenchSourceEpisodeIndexV1,
    frame_index: int,
) -> npt.NDArray[np.float32]:
    """Decode one explicitly indexed source frame into world coordinates."""

    _require(
        type(frame_index) is int and 0 <= frame_index < len(episode.pcd_paths),
        "RGBench source frame index is invalid",
    )
    camera = read_binary_xyzrgb_pcd_v1(episode.pcd_paths[frame_index])
    return camera_points_to_world_v1(camera, episode.world_to_camera)


def deterministic_farthest_points_v1(
    points_m: np.ndarray,
    *,
    count: int,
) -> npt.NDArray[np.float32]:
    """Select unique material nodes with canonical ordering and deterministic ties."""

    points = np.asarray(points_m, dtype=np.float32)
    _require(points.ndim == 2 and points.shape[1] == 3, "points must have shape (N,3)")
    _require(len(points) > 0 and np.all(np.isfinite(points)), "points must be finite")
    unique = np.unique(points, axis=0)
    order = np.lexsort((unique[:, 2], unique[:, 1], unique[:, 0]))
    canonical = unique[order]
    _require(
        type(count) is int and 2 <= count <= len(canonical), "sample count is invalid"
    )
    selected: npt.NDArray[np.int64] = np.empty(count, dtype=np.int64)
    selected[0] = 0
    minimum_squared = np.sum(np.square(canonical - canonical[0]), axis=1)
    for index in range(1, count):
        minimum_squared[selected[:index]] = -1.0
        chosen = int(np.argmax(minimum_squared))
        _require(
            minimum_squared[chosen] > 0.0, "point cloud has too few unique positions"
        )
        selected[index] = chosen
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum(np.square(canonical - canonical[chosen]), axis=1),
        )
    return np.asarray(canonical[selected], dtype=np.float32)


def spring_graph_component_count_v1(graph: PhysTwinSpringGraph) -> int:
    """Count connected components in the object-only spring topology."""

    object_count = int(graph.num_object_points or 0)
    _require(object_count > 0, "spring graph object count is invalid")
    edges = np.asarray(graph.springs[: graph.num_object_springs], dtype=np.int64)
    _require(len(edges) > 0, "spring graph has no object edges")
    parents: npt.NDArray[np.int64] = np.arange(object_count, dtype=np.int64)

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = int(parents[value])
        return value

    for first_raw, second_raw in edges:
        first, second = find(int(first_raw)), find(int(second_raw))
        if first != second:
            parents[max(first, second)] = min(first, second)
    return len({find(index) for index in range(object_count)})


def build_rgbench_matphys_graph_v1(
    initial_points_world_m: np.ndarray,
    initial_controller_points_m: np.ndarray,
    *,
    node_count: int,
    total_mass_kg: float,
    object_radius_m: float,
    object_max_neighbours: int,
    controller_radius_m: float,
    controller_max_neighbours: int,
) -> PhysTwinSpringGraph:
    """Build the deterministic source graph and require usable bilateral anchors."""

    _require(np.isfinite(total_mass_kg) and total_mass_kg > 0.0, "mass is invalid")
    nodes = deterministic_farthest_points_v1(initial_points_world_m, count=node_count)
    controls = np.asarray(initial_controller_points_m, dtype=np.float32)
    _require(controls.shape == (2, 3), "RGBench controls must be bilateral")
    graph = build_phystwin_spring_graph(
        nodes,
        controls,
        config=PhysTwinSpringGraphConfig(
            object_radius=float(object_radius_m),
            object_max_neighbours=int(object_max_neighbours),
            controller_radius=float(controller_radius_m),
            controller_max_neighbours=int(controller_max_neighbours),
        ),
    )
    _require(graph.num_object_springs > 0, "RGBench object graph is empty")
    _require(
        spring_graph_component_count_v1(graph) == 1,
        "RGBench object graph is disconnected",
    )
    controller_edges = np.asarray(graph.springs[graph.num_object_springs :])
    _require(len(controller_edges) >= 2, "RGBench controller graph is empty")
    object_count = len(nodes)
    controller_ids = controller_edges[:, 0]
    if np.any(controller_ids < object_count):
        controller_ids = controller_edges[:, 1]
    _require(
        set((controller_ids - object_count).tolist()) == {0, 1},
        "RGBench controller graph does not anchor both arms",
    )
    masses: npt.NDArray[np.float32] = np.full(
        len(graph.vertices), total_mass_kg / node_count, dtype=np.float32
    )
    masses[node_count:] = 1.0
    return replace(graph, masses=masses)


__all__ = [
    "LEFT_BASE_OFFSET_M",
    "RIGHT_BASE_OFFSET_M",
    "RGBenchPoseTrajectoryV1",
    "RGBenchSourceEpisodeIndexV1",
    "amended_source_cell_v1",
    "build_rgbench_matphys_graph_v1",
    "camera_points_to_world_v1",
    "deterministic_farthest_points_v1",
    "interpolate_pose_positions_v1",
    "load_episode_world_points_v1",
    "load_rgbench_source_episode_index_v1",
    "load_world_to_camera_v1",
    "pcd_timestamp_s_v1",
    "read_binary_xyzrgb_pcd_v1",
    "read_pose_trajectory_csv_v1",
    "resolve_amended_source_episode_dir_v1",
    "spring_graph_component_count_v1",
]
