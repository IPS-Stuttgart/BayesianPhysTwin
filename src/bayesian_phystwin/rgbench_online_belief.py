"""RGBench point-cloud and mesh helpers for guarded online belief updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_PCD_NUMPY_TYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}


def load_binary_pcd_xyz(path: str | Path) -> np.ndarray:
    """Load XYZ from a PCD ``DATA binary`` archive without Open3D."""

    source = Path(path)
    _require(source.is_file(), f"PCD file does not exist: {source}")
    header: dict[str, list[str]] = {}
    with source.open("rb") as stream:
        while True:
            raw_line = stream.readline()
            _require(bool(raw_line), f"{source} has no DATA header")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"{source} has a non-ASCII PCD header") from error
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            key = fields[0].upper()
            header[key] = fields[1:]
            if key == "DATA":
                break
        _require(header["DATA"] == ["binary"], f"{source} is not binary PCD")
        names = header.get("FIELDS", [])
        sizes = [int(value) for value in header.get("SIZE", [])]
        types = [value.upper() for value in header.get("TYPE", [])]
        counts = [int(value) for value in header.get("COUNT", [])]
        _require(names and len(names) == len(sizes) == len(types), "invalid PCD fields")
        if not counts:
            counts = [1] * len(names)
        _require(len(counts) == len(names), "invalid PCD counts")
        _require({"x", "y", "z"} <= set(names), f"{source} has no XYZ fields")
        point_count = int(header.get("POINTS", ["0"])[0])
        _require(point_count >= 1, f"{source} has no points")
        dtype_fields: list[tuple[str, object]] = []
        for name, size, type_name, count in zip(
            names,
            sizes,
            types,
            counts,
            strict=True,
        ):
            scalar_type = _PCD_NUMPY_TYPES.get((type_name, size))
            _require(
                scalar_type is not None,
                f"{source} has unsupported PCD type {type_name}{size}",
            )
            dtype_fields.append(
                (name, scalar_type)
                if count == 1
                else (name, scalar_type, (count,))
            )
        values = np.fromfile(stream, dtype=np.dtype(dtype_fields), count=point_count)
    _require(len(values) == point_count, f"{source} ended before all points")
    points = np.column_stack((values["x"], values["y"], values["z"])).astype(
        np.float64,
        copy=False,
    )
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    _require(len(points) >= 1, f"{source} has no finite XYZ points")
    points.setflags(write=False)
    return points


def transform_points(points_m: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to an ``(N, 3)`` point cloud."""

    points = np.asarray(points_m, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3,
        "points_m must have shape (N, 3)",
    )
    _require(matrix.shape == (4, 4), "transform must have shape (4, 4)")
    _require(
        np.all(np.isfinite(points)) and np.all(np.isfinite(matrix)),
        "points and transform must be finite",
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    transformed = (matrix @ homogeneous.T).T[:, :3]
    transformed.setflags(write=False)
    return transformed


def load_rgbbench_world_cloud(
    pcd_path: str | Path,
    world_to_camera_path: str | Path,
) -> np.ndarray:
    """Load one RGBench segmented cloud and transform it into world coordinates."""

    transform_path = Path(world_to_camera_path)
    matrix = np.asarray(
        json.loads(transform_path.read_text(encoding="utf-8")),
        dtype=np.float64,
    )
    _require(matrix.shape == (4, 4), "world-to-camera transform shape changed")
    return transform_points(load_binary_pcd_xyz(pcd_path), np.linalg.inv(matrix))


def load_obj_triangles(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices and triangulate polygon faces deterministically."""

    source = Path(path)
    _require(source.is_file(), f"OBJ file does not exist: {source}")
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    with source.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.startswith("v "):
                fields = line.split()
                _require(len(fields) >= 4, f"invalid vertex at line {line_number}")
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                fields = line.split()[1:]
                _require(len(fields) >= 3, f"invalid face at line {line_number}")
                indices: list[int] = []
                for field in fields:
                    raw_index = int(field.split("/", maxsplit=1)[0])
                    index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    _require(
                        0 <= index < len(vertices),
                        f"invalid face index at line {line_number}",
                    )
                    indices.append(index)
                triangles.extend(
                    (indices[0], indices[offset], indices[offset + 1])
                    for offset in range(1, len(indices) - 1)
                )
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(triangles, dtype=np.int64)
    _require(
        vertex_array.ndim == 2
        and vertex_array.shape[1] == 3
        and len(vertex_array) >= 1,
        "OBJ has no vertices",
    )
    _require(
        face_array.ndim == 2
        and face_array.shape[1] == 3
        and len(face_array) >= 1,
        "OBJ has no faces",
    )
    _require(
        np.all(np.isfinite(vertex_array)),
        "OBJ contains non-finite vertices",
    )
    vertex_array.setflags(write=False)
    face_array.setflags(write=False)
    return vertex_array, face_array


def evaluation_pcd_paths(
    capture_root: str | Path,
    *,
    master_start_time_s: float,
    camera_delay_s: float,
    start_calculate_time_s: float,
    end_calculate_time_s: float,
    expected_count: int,
    expected_name_sha256: str,
) -> tuple[Path, ...]:
    """Select and verify the frozen RGBench evaluation point-cloud filenames."""

    cloud_dir = Path(capture_root) / "segment_pcds"
    paths = tuple(sorted(cloud_dir.glob("pointcloud_*_segmented.pcd")))
    selected: list[Path] = []
    for path in paths:
        absolute_time = float(
            path.name.removeprefix("pointcloud_").removesuffix("_segmented.pcd")
        )
        current_time = absolute_time - master_start_time_s + camera_delay_s
        if (
            start_calculate_time_s - 1e-9
            <= current_time
            <= end_calculate_time_s + 1e-9
        ):
            selected.append(path)
    _require(len(selected) == expected_count, "evaluation frame count changed")
    digest = hashlib.sha256(
        b"rgbbench-evaluation-point-cloud-names-v1\0"
        + "\n".join(path.name for path in selected).encode("ascii")
    ).hexdigest()
    _require(digest == expected_name_sha256, "evaluation frame names changed")
    return tuple(selected)


def real_to_sim_l1_chamfer_m(
    real_points_m: np.ndarray,
    simulated_points_m: np.ndarray,
) -> float:
    """Return RGBench's primary real-to-simulation Manhattan Chamfer."""

    try:
        from scipy.spatial import cKDTree
    except (ImportError, OSError) as error:
        raise RuntimeError("RGBench Chamfer evaluation requires scipy") from error
    real = np.asarray(real_points_m, dtype=np.float64)
    simulated = np.asarray(simulated_points_m, dtype=np.float64)
    _require(
        real.ndim == simulated.ndim == 2
        and real.shape[1] == simulated.shape[1] == 3,
        "point clouds must have shape (N, 3)",
    )
    _require(
        len(real) >= 1
        and len(simulated) >= 1
        and np.all(np.isfinite(real))
        and np.all(np.isfinite(simulated)),
        "point clouds must be finite and nonempty",
    )
    distances = cKDTree(simulated).query(real, k=1, p=1, workers=-1)[0]
    return float(np.mean(distances))


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
