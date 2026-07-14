"""Access-independent PokeFlex contract, fixture, and preflight audit."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PINNED_POKEFLEX_REPOSITORY = "https://github.com/pokeflex-dataset/reconstruction"
PINNED_POKEFLEX_COMMIT = "aaa8726072834a95bbe97e1a113588968c36e185"
POKEFLEX_TERMS_URL = "https://pokeflex.ait.ethz.ch/"
POKEFLEX_PREFLIGHT_SCHEMA_VERSION = 1
POKEFLEX_READINESS_CONFIG_SCHEMA_VERSION = 1
_CANONICAL_READINESS_CONFIG_SHA256 = (
    "256f6c0585a1eb592583b0a0c017e116baed9126f12119e80f866cd174b58070"
)
_MESH_FRAME_PATTERN = re.compile(r"(?:mesh-)?f?(\d{5})")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PokeFlexReadinessConfig:
    """Outcome-free choices fixed before the gated dataset is inspected."""

    protocol_id: str = "causal4d-pokeflex-public-readiness-v1"
    upstream_repository: str = PINNED_POKEFLEX_REPOSITORY
    upstream_commit: str = PINNED_POKEFLEX_COMMIT
    prefix_frame_count: int = 6
    minimum_future_frame_count: int = 6
    topology_sample_count: int = 9
    minimum_calibrated_camera_count: int = 3
    minimum_takes_per_object_for_cross_take: int = 5
    development_fraction: float = 0.60
    calibration_fraction: float = 0.20
    target_fraction: float = 0.20
    split_seed: str = "causal4d-pokeflex-public-v1"
    public_force_axis_index: int = 1
    public_force_threshold_n: float = 3.0
    expected_mesh_unit: str = "millimetre"
    expected_robot_translation_unit: str = "metre"
    minimum_plausible_mesh_extent_mm: float = 10.0
    maximum_plausible_mesh_extent_mm: float = 2000.0
    maximum_plausible_robot_translation_m: float = 10.0

    def __post_init__(self) -> None:
        _require(bool(self.protocol_id), "protocol_id is empty")
        _require(
            self.upstream_repository == PINNED_POKEFLEX_REPOSITORY,
            "unexpected PokeFlex upstream repository",
        )
        _require(
            self.upstream_commit == PINNED_POKEFLEX_COMMIT,
            "unexpected PokeFlex upstream commit",
        )
        _require(
            self.prefix_frame_count >= 2, "prefix must contain at least two frames"
        )
        _require(
            self.minimum_future_frame_count >= 2,
            "future must contain at least two frames",
        )
        _require(
            self.topology_sample_count >= 3,
            "topology audit needs at least three sampled meshes",
        )
        _require(
            self.minimum_calibrated_camera_count >= 1,
            "at least one camera is required",
        )
        _require(
            self.minimum_takes_per_object_for_cross_take >= 5,
            "cross-take protocol needs at least five takes per object",
        )
        fractions = (
            self.development_fraction,
            self.calibration_fraction,
            self.target_fraction,
        )
        _require(
            all(value > 0.0 for value in fractions) and np.isclose(sum(fractions), 1.0),
            "split fractions must be positive and sum to one",
        )
        _require(
            self.public_force_axis_index in {0, 1, 2},
            "force axis must select x, y, or z",
        )
        _require(
            self.public_force_threshold_n > 0.0, "force threshold must be positive"
        )
        _require(
            0.0
            < self.minimum_plausible_mesh_extent_mm
            < self.maximum_plausible_mesh_extent_mm,
            "mesh extent bounds are invalid",
        )
        _require(
            self.maximum_plausible_robot_translation_m > 0.0,
            "robot translation bound must be positive",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PokeFlexReadinessConfig:
        fields = cls.__dataclass_fields__
        unknown = set(value) - set(fields)
        _require(not unknown, f"unknown readiness config fields: {sorted(unknown)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class PokeFlexEpisode:
    """Metadata-only identity for one public object/take directory."""

    object_id: str
    take_id: str
    relative_take_path: str
    layout: str

    def __post_init__(self) -> None:
        _require(bool(self.object_id), "PokeFlex object id is empty")
        _require(bool(self.take_id), "PokeFlex take id is empty")
        relative = Path(self.relative_take_path)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            "PokeFlex relative take path is unsafe",
        )
        _require(self.layout in {"raw", "processed", "unknown"}, "unknown take layout")

    @property
    def episode_id(self) -> str:
        return f"{self.object_id}/{self.take_id}"


def discover_pokeflex_episodes(dataset_root: str | Path) -> tuple[PokeFlexEpisode, ...]:
    """Discover episodes using paths only, before any outcome is read."""

    root = Path(dataset_root).resolve()
    _require(root.is_dir(), "PokeFlex dataset root does not exist")
    episodes = []
    for robot_path in sorted(root.glob("*/*/robot_data.json")):
        take_root = robot_path.parent
        relative = take_root.relative_to(root)
        _require(
            len(relative.parts) >= 2, "take must be nested under an object directory"
        )
        episodes.append(
            PokeFlexEpisode(
                object_id=relative.parts[-2],
                take_id=relative.parts[-1],
                relative_take_path=relative.as_posix(),
                layout=_detect_layout(take_root),
            )
        )
    return tuple(episodes)


def readiness_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return _sha256_bytes(_canonical_bytes(canonical))


def load_readiness_config(path: str | Path) -> PokeFlexReadinessConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == POKEFLEX_READINESS_CONFIG_SCHEMA_VERSION,
        "unsupported PokeFlex readiness config schema",
    )
    _require(
        payload.get("config_sha256") == readiness_config_sha256(payload),
        "PokeFlex readiness config checksum mismatch",
    )
    if _CANONICAL_READINESS_CONFIG_SHA256:
        _require(
            payload["config_sha256"] == _CANONICAL_READINESS_CONFIG_SHA256,
            "PokeFlex readiness config differs from the canonical lock",
        )
    return PokeFlexReadinessConfig.from_mapping(payload["config"])


def _frame_id(path: Path) -> int | None:
    match = _MESH_FRAME_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def _even_sample(values: Sequence[Path], count: int) -> list[Path]:
    if len(values) <= count:
        return list(values)
    indices = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[int(index)] for index in np.unique(indices)]


def _resolve_obj_index(raw: int, vertex_count: int) -> int:
    if raw > 0:
        return raw - 1
    if raw < 0:
        return vertex_count + raw
    raise ValueError("OBJ indices are one-based and cannot be zero")


def _obj_summary(path: Path) -> dict[str, Any]:
    vertices: list[tuple[float, float, float]] = []
    raw_faces: list[tuple[int, ...]] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                _require(len(fields) >= 4, f"invalid OBJ vertex in {path.name}")
                vertex = tuple(float(value) for value in fields[1:4])
                _require(
                    all(np.isfinite(vertex)), f"non-finite OBJ vertex in {path.name}"
                )
                vertices.append(vertex)
            elif line.startswith("f "):
                fields = line.split()[1:]
                _require(len(fields) >= 3, f"invalid OBJ face in {path.name}")
                raw_faces.append(
                    tuple(int(field.split("/", maxsplit=1)[0]) for field in fields)
                )
    _require(vertices, f"OBJ contains no vertices: {path.name}")
    _require(raw_faces, f"OBJ contains no faces: {path.name}")
    faces = [
        tuple(_resolve_obj_index(index, len(vertices)) for index in face)
        for face in raw_faces
    ]
    _require(
        all(0 <= index < len(vertices) for face in faces for index in face),
        f"OBJ face index is out of range: {path.name}",
    )
    vertex_array = np.asarray(vertices, dtype=float)
    extent = np.max(vertex_array, axis=0) - np.min(vertex_array, axis=0)
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "frame": _frame_id(path),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "topology_sha256": _sha256_bytes(_canonical_bytes(faces)),
        "bounding_box_extent_raw": extent.tolist(),
        "bounding_box_diagonal_raw": float(np.linalg.norm(extent)),
    }


def _longest_consecutive_run(values: Sequence[int]) -> int:
    if not values:
        return 0
    longest = current = 1
    for previous, value in zip(values, values[1:], strict=False):
        if value == previous + 1:
            current += 1
            longest = max(longest, current)
        elif value != previous:
            current = 1
    return longest


def _mesh_inventory_digest(paths: Sequence[Path], take_root: Path) -> str:
    inventory = [
        {
            "path": path.relative_to(take_root).as_posix(),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return _sha256_bytes(_canonical_bytes(inventory))


def _material_identity_record(
    take_root: Path, mesh_inventory_sha256: str
) -> dict[str, Any]:
    path = take_root / "causal4d_material_identity.json"
    if not path.is_file():
        return {
            "status": "unverified_by_public_code",
            "verified": False,
            "reason": (
                "Per-frame meshes and faces are loaded independently; sampled topology "
                "consistency does not establish material vertex identity."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    verified = bool(
        payload.get("verified") is True
        and payload.get("vertex_ids_persist") is True
        and payload.get("mesh_inventory_sha256") == mesh_inventory_sha256
        and isinstance(payload.get("evidence"), str)
        and payload["evidence"]
    )
    return {
        "status": "verified_companion_artifact"
        if verified
        else "invalid_companion_artifact",
        "verified": verified,
        "artifact": {
            "path": path.name,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        },
    }


def _mesh_summary(
    take_root: Path,
    layout: str,
    config: PokeFlexReadinessConfig,
) -> dict[str, Any]:
    mesh_root = take_root / ("meshes" if layout == "raw" else "triangle_meshes")
    obj_paths = sorted(
        mesh_root.glob("*.obj"), key=lambda path: (_frame_id(path) or -1, path.name)
    )
    ply_count = len(list(mesh_root.glob("*.ply"))) if mesh_root.is_dir() else 0
    if not obj_paths:
        return {
            "mesh_directory": mesh_root.name,
            "obj_count": 0,
            "ply_count": ply_count,
            "schema_valid": False,
            "reason": "no OBJ meshes available for topology audit",
        }
    frame_ids = [_frame_id(path) for path in obj_paths]
    valid_frame_ids = [value for value in frame_ids if value is not None]
    sampled = [
        _obj_summary(path)
        for path in _even_sample(obj_paths, config.topology_sample_count)
    ]
    topology_pairs = {
        (item["vertex_count"], item["face_count"], item["topology_sha256"])
        for item in sampled
    }
    inventory_sha256 = _mesh_inventory_digest(obj_paths, take_root)
    diagonals = [item["bounding_box_diagonal_raw"] for item in sampled]
    unit_plausible = all(
        config.minimum_plausible_mesh_extent_mm
        <= value
        <= config.maximum_plausible_mesh_extent_mm
        for value in diagonals
    )
    return {
        "mesh_directory": mesh_root.name,
        "obj_count": len(obj_paths),
        "ply_count": ply_count,
        "frame_ids_complete": len(valid_frame_ids) == len(obj_paths),
        "first_frame": min(valid_frame_ids) if valid_frame_ids else None,
        "last_frame": max(valid_frame_ids) if valid_frame_ids else None,
        "longest_consecutive_frame_run": _longest_consecutive_run(
            sorted(valid_frame_ids)
        ),
        "mesh_inventory_sha256": inventory_sha256,
        "sampled_mesh_count": len(sampled),
        "sampled_meshes": sampled,
        "sampled_topology_consistent": len(topology_pairs) == 1,
        "expected_unit": config.expected_mesh_unit,
        "sampled_extent_unit_plausible": unit_plausible,
        "material_identity": _material_identity_record(take_root, inventory_sha256),
        "schema_valid": bool(valid_frame_ids and unit_plausible),
    }


def _finite_vector(value: Any, minimum_size: int) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(result) < minimum_size or not np.all(np.isfinite(result)):
        return None
    return result


def _valid_transform(value: Any) -> bool:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return False
    rotation = matrix[:3, :3]
    return bool(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=1e-4)
        and np.linalg.det(rotation) > 0.0
    )


def _first_key(record: Mapping[str, Any], candidates: Sequence[str]) -> str | None:
    return next((key for key in candidates if key in record), None)


def _nonnegative_frame_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return None


def _robot_summary(path: Path, config: PokeFlexReadinessConfig) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        return {
            "schema_valid": False,
            "reason": "robot_data.json is not a nonempty list",
        }
    timestamp_keys = ("timestamp_s", "timestamp", "time_s", "time")
    contact_keys = (
        "contact_location_world_m",
        "contact_location",
        "contact_point",
        "contact_position",
    )
    command_keys = (
        "commanded_pose_world",
        "commanded_pose",
        "command",
        "u_cmd",
    )
    frames = []
    forces = []
    valid_transforms = []
    timestamps = []
    explicit_contacts = []
    commands = []
    translations = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            continue
        frame = _nonnegative_frame_id(item.get("frame", index))
        if frame is not None:
            frames.append(frame)
        force = _finite_vector(item.get("forces"), 3)
        if force is not None:
            forces.append(force)
        transform = item.get("T_WT")
        valid_transforms.append(_valid_transform(transform))
        if valid_transforms[-1]:
            translations.append(np.asarray(transform, dtype=float)[:3, 3])
        timestamp_key = _first_key(item, timestamp_keys)
        if timestamp_key is not None:
            timestamp = _finite_vector([item[timestamp_key]], 1)
            if timestamp is not None:
                timestamps.append(float(timestamp[0]))
        contact_key = _first_key(item, contact_keys)
        explicit_contacts.append(
            contact_key is not None and _finite_vector(item[contact_key], 3) is not None
        )
        command_key = _first_key(item, command_keys)
        commands.append(command_key is not None)
    force_axis = config.public_force_axis_index
    force_contact_frames = [
        frame
        for frame, force in zip(frames, forces, strict=False)
        if len(force) > force_axis
        and force[force_axis] > config.public_force_threshold_n
    ]
    timestamp_monotonic = bool(
        len(timestamps) == len(payload)
        and all(
            second > first
            for first, second in zip(timestamps, timestamps[1:], strict=False)
        )
    )
    max_translation = max(
        (float(np.linalg.norm(value)) for value in translations), default=None
    )
    translation_plausible = bool(
        max_translation is not None
        and max_translation <= config.maximum_plausible_robot_translation_m
    )
    frame_unique = len(frames) == len(set(frames)) == len(payload)
    return {
        "artifact": {
            "path": path.name,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        },
        "record_count": len(payload),
        "frame_count": len(frames),
        "frames_unique": frame_unique,
        "first_frame": min(frames) if frames else None,
        "last_frame": max(frames) if frames else None,
        "force_vector_available": len(forces) == len(payload),
        "full_wrench_available": bool(
            forces and all(len(value) >= 6 for value in forces)
        ),
        "tool_transform_available": bool(valid_transforms and all(valid_transforms)),
        "timestamp_available": len(timestamps) == len(payload),
        "timestamp_strictly_monotonic": timestamp_monotonic,
        "explicit_contact_location_available": bool(
            explicit_contacts and all(explicit_contacts)
        ),
        "commanded_action_available": bool(commands and all(commands)),
        "force_contact_frame_count": len(force_contact_frames),
        "force_contact_first_frame": min(force_contact_frames)
        if force_contact_frames
        else None,
        "force_contact_last_frame": max(force_contact_frames)
        if force_contact_frames
        else None,
        "maximum_tool_translation_raw": max_translation,
        "expected_translation_unit": config.expected_robot_translation_unit,
        "translation_unit_plausible": translation_plausible,
        "schema_valid": bool(
            frame_unique
            and len(forces) == len(payload)
            and valid_transforms
            and all(valid_transforms)
            and translation_plausible
        ),
        "public_contact_frame_rule": {
            "force_axis_index": force_axis,
            "positive_threshold_n": config.public_force_threshold_n,
            "source": "pinned upstream dataset/preprocess.py",
            "ground_truth_contact_claimed": False,
        },
    }


def _matrix_shape_valid(value: Any, allowed: set[tuple[int, ...]]) -> bool:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(matrix.shape in allowed and np.all(np.isfinite(matrix)))


def _camera_parameter_valid(payload: Mapping[str, Any]) -> bool:
    intrinsic_keys = ("intrinsics", "color_intrinsics", "depth_intrinsics")
    extrinsic_keys = ("extrinsics", "color_extrinsics", "depth_extrinsics")
    intrinsic_valid = any(
        key in payload and _matrix_shape_valid(payload[key], {(3, 3)})
        for key in intrinsic_keys
    )
    extrinsic_valid = any(
        key in payload
        and _matrix_shape_valid(payload[key], {(3, 4), (4, 4), (1, 4, 4)})
        for key in extrinsic_keys
    )
    return intrinsic_valid and extrinsic_valid


def _camera_summary(take_root: Path, layout: str) -> dict[str, Any]:
    cameras = []
    if layout == "raw":
        parameter_paths = sorted(
            path
            for family in ("volucam", "kinect", "realsense")
            for path in (take_root / family).glob("*/camera_parameters.json")
        )
        for path in parameter_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            camera_root = path.parent
            cameras.append(
                {
                    "camera_id": camera_root.relative_to(take_root).as_posix(),
                    "parameter_sha256": _sha256_file(path),
                    "calibration_valid": isinstance(payload, Mapping)
                    and _camera_parameter_valid(payload),
                    "color_file_count": len(list((camera_root / "color").glob("*"))),
                    "depth_file_count": len(list((camera_root / "depth").glob("*"))),
                }
            )
    else:
        path = take_root / "images" / "camera_parameters.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                for camera_id, parameters in sorted(payload.items()):
                    cameras.append(
                        {
                            "camera_id": str(camera_id),
                            "parameter_sha256": _sha256_file(path),
                            "calibration_valid": isinstance(parameters, Mapping)
                            and _camera_parameter_valid(parameters),
                            "color_file_count": 0,
                            "depth_file_count": 0,
                        }
                    )
    return {
        "camera_count": len(cameras),
        "calibrated_camera_count": sum(
            camera["calibration_valid"] for camera in cameras
        ),
        "cameras": cameras,
    }


def _detect_layout(take_root: Path) -> str:
    if (take_root / "meshes").is_dir():
        return "raw"
    if (take_root / "triangle_meshes").is_dir():
        return "processed"
    return "unknown"


def _take_summary(
    dataset_root: Path,
    episode: PokeFlexEpisode,
    config: PokeFlexReadinessConfig,
) -> dict[str, Any]:
    take_root = dataset_root / episode.relative_take_path
    object_id = episode.object_id
    take_id = episode.take_id
    episode_id = episode.episode_id
    layout = episode.layout
    robot_path = take_root / "robot_data.json"
    try:
        robot = (
            _robot_summary(robot_path, config)
            if robot_path.is_file()
            else {"schema_valid": False, "reason": "robot_data.json is missing"}
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        robot = {"schema_valid": False, "reason": f"robot audit failed: {error}"}
    try:
        mesh = (
            _mesh_summary(take_root, layout, config)
            if layout != "unknown"
            else {"schema_valid": False, "reason": "mesh directory is missing"}
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        mesh = {"schema_valid": False, "reason": f"mesh audit failed: {error}"}
    try:
        cameras = (
            _camera_summary(take_root, layout)
            if layout != "unknown"
            else {
                "camera_count": 0,
                "calibrated_camera_count": 0,
                "cameras": [],
            }
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        cameras = {
            "camera_count": 0,
            "calibrated_camera_count": 0,
            "cameras": [],
            "reason": f"camera audit failed: {error}",
        }
    mesh_frames = {
        item["frame"]
        for item in mesh.get("sampled_meshes", [])
        if item["frame"] is not None
    }
    first_robot = robot.get("first_frame")
    last_robot = robot.get("last_frame")
    sampled_frames_within_robot = bool(
        mesh_frames
        and first_robot is not None
        and last_robot is not None
        and all(first_robot <= frame <= last_robot for frame in mesh_frames)
    )
    required_run = config.prefix_frame_count + config.minimum_future_frame_count
    sequence_ready = mesh.get("longest_consecutive_frame_run", 0) >= required_run
    camera_ready = (
        cameras["calibrated_camera_count"] >= config.minimum_calibrated_camera_count
    )
    geometry_ready = bool(
        robot.get("schema_valid")
        and mesh.get("schema_valid")
        and sampled_frames_within_robot
        and sequence_ready
        and camera_ready
    )
    return {
        "episode_id": episode_id,
        "object_id": object_id,
        "take_id": take_id,
        "layout": layout,
        "robot": robot,
        "mesh": mesh,
        "cameras": cameras,
        "frame_alignment": {
            "sampled_mesh_frames_within_robot_range": sampled_frames_within_robot,
            "required_consecutive_mesh_frames": required_run,
            "sequence_ready": sequence_ready,
        },
        "capabilities": {
            "geometry_observation_ready": geometry_ready,
            "measured_intervention_ready": bool(
                robot.get("force_vector_available")
                and robot.get("tool_transform_available")
            ),
            "explicit_contact_location_ready": bool(
                robot.get("explicit_contact_location_available")
            ),
            "contact_candidate_from_pose_wrench_ready": bool(
                robot.get("force_vector_available")
                and robot.get("tool_transform_available")
            ),
            "delay_inference_ready": bool(
                robot.get("timestamp_available")
                and robot.get("timestamp_strictly_monotonic")
            ),
            "command_vs_measured_ready": bool(robot.get("commanded_action_available")),
            "identity_dependent_track_metric_ready": bool(
                mesh.get("material_identity", {}).get("verified")
            ),
            "sampled_topology_consistent": bool(
                mesh.get("sampled_topology_consistent")
            ),
        },
        "eligible_for_metadata_split": geometry_ready,
    }


def _split_rank(config: PokeFlexReadinessConfig, episode_id: str) -> str:
    return _sha256_bytes(f"{config.split_seed}:{episode_id}".encode("utf-8"))


def assign_metadata_only_splits(
    takes: Sequence[Mapping[str, Any]],
    config: PokeFlexReadinessConfig,
) -> dict[str, Any]:
    assignments = []
    readiness_by_object = {}
    object_ids = sorted({str(take["object_id"]) for take in takes})
    for object_id in object_ids:
        candidates = [
            take
            for take in takes
            if take["object_id"] == object_id and take["eligible_for_metadata_split"]
        ]
        ranked = sorted(
            candidates, key=lambda take: _split_rank(config, take["episode_id"])
        )
        enough = len(ranked) >= config.minimum_takes_per_object_for_cross_take
        if enough:
            calibration_count = max(
                1, int(round(len(ranked) * config.calibration_fraction))
            )
            target_count = max(1, int(round(len(ranked) * config.target_fraction)))
            development_count = len(ranked) - calibration_count - target_count
            if development_count < 1:
                development_count = 1
                target_count = len(ranked) - calibration_count - development_count
            boundaries = (
                development_count,
                development_count + calibration_count,
            )
        else:
            boundaries = (len(ranked), len(ranked))
        counts = {"development": 0, "calibration": 0, "target": 0}
        for rank, take in enumerate(ranked):
            split = (
                "development"
                if rank < boundaries[0]
                else "calibration"
                if rank < boundaries[1]
                else "target"
            )
            counts[split] += 1
            assignments.append(
                {
                    "episode_id": take["episode_id"],
                    "object_id": object_id,
                    "take_id": take["take_id"],
                    "split": split,
                    "metadata_rank_sha256": _split_rank(config, take["episode_id"]),
                }
            )
        readiness_by_object[object_id] = {
            "eligible_take_count": len(ranked),
            "minimum_required": config.minimum_takes_per_object_for_cross_take,
            "cross_take_ready": enough
            and counts["development"] > 0
            and counts["calibration"] > 0
            and counts["target"] > 0,
            "split_counts": counts,
        }
    assignments.sort(key=lambda value: value["episode_id"])
    split_counts = {
        split: sum(value["split"] == split for value in assignments)
        for split in ("development", "calibration", "target")
    }
    return {
        "method": "per-object SHA-256 rank from metadata-only object/take ids",
        "seed": config.split_seed,
        "outcome_metrics_used": False,
        "assignments": assignments,
        "split_counts": split_counts,
        "objects": readiness_by_object,
        "cross_take_ready": any(
            value["cross_take_ready"] for value in readiness_by_object.values()
        ),
        "nominal_90_session_conformal_ready": split_counts["calibration"] >= 9,
    }


def preflight_result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return _sha256_bytes(_canonical_bytes(canonical))


def preflight_pokeflex_dataset(
    dataset_root: str | Path,
    config: PokeFlexReadinessConfig | None = None,
) -> dict[str, Any]:
    """Audit PokeFlex compatibility without fitting or evaluating a model."""

    cfg = config or PokeFlexReadinessConfig()
    root = Path(dataset_root).resolve()
    _require(root.is_dir(), "PokeFlex dataset root does not exist")
    episodes = discover_pokeflex_episodes(root)
    if not episodes and (root / "robot_data.json").is_file():
        raise ValueError("dataset root must include object/take directory levels")
    takes = [_take_summary(root, episode, cfg) for episode in episodes]
    split = assign_metadata_only_splits(takes, cfg)
    eligible = [take for take in takes if take["eligible_for_metadata_split"]]
    inventory = [
        {
            "episode_id": take["episode_id"],
            "layout": take["layout"],
            "robot_sha256": take["robot"].get("artifact", {}).get("sha256"),
            "mesh_inventory_sha256": take["mesh"].get("mesh_inventory_sha256"),
        }
        for take in takes
    ]
    capability_gates = {
        "adapter_schema_ready": bool(eligible),
        "factual_geometry_continuation_ready": any(
            take["capabilities"]["geometry_observation_ready"]
            and take["capabilities"]["measured_intervention_ready"]
            for take in takes
        ),
        "cross_take_interventional_evaluation_ready": split["cross_take_ready"],
        "explicit_contact_abduction_ready": any(
            take["capabilities"]["explicit_contact_location_ready"] for take in takes
        ),
        "pose_wrench_contact_candidate_ready": any(
            take["capabilities"]["contact_candidate_from_pose_wrench_ready"]
            for take in takes
        ),
        "delay_inference_ready": any(
            take["capabilities"]["delay_inference_ready"] for take in takes
        ),
        "command_vs_measured_separation_ready": any(
            take["capabilities"]["command_vs_measured_ready"] for take in takes
        ),
        "identity_dependent_track_metric_ready": any(
            take["capabilities"]["identity_dependent_track_metric_ready"]
            for take in takes
        ),
        "nominal_90_session_conformal_ready": split[
            "nominal_90_session_conformal_ready"
        ],
    }
    result: dict[str, Any] = {
        "schema_version": POKEFLEX_PREFLIGHT_SCHEMA_VERSION,
        "artifact_kind": "PublicPokeFlexPreflight",
        "protocol_id": cfg.protocol_id,
        "upstream": {
            "repository": cfg.upstream_repository,
            "commit": cfg.upstream_commit,
            "terms_url": POKEFLEX_TERMS_URL,
        },
        "information_boundary": {
            "directory_and_schema_metadata_read": True,
            "mesh_geometry_sampled_for_topology_and_units": True,
            "prediction_metrics_computed": False,
            "model_parameters_fitted": False,
            "outcome_based_take_selection": False,
            "target_split_selected_from_metadata_hash_only": True,
        },
        "dataset_inventory": {
            "object_count": len({take["object_id"] for take in takes}),
            "take_count": len(takes),
            "eligible_take_count": len(eligible),
            "inventory_sha256": _sha256_bytes(_canonical_bytes(inventory)),
            "raw_data_embedded": False,
        },
        "takes": takes,
        "metadata_only_split": split,
        "capability_gates": capability_gates,
        "metric_contract": {
            "geometry_metrics_allowed": ["symmetric_chamfer", "point_to_surface"],
            "identity_metrics_allowed_only_when_verified": [
                "material_track_error",
                "per_vertex_state_error",
            ],
            "individual_counterfactual_ground_truth_claim_allowed": False,
            "session_calibration_claim_allowed": capability_gates[
                "nominal_90_session_conformal_ready"
            ],
        },
        "claim_boundary": (
            "Public-data readiness and held-out interventional prediction only. "
            "No physical registration, individual counterfactual ground truth, "
            "material correspondence, or calibration claim is implied by schema checks."
        ),
        "preflight_passed": bool(
            capability_gates["adapter_schema_ready"]
            and capability_gates["factual_geometry_continuation_ready"]
        ),
    }
    result["result_sha256"] = preflight_result_sha256(result)
    return result


def validate_preflight_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == 1, "unsupported preflight schema")
    _require(
        payload.get("artifact_kind") == "PublicPokeFlexPreflight",
        "unexpected preflight artifact kind",
    )
    _require(
        payload.get("result_sha256") == preflight_result_sha256(payload),
        "preflight result checksum mismatch",
    )
    _require(
        payload["information_boundary"]["prediction_metrics_computed"] is False
        and payload["information_boundary"]["model_parameters_fitted"] is False
        and payload["metadata_only_split"]["outcome_metrics_used"] is False,
        "preflight crossed the outcome-free information boundary",
    )
    return {
        "passed": True,
        "preflight_passed": bool(payload["preflight_passed"]),
        "result_sha256": payload["result_sha256"],
        "take_count": payload["dataset_inventory"]["take_count"],
    }


def write_preflight_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _fixture_mesh(frame: int, mutate_topology: bool) -> str:
    displacement = 0.5 * np.sin(frame / 4.0)
    vertices = [
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0 + displacement),
    ]
    faces = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    if mutate_topology:
        vertices.append((50.0, 50.0, 50.0))
        faces.append((1, 2, 5))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines.extend("f " + " ".join(str(index) for index in face) for face in faces)
    return "\n".join(lines) + "\n"


def _fixture_camera_parameters(camera_index: int) -> dict[str, Any]:
    transform = np.eye(4)
    transform[0, 3] = 0.1 * camera_index
    return {
        "intrinsics": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "extrinsics": transform.tolist(),
    }


def write_synthetic_pokeflex_fixture(
    root: str | Path,
    *,
    object_count: int = 1,
    takes_per_object: int = 5,
    frame_count: int = 16,
    mutate_last_mesh_topology: bool = False,
    include_material_identity: bool = False,
) -> dict[str, Any]:
    """Write a tiny public-schema fixture containing no real PokeFlex data."""

    _require(object_count >= 1, "fixture needs at least one object")
    _require(takes_per_object >= 1, "fixture needs at least one take")
    _require(frame_count >= 12, "fixture needs at least twelve frames")
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    episode_ids = []
    for object_index in range(object_count):
        object_id = f"fixture_plush_{object_index:02d}"
        for take_index in range(takes_per_object):
            take_id = f"poke_{take_index:03d}"
            take_root = output / object_id / take_id
            mesh_root = take_root / "meshes"
            mesh_root.mkdir(parents=True, exist_ok=True)
            robot_records = []
            for frame in range(1, frame_count + 1):
                mutate = bool(
                    mutate_last_mesh_topology
                    and object_index == object_count - 1
                    and take_index == takes_per_object - 1
                    and frame == frame_count
                )
                (mesh_root / f"mesh-f{frame:05d}.obj").write_text(
                    _fixture_mesh(frame, mutate), encoding="utf-8"
                )
                transform = np.eye(4)
                transform[:3, 3] = [
                    0.01 * take_index,
                    0.0,
                    0.25 - 0.001 * frame,
                ]
                force = 4.0 if 4 <= frame <= frame_count - 2 else 0.0
                robot_records.append(
                    {
                        "frame": frame,
                        "timestamp_s": frame / 30.0,
                        "forces": [0.0, force, 0.0, 0.0, 0.0, 0.0],
                        "T_WT": transform.tolist(),
                        "contact_location_world_m": transform[:3, 3].tolist(),
                        "commanded_pose_world": transform.tolist(),
                    }
                )
            (take_root / "robot_data.json").write_text(
                json.dumps(robot_records, indent=2) + "\n", encoding="utf-8"
            )
            for camera_index in range(3):
                camera_root = take_root / "volucam" / str(camera_index)
                (camera_root / "color").mkdir(parents=True, exist_ok=True)
                (camera_root / "depth").mkdir(parents=True, exist_ok=True)
                (camera_root / "camera_parameters.json").write_text(
                    json.dumps(_fixture_camera_parameters(camera_index), indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                (camera_root / "color" / "00001.png").write_bytes(b"fixture")
                (camera_root / "depth" / "00001.png").write_bytes(b"fixture")
            if include_material_identity:
                obj_paths = sorted(mesh_root.glob("*.obj"))
                inventory_sha256 = _mesh_inventory_digest(obj_paths, take_root)
                (take_root / "causal4d_material_identity.json").write_text(
                    json.dumps(
                        {
                            "verified": True,
                            "vertex_ids_persist": True,
                            "mesh_inventory_sha256": inventory_sha256,
                            "evidence": "synthetic fixture construction",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            episode_ids.append(f"{object_id}/{take_id}")
    descriptor = {
        "artifact_kind": "SyntheticPokeFlexFixture",
        "contains_real_pokeflex_data": False,
        "object_count": object_count,
        "takes_per_object": takes_per_object,
        "frame_count": frame_count,
        "mutate_last_mesh_topology": mutate_last_mesh_topology,
        "include_material_identity": include_material_identity,
        "episode_ids": episode_ids,
    }
    descriptor["descriptor_sha256"] = _sha256_bytes(_canonical_bytes(descriptor))
    (output / "fixture_manifest.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return descriptor
