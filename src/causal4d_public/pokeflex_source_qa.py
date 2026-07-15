"""Source-only geometry and intervention QA for the locked PokeFlex pilot."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d_public.pokeflex import (
    PINNED_POKEFLEX_COMMIT,
    validate_preflight_result,
)


POKEFLEX_SOURCE_QA_POLICY_SCHEMA_VERSION = 1
POKEFLEX_SOURCE_QA_ARTIFACT_SCHEMA_VERSION = 1
POKEFLEX_SOURCE_QA_POLICY_ID = "causal4d-pokeflex-source-qa-v1"
CANONICAL_POKEFLEX_SOURCE_QA_POLICY_SHA256 = (
    "6abf69df5299f997caa223c19778c500dfe407d28b78eb068bb567b7f0b56836"
)
_MESH_FRAME_PATTERN = re.compile(r"mesh-f(\d{5})\.obj$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PokeFlexSourceQaConfig:
    policy_id: str = POKEFLEX_SOURCE_QA_POLICY_ID
    upstream_commit: str = PINNED_POKEFLEX_COMMIT
    readiness_config_sha256: str = (
        "256f6c0585a1eb592583b0a0c017e116baed9126f12119e80f866cd174b58070"
    )
    expected_preflight_result_sha256: str | None = None
    expected_object_id: str | None = None
    expected_development_take_ids: tuple[str, ...] = ()
    prefix_frame_count: int = 6
    minimum_future_frame_count: int = 6
    force_axis_index: int = 1
    force_threshold_n: float = 3.0
    contact_proximity_mm: float = 20.0
    minimum_force_contact_frame_count: int = 6
    minimum_force_proximity_agreement_fraction: float = 0.75
    maximum_minimum_active_tool_surface_distance_mm: float = 5.0
    minimum_closed_edge_fraction: float = 0.95
    maximum_nonmanifold_edge_count: int = 0
    maximum_shared_graph_rigid_chamfer_mm: float = 5.0
    icp_max_points: int = 3000
    icp_iterations: int = 20
    icp_trim_quantile: float = 0.90

    def __post_init__(self) -> None:
        _require(self.policy_id == POKEFLEX_SOURCE_QA_POLICY_ID, "policy id changed")
        _require(
            self.upstream_commit == PINNED_POKEFLEX_COMMIT,
            "PokeFlex upstream commit changed",
        )
        _require(self.prefix_frame_count >= 2, "prefix is too short")
        _require(self.minimum_future_frame_count >= 2, "future is too short")
        _require(self.force_axis_index in {0, 1, 2}, "force axis is invalid")
        _require(self.force_threshold_n > 0.0, "force threshold must be positive")
        _require(self.contact_proximity_mm > 0.0, "contact distance must be positive")
        _require(
            self.minimum_force_contact_frame_count >= 1,
            "contact-frame gate must be positive",
        )
        _require(
            0.0 < self.minimum_force_proximity_agreement_fraction <= 1.0,
            "force/proximity agreement gate is invalid",
        )
        _require(
            0.0 < self.minimum_closed_edge_fraction <= 1.0,
            "closed-edge gate is invalid",
        )
        _require(
            self.maximum_nonmanifold_edge_count >= 0,
            "nonmanifold-edge gate is invalid",
        )
        _require(self.icp_max_points >= 100, "ICP support is too small")
        _require(self.icp_iterations >= 1, "ICP iterations must be positive")
        _require(0.5 <= self.icp_trim_quantile <= 1.0, "ICP trim is invalid")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PokeFlexSourceQaConfig:
        fields = cls.__dataclass_fields__
        unknown = set(value) - set(fields)
        _require(not unknown, f"unknown source-QA fields: {sorted(unknown)}")
        payload = dict(value)
        if "expected_development_take_ids" in payload:
            payload["expected_development_take_ids"] = tuple(
                map(str, payload["expected_development_take_ids"])
            )
        return cls(**payload)


def source_qa_policy_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def validate_source_qa_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == POKEFLEX_SOURCE_QA_POLICY_SCHEMA_VERSION,
        "unsupported PokeFlex source-QA policy schema",
    )
    _require(
        payload.get("artifact_kind") == "PublicPokeFlexSourceQaPolicy",
        "unexpected PokeFlex source-QA policy kind",
    )
    observed = source_qa_policy_sha256(payload)
    _require(payload.get("config_sha256") == observed, "source-QA checksum mismatch")
    if CANONICAL_POKEFLEX_SOURCE_QA_POLICY_SHA256:
        _require(
            observed == CANONICAL_POKEFLEX_SOURCE_QA_POLICY_SHA256,
            "source-QA policy differs from the canonical lock",
        )
    config = PokeFlexSourceQaConfig.from_mapping(payload["config"])
    boundary = payload.get("information_boundary")
    _require(
        boundary
        == {
            "source_meshes_and_robot_records_only": True,
            "calibration_take_data_allowed": False,
            "target_take_data_allowed": False,
            "prediction_metrics_allowed": False,
            "model_fitting_allowed": False,
        },
        "source-QA information boundary changed",
    )
    return {"passed": True, "config_sha256": observed, "config": config}


def load_source_qa_policy(path: str | Path) -> PokeFlexSourceQaConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_source_qa_policy(payload)["config"]


def _mesh_frame(path: Path) -> int | None:
    match = _MESH_FRAME_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def _obj_geometry(
    path: Path, *, include_faces: bool = False
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                _require(len(fields) >= 4, f"invalid OBJ vertex in {path.name}")
                vertices.append(tuple(map(float, fields[1:4])))
            elif include_faces and line.startswith("f "):
                raw = [int(value.split("/", maxsplit=1)[0]) for value in line.split()[1:]]
                _require(len(raw) >= 3, f"invalid OBJ face in {path.name}")
                faces.append(
                    tuple(value - 1 if value > 0 else len(vertices) + value for value in raw)
                )
    vertex_array = np.asarray(vertices, dtype=np.float64)
    _require(
        vertex_array.ndim == 2 and vertex_array.shape[1] == 3 and len(vertex_array),
        f"OBJ contains no vertices: {path.name}",
    )
    _require(np.all(np.isfinite(vertex_array)), f"OBJ has non-finite vertices: {path.name}")
    if include_faces:
        _require(faces, f"OBJ contains no faces: {path.name}")
        _require(
            all(0 <= index < len(vertex_array) for face in faces for index in face),
            f"OBJ face index is out of range: {path.name}",
        )
    return vertex_array, tuple(faces)


def _edge_summary(faces: Sequence[Sequence[int]]) -> dict[str, Any]:
    counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in zip(face, (*face[1:], face[0]), strict=True):
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    _require(bool(counts), "mesh contains no edges")
    boundary = sum(value == 1 for value in counts.values())
    nonmanifold = sum(value > 2 for value in counts.values())
    closed = sum(value == 2 for value in counts.values())
    return {
        "edge_count": len(counts),
        "boundary_edge_count": boundary,
        "nonmanifold_edge_count": nonmanifold,
        "closed_edge_fraction": float(closed / len(counts)),
    }


def _valid_transform(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (4, 4), "tool transform must be 4 x 4")
    _require(np.all(np.isfinite(matrix)), "tool transform is non-finite")
    rotation = matrix[:3, :3]
    _require(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0),
        "tool transform has an invalid homogeneous row",
    )
    _require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=1e-4)
        and np.linalg.det(rotation) > 0.0,
        "tool transform has an invalid rotation",
    )
    return matrix


def _mesh_inventory_sha256(paths: Sequence[Path], take_root: Path) -> str:
    inventory = [
        {
            "path": path.relative_to(take_root).as_posix(),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return hashlib.sha256(_canonical_bytes(inventory)).hexdigest()


def audit_pokeflex_source_take(
    take_root: str | Path,
    config: PokeFlexSourceQaConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Audit one source take and return its first-frame surface for cross-take QA."""

    cfg = config or PokeFlexSourceQaConfig()
    root = Path(take_root).resolve()
    robot_path = root / "robot_data.json"
    mesh_root = root / "meshes"
    _require(robot_path.is_file(), "source take has no robot_data.json")
    _require(mesh_root.is_dir(), "source take has no meshes directory")
    robot_payload = json.loads(robot_path.read_text(encoding="utf-8"))
    _require(isinstance(robot_payload, list) and robot_payload, "robot log is empty")
    robot_by_frame: dict[int, Mapping[str, Any]] = {}
    for item in robot_payload:
        _require(isinstance(item, Mapping), "robot record is not an object")
        value = item.get("frame")
        _require(
            isinstance(value, (int, str)) and not isinstance(value, bool),
            "robot frame id is invalid",
        )
        frame = int(value)
        _require(frame >= 0 and frame not in robot_by_frame, "robot frame ids repeat")
        robot_by_frame[frame] = item
    mesh_paths = sorted(
        (path for path in mesh_root.glob("mesh-f*.obj") if _mesh_frame(path) is not None),
        key=lambda path: int(_mesh_frame(path) or -1),
    )
    mesh_by_frame = {int(_mesh_frame(path) or -1): path for path in mesh_paths}
    frames = sorted(robot_by_frame)
    _require(frames == sorted(mesh_by_frame), "robot and mesh frames are not aligned")
    _require(
        len(frames) >= cfg.prefix_frame_count + cfg.minimum_future_frame_count,
        "source take is too short",
    )

    distances_mm = []
    force_values_n = []
    vertex_counts = []
    first_vertices: np.ndarray | None = None
    first_faces: tuple[tuple[int, ...], ...] = ()
    for index, frame in enumerate(frames):
        vertices, faces = _obj_geometry(
            mesh_by_frame[frame], include_faces=index == 0
        )
        if index == 0:
            first_vertices = vertices
            first_faces = faces
        item = robot_by_frame[frame]
        transform = _valid_transform(item.get("T_WT"))
        force = np.asarray(item.get("forces"), dtype=np.float64).reshape(-1)
        _require(
            len(force) > cfg.force_axis_index and np.all(np.isfinite(force)),
            "robot force vector is invalid",
        )
        tool_mm = 1000.0 * transform[:3, 3]
        distances_mm.append(float(np.min(np.linalg.norm(vertices - tool_mm, axis=1))))
        force_values_n.append(float(force[cfg.force_axis_index]))
        vertex_counts.append(len(vertices))
    _require(first_vertices is not None, "source take has no first mesh")

    distance = np.asarray(distances_mm, dtype=np.float64)
    force_axis = np.asarray(force_values_n, dtype=np.float64)
    active = force_axis > cfg.force_threshold_n
    close = distance <= cfg.contact_proximity_mm
    active_count = int(np.sum(active))
    agreement = float(np.sum(active & close) / active_count) if active_count else 0.0
    minimum_active_distance = float(np.min(distance[active])) if active_count else None
    edge = _edge_summary(first_faces)
    contact_ready = bool(
        active_count >= cfg.minimum_force_contact_frame_count
        and agreement >= cfg.minimum_force_proximity_agreement_fraction
        and minimum_active_distance is not None
        and minimum_active_distance
        <= cfg.maximum_minimum_active_tool_surface_distance_mm
    )
    mesh_ready = bool(
        edge["closed_edge_fraction"] >= cfg.minimum_closed_edge_fraction
        and edge["nonmanifold_edge_count"]
        <= cfg.maximum_nonmanifold_edge_count
    )
    record = {
        "take_id": root.name,
        "frame_count": len(frames),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "robot_sha256": _sha256_file(robot_path),
        "first_mesh_sha256": _sha256_file(mesh_by_frame[frames[0]]),
        "mesh_inventory_sha256": _mesh_inventory_sha256(mesh_paths, root),
        "vertex_count_min": int(min(vertex_counts)),
        "vertex_count_max": int(max(vertex_counts)),
        "vertex_count_constant": len(set(vertex_counts)) == 1,
        "first_mesh": {
            "vertex_count": len(first_vertices),
            "face_count": len(first_faces),
            "centroid_mm": first_vertices.mean(axis=0).tolist(),
            "extent_mm": np.ptp(first_vertices, axis=0).tolist(),
            **edge,
        },
        "contact_alignment": {
            "force_axis_index": cfg.force_axis_index,
            "force_threshold_n": cfg.force_threshold_n,
            "proximity_threshold_mm": cfg.contact_proximity_mm,
            "force_active_frame_count": active_count,
            "force_active_proximity_agreement_fraction": agreement,
            "minimum_active_tool_surface_distance_mm": minimum_active_distance,
            "median_active_tool_surface_distance_mm": float(np.median(distance[active]))
            if active_count
            else None,
            "median_inactive_tool_surface_distance_mm": float(
                np.median(distance[~active])
            )
            if np.any(~active)
            else None,
            "passed": contact_ready,
        },
        "gates": {
            "frame_alignment_ready": True,
            "surface_graph_geometry_ready": mesh_ready,
            "pose_wrench_contact_candidate_ready": contact_ready,
            "take_specific_graph_backend_ready": mesh_ready and contact_ready,
            "material_vertex_identity_ready": False,
        },
    }
    return record, first_vertices


def _scipy_tree(points: np.ndarray):
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - optional integration
        raise RuntimeError("PokeFlex source QA requires scipy") from error
    return cKDTree(points)


def _sample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _symmetric_chamfer_mm(source: np.ndarray, target: np.ndarray) -> float:
    source_to_target = _scipy_tree(target).query(source, k=1)[0]
    target_to_source = _scipy_tree(source).query(target, k=1)[0]
    return float(0.5 * (np.mean(source_to_target) + np.mean(target_to_source)))


def _rigid_icp_summary(
    source: np.ndarray,
    target: np.ndarray,
    config: PokeFlexSourceQaConfig,
) -> dict[str, float]:
    source_points = _sample_points(source, config.icp_max_points)
    target_points = _sample_points(target, config.icp_max_points)
    source_points = source_points - source_points.mean(axis=0)
    target_points = target_points - target_points.mean(axis=0)
    translated_score = _symmetric_chamfer_mm(source_points, target_points)
    transformed = source_points.copy()
    total_rotation = np.eye(3)
    target_tree = _scipy_tree(target_points)
    for _ in range(config.icp_iterations):
        distances, indices = target_tree.query(transformed, k=1)
        cutoff = float(np.quantile(distances, config.icp_trim_quantile))
        keep = distances <= cutoff
        rotation, translation = _kabsch(
            transformed[keep], target_points[indices[keep]]
        )
        transformed = transformed @ rotation.T + translation
        total_rotation = rotation @ total_rotation
    cosine = np.clip((np.trace(total_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "translation_aligned_chamfer_mm": translated_score,
        "rigid_icp_chamfer_mm": _symmetric_chamfer_mm(
            transformed, target_points
        ),
        "rigid_icp_rotation_degrees": float(np.degrees(np.arccos(cosine))),
    }


def source_qa_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def run_pokeflex_source_qa(
    dataset_root: str | Path,
    preflight: Mapping[str, Any],
    config: PokeFlexSourceQaConfig | None = None,
) -> dict[str, Any]:
    """Audit only metadata-assigned development takes; never open other takes."""

    cfg = config or PokeFlexSourceQaConfig()
    validation = validate_preflight_result(preflight)
    observed_preflight_hash = validation["result_sha256"]
    if cfg.expected_preflight_result_sha256 is not None:
        _require(
            observed_preflight_hash == cfg.expected_preflight_result_sha256,
            "source QA received an unexpected preflight artifact",
        )
    assignments = preflight["metadata_only_split"]["assignments"]
    development = sorted(
        (
            assignment
            for assignment in assignments
            if assignment["split"] == "development"
        ),
        key=lambda assignment: str(assignment["take_id"]),
    )
    development_ids = tuple(str(value["take_id"]) for value in development)
    if cfg.expected_development_take_ids:
        _require(
            development_ids == tuple(sorted(cfg.expected_development_take_ids)),
            "development take set differs from the source-QA lock",
        )
    _require(bool(development), "preflight contains no development takes")
    object_ids = {str(value["object_id"]) for value in development}
    _require(len(object_ids) == 1, "source QA currently requires one object")
    object_id = next(iter(object_ids))
    if cfg.expected_object_id is not None:
        _require(object_id == cfg.expected_object_id, "source-QA object changed")

    root = Path(dataset_root).resolve()
    take_records = []
    first_surfaces: dict[str, np.ndarray] = {}
    for assignment in development:
        take_id = str(assignment["take_id"])
        take_root = root / object_id / take_id
        record, first_surface = audit_pokeflex_source_take(take_root, cfg)
        take_records.append(record)
        first_surfaces[take_id] = first_surface

    pairwise = []
    for index, first in enumerate(development_ids):
        for second in development_ids[index + 1 :]:
            pairwise.append(
                {
                    "first_take_id": first,
                    "second_take_id": second,
                    **_rigid_icp_summary(
                        first_surfaces[first], first_surfaces[second], cfg
                    ),
                }
            )
    maximum_icp = max(
        (record["rigid_icp_chamfer_mm"] for record in pairwise), default=0.0
    )
    take_specific_ready = all(
        record["gates"]["take_specific_graph_backend_ready"]
        for record in take_records
    )
    all_take_ids = {str(value["take_id"]) for value in assignments}
    unopened = sorted(all_take_ids - set(development_ids))
    result: dict[str, Any] = {
        "schema_version": POKEFLEX_SOURCE_QA_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "PublicPokeFlexSourceQa",
        "policy_id": cfg.policy_id,
        "upstream_commit": cfg.upstream_commit,
        "preflight_result_sha256": observed_preflight_hash,
        "information_boundary": {
            "opened_take_ids": list(development_ids),
            "unopened_take_ids": unopened,
            "development_data_only": True,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "prediction_metrics_computed": False,
            "model_parameters_fitted": False,
            "raw_data_embedded": False,
        },
        "object_id": object_id,
        "takes": take_records,
        "cross_take_first_frame_alignment": {
            "pair_count": len(pairwise),
            "pairs": pairwise,
            "maximum_rigid_icp_chamfer_mm": float(maximum_icp),
            "single_canonical_graph_threshold_mm": (
                cfg.maximum_shared_graph_rigid_chamfer_mm
            ),
        },
        "capability_gates": {
            "source_take_specific_graph_backend_ready": take_specific_ready,
            "single_shared_canonical_graph_ready": bool(
                pairwise
                and maximum_icp <= cfg.maximum_shared_graph_rigid_chamfer_mm
            ),
            "shared_parameters_across_take_specific_graphs_ready": take_specific_ready,
            "pose_wrench_contact_candidate_ready": all(
                record["gates"]["pose_wrench_contact_candidate_ready"]
                for record in take_records
            ),
            "material_identity_metrics_ready": False,
        },
        "source_qa_passed": take_specific_ready,
        "claim_boundary": (
            "Development-take geometry and pose/wrench compatibility only. "
            "No prediction, parameter fit, material identity, calibration, or "
            "target result is established."
        ),
    }
    result["result_sha256"] = source_qa_artifact_sha256(result)
    return result


def validate_source_qa_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == POKEFLEX_SOURCE_QA_ARTIFACT_SCHEMA_VERSION,
        "unsupported PokeFlex source-QA artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "PublicPokeFlexSourceQa",
        "unexpected PokeFlex source-QA artifact kind",
    )
    _require(
        payload.get("result_sha256") == source_qa_artifact_sha256(payload),
        "PokeFlex source-QA artifact checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(boundary.get("development_data_only") is True, "source boundary changed")
    _require(boundary.get("calibration_take_data_read") is False, "calibration opened")
    _require(boundary.get("target_take_data_read") is False, "target opened")
    _require(
        boundary.get("prediction_metrics_computed") is False,
        "source QA computed prediction metrics",
    )
    _require(
        boundary.get("model_parameters_fitted") is False,
        "source QA fitted model parameters",
    )
    return {
        "passed": True,
        "source_qa_passed": bool(payload["source_qa_passed"]),
        "result_sha256": payload["result_sha256"],
        "opened_take_count": len(boundary["opened_take_ids"]),
    }


def write_source_qa_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
