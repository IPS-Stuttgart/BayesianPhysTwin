"""Target-free RGBench mesh reduction with explicit contact lineage."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import shutil
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rgbench_online_belief import load_obj_triangles, sha256_file

CONTRACT = "rgbbench-isotropic-mesh-v2"
_ARTIFACT_SALT = b"rgbbench-isotropic-mesh-artifact-v2\0"
_MANIFEST_SALT = b"rgbbench-isotropic-mesh-manifest-v2\0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_sha256(payload: dict[str, Any], salt: bytes) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        salt
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RGBenchIsotropicMeshConfig:
    """Frozen, outcome-blind physical-backend admission settings."""

    identity_max_vertices: int = 12_000
    physical_min_vertices: int = 128
    remesh_iterations: int = 5
    target_edge_lengths_um: tuple[int, ...] = tuple(range(8_000, 20_001, 250))
    maximum_surface_distance_um: int = 3_000
    maximum_source_mean_distance_um: int = 5_000
    maximum_source_p99_distance_um: int = 10_000
    maximum_source_distance_um: int = 15_000
    feature_angle_degrees: float = 30.0

    def __post_init__(self) -> None:
        _require(
            self.identity_max_vertices >= self.physical_min_vertices >= 128,
            "invalid physical vertex limits",
        )
        _require(self.remesh_iterations >= 1, "remesh_iterations must be positive")
        _require(
            len(self.target_edge_lengths_um) >= 1
            and tuple(sorted(set(self.target_edge_lengths_um)))
            == self.target_edge_lengths_um
            and all(value > 0 for value in self.target_edge_lengths_um),
            "target edge lengths must be unique, sorted, and positive",
        )
        _require(
            0
            < self.maximum_surface_distance_um
            <= self.maximum_source_p99_distance_um
            <= self.maximum_source_distance_um,
            "invalid surface-distance limits",
        )
        _require(
            0
            < self.maximum_source_mean_distance_um
            <= self.maximum_source_p99_distance_um,
            "invalid mean-distance limit",
        )
        _require(
            math.isfinite(self.feature_angle_degrees)
            and 0.0 < self.feature_angle_degrees < 180.0,
            "feature angle must lie in (0, 180)",
        )


@dataclass(frozen=True)
class MeshAdmissionAttempt:
    """One target-free identity or remeshing candidate."""

    mode: str
    target_edge_length_um: int | None
    vertex_count: int
    face_count: int
    connected_component_count: int
    edge_manifold: bool
    vertex_manifold: bool
    orientable: bool
    degenerate_face_count: int
    duplicate_face_count: int
    self_intersection_face_count: int
    source_mean_distance_m: float
    source_p99_distance_m: float
    source_max_distance_m: float
    maximum_pin_snap_distance_m: float
    accepted: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(self.mode in {"identity", "isotropic_remesh"}, "unknown mode")
        _require(
            (self.mode == "identity") == (self.target_edge_length_um is None),
            "identity candidates cannot have a target edge length",
        )
        _require(
            self.vertex_count >= 1 and self.face_count >= 1,
            "candidate mesh is empty",
        )
        _require(
            self.connected_component_count >= 1,
            "candidate has no connected component",
        )
        _require(
            self.degenerate_face_count >= 0
            and self.duplicate_face_count >= 0
            and self.self_intersection_face_count >= 0,
            "candidate contains an invalid face count",
        )
        for value in (
            self.source_mean_distance_m,
            self.source_p99_distance_m,
            self.source_max_distance_m,
            self.maximum_pin_snap_distance_m,
        ):
            _require(math.isfinite(value) and value >= 0.0, "invalid distance")
        _require(
            self.accepted == (len(self.rejection_reasons) == 0),
            "acceptance and rejection reasons disagree",
        )


@dataclass(frozen=True)
class RGBenchIsotropicMeshArtifact:
    """Immutable physical mesh and contact-lineage descriptor."""

    garment: str
    mode: str
    source_mesh_relative_path: str
    source_mesh_sha256: str
    source_vertex_count: int
    source_face_count: int
    cloth_parameters_relative_path: str
    cloth_parameters_sha256: str
    source_fling_pin_indices: tuple[int, int]
    source_fling_pin_positions_m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    derived_mesh_relative_path: str
    derived_mesh_sha256: str
    derived_vertex_count: int
    derived_face_count: int
    derived_fling_pin_indices: tuple[int, int]
    selected_target_edge_length_um: int | None
    pymeshlab_version: str | None
    config: RGBenchIsotropicMeshConfig
    attempts: tuple[MeshAdmissionAttempt, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.garment), "garment is empty")
        _require(self.mode in {"identity", "isotropic_remesh"}, "unknown mode")
        _require(
            (self.mode == "identity")
            == (self.selected_target_edge_length_um is None),
            "selected edge length and mode disagree",
        )
        for path in (
            self.source_mesh_relative_path,
            self.cloth_parameters_relative_path,
            self.derived_mesh_relative_path,
        ):
            _require(bool(path) and not Path(path).is_absolute(), "path must be relative")
        for digest in (
            self.source_mesh_sha256,
            self.cloth_parameters_sha256,
            self.derived_mesh_sha256,
            self.artifact_sha256,
        ):
            _require(_valid_sha256(digest), "artifact contains an invalid SHA-256")
        _require(
            self.source_vertex_count >= self.config.physical_min_vertices
            and self.derived_vertex_count >= self.config.physical_min_vertices,
            "mesh violates the physical minimum",
        )
        _require(
            self.source_face_count >= 1 and self.derived_face_count >= 1,
            "mesh has no faces",
        )
        for index in self.source_fling_pin_indices:
            _require(
                0 <= index < self.source_vertex_count,
                "source pin index is out of bounds",
            )
        for index in self.derived_fling_pin_indices:
            _require(
                0 <= index < self.derived_vertex_count,
                "derived pin index is out of bounds",
            )
        positions = np.asarray(self.source_fling_pin_positions_m, dtype=np.float64)
        _require(
            positions.shape == (2, 3) and np.all(np.isfinite(positions)),
            "source pin positions are invalid",
        )
        _require(len(self.attempts) >= 1, "artifact has no admission attempt")
        _require(
            sum(attempt.accepted for attempt in self.attempts) == 1
            and self.attempts[-1].accepted,
            "artifact must terminate at exactly one accepted candidate",
        )
        _require(
            _canonical_sha256(self.descriptor(), _ARTIFACT_SALT)
            == self.artifact_sha256,
            "artifact digest changed",
        )

    def descriptor(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "schema_version": 1,
            "artifact_kind": "RGBenchIsotropicMeshArtifact",
            "contract": CONTRACT,
            "garment": self.garment,
            "mode": self.mode,
            "source_mesh_relative_path": self.source_mesh_relative_path,
            "source_mesh_sha256": self.source_mesh_sha256,
            "source_vertex_count": self.source_vertex_count,
            "source_face_count": self.source_face_count,
            "cloth_parameters_relative_path": self.cloth_parameters_relative_path,
            "cloth_parameters_sha256": self.cloth_parameters_sha256,
            "source_fling_pin_indices": list(self.source_fling_pin_indices),
            "source_fling_pin_positions_m": [
                list(position) for position in self.source_fling_pin_positions_m
            ],
            "derived_mesh_relative_path": self.derived_mesh_relative_path,
            "derived_mesh_sha256": self.derived_mesh_sha256,
            "derived_vertex_count": self.derived_vertex_count,
            "derived_face_count": self.derived_face_count,
            "derived_fling_pin_indices": list(self.derived_fling_pin_indices),
            "selected_target_edge_length_um": self.selected_target_edge_length_um,
            "pymeshlab_version": self.pymeshlab_version,
            "config": asdict(self.config),
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "information_boundary": {
                "source_mesh_coordinates_read": True,
                "cloth_parameter_metadata_read": True,
                "robot_trajectory_coordinates_read": False,
                "object_point_cloud_coordinates_read": False,
                "object_outcomes_read": False,
                "selection_uses_only_mesh_admission": True,
                "fling_contacts_preserved_from_released_configuration": True,
                "grasp_and_fold_contacts_remain_upstream_automatic": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class RGBenchIsotropicMeshManifest:
    """Sorted collection of all primary-garment physical mesh artifacts."""

    rgbbench_commit: str
    dataset_revision: str
    dataset_manifest_artifact_sha256: str
    dataset_manifest_file_sha256: str
    artifacts: tuple[RGBenchIsotropicMeshArtifact, ...]
    artifact_files: tuple[str, ...]
    artifact_file_sha256s: tuple[str, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(len(self.artifacts) >= 1, "manifest has no artifacts")
        _require(bool(self.rgbbench_commit), "RGBench commit is empty")
        _require(bool(self.dataset_revision), "dataset revision is empty")
        _require(
            _valid_sha256(self.dataset_manifest_artifact_sha256)
            and _valid_sha256(self.dataset_manifest_file_sha256),
            "dataset-manifest provenance is invalid",
        )
        garments = tuple(artifact.garment for artifact in self.artifacts)
        _require(
            garments == tuple(sorted(garments)) and len(set(garments)) == len(garments),
            "manifest garments are not unique and sorted",
        )
        _require(
            len(self.artifacts)
            == len(self.artifact_files)
            == len(self.artifact_file_sha256s),
            "manifest artifact lists differ in length",
        )
        _require(
            all(_valid_sha256(value) for value in self.artifact_file_sha256s),
            "manifest contains an invalid file digest",
        )
        _require(
            _canonical_sha256(self.descriptor(), _MANIFEST_SALT)
            == self.artifact_sha256,
            "manifest digest changed",
        )

    def descriptor(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "schema_version": 1,
            "artifact_kind": "RGBenchIsotropicMeshManifest",
            "contract": CONTRACT,
            "rgbbench_commit": self.rgbbench_commit,
            "dataset_revision": self.dataset_revision,
            "dataset_manifest_artifact_sha256": (
                self.dataset_manifest_artifact_sha256
            ),
            "dataset_manifest_file_sha256": self.dataset_manifest_file_sha256,
            "artifacts": [
                {
                    "garment": artifact.garment,
                    "artifact_sha256": artifact.artifact_sha256,
                    "artifact_file": artifact_file,
                    "artifact_file_sha256": artifact_file_sha256,
                    "derived_mesh_relative_path": artifact.derived_mesh_relative_path,
                    "derived_mesh_sha256": artifact.derived_mesh_sha256,
                }
                for artifact, artifact_file, artifact_file_sha256 in zip(
                    self.artifacts,
                    self.artifact_files,
                    self.artifact_file_sha256s,
                    strict=True,
                )
            ],
            "information_boundary": {
                "object_outcomes_read": False,
                "object_point_cloud_coordinates_read": False,
                "all_meshes_selected_before_physical_outcome_evaluation": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class _Topology:
    connected_component_count: int
    edge_manifold: bool
    vertex_manifold: bool
    orientable: bool


def _validate_mesh_arrays(vertices: np.ndarray, faces: np.ndarray) -> None:
    _require(
        vertices.ndim == 2
        and vertices.shape[1] == 3
        and len(vertices) >= 1
        and np.all(np.isfinite(vertices)),
        "vertices must be a finite (N, 3) array",
    )
    _require(
        faces.ndim == 2
        and faces.shape[1] == 3
        and len(faces) >= 1
        and np.issubdtype(faces.dtype, np.integer),
        "faces must be an integer (F, 3) array",
    )
    _require(
        int(np.min(faces)) >= 0 and int(np.max(faces)) < len(vertices),
        "face index is out of bounds",
    )
    _require(
        np.all(faces[:, 0] != faces[:, 1])
        and np.all(faces[:, 0] != faces[:, 2])
        and np.all(faces[:, 1] != faces[:, 2]),
        "mesh contains a degenerate face",
    )


def _mesh_topology(vertices: np.ndarray, faces: np.ndarray) -> _Topology:
    _validate_mesh_arrays(vertices, faces)
    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    vertex_faces: list[list[int]] = [[] for _ in range(len(vertices))]
    for face_index, (a, b, c) in enumerate(faces.tolist()):
        for vertex in (a, b, c):
            vertex_faces[vertex].append(face_index)
        for start, stop in ((a, b), (b, c), (c, a)):
            edge = (min(start, stop), max(start, stop))
            direction = 1 if (start, stop) == edge else -1
            edge_faces[edge].append((face_index, direction))

    edge_manifold = all(len(incidents) <= 2 for incidents in edge_faces.values())
    face_neighbors: list[list[tuple[int, int]]] = [[] for _ in range(len(faces))]
    for incidents in edge_faces.values():
        if len(incidents) == 2:
            (left, left_direction), (right, right_direction) = incidents
            relation = -left_direction * right_direction
            face_neighbors[left].append((right, relation))
            face_neighbors[right].append((left, relation))

    orientation: list[int | None] = [None] * len(faces)
    connected_components = 0
    orientable = True
    for seed in range(len(faces)):
        if orientation[seed] is not None:
            continue
        connected_components += 1
        orientation[seed] = 1
        queue: deque[int] = deque([seed])
        while queue:
            face = queue.popleft()
            assert orientation[face] is not None
            for neighbor, relation in face_neighbors[face]:
                required = orientation[face] * relation
                if orientation[neighbor] is None:
                    orientation[neighbor] = required
                    queue.append(neighbor)
                elif orientation[neighbor] != required:
                    orientable = False

    vertex_manifold = edge_manifold
    if vertex_manifold:
        for vertex, incidents in enumerate(vertex_faces):
            if not incidents:
                vertex_manifold = False
                break
            incident_set = set(incidents)
            local_neighbors: dict[int, set[int]] = {
                face: set() for face in incidents
            }
            boundary_edges = 0
            for edge, edge_incidents in edge_faces.items():
                if vertex not in edge:
                    continue
                if len(edge_incidents) == 1:
                    boundary_edges += 1
                elif len(edge_incidents) == 2:
                    left = edge_incidents[0][0]
                    right = edge_incidents[1][0]
                    if left in incident_set and right in incident_set:
                        local_neighbors[left].add(right)
                        local_neighbors[right].add(left)
            visited = {incidents[0]}
            frontier = [incidents[0]]
            while frontier:
                face = frontier.pop()
                for neighbor in local_neighbors[face]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append(neighbor)
            if visited != incident_set or boundary_edges not in {0, 2}:
                vertex_manifold = False
                break

    return _Topology(
        connected_component_count=connected_components,
        edge_manifold=edge_manifold,
        vertex_manifold=vertex_manifold,
        orientable=edge_manifold and orientable,
    )


def _pymeshlab_self_intersection_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> int:
    try:
        import pymeshlab
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "RGBench isotropic mesh validation requires PyMeshLab"
        ) from error
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces),
        "candidate",
    )
    mesh_set.set_selection_none()
    mesh_set.compute_selection_by_self_intersections_per_face()
    return int(mesh_set.current_mesh().selected_face_number())


def _remesh_isotropic(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_edge_length_m: float,
    config: RGBenchIsotropicMeshConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import pymeshlab
    except (ImportError, OSError) as error:
        raise RuntimeError("RGBench isotropic remeshing requires PyMeshLab") from error
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces),
        "source",
    )
    mesh_set.meshing_isotropic_explicit_remeshing(
        iterations=config.remesh_iterations,
        adaptive=False,
        selectedonly=False,
        targetlen=pymeshlab.PureValue(target_edge_length_m),
        featuredeg=config.feature_angle_degrees,
        checksurfdist=True,
        maxsurfdist=pymeshlab.PureValue(
            config.maximum_surface_distance_um * 1e-6
        ),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True,
    )
    mesh = mesh_set.current_mesh()
    remeshed_vertices = np.asarray(mesh.vertex_matrix(), dtype=np.float64).copy()
    remeshed_faces = np.asarray(mesh.face_matrix(), dtype=np.int64).copy()
    return (
        remeshed_vertices,
        remeshed_faces,
        importlib.metadata.version("pymeshlab"),
    )


def _source_distances(
    source_vertices: np.ndarray,
    derived_vertices: np.ndarray,
) -> tuple[float, float, float]:
    if np.array_equal(source_vertices, derived_vertices):
        return 0.0, 0.0, 0.0
    try:
        from scipy.spatial import cKDTree
    except (ImportError, OSError, ValueError):
        _require(
            len(source_vertices) * len(derived_vertices) <= 5_000_000,
            "large mesh-distance validation requires a working scipy",
        )
        distances = np.empty(len(source_vertices), dtype=np.float64)
        for start in range(0, len(source_vertices), 256):
            stop = min(start + 256, len(source_vertices))
            difference = (
                source_vertices[start:stop, None, :]
                - derived_vertices[None, :, :]
            )
            distances[start:stop] = np.sqrt(
                np.min(np.sum(difference * difference, axis=2), axis=1)
            )
    else:
        distances = np.asarray(
            cKDTree(derived_vertices).query(
                source_vertices,
                k=1,
                workers=-1,
            )[0],
            dtype=np.float64,
        )
    return (
        float(np.mean(distances)),
        float(np.quantile(distances, 0.99)),
        float(np.max(distances)),
    )


def _snap_pin_positions(
    vertices: np.ndarray,
    pin_positions: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], float]:
    snapped = np.asarray(vertices, dtype=np.float64).copy()
    pins = np.asarray(pin_positions, dtype=np.float64)
    _require(pins.shape == (2, 3), "pin_positions must have shape (2, 3)")
    selected: list[int] = []
    distances: list[float] = []
    for pin in pins:
        candidates = np.linalg.norm(snapped - pin, axis=1)
        order = np.lexsort((np.arange(len(snapped)), candidates))
        index = next(int(value) for value in order if int(value) not in selected)
        selected.append(index)
        distances.append(float(candidates[index]))
        snapped[index] = pin
    return snapped, (selected[0], selected[1]), max(distances)


def write_obj_triangles(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Write a no-UV triangle OBJ while preserving array index order."""

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    _validate_mesh_arrays(vertex_array, face_array)
    destination = Path(path)
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        for x, y, z in vertex_array:
            stream.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in face_array:
            stream.write(f"f {a + 1} {b + 1} {c + 1}\n")
    temporary.replace(destination)


def _candidate_attempt(
    *,
    mode: str,
    target_edge_length_um: int | None,
    source_vertices: np.ndarray,
    derived_vertices: np.ndarray,
    derived_faces: np.ndarray,
    maximum_pin_snap_distance_m: float,
    config: RGBenchIsotropicMeshConfig,
    self_intersection_counter: Callable[[np.ndarray, np.ndarray], int],
) -> MeshAdmissionAttempt:
    topology = _mesh_topology(derived_vertices, derived_faces)
    face_points = derived_vertices[derived_faces]
    double_areas = np.linalg.norm(
        np.cross(
            face_points[:, 1] - face_points[:, 0],
            face_points[:, 2] - face_points[:, 0],
        ),
        axis=1,
    )
    degenerate_faces = int(np.count_nonzero(double_areas <= 1e-15))
    canonical_faces = np.sort(derived_faces, axis=1)
    duplicate_faces = int(
        len(canonical_faces) - len(np.unique(canonical_faces, axis=0))
    )
    self_intersections = self_intersection_counter(
        derived_vertices,
        derived_faces,
    )
    mean_distance, p99_distance, max_distance = _source_distances(
        source_vertices,
        derived_vertices,
    )
    reasons: list[str] = []
    if len(derived_vertices) < config.physical_min_vertices:
        reasons.append("below_physical_vertex_minimum")
    if len(derived_vertices) > config.identity_max_vertices:
        reasons.append("above_physical_vertex_maximum")
    if topology.connected_component_count != 1:
        reasons.append("multiple_connected_components")
    if not topology.edge_manifold:
        reasons.append("non_manifold_edges")
    if not topology.vertex_manifold:
        reasons.append("non_manifold_vertices")
    if not topology.orientable:
        reasons.append("non_orientable")
    if degenerate_faces != 0:
        reasons.append("degenerate_faces")
    if duplicate_faces != 0:
        reasons.append("duplicate_faces")
    if self_intersections != 0:
        reasons.append("self_intersections")
    if mean_distance > config.maximum_source_mean_distance_um * 1e-6:
        reasons.append("mean_surface_distortion")
    if p99_distance > config.maximum_source_p99_distance_um * 1e-6:
        reasons.append("p99_surface_distortion")
    if max_distance > config.maximum_source_distance_um * 1e-6:
        reasons.append("maximum_surface_distortion")
    return MeshAdmissionAttempt(
        mode=mode,
        target_edge_length_um=target_edge_length_um,
        vertex_count=len(derived_vertices),
        face_count=len(derived_faces),
        connected_component_count=topology.connected_component_count,
        edge_manifold=topology.edge_manifold,
        vertex_manifold=topology.vertex_manifold,
        orientable=topology.orientable,
        degenerate_face_count=degenerate_faces,
        duplicate_face_count=duplicate_faces,
        self_intersection_face_count=self_intersections,
        source_mean_distance_m=mean_distance,
        source_p99_distance_m=p99_distance,
        source_max_distance_m=max_distance,
        maximum_pin_snap_distance_m=maximum_pin_snap_distance_m,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def build_isotropic_mesh_artifact(
    *,
    garment: str,
    source_mesh: str | Path,
    source_mesh_relative_path: str,
    cloth_parameters: str | Path,
    cloth_parameters_relative_path: str,
    source_fling_pin_indices: tuple[int, int],
    derived_mesh: str | Path,
    derived_mesh_relative_path: str,
    config: RGBenchIsotropicMeshConfig | None = None,
    remesher: Callable[
        [np.ndarray, np.ndarray, float, RGBenchIsotropicMeshConfig],
        tuple[np.ndarray, np.ndarray, str],
    ] = _remesh_isotropic,
    self_intersection_counter: Callable[
        [np.ndarray, np.ndarray], int
    ] = _pymeshlab_self_intersection_faces,
) -> RGBenchIsotropicMeshArtifact:
    """Select and write one target-free physical mesh artifact."""

    cfg = config or RGBenchIsotropicMeshConfig()
    source_path = Path(source_mesh)
    parameters_path = Path(cloth_parameters)
    derived_path = Path(derived_mesh)
    _require(source_path.is_file(), "source mesh does not exist")
    _require(parameters_path.is_file(), "cloth parameters do not exist")
    _require(not derived_path.exists(), "refusing to overwrite derived mesh")
    source_vertices, source_faces = load_obj_triangles(source_path)
    for index in source_fling_pin_indices:
        _require(
            0 <= index < len(source_vertices),
            "source fling pin is out of bounds",
        )
    pin_positions = source_vertices[np.asarray(source_fling_pin_indices)]
    attempts: list[MeshAdmissionAttempt] = []
    selected_vertices: np.ndarray | None = None
    selected_faces: np.ndarray | None = None
    selected_pins: tuple[int, int] | None = None
    selected_edge: int | None = None
    pymeshlab_version: str | None = None
    mode = "identity"

    if len(source_vertices) <= cfg.identity_max_vertices:
        identity_attempt = _candidate_attempt(
            mode="identity",
            target_edge_length_um=None,
            source_vertices=source_vertices,
            derived_vertices=source_vertices,
            derived_faces=source_faces,
            maximum_pin_snap_distance_m=0.0,
            config=cfg,
            self_intersection_counter=self_intersection_counter,
        )
        attempts.append(identity_attempt)
        if identity_attempt.accepted:
            selected_vertices = source_vertices
            selected_faces = source_faces
            selected_pins = source_fling_pin_indices

    if selected_vertices is None:
        mode = "isotropic_remesh"
        for edge_um in cfg.target_edge_lengths_um:
            vertices, faces, version = remesher(
                source_vertices,
                source_faces,
                edge_um * 1e-6,
                cfg,
            )
            _validate_mesh_arrays(vertices, faces)
            vertices, pins, snap_distance = _snap_pin_positions(
                vertices,
                pin_positions,
            )
            attempt = _candidate_attempt(
                mode=mode,
                target_edge_length_um=edge_um,
                source_vertices=source_vertices,
                derived_vertices=vertices,
                derived_faces=faces,
                maximum_pin_snap_distance_m=snap_distance,
                config=cfg,
                self_intersection_counter=self_intersection_counter,
            )
            attempts.append(attempt)
            if attempt.accepted:
                selected_vertices = vertices
                selected_faces = faces
                selected_pins = pins
                selected_edge = edge_um
                pymeshlab_version = version
                break

    _require(
        selected_vertices is not None
        and selected_faces is not None
        and selected_pins is not None,
        f"{garment} has no admissible mesh candidate",
    )
    if mode == "identity":
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = derived_path.with_suffix(derived_path.suffix + ".tmp")
        shutil.copyfile(source_path, temporary)
        temporary.replace(derived_path)
    else:
        write_obj_triangles(derived_path, selected_vertices, selected_faces)

    reloaded_vertices, reloaded_faces = load_obj_triangles(derived_path)
    _require(
        len(reloaded_vertices) == len(selected_vertices)
        and np.array_equal(reloaded_faces, selected_faces),
        "derived OBJ changed mesh indexing",
    )
    _require(
        np.array_equal(
            reloaded_vertices[np.asarray(selected_pins)],
            pin_positions,
        ),
        "derived OBJ did not preserve exact fling pin coordinates",
    )
    source_positions = tuple(
        tuple(float(value) for value in position) for position in pin_positions
    )
    descriptor = {
        "schema_version": 1,
        "artifact_kind": "RGBenchIsotropicMeshArtifact",
        "contract": CONTRACT,
        "garment": garment,
        "mode": mode,
        "source_mesh_relative_path": source_mesh_relative_path,
        "source_mesh_sha256": sha256_file(source_path),
        "source_vertex_count": len(source_vertices),
        "source_face_count": len(source_faces),
        "cloth_parameters_relative_path": cloth_parameters_relative_path,
        "cloth_parameters_sha256": sha256_file(parameters_path),
        "source_fling_pin_indices": list(source_fling_pin_indices),
        "source_fling_pin_positions_m": [
            list(position) for position in source_positions
        ],
        "derived_mesh_relative_path": derived_mesh_relative_path,
        "derived_mesh_sha256": sha256_file(derived_path),
        "derived_vertex_count": len(reloaded_vertices),
        "derived_face_count": len(reloaded_faces),
        "derived_fling_pin_indices": list(selected_pins),
        "selected_target_edge_length_um": selected_edge,
        "pymeshlab_version": pymeshlab_version,
        "config": asdict(cfg),
        "attempts": [asdict(attempt) for attempt in attempts],
        "information_boundary": {
            "source_mesh_coordinates_read": True,
            "cloth_parameter_metadata_read": True,
            "robot_trajectory_coordinates_read": False,
            "object_point_cloud_coordinates_read": False,
            "object_outcomes_read": False,
            "selection_uses_only_mesh_admission": True,
            "fling_contacts_preserved_from_released_configuration": True,
            "grasp_and_fold_contacts_remain_upstream_automatic": True,
        },
        "artifact_sha256": "0" * 64,
    }
    descriptor["artifact_sha256"] = _canonical_sha256(
        descriptor,
        _ARTIFACT_SALT,
    )
    return RGBenchIsotropicMeshArtifact(
        garment=garment,
        mode=mode,
        source_mesh_relative_path=source_mesh_relative_path,
        source_mesh_sha256=descriptor["source_mesh_sha256"],
        source_vertex_count=len(source_vertices),
        source_face_count=len(source_faces),
        cloth_parameters_relative_path=cloth_parameters_relative_path,
        cloth_parameters_sha256=descriptor["cloth_parameters_sha256"],
        source_fling_pin_indices=source_fling_pin_indices,
        source_fling_pin_positions_m=source_positions,
        derived_mesh_relative_path=derived_mesh_relative_path,
        derived_mesh_sha256=descriptor["derived_mesh_sha256"],
        derived_vertex_count=len(reloaded_vertices),
        derived_face_count=len(reloaded_faces),
        derived_fling_pin_indices=selected_pins,
        selected_target_edge_length_um=selected_edge,
        pymeshlab_version=pymeshlab_version,
        config=cfg,
        attempts=tuple(attempts),
        artifact_sha256=descriptor["artifact_sha256"],
    )


def load_isotropic_mesh_artifact(
    path: str | Path,
) -> RGBenchIsotropicMeshArtifact:
    """Load and validate one mesh artifact JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == "RGBenchIsotropicMeshArtifact"
        and payload.get("contract") == CONTRACT,
        "not an RGBench isotropic mesh artifact",
    )
    return RGBenchIsotropicMeshArtifact(
        garment=str(payload["garment"]),
        mode=str(payload["mode"]),
        source_mesh_relative_path=str(payload["source_mesh_relative_path"]),
        source_mesh_sha256=str(payload["source_mesh_sha256"]),
        source_vertex_count=int(payload["source_vertex_count"]),
        source_face_count=int(payload["source_face_count"]),
        cloth_parameters_relative_path=str(
            payload["cloth_parameters_relative_path"]
        ),
        cloth_parameters_sha256=str(payload["cloth_parameters_sha256"]),
        source_fling_pin_indices=tuple(
            int(value) for value in payload["source_fling_pin_indices"]
        ),
        source_fling_pin_positions_m=tuple(
            tuple(float(value) for value in position)
            for position in payload["source_fling_pin_positions_m"]
        ),
        derived_mesh_relative_path=str(payload["derived_mesh_relative_path"]),
        derived_mesh_sha256=str(payload["derived_mesh_sha256"]),
        derived_vertex_count=int(payload["derived_vertex_count"]),
        derived_face_count=int(payload["derived_face_count"]),
        derived_fling_pin_indices=tuple(
            int(value) for value in payload["derived_fling_pin_indices"]
        ),
        selected_target_edge_length_um=(
            None
            if payload["selected_target_edge_length_um"] is None
            else int(payload["selected_target_edge_length_um"])
        ),
        pymeshlab_version=(
            None
            if payload["pymeshlab_version"] is None
            else str(payload["pymeshlab_version"])
        ),
        config=RGBenchIsotropicMeshConfig(
            **{
                **payload["config"],
                "target_edge_lengths_um": tuple(
                    payload["config"]["target_edge_lengths_um"]
                ),
            }
        ),
        attempts=tuple(
            MeshAdmissionAttempt(
                **{
                    **attempt,
                    "rejection_reasons": tuple(attempt["rejection_reasons"]),
                }
            )
            for attempt in payload["attempts"]
        ),
        artifact_sha256=str(payload["artifact_sha256"]),
    )


def build_isotropic_mesh_manifest(
    artifact_paths: tuple[Path, ...],
    *,
    root: Path,
    rgbbench_commit: str,
    dataset_revision: str,
    dataset_manifest_artifact_sha256: str,
    dataset_manifest_file_sha256: str,
) -> RGBenchIsotropicMeshManifest:
    """Load sorted per-garment artifacts and bind their files."""

    artifacts_and_paths = sorted(
        (
            (load_isotropic_mesh_artifact(path), path)
            for path in artifact_paths
        ),
        key=lambda item: item[0].garment,
    )
    artifacts = tuple(item[0] for item in artifacts_and_paths)
    relative_files = tuple(
        str(path.resolve().relative_to(root.resolve()))
        for _, path in artifacts_and_paths
    )
    file_digests = tuple(sha256_file(path) for _, path in artifacts_and_paths)
    descriptor = {
        "schema_version": 1,
        "artifact_kind": "RGBenchIsotropicMeshManifest",
        "contract": CONTRACT,
        "rgbbench_commit": rgbbench_commit,
        "dataset_revision": dataset_revision,
        "dataset_manifest_artifact_sha256": dataset_manifest_artifact_sha256,
        "dataset_manifest_file_sha256": dataset_manifest_file_sha256,
        "artifacts": [
            {
                "garment": artifact.garment,
                "artifact_sha256": artifact.artifact_sha256,
                "artifact_file": artifact_file,
                "artifact_file_sha256": file_digest,
                "derived_mesh_relative_path": artifact.derived_mesh_relative_path,
                "derived_mesh_sha256": artifact.derived_mesh_sha256,
            }
            for artifact, artifact_file, file_digest in zip(
                artifacts,
                relative_files,
                file_digests,
                strict=True,
            )
        ],
        "information_boundary": {
            "object_outcomes_read": False,
            "object_point_cloud_coordinates_read": False,
            "all_meshes_selected_before_physical_outcome_evaluation": True,
        },
        "artifact_sha256": "0" * 64,
    }
    descriptor["artifact_sha256"] = _canonical_sha256(
        descriptor,
        _MANIFEST_SALT,
    )
    return RGBenchIsotropicMeshManifest(
        rgbbench_commit=rgbbench_commit,
        dataset_revision=dataset_revision,
        dataset_manifest_artifact_sha256=dataset_manifest_artifact_sha256,
        dataset_manifest_file_sha256=dataset_manifest_file_sha256,
        artifacts=artifacts,
        artifact_files=relative_files,
        artifact_file_sha256s=file_digests,
        artifact_sha256=descriptor["artifact_sha256"],
    )


def load_isotropic_mesh_manifest(
    path: str | Path,
) -> RGBenchIsotropicMeshManifest:
    """Load a manifest and verify every bound artifact and derived mesh."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == "RGBenchIsotropicMeshManifest"
        and payload.get("contract") == CONTRACT,
        "not an RGBench isotropic mesh manifest",
    )
    entries = payload.get("artifacts")
    _require(isinstance(entries, list) and entries, "manifest has no artifacts")
    artifacts: list[RGBenchIsotropicMeshArtifact] = []
    artifact_files: list[str] = []
    artifact_file_sha256s: list[str] = []
    for entry in entries:
        _require(isinstance(entry, dict), "manifest contains a non-object entry")
        artifact_relative = str(entry["artifact_file"])
        artifact_path = source.parent / artifact_relative
        expected_file_sha256 = str(entry["artifact_file_sha256"])
        _require(
            artifact_path.is_file()
            and sha256_file(artifact_path) == expected_file_sha256,
            "bound mesh artifact file changed",
        )
        artifact = load_isotropic_mesh_artifact(artifact_path)
        _require(
            artifact.garment == entry["garment"]
            and artifact.artifact_sha256 == entry["artifact_sha256"]
            and artifact.derived_mesh_relative_path
            == entry["derived_mesh_relative_path"]
            and artifact.derived_mesh_sha256 == entry["derived_mesh_sha256"],
            "mesh artifact summary changed",
        )
        derived_mesh = source.parent / artifact.derived_mesh_relative_path
        _require(
            derived_mesh.is_file()
            and sha256_file(derived_mesh) == artifact.derived_mesh_sha256,
            "bound derived mesh changed",
        )
        artifacts.append(artifact)
        artifact_files.append(artifact_relative)
        artifact_file_sha256s.append(expected_file_sha256)
    return RGBenchIsotropicMeshManifest(
        rgbbench_commit=str(payload["rgbbench_commit"]),
        dataset_revision=str(payload["dataset_revision"]),
        dataset_manifest_artifact_sha256=str(
            payload["dataset_manifest_artifact_sha256"]
        ),
        dataset_manifest_file_sha256=str(
            payload["dataset_manifest_file_sha256"]
        ),
        artifacts=tuple(artifacts),
        artifact_files=tuple(artifact_files),
        artifact_file_sha256s=tuple(artifact_file_sha256s),
        artifact_sha256=str(payload["artifact_sha256"]),
    )


def write_json_once(path: str | Path, payload: dict[str, Any]) -> None:
    """Write canonical human-readable JSON without overwriting."""

    destination = Path(path)
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
