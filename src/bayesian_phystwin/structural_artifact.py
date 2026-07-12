"""Typed artifacts for hierarchical structural calibration of PhysTwin."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


STRUCTURAL_CORRECTION_SCHEMA_VERSION = 1
STRUCTURAL_RANK_CANDIDATES = (4, 8, 16)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _json_data(values: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def nominal_rest_geometry_sha256(
    rest_positions: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    num_object_springs: int,
) -> str:
    """Hash the object embedding, topology, and released object rest lengths."""

    positions = np.asarray(rest_positions, dtype=np.float32)
    edges = np.asarray(springs, dtype=np.int32)
    lengths = np.asarray(rest_lengths, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("rest_positions must have shape (N, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("rest_positions must be finite")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(lengths):
        raise ValueError("springs and rest_lengths must agree")
    if not 0 < num_object_springs <= len(edges):
        raise ValueError("num_object_springs must lie in (0, S]")
    object_edges = edges[:num_object_springs]
    object_lengths = lengths[:num_object_springs]
    if np.any(object_edges < 0) or np.any(object_edges >= len(positions)):
        raise ValueError("object spring endpoint exceeds rest_positions")
    if np.any(object_lengths <= 0.0) or not np.all(np.isfinite(object_lengths)):
        raise ValueError("object rest lengths must be positive and finite")
    digest = hashlib.sha256()
    for name, values in (
        ("rest_positions", positions),
        ("object_springs", object_edges),
        ("object_rest_lengths", object_lengths),
    ):
        digest.update(name.encode("ascii"))
        digest.update(_array_sha256(values).encode("ascii"))
    return digest.hexdigest()


def _proper_rotation(values: np.ndarray, *, name: str) -> np.ndarray:
    rotation = _readonly(values)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=1e-8):
        raise ValueError(f"{name} must be orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError(f"{name} must be a proper rotation")
    return rotation


@dataclass(frozen=True)
class StructuralSessionCorrection:
    """Session placement, settled state, and gravity-frame correction."""

    session_id: str
    frame_linear: np.ndarray
    frame_translation_m: np.ndarray
    settled_state_coefficients: np.ndarray
    gravity_correction_mps2: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be nonempty")
        linear = _proper_rotation(self.frame_linear, name="frame_linear")
        translation = _readonly(self.frame_translation_m)
        settled = _readonly(self.settled_state_coefficients)
        gravity = _readonly(self.gravity_correction_mps2)
        if translation.shape != (3,) or gravity.shape != (3,):
            raise ValueError("frame translation and gravity correction must be 3-vectors")
        if settled.ndim != 1:
            raise ValueError("settled_state_coefficients must be a vector")
        if not all(np.all(np.isfinite(value)) for value in (translation, settled, gravity)):
            raise ValueError("session correction arrays must be finite")
        object.__setattr__(self, "frame_linear", linear)
        object.__setattr__(self, "frame_translation_m", translation)
        object.__setattr__(self, "settled_state_coefficients", settled)
        object.__setattr__(self, "gravity_correction_mps2", gravity)


@dataclass(frozen=True)
class StructuralTwinCorrection:
    """Object-persistent and session-specific structural calibration artifact."""

    nominal_rest_geometry_hash: str
    graph_basis: np.ndarray
    graph_frequencies: np.ndarray
    persistent_rest_coefficients: np.ndarray
    persistent_coefficient_covariance: np.ndarray | None
    sessions: tuple[StructuralSessionCorrection, ...]
    support_node_indices: np.ndarray
    surface_triangles: np.ndarray
    validity_tetrahedra: np.ndarray
    support_model: Mapping[str, Any]
    allowed_edge_strain: float
    fit_session_ids: tuple[str, ...]
    information_boundary: Mapping[str, Any]
    source_checksums: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_sha256(self.nominal_rest_geometry_hash):
            raise ValueError("nominal_rest_geometry_hash must be a SHA-256 digest")
        basis = _readonly(self.graph_basis)
        frequencies = _readonly(self.graph_frequencies)
        coefficients = _readonly(self.persistent_rest_coefficients)
        support_nodes = _readonly(self.support_node_indices, dtype=np.int64)
        triangles = _readonly(self.surface_triangles, dtype=np.int64)
        tetrahedra = _readonly(self.validity_tetrahedra, dtype=np.int64)
        if basis.ndim != 3 or basis.shape[1] != 3:
            raise ValueError("graph_basis must have shape (N, 3, R)")
        rank = basis.shape[2]
        if rank < 1 or frequencies.shape != (rank,) or coefficients.shape != (rank,):
            raise ValueError("basis, graph frequencies, and coefficients disagree")
        if not all(np.all(np.isfinite(value)) for value in (basis, frequencies, coefficients)):
            raise ValueError("structural basis arrays must be finite")
        if np.any(frequencies <= 0.0):
            raise ValueError("graph frequencies must be positive after rigid-mode removal")
        gram = basis.reshape(-1, rank).T @ basis.reshape(-1, rank)
        if not np.allclose(gram, np.eye(rank), atol=1e-7, rtol=1e-7):
            raise ValueError("graph_basis must be orthonormal")
        if support_nodes.ndim != 1 or np.any(support_nodes < 0):
            raise ValueError("support_node_indices must be a nonnegative vector")
        if len(np.unique(support_nodes)) != len(support_nodes):
            raise ValueError("support_node_indices must be unique")
        if np.any(support_nodes >= basis.shape[0]):
            raise ValueError("support node exceeds graph basis")
        if len(support_nodes) and not np.array_equal(
            basis[support_nodes], np.zeros_like(basis[support_nodes])
        ):
            raise ValueError("graph basis must be exactly zero on support anchors")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            raise ValueError("surface_triangles must have shape (F, 3)")
        if tetrahedra.ndim != 2 or tetrahedra.shape[1:] != (4,):
            raise ValueError("validity_tetrahedra must have shape (C, 4)")
        for name, cells in (("surface triangle", triangles), ("validity tetrahedron", tetrahedra)):
            if np.any(cells < 0) or np.any(cells >= basis.shape[0]):
                raise ValueError(f"{name} index exceeds graph basis")
        covariance = self.persistent_coefficient_covariance
        if covariance is not None:
            covariance = _readonly(covariance)
            if covariance.shape != (rank, rank):
                raise ValueError("persistent coefficient covariance must be R by R")
            if not np.all(np.isfinite(covariance)) or not np.allclose(
                covariance, covariance.T, atol=1e-10, rtol=1e-10
            ):
                raise ValueError("persistent coefficient covariance must be finite and symmetric")
            if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
                raise ValueError("persistent coefficient covariance must be positive semidefinite")
        if not 0.0 < self.allowed_edge_strain < 1.0:
            raise ValueError("allowed_edge_strain must lie in (0, 1)")
        if not self.sessions or len({value.session_id for value in self.sessions}) != len(
            self.sessions
        ):
            raise ValueError("sessions must be nonempty and uniquely identified")
        if any(len(value.settled_state_coefficients) != rank for value in self.sessions):
            raise ValueError("session settled coefficients must match the graph rank")
        session_ids = {value.session_id for value in self.sessions}
        if not self.fit_session_ids or not set(self.fit_session_ids) <= session_ids:
            raise ValueError("fit_session_ids must identify artifact sessions")
        if len(set(self.fit_session_ids)) != len(self.fit_session_ids):
            raise ValueError("fit_session_ids must be unique")
        boundary = _json_data(self.information_boundary, name="information_boundary")
        required_boundary = {
            "persistent_fit_uses_o_minus_only": True,
            "future_frames_used_for_fit": False,
            "target_outcomes_used_for_selection": False,
            "fit_mode": "map",
        }
        if any(boundary.get(key) != value for key, value in required_boundary.items()):
            raise ValueError("structural artifact violates its information boundary")
        checksums = dict(self.source_checksums)
        if not checksums or any(not key or not _is_sha256(value) for key, value in checksums.items()):
            raise ValueError("source_checksums must map names to SHA-256 digests")
        object.__setattr__(self, "graph_basis", basis)
        object.__setattr__(self, "graph_frequencies", frequencies)
        object.__setattr__(self, "persistent_rest_coefficients", coefficients)
        object.__setattr__(self, "persistent_coefficient_covariance", covariance)
        object.__setattr__(self, "support_node_indices", support_nodes)
        object.__setattr__(self, "surface_triangles", triangles)
        object.__setattr__(self, "validity_tetrahedra", tetrahedra)
        object.__setattr__(self, "support_model", _json_data(self.support_model, name="support_model"))
        object.__setattr__(self, "information_boundary", boundary)
        object.__setattr__(self, "source_checksums", dict(sorted(checksums.items())))
        object.__setattr__(self, "metadata", _json_data(self.metadata, name="metadata"))

    @property
    def rank(self) -> int:
        return self.graph_basis.shape[2]

    @property
    def posterior_uncertainty_estimated(self) -> bool:
        return self.persistent_coefficient_covariance is not None

    def session(self, session_id: str) -> StructuralSessionCorrection:
        for value in self.sessions:
            if value.session_id == session_id:
                return value
        raise KeyError(session_id)

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_CORRECTION_SCHEMA_VERSION,
            "artifact_kind": "StructuralTwinCorrection",
            "nominal_rest_geometry_hash": self.nominal_rest_geometry_hash,
            "rank": self.rank,
            "posterior_uncertainty_estimated": self.posterior_uncertainty_estimated,
            "session_ids": [value.session_id for value in self.sessions],
            "support_model": self.support_model,
            "allowed_edge_strain": self.allowed_edge_strain,
            "fit_session_ids": list(self.fit_session_ids),
            "information_boundary": self.information_boundary,
            "source_checksums": self.source_checksums,
            "metadata": self.metadata,
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        arrays = {
            "graph_basis": self.graph_basis,
            "graph_frequencies": self.graph_frequencies,
            "persistent_rest_coefficients": self.persistent_rest_coefficients,
            "support_node_indices": self.support_node_indices,
            "surface_triangles": self.surface_triangles,
            "validity_tetrahedra": self.validity_tetrahedra,
            "session_frame_linear": np.stack([value.frame_linear for value in self.sessions]),
            "session_frame_translation_m": np.stack(
                [value.frame_translation_m for value in self.sessions]
            ),
            "session_settled_state_coefficients": np.stack(
                [value.settled_state_coefficients for value in self.sessions]
            ),
            "session_gravity_correction_mps2": np.stack(
                [value.gravity_correction_mps2 for value in self.sessions]
            ),
        }
        if self.persistent_coefficient_covariance is not None:
            arrays["persistent_coefficient_covariance"] = (
                self.persistent_coefficient_covariance
            )
        return arrays

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self._scalar_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in sorted(self._array_payload().items()):
            digest.update(name.encode("ascii"))
            digest.update(_array_sha256(values).encode("ascii"))
        return digest.hexdigest()


def _graph_laplacian(node_count: int, springs: np.ndarray):
    try:
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("graph basis construction requires scipy") from error
    edges = np.asarray(springs, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or not len(edges):
        raise ValueError("springs must have nonempty shape (S, 2)")
    if np.any(edges < 0) or np.any(edges >= node_count):
        raise ValueError("spring endpoint exceeds node_count")
    row = np.concatenate((edges[:, 0], edges[:, 1]))
    column = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(row)), (row, column)), shape=(node_count, node_count)
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0.0):
        raise ValueError("every structural node must belong to the spring graph")
    inverse_root = 1.0 / np.sqrt(degree)
    normalized_adjacency = sparse.diags(inverse_root) @ adjacency @ sparse.diags(
        inverse_root
    )
    return sparse.eye(node_count, format="csr") - normalized_adjacency


def _smallest_graph_modes(laplacian, count: int) -> tuple[np.ndarray, np.ndarray]:
    node_count = laplacian.shape[0]
    count = min(max(count, 2), node_count)
    if node_count <= 256 or count == node_count:
        values, vectors = np.linalg.eigh(laplacian.toarray())
        return values[:count], vectors[:, :count]
    try:
        from scipy.sparse.linalg import eigsh
    except ImportError as error:
        raise RuntimeError("large graph basis construction requires scipy") from error
    values, vectors = eigsh(laplacian, k=min(count, node_count - 1), which="SM")
    order = np.argsort(values)
    return values[order], vectors[:, order]


def _rigid_fields(rest_positions: np.ndarray, free_nodes: np.ndarray) -> np.ndarray:
    points = np.asarray(rest_positions, dtype=float)
    centered = points - np.mean(points[free_nodes], axis=0)
    fields = []
    for axis in np.eye(3):
        field = np.zeros_like(points)
        field[free_nodes] = axis
        fields.append(field.reshape(-1))
    for axis in np.eye(3):
        field = np.zeros_like(points)
        field[free_nodes] = np.cross(axis, centered[free_nodes])
        fields.append(field.reshape(-1))
    matrix = np.column_stack(fields)
    matrix = matrix[:, np.linalg.norm(matrix, axis=0) > 1e-12]
    return np.linalg.qr(matrix, mode="reduced")[0]


def build_rigid_free_graph_basis(
    rest_positions: np.ndarray,
    springs: np.ndarray,
    *,
    rank: int,
    support_node_indices: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build low-frequency vector fields with rigid modes and anchors removed."""

    positions = np.asarray(rest_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(
        np.isfinite(positions)
    ):
        raise ValueError("rest_positions must have finite shape (N, 3)")
    if rank < 1 or rank >= 3 * len(positions) - 6:
        raise ValueError("rank is incompatible with the graph size")
    anchors = np.asarray(tuple(support_node_indices), dtype=np.int64)
    if anchors.ndim != 1 or np.any(anchors < 0) or np.any(anchors >= len(positions)):
        raise ValueError("support anchors exceed the structural graph")
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError("support anchors must be unique")
    free_mask = np.ones(len(positions), dtype=bool)
    free_mask[anchors] = False
    free_nodes = np.flatnonzero(free_mask)
    if 3 * len(free_nodes) - 6 < rank:
        raise ValueError("support anchors leave too few free structural degrees")

    laplacian = _graph_laplacian(len(positions), springs)
    scalar_count = min(len(positions), max(6, int(np.ceil(rank / 3)) + 5))
    rigid = _rigid_fields(positions, free_nodes)
    chosen: list[np.ndarray] = []
    chosen_frequencies: list[float] = []
    while True:
        values, vectors = _smallest_graph_modes(laplacian, scalar_count)
        candidates = []
        for mode_index, frequency in enumerate(values):
            if frequency <= 1e-10:
                continue
            for axis_index in range(3):
                field = np.zeros_like(positions)
                field[free_nodes, axis_index] = vectors[free_nodes, mode_index]
                candidates.append((float(frequency), axis_index, field.reshape(-1)))
        candidates.sort(key=lambda value: (value[0], value[1]))
        chosen = []
        chosen_frequencies = []
        for frequency, _, candidate in candidates:
            candidate = candidate - rigid @ (rigid.T @ candidate)
            for previous in chosen:
                candidate = candidate - previous * float(previous @ candidate)
            norm = float(np.linalg.norm(candidate))
            if norm <= 1e-10:
                continue
            candidate /= norm
            candidate[3 * anchors[:, None] + np.arange(3)] = 0.0
            chosen.append(candidate)
            chosen_frequencies.append(frequency)
            if len(chosen) == rank:
                break
        if len(chosen) == rank:
            break
        if scalar_count >= len(positions):
            raise ValueError("could not construct the requested rigid-free graph rank")
        scalar_count = min(len(positions), scalar_count + max(4, int(np.ceil(rank / 3))))

    basis = np.column_stack(chosen).reshape(len(positions), 3, rank)
    flat = basis.reshape(-1, rank)
    rigid_overlap = np.max(np.abs(rigid.T @ flat), initial=0.0)
    diagnostics = {
        "rank": rank,
        "scalar_mode_count": scalar_count,
        "support_node_count": len(anchors),
        "maximum_rigid_mode_overlap": float(rigid_overlap),
        "orthonormality_maximum_error": float(
            np.max(np.abs(flat.T @ flat - np.eye(rank)), initial=0.0)
        ),
    }
    return basis, np.asarray(chosen_frequencies), diagnostics


def structural_displacement(
    graph_basis: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    basis = np.asarray(graph_basis, dtype=float)
    values = np.asarray(coefficients, dtype=float)
    if basis.ndim != 3 or basis.shape[1] != 3 or values.shape != (basis.shape[2],):
        raise ValueError("basis and structural coefficients disagree")
    return np.tensordot(basis, values, axes=(2, 0))


def _signed_tetrahedron_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    first = points[cells[:, 1]] - points[cells[:, 0]]
    second = points[cells[:, 2]] - points[cells[:, 0]]
    third = points[cells[:, 3]] - points[cells[:, 0]]
    return np.einsum("ij,ij->i", np.cross(first, second), third) / 6.0


def _segment_triangle_intersection(
    start: np.ndarray,
    stop: np.ndarray,
    triangle: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> bool:
    direction = stop - start
    edge_a = triangle[1] - triangle[0]
    edge_b = triangle[2] - triangle[0]
    cross = np.cross(direction, edge_b)
    determinant = float(edge_a @ cross)
    if abs(determinant) <= tolerance:
        return False
    inverse = 1.0 / determinant
    relative = start - triangle[0]
    u = inverse * float(relative @ cross)
    if u <= tolerance or u >= 1.0 - tolerance:
        return False
    q = np.cross(relative, edge_a)
    v = inverse * float(direction @ q)
    if v <= tolerance or u + v >= 1.0 - tolerance:
        return False
    distance = inverse * float(edge_b @ q)
    return tolerance < distance < 1.0 - tolerance


def _triangle_intersection_count(points: np.ndarray, faces: np.ndarray) -> int:
    intersections = 0
    bounds_min = np.min(points[faces], axis=1)
    bounds_max = np.max(points[faces], axis=1)
    for first in range(len(faces)):
        for second in range(first + 1, len(faces)):
            if set(map(int, faces[first])) & set(map(int, faces[second])):
                continue
            if np.any(bounds_max[first] < bounds_min[second]) or np.any(
                bounds_max[second] < bounds_min[first]
            ):
                continue
            first_triangle = points[faces[first]]
            second_triangle = points[faces[second]]
            first_edges = ((0, 1), (1, 2), (2, 0))
            hit = any(
                _segment_triangle_intersection(
                    first_triangle[start], first_triangle[stop], second_triangle
                )
                for start, stop in first_edges
            ) or any(
                _segment_triangle_intersection(
                    second_triangle[start], second_triangle[stop], first_triangle
                )
                for start, stop in first_edges
            )
            intersections += int(hit)
    return intersections


@dataclass(frozen=True)
class CorrectedRestGeometry:
    rest_positions: np.ndarray
    rest_lengths: np.ndarray
    persistent_displacement: np.ndarray
    diagnostics: Mapping[str, Any]


def corrected_rest_geometry(
    correction: StructuralTwinCorrection,
    nominal_rest_positions: np.ndarray,
    springs: np.ndarray,
    nominal_rest_lengths: np.ndarray,
    *,
    num_object_springs: int,
) -> CorrectedRestGeometry:
    """Apply a realizable embedded correction and reject implausible geometry."""

    positions = np.asarray(nominal_rest_positions)
    edges = np.asarray(springs, dtype=np.int64)
    released = np.asarray(nominal_rest_lengths)
    digest = nominal_rest_geometry_sha256(
        positions,
        edges,
        released,
        num_object_springs=num_object_springs,
    )
    if digest != correction.nominal_rest_geometry_hash:
        raise ValueError("structural correction nominal rest geometry mismatch")
    if positions.shape != correction.graph_basis.shape[:2]:
        raise ValueError("structural correction node count changed")
    displacement = structural_displacement(
        correction.graph_basis, correction.persistent_rest_coefficients
    )
    zero_correction = bool(
        np.array_equal(
            correction.persistent_rest_coefficients,
            np.zeros_like(correction.persistent_rest_coefficients),
        )
    )
    corrected_positions = positions.copy() if zero_correction else positions + displacement
    corrected_lengths = released.copy()
    object_edges = edges[:num_object_springs]
    if zero_correction:
        object_lengths = released[:num_object_springs].astype(float)
    else:
        object_lengths = np.linalg.norm(
            corrected_positions[object_edges[:, 0]]
            - corrected_positions[object_edges[:, 1]],
            axis=1,
        )
        corrected_lengths[:num_object_springs] = object_lengths.astype(
            corrected_lengths.dtype, copy=False
        )
    released_object = released[:num_object_springs].astype(float)
    relative_strain = object_lengths / released_object - 1.0
    maximum_strain = float(np.max(np.abs(relative_strain), initial=0.0))
    if maximum_strain > correction.allowed_edge_strain + 1e-12:
        raise ValueError(
            "structural correction exceeds allowed edge strain: "
            f"{maximum_strain:.6f} > {correction.allowed_edge_strain:.6f}"
        )
    if np.any(object_lengths <= 1e-10) or not np.all(np.isfinite(corrected_positions)):
        raise ValueError("structural correction produced collapsed or nonfinite geometry")
    inverted_tetrahedra = 0
    minimum_volume_ratio = None
    if len(correction.validity_tetrahedra):
        nominal_volume = _signed_tetrahedron_volumes(
            positions.astype(float), correction.validity_tetrahedra
        )
        corrected_volume = _signed_tetrahedron_volumes(
            corrected_positions.astype(float), correction.validity_tetrahedra
        )
        active = np.abs(nominal_volume) > 1e-12
        ratios = np.ones_like(nominal_volume)
        ratios[active] = corrected_volume[active] / nominal_volume[active]
        inverted_tetrahedra = int(np.sum(ratios[active] <= 0.0))
        minimum_volume_ratio = float(np.min(ratios[active], initial=1.0))
        if inverted_tetrahedra:
            raise ValueError("structural correction inverted validity tetrahedra")
    self_intersections = 0
    if len(correction.surface_triangles):
        nominal_intersections = _triangle_intersection_count(
            positions.astype(float), correction.surface_triangles
        )
        corrected_intersections = _triangle_intersection_count(
            corrected_positions.astype(float), correction.surface_triangles
        )
        self_intersections = max(0, corrected_intersections - nominal_intersections)
        if self_intersections:
            raise ValueError("structural correction introduced surface self-intersections")
    absolute = np.abs(relative_strain)
    diagnostics = {
        "zero_correction": zero_correction,
        "maximum_absolute_edge_strain": maximum_strain,
        "absolute_edge_strain_percentiles": {
            str(percentile): float(np.percentile(absolute, percentile))
            for percentile in (50, 90, 95, 99)
        },
        "support_anchor_maximum_displacement_m": float(
            np.max(
                np.linalg.norm(displacement[correction.support_node_indices], axis=1),
                initial=0.0,
            )
        ),
        "validity_tetrahedron_count": len(correction.validity_tetrahedra),
        "minimum_signed_volume_ratio": minimum_volume_ratio,
        "inverted_tetrahedron_count": inverted_tetrahedra,
        "surface_triangle_count": len(correction.surface_triangles),
        "introduced_self_intersection_count": self_intersections,
    }
    return CorrectedRestGeometry(
        rest_positions=corrected_positions,
        rest_lengths=corrected_lengths,
        persistent_displacement=displacement,
        diagnostics=diagnostics,
    )


def identity_structural_twin_correction(
    nominal_rest_positions: np.ndarray,
    springs: np.ndarray,
    nominal_rest_lengths: np.ndarray,
    *,
    num_object_springs: int,
    graph_basis: np.ndarray,
    graph_frequencies: np.ndarray,
    session_ids: Sequence[str],
    support_node_indices: Sequence[int] = (),
    surface_triangles: np.ndarray | None = None,
    validity_tetrahedra: np.ndarray | None = None,
    source_checksums: Mapping[str, str],
    allowed_edge_strain: float = 0.10,
) -> StructuralTwinCorrection:
    """Construct a MAP identity artifact for exact backend parity checks."""

    basis = np.asarray(graph_basis, dtype=float)
    rank = basis.shape[2]
    sessions = tuple(
        StructuralSessionCorrection(
            session_id=session_id,
            frame_linear=np.eye(3),
            frame_translation_m=np.zeros(3),
            settled_state_coefficients=np.zeros(rank),
            gravity_correction_mps2=np.zeros(3),
        )
        for session_id in session_ids
    )
    return StructuralTwinCorrection(
        nominal_rest_geometry_hash=nominal_rest_geometry_sha256(
            nominal_rest_positions,
            springs,
            nominal_rest_lengths,
            num_object_springs=num_object_springs,
        ),
        graph_basis=basis,
        graph_frequencies=np.asarray(graph_frequencies, dtype=float),
        persistent_rest_coefficients=np.zeros(rank),
        persistent_coefficient_covariance=None,
        sessions=sessions,
        support_node_indices=np.asarray(tuple(support_node_indices), dtype=np.int64),
        surface_triangles=np.empty((0, 3), dtype=np.int64)
        if surface_triangles is None
        else np.asarray(surface_triangles, dtype=np.int64),
        validity_tetrahedra=np.empty((0, 4), dtype=np.int64)
        if validity_tetrahedra is None
        else np.asarray(validity_tetrahedra, dtype=np.int64),
        support_model={"kind": "declared_node_anchors"},
        allowed_edge_strain=allowed_edge_strain,
        fit_session_ids=tuple(session_ids),
        information_boundary={
            "persistent_fit_uses_o_minus_only": True,
            "future_frames_used_for_fit": False,
            "target_outcomes_used_for_selection": False,
            "fit_mode": "map",
        },
        source_checksums=source_checksums,
        metadata={"identity_artifact": True, "posterior_stage": "deferred"},
    )


def write_structural_twin_correction(
    output_dir: str | Path,
    correction: StructuralTwinCorrection,
) -> dict[str, Any]:
    """Write a checksummed JSON/NPZ structural artifact pair."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "structural_twin_correction.npz"
    manifest_path = output / "structural_twin_correction.json"
    np.savez_compressed(archive_path, **correction._array_payload())
    manifest = {
        **correction._scalar_payload(),
        "artifact_id": correction.artifact_id,
        "array_archive": {
            "path": archive_path.name,
            "sha256": _file_sha256(archive_path),
            "arrays": {
                name: {
                    "shape": list(values.shape),
                    "dtype": values.dtype.str,
                    "sha256": _array_sha256(values),
                }
                for name, values in sorted(correction._array_payload().items())
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "artifact_id": correction.artifact_id,
        "manifest_path": str(manifest_path.resolve()),
        "archive_path": str(archive_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "archive_sha256": manifest["array_archive"]["sha256"],
    }


def load_structural_twin_correction(
    manifest_path: str | Path,
) -> StructuralTwinCorrection:
    """Load and fully validate a typed structural correction artifact."""

    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != STRUCTURAL_CORRECTION_SCHEMA_VERSION:
        raise ValueError("unsupported structural correction schema")
    if manifest.get("artifact_kind") != "StructuralTwinCorrection":
        raise ValueError("wrong structural artifact kind")
    archive_descriptor = manifest["array_archive"]
    archive_path = source.parent / archive_descriptor["path"]
    if _file_sha256(archive_path) != archive_descriptor["sha256"]:
        raise ValueError("structural correction archive checksum mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name, descriptor in archive_descriptor["arrays"].items():
        if name not in arrays or _array_sha256(arrays[name]) != descriptor["sha256"]:
            raise ValueError(f"structural correction array checksum mismatch: {name}")
    session_ids = tuple(map(str, manifest["session_ids"]))
    sessions = tuple(
        StructuralSessionCorrection(
            session_id=session_id,
            frame_linear=arrays["session_frame_linear"][index],
            frame_translation_m=arrays["session_frame_translation_m"][index],
            settled_state_coefficients=arrays[
                "session_settled_state_coefficients"
            ][index],
            gravity_correction_mps2=arrays[
                "session_gravity_correction_mps2"
            ][index],
        )
        for index, session_id in enumerate(session_ids)
    )
    correction = StructuralTwinCorrection(
        nominal_rest_geometry_hash=manifest["nominal_rest_geometry_hash"],
        graph_basis=arrays["graph_basis"],
        graph_frequencies=arrays["graph_frequencies"],
        persistent_rest_coefficients=arrays["persistent_rest_coefficients"],
        persistent_coefficient_covariance=arrays.get(
            "persistent_coefficient_covariance"
        ),
        sessions=sessions,
        support_node_indices=arrays["support_node_indices"],
        surface_triangles=arrays["surface_triangles"],
        validity_tetrahedra=arrays["validity_tetrahedra"],
        support_model=manifest["support_model"],
        allowed_edge_strain=float(manifest["allowed_edge_strain"]),
        fit_session_ids=tuple(map(str, manifest["fit_session_ids"])),
        information_boundary=manifest["information_boundary"],
        source_checksums=manifest["source_checksums"],
        metadata=manifest.get("metadata", {}),
    )
    if correction.artifact_id != manifest["artifact_id"]:
        raise ValueError("structural correction artifact id mismatch")
    return correction
