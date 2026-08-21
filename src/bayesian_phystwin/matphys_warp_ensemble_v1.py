"""Official-PhysTwin replay contracts for target-excluded MatPhys ensembles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from .matphys_fold_ensemble_v1 import trajectory_ensemble_moments

MATPHYS_WARP_ENSEMBLE_SCHEMA: Final = (
    "bayesian-phystwin.matphys-warp-trajectory-ensemble"
)
MATPHYS_WARP_ENSEMBLE_VERSION: Final = 1
MATPHYS_WARP_ENSEMBLE_PROTOCOL: Final = (
    "target-excluded-matphys-fields-official-phystwin-warp-v1"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Hash one ordinary file without normalizing its content."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MatPhysWarpReplayGraph:
    """Canonical object graph plus deterministic controller attachments."""

    vertices: npt.NDArray[np.float32]
    springs: npt.NDArray[np.int32]
    rest_lengths: npt.NDArray[np.float32]
    masses: npt.NDArray[np.float32]
    num_object_springs: int
    num_controller_springs: int
    controller_group_count: int


@dataclass(frozen=True)
class MatPhysSpringEnsemble:
    """Validated MatPhys fields aligned to one canonical object graph."""

    incumbent_spring_y_pa: npt.NDArray[np.float32]
    member_spring_y_pa: npt.NDArray[np.float32]
    graph_points_m: npt.NDArray[np.float32]
    graph_edges: npt.NDArray[np.int64]


def _finite_points(value: np.ndarray, *, name: str) -> npt.NDArray[np.float32]:
    points = np.asarray(value, dtype=np.float32)
    _require(points.ndim == 2 and points.shape[1] == 3, f"{name} must have shape (N,3)")
    _require(len(points) > 0 and np.all(np.isfinite(points)), f"{name} must be finite")
    return points


def load_matphys_spring_ensemble(
    path: str | Path,
    *,
    expected_member_count: int,
) -> MatPhysSpringEnsemble:
    """Load the exact graph-aligned spring arrays emitted by the fold producer."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "incumbent_spring_y_pa",
            "member_spring_y_pa",
            "graph_points_m",
            "graph_edges",
        }
        missing = required - set(archive.files)
        _require(
            not missing, "spring ensemble is missing: " + ", ".join(sorted(missing))
        )
        incumbent = np.asarray(archive["incumbent_spring_y_pa"], dtype=np.float32)
        members = np.asarray(archive["member_spring_y_pa"], dtype=np.float32)
        points = _finite_points(archive["graph_points_m"], name="graph_points_m")
        edges = np.asarray(archive["graph_edges"], dtype=np.int64)
    _require(
        incumbent.ndim == 1 and len(incumbent) > 0, "incumbent spring field is invalid"
    )
    _require(
        members.shape == (expected_member_count, len(incumbent)),
        "member spring fields do not match the registered ensemble",
    )
    _require(
        np.all(np.isfinite(incumbent))
        and np.all(incumbent > 0.0)
        and np.all(np.isfinite(members))
        and np.all(members > 0.0),
        "spring fields must be finite and positive",
    )
    _require(
        edges.shape == (len(incumbent), 2)
        and np.all(edges >= 0)
        and np.all(edges < len(points)),
        "graph edges do not match the spring field",
    )
    return MatPhysSpringEnsemble(
        incumbent_spring_y_pa=incumbent,
        member_spring_y_pa=members,
        graph_points_m=points,
        graph_edges=edges,
    )


def load_registered_replay_graph(
    path: str | Path,
    *,
    expected_points_m: np.ndarray,
    expected_edges: np.ndarray,
    controller_reference_m: np.ndarray,
    controller_radius_m: float,
    controller_patch_size: int,
) -> MatPhysWarpReplayGraph:
    """Load the frozen object graph and rebuild only controller attachments.

    This is the relevant subset of the released Causal4D reusable-graph adapter.
    Object vertices, topology, rest lengths, and masses are never reconstructed.
    """

    _require(
        np.isfinite(controller_radius_m) and controller_radius_m > 0.0,
        "controller radius must be finite and positive",
    )
    _require(controller_patch_size >= 1, "controller patch size must be positive")
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "vertices",
            "springs",
            "rest_lengths",
            "masses",
            "contact_anchor_indices",
        }
        missing = required - set(archive.files)
        _require(
            not missing, "registered graph is missing: " + ", ".join(sorted(missing))
        )
        vertices = _finite_points(archive["vertices"], name="vertices")
        springs = np.asarray(archive["springs"], dtype=np.int32)
        rest_lengths = np.asarray(archive["rest_lengths"], dtype=np.float32)
        masses = np.asarray(archive["masses"], dtype=np.float32)
        anchors = np.asarray(archive["contact_anchor_indices"], dtype=np.int64)

    expected_points = np.asarray(expected_points_m, dtype=np.float32)
    expected_links = np.asarray(expected_edges, dtype=np.int64)
    _require(
        np.array_equal(vertices, expected_points),
        "registered graph vertices differ from the MatPhys graph",
    )
    _require(
        np.array_equal(springs.astype(np.int64), expected_links),
        "registered graph edges differ from the MatPhys graph",
    )
    _require(
        rest_lengths.shape == (len(springs),)
        and np.all(np.isfinite(rest_lengths))
        and np.all(rest_lengths > 0.0),
        "registered rest lengths are invalid",
    )
    _require(
        masses.shape == (len(vertices),)
        and np.all(np.isfinite(masses))
        and np.all(masses > 0.0),
        "registered masses are invalid",
    )
    _require(
        anchors.ndim == 1
        and len(anchors) > 0
        and np.all(anchors >= 0)
        and np.all(anchors < len(vertices)),
        "registered contact anchors are invalid",
    )
    controls = _finite_points(controller_reference_m, name="controller_reference_m")
    _require(
        len(controls) % len(anchors) == 0,
        "controller points cannot be divided among registered anchors",
    )
    group_size = len(controls) // len(anchors)
    adjacency = coo_matrix(
        (
            np.concatenate((rest_lengths, rest_lengths)),
            (
                np.concatenate((springs[:, 0], springs[:, 1])),
                np.concatenate((springs[:, 1], springs[:, 0])),
            ),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    controller_springs: list[tuple[int, int]] = []
    controller_rest_lengths: list[float] = []
    for group_index, anchor_value in enumerate(anchors):
        start = group_index * group_size
        stop = start + group_size
        group = controls[start:stop]
        controller_distance = cdist(vertices, group)
        nearest_controller = np.argmin(controller_distance, axis=1)
        nearest_distance = controller_distance[
            np.arange(len(vertices)), nearest_controller
        ]
        anchor = int(anchor_value)
        _require(
            nearest_distance[anchor] <= controller_radius_m,
            "registered contact anchor is outside the controller radius",
        )
        graph_distance = np.asarray(dijkstra(adjacency, indices=anchor, directed=False))
        admissible = np.flatnonzero(
            np.isfinite(graph_distance) & (nearest_distance <= controller_radius_m)
        )
        order = np.lexsort(
            (
                admissible,
                nearest_distance[admissible],
                graph_distance[admissible],
            )
        )
        selected = admissible[order[:controller_patch_size]]
        if anchor not in selected:
            selected = np.concatenate((np.asarray([anchor]), selected[:-1]))
        for node_index in selected:
            local_index = int(nearest_controller[int(node_index)])
            controller_springs.append(
                (len(vertices) + start + local_index, int(node_index))
            )
            controller_rest_lengths.append(float(nearest_distance[int(node_index)]))

    controller_edges = np.asarray(controller_springs, dtype=np.int32)
    controller_rest = np.asarray(controller_rest_lengths, dtype=np.float32)
    _require(len(controller_edges) > 0, "registered graph has no controller springs")
    return MatPhysWarpReplayGraph(
        vertices=np.concatenate((vertices, controls), axis=0).astype(np.float32),
        springs=np.concatenate((springs, controller_edges), axis=0).astype(np.int32),
        rest_lengths=np.concatenate((rest_lengths, controller_rest)).astype(np.float32),
        masses=np.concatenate((masses, np.ones(len(controls), dtype=np.float32))),
        num_object_springs=len(springs),
        num_controller_springs=len(controller_edges),
        controller_group_count=len(anchors),
    )


def trajectory_ensemble_arrays(
    incumbent_trajectory_m: np.ndarray,
    member_trajectories_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate replay outputs and expose duplicate-safe physical moments."""

    incumbent = np.asarray(incumbent_trajectory_m)
    members = np.asarray(member_trajectories_m)
    _require(
        incumbent.ndim == 3 and incumbent.shape[-1] == 3,
        "incumbent trajectory must have shape (T,N,3)",
    )
    _require(
        members.ndim == 4 and members.shape[1:] == incumbent.shape,
        "member trajectories must have shape (M,T,N,3)",
    )
    _require(
        np.issubdtype(incumbent.dtype, np.floating)
        and np.all(np.isfinite(incumbent))
        and np.all(np.isfinite(members)),
        "replay trajectories must be finite",
    )
    moments = trajectory_ensemble_moments(members)
    return {
        "incumbent_trajectory_m": incumbent,
        "member_trajectories_m": members,
        "member_mean_trajectory_m": moments.mean_m,
        "member_covariance_m2": moments.covariance_m2,
        "unique_member_indices": moments.unique_member_indices,
    }


__all__ = [
    "MATPHYS_WARP_ENSEMBLE_PROTOCOL",
    "MATPHYS_WARP_ENSEMBLE_SCHEMA",
    "MATPHYS_WARP_ENSEMBLE_VERSION",
    "MatPhysSpringEnsemble",
    "MatPhysWarpReplayGraph",
    "file_sha256",
    "load_matphys_spring_ensemble",
    "load_registered_replay_graph",
    "trajectory_ensemble_arrays",
]
