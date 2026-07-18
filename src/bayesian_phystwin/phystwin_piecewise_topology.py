"""Typed piecewise-topology proposals for official PhysTwin Warp rollouts."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .phystwin_graph import (
    PhysTwinPiecewiseSpringGraphConfig,
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    TransferredSpringField,
    build_phystwin_spring_graph,
    build_piecewise_phystwin_spring_graph,
    transfer_teacher_spring_field,
)


PIECEWISE_TOPOLOGY_CONTRACT = "phystwin-piecewise-topology-v1"


@dataclass(frozen=True)
class PiecewiseTopologyArtifact:
    """A complete graph and spring initialization for one proposal."""

    graph: PhysTwinSpringGraph
    reference_spring_y: np.ndarray
    region_assignments: np.ndarray
    object_radii: np.ndarray
    object_max_neighbours: np.ndarray
    transfer: TransferredSpringField
    diagnostics: dict[str, object]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_topology_diagnostics(graph: PhysTwinSpringGraph) -> dict[str, object]:
    """Return connectivity and degree diagnostics for object springs."""

    if graph.num_object_points is None:
        raise ValueError("topology diagnostics require num_object_points")
    count = int(graph.num_object_points)
    parent = np.arange(count, dtype=np.int64)
    degree = np.zeros(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first_raw, second_raw in graph.springs[: graph.num_object_springs]:
        first, second = int(first_raw), int(second_raw)
        if not 0 <= first < count or not 0 <= second < count:
            raise ValueError("object spring crosses the controller boundary")
        degree[first] += 1
        degree[second] += 1
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root
    component_count = len({find(index) for index in range(count)})
    return {
        "object_component_count": int(component_count),
        "isolated_object_point_count": int(np.sum(degree == 0)),
        "minimum_object_degree": int(np.min(degree)),
        "median_object_degree": float(np.median(degree)),
        "maximum_object_degree": int(np.max(degree)),
        "object_spring_count": int(graph.num_object_springs),
        "controller_spring_count": int(len(graph.springs) - graph.num_object_springs),
    }


def build_piecewise_topology_candidate(
    structure_points: np.ndarray,
    controller_points: np.ndarray,
    region_assignments: np.ndarray,
    teacher_spring_y: Sequence[float] | np.ndarray,
    *,
    teacher_config: PhysTwinSpringGraphConfig,
    radius_multipliers: Sequence[float],
    neighbour_multipliers: Sequence[float],
    object_log_scale: float = 0.0,
    controller_log_scale: float = 0.0,
) -> PiecewiseTopologyArtifact:
    """Build one bounded topology proposal around the released teacher."""

    assignments = np.asarray(region_assignments, dtype=np.int64).reshape(-1)
    if len(assignments) != len(structure_points) or np.any(assignments < 0):
        raise ValueError("region assignments must label every structure point")
    region_count = int(np.max(assignments)) + 1
    radius_scale = np.asarray(radius_multipliers, dtype=float).reshape(-1)
    neighbour_scale = np.asarray(neighbour_multipliers, dtype=float).reshape(-1)
    if len(radius_scale) != region_count or len(neighbour_scale) != region_count:
        raise ValueError("topology multipliers must match the region count")
    if (
        np.any(~np.isfinite(radius_scale))
        or np.any(radius_scale <= 0.0)
        or np.any(~np.isfinite(neighbour_scale))
        or np.any(neighbour_scale <= 0.0)
    ):
        raise ValueError("topology multipliers must be finite and positive")
    if not np.isfinite(object_log_scale) or not np.isfinite(controller_log_scale):
        raise ValueError("spring log scales must be finite")

    teacher = build_phystwin_spring_graph(
        structure_points,
        controller_points,
        config=teacher_config,
    )
    radii = teacher_config.object_radius * radius_scale
    maximums = np.maximum(
        2,
        np.rint(teacher_config.object_max_neighbours * neighbour_scale).astype(int),
    )
    candidate = build_piecewise_phystwin_spring_graph(
        structure_points,
        controller_points,
        assignments,
        config=PhysTwinPiecewiseSpringGraphConfig(
            object_radii=tuple(float(value) for value in radii),
            object_max_neighbours=tuple(int(value) for value in maximums),
            controller_radius=teacher_config.controller_radius,
            controller_max_neighbours=teacher_config.controller_max_neighbours,
        ),
    )
    transfer = transfer_teacher_spring_field(teacher, candidate, teacher_spring_y)
    reference = transfer.spring_y.copy()
    reference[: candidate.num_object_springs] *= float(np.exp(object_log_scale))
    reference[candidate.num_object_springs :] *= float(
        np.exp(controller_log_scale)
    )
    if np.any(~np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError("scaled candidate spring field is invalid")
    diagnostics = object_topology_diagnostics(candidate)
    return PiecewiseTopologyArtifact(
        graph=candidate,
        reference_spring_y=reference,
        region_assignments=assignments.astype(np.int32),
        object_radii=radii.astype(np.float64),
        object_max_neighbours=maximums.astype(np.int32),
        transfer=transfer,
        diagnostics=diagnostics,
    )


def write_piecewise_topology_artifact(
    path: str | Path,
    artifact: PiecewiseTopologyArtifact,
) -> dict[str, object]:
    """Write a non-pickle topology artifact and return its file identity."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = artifact.graph
    np.savez_compressed(
        output,
        contract=np.asarray(PIECEWISE_TOPOLOGY_CONTRACT),
        vertices=np.asarray(graph.vertices, dtype=np.float32),
        springs=np.asarray(graph.springs, dtype=np.int32),
        rest_lengths=np.asarray(graph.rest_lengths, dtype=np.float32),
        masses=np.asarray(graph.masses, dtype=np.float32),
        num_object_springs=np.asarray(graph.num_object_springs, dtype=np.int64),
        num_object_points=np.asarray(graph.num_object_points, dtype=np.int64),
        reference_spring_y=np.asarray(artifact.reference_spring_y, dtype=np.float32),
        transferred_teacher_spring_y=np.asarray(
            artifact.transfer.spring_y, dtype=np.float32
        ),
        region_assignments=np.asarray(artifact.region_assignments, dtype=np.int32),
        object_radii=np.asarray(artifact.object_radii, dtype=np.float64),
        object_max_neighbours=np.asarray(
            artifact.object_max_neighbours, dtype=np.int32
        ),
        exact_edge_count=np.asarray(artifact.transfer.exact_edge_count, dtype=np.int64),
        interpolated_edge_count=np.asarray(
            artifact.transfer.interpolated_edge_count, dtype=np.int64
        ),
        removed_teacher_edge_count=np.asarray(
            artifact.transfer.removed_teacher_edge_count, dtype=np.int64
        ),
        diagnostics_json=np.asarray(json.dumps(artifact.diagnostics, sort_keys=True)),
    )
    return {"path": str(output), "sha256": _sha256(output)}


def load_piecewise_topology_artifact(path: str | Path) -> PiecewiseTopologyArtifact:
    """Load and validate a typed topology proposal with pickle disabled."""

    with np.load(path, allow_pickle=False) as archive:
        if str(archive["contract"].item()) != PIECEWISE_TOPOLOGY_CONTRACT:
            raise ValueError("unsupported piecewise topology contract")
        graph = PhysTwinSpringGraph(
            vertices=np.asarray(archive["vertices"], dtype=np.float32),
            springs=np.asarray(archive["springs"], dtype=np.int32),
            rest_lengths=np.asarray(archive["rest_lengths"], dtype=np.float32),
            masses=np.asarray(archive["masses"], dtype=np.float32),
            num_object_springs=int(archive["num_object_springs"].item()),
            num_object_points=int(archive["num_object_points"].item()),
        )
        artifact = PiecewiseTopologyArtifact(
            graph=graph,
            reference_spring_y=np.asarray(
                archive["reference_spring_y"], dtype=np.float32
            ),
            region_assignments=np.asarray(
                archive["region_assignments"], dtype=np.int32
            ),
            object_radii=np.asarray(archive["object_radii"], dtype=np.float64),
            object_max_neighbours=np.asarray(
                archive["object_max_neighbours"], dtype=np.int32
            ),
            transfer=TransferredSpringField(
                spring_y=np.asarray(
                    archive["transferred_teacher_spring_y"], dtype=np.float32
                ),
                exact_edge_count=int(archive["exact_edge_count"].item()),
                interpolated_edge_count=int(archive["interpolated_edge_count"].item()),
                removed_teacher_edge_count=int(
                    archive["removed_teacher_edge_count"].item()
                ),
            ),
            diagnostics=json.loads(str(archive["diagnostics_json"].item())),
        )
    if (
        len(graph.springs) != len(graph.rest_lengths)
        or len(graph.springs) != len(artifact.reference_spring_y)
        or len(graph.springs) != len(artifact.transfer.spring_y)
    ):
        raise ValueError("piecewise topology spring arrays disagree")
    if len(graph.vertices) != len(graph.masses):
        raise ValueError("piecewise topology vertex arrays disagree")
    if len(artifact.region_assignments) != graph.num_object_points:
        raise ValueError("piecewise topology region assignments disagree")
    if np.any(~np.isfinite(graph.vertices)) or np.any(~np.isfinite(graph.rest_lengths)):
        raise ValueError("piecewise topology contains non-finite geometry")
    if np.any(graph.rest_lengths <= 0.0) or np.any(artifact.reference_spring_y <= 0.0):
        raise ValueError("piecewise topology contains non-positive spring data")
    if object_topology_diagnostics(graph) != artifact.diagnostics:
        raise ValueError("piecewise topology diagnostics do not reproduce")
    return artifact


def build_piecewise_topology_from_files(
    final_data_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    partition_path: str | Path,
    output_path: str | Path,
    *,
    radius_multipliers: Sequence[float],
    neighbour_multipliers: Sequence[float],
    object_log_scale: float = 0.0,
    controller_log_scale: float = 0.0,
) -> dict[str, object]:
    """Build a proposal from future-blind PhysTwin prefix inputs."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in GPU environment
        raise RuntimeError("piecewise topology export requires torch") from error
    with Path(final_data_path).open("rb") as handle:
        final_data = pickle.load(handle)
    with Path(optimal_params_path).open("rb") as handle:
        optimal = pickle.load(handle)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older Torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(final_data, Mapping) or not isinstance(optimal, Mapping):
        raise ValueError("PhysTwin inputs must contain dictionaries")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("PhysTwin checkpoint must contain a dictionary")
    with np.load(partition_path, allow_pickle=False) as partition:
        assignments = np.asarray(partition["part_assignments"], dtype=np.int32)
    structure_points = np.concatenate(
        (
            np.asarray(final_data["object_points"])[0],
            np.asarray(final_data["surface_points"]),
            np.asarray(final_data["interior_points"]),
        ),
        axis=0,
    )
    artifact = build_piecewise_topology_candidate(
        structure_points,
        np.asarray(final_data["controller_points"])[0],
        assignments,
        np.asarray(checkpoint["spring_Y"], dtype=np.float32),
        teacher_config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
        radius_multipliers=radius_multipliers,
        neighbour_multipliers=neighbour_multipliers,
        object_log_scale=object_log_scale,
        controller_log_scale=controller_log_scale,
    )
    identity = write_piecewise_topology_artifact(output_path, artifact)
    return {
        "schema_version": 1,
        "contract": PIECEWISE_TOPOLOGY_CONTRACT,
        "artifact": identity,
        "inputs": {
            "final_data": {"path": str(Path(final_data_path).resolve()), "sha256": _sha256(final_data_path)},
            "optimal_params": {"path": str(Path(optimal_params_path).resolve()), "sha256": _sha256(optimal_params_path)},
            "checkpoint": {"path": str(Path(checkpoint_path).resolve()), "sha256": _sha256(checkpoint_path)},
            "partition": {"path": str(Path(partition_path).resolve()), "sha256": _sha256(partition_path)},
        },
        "search_coordinates": {
            "radius_multipliers": [float(value) for value in radius_multipliers],
            "neighbour_multipliers": [float(value) for value in neighbour_multipliers],
            "object_log_scale": float(object_log_scale),
            "controller_log_scale": float(controller_log_scale),
        },
        "diagnostics": artifact.diagnostics,
        "transfer": {
            "exact_edge_count": artifact.transfer.exact_edge_count,
            "interpolated_edge_count": artifact.transfer.interpolated_edge_count,
            "removed_teacher_edge_count": artifact.transfer.removed_teacher_edge_count,
        },
        "future_observations_used": False,
    }
