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
MATPHYS_WARP_ENSEMBLE_VERSION: Final = 2
MATPHYS_WARP_ENSEMBLE_PROTOCOL: Final = (
    "target-excluded-matphys-fields-official-phystwin-warp-v2"
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


def hierarchical_trajectory_ensemble_arrays(
    incumbent_replicates_m: np.ndarray,
    member_replicates_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Separate checkpoint disagreement from official-Warp replay variation.

    The total covariance follows the law of total variance: covariance across
    per-checkpoint replay means plus the mean covariance across repeated Warp
    executions within each checkpoint. The incumbent is kept separate because
    downstream Bayesian-PhysTwin uses an unchanged stronger baseline mean.
    """

    incumbent = np.asarray(incumbent_replicates_m)
    members = np.asarray(member_replicates_m)
    _require(
        incumbent.ndim == 4 and incumbent.shape[0] >= 2 and incumbent.shape[-1] == 3,
        "incumbent_replicates_m must have shape (R,T,N,3) with R>=2",
    )
    _require(
        members.ndim == 5
        and members.shape[0] >= 2
        and members.shape[1] >= 2
        and members.shape[2:] == incumbent.shape[1:],
        "member_replicates_m must have shape (M,R,T,N,3) with M,R>=2",
    )
    _require(
        np.issubdtype(incumbent.dtype, np.floating)
        and np.issubdtype(members.dtype, np.floating)
        and np.all(np.isfinite(incumbent))
        and np.all(np.isfinite(members)),
        "hierarchical replay trajectories must be finite floating arrays",
    )
    member_means = np.mean(members.astype(np.float64), axis=1)
    ensemble_mean = np.mean(member_means, axis=0)
    between_centered = member_means - ensemble_mean[None]
    between = np.einsum(
        "mtni,mtnj->tnij", between_centered, between_centered, optimize=True
    ) / len(member_means)
    within_centered = members.astype(np.float64) - member_means[:, None]
    within_by_member = (
        np.einsum("mrtni,mrtnj->mtnij", within_centered, within_centered, optimize=True)
        / members.shape[1]
    )
    within = np.mean(within_by_member, axis=0)
    total = between + within
    incumbent_mean = np.mean(incumbent.astype(np.float64), axis=0)
    incumbent_centered = incumbent.astype(np.float64) - incumbent_mean[None]
    incumbent_replay = np.einsum(
        "rtni,rtnj->tnij", incumbent_centered, incumbent_centered, optimize=True
    ) / len(incumbent)
    for name, covariance in {
        "between-member": between,
        "within-member": within,
        "total": total,
        "incumbent replay": incumbent_replay,
    }.items():
        covariance[...] = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
        _require(
            float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-12,
            f"{name} covariance is not PSD",
        )
    return {
        "incumbent_replicates_m": incumbent,
        "incumbent_replay_mean_m": incumbent_mean,
        "incumbent_replay_covariance_m2": incumbent_replay,
        "member_replicates_m": members,
        "member_replay_means_m": member_means,
        "member_mean_trajectory_m": ensemble_mean,
        "between_member_covariance_m2": between,
        "within_member_replay_covariance_m2": within,
        "member_total_covariance_m2": total,
    }


def baseline_relative_trajectory_ensemble_arrays(
    incumbent_replicates_m: np.ndarray,
    member_trajectories_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Estimate covariance around an intentionally unchanged baseline mean.

    A covariance-only Bayesian-PhysTwin arm does not replace its stronger point
    mean with the MatPhys ensemble mean. Centering fold trajectories around
    that unused ensemble mean would therefore erase a displacement shared by
    every physical member. This estimator instead takes the second moment of
    each member's displacement from the incumbent replay mean and adds a
    separately measured, shared incumbent replay floor.

    Repeating every member is unnecessary under this registered shared-floor
    approximation: numerical replay variation is measured using at least two
    incumbent executions, while each independently trained member is replayed
    once. The output remains raw model-family evidence and still requires
    source-only scalar calibration before a predictive claim.
    """

    incumbent = np.asarray(incumbent_replicates_m)
    members = np.asarray(member_trajectories_m)
    _require(
        incumbent.ndim == 4 and incumbent.shape[0] >= 2 and incumbent.shape[-1] == 3,
        "incumbent_replicates_m must have shape (R,T,N,3) with R>=2",
    )
    _require(
        members.ndim == 4
        and members.shape[0] >= 2
        and members.shape[1:] == incumbent.shape[1:],
        "member_trajectories_m must have shape (M,T,N,3) with M>=2",
    )
    _require(
        np.issubdtype(incumbent.dtype, np.floating)
        and np.issubdtype(members.dtype, np.floating)
        and np.all(np.isfinite(incumbent))
        and np.all(np.isfinite(members)),
        "baseline-relative replay trajectories must be finite floating arrays",
    )
    incumbent64 = incumbent.astype(np.float64)
    members64 = members.astype(np.float64)
    incumbent_mean = np.mean(incumbent64, axis=0)
    replay_centered = incumbent64 - incumbent_mean[None]
    replay_floor = np.einsum(
        "rtni,rtnj->tnij", replay_centered, replay_centered, optimize=True
    ) / len(incumbent64)
    member_delta = members64 - incumbent_mean[None]
    model_second_moment = np.einsum(
        "mtni,mtnj->tnij", member_delta, member_delta, optimize=True
    ) / len(members64)
    total = model_second_moment + replay_floor
    for name, covariance in {
        "incumbent replay": replay_floor,
        "baseline-relative model": model_second_moment,
        "baseline-relative total": total,
    }.items():
        covariance[...] = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
        _require(
            float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-12,
            f"{name} covariance is not PSD",
        )
    return {
        "incumbent_replicates_m": incumbent,
        "incumbent_replay_mean_m": incumbent_mean,
        "incumbent_replay_covariance_m2": replay_floor,
        "member_trajectories_m": members,
        "member_mean_trajectory_m": np.mean(members64, axis=0),
        "member_delta_from_incumbent_m": member_delta,
        "baseline_relative_model_second_moment_m2": model_second_moment,
        "baseline_relative_total_covariance_m2": total,
    }


__all__ = [
    "MATPHYS_WARP_ENSEMBLE_PROTOCOL",
    "MATPHYS_WARP_ENSEMBLE_SCHEMA",
    "MATPHYS_WARP_ENSEMBLE_VERSION",
    "MatPhysSpringEnsemble",
    "MatPhysWarpReplayGraph",
    "baseline_relative_trajectory_ensemble_arrays",
    "file_sha256",
    "hierarchical_trajectory_ensemble_arrays",
    "load_matphys_spring_ensemble",
    "load_registered_replay_graph",
    "trajectory_ensemble_arrays",
]
