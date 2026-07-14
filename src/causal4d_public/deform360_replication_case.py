"""Bridge sampled Deform360 observations into official-Warp forecast cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .deform360_replication_graph import build_sparse_graph_for_stratum
from .deform360_replication_warp import (
    Deform360WarpForecastCase,
    sparse_trajectory_chamfer_m,
)
from .deform360_rope_predict import select_visual_contact_patch


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ReplicationWarpObservation:
    """A full-rate physical case plus sparse future visual-hull references."""

    case: Deform360WarpForecastCase
    raw_hull_frame_indices: np.ndarray
    reference_hulls_m: tuple[np.ndarray, ...]
    prefix_endpoint_frame: int
    contact_associations: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        frames = np.asarray(self.raw_hull_frame_indices, dtype=np.int32)
        _require(frames.ndim == 1 and len(frames) >= 1, "hull frames are invalid")
        _require(np.all(np.diff(frames) > 0), "hull frames are not increasing")
        _require(
            int(frames[0]) == self.prefix_endpoint_frame,
            "first hull is not the prefix endpoint",
        )
        _require(len(self.reference_hulls_m) == len(frames), "hull count differs")
        _require(
            len(self.case.controller_positions_m)
            > int(frames[-1]) - self.prefix_endpoint_frame,
            "case does not reach the final hull",
        )
        copied = frames.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "raw_hull_frame_indices", copied)


def _robot_arrays(state: Any) -> tuple[np.ndarray, np.ndarray]:
    openings = np.asarray(state.openings, dtype=np.float64)
    transforms = np.asarray(state.T_worlds, dtype=np.float64)
    if openings.ndim == 1:
        openings = openings[:, None]
    if transforms.ndim == 3:
        transforms = transforms[:, None]
    _require(openings.ndim == 2, "robot openings must be (T,C)")
    _require(
        transforms.shape == (*openings.shape, 4, 4),
        "robot transforms must be (T,C,4,4)",
    )
    return openings, transforms


def build_replication_warp_observation(
    episode_dir: str | Path,
    episode_id: str,
    stratum: str,
    raw_hull_frame_indices: Sequence[int],
    reference_hulls_m: Sequence[np.ndarray],
    contact_active_full_rate: np.ndarray,
    *,
    dt_seconds: float = 1.0 / 30.0,
    selected_taxel_count: int = 8,
) -> ReplicationWarpObservation:
    """Attach released gripper taxels to a prefix graph without future geometry."""

    directory = Path(episode_dir).resolve()
    frames = np.asarray(raw_hull_frame_indices, dtype=np.int32)
    hulls = tuple(np.asarray(hull, dtype=np.float64) for hull in reference_hulls_m)
    _require(len(frames) == len(hulls) >= 1, "hull frames and values differ")
    graph = build_sparse_graph_for_stratum(hulls[0], stratum)
    try:
        from deform360.processing.control_points_stage import gripper_taxel_points
        from deform360.robot import load_robot_state
    except ImportError as error:  # pragma: no cover - host integration
        raise RuntimeError("the pinned Deform360 runtime is required") from error
    state = load_robot_state(directory / "robot" / "robot.npz")
    openings, transforms = _robot_arrays(state)
    schedule = np.asarray(contact_active_full_rate, dtype=bool)
    _require(schedule.shape == openings.shape, "contact schedule differs from robot")
    prefix_endpoint = int(frames[0])
    _require(0 <= prefix_endpoint < len(openings) - 1, "prefix endpoint is invalid")
    selected_taxels = []
    offsets = []
    nodes = []
    rest_lengths = []
    associations = []
    for axis in range(openings.shape[1]):
        taxels = gripper_taxel_points(
            float(openings[prefix_endpoint, axis]),
            transforms[prefix_endpoint, axis],
        )
        selected, patch, node, offset, diagnostics = select_visual_contact_patch(
            taxels,
            graph.positions_m,
            taxel_count=selected_taxel_count,
        )
        registered_patch = patch + offset
        rest = max(
            float(np.linalg.norm(graph.positions_m[node] - registered_patch)), 1e-4
        )
        selected_taxels.append(selected)
        offsets.append(offset)
        nodes.append(node)
        rest_lengths.append(rest)
        associations.append(
            {
                "robot_axis": axis,
                "selected_taxel_indices": selected.astype(int).tolist(),
                "contact_node_index": node,
                "contact_patch_world_m": patch.tolist(),
                "registered_contact_patch_world_m": registered_patch.tolist(),
                "contact_offset_m": offset.tolist(),
                "contact_rest_length_m": rest,
                **diagnostics,
            }
        )
    raw_case_frames = np.arange(prefix_endpoint, len(openings), dtype=np.int32)
    controllers = np.empty(
        (len(raw_case_frames), openings.shape[1], 3), dtype=np.float64
    )
    for output_index, raw_frame in enumerate(raw_case_frames):
        for axis in range(openings.shape[1]):
            taxels = gripper_taxel_points(
                float(openings[raw_frame, axis]), transforms[raw_frame, axis]
            )
            controllers[output_index, axis] = np.mean(
                taxels[selected_taxels[axis]], axis=0
            ) + offsets[axis]
    case = Deform360WarpForecastCase(
        episode_id=episode_id,
        graph=graph,
        controller_positions_m=controllers,
        contact_active=schedule[prefix_endpoint:],
        contact_node_indices=tuple(nodes),
        contact_rest_lengths_m=np.asarray(rest_lengths, dtype=np.float64),
        dt_seconds=dt_seconds,
    )
    return ReplicationWarpObservation(
        case=case,
        raw_hull_frame_indices=frames,
        reference_hulls_m=hulls,
        prefix_endpoint_frame=prefix_endpoint,
        contact_associations=tuple(associations),
    )


def score_replication_warp_prediction(
    observation: ReplicationWarpObservation,
    full_rate_prediction_m: np.ndarray,
) -> dict[str, object]:
    """Score untouched future hull frames, excluding the fitted prefix endpoint."""

    prediction = np.asarray(full_rate_prediction_m, dtype=np.float64)
    _require(
        len(observation.raw_hull_frame_indices) >= 2,
        "future hull references are required for scoring",
    )
    relative = (
        observation.raw_hull_frame_indices[1:] - observation.prefix_endpoint_frame
    )
    _require(
        prediction.ndim == 3 and int(relative[-1]) < len(prediction),
        "prediction does not cover the reference future",
    )
    return sparse_trajectory_chamfer_m(
        observation.reference_hulls_m[1:], prediction[relative]
    )


def score_constant_persistence(
    observation: ReplicationWarpObservation,
) -> dict[str, object]:
    """Score a prefix-end graph held fixed through the future."""

    _require(
        len(observation.reference_hulls_m) >= 2,
        "future hull references are required for scoring",
    )
    count = len(observation.reference_hulls_m) - 1
    prediction = np.repeat(
        observation.case.graph.positions_m[None], count, axis=0
    )
    return sparse_trajectory_chamfer_m(
        observation.reference_hulls_m[1:], prediction
    )


__all__ = [
    "ReplicationWarpObservation",
    "build_replication_warp_observation",
    "score_constant_persistence",
    "score_replication_warp_prediction",
]
