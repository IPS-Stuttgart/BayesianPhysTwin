"""Target-free case/window/query selection for TAPNext++ source transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .phystwin_tapnextpp_competence import PhysTwinTAPNextPPCompetenceConfig
from .tapnextpp_depth_completion import TAPNextPPDepthCompletionConfig

TRANSFER_PROTOCOL_ID = "phystwin-tapnextpp-depth-completion-transfer-v1"
TRANSFER_PROTOCOL_STATUS = (
    "locked-before-source-artifact-staging-and-tapnextpp-prediction"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def deterministic_farthest_point_indices(
    points: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select deterministic spatial coverage, starting from the first row."""

    values = np.asarray(points, dtype=np.float64)
    _require(
        values.ndim == 2 and values.shape[1] == 3,
        "points must have shape (N, 3)",
    )
    _require(np.all(np.isfinite(values)), "points are not finite")
    _require(1 <= count <= len(values), "selection count lies outside the points")
    selected = [0]
    distance = np.linalg.norm(values - values[0], axis=1)
    distance[0] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(
            distance,
            np.linalg.norm(values - values[index], axis=1),
        )
        distance[np.asarray(selected, dtype=np.int64)] = -np.inf
    result = np.asarray(selected, dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhysicalMotionWindow:
    """A causal prefix window chosen only from a physical rollout."""

    start_frame: int
    end_frame_exclusive: int
    rms_endpoint_displacement_m: float
    sampled_node_indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.sampled_node_indices, dtype=np.int64).copy()
        indices.setflags(write=False)
        _require(self.start_frame >= 0, "window start must be nonnegative")
        _require(
            self.end_frame_exclusive > self.start_frame + 1,
            "window must contain at least two frames",
        )
        _require(
            np.isfinite(self.rms_endpoint_displacement_m)
            and self.rms_endpoint_displacement_m >= 0.0,
            "window displacement must be finite and nonnegative",
        )
        _require(indices.ndim == 1 and len(indices), "sampled nodes are empty")
        object.__setattr__(self, "sampled_node_indices", indices)


@dataclass(frozen=True)
class TAPNextPPTransferCasePlan:
    """Outcome-blind source case plan frozen before tracker execution."""

    tracker_config: PhysTwinTAPNextPPCompetenceConfig
    physical_window: PhysicalMotionWindow

    @property
    def case_name(self) -> str:
        return self.tracker_config.case_name

    @property
    def selected_identity_ids(self) -> tuple[int, ...]:
        return self.tracker_config.selected_identity_ids


def select_physical_motion_window(
    physical_trajectory_m: np.ndarray,
    *,
    train_end_frame_exclusive: int,
    window_frames: int = 20,
    sampled_node_count: int = 64,
) -> PhysicalMotionWindow:
    """Choose the earliest maximum-displacement window inside the prefix."""

    trajectory = np.asarray(physical_trajectory_m, dtype=np.float64)
    _require(
        trajectory.ndim == 3 and trajectory.shape[2] == 3,
        "physical trajectory must have shape (T, N, 3)",
    )
    _require(np.all(np.isfinite(trajectory)), "physical trajectory is not finite")
    _require(window_frames >= 2, "window must contain at least two frames")
    _require(
        window_frames <= train_end_frame_exclusive <= len(trajectory),
        "training boundary is incompatible with the physical trajectory",
    )
    count = min(sampled_node_count, trajectory.shape[1])
    _require(count >= 1, "physical trajectory contains no nodes")
    sampled = deterministic_farthest_point_indices(trajectory[0], count)
    endpoint_offset = window_frames - 1
    candidate_count = train_end_frame_exclusive - window_frames + 1
    displacement = np.empty(candidate_count, dtype=np.float64)
    for start in range(candidate_count):
        delta = (
            trajectory[start + endpoint_offset, sampled]
            - trajectory[start, sampled]
        )
        displacement[start] = np.sqrt(np.mean(np.sum(np.square(delta), axis=1)))
    maximum = float(np.max(displacement))
    tied = np.flatnonzero(
        np.isclose(displacement, maximum, rtol=1e-12, atol=1e-15)
    )
    start = int(tied[0])
    return PhysicalMotionWindow(
        start_frame=start,
        end_frame_exclusive=start + window_frames,
        rms_endpoint_displacement_m=float(displacement[start]),
        sampled_node_indices=sampled,
    )


def select_query_identity_ids(
    manual_tracks_world_m: np.ndarray,
    *,
    source_frame: int,
    maximum_query_count: int = 4,
    minimum_query_count: int = 3,
) -> np.ndarray:
    """Select spatially diverse identities using the query frame only."""

    tracks = np.asarray(manual_tracks_world_m, dtype=np.float64)
    _require(
        tracks.ndim == 3 and tracks.shape[2] == 3,
        "manual tracks must have shape (T, N, 3)",
    )
    _require(0 <= source_frame < len(tracks), "source frame lies outside tracks")
    finite = np.flatnonzero(np.all(np.isfinite(tracks[source_frame]), axis=1))
    _require(
        len(finite) >= minimum_query_count,
        "too few finite material identities at the source frame",
    )
    count = min(maximum_query_count, len(finite))
    local = deterministic_farthest_point_indices(tracks[source_frame, finite], count)
    selected = finite[local].astype(np.int64, copy=True)
    selected.setflags(write=False)
    return selected


def validate_transfer_protocol(protocol: dict[str, Any]) -> None:
    """Validate the frozen eight-case source-transfer contract."""

    _require(
        protocol.get("protocol_id") == TRANSFER_PROTOCOL_ID,
        "transfer protocol ID changed",
    )
    _require(
        protocol.get("status") == TRANSFER_PROTOCOL_STATUS,
        "transfer protocol is not prediction-locked",
    )
    cases = protocol.get("fixed_source_cases")
    _require(
        isinstance(cases, list)
        and len(cases) == 8
        and len(set(cases)) == len(cases)
        and all(
            isinstance(case, str)
            and bool(case)
            and "/" not in case
            and "\\" not in case
            for case in cases
        ),
        "transfer source cases are invalid",
    )
    selection = protocol.get("selection")
    _require(isinstance(selection, dict), "selection config is missing")
    _require(int(selection.get("window_frames", 0)) >= 2, "window is too short")
    _require(
        int(selection.get("sampled_physical_node_count", 0)) >= 1,
        "sampled physical-node count is invalid",
    )
    _require(
        int(selection.get("maximum_query_count", 0))
        >= int(selection.get("minimum_query_count", 1))
        >= 3,
        "query-count contract is invalid",
    )
    selected_cameras = selection.get("selected_cameras")
    _require(
        isinstance(selected_cameras, list)
        and len(selected_cameras) >= 2
        and len(set(selected_cameras)) == len(selected_cameras),
        "selected-camera contract is invalid",
    )
    tracker_fixed = protocol.get("tracker_fixed_config")
    _require(isinstance(tracker_fixed, dict), "tracker config is missing")
    dynamic = {
        "case_name": "validation_case",
        "source_frame_start": 0,
        "source_frame_end_exclusive": int(selection["window_frames"]),
        "selected_identity_ids": tuple(
            range(int(selection["minimum_query_count"]))
        ),
        "selected_cameras": tuple(int(value) for value in selected_cameras),
    }
    PhysTwinTAPNextPPCompetenceConfig(**dynamic, **tracker_fixed)
    _require(
        protocol.get("depth_completion_config")
        == TAPNextPPDepthCompletionConfig().__dict__,
        "depth-completion config changed",
    )
    gates = protocol.get("per_case_gates")
    _require(
        isinstance(gates, dict)
        and 0.0 < float(gates.get("minimum_supported_fraction", 0.0)) <= 1.0
        and float(gates.get("maximum_identity_rmse_m", 0.0)) > 0.0,
        "per-case transfer gates are invalid",
    )
    aggregate = protocol.get("aggregate_gates")
    _require(
        isinstance(aggregate, dict)
        and 1
        <= int(aggregate.get("minimum_passing_case_count", 0))
        <= len(cases),
        "aggregate transfer gates are invalid",
    )
    retention = protocol.get("retention_policy")
    _require(
        isinstance(retention, dict)
        and retention.get("technical_failures_retained") is True
        and retention.get("failed_cases_replaced") is False
        and retention.get("all_eight_fixed_cases_required_in_final_accounting")
        is True,
        "failure-retention policy changed",
    )


def plan_transfer_case(
    case_name: str,
    physical_trajectory_m: np.ndarray,
    manual_tracks_world_m: np.ndarray,
    *,
    train_end_frame_exclusive: int,
    protocol: dict[str, Any],
) -> TAPNextPPTransferCasePlan:
    """Build one deterministic case plan without opening prefix outcomes."""

    validate_transfer_protocol(protocol)
    _require(
        case_name in protocol["fixed_source_cases"],
        "case lies outside the frozen source panel",
    )
    selection = protocol["selection"]
    window = select_physical_motion_window(
        physical_trajectory_m,
        train_end_frame_exclusive=train_end_frame_exclusive,
        window_frames=int(selection["window_frames"]),
        sampled_node_count=int(selection["sampled_physical_node_count"]),
    )
    identities = select_query_identity_ids(
        manual_tracks_world_m,
        source_frame=window.start_frame,
        maximum_query_count=int(selection["maximum_query_count"]),
        minimum_query_count=int(selection["minimum_query_count"]),
    )
    config = PhysTwinTAPNextPPCompetenceConfig(
        case_name=case_name,
        source_frame_start=window.start_frame,
        source_frame_end_exclusive=window.end_frame_exclusive,
        selected_identity_ids=tuple(int(value) for value in identities),
        selected_cameras=tuple(
            int(value) for value in selection["selected_cameras"]
        ),
        **protocol["tracker_fixed_config"],
    )
    return TAPNextPPTransferCasePlan(
        tracker_config=config,
        physical_window=window,
    )
