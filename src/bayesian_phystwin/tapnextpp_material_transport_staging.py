"""Outcome-blind staging for fixed material-identity TAPNext++ transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .phystwin_official_evaluation import _nearest_distances
from .phystwin_tapnextpp_competence import PhysTwinTAPNextPPCompetenceConfig
from .tapnextpp_depth_completion import TAPNextPPDepthCompletionConfig
from .tapnextpp_transfer_staging import deterministic_farthest_point_indices

PROVIDER_PROTOCOL_ID = "phystwin-tapnextpp-material-transport-provider-source-v1"
PROVIDER_PROTOCOL_STATUS = "locked-before-provider-source-outcomes"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class MaterialTransportCasePlan:
    """Terminal prefix window and immutable frame-zero graph attachments."""

    tracker_config: PhysTwinTAPNextPPCompetenceConfig
    material_node_indices: np.ndarray
    frame_zero_attachment_distance_m: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.material_node_indices, dtype=np.int64).copy()
        distances = np.asarray(
            self.frame_zero_attachment_distance_m,
            dtype=np.float64,
        ).copy()
        count = len(self.tracker_config.selected_identity_ids)
        _require(nodes.shape == (count,), "material-node shape changed")
        _require(distances.shape == (count,), "attachment-distance shape changed")
        _require(np.all(nodes >= 0), "material nodes must be nonnegative")
        _require(
            np.all(np.isfinite(distances)) and np.all(distances >= 0.0),
            "attachment distances must be finite and nonnegative",
        )
        nodes.setflags(write=False)
        distances.setflags(write=False)
        object.__setattr__(self, "material_node_indices", nodes)
        object.__setattr__(self, "frame_zero_attachment_distance_m", distances)


def validate_material_transport_provider_protocol(protocol: dict[str, Any]) -> None:
    """Validate the frozen new-case provider and material-attachment contract."""

    _require(
        protocol.get("protocol_id") == PROVIDER_PROTOCOL_ID,
        "material-transport provider protocol ID changed",
    )
    _require(
        protocol.get("status") == PROVIDER_PROTOCOL_STATUS,
        "material-transport provider protocol is not locked",
    )
    cases = protocol.get("fixed_source_cases")
    excluded = protocol.get("excluded_prior_assimilation_cases")
    _require(
        isinstance(cases, list)
        and len(cases) == 14
        and len(set(cases)) == 14
        and all(isinstance(case, str) and case for case in cases),
        "material-transport source panel changed",
    )
    _require(
        isinstance(excluded, list)
        and len(excluded) == 8
        and not set(cases).intersection(excluded),
        "source panel overlaps the opened sparse-assimilation panel",
    )
    selection = protocol.get("selection")
    _require(isinstance(selection, dict), "selection config is missing")
    _require(int(selection.get("window_frames", 0)) >= 2, "window is too short")
    _require(
        int(selection.get("maximum_query_count", 0))
        >= int(selection.get("minimum_query_count", 1))
        >= 3,
        "query-count contract is invalid",
    )
    cameras = selection.get("selected_cameras")
    _require(
        isinstance(cameras, list)
        and len(cameras) >= 2
        and len(cameras) == len(set(cameras)),
        "selected-camera contract is invalid",
    )
    dynamic = {
        "case_name": "validation_case",
        "source_frame_start": 0,
        "source_frame_end_exclusive": int(selection["window_frames"]),
        "selected_identity_ids": tuple(
            range(int(selection["minimum_query_count"]))
        ),
        "selected_cameras": tuple(int(camera) for camera in cameras),
    }
    PhysTwinTAPNextPPCompetenceConfig(
        **dynamic,
        **protocol["tracker_fixed_config"],
    )
    _require(
        protocol.get("depth_completion_config")
        == TAPNextPPDepthCompletionConfig().__dict__,
        "depth-completion config changed",
    )
    retention = protocol.get("retention_policy")
    _require(
        isinstance(retention, dict)
        and retention.get("all_fourteen_cases_required") is True
        and retention.get("technical_failures_retained") is True
        and retention.get("failed_cases_replaced") is False,
        "failure-retention policy changed",
    )
    _require(
        protocol.get("information_boundary", {}).get("held_v8_accessed") is False,
        "held-v8 boundary changed",
    )


def plan_material_transport_case(
    case_name: str,
    manual_tracks_world_m: np.ndarray,
    physical_trajectory_m: np.ndarray,
    *,
    train_end_frame_exclusive: int,
    protocol: dict[str, Any],
) -> MaterialTransportCasePlan:
    """Select fixed identities and nodes without opening a withheld prefix row."""

    validate_material_transport_provider_protocol(protocol)
    _require(case_name in protocol["fixed_source_cases"], "case is outside panel")
    tracks = np.asarray(manual_tracks_world_m, dtype=np.float64)
    physical = np.asarray(physical_trajectory_m, dtype=np.float64)
    _require(
        tracks.ndim == 3 and tracks.shape[2] == 3,
        "manual tracks must have shape (T, N, 3)",
    )
    _require(
        physical.ndim == 3 and physical.shape[2] == 3 and len(physical),
        "physical trajectory must have shape (T, M, 3)",
    )
    selection = protocol["selection"]
    window_frames = int(selection["window_frames"])
    source_start = train_end_frame_exclusive - window_frames
    _require(
        0 <= source_start < train_end_frame_exclusive <= len(tracks),
        "terminal provider window lies outside manual tracks",
    )
    _require(
        train_end_frame_exclusive <= len(physical),
        "terminal provider window lies outside physical trajectory",
    )
    finite = np.flatnonzero(
        np.all(np.isfinite(tracks[0]), axis=1)
        & np.all(np.isfinite(tracks[source_start]), axis=1)
    )
    minimum = int(selection["minimum_query_count"])
    _require(len(finite) >= minimum, "too few frame-zero material identities")
    count = min(int(selection["maximum_query_count"]), len(finite))
    selected_local = deterministic_farthest_point_indices(tracks[0, finite], count)
    identities = finite[selected_local]
    distances, nodes = _nearest_distances(
        physical[0],
        tracks[0, identities],
        p=2,
    )
    _require(
        np.all(distances <= float(selection["maximum_frame_zero_attachment_m"])),
        "a frame-zero identity is too far from the physical graph",
    )
    tracker_config = PhysTwinTAPNextPPCompetenceConfig(
        case_name=case_name,
        source_frame_start=source_start,
        source_frame_end_exclusive=train_end_frame_exclusive,
        selected_identity_ids=tuple(int(identity) for identity in identities),
        selected_cameras=tuple(
            int(camera) for camera in selection["selected_cameras"]
        ),
        **protocol["tracker_fixed_config"],
    )
    return MaterialTransportCasePlan(
        tracker_config=tracker_config,
        material_node_indices=nodes,
        frame_zero_attachment_distance_m=distances,
    )
