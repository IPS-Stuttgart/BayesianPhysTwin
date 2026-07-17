"""Graph-local trust for reusable Deform360 PhysTwin action responses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from .deform360_phystwin_trust import (
    CausalTrustEpisode,
    score_causal_trust_interval,
)
from .deform360_reusable_graph import CanonicalDeform360Graph


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def dynamic_contact_anchor_indices(
    object_points_m: np.ndarray,
    controller_reference_m: np.ndarray,
    *,
    controller_group_size: int,
    maximum_contact_distance_m: float,
) -> np.ndarray:
    """Recover the same per-episode contact anchors used by the Warp adapter."""

    points = np.asarray(object_points_m, dtype=np.float64)
    controls = np.asarray(controller_reference_m, dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and controls.ndim == 2
        and controls.shape[1] == 3
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(controls)),
        "dynamic contact geometry must contain finite 3D points",
    )
    _require(
        controller_group_size >= 1 and len(controls) % controller_group_size == 0,
        "dynamic controller points do not form locked groups",
    )
    _require(
        np.isfinite(maximum_contact_distance_m) and maximum_contact_distance_m > 0.0,
        "dynamic contact radius must be finite and positive",
    )
    anchors = []
    for start in range(0, len(controls), controller_group_size):
        group = controls[start : start + controller_group_size]
        distance = cdist(points, group)
        node, _ = np.unravel_index(np.argmin(distance), distance.shape)
        _require(
            float(np.min(distance)) <= maximum_contact_distance_m,
            "episode contact anchor is outside the controller radius",
        )
        anchors.append(int(node))
    return np.asarray(anchors, dtype=np.int64)


def graph_contact_distance_m(
    graph: CanonicalDeform360Graph,
    *,
    contact_anchor_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return shortest material-graph distance to any registered contact anchor."""

    anchors = (
        np.asarray(graph.contact_anchor_indices, dtype=np.int64)
        if contact_anchor_indices is None
        else np.asarray(contact_anchor_indices, dtype=np.int64)
    )
    _require(
        anchors.ndim == 1
        and len(anchors) > 0
        and np.all(anchors >= 0)
        and np.all(anchors < len(graph.vertices)),
        "graph has no valid contact anchor",
    )
    edges = np.asarray(graph.springs, dtype=np.int64)
    lengths = np.asarray(graph.rest_lengths, dtype=np.float64)
    _require(
        edges.ndim == 2 and edges.shape[1] == 2 and lengths.shape == (len(edges),),
        "graph springs and rest lengths differ",
    )
    adjacency = coo_matrix(
        (
            np.concatenate((lengths, lengths)),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(len(graph.vertices), len(graph.vertices)),
    ).tocsr()
    distance = np.asarray(
        dijkstra(
            adjacency,
            indices=anchors,
            directed=False,
        ),
        dtype=np.float64,
    )
    if distance.ndim == 2:
        distance = np.min(distance, axis=0)
    _require(
        distance.shape == (len(graph.vertices),) and np.all(np.isfinite(distance)),
        "contact does not reach the complete material graph",
    )
    return distance


def graph_readout_action_support(
    readout_weights: np.ndarray,
    node_contact_distance_m: np.ndarray,
    *,
    length_scale_m: float,
) -> np.ndarray:
    """Lift an exponential graph-distance support prior to target identities."""

    weights = np.asarray(readout_weights, dtype=np.float64)
    distance = np.asarray(node_contact_distance_m, dtype=np.float64)
    _require(
        weights.ndim == 2 and weights.shape[1] == len(distance),
        "readout and graph distance axes differ",
    )
    _require(
        np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6),
        "readout must contain finite convex weights",
    )
    _require(
        np.all(np.isfinite(distance)) and np.all(distance >= 0.0),
        "graph distances must be finite and non-negative",
    )
    _require(
        np.isfinite(length_scale_m) and length_scale_m > 0.0,
        "support length scale must be finite and positive",
    )
    node_support = np.exp(-distance / float(length_scale_m))
    support = weights @ node_support
    return np.clip(support, 0.0, 1.0)


@dataclass(frozen=True)
class GraphActionSupportEpisode:
    """One source episode with frame-zero graph-to-observation support."""

    episode: CausalTrustEpisode
    readout_weights: np.ndarray
    node_contact_distance_m: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.readout_weights, dtype=np.float64)
        distance = np.asarray(self.node_contact_distance_m, dtype=np.float64)
        _require(
            weights.shape == (self.episode.target_m.shape[1], len(distance)),
            "action-support readout axes differ from the episode and graph",
        )
        copied_weights = weights.copy()
        copied_distance = distance.copy()
        copied_weights.setflags(write=False)
        copied_distance.setflags(write=False)
        object.__setattr__(self, "readout_weights", copied_weights)
        object.__setattr__(self, "node_contact_distance_m", copied_distance)

    def support(self, length_scale_m: float) -> np.ndarray:
        return graph_readout_action_support(
            self.readout_weights,
            self.node_contact_distance_m,
            length_scale_m=length_scale_m,
        )


def graph_action_support_prediction(
    source: GraphActionSupportEpisode,
    *,
    action_response: float,
    length_scale_m: float,
) -> np.ndarray:
    """Apply physical response only where contact geometry supports the action."""

    _require(
        np.isfinite(action_response) and 0.0 <= action_response <= 1.0,
        "action response must lie in [0, 1]",
    )
    episode = source.episode
    support = source.support(length_scale_m)
    response = episode.driven_m - episode.zero_action_m
    return episode.target_m[:1] + (
        float(action_response) * support[None, :, None] * response
    )


def fit_source_graph_action_support(
    source_episodes: Sequence[GraphActionSupportEpisode],
    *,
    length_scale_grid_m: Sequence[float],
    action_response_grid: Sequence[float],
    transfer_episodes: Sequence[GraphActionSupportEpisode] = (),
) -> dict[str, Any]:
    """Select one graph-support prior on source training frames only."""

    source = tuple(source_episodes)
    _require(len(source) >= 2, "graph action support needs at least two sources")
    ids = tuple(item.episode.episode_id for item in source)
    _require(len(set(ids)) == len(ids), "source episode identities repeat")
    transfer = tuple(transfer_episodes)
    transfer_ids = tuple(item.episode.episode_id for item in transfer)
    _require(
        len(set(transfer_ids)) == len(transfer_ids),
        "transfer episode identities repeat",
    )
    _require(
        not set(ids).intersection(transfer_ids),
        "selection and transfer episode identities overlap",
    )
    scales = tuple(float(value) for value in length_scale_grid_m)
    actions = tuple(float(value) for value in action_response_grid)
    _require(
        bool(scales) and all(np.isfinite(value) and value > 0.0 for value in scales),
        "support scale grid is invalid",
    )
    _require(
        bool(actions)
        and 0.0 in actions
        and all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in actions),
        "action response grid must contain persistence and lie in [0, 1]",
    )

    candidates = []
    for scale in scales:
        for action in actions:
            if action == 0.0 and scale != scales[0]:
                continue
            by_episode = {}
            for item in source:
                episode = item.episode
                predicted = graph_action_support_prediction(
                    item,
                    action_response=action,
                    length_scale_m=scale,
                )
                train = score_causal_trust_interval(
                    episode,
                    predicted,
                    0,
                    episode.train_stop_frame,
                )
                tail = score_causal_trust_interval(
                    episode,
                    predicted,
                    episode.train_stop_frame,
                    len(episode.target_m),
                )
                by_episode[episode.episode_id] = {
                    "mean_action_support": float(np.mean(item.support(scale))),
                    "train": train,
                    "untouched_tail": tail,
                }
            candidates.append(
                {
                    "length_scale_m": scale,
                    "action_response": action,
                    "pooled_train_relative_score_vs_persistence": float(
                        np.mean(
                            [
                                row["train"]["relative_score_vs_persistence"]
                                for row in by_episode.values()
                            ]
                        )
                    ),
                    "pooled_tail_relative_score_vs_persistence": float(
                        np.mean(
                            [
                                row["untouched_tail"]["relative_score_vs_persistence"]
                                for row in by_episode.values()
                            ]
                        )
                    ),
                    "by_episode": by_episode,
                }
            )

    selected = min(
        candidates,
        key=lambda row: (
            row["pooled_train_relative_score_vs_persistence"],
            row["action_response"],
            row["length_scale_m"],
        ),
    )
    tail_oracle = min(
        candidates,
        key=lambda row: row["pooled_tail_relative_score_vs_persistence"],
    )
    held_out_transfer = {}
    for item in transfer:
        episode = item.episode
        predicted = graph_action_support_prediction(
            item,
            action_response=float(selected["action_response"]),
            length_scale_m=float(selected["length_scale_m"]),
        )
        held_out_transfer[episode.episode_id] = {
            "mean_action_support": float(
                np.mean(item.support(float(selected["length_scale_m"])))
            ),
            "train": score_causal_trust_interval(
                episode,
                predicted,
                0,
                episode.train_stop_frame,
            ),
            "untouched_tail": score_causal_trust_interval(
                episode,
                predicted,
                episode.train_stop_frame,
                len(episode.target_m),
            ),
        }
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360SourceGraphActionSupportFit",
        "method": {
            "support_kernel": "exp(-material_graph_distance / length_scale_m)",
            "autonomous_drift_trust": 0.0,
            "persistence_fallback_in_grid": True,
            "action_support_uses_simulator_residual": False,
            "action_support_uses_future_object_observations": False,
        },
        "source_episode_ids": list(ids),
        "held_out_source_transfer_episode_ids": list(transfer_ids),
        "length_scale_grid_m": list(scales),
        "action_response_grid": list(actions),
        "selected": selected,
        "untouched_tail_oracle_diagnostic": tail_oracle,
        "held_out_source_transfer": held_out_transfer,
        "candidate_table": candidates,
        "information_boundary": {
            "selection_uses_source_train_frames_only": True,
            "source_tails_used_for_selection": False,
            "source_tails_used_for_exploratory_transfer_evaluation": True,
            "held_out_source_transfer_used_for_selection": False,
            "calibration_outcomes_read": False,
            "target_initial_frame_read": False,
            "target_future_read": False,
        },
        "claim_boundary": (
            "source-only cross-object action-support discovery; independent "
            "preregistered evaluation is required before a SOTA claim"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


__all__ = [
    "GraphActionSupportEpisode",
    "dynamic_contact_anchor_indices",
    "fit_source_graph_action_support",
    "graph_action_support_prediction",
    "graph_contact_distance_m",
    "graph_readout_action_support",
]
