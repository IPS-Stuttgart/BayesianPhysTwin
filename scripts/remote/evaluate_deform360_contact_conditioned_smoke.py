#!/usr/bin/env python3
"""Score one opened source-only contact-conditioned PhysTwin smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_action_support import (
    graph_contact_distance_m,
    graph_readout_action_support,
)
from causal4d_public.deform360_independent_source import sha256_file
from causal4d_public.deform360_phystwin_trust import (
    CausalTrustEpisode,
    score_causal_trust_interval,
)
from causal4d_public.deform360_reusable_graph import (
    load_canonical_deform360_graph,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--prediction-input", type=Path, required=True)
    parser.add_argument("--target-data", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--driven-trajectory", type=Path, required=True)
    parser.add_argument("--zero-trajectory", type=Path, required=True)
    parser.add_argument("--old-prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-scale-m", type=float, default=0.12)
    parser.add_argument("--action-response", type=float, default=0.9)
    return parser.parse_args()


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"pickle is not a dictionary: {path}")
    return payload


def _load_trajectory(path: Path, node_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as stored:
        trajectory = np.asarray(stored["vertices"], dtype=np.float64)
    if (
        trajectory.ndim != 3
        or trajectory.shape[1] < node_count
        or trajectory.shape[2] != 3
        or not np.all(np.isfinite(trajectory))
    ):
        raise ValueError(f"invalid PhysTwin trajectory: {path}")
    return trajectory[:, :node_count]


def _metric_summary(
    episode: CausalTrustEpisode, predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    intervals = {
        "future": (1, 76),
        "early": (1, 26),
        "middle": (26, 51),
        "late": (51, 76),
    }
    summary: dict[str, Any] = {}
    for method, prediction in predictions.items():
        by_interval = {}
        for name, (start, stop) in intervals.items():
            scored = score_causal_trust_interval(episode, prediction, start, stop)
            scored["track_improvement_fraction"] = 1.0 - (
                float(scored["track_rmse_m"])
                / float(scored["persistence_track_rmse_m"])
            )
            scored["chamfer_improvement_fraction"] = 1.0 - (
                float(scored["chamfer_m"]) / float(scored["persistence_chamfer_m"])
            )
            by_interval[name] = scored
        summary[method] = by_interval
    return summary


def main() -> int:
    args = _parse_args()
    prediction_input = _load_pickle(args.prediction_input)
    target_data = _load_pickle(args.target_data)
    initial = np.asarray(prediction_input["object_points"][0], dtype=np.float64)
    target = np.asarray(target_data["object_points"], dtype=np.float64)
    visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
    validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
    if target.shape != (76, len(initial), 3):
        raise ValueError("target and frame-zero prediction identities differ")
    if not np.array_equal(target[0].astype(np.float32), initial.astype(np.float32)):
        raise ValueError("target frame zero differs from prediction-only input")

    graph = load_canonical_deform360_graph(args.graph)
    with np.load(args.readout, allow_pickle=False) as stored:
        readout_weights = np.asarray(stored["readout_weights"], dtype=np.float64)
        graph_sha = str(np.asarray(stored["canonical_graph_sha256"]).item())
    if graph_sha != graph.sha256 or readout_weights.shape != (
        len(initial),
        len(graph.vertices),
    ):
        raise ValueError("readout and contact-conditioned graph differ")

    driven_nodes = _load_trajectory(args.driven_trajectory, len(graph.vertices))
    zero_nodes = _load_trajectory(args.zero_trajectory, len(graph.vertices))
    if driven_nodes.shape != zero_nodes.shape or len(driven_nodes) != len(target):
        raise ValueError("driven, zero, and target frame axes differ")
    driven = np.einsum("mn,tnc->tmc", readout_weights, driven_nodes, optimize=True)
    zero = np.einsum("mn,tnc->tmc", readout_weights, zero_nodes, optimize=True)
    offset = initial - zero[0]
    driven += offset[None]
    zero += offset[None]
    response = driven - zero
    support = graph_readout_action_support(
        readout_weights,
        graph_contact_distance_m(graph),
        length_scale_m=args.length_scale_m,
    )
    persistence = np.repeat(initial[None], len(target), axis=0)
    predictions = {
        "persistence": persistence,
        "contact_conditioned_graph_support": initial[None]
        + args.action_response * support[None, :, None] * response,
        "contact_conditioned_direct_response": initial[None] + response,
    }
    if args.old_prediction is not None:
        with np.load(args.old_prediction, allow_pickle=False) as stored:
            predictions["frozen_frame_zero_graph_support"] = np.asarray(
                stored["prediction_m"], dtype=np.float64
            )

    episode = CausalTrustEpisode(
        episode_id=f"{args.object_id}/{args.episode_id}",
        target_m=target,
        visibility=visibility,
        validity=validity,
        driven_m=driven,
        zero_action_m=zero,
        train_stop_frame=60,
        source_data_sha256=sha256_file(args.target_data),
        driven_trajectory_sha256=sha256_file(args.driven_trajectory),
        zero_action_trajectory_sha256=sha256_file(args.zero_trajectory),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ContactConditionedSourceSmoke",
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "method": {
            "length_scale_m": args.length_scale_m,
            "action_response": args.action_response,
            "contact_anchor_count": len(graph.contact_anchor_indices),
        },
        "metrics": _metric_summary(episode, predictions),
        "input_sha256": {
            "prediction_input": sha256_file(args.prediction_input),
            "target_data": sha256_file(args.target_data),
            "graph": sha256_file(args.graph),
            "readout": sha256_file(args.readout),
            "driven_trajectory": sha256_file(args.driven_trajectory),
            "zero_trajectory": sha256_file(args.zero_trajectory),
            "old_prediction": (
                None
                if args.old_prediction is None
                else sha256_file(args.old_prediction)
            ),
        },
        "information_boundary": {
            "prediction_uses_object_observation_frames": [0],
            "known_future_robot_action_used": True,
            "future_tactile_used": False,
            "opened_source_future_used_for_scoring": True,
            "calibration_outcome_read": False,
            "target_outcome_read": False,
        },
        "claim_boundary": (
            "exploratory result on an already examined public source episode; "
            "not independent or state-of-the-art evidence"
        ),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
