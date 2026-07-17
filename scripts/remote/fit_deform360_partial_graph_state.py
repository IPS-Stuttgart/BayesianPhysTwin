#!/usr/bin/env python3
"""Fit a source-authorized Deform360 episode to one canonical material graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_dense_reusable_panel import (
    authorize_dense_panel_episode,
    load_dense_reusable_panel_config,
)
from causal4d_public.deform360_partial_graph_state import (
    PartialGraphStateConfig,
    evaluate_partial_graph_state,
    fit_partial_graph_state,
)
from causal4d_public.deform360_reusable_graph import (
    load_canonical_deform360_graph,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError("PhysTwin final_data must contain a dictionary")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--phase", choices=("source", "calibration"), required=True)
    parser.add_argument("--source-admission-passed", action="store_true")
    parser.add_argument("--canonical-graph", type=Path, required=True)
    parser.add_argument("--episode-final-data", type=Path, required=True)
    parser.add_argument("--simulator-final-data", type=Path, required=True)
    parser.add_argument("--state-artifact", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--state-frame",
        type=int,
        default=0,
        help=(
            "Opt-in observed state frame. Nonzero values must equal the final "
            "frame of the training interval declared by --split-json."
        ),
    )
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = (
        args.repo / "configs/causal4d_public/deform360_dense_reusable_panel_v1.json"
    )
    protocol = load_dense_reusable_panel_config(config_path)
    authorization = authorize_dense_panel_episode(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        phase=args.phase,
        source_admission_passed=args.source_admission_passed,
    )
    method = protocol["config"]["dense_reusable_method"]
    object_protocol = {row["object_id"]: row for row in protocol["config"]["cohort"]}[
        args.object_id
    ]
    state_policy = method["partial_graph_state_completion"]
    config = PartialGraphStateConfig(
        start_count=int(state_policy["start_count"]),
        anchor_count=int(state_policy["anchor_count"]),
        iterations=int(state_policy["iterations"]),
        learning_rate=float(state_policy["learning_rate"]),
        observation_scale_m=float(state_policy["observation_scale_m"]),
        hidden_node_distance_cap_m=float(state_policy["hidden_node_distance_cap_m"]),
        hidden_node_fit_weight=float(state_policy["hidden_node_fit_weight"]),
        edge_strain_weight=float(state_policy["edge_strain_weight"]),
        bridge_strain_weight=float(state_policy["bridge_strain_weight"]),
        contact_anchor_weight=float(state_policy["contact_anchor_weight"]),
        controller_group_size=int(method["controller_input_group_size"]),
        contact_clearance_m=0.1 * float(method["object_radius_m"]),
        readout_neighbour_count=int(state_policy["readout_neighbour_count"]),
        readout_geometry_scale_m=float(state_policy["readout_geometry_scale_m"]),
        readout_color_scale=float(state_policy["readout_color_scale"]),
        readout_color_weight=float(state_policy["readout_color_weight"]),
        measurement_variance_m2=float(state_policy["measurement_variance_m2"]),
        maximum_supported_distance_m=float(
            state_policy["maximum_supported_distance_m"]
        ),
        minimum_observed_target_fraction=float(
            state_policy["minimum_observed_target_fraction"]
        ),
        minimum_effective_target_reliability=float(
            state_policy["minimum_effective_target_reliability"]
        ),
        maximum_p99_relative_edge_strain=float(
            state_policy["maximum_p99_relative_edge_strain"]
        ),
        maximum_bridge_relative_edge_strain=float(
            state_policy["maximum_bridge_relative_edge_strain"]
        ),
        maximum_contact_anchor_error_m=float(
            state_policy["maximum_contact_anchor_error_m"]
        ),
    )
    canonical = load_canonical_deform360_graph(args.canonical_graph)
    episode = _load_pickle(args.episode_final_data)
    total_frame_count = len(episode["object_points"])
    if not 0 <= args.state_frame < total_frame_count:
        raise ValueError("state frame is outside the episode")
    split_sha256 = None
    if args.state_frame:
        if args.split_json is None:
            raise ValueError("nonzero state frame requires --split-json")
        split = json.loads(args.split_json.read_text(encoding="utf-8"))
        if int(split.get("frame_len", -1)) != total_frame_count:
            raise ValueError("split frame count differs from episode")
        train = split.get("train")
        if train != [0, args.state_frame + 1]:
            raise ValueError("state frame must be the final training frame")
        split_sha256 = _sha256_file(args.split_json)
    prefix_frames = int(method["temporal_prefix_frame_count"])
    prefix_start = max(0, args.state_frame - prefix_frames + 1)
    prefix_stop = args.state_frame + 1
    visibility = np.asarray(
        episode["object_visibilities"][prefix_start:prefix_stop],
        dtype=bool,
    )
    validity = np.asarray(
        episode["object_motions_valid"][prefix_start:prefix_stop],
        dtype=bool,
    )
    candidate_reliability = np.mean(visibility & validity, axis=0)
    observed_points = np.asarray(episode["object_points"])[args.state_frame]
    observed_colors = np.asarray(episode["object_colors"])[args.state_frame]
    observed_controller = np.asarray(episode["controller_points"])[args.state_frame]
    if args.episode_id == int(object_protocol["canonical_reference_episode_id"]):
        if args.state_frame == 0:
            result = evaluate_partial_graph_state(
                canonical,
                canonical.vertices,
                observed_points,
                observed_colors,
                config=config,
                candidate_reliability=candidate_reliability,
                controller_points=observed_controller,
            )
        else:
            result = fit_partial_graph_state(
                canonical,
                observed_points,
                observed_colors,
                config=config,
                candidate_reliability=candidate_reliability,
                controller_points=observed_controller,
                device=args.device,
            )
    else:
        result = fit_partial_graph_state(
            canonical,
            observed_points,
            observed_colors,
            config=config,
            candidate_reliability=candidate_reliability,
            controller_points=observed_controller,
            device=args.device,
        )

    frame_count = total_frame_count - args.state_frame
    simulator_data = dict(episode)
    simulator_data["object_points"] = np.repeat(
        result.vertices[None], frame_count, axis=0
    ).astype(np.float32)
    simulator_data["object_colors"] = np.repeat(
        canonical.colors[None], frame_count, axis=0
    ).astype(np.float32)
    supported = (
        result.source_to_target_distance_m <= config.maximum_supported_distance_m
    )
    simulator_data["object_visibilities"] = np.repeat(
        supported[None], frame_count, axis=0
    )
    simulator_data["object_motions_valid"] = np.repeat(
        supported[None], frame_count, axis=0
    )
    simulator_data["controller_points"] = np.asarray(episode["controller_points"])[
        args.state_frame :
    ].copy()
    simulator_data["surface_points"] = np.empty((0, 3), dtype=np.float32)
    simulator_data["interior_points"] = np.empty((0, 3), dtype=np.float32)
    simulator_data["reusable_graph_registration"] = {
        "canonical_graph_sha256": canonical.sha256,
        "association_mode": "partial_graph_state_completion",
        "target_readout_is_external": True,
        "state_frame": int(args.state_frame),
        "passed": bool(result.metrics["passed"]),
    }
    args.simulator_final_data.parent.mkdir(parents=True, exist_ok=True)
    with args.simulator_final_data.open("wb") as stream:
        pickle.dump(simulator_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    args.state_artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.state_artifact,
        vertices=result.vertices,
        readout_weights=result.readout_weights,
        readout_covariance_m2=result.readout_covariance_m2,
        target_prior_reliability=result.target_prior_reliability,
        state_covariance_m2=result.state_covariance_m2,
        source_to_target_distance_m=result.source_to_target_distance_m,
        target_to_source_distance_m=result.target_to_source_distance_m,
        relative_edge_strain=result.relative_edge_strain,
        canonical_graph_sha256=np.asarray(canonical.sha256),
        state_frame=np.asarray(args.state_frame, dtype=np.int64),
    )
    summary = {
        "schema_version": 1,
        "artifact_kind": "Deform360PartialGraphStateCompletion",
        "protocol_id": authorization["protocol_id"],
        "protocol_config_sha256": authorization["config_sha256"],
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "phase": args.phase,
        "state_frame": int(args.state_frame),
        "canonical_graph_sha256": canonical.sha256,
        "canonical_graph": {
            "node_count": len(canonical.vertices),
            "observed_node_count": canonical.observed_node_count,
            "latent_node_count": canonical.latent_node_count,
            "object_spring_count": len(canonical.springs),
            "bridge_spring_count": canonical.bridge_spring_count,
        },
        "metrics": result.metrics,
        "input_sha256": {
            "canonical_graph": _sha256_file(args.canonical_graph),
            "episode_final_data": _sha256_file(args.episode_final_data),
            "split_json": split_sha256,
        },
        "output_sha256": {
            "simulator_final_data": _sha256_file(args.simulator_final_data),
            "state_artifact": _sha256_file(args.state_artifact),
        },
        "information_boundary": {
            "observed_object_geometry_frames_used": [int(args.state_frame)],
            "reliability_prefix_frame_range": [prefix_start, prefix_stop],
            "simulator_residual_used": False,
            "future_object_frames_used": False,
            "held_out_test_frames_used": False,
            "target_access": False,
        },
        "passed": bool(result.metrics["passed"]),
        "claim_boundary": (
            "current state and uncertain target readout from partial observed "
            "geometry; canonical topology and rest lengths remain unchanged"
        ),
    }
    summary["result_sha256"] = hashlib.sha256(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
