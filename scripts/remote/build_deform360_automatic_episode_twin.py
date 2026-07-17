#!/usr/bin/env python3
"""Build a frame-zero episode twin as a benchmark-fair control arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_graph import PhysTwinSpringGraphConfig
from causal4d_public.deform360_dense_reusable_panel import (
    authorize_dense_panel_episode,
    load_dense_reusable_panel_config,
)
from causal4d_public.deform360_independent_source import (
    validate_prediction_only_bundle,
)
from causal4d_public.deform360_partial_graph_state import (
    PartialGraphStateConfig,
    evaluate_partial_graph_state,
)
from causal4d_public.deform360_reusable_graph import (
    ReusableGraphRegistrationConfig,
    build_canonical_deform360_graph,
    write_canonical_deform360_graph,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_sha256(payload: dict[str, Any]) -> str:
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
    parser.add_argument("--episode-final-data", type=Path, required=True)
    parser.add_argument("--episode-graph", type=Path, required=True)
    parser.add_argument("--simulator-final-data", type=Path, required=True)
    parser.add_argument("--state-artifact", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--prediction-only-input",
        action="store_true",
        help=(
            "Require a frame-zero-only bundle whose object geometry is constant "
            "over the known-action prediction horizon."
        ),
    )
    parser.add_argument(
        "--canonical-node-count",
        type=int,
        help=(
            "Opt-in source-only capacity diagnostic. The frozen panel value "
            "is used when this argument is omitted."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_path = (
        args.repo / "configs/causal4d_public/deform360_dense_reusable_panel_v1.json"
    )
    protocol = load_dense_reusable_panel_config(protocol_path)
    authorization = authorize_dense_panel_episode(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        phase=args.phase,
        source_admission_passed=args.source_admission_passed,
    )
    method = protocol["config"]["dense_reusable_method"]
    registration = method["canonical_episode_registration"]
    state_policy = method["partial_graph_state_completion"]
    configured_node_count = int(method["canonical_surface_node_count"])
    canonical_node_count = (
        configured_node_count
        if args.canonical_node_count is None
        else int(args.canonical_node_count)
    )
    minimum_node_count = int(method["minimum_canonical_surface_node_count"])
    if canonical_node_count < minimum_node_count:
        raise ValueError("requested canonical node count is below the panel minimum")
    if int(method["temporal_prefix_frame_count"]) != 1:
        raise ValueError("automatic episode twin requires a frame-zero-only lock")
    if int(state_policy["uses_prefix_visibility_frame_count"]) != 1:
        raise ValueError("automatic episode twin cannot use post-initial reliability")

    registration_config = ReusableGraphRegistrationConfig(
        canonical_node_count=canonical_node_count,
        geometry_sigma_m=float(registration["geometry_sigma_m"]),
        color_sigma=float(registration["color_sigma"]),
        color_cost_weight=float(registration["color_cost_weight"]),
        assignment_temperature=float(registration["assignment_temperature"]),
        measurement_variance_m2=float(registration["measurement_variance_m2"]),
        maximum_match_distance_m=float(registration["maximum_match_distance_m"]),
        minimum_match_fraction=float(method["minimum_temporal_match_fraction"]),
        minimum_effective_reliable_fraction=float(
            method["minimum_effective_reliable_match_fraction"]
        ),
        icp_iterations=int(registration["icp_iterations"]),
        trim_fraction=float(registration["trim_fraction"]),
        use_pca_multistart=bool(registration["use_pca_multistart"]),
    )
    spring_config = PhysTwinSpringGraphConfig(
        object_radius=float(method["object_radius_m"]),
        object_max_neighbours=int(method["object_max_neighbours"]),
        controller_radius=float(method["controller_radius_m"]),
        controller_max_neighbours=int(method["controller_max_neighbours"]),
    )
    state_config = PartialGraphStateConfig(
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

    episode = _load_pickle(args.episode_final_data)
    points = np.asarray(episode["object_points"])
    colors = np.asarray(episode["object_colors"])
    visibility = np.asarray(episode["object_visibilities"], dtype=bool)
    validity = np.asarray(episode["object_motions_valid"], dtype=bool)
    controllers = np.asarray(episode["controller_points"])
    prediction_input_validation = None
    if args.prediction_only_input:
        prediction_input_validation = validate_prediction_only_bundle(
            episode,
            object_id=args.object_id,
            episode_id=args.episode_id,
        )
    if points.ndim != 3 or points.shape[0] < 2 or points.shape[2] != 3:
        raise ValueError("object_points must have shape (T, N, 3) with T >= 2")
    if colors.shape != points.shape:
        raise ValueError("object colors must match object points")
    if visibility.shape != points.shape[:2] or validity.shape != points.shape[:2]:
        raise ValueError("visibility and validity must match object point axes")
    if controllers.ndim != 3 or controllers.shape[0] != points.shape[0]:
        raise ValueError("controller trajectory must share the episode frame axis")
    effective_node_count = min(canonical_node_count, points.shape[1])
    if effective_node_count < minimum_node_count:
        raise ValueError("frame-zero point count is below the panel minimum")
    registration_config = replace(
        registration_config,
        canonical_node_count=effective_node_count,
    )

    canonical = build_canonical_deform360_graph(
        points[0],
        colors[0],
        registration_config=registration_config,
        spring_config=spring_config,
        reference_controller_points=controllers[0],
        controller_group_size=int(method["controller_input_group_size"]),
        contact_clearance_m=state_config.contact_clearance_m,
    )
    graph_descriptor = write_canonical_deform360_graph(args.episode_graph, canonical)
    candidate_reliability = np.asarray(visibility[0] & validity[0], dtype=np.float64)
    state = evaluate_partial_graph_state(
        canonical,
        canonical.vertices,
        points[0],
        colors[0],
        config=state_config,
        candidate_reliability=candidate_reliability,
        controller_points=controllers[0],
    )

    frame_count = points.shape[0]
    simulator_data = dict(episode)
    simulator_data["object_points"] = np.repeat(
        state.vertices[None], frame_count, axis=0
    ).astype(np.float32)
    simulator_data["object_colors"] = np.repeat(
        canonical.colors[None], frame_count, axis=0
    ).astype(np.float32)
    supported = (
        state.source_to_target_distance_m <= state_config.maximum_supported_distance_m
    )
    simulator_data["object_visibilities"] = np.repeat(
        supported[None], frame_count, axis=0
    )
    simulator_data["object_motions_valid"] = np.repeat(
        supported[None], frame_count, axis=0
    )
    simulator_data["surface_points"] = np.empty((0, 3), dtype=np.float32)
    simulator_data["interior_points"] = np.empty((0, 3), dtype=np.float32)
    simulator_data["reusable_graph_registration"] = {
        "canonical_graph_sha256": canonical.sha256,
        "association_mode": "automatic_episode_graph_from_frame_zero",
        "target_readout_is_external": True,
        "state_frame": 0,
        "passed": bool(state.metrics["passed"]),
    }
    args.simulator_final_data.parent.mkdir(parents=True, exist_ok=True)
    with args.simulator_final_data.open("wb") as stream:
        pickle.dump(simulator_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    args.state_artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.state_artifact,
        vertices=state.vertices,
        readout_weights=state.readout_weights,
        readout_covariance_m2=state.readout_covariance_m2,
        target_prior_reliability=state.target_prior_reliability,
        state_covariance_m2=state.state_covariance_m2,
        source_to_target_distance_m=state.source_to_target_distance_m,
        target_to_source_distance_m=state.target_to_source_distance_m,
        relative_edge_strain=state.relative_edge_strain,
        canonical_graph_sha256=np.asarray(canonical.sha256),
        state_frame=np.asarray(0, dtype=np.int64),
    )

    summary = {
        "schema_version": 1,
        "artifact_kind": "Deform360AutomaticEpisodeTwin",
        "protocol_id": authorization["protocol_id"],
        "protocol_config_sha256": authorization["config_sha256"],
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "phase": args.phase,
        "graph_mode": "episode_specific_frame_zero_control",
        "capacity_diagnostic": {
            "configured_canonical_node_count": configured_node_count,
            "requested_canonical_node_count": canonical_node_count,
            "effective_canonical_node_count": effective_node_count,
            "source_only_override": effective_node_count != configured_node_count,
            "capacity_is_a_maximum": True,
        },
        "graph": graph_descriptor,
        "state_metrics": state.metrics,
        "input_sha256": {
            "episode_final_data": _sha256_file(args.episode_final_data),
        },
        "output_sha256": {
            "episode_graph": _sha256_file(args.episode_graph),
            "simulator_final_data": _sha256_file(args.simulator_final_data),
            "state_artifact": _sha256_file(args.state_artifact),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_robot_action_available": True,
            "post_initial_object_observation_used": False,
            "simulator_residual_used": False,
            "target_access": False,
            "prediction_only_input_required": args.prediction_only_input,
            "future_object_tracks_present": (
                False if args.prediction_only_input else None
            ),
        },
        "prediction_input_validation": prediction_input_validation,
        "passed": bool(state.metrics["passed"]),
        "claim_boundary": (
            "benchmark-fair automatic frame-zero episode-twin control; physical "
            "parameters may be pooled across source episodes, but object topology "
            "and rest geometry are rebuilt automatically for each episode"
        ),
    }
    summary["result_sha256"] = _result_sha256(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
