#!/usr/bin/env python3
"""Build one frame-zero automatic twin for the dynamic TAPNext++ protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import sys
import types
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_dynamic_tapnextpp_cohort import (
    dynamic_provider_case_record,
    load_dynamic_provider_cohort_lock,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256
from bayesian_phystwin.tapnextpp_dynamic_multiview import PROTOCOL_ID


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_frozen_upstream(repo: Path) -> dict[str, Any]:
    """Load numerical modules only from the separately checksummed source tree."""

    import bayesian_phystwin

    source = repo.resolve() / "src"
    causal_path = source / "causal4d_public"
    bayesian_path = source / "bayesian_phystwin"
    _require(causal_path.is_dir(), "frozen causal4d_public package is missing")
    _require(bayesian_path.is_dir(), "frozen Bayesian-PhysTwin package is missing")
    _require(
        "causal4d_public" not in sys.modules,
        "causal4d_public loaded before frozen runtime binding",
    )
    _require(
        "bayesian_phystwin.phystwin_graph" not in sys.modules,
        "local PhysTwin graph loaded before frozen runtime binding",
    )
    if str(bayesian_path) not in bayesian_phystwin.__path__:
        bayesian_phystwin.__path__.insert(0, str(bayesian_path))
    causal4d_public = types.ModuleType("causal4d_public")
    causal4d_public.__path__ = [str(causal_path)]
    causal4d_public.__package__ = "causal4d_public"
    sys.modules["causal4d_public"] = causal4d_public
    dense = importlib.import_module("causal4d_public.deform360_dense_reusable_panel")
    independent = importlib.import_module(
        "causal4d_public.deform360_independent_source"
    )
    partial = importlib.import_module("causal4d_public.deform360_partial_graph_state")
    graph = importlib.import_module("causal4d_public.deform360_reusable_graph")
    return {
        "load_dense_reusable_panel_config": dense.load_dense_reusable_panel_config,
        "validate_prediction_only_bundle": independent.validate_prediction_only_bundle,
        "PartialGraphStateConfig": partial.PartialGraphStateConfig,
        "evaluate_partial_graph_state": partial.evaluate_partial_graph_state,
        "ReusableGraphRegistrationConfig": graph.ReusableGraphRegistrationConfig,
        "build_canonical_deform360_graph": graph.build_canonical_deform360_graph,
        "write_canonical_deform360_graph": graph.write_canonical_deform360_graph,
        "PhysTwinSpringGraphConfig": graph.PhysTwinSpringGraphConfig,
    }


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    _require(isinstance(payload, dict), "PhysTwin input must contain a dictionary")
    return payload


def _load_protocol(path: Path, cohort: Mapping[str, Any]) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(protocol, dict) and protocol.get("protocol_id") == PROTOCOL_ID,
        "dynamic provider protocol changed",
    )
    _require(
        file_sha256(path)
        == cohort["bindings"]["provider_protocol_file_sha256"],
        "dynamic provider protocol differs from the cohort lock",
    )
    return protocol


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--partition", choices=("source", "target"), required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--episode-final-data", type=Path, required=True)
    parser.add_argument("--episode-graph", type=Path, required=True)
    parser.add_argument("--simulator-final-data", type=Path, required=True)
    parser.add_argument("--state-artifact", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--canonical-node-count", type=int, default=384)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cohort = load_dynamic_provider_cohort_lock(args.cohort_lock)
    protocol = _load_protocol(args.protocol, cohort)
    authorization = dynamic_provider_case_record(
        cohort,
        object_id=args.object_id,
        episode_id=args.episode_id,
        partition=args.partition,
    )
    upstream = _load_frozen_upstream(args.source_repo)
    load_dense_config = upstream["load_dense_reusable_panel_config"]
    validate_prediction_only_bundle = upstream["validate_prediction_only_bundle"]
    PartialGraphStateConfig = upstream["PartialGraphStateConfig"]
    evaluate_partial_graph_state = upstream["evaluate_partial_graph_state"]
    ReusableGraphRegistrationConfig = upstream["ReusableGraphRegistrationConfig"]
    build_canonical_graph = upstream["build_canonical_deform360_graph"]
    write_canonical_graph = upstream["write_canonical_deform360_graph"]
    PhysTwinSpringGraphConfig = upstream["PhysTwinSpringGraphConfig"]

    dense_config_path = (
        args.source_repo.resolve()
        / "configs/causal4d_public/deform360_dense_reusable_panel_v1.json"
    )
    dense_protocol = load_dense_config(dense_config_path)
    method = dense_protocol["config"]["dense_reusable_method"]
    registration = method["canonical_episode_registration"]
    state_policy = method["partial_graph_state_completion"]
    configured_node_count = int(method["canonical_surface_node_count"])
    canonical_node_count = int(args.canonical_node_count)
    minimum_node_count = int(method["minimum_canonical_surface_node_count"])
    _require(canonical_node_count == 384, "dynamic physical node count changed")
    _require(canonical_node_count >= minimum_node_count, "node count below minimum")
    _require(int(method["temporal_prefix_frame_count"]) == 1, "prefix changed")
    _require(
        int(state_policy["uses_prefix_visibility_frame_count"]) == 1,
        "state completion uses future visibility",
    )

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
    validation = validate_prediction_only_bundle(
        episode,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    points = np.asarray(episode["object_points"])
    colors = np.asarray(episode["object_colors"])
    visibility = np.asarray(episode["object_visibilities"], dtype=bool)
    validity = np.asarray(episode["object_motions_valid"], dtype=bool)
    controllers = np.asarray(episode["controller_points"])
    effective_node_count = min(canonical_node_count, points.shape[1])
    _require(effective_node_count >= minimum_node_count, "too few frame-zero points")
    registration_config = replace(
        registration_config,
        canonical_node_count=effective_node_count,
    )

    canonical = build_canonical_graph(
        points[0],
        colors[0],
        registration_config=registration_config,
        spring_config=spring_config,
        reference_controller_points=controllers[0],
        controller_group_size=int(method["controller_input_group_size"]),
        contact_clearance_m=state_config.contact_clearance_m,
    )
    graph_descriptor = write_canonical_graph(args.episode_graph, canonical)
    candidate_reliability = np.asarray(
        visibility[0] & validity[0],
        dtype=np.float64,
    )
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
    simulator_data["controller_points"] = controllers.astype(np.float32)
    simulator_data["object_points"] = np.repeat(
        state.vertices[None],
        frame_count,
        axis=0,
    ).astype(np.float32)
    simulator_data["object_colors"] = np.repeat(
        canonical.colors[None],
        frame_count,
        axis=0,
    ).astype(np.float32)
    supported = (
        state.source_to_target_distance_m <= state_config.maximum_supported_distance_m
    )
    simulator_data["object_visibilities"] = np.repeat(
        supported[None],
        frame_count,
        axis=0,
    )
    simulator_data["object_motions_valid"] = np.repeat(
        supported[None],
        frame_count,
        axis=0,
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
    with args.simulator_final_data.open("wb") as handle:
        pickle.dump(simulator_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicTAPNextPPAutomaticEpisodeTwin",
        "protocol_id": PROTOCOL_ID,
        "provider_protocol_file_sha256": file_sha256(args.protocol),
        "cohort_lock_sha256": cohort["cohort_lock_sha256"],
        "cohort_lock_file_sha256": file_sha256(args.cohort_lock),
        "partition": args.partition,
        **authorization,
        "graph_mode": "episode_specific_frame_zero_control",
        "capacity_diagnostic": {
            "configured_canonical_node_count": configured_node_count,
            "requested_canonical_node_count": canonical_node_count,
            "effective_canonical_node_count": effective_node_count,
            "capacity_is_a_maximum": True,
        },
        "graph": graph_descriptor,
        "state_metrics": state.metrics,
        "input_sha256": {
            "episode_final_data": file_sha256(args.episode_final_data),
            "dense_numeric_config": file_sha256(dense_config_path),
        },
        "output_sha256": {
            "episode_graph": file_sha256(args.episode_graph),
            "simulator_final_data": file_sha256(args.simulator_final_data),
            "state_artifact": file_sha256(args.state_artifact),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_robot_action_available": True,
            "post_initial_object_observation_used": False,
            "simulator_residual_used": False,
            "target_access": False,
            "prediction_only_input_required": True,
            "future_object_tracks_present": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
        },
        "prediction_input_validation": validation,
        "passed": bool(state.metrics["passed"]),
        "claim_boundary": (
            "automatic frame-zero twin under the frozen open-27 numerical "
            "contract, bound to the dynamic TAPNext++ cohort"
        ),
        "protocol_snapshot": {
            "physical_backbone": protocol["state_update"]["physical_backbone"],
            "state_coordinates": protocol["state_update"]["state_coordinates"],
        },
    }
    summary["result_sha256"] = _canonical_sha256(
        summary,
        digest_key="result_sha256",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
