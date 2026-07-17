#!/usr/bin/env python3
"""Build or apply one source-locked canonical graph across Deform360 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_graph import PhysTwinSpringGraphConfig
from causal4d_public.deform360_dense_reusable_panel import (
    authorize_dense_panel_episode,
    load_dense_reusable_panel_config,
)
from causal4d_public.deform360_reusable_graph import (
    ReusableGraphRegistrationConfig,
    build_canonical_deform360_graph,
    canonical_reference_registration,
    episode_registration_summary,
    load_canonical_deform360_graph,
    register_canonical_graph_to_episode,
    registered_episode_data,
    write_canonical_deform360_graph,
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


def _write_pickle(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--phase", choices=("source", "calibration"), required=True)
    parser.add_argument("--source-admission-passed", action="store_true")
    parser.add_argument("--reference-final-data", type=Path, required=True)
    parser.add_argument("--episode-final-data", type=Path, required=True)
    parser.add_argument("--canonical-graph", type=Path, required=True)
    parser.add_argument("--registered-final-data", type=Path)
    parser.add_argument("--registration-arrays", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--build-canonical", action="store_true")
    parser.add_argument("--canonical-only", action="store_true")
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
    cohort = {row["object_id"]: row for row in protocol["config"]["cohort"]}
    object_protocol = cohort[args.object_id]
    method = protocol["config"]["dense_reusable_method"]
    association = method["canonical_episode_registration"]
    registration_config = ReusableGraphRegistrationConfig(
        canonical_node_count=int(method["canonical_surface_node_count"]),
        geometry_sigma_m=float(association["geometry_sigma_m"]),
        color_sigma=float(association["color_sigma"]),
        color_cost_weight=float(association["color_cost_weight"]),
        assignment_temperature=float(association["assignment_temperature"]),
        measurement_variance_m2=float(association["measurement_variance_m2"]),
        maximum_match_distance_m=float(association["maximum_match_distance_m"]),
        minimum_match_fraction=float(method["minimum_temporal_match_fraction"]),
        minimum_effective_reliable_fraction=float(
            method["minimum_effective_reliable_match_fraction"]
        ),
        icp_iterations=int(association["icp_iterations"]),
        trim_fraction=float(association["trim_fraction"]),
        use_pca_multistart=bool(association["use_pca_multistart"]),
    )
    spring_config = PhysTwinSpringGraphConfig(
        object_radius=float(method["object_radius_m"]),
        object_max_neighbours=int(method["object_max_neighbours"]),
        controller_radius=float(method["controller_radius_m"]),
        controller_max_neighbours=int(method["controller_max_neighbours"]),
    )

    reference = _load_pickle(args.reference_final_data)
    episode = _load_pickle(args.episode_final_data)
    reference_points = np.asarray(reference["object_points"])[0]
    reference_colors = np.asarray(reference["object_colors"])[0]
    if args.build_canonical:
        if args.episode_id != int(object_protocol["canonical_reference_episode_id"]):
            raise ValueError("only the locked reference episode may build the graph")
        canonical = build_canonical_deform360_graph(
            reference_points,
            reference_colors,
            registration_config=registration_config,
            spring_config=spring_config,
            reference_controller_points=np.asarray(reference["controller_points"])[0],
            controller_group_size=int(method["controller_input_group_size"]),
            contact_clearance_m=0.1 * float(method["object_radius_m"]),
        )
        canonical_descriptor = write_canonical_deform360_graph(
            args.canonical_graph,
            canonical,
        )
    else:
        canonical = load_canonical_deform360_graph(args.canonical_graph)
        canonical_descriptor = {
            "path": str(args.canonical_graph.resolve()),
            "reusable_graph_sha256": canonical.sha256,
            "node_count": len(canonical.vertices),
            "object_spring_count": len(canonical.springs),
            "bridge_spring_count": canonical.bridge_spring_count,
            "observed_node_count": canonical.observed_node_count,
            "latent_node_count": canonical.latent_node_count,
            "contact_anchor_count": len(canonical.contact_anchor_indices),
            "contact_chain_spring_count": canonical.contact_chain_spring_count,
        }
    if canonical.observed_node_count != registration_config.canonical_node_count:
        raise ValueError("canonical node count differs from the locked protocol")
    if args.canonical_only:
        if not args.build_canonical:
            raise ValueError("canonical-only mode must build the source graph")
        summary = {
            "schema_version": 1,
            "artifact_kind": "Deform360CanonicalReusableGraphBuild",
            "protocol_id": authorization["protocol_id"],
            "protocol_config_sha256": authorization["config_sha256"],
            "object_id": args.object_id,
            "episode_id": args.episode_id,
            "phase": args.phase,
            "canonical_graph": canonical_descriptor,
            "input_sha256": {
                "reference_final_data": _sha256_file(args.reference_final_data),
            },
            "output_sha256": {
                "canonical_graph": _sha256_file(args.canonical_graph),
            },
            "information_boundary": {
                "source_only": True,
                "simulator_residual_used": False,
                "future_object_frames_used": False,
                "target_access": False,
            },
            "passed": True,
            "claim_boundary": (
                "canonical topology and rest geometry only; episode state and "
                "readout association require a separately gated artifact"
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
        return 0
    if args.registered_final_data is None or args.registration_arrays is None:
        raise ValueError("episode registration output paths are required")

    prefix_frames = int(method["temporal_prefix_frame_count"])
    episode_points = np.asarray(episode["object_points"])[0]
    episode_colors = np.asarray(episode["object_colors"])[0]
    visibility = np.asarray(episode["object_visibilities"][:prefix_frames], dtype=bool)
    validity = np.asarray(episode["object_motions_valid"][:prefix_frames], dtype=bool)
    candidate_reliability = np.mean(visibility & validity, axis=0)
    if args.episode_id == int(object_protocol["canonical_reference_episode_id"]):
        if _sha256_file(args.episode_final_data) != _sha256_file(
            args.reference_final_data
        ):
            raise ValueError("reference registration must use the reference data")
        registration = canonical_reference_registration(
            canonical,
            config=registration_config,
            candidate_reliability=candidate_reliability,
        )
    else:
        registration = register_canonical_graph_to_episode(
            canonical,
            episode_points,
            episode_colors,
            config=registration_config,
            candidate_reliability=candidate_reliability,
        )
    registered = registered_episode_data(
        episode,
        registration,
        canonical_graph_sha256=canonical.sha256,
    )
    _write_pickle(args.registered_final_data, registered)
    args.registration_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.registration_arrays,
        rotation=registration.rotation,
        translation=registration.translation,
        target_indices=registration.target_indices,
        geometric_error_m=registration.geometric_error_m,
        color_error=registration.color_error,
        assignment_probability=registration.assignment_probability,
        assignment_entropy=registration.assignment_entropy,
        prior_reliability=registration.prior_reliability,
        observation_covariance_m2=registration.observation_covariance_m2,
    )
    summary = episode_registration_summary(
        registration,
        canonical_graph_sha256=canonical.sha256,
        information_boundary={
            "phase": args.phase,
            "observed_prefix_frame_count": prefix_frames,
            "simulator_residual_used": False,
            "future_object_frames_used": False,
            "target_access": False,
        },
    )
    summary.update(
        {
            "protocol_id": authorization["protocol_id"],
            "protocol_config_sha256": authorization["config_sha256"],
            "object_id": args.object_id,
            "episode_id": args.episode_id,
            "phase": args.phase,
            "canonical_reference_episode_id": int(
                object_protocol["canonical_reference_episode_id"]
            ),
            "canonical_graph": canonical_descriptor,
            "input_sha256": {
                "reference_final_data": _sha256_file(args.reference_final_data),
                "episode_final_data": _sha256_file(args.episode_final_data),
            },
            "output_sha256": {
                "canonical_graph": _sha256_file(args.canonical_graph),
                "registered_final_data": _sha256_file(args.registered_final_data),
                "registration_arrays": _sha256_file(args.registration_arrays),
            },
        }
    )
    canonical_summary = dict(summary)
    canonical_summary.pop("result_sha256", None)
    summary["result_sha256"] = hashlib.sha256(
        json.dumps(
            canonical_summary,
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
    return 0 if registration.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
