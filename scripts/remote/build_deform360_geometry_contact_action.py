#!/usr/bin/env python3
"""Build a target-tactile-free action from known motion and frame-zero geometry."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from causal4d_public.deform360_contact_conditioned_action import (
    condition_controller_action,
    geometry_latched_contact_schedule,
    write_contact_conditioned_action_artifact,
)
from causal4d_public.deform360_independent_source import (
    sha256_array,
    sha256_file,
    validate_prediction_only_bundle,
)
from causal4d_public.deform360_reusable_trust_masks import (
    GEOMETRY_CONTACT_MASK_ADDENDUM_ID,
    SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID,
    authorize_reusable_trust_mask_episode,
    load_reusable_trust_mask_addendum,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-input", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--operation", choices=("fit", "held-prediction"), required=True
    )
    parser.add_argument("--fresh-parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--mask-addendum", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_trust_mask_addendum(
        args.fresh_parent_lock,
        args.physics_addendum,
        args.execution_lock,
        args.mask_addendum,
    )
    if protocol["mask_addendum"]["protocol_id"] not in {
        GEOMETRY_CONTACT_MASK_ADDENDUM_ID,
        SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID,
    }:
        raise ValueError("geometry contact action requires a geometry-contact addendum")
    authorization = authorize_reusable_trust_mask_episode(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        operation=args.operation,
    )
    with args.prediction_input.open("rb") as stream:
        prediction = pickle.load(stream)
    validate_prediction_only_bundle(
        prediction,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    source_controller_points = np.asarray(prediction["controller_points"])
    controller_points = np.asarray(source_controller_points, dtype=np.float64)
    initial_object_points = np.asarray(prediction["object_points"][0], dtype=np.float64)
    policy = protocol["mask_addendum"]["geometry_contact_policy"]
    schedule, distances = geometry_latched_contact_schedule(
        controller_points,
        initial_object_points,
        controller_group_size=int(policy["controller_group_size"]),
        maximum_contact_distance_m=float(policy["maximum_contact_distance_m"]),
        confirmation_frames=int(policy["confirmation_frames"]),
    )
    action = condition_controller_action(
        controller_points,
        schedule,
        initial_object_points,
        controller_group_size=int(policy["controller_group_size"]),
        maximum_contact_distance_m=float(policy["maximum_contact_distance_m"]),
    )
    payload = write_contact_conditioned_action_artifact(
        args.archive,
        action,
        object_id=args.object_id,
        episode_id=args.episode_id,
        source_controller_sha256=sha256_array(source_controller_points),
        contact_policy_result_sha256=protocol["mask_addendum_file_sha256"],
        information_boundary={
            "known_future_robot_action_used": True,
            "future_object_observations_used": False,
            "target_tactile_used": False,
            "source_tactile_used": False,
            "contact_model_kind": "geometry_onset_then_latched",
            "dynamic_features": ["known_controller_to_frame_zero_object_distance_m"],
            "onset_geometry_uses_frame_zero_object_only": True,
            "release_inferred_from_initial_geometry": False,
            "authorization": authorization,
            "input_sha256": {
                "prediction_input": sha256_file(args.prediction_input),
                "mask_addendum": protocol["mask_addendum_file_sha256"],
            },
            "contact_schedule_summary": {
                "predicted_active_fraction": float(np.mean(schedule)),
                "predicted_transition_count": int(
                    np.sum(schedule[1:] != schedule[:-1])
                ),
                "minimum_group_distance_m": np.min(distances, axis=0).tolist(),
                "retained_group_count": action.retained_group_count,
                "source_group_count": action.source_group_count,
            },
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not action.falls_back_to_persistence else 2


if __name__ == "__main__":
    raise SystemExit(main())
