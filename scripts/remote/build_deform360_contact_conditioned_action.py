#!/usr/bin/env python3
"""Build a source-trained, target-tactile-free Deform360 contact action."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from causal4d_public.deform360_contact_conditioned_action import (
    condition_controller_action,
    write_contact_conditioned_action_artifact,
)
from causal4d_public.deform360_independent_source import (
    sha256_array,
    sha256_file,
    validate_prediction_only_bundle,
)
from causal4d_public.deform360_replication_contact import (
    ReplicationOpeningContactModel,
    causal_confirmed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-input", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--contact-model-json", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-group-size", type=int, default=768)
    parser.add_argument("--maximum-contact-distance-m", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
    if controller_points.shape[1] % args.controller_group_size:
        raise ValueError("controller points do not form complete gripper groups")
    group_count = controller_points.shape[1] // args.controller_group_size

    with np.load(args.robot_state, allow_pickle=False) as robot:
        openings = np.asarray(robot["openings"], dtype=np.float64)
    if openings.ndim == 1:
        openings = openings[:, None]
    openings = openings[: len(controller_points)]
    if openings.shape != (len(controller_points), group_count):
        raise ValueError("robot openings differ from the prediction controller groups")

    model_payload = json.loads(args.contact_model_json.read_text(encoding="utf-8"))
    model = ReplicationOpeningContactModel(**model_payload)
    contact_active = np.column_stack(
        [
            causal_confirmed(
                openings[:, group] <= model.opening_threshold_m,
                model.confirmation_frames,
            )
            for group in range(group_count)
        ]
    )
    action = condition_controller_action(
        controller_points,
        contact_active,
        initial_object_points,
        controller_group_size=args.controller_group_size,
        maximum_contact_distance_m=args.maximum_contact_distance_m,
    )
    payload = write_contact_conditioned_action_artifact(
        args.archive,
        action,
        object_id=args.object_id,
        episode_id=args.episode_id,
        source_controller_sha256=sha256_array(source_controller_points),
        contact_model_result_sha256=sha256_file(args.contact_model_json),
        information_boundary={
            "known_future_robot_action_used": True,
            "future_object_observations_used": False,
            "target_tactile_used": False,
            "source_tactile_used_to_fit_contact_model": True,
            "contact_model_kind": "source_fitted_opening_with_onset_geometry_gate",
            "dynamic_features": ["gripper_openness_m"],
            "onset_geometry_uses_frame_zero_object_only": True,
            "input_sha256": {
                "prediction_input": sha256_file(args.prediction_input),
                "robot_state": sha256_file(args.robot_state),
                "contact_model": sha256_file(args.contact_model_json),
            },
            "contact_schedule_summary": {
                "predicted_active_fraction": float(np.mean(contact_active)),
                "predicted_transition_count": int(
                    np.sum(contact_active[1:] != contact_active[:-1])
                ),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
