#!/usr/bin/env python3
"""Build a strict one-frame Deform360 action from a fitted contact hazard."""

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
from causal4d_public.deform360_replication_controls import (
    predict_causal_contact_transition,
)
from causal4d_public.deform360_replication_transition import (
    validate_transition_fit_artifact,
)
from causal4d_public.deform360_reusable_contact_transition import (
    load_contact_transition_addendum,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-input", type=Path, required=True)
    parser.add_argument("--feature-rollout", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--transition-fit", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-group-size", type=int, default=768)
    parser.add_argument("--maximum-contact-distance-m", type=float, default=0.03)
    parser.add_argument("--dt-seconds", type=float, default=1.0 / 30.0)
    return parser.parse_args()


def _load_openings(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as robot:
        key = "openings_m" if "openings_m" in robot.files else "openings"
        openings = np.asarray(robot[key], dtype=np.float64)
    return openings[:, None] if openings.ndim == 1 else openings


def _load_feature_rollout(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as stored:
        keys = (
            "predicted_object_positions_m",
            "object_positions_m",
            "positions_m",
        )
        present = [key for key in keys if key in stored.files]
        if len(present) != 1:
            raise ValueError(
                "feature rollout must contain exactly one supported position array"
            )
        return np.asarray(stored[present[0]], dtype=np.float64)


def main() -> int:
    args = _parse_args()
    addendum = load_contact_transition_addendum(args.addendum)
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
    if (
        args.controller_group_size < 1
        or controller_points.shape[1] % args.controller_group_size
    ):
        raise ValueError("controller points do not form complete gripper groups")
    group_count = controller_points.shape[1] // args.controller_group_size
    controller_centers = controller_points.reshape(
        len(controller_points), group_count, args.controller_group_size, 3
    ).mean(axis=2)

    openings = _load_openings(args.robot_state)[: len(controller_points)]
    feature_positions = _load_feature_rollout(args.feature_rollout)
    if openings.shape != (len(controller_points), group_count):
        raise ValueError("robot openings differ from controller groups")
    if (
        feature_positions.ndim != 3
        or feature_positions.shape[0] != len(controller_points)
        or feature_positions.shape[2] != 3
    ):
        raise ValueError("feature rollout must have shape (T,N,3)")

    transition_payload = json.loads(args.transition_fit.read_text(encoding="utf-8"))
    transition_model = validate_transition_fit_artifact(transition_payload)
    probabilities, schedule = predict_causal_contact_transition(
        transition_model,
        openings,
        controller_centers,
        feature_positions,
        dt_seconds=args.dt_seconds,
        initial_contact_state=None,
    )
    action = condition_controller_action(
        controller_points,
        schedule,
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
        contact_model_result_sha256=transition_payload["result_sha256"],
        information_boundary={
            "known_future_robot_action_used": True,
            "future_object_observations_used": False,
            "target_tactile_used": False,
            "source_tactile_used_to_fit_contact_model": True,
            "contact_model_kind": "source_fitted_causal_onset_release_hazard",
            "initial_contact_state": "frame-zero onset hazard",
            "feature_rollout_kind": addendum["config"]["model"]["feature_rollout"],
            "dynamic_features": list(transition_model.feature_names),
            "input_sha256": {
                "prediction_input": sha256_file(args.prediction_input),
                "feature_rollout": sha256_file(args.feature_rollout),
                "robot_state": sha256_file(args.robot_state),
                "transition_fit": sha256_file(args.transition_fit),
                "addendum": sha256_file(args.addendum),
            },
            "contact_schedule_summary": {
                "predicted_active_fraction": float(np.mean(schedule)),
                "predicted_transition_count": int(
                    np.sum(schedule[1:] != schedule[:-1])
                ),
                "mean_active_probability": float(np.mean(probabilities)),
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
