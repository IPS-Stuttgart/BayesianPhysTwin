#!/usr/bin/env python3
"""Score the locked causal contact-transport grid on one source episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_causal_transport import (
    CausalContactTransportConfig,
    causal_contact_transport_prediction,
)
from causal4d_public.deform360_causal_transport_method import (
    causal_transport_candidates,
    load_causal_transport_method,
)
from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_reusable_sota_selection import (
    score_reusable_sota_trajectory,
)
from causal4d_public.deform360_reusable_sota_window import (
    authorize_development_fit_window,
    load_reusable_sota_window,
)
from causal4d_public.deform360_sota_processing import (
    authorize_development_processing,
    validate_development_final_data_input,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--window-addendum", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--development-observations", type=Path, required=True)
    parser.add_argument("--source-final-data", type=Path, required=True)
    parser.add_argument("--source-robot-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _aligned_openings(
    robot_path: Path,
    *,
    frame_count: int,
    group_count: int,
    expected_raw_frame_count: int,
) -> np.ndarray:
    with np.load(robot_path, allow_pickle=False) as archive:
        _require("openings" in archive, "robot state lacks gripper openings")
        openings = np.asarray(archive["openings"], dtype=np.float64)
    if openings.ndim == 1:
        openings = openings[:, None]
    _require(
        openings.ndim == 2
        and openings.shape == (expected_raw_frame_count, group_count)
        and np.all(np.isfinite(openings)),
        "robot openings differ from the locked staged action window",
    )
    # Deform360's public reconstruction drops the final tracking tail.
    return openings[:frame_count].copy()


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_sota_config(args.protocol)
    window = load_reusable_sota_window(args.window_addendum)
    method = load_causal_transport_method(args.method)
    processing_authorization = authorize_development_processing(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        role="fit",
    )
    window_authorization = authorize_development_fit_window(
        protocol,
        window,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    _require(
        method["config"]["parent_config_sha256"] == protocol["config_sha256"]
        and method["config"]["window_config_sha256"] == window["config_sha256"],
        "causal-transport source bank uses another parent lock",
    )
    _require(
        args.episode_id in method["config"]["selection"]["fit_episode_ids"],
        "episode is outside the causal-transport source panel",
    )
    observations = json.loads(
        args.development_observations.read_text(encoding="utf-8")
    )
    final_data_validation = validate_development_final_data_input(
        observations,
        authorization=processing_authorization,
        final_data_path=args.source_final_data,
    )
    with args.source_final_data.open("rb") as stream:
        source = pickle.load(stream)
    target = np.asarray(source["object_points"], dtype=np.float64)
    controllers = np.asarray(source["controller_points"], dtype=np.float64)
    frames = window["config"]["frame_protocol"]
    frame_count = int(frames["processed_frame_count"])
    raw_count = int(window["config"]["window_selection"]["window_length_frames"])
    contact = method["config"]["contact_policy"]
    group_size = int(contact["controller_group_size"])
    _require(
        target.ndim == controllers.ndim == 3
        and target.shape[0] == controllers.shape[0] == frame_count
        and target.shape[2] == controllers.shape[2] == 3
        and controllers.shape[1] % group_size == 0
        and int(final_data_validation["point_frame_count"]) == frame_count,
        "source trajectories differ from the locked causal-transport horizon",
    )
    group_count = controllers.shape[1] // group_size
    openings = _aligned_openings(
        args.source_robot_state,
        frame_count=frame_count,
        group_count=group_count,
        expected_raw_frame_count=raw_count,
    )
    ranges = {
        "full": frames["evaluation_range_half_open"],
        **frames["horizon_ranges_half_open"],
    }
    persistence = np.repeat(target[:1], frame_count, axis=0)
    persistence_metrics = score_reusable_sota_trajectory(
        target,
        persistence,
        horizon_ranges_half_open=ranges,
    )

    records = []
    for candidate in causal_transport_candidates(method):
        config = CausalContactTransportConfig(
            controller_group_size=group_size,
            maximum_contact_distance_m=float(
                contact["maximum_contact_distance_m"]
            ),
            opening_contact_threshold_m=float(
                contact["opening_contact_threshold_m"]
            ),
            confirmation_frames=int(contact["confirmation_frames"]),
            base_support_scale_m=float(candidate["base_support_scale_m"]),
            support_growth_per_travel=float(
                candidate["support_growth_per_travel"]
            ),
            initial_contact_gain=float(candidate["initial_contact_gain"]),
            acquired_contact_gain=float(candidate["acquired_contact_gain"]),
            transform_mode=str(candidate["transform_mode"]),
        )
        result = causal_contact_transport_prediction(
            target[0],
            controllers,
            openings,
            config=config,
        )
        metrics = score_reusable_sota_trajectory(
            target,
            result.prediction_m,
            horizon_ranges_half_open=ranges,
        )
        records.append(
            {
                **candidate,
                "valid": True,
                "metrics": metrics,
                "diagnostics": result.diagnostics(),
                "prediction_array_sha256": _array_sha256(result.prediction_m),
            }
        )

    candidates = causal_transport_candidates(method)
    _require(
        len(records) == len(candidates) == 49
        and records[0]["label"] == "persistence"
        and records[0]["diagnostics"]["exact_persistence"] is True
        and records[0]["metrics"] == persistence_metrics,
        "causal-transport persistence control is not exact",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalTransportSourceCandidateBank",
        "protocol_id": method["config"]["protocol_id"],
        "method_config_sha256": method["config_sha256"],
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "processing_authorization": processing_authorization,
        "window_authorization": window_authorization,
        "candidate_count": len(records),
        "candidate_order": [record["label"] for record in records],
        "persistence_metrics": persistence_metrics,
        "records": records,
        "input_sha256": {
            "development_observations": _sha256_file(
                args.development_observations
            ),
            "source_final_data": _sha256_file(args.source_final_data),
            "source_robot_state": _sha256_file(args.source_robot_state),
            "initial_object_points": _array_sha256(target[0]),
            "controller_points": _array_sha256(controllers),
            "aligned_openings": _array_sha256(openings),
        },
        "information_boundary": {
            "source_future_outcome_used_for_candidate_scoring": True,
            "source_future_outcome_used_for_prediction": False,
            "known_future_robot_action_used": True,
            "known_future_opening_used": True,
            "future_tactile_used_for_prediction": False,
            "held_development_outcome_read": False,
            "confirmatory_object_read": False,
            "pokeflex_target_read": False,
        },
        "passed": True,
        "claim_boundary": (
            "source-only causal-transport scoring; no held outcome or direct "
            "Deform360 Table 4 claim"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    _require(not args.output.exists(), f"source bank exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "object_id": args.object_id,
                "episode_id": args.episode_id,
                "candidate_count": len(records),
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
