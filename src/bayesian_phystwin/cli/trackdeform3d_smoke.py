"""Prepare, predict, and score the leakage-safe TrackDeform3D smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.trackdeform3d_smoke import (
    TrackDeform3DEvaluatorTarget,
    TrackDeform3DPredictionInput,
    TrackDeform3DSmokeConfig,
    TrackDeform3DSmokePrediction,
    evaluate_trackdeform3d_smoke,
    predict_trackdeform3d_smoke,
    split_trackdeform3d_carriers,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_pose_translations(path: Path, indices: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as stored:
        return np.stack(
            [np.asarray(stored[f"arr_{int(index)}"][:3]) for index in indices]
        )


def _to_camera_m(translations_m: np.ndarray, base_to_camera: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([translations_m, np.ones(len(translations_m))])
    return (homogeneous @ base_to_camera.T)[:, :3]


def _prepare(args: argparse.Namespace) -> None:
    config = TrackDeform3DSmokeConfig()
    total = config.prefix_frames + config.future_frames
    with np.load(args.tracker_npz, allow_pickle=False) as stored:
        trajectory_m = np.asarray(stored["full"], dtype=float)[:total] / 1000.0
        edges = np.asarray(stored["edge_connection"], dtype=np.int64)
        rest_lengths_m = np.asarray(stored["reference_lengths"], dtype=float) / 1000.0
    indices = np.arange(args.clip_start, args.clip_start + total, dtype=np.int64)
    left = _load_pose_translations(args.chunk_dir / "left_arm_poses.npz", indices)
    right = _load_pose_translations(args.chunk_dir / "right_arm_poses.npz", indices)
    with np.load(args.calibration, allow_pickle=False) as calibration:
        left_camera = _to_camera_m(left, calibration["T_left_base2cam"])
        right_camera = _to_camera_m(right, calibration["T_right_base2cam"])
    action = np.stack([left_camera, right_camera], axis=1)
    prediction_input, target = split_trackdeform3d_carriers(
        trajectory_m,
        edges,
        rest_lengths_m,
        action,
        config=config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "prediction_input.npz"
    target_path = args.output_dir / "evaluator_target.npz"
    np.savez_compressed(
        input_path,
        frame_zero_points_m=prediction_input.frame_zero_points_m,
        edges=prediction_input.edges,
        rest_lengths_m=prediction_input.rest_lengths_m,
        end_effector_positions_m=prediction_input.end_effector_positions_m,
        observed_identity_ids=prediction_input.observed_identity_ids,
        observed_prefix_points_m=prediction_input.observed_prefix_points_m,
    )
    np.savez_compressed(
        target_path,
        hidden_identity_ids=target.hidden_identity_ids,
        hidden_future_points_m=target.hidden_future_points_m,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "trackdeform3d-public-smoke-v1",
        "clip_start": args.clip_start,
        "config": asdict(config),
        "prediction_input_sha256": _file_sha256(input_path),
        "evaluator_target_sha256": _file_sha256(target_path),
        "source": {
            "tracker_npz_sha256": _file_sha256(args.tracker_npz),
            "left_arm_poses_sha256": _file_sha256(
                args.chunk_dir / "left_arm_poses.npz"
            ),
            "right_arm_poses_sha256": _file_sha256(
                args.chunk_dir / "right_arm_poses.npz"
            ),
            "calibration_sha256": _file_sha256(args.calibration),
        },
        "information_boundary": {
            "prediction_input_contains_hidden_future": False,
            "prediction_process_requires_evaluator_target": False,
            "future_robot_action_is_known": True,
            "tracker_output_is_pseudo_observation": True,
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _write_json(args.output_dir / "carrier_manifest.json", manifest)


def _load_prediction_input(path: Path) -> TrackDeform3DPredictionInput:
    with np.load(path, allow_pickle=False) as stored:
        return TrackDeform3DPredictionInput(
            frame_zero_points_m=stored["frame_zero_points_m"],
            edges=stored["edges"],
            rest_lengths_m=stored["rest_lengths_m"],
            end_effector_positions_m=stored["end_effector_positions_m"],
            observed_identity_ids=stored["observed_identity_ids"],
            observed_prefix_points_m=stored["observed_prefix_points_m"],
        )


def _predict(args: argparse.Namespace) -> None:
    config = TrackDeform3DSmokeConfig()
    prediction = predict_trackdeform3d_smoke(
        _load_prediction_input(args.prediction_input),
        config=config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "prediction.npz"
    variance = (
        prediction.guarded_variance_m2
        if prediction.guarded_variance_m2 is not None
        else np.empty((0,), dtype=float)
    )
    np.savez_compressed(
        prediction_path,
        observed_identity_ids=prediction.observed_identity_ids,
        persistence_m=prediction.persistence_m,
        constant_velocity_m=prediction.constant_velocity_m,
        physical_m=prediction.physical_m,
        guarded_bayesian_m=prediction.guarded_bayesian_m,
        guarded_variance_m2=variance,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "trackdeform3d-public-smoke-v1",
        "prediction_input_sha256": _file_sha256(args.prediction_input),
        "prediction_sha256": _file_sha256(prediction_path),
        "config": asdict(config),
        "gate": prediction.gate,
        "information_boundary": {
            "evaluator_target_opened": False,
            "future_object_trajectory_accepted_by_predictor": False,
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _write_json(args.output_dir / "prediction_manifest.json", manifest)


def _load_prediction(path: Path, manifest_path: Path) -> TrackDeform3DSmokePrediction:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as stored:
        variance = np.asarray(stored["guarded_variance_m2"])
        return TrackDeform3DSmokePrediction(
            observed_identity_ids=stored["observed_identity_ids"],
            persistence_m=stored["persistence_m"],
            constant_velocity_m=stored["constant_velocity_m"],
            physical_m=stored["physical_m"],
            guarded_bayesian_m=stored["guarded_bayesian_m"],
            guarded_variance_m2=None if variance.size == 0 else variance,
            gate=manifest["gate"],
        )


def _score(args: argparse.Namespace) -> None:
    config = TrackDeform3DSmokeConfig()
    prediction = _load_prediction(args.prediction, args.prediction_manifest)
    with np.load(args.evaluator_target, allow_pickle=False) as stored:
        target = TrackDeform3DEvaluatorTarget(
            hidden_identity_ids=stored["hidden_identity_ids"],
            hidden_future_points_m=stored["hidden_future_points_m"],
        )
    result = evaluate_trackdeform3d_smoke(
        prediction,
        target,
        nominal_coverage=config.nominal_coverage,
    )
    result["prediction_sha256"] = _file_sha256(args.prediction)
    result["evaluator_target_sha256"] = _file_sha256(args.evaluator_target)
    result["result_sha256"] = _canonical_sha256(result)
    _write_json(args.output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--tracker-npz", type=Path, required=True)
    prepare.add_argument("--chunk-dir", type=Path, required=True)
    prepare.add_argument("--calibration", type=Path, required=True)
    prepare.add_argument("--clip-start", type=int, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.set_defaults(handler=_prepare)

    predict = commands.add_parser("predict")
    predict.add_argument("--prediction-input", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.set_defaults(handler=_predict)

    score = commands.add_parser("score")
    score.add_argument("--prediction", type=Path, required=True)
    score.add_argument("--prediction-manifest", type=Path, required=True)
    score.add_argument("--evaluator-target", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=_score)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
