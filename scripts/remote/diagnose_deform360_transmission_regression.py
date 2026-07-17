#!/usr/bin/env python3
"""Cross-fit source-only action transmission on the exhausted Deform360 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d_public.deform360_independent_source import sha256_file
from causal4d_public.deform360_phystwin_trust import (
    CausalTrustEpisode,
    score_causal_trust_interval,
)


RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FEATURE_SETS = {
    "closure_only": ("mean_minimum_gripper_closure",),
    "closure_motion": (
        "mean_minimum_gripper_closure",
        "all_grippers_closed_fraction_075",
        "mean_closed_weighted_path_length_m",
        "mean_displacement_from_window_start_m",
        "bimanual",
    ),
    "full_action": (
        "mean_closure",
        "mean_minimum_gripper_closure",
        "all_grippers_closed_fraction_075",
        "mean_closed_weighted_path_length_m",
        "mean_displacement_from_window_start_m",
        "mean_gripper_path_length_m",
        "bimanual",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-diagnosis", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return payload


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"pickle is not a dictionary: {path}")
    return payload


def _first_existing(directory: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of {tuple(names)} exists in {directory}")


def _episode_dir(root: Path, object_id: str, episode_id: int) -> Path:
    return root / f"{object_id}-ep{episode_id:04d}"


def _ridge_fit(
    features: np.ndarray, labels: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-8)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ y,
    )
    return coefficients, mean, scale


def _ridge_predict(
    model: tuple[np.ndarray, np.ndarray, np.ndarray], features: np.ndarray
) -> np.ndarray:
    coefficients, mean, scale = model
    x = np.asarray(features, dtype=np.float64)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    return np.clip(design @ coefficients, 0.0, 1.2)


def _score_alpha(row: Mapping[str, Any], alpha: float, interval: str) -> dict[str, Any]:
    prediction = row["persistence"] + float(alpha) * row["response"]
    bounds = {"future": (1, 76), "late": (51, 76)}
    start, stop = bounds[interval]
    return score_causal_trust_interval(row["episode"], prediction, start, stop)


def _aggregate(rows: Sequence[Mapping[str, Any]], alpha: np.ndarray, interval: str):
    scored = [_score_alpha(row, value, interval) for row, value in zip(rows, alpha)]
    track = np.asarray([float(item["track_rmse_m"]) for item in scored])
    track_base = np.asarray(
        [float(item["persistence_track_rmse_m"]) for item in scored]
    )
    chamfer = np.asarray([float(item["chamfer_m"]) for item in scored])
    chamfer_base = np.asarray([float(item["persistence_chamfer_m"]) for item in scored])
    return {
        "track_improvement_fraction": float(1.0 - np.mean(track) / np.mean(track_base)),
        "chamfer_improvement_fraction": float(
            1.0 - np.mean(chamfer) / np.mean(chamfer_base)
        ),
        "joint_win_count": int(np.sum((track < track_base) & (chamfer < chamfer_base))),
        "relative_score_vs_persistence": float(
            0.5
            * (
                np.mean(track) / np.mean(track_base)
                + np.mean(chamfer) / np.mean(chamfer_base)
            )
        ),
    }


def _features(row: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    return np.asarray([float(row["features"][name]) for name in names])


def _nested_cross_fit(
    rows: list[dict[str, Any]], feature_names: Sequence[str]
) -> dict[str, Any]:
    objects = sorted({str(row["object_id"]) for row in rows})
    selected_alpha = np.zeros(len(rows), dtype=np.float64)
    folds = []
    for held_object in objects:
        train_indices = [
            index for index, row in enumerate(rows) if row["object_id"] != held_object
        ]
        held_indices = [
            index for index, row in enumerate(rows) if row["object_id"] == held_object
        ]
        inner_objects = sorted(
            {str(rows[index]["object_id"]) for index in train_indices}
        )
        ridge_scores = []
        for ridge in RIDGE_GRID:
            inner_reference = []
            inner_alpha = []
            for validation_object in inner_objects:
                fit_indices = [
                    index
                    for index in train_indices
                    if rows[index]["object_id"] != validation_object
                ]
                validation_indices = [
                    index
                    for index in train_indices
                    if rows[index]["object_id"] == validation_object
                ]
                model = _ridge_fit(
                    np.stack(
                        [_features(rows[index], feature_names) for index in fit_indices]
                    ),
                    np.asarray([rows[index]["oracle_alpha"] for index in fit_indices]),
                    ridge,
                )
                predicted = _ridge_predict(
                    model,
                    np.stack(
                        [
                            _features(rows[index], feature_names)
                            for index in validation_indices
                        ]
                    ),
                )
                inner_reference.extend(
                    rows[index]["oracle_alpha"] for index in validation_indices
                )
                inner_alpha.extend(predicted.tolist())
            ridge_scores.append(
                {
                    "ridge": ridge,
                    "inner_oracle_alpha_mse": float(
                        np.mean(
                            (np.asarray(inner_alpha) - np.asarray(inner_reference)) ** 2
                        )
                    ),
                }
            )
        selected_ridge = min(
            ridge_scores,
            key=lambda item: (item["inner_oracle_alpha_mse"], item["ridge"]),
        )["ridge"]
        model = _ridge_fit(
            np.stack(
                [_features(rows[index], feature_names) for index in train_indices]
            ),
            np.asarray([rows[index]["oracle_alpha"] for index in train_indices]),
            float(selected_ridge),
        )
        held_prediction = _ridge_predict(
            model,
            np.stack([_features(rows[index], feature_names) for index in held_indices]),
        )
        selected_alpha[held_indices] = held_prediction
        folds.append(
            {
                "held_out_object": held_object,
                "selected_ridge": selected_ridge,
                "inner_candidates": ridge_scores,
            }
        )
    return {
        "feature_names": list(feature_names),
        "folds": folds,
        "selected_alpha_by_episode": {
            row["episode_key"]: float(alpha) for row, alpha in zip(rows, selected_alpha)
        },
        "future": _aggregate(rows, selected_alpha, "future"),
        "late": _aggregate(rows, selected_alpha, "late"),
    }


def main() -> int:
    args = _parse_args()
    diagnosis = _load_json(args.failure_diagnosis)
    rows = []
    input_hashes: dict[str, Any] = {}
    for record in diagnosis["episodes"]:
        object_id = str(record["object_id"])
        episode_id = int(record["episode_id"])
        result_dir = _episode_dir(args.result_root, object_id, episode_id)
        stage_dir = (
            _episode_dir(args.stage_root, object_id, episode_id) / "episode_0000"
        )
        prediction_path = _first_existing(
            result_dir, ("prediction.npz", "sealed_prediction.npz")
        )
        target_path = result_dir / "target_data.pkl"
        robot_path = stage_dir / "robot" / "robot.npz"
        alignment_path = stage_dir / "action_aligned_source_staging.json"
        with np.load(prediction_path, allow_pickle=False) as stored:
            prediction = np.asarray(stored["prediction_m"], dtype=np.float64)
            persistence = np.asarray(stored["persistence_m"], dtype=np.float64)
            driven = np.asarray(stored["driven_readout_m"], dtype=np.float64)
            zero = np.asarray(stored["zero_action_readout_m"], dtype=np.float64)
        target_data = _load_pickle(target_path)
        target = np.asarray(target_data["object_points"], dtype=np.float64)
        episode = CausalTrustEpisode(
            episode_id=str(record["episode_key"]),
            target_m=target,
            visibility=np.asarray(target_data["object_visibilities"], dtype=bool),
            validity=np.asarray(target_data["object_motions_valid"], dtype=bool),
            driven_m=driven,
            zero_action_m=zero,
            train_stop_frame=60,
            source_data_sha256=sha256_file(target_path),
            driven_trajectory_sha256=sha256_file(prediction_path),
            zero_action_trajectory_sha256=sha256_file(prediction_path),
        )
        alignment = _load_json(alignment_path)
        locked = alignment["action_summary"]["locked_window"]
        features = dict(record["action_features"])
        features.update(
            {
                "mean_closed_weighted_path_length_m": float(
                    locked["mean_closed_weighted_path_length_m"]
                ),
                "mean_displacement_from_window_start_m": float(
                    locked["mean_displacement_from_window_start_m"]
                ),
                "mean_gripper_path_length_m": float(
                    locked["mean_gripper_path_length_m"]
                ),
                "bimanual": float(alignment["action_summary"]["gripper_count"] == 2),
            }
        )
        rows.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "episode_key": record["episode_key"],
                "features": features,
                "oracle_alpha": float(record["best_nonnegative_alpha"]),
                "episode": episode,
                "persistence": persistence,
                "response": (prediction - persistence) / 0.9,
            }
        )
        input_hashes[str(record["episode_key"])] = {
            "prediction": sha256_file(prediction_path),
            "target": sha256_file(target_path),
            "robot": sha256_file(robot_path),
            "alignment": sha256_file(alignment_path),
        }

    methods = {
        name: _nested_cross_fit(rows, features)
        for name, features in FEATURE_SETS.items()
    }
    fixed_alpha = np.full(len(rows), 0.9)
    oracle_alpha = np.asarray([row["oracle_alpha"] for row in rows])
    closure_alpha_by_episode = diagnosis["exploratory_group_cross_fit"][
        "selected_alpha_by_episode"
    ]
    closure_alpha = np.asarray(
        [closure_alpha_by_episode[row["episode_key"]] for row in rows]
    )
    baselines = {}
    for name, alpha in {
        "persistence": np.zeros(len(rows)),
        "frozen_fixed_response": fixed_alpha,
        "cross_fitted_binary_closure": closure_alpha,
        "non_deployable_alpha_oracle": oracle_alpha,
    }.items():
        baselines[name] = {
            "future": _aggregate(rows, alpha, "future"),
            "late": _aggregate(rows, alpha, "late"),
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SourceTransmissionRegressionDiagnosis",
        "episode_count": len(rows),
        "feature_sets": methods,
        "baselines": baselines,
        "input_sha256": {
            "failure_diagnosis": sha256_file(args.failure_diagnosis),
            "episodes": input_hashes,
        },
        "information_boundary": {
            "independent_source_outcomes_read": True,
            "calibration_outcomes_read": False,
            "target_actions_read": False,
            "target_initial_frames_read": False,
            "target_outcomes_read": False,
            "outer_cross_fit_unit": "object",
            "ridge_selected_by_nested_object_cross_fit": True,
        },
        "claim_boundary": (
            "post-failure exploratory source diagnosis; feature-family selection "
            "requires a new unused-object lock"
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
    print(
        json.dumps(
            {
                "baselines": baselines,
                "feature_sets": {
                    name: {"future": value["future"], "late": value["late"]}
                    for name, value in methods.items()
                },
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
