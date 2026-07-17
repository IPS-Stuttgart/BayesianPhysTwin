#!/usr/bin/env python3
"""Cross-fit same-object PhysTwin trust on the exhausted Deform360 source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d_public.deform360_independent_source import sha256_file
from causal4d_public.deform360_phystwin_trust import (
    CausalTrustEpisode,
    score_causal_trust_interval,
)


RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-diagnosis", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path)
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


def _aggregate(
    rows: Sequence[Mapping[str, Any]], alpha: np.ndarray, interval: str
) -> dict[str, Any]:
    scored = [_score_alpha(row, value, interval) for row, value in zip(rows, alpha)]
    track = np.asarray([float(item["track_rmse_m"]) for item in scored])
    track_base = np.asarray(
        [float(item["persistence_track_rmse_m"]) for item in scored]
    )
    chamfer = np.asarray([float(item["chamfer_m"]) for item in scored])
    chamfer_base = np.asarray(
        [float(item["persistence_chamfer_m"]) for item in scored]
    )
    track_change = track / track_base - 1.0
    chamfer_change = chamfer / chamfer_base - 1.0
    return {
        "track_rmse_m": float(np.mean(track)),
        "persistence_track_rmse_m": float(np.mean(track_base)),
        "track_improvement_fraction": float(1.0 - np.mean(track) / np.mean(track_base)),
        "chamfer_m": float(np.mean(chamfer)),
        "persistence_chamfer_m": float(np.mean(chamfer_base)),
        "chamfer_improvement_fraction": float(
            1.0 - np.mean(chamfer) / np.mean(chamfer_base)
        ),
        "joint_win_count": int(np.sum((track < track_base) & (chamfer < chamfer_base))),
        "track_degradation_count": int(np.sum(track > track_base)),
        "chamfer_degradation_count": int(np.sum(chamfer > chamfer_base)),
        "maximum_track_degradation_fraction": float(np.max(track_change)),
        "maximum_chamfer_degradation_fraction": float(np.max(chamfer_change)),
        "relative_score_vs_persistence": float(
            0.5
            * (
                np.mean(track) / np.mean(track_base)
                + np.mean(chamfer) / np.mean(chamfer_base)
            )
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-6))


def _robot_features(robot_path: Path) -> dict[str, float]:
    with np.load(robot_path, allow_pickle=False) as stored:
        action = np.asarray(stored["actions"], dtype=np.float64)
        openings = np.asarray(stored["openings"], dtype=np.float64)
    if action.ndim == 3 and action.shape[1:] == (5, 3):
        action = action[:, None]
    if openings.ndim == 1:
        openings = openings[:, None]
    if action.ndim != 4 or action.shape[2:] != (5, 3):
        raise ValueError(f"unexpected robot action shape: {action.shape}")
    if openings.shape != action.shape[:2]:
        raise ValueError(
            f"robot opening shape {openings.shape} does not match {action.shape[:2]}"
        )
    centres = np.mean(action, axis=2)
    order = np.argsort(centres[0, :, 0])
    centres = centres[:, order]
    openings = openings[:, order]
    steps = np.diff(centres, axis=0)
    delta = centres[-1] - centres[0]
    axis_path = np.sum(np.abs(steps), axis=0)
    path = np.sum(np.linalg.norm(steps, axis=-1), axis=0)
    features: dict[str, float] = {
        "gripper_count": float(centres.shape[1]),
        "bimanual": float(centres.shape[1] == 2),
        "mean_gripper_path_m": float(np.mean(path)),
        "max_gripper_path_m": float(np.max(path)),
        "mean_endpoint_displacement_m": float(np.mean(np.linalg.norm(delta, axis=-1))),
        "mean_vertical_displacement_m": float(np.mean(delta[:, 2])),
        "mean_absolute_vertical_displacement_m": float(np.mean(np.abs(delta[:, 2]))),
        "mean_horizontal_displacement_m": float(np.mean(np.linalg.norm(delta[:, :2], axis=-1))),
        "mean_vertical_path_m": float(np.mean(axis_path[:, 2])),
        "mean_horizontal_path_m": float(np.mean(np.linalg.norm(axis_path[:, :2], axis=-1))),
        "mean_opening_start_m": float(np.mean(openings[0])),
        "mean_opening_end_m": float(np.mean(openings[-1])),
        "mean_opening_change_m": float(np.mean(openings[-1] - openings[0])),
        "minimum_opening_m": float(np.min(openings)),
    }
    features["vertical_to_horizontal_path_ratio"] = _safe_ratio(
        features["mean_vertical_path_m"], features["mean_horizontal_path_m"]
    )
    for gripper_index in range(min(2, centres.shape[1])):
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            features[f"gripper_{gripper_index}_delta_{axis_name}_m"] = float(
                delta[gripper_index, axis_index]
            )
            features[f"gripper_{gripper_index}_axis_path_{axis_name}_m"] = float(
                axis_path[gripper_index, axis_index]
            )
    for gripper_index in range(centres.shape[1], 2):
        for axis_name in ("x", "y", "z"):
            features[f"gripper_{gripper_index}_delta_{axis_name}_m"] = 0.0
            features[f"gripper_{gripper_index}_axis_path_{axis_name}_m"] = 0.0
    if centres.shape[1] == 2:
        separation = np.linalg.norm(centres[:, 1] - centres[:, 0], axis=-1)
        features.update(
            {
                "gripper_separation_start_m": float(separation[0]),
                "gripper_separation_end_m": float(separation[-1]),
                "gripper_separation_change_m": float(separation[-1] - separation[0]),
                "gripper_separation_range_m": float(np.ptp(separation)),
                "gripper_delta_dot_m2": float(np.dot(delta[0], delta[1])),
            }
        )
    else:
        features.update(
            {
                "gripper_separation_start_m": 0.0,
                "gripper_separation_end_m": 0.0,
                "gripper_separation_change_m": 0.0,
                "gripper_separation_range_m": 0.0,
                "gripper_delta_dot_m2": 0.0,
            }
        )
    return features


def _response_features(
    response: np.ndarray,
    persistence: np.ndarray,
    robot_features: Mapping[str, float],
) -> dict[str, float]:
    relative = np.asarray(response - response[0:1], dtype=np.float64)
    displacement = np.linalg.norm(relative, axis=-1)
    frame_rms = np.sqrt(np.mean(relative**2, axis=(1, 2)))
    geometry = np.asarray(persistence[0], dtype=np.float64)
    extents = np.ptp(geometry, axis=0)
    singular_values = np.linalg.svd(
        geometry - np.mean(geometry, axis=0, keepdims=True),
        full_matrices=False,
        compute_uv=False,
    )
    pca_scales = singular_values / np.sqrt(max(len(geometry) - 1, 1))
    features = {
        "response_rms_m": float(np.sqrt(np.mean(relative[1:] ** 2))),
        "response_mean_displacement_m": float(np.mean(displacement[1:])),
        "response_early_mean_displacement_m": float(np.mean(displacement[1:26])),
        "response_late_mean_displacement_m": float(np.mean(displacement[51:76])),
        "response_endpoint_mean_displacement_m": float(np.mean(displacement[-1])),
        "response_p90_displacement_m": float(np.quantile(displacement[1:], 0.9)),
        "response_max_displacement_m": float(np.max(displacement[1:])),
        "response_frame_rms_growth_m": float(frame_rms[-1] - frame_rms[1]),
        "geometry_extent_x_m": float(extents[0]),
        "geometry_extent_y_m": float(extents[1]),
        "geometry_extent_z_m": float(extents[2]),
        "geometry_diagonal_m": float(np.linalg.norm(extents)),
        "geometry_pca_0_m": float(pca_scales[0]),
        "geometry_pca_1_m": float(pca_scales[1]),
        "geometry_pca_2_m": float(pca_scales[2]),
    }
    features["response_late_to_early_ratio"] = _safe_ratio(
        features["response_late_mean_displacement_m"],
        features["response_early_mean_displacement_m"],
    )
    features["response_to_action_path_ratio"] = _safe_ratio(
        features["response_mean_displacement_m"],
        float(robot_features["mean_gripper_path_m"]),
    )
    features["response_to_geometry_ratio"] = _safe_ratio(
        features["response_mean_displacement_m"], features["geometry_diagonal_m"]
    )
    return features


def _feature_vector(row: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    return np.asarray([float(row["features"][name]) for name in names])


def _nested_episode_cross_fit(
    rows: list[dict[str, Any]], feature_names: Sequence[str]
) -> dict[str, Any]:
    selected_alpha = np.zeros(len(rows), dtype=np.float64)
    folds = []
    for held_index, held_row in enumerate(rows):
        train_indices = [index for index in range(len(rows)) if index != held_index]
        ridge_scores = []
        for ridge in RIDGE_GRID:
            inner_reference = []
            inner_alpha = []
            for validation_index in train_indices:
                fit_indices = [
                    index for index in train_indices if index != validation_index
                ]
                model = _ridge_fit(
                    np.stack(
                        [
                            _feature_vector(rows[index], feature_names)
                            for index in fit_indices
                        ]
                    ),
                    np.asarray([rows[index]["oracle_alpha"] for index in fit_indices]),
                    ridge,
                )
                predicted = _ridge_predict(
                    model,
                    _feature_vector(rows[validation_index], feature_names)[None],
                )
                inner_reference.append(float(rows[validation_index]["oracle_alpha"]))
                inner_alpha.append(float(predicted[0]))
            inner_error = np.asarray(inner_alpha) - np.asarray(inner_reference)
            ridge_scores.append(
                {
                    "ridge": ridge,
                    "inner_oracle_alpha_mse": float(np.mean(inner_error**2)),
                }
            )
        selected_ridge = min(
            ridge_scores,
            key=lambda item: (item["inner_oracle_alpha_mse"], item["ridge"]),
        )["ridge"]
        model = _ridge_fit(
            np.stack(
                [_feature_vector(rows[index], feature_names) for index in train_indices]
            ),
            np.asarray([rows[index]["oracle_alpha"] for index in train_indices]),
            float(selected_ridge),
        )
        selected_alpha[held_index] = _ridge_predict(
            model, _feature_vector(held_row, feature_names)[None]
        )[0]
        folds.append(
            {
                "held_out_episode": held_row["episode_key"],
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


def _object_mean_cross_fit(rows: list[dict[str, Any]]) -> np.ndarray:
    selected = []
    for held_index, held_row in enumerate(rows):
        matches = [
            row["oracle_alpha"]
            for index, row in enumerate(rows)
            if index != held_index and row["object_id"] == held_row["object_id"]
        ]
        if not matches:
            matches = [
                row["oracle_alpha"] for index, row in enumerate(rows) if index != held_index
            ]
        selected.append(float(np.mean(matches)))
    return np.asarray(selected, dtype=np.float64)


def _fit_full_source_candidate(
    rows: list[dict[str, Any]], feature_names: Sequence[str]
) -> dict[str, Any]:
    closure_feature = "mean_minimum_gripper_closure"
    closure_values = sorted(
        {float(row["features"][closure_feature]) for row in rows}
    )
    threshold_candidates: list[dict[str, Any]] = [
        {"mode": "accept_all", "threshold": None}
    ]
    threshold_candidates.extend(
        {
            "mode": "threshold",
            "threshold": float((left + right) / 2.0),
        }
        for left, right in zip(closure_values, closure_values[1:])
    )
    threshold_candidates.append({"mode": "accept_none", "threshold": None})
    fixed_scores = [_score_alpha(row, 0.9, "future") for row in rows]
    fixed_track = np.asarray([item["track_rmse_m"] for item in fixed_scores])
    fixed_chamfer = np.asarray([item["chamfer_m"] for item in fixed_scores])
    persistence_track = np.asarray(
        [item["persistence_track_rmse_m"] for item in fixed_scores]
    )
    persistence_chamfer = np.asarray(
        [item["persistence_chamfer_m"] for item in fixed_scores]
    )
    closure_scores = []
    for candidate in threshold_candidates:
        if candidate["mode"] == "accept_all":
            accepted = np.ones(len(rows), dtype=bool)
        elif candidate["mode"] == "accept_none":
            accepted = np.zeros(len(rows), dtype=bool)
        else:
            accepted = np.asarray(
                [
                    row["features"][closure_feature] >= candidate["threshold"]
                    for row in rows
                ],
                dtype=bool,
            )
        selected_track = np.where(accepted, fixed_track, persistence_track)
        selected_chamfer = np.where(accepted, fixed_chamfer, persistence_chamfer)
        relative_score = 0.5 * (
            float(np.mean(selected_track) / np.mean(persistence_track))
            + float(np.mean(selected_chamfer) / np.mean(persistence_chamfer))
        )
        closure_scores.append(
            {
                **candidate,
                "future_relative_score": relative_score,
            }
        )
    selected_closure = min(
        closure_scores,
        key=lambda item: (
            item["future_relative_score"],
            item["mode"],
            -1.0 if item["threshold"] is None else item["threshold"],
        ),
    )

    ridge_scores = []
    for ridge in RIDGE_GRID:
        reference = []
        predicted = []
        for held_index, held_row in enumerate(rows):
            train_indices = [index for index in range(len(rows)) if index != held_index]
            model = _ridge_fit(
                np.stack(
                    [
                        _feature_vector(rows[index], feature_names)
                        for index in train_indices
                    ]
                ),
                np.asarray([rows[index]["oracle_alpha"] for index in train_indices]),
                ridge,
            )
            predicted.append(
                float(
                    _ridge_predict(
                        model, _feature_vector(held_row, feature_names)[None]
                    )[0]
                )
            )
            reference.append(float(held_row["oracle_alpha"]))
        error = np.asarray(predicted) - np.asarray(reference)
        ridge_scores.append(
            {
                "ridge": ridge,
                "leave_one_episode_out_oracle_alpha_mse": float(np.mean(error**2)),
            }
        )
    selected_ridge = min(
        ridge_scores,
        key=lambda item: (
            item["leave_one_episode_out_oracle_alpha_mse"],
            item["ridge"],
        ),
    )["ridge"]
    coefficients, feature_mean, feature_scale = _ridge_fit(
        np.stack([_feature_vector(row, feature_names) for row in rows]),
        np.asarray([row["oracle_alpha"] for row in rows]),
        float(selected_ridge),
    )
    return {
        "artifact_kind": "Deform360ReusableTwinTrustCandidate",
        "schema_version": 1,
        "policy": "closure gate AND simulator-response self-diagnostic",
        "closure_feature": closure_feature,
        "closure_rule": {
            "mode": selected_closure["mode"],
            "threshold": selected_closure["threshold"],
        },
        "reference_response_alpha": 0.9,
        "maximum_alpha": 1.2,
        "feature_names": list(feature_names),
        "ridge": selected_ridge,
        "coefficients": coefficients.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "closure_search": closure_scores,
        "ridge_search": ridge_scores,
        "fit_episode_keys": [row["episode_key"] for row in rows],
        "information_boundary": {
            "fit_uses_exhausted_source_outcomes": True,
            "application_uses_known_robot_action": True,
            "application_uses_frame_zero_geometry": True,
            "application_uses_predicted_simulator_response": True,
            "application_uses_post_initial_object_observation": False,
            "application_uses_tactile": False,
            "application_uses_symbolic_action_label": False,
        },
    }


def main() -> int:
    args = _parse_args()
    diagnosis = _load_json(args.failure_diagnosis)
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, Any] = {}
    for record in diagnosis["episodes"]:
        object_id = str(record["object_id"])
        episode_id = int(record["episode_id"])
        result_dir = _episode_dir(args.result_root, object_id, episode_id)
        stage_dir = _episode_dir(args.stage_root, object_id, episode_id) / "episode_0000"
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
        robot_features = _robot_features(robot_path)
        response = (prediction - persistence) / 0.9
        features = dict(record["action_features"])
        features.update(robot_features)
        features.update(_response_features(response, persistence, robot_features))
        rows.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "episode_key": record["episode_key"],
                "features": features,
                "oracle_alpha": float(record["best_nonnegative_alpha"]),
                "episode": episode,
                "persistence": persistence,
                "response": response,
            }
        )
        input_hashes[str(record["episode_key"])] = {
            "prediction": sha256_file(prediction_path),
            "target": sha256_file(target_path),
            "robot": sha256_file(robot_path),
            "alignment": sha256_file(alignment_path),
        }

    object_ids = sorted({row["object_id"] for row in rows})
    for row in rows:
        for object_id in object_ids:
            row["features"][f"object::{object_id}"] = float(
                row["object_id"] == object_id
            )

    action_names = [
        "mean_closure",
        "mean_minimum_gripper_closure",
        "all_grippers_closed_fraction_075",
        "gripper_count",
        "bimanual",
        "mean_gripper_path_m",
        "max_gripper_path_m",
        "mean_endpoint_displacement_m",
        "mean_vertical_displacement_m",
        "mean_absolute_vertical_displacement_m",
        "mean_horizontal_displacement_m",
        "mean_vertical_path_m",
        "mean_horizontal_path_m",
        "vertical_to_horizontal_path_ratio",
        "mean_opening_start_m",
        "mean_opening_end_m",
        "mean_opening_change_m",
        "minimum_opening_m",
        "gripper_0_delta_x_m",
        "gripper_0_delta_y_m",
        "gripper_0_delta_z_m",
        "gripper_1_delta_x_m",
        "gripper_1_delta_y_m",
        "gripper_1_delta_z_m",
        "gripper_separation_start_m",
        "gripper_separation_end_m",
        "gripper_separation_change_m",
        "gripper_separation_range_m",
        "gripper_delta_dot_m2",
    ]
    response_names = [
        "response_rms_m",
        "response_mean_displacement_m",
        "response_early_mean_displacement_m",
        "response_late_mean_displacement_m",
        "response_endpoint_mean_displacement_m",
        "response_p90_displacement_m",
        "response_max_displacement_m",
        "response_frame_rms_growth_m",
        "response_late_to_early_ratio",
        "response_to_action_path_ratio",
        "response_to_geometry_ratio",
        "geometry_extent_x_m",
        "geometry_extent_y_m",
        "geometry_extent_z_m",
        "geometry_diagonal_m",
        "geometry_pca_0_m",
        "geometry_pca_1_m",
        "geometry_pca_2_m",
    ]
    object_names = [f"object::{object_id}" for object_id in object_ids]
    feature_sets = {
        "same_object_intercept": object_names,
        "action_kinematics": action_names,
        "simulation_self_diagnostic": action_names + response_names,
        "same_object_simulation_self_diagnostic": (
            object_names + action_names + response_names
        ),
    }
    methods = {
        name: _nested_episode_cross_fit(rows, feature_names)
        for name, feature_names in feature_sets.items()
    }
    full_source_candidate = _fit_full_source_candidate(
        rows, feature_sets["simulation_self_diagnostic"]
    )
    full_source_candidate["fit_runtime"] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    full_source_candidate["source_input_sha256"] = {
        "failure_diagnosis": sha256_file(args.failure_diagnosis),
        "episodes": input_hashes,
    }
    candidate_canonical = json.dumps(
        full_source_candidate,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    full_source_candidate["result_sha256"] = hashlib.sha256(
        candidate_canonical
    ).hexdigest()

    fixed_alpha = np.full(len(rows), 0.9)
    oracle_alpha = np.asarray([row["oracle_alpha"] for row in rows])
    object_mean_alpha = _object_mean_cross_fit(rows)
    closure_alpha_by_episode = diagnosis["exploratory_group_cross_fit"][
        "selected_alpha_by_episode"
    ]
    closure_alpha = np.asarray(
        [closure_alpha_by_episode[row["episode_key"]] for row in rows]
    )
    self_diagnostic_alpha_by_episode = methods["simulation_self_diagnostic"][
        "selected_alpha_by_episode"
    ]
    self_diagnostic_alpha = np.asarray(
        [self_diagnostic_alpha_by_episode[row["episode_key"]] for row in rows]
    )
    closure_accepts = closure_alpha > 0.0
    intersected_alpha = np.where(closure_accepts, self_diagnostic_alpha, 0.0)
    conservative_intersected_alpha = np.minimum(intersected_alpha, closure_alpha)
    baselines = {}
    for name, alpha in {
        "persistence": np.zeros(len(rows)),
        "frozen_fixed_response": fixed_alpha,
        "cross_fitted_binary_closure": closure_alpha,
        "closure_then_self_diagnostic": intersected_alpha,
        "conservative_closure_self_intersection": conservative_intersected_alpha,
        "leave_one_episode_out_object_mean": object_mean_alpha,
        "non_deployable_alpha_oracle": oracle_alpha,
    }.items():
        baselines[name] = {
            "future": _aggregate(rows, alpha, "future"),
            "late": _aggregate(rows, alpha, "late"),
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SameObjectTrustDiagnosis",
        "episode_count": len(rows),
        "object_count": len(object_ids),
        "feature_sets": methods,
        "baselines": baselines,
        "full_source_candidate": full_source_candidate,
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
            "outer_cross_fit_unit": "episode",
            "same_object_training_episodes_permitted": True,
            "ridge_selected_by_nested_episode_cross_fit": True,
            "held_episode_simulator_response_used": True,
            "held_episode_object_outcome_used": False,
            "symbolic_action_label_used": False,
            "tactile_used": False,
        },
        "claim_boundary": (
            "post-failure exploratory diagnosis matched to same-object unseen-episode "
            "generalization; any selected feature family requires a fresh protocol lock"
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
    if args.candidate_output is not None:
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(
            json.dumps(
                full_source_candidate, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
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
