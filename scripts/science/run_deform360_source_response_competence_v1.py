#!/usr/bin/env python3
"""Qualify a target-closed Bayesian robot-to-tactile source response model.

The experiment uses one registered Deform360 source episode. It predicts the
next tactile response from tactile state and robot action on a chronological
train/calibration/test split. It compares state-only, action-conditioned
full-covariance, action-conditioned diagonal-covariance, and block-permuted
action controls, plus persistence and last-delta point baselines.

This is a source competence diagnostic. Frames are nested observations, not
independent experimental units, and no target or paper claim is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

Z90 = 1.6448536269514722


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    sine = float(np.sin(angle))
    if abs(sine) < 1e-7:
        values, vectors = np.linalg.eigh((rotation + np.eye(3)) * 0.5)
        axis = vectors[:, int(np.argmax(values))]
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return angle * axis
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    return angle * axis


def _standardize(
    reference: np.ndarray,
    arrays: Iterable[np.ndarray],
    *,
    minimum_scale: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mean = np.mean(reference, axis=0)
    scale = np.maximum(np.std(reference, axis=0), minimum_scale)
    return mean, scale, [(array - mean) / scale for array in arrays]


def _with_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features), dtype=np.float64), features])


def _block_permute(
    values: np.ndarray,
    split_slices: Iterable[slice],
    *,
    block_size: int,
    seed: int,
) -> np.ndarray:
    result = values.copy()
    rng = np.random.default_rng(seed)
    for split in split_slices:
        section = values[split]
        count = len(section)
        if count <= 1:
            continue
        blocks = [
            np.arange(start, min(start + block_size, count), dtype=np.int64)
            for start in range(0, count, block_size)
        ]
        if len(blocks) == 1:
            order = np.roll(np.arange(count), max(1, count // 2))
        else:
            permutation = rng.permutation(len(blocks))
            if np.array_equal(permutation, np.arange(len(blocks))):
                permutation = np.roll(permutation, 1)
            order = np.concatenate([blocks[int(index)] for index in permutation])
        result[split] = section[order]
    return result


def _fit(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    eigenvalue_floor: float,
) -> dict[str, Any]:
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    gram = design.T @ design
    precision = gram + float(ridge) * penalty
    precision += np.eye(precision.shape[0], dtype=np.float64) * 1e-12
    precision_inverse = np.linalg.pinv(precision, rcond=1e-12, hermitian=True)
    coefficients = precision_inverse @ design.T @ targets
    residual = targets - design @ coefficients
    effective_df = float(np.trace(precision_inverse @ gram))
    residual_df = max(float(len(design)) - effective_df, 1.0)
    covariance = residual.T @ residual / residual_df
    covariance = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalue_floor), 1e-12)
    covariance = (vectors * np.maximum(values, floor)) @ vectors.T
    covariance = (covariance + covariance.T) * 0.5
    return {
        "coefficients": coefficients,
        "precision_inverse": precision_inverse,
        "residual_covariance": covariance,
        "ridge": float(ridge),
        "effective_degrees_of_freedom": effective_df,
    }


def _predict(
    model: dict[str, Any],
    design: np.ndarray,
    *,
    covariance_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = design @ model["coefficients"]
    leverage = 1.0 + np.einsum(
        "ni,ij,nj->n",
        design,
        model["precision_inverse"],
        design,
        optimize=True,
    )
    leverage = np.clip(leverage, 1.0, 1e6)
    covariance = (
        leverage[:, None, None]
        * float(covariance_scale)
        * model["residual_covariance"][None, :, :]
    )
    return mean, covariance


def _probabilistic_metrics(
    targets: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    diagonal: bool,
) -> dict[str, float]:
    dimension = targets.shape[1]
    errors = targets - means
    marginal_variances = np.maximum(
        np.diagonal(covariances, axis1=1, axis2=2), 1e-12
    )
    marginal_standard_deviations = np.sqrt(marginal_variances)
    if diagonal:
        squared = np.sum(errors * errors / marginal_variances, axis=1)
        log_determinants = np.sum(np.log(marginal_variances), axis=1)
    else:
        squared_values = []
        log_determinants_values = []
        for error, covariance in zip(errors, covariances, strict=True):
            covariance = (covariance + covariance.T) * 0.5
            try:
                factor = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                factor = np.linalg.cholesky(
                    covariance + np.eye(dimension, dtype=np.float64) * 1e-9
                )
            whitened = np.linalg.solve(factor, error)
            squared_values.append(float(whitened @ whitened))
            log_determinants_values.append(
                float(2.0 * np.log(np.diag(factor)).sum())
            )
        squared = np.asarray(squared_values, dtype=np.float64)
        log_determinants = np.asarray(log_determinants_values, dtype=np.float64)
    nll = 0.5 * (
        dimension * np.log(2.0 * np.pi) + log_determinants + squared
    )
    covered = np.abs(errors) <= Z90 * marginal_standard_deviations
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
        "nll_per_dimension": float(np.mean(nll) / dimension),
        "normalized_joint_nees": float(np.mean(squared) / dimension),
        "marginal_90_coverage": float(np.mean(covered)),
        "mean_marginal_90_interval_width": float(
            np.mean(2.0 * Z90 * marginal_standard_deviations)
        ),
    }


def _point_metrics(targets: np.ndarray, means: np.ndarray) -> dict[str, float]:
    errors = targets - means
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
    }


def _tune(
    train_design: np.ndarray,
    train_targets: np.ndarray,
    calibration_design: np.ndarray,
    calibration_targets: np.ndarray,
    *,
    ridge_grid: Iterable[float],
    covariance_scale_grid: Iterable[float],
    eigenvalue_floor: float,
    diagonal: bool,
) -> tuple[dict[str, Any], float, dict[str, float]]:
    best_key: tuple[float, float, float] | None = None
    best_value: tuple[dict[str, Any], float, dict[str, float]] | None = None
    for ridge in ridge_grid:
        model = _fit(
            train_design,
            train_targets,
            ridge=float(ridge),
            eigenvalue_floor=eigenvalue_floor,
        )
        for scale in covariance_scale_grid:
            means, covariances = _predict(
                model, calibration_design, covariance_scale=float(scale)
            )
            metrics = _probabilistic_metrics(
                calibration_targets,
                means,
                covariances,
                diagonal=diagonal,
            )
            key = (
                metrics["nll_per_dimension"],
                float(ridge),
                float(scale),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_value = (model, float(scale), metrics)
    if best_value is None:
        raise RuntimeError("empty tuning grid")
    return best_value


def _robot_features(
    transforms: np.ndarray,
    openings: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    frame_count, gripper_count = transforms.shape[:2]
    blocks = []
    names = []
    for gripper in range(gripper_count):
        positions = transforms[:, gripper, :3, 3]
        rotations = transforms[:, gripper, :3, :3]
        delta_position = np.diff(positions, axis=0)
        delta_rotation = np.empty((frame_count - 1, 3), dtype=np.float64)
        for frame in range(frame_count - 1):
            delta_rotation[frame] = _rotation_log(
                rotations[frame].T @ rotations[frame + 1]
            )
        delta_opening = np.diff(openings[:, gripper])[:, None]
        current_opening = openings[:-1, gripper, None]
        blocks.extend(
            [delta_position, delta_rotation, delta_opening, current_opening]
        )
        prefix = f"gripper_{gripper}"
        names.extend(
            [
                f"{prefix}_delta_x_m",
                f"{prefix}_delta_y_m",
                f"{prefix}_delta_z_m",
                f"{prefix}_delta_rot_x_rad",
                f"{prefix}_delta_rot_y_rad",
                f"{prefix}_delta_rot_z_rad",
                f"{prefix}_delta_opening_m",
                f"{prefix}_opening_m",
            ]
        )
    return np.concatenate(blocks, axis=1), names


def _tactile_features(
    root: Path,
    sensor_names: Iterable[str],
) -> tuple[np.ndarray, list[str], np.ndarray, list[dict[str, Any]], list[Path]]:
    row_coordinates = np.linspace(-1.0, 1.0, 16)[:, None]
    column_coordinates = np.linspace(-1.0, 1.0, 32)[None, :]
    blocks = []
    names = []
    total_signal = None
    records = []
    paths = []
    frame_count = None
    for sensor in sensor_names:
        path = root / sensor / "synced_tactile.npy"
        raw = np.load(path, allow_pickle=False, mmap_mode="r")
        if raw.ndim != 3 or raw.shape[1:] != (16, 32):
            raise ValueError(f"unexpected tactile shape for {sensor}: {raw.shape}")
        if frame_count is None:
            frame_count = int(raw.shape[0])
        elif int(raw.shape[0]) != frame_count:
            raise ValueError("tactile frame counts disagree")
        positive = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
        signal = positive.sum(axis=(1, 2))
        active = np.count_nonzero(positive > 0.0, axis=(1, 2))
        denominator = np.maximum(signal, np.finfo(np.float64).tiny)
        centroid_x = (positive * column_coordinates).sum(axis=(1, 2)) / denominator
        centroid_y = (positive * row_coordinates).sum(axis=(1, 2)) / denominator
        inactive = signal <= 0.0
        centroid_x[inactive] = 0.0
        centroid_y[inactive] = 0.0
        blocks.append(
            np.column_stack(
                [
                    np.log1p(signal),
                    active / float(16 * 32),
                    centroid_x,
                    centroid_y,
                ]
            )
        )
        names.extend(
            [
                f"{sensor}:log_total_positive_signal",
                f"{sensor}:active_taxel_fraction",
                f"{sensor}:weighted_centroid_x",
                f"{sensor}:weighted_centroid_y",
            ]
        )
        total_signal = signal if total_signal is None else total_signal + signal
        records.append(
            {
                "sensor": sensor,
                "shape": [int(value) for value in raw.shape],
                "dtype": str(raw.dtype),
                "positive_frame_count": int(np.count_nonzero(signal > 0.0)),
                "maximum_positive_signal": float(np.max(signal)),
            }
        )
        paths.append(path)
    if total_signal is None:
        raise ValueError("no tactile sensors configured")
    return np.concatenate(blocks, axis=1), names, total_signal, records, paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workflow-run-id", required=True, type=int)
    parser.add_argument("--workflow-run-attempt", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.source_episode_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    observed_object = root.parent.name
    observed_episode = int(root.name.split("_")[-1])
    expected = protocol["source_object"], int(protocol["source_episode"])
    if (observed_object, observed_episode) != expected:
        raise ValueError("source episode identity differs from protocol")

    robot_path = root / "robot" / "robot.npz"
    with np.load(robot_path, allow_pickle=False) as robot:
        transforms = np.asarray(robot["T_worlds"], dtype=np.float64)
        openings = np.asarray(robot["openings"], dtype=np.float64)
        bimanual = bool(np.asarray(robot["bimanual"]).item())
        robot_format = int(np.asarray(robot["format_version"]).item())
    if transforms.ndim == 3:
        transforms = transforms[:, None, :, :]
        openings = openings[:, None]
    if transforms.ndim != 4 or transforms.shape[-2:] != (4, 4):
        raise ValueError(f"unexpected T_worlds shape: {transforms.shape}")
    if openings.shape != transforms.shape[:2]:
        raise ValueError("openings shape disagrees with T_worlds")
    if not np.isfinite(transforms).all() or not np.isfinite(openings).all():
        raise ValueError("robot state contains non-finite values")

    robot_features, robot_names = _robot_features(transforms, openings)
    (
        tactile_features,
        tactile_names,
        total_signal,
        tactile_records,
        tactile_paths,
    ) = _tactile_features(
        root, protocol["response_contract"]["tactile_sensors"]
    )
    frame_count = tactile_features.shape[0]
    if transforms.shape[0] != frame_count:
        raise ValueError("robot and tactile frame counts differ")

    contact_indices = np.flatnonzero(total_signal > 0.0)
    if len(contact_indices) == 0:
        raise ValueError("source episode contains no positive tactile signal")
    margin = int(protocol["design"]["contact_margin_frames"])
    first_transition = max(1, int(contact_indices[0]) - margin)
    last_transition = min(frame_count - 2, int(contact_indices[-1]) + margin)
    transition_frames = np.arange(
        first_transition, last_transition + 1, dtype=np.int64
    )
    sample_count = len(transition_frames)
    train_count = int(np.floor(sample_count * protocol["design"]["train_fraction"]))
    calibration_count = int(
        np.floor(sample_count * protocol["design"]["calibration_fraction"])
    )
    test_count = sample_count - train_count - calibration_count
    if min(train_count, calibration_count, test_count) <= 0:
        raise ValueError("chronological split is empty")
    train = slice(0, train_count)
    calibration = slice(train_count, train_count + calibration_count)
    test = slice(train_count + calibration_count, sample_count)

    current_raw = tactile_features[transition_frames]
    previous_raw = tactile_features[transition_frames - 1]
    targets_raw = tactile_features[transition_frames + 1]
    minimum_output_std = float(
        protocol["response_contract"]["minimum_output_train_std"]
    )
    active_outputs = np.std(targets_raw[train], axis=0) >= minimum_output_std
    if int(np.count_nonzero(active_outputs)) < 2:
        raise ValueError("fewer than two nondegenerate tactile outputs")
    selected_names = [
        name
        for name, active in zip(tactile_names, active_outputs, strict=True)
        if active
    ]
    current_raw = current_raw[:, active_outputs]
    previous_raw = previous_raw[:, active_outputs]
    targets_raw = targets_raw[:, active_outputs]

    target_mean, target_scale, transformed = _standardize(
        targets_raw[train],
        [targets_raw, current_raw, previous_raw],
        minimum_scale=minimum_output_std,
    )
    targets, current, previous = transformed
    state_raw = np.concatenate([current, current - previous], axis=1)
    state_mean, state_scale, [state_features] = _standardize(
        state_raw[train], [state_raw]
    )
    action_raw = robot_features[transition_frames]
    action_mean, action_scale, [action_features] = _standardize(
        action_raw[train], [action_raw]
    )
    state_design = _with_intercept(state_features)
    action_design = _with_intercept(
        np.concatenate([state_features, action_features], axis=1)
    )
    permuted_actions = _block_permute(
        action_features,
        [train, calibration, test],
        block_size=int(protocol["design"]["permutation_block_frames"]),
        seed=int(protocol["design"]["permutation_seed"]),
    )
    permuted_design = _with_intercept(
        np.concatenate([state_features, permuted_actions], axis=1)
    )

    tune_options = {
        "ridge_grid": protocol["design"]["ridge_grid"],
        "covariance_scale_grid": protocol["design"]["covariance_scale_grid"],
        "eigenvalue_floor": float(
            protocol["design"]["covariance_eigenvalue_floor"]
        ),
    }
    specifications = {
        "state_only_bayesian_full_covariance": (state_design, False),
        "action_conditioned_bayesian_diagonal_covariance": (action_design, True),
        "action_conditioned_bayesian_full_covariance": (action_design, False),
        "action_block_permuted_bayesian_full_covariance": (
            permuted_design,
            False,
        ),
    }
    methods: dict[str, Any] = {
        "persistence": _point_metrics(targets[test], current[test]),
        "last_delta": _point_metrics(
            targets[test], current[test] + current[test] - previous[test]
        ),
    }
    calibration_selection = {}
    for name, (design, diagonal) in specifications.items():
        model, covariance_scale, calibration_metrics = _tune(
            design[train],
            targets[train],
            design[calibration],
            targets[calibration],
            diagonal=diagonal,
            **tune_options,
        )
        means, covariances = _predict(
            model, design[test], covariance_scale=covariance_scale
        )
        metrics = _probabilistic_metrics(
            targets[test], means, covariances, diagonal=diagonal
        )
        metrics["ridge"] = float(model["ridge"])
        metrics["covariance_scale"] = float(covariance_scale)
        metrics["effective_degrees_of_freedom"] = float(
            model["effective_degrees_of_freedom"]
        )
        methods[name] = metrics
        calibration_selection[name] = calibration_metrics

    state_metrics = methods["state_only_bayesian_full_covariance"]
    diagonal_metrics = methods[
        "action_conditioned_bayesian_diagonal_covariance"
    ]
    action_metrics = methods["action_conditioned_bayesian_full_covariance"]
    permuted_metrics = methods[
        "action_block_permuted_bayesian_full_covariance"
    ]
    gains = {
        "action_vs_state_nll_per_dimension": float(
            state_metrics["nll_per_dimension"]
            - action_metrics["nll_per_dimension"]
        ),
        "full_vs_diagonal_nll_per_dimension": float(
            diagonal_metrics["nll_per_dimension"]
            - action_metrics["nll_per_dimension"]
        ),
        "true_vs_block_permuted_action_nll_per_dimension": float(
            permuted_metrics["nll_per_dimension"]
            - action_metrics["nll_per_dimension"]
        ),
        "action_vs_persistence_rmse": float(
            methods["persistence"]["rmse"] - action_metrics["rmse"]
        ),
        "action_rmse_ratio_to_persistence": float(
            action_metrics["rmse"]
            / max(methods["persistence"]["rmse"], 1e-12)
        ),
    }

    gates = protocol["qualification_gates"]
    checks = {
        "minimum_train_transitions": train_count
        >= int(gates["minimum_train_transitions"]),
        "minimum_calibration_transitions": calibration_count
        >= int(gates["minimum_calibration_transitions"]),
        "minimum_test_transitions": test_count
        >= int(gates["minimum_test_transitions"]),
        "action_beats_state_nll": gains["action_vs_state_nll_per_dimension"]
        >= float(gates["minimum_action_vs_state_nll_gain_per_dimension"]),
        "full_covariance_beats_diagonal_nll": gains[
            "full_vs_diagonal_nll_per_dimension"
        ]
        >= float(gates["minimum_full_vs_diagonal_nll_gain_per_dimension"]),
        "true_pairing_beats_block_permutation_nll": gains[
            "true_vs_block_permuted_action_nll_per_dimension"
        ]
        >= float(gates["minimum_true_vs_permuted_action_nll_gain_per_dimension"]),
        "action_mean_not_materially_worse_than_persistence": gains[
            "action_rmse_ratio_to_persistence"
        ]
        <= float(gates["maximum_action_rmse_ratio_to_persistence"]),
        "normalized_joint_nees_in_range": float(
            gates["minimum_normalized_joint_nees"]
        )
        <= action_metrics["normalized_joint_nees"]
        <= float(gates["maximum_normalized_joint_nees"]),
        "marginal_coverage_in_range": float(gates["minimum_marginal_coverage"])
        <= action_metrics["marginal_90_coverage"]
        <= float(gates["maximum_marginal_coverage"]),
    }
    qualified = all(checks.values())

    input_hashes = {"robot/robot.npz": _sha256(robot_path)}
    for path in tactile_paths:
        input_hashes[str(path.relative_to(root))] = _sha256(path)
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-source-response-competence-v1",
        "repository": args.repository,
        "revision": args.revision,
        "workflow_run_id": args.workflow_run_id,
        "workflow_run_attempt": args.workflow_run_attempt,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "required_runner_label": "gpuserver4090",
        "source_object": protocol["source_object"],
        "source_episode": int(protocol["source_episode"]),
        "source_episode_root": str(root),
        "protocol_sha256": _sha256(protocol_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "input_sha256": input_hashes,
        "robot": {
            "format_version": robot_format,
            "bimanual": bimanual,
            "frame_count": frame_count,
            "gripper_count": int(transforms.shape[1]),
            "transition_feature_names": robot_names,
            "transition_feature_dimension": len(robot_names),
        },
        "tactile": {
            "sensor_records": tactile_records,
            "configured_output_names": tactile_names,
            "selected_output_names": selected_names,
            "selected_output_dimension": len(selected_names),
            "contact_frame_count": int(np.count_nonzero(total_signal > 0.0)),
            "contact_fraction": float(np.mean(total_signal > 0.0)),
        },
        "chronological_split": {
            "first_transition_frame": int(transition_frames[0]),
            "last_transition_frame": int(transition_frames[-1]),
            "sample_count": sample_count,
            "train_count": train_count,
            "calibration_count": calibration_count,
            "test_count": test_count,
        },
        "calibration_selection": calibration_selection,
        "test_methods": methods,
        "test_gains": gains,
        "qualification_checks": checks,
        "source_response_competence_qualified": qualified,
        "decision": (
            "source-response-competence-qualified"
            if qualified
            else "source-response-competence-not-qualified"
        ),
        "standardization": {
            "target_mean": target_mean.astype(float).tolist(),
            "target_scale": target_scale.astype(float).tolist(),
            "state_feature_mean": state_mean.astype(float).tolist(),
            "state_feature_scale": state_scale.astype(float).tolist(),
            "action_feature_mean": action_mean.astype(float).tolist(),
            "action_feature_scale": action_scale.astype(float).tolist(),
        },
        "information_boundary": {
            "source_robot_payload_opened": True,
            "source_tactile_payloads_opened": True,
            "source_camera_pixels_opened": False,
            "target_directory_contents_listed": False,
            "target_numeric_payload_opened": False,
            "target_scoring_performed": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": False,
        },
        "statistical_scope": protocol["statistical_scope"],
        "claim_boundary": protocol["claim_boundary"],
    }
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    _write_json(output / "result.json", result)
    (output / "report.md").write_text(
        "# Deform360 source-response competence v1\n\n"
        f"Decision: `{result['decision']}`\n\n"
        f"Chronological transitions: `{train_count} / {calibration_count} / "
        f"{test_count}` train/calibration/test\n\n"
        f"Selected tactile response dimension: `{len(selected_names)}`\n\n"
        "## Test gains (positive is favorable)\n\n"
        f"- action vs state NLL/dim: "
        f"`{gains['action_vs_state_nll_per_dimension']:.6f}`\n"
        f"- full vs diagonal NLL/dim: "
        f"`{gains['full_vs_diagonal_nll_per_dimension']:.6f}`\n"
        f"- true vs block-permuted action NLL/dim: "
        f"`{gains['true_vs_block_permuted_action_nll_per_dimension']:.6f}`\n"
        f"- action vs persistence RMSE: "
        f"`{gains['action_vs_persistence_rmse']:.6f}`\n\n"
        f"Normalized joint NEES: "
        f"`{action_metrics['normalized_joint_nees']:.6f}`\n\n"
        f"Marginal 90% coverage: "
        f"`{action_metrics['marginal_90_coverage']:.6f}`\n\n"
        "Frames are nested observations from one source episode; this is not "
        "an independent-object result or paper claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
