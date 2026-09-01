#!/usr/bin/env python3
"""Run a target-closed Deform360 tactile-innovation competence gate.

The registered source episode is transformed with a train-only robust tactile
baseline. The model predicts baseline-corrected tactile feature increments over
one of several preregistered horizons. It tests whether causal multiscale robot
action histories and a state-gated action exposure improve held-out predictive
log likelihood over state-only, ungated, diagonal-covariance, and block-
permuted controls.

Frames are dependent observations nested within one source episode. No target
access, cross-object claim, or paper claim is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

Z90 = 1.6448536269514722
MAD_NORMALIZER = 1.482602218505602


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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
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


def _fit(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    eigenvalue_floor: float,
) -> dict[str, Any]:
    if design.ndim != 2 or targets.ndim != 2 or len(design) != len(targets):
        raise ValueError("invalid regression arrays")
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
    model: Mapping[str, Any],
    design: np.ndarray,
    *,
    covariance_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = design @ np.asarray(model["coefficients"])
    precision_inverse = np.asarray(model["precision_inverse"])
    leverage = 1.0 + np.einsum(
        "ni,ij,nj->n", design, precision_inverse, design, optimize=True
    )
    leverage = np.clip(leverage, 1.0, 1e6)
    covariance = (
        leverage[:, None, None]
        * float(covariance_scale)
        * np.asarray(model["residual_covariance"])[None, :, :]
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
        log_determinant_values = []
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
            log_determinant_values.append(
                float(2.0 * np.log(np.diag(factor)).sum())
            )
        squared = np.asarray(squared_values, dtype=np.float64)
        log_determinants = np.asarray(
            log_determinant_values, dtype=np.float64
        )
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


def _block_permute(
    values: np.ndarray,
    split_slices: Sequence[slice],
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


def _robust_dynamic_tactile_features(
    values_by_sensor: Mapping[str, np.ndarray],
    *,
    baseline_frame_count: int,
    mad_threshold: float,
    minimum_noise_scale: float,
    maximum_standardized_residual: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """Create preload-removed activity features from a train-only prefix."""
    rows = np.linspace(-1.0, 1.0, 16, dtype=np.float64)[:, None]
    columns = np.linspace(-1.0, 1.0, 32, dtype=np.float64)[None, :]
    feature_blocks = []
    names: list[str] = []
    total_activity = None
    records = []
    frame_count = None
    for sensor in sorted(values_by_sensor):
        values = np.asarray(values_by_sensor[sensor], dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (16, 32):
            raise ValueError(f"unexpected tactile shape for {sensor}: {values.shape}")
        if frame_count is None:
            frame_count = int(values.shape[0])
        elif int(values.shape[0]) != frame_count:
            raise ValueError("tactile frame counts disagree")
        if not 4 <= baseline_frame_count < values.shape[0]:
            raise ValueError("baseline frame count is outside the episode")
        prefix = values[:baseline_frame_count]
        baseline = np.median(prefix, axis=0)
        mad = MAD_NORMALIZER * np.median(
            np.abs(prefix - baseline[None, :, :]), axis=0
        )
        positive_mad = mad[mad > minimum_noise_scale]
        adaptive_floor = (
            float(np.percentile(positive_mad, 10.0)) * 0.25
            if len(positive_mad)
            else minimum_noise_scale
        )
        sensor_floor = max(float(minimum_noise_scale), adaptive_floor)
        scale = np.maximum(mad, sensor_floor)
        standardized = np.clip(
            (values - baseline[None, :, :]) / scale[None, :, :],
            -float(maximum_standardized_residual),
            float(maximum_standardized_residual),
        )
        activity = np.maximum(np.abs(standardized) - float(mad_threshold), 0.0)
        energy = np.sum(activity, axis=(1, 2))
        active_fraction = np.mean(activity > 0.0, axis=(1, 2))
        denominator = np.maximum(energy, np.finfo(np.float64).tiny)
        centroid_x = np.sum(activity * columns, axis=(1, 2)) / denominator
        centroid_y = np.sum(activity * rows, axis=(1, 2)) / denominator
        inactive = energy <= 0.0
        centroid_x[inactive] = 0.0
        centroid_y[inactive] = 0.0
        feature_blocks.append(
            np.column_stack(
                [
                    np.log1p(energy),
                    active_fraction,
                    centroid_x,
                    centroid_y,
                ]
            )
        )
        names.extend(
            [
                f"{sensor}:log_dynamic_energy",
                f"{sensor}:dynamic_active_taxel_fraction",
                f"{sensor}:dynamic_centroid_x",
                f"{sensor}:dynamic_centroid_y",
            ]
        )
        total_activity = energy if total_activity is None else total_activity + energy
        records.append(
            {
                "sensor": sensor,
                "shape": [int(value) for value in values.shape],
                "baseline_frame_count": int(baseline_frame_count),
                "baseline_sha256": _array_sha256(baseline),
                "noise_scale_sha256": _array_sha256(scale),
                "median_baseline": float(np.median(baseline)),
                "median_noise_scale": float(np.median(scale)),
                "sensor_noise_floor": float(sensor_floor),
                "dynamic_frame_count": int(np.count_nonzero(energy > 0.0)),
                "dynamic_frame_fraction": float(np.mean(energy > 0.0)),
                "maximum_dynamic_energy": float(np.max(energy)),
            }
        )
    if total_activity is None:
        raise ValueError("no tactile sensors configured")
    return (
        np.concatenate(feature_blocks, axis=1),
        np.asarray(total_activity, dtype=np.float64),
        names,
        records,
    )


def _robot_primitives(
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
        blocks.extend([delta_position, delta_rotation, delta_opening])
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
            ]
        )
    return np.concatenate(blocks, axis=1), names


def _action_history_features(
    primitives: np.ndarray,
    openings: np.ndarray,
    times: np.ndarray,
    *,
    horizon: int,
    history_windows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return action context and the horizon exposure used by the gate."""
    primitive_dimension = primitives.shape[1]
    exposure = np.stack(
        [np.sum(primitives[t : t + horizon], axis=0) for t in times]
    )
    context_blocks = [exposure]
    context_names = [
        f"horizon_exposure:{index}" for index in range(primitive_dimension)
    ]
    for window in history_windows:
        block = np.stack(
            [np.mean(primitives[t - window : t], axis=0) for t in times]
        )
        context_blocks.append(block)
        context_names.extend(
            [
                f"past_mean_w{window}:{index}"
                for index in range(primitive_dimension)
            ]
        )
    current_opening = openings[times]
    context_blocks.append(current_opening)
    context_names.extend(
        [f"current_opening:{index}" for index in range(openings.shape[1])]
    )
    exposure_names = [
        f"gated_horizon_exposure:{index}"
        for index in range(primitive_dimension)
    ]
    return (
        np.concatenate(context_blocks, axis=1),
        exposure,
        context_names,
        exposure_names,
    )


def _active_output_mask(
    tactile_features: np.ndarray,
    *,
    nominal_train_end_frame: int,
    history_start_frame: int,
    minimum_increment_std: float,
) -> np.ndarray:
    increments = np.diff(tactile_features, axis=0)
    start = max(int(history_start_frame), 1)
    stop = min(int(nominal_train_end_frame), tactile_features.shape[0]) - 1
    if stop <= start:
        raise ValueError("no train-only increments available for output selection")
    scale = np.std(increments[start:stop], axis=0)
    return scale >= float(minimum_increment_std)


def _build_horizon_dataset(
    tactile_features: np.ndarray,
    total_activity: np.ndarray,
    robot_primitives: np.ndarray,
    openings: np.ndarray,
    active_outputs: np.ndarray,
    *,
    horizon: int,
    history_windows: Sequence[int],
    train_fraction: float,
    calibration_fraction: float,
    minimum_scale: float,
    block_size: int,
    permutation_seed: int,
) -> dict[str, Any]:
    maximum_history = max(history_windows)
    start = max(2, maximum_history)
    times = np.arange(start, tactile_features.shape[0] - horizon, dtype=np.int64)
    sample_count = len(times)
    train_count = int(np.floor(sample_count * train_fraction))
    calibration_count = int(np.floor(sample_count * calibration_fraction))
    test_count = sample_count - train_count - calibration_count
    if min(train_count, calibration_count, test_count) <= 0:
        raise ValueError("chronological split is empty")
    train = slice(0, train_count)
    calibration = slice(train_count, train_count + calibration_count)
    test = slice(train_count + calibration_count, sample_count)

    selected = tactile_features[:, active_outputs]
    target_raw = selected[times + horizon] - selected[times]
    target_mean, target_scale, [targets] = _standardize(
        target_raw[train], [target_raw], minimum_scale=minimum_scale
    )
    current = selected[times]
    previous_increment = selected[times] - selected[times - 1]
    older_increment = selected[times - 1] - selected[times - 2]
    state_raw = np.concatenate(
        [current, previous_increment, older_increment], axis=1
    )
    state_mean, state_scale, [state_features] = _standardize(
        state_raw[train], [state_raw], minimum_scale=minimum_scale
    )

    action_raw, exposure_raw, action_names, exposure_names = (
        _action_history_features(
            robot_primitives,
            openings,
            times,
            horizon=horizon,
            history_windows=history_windows,
        )
    )
    action_mean, action_scale, [action_features] = _standardize(
        action_raw[train], [action_raw], minimum_scale=minimum_scale
    )
    exposure_dimension = exposure_raw.shape[1]
    exposure_features = action_features[:, :exposure_dimension]

    gate_raw = np.log1p(np.maximum(total_activity[times], 0.0))
    gate_reference = float(np.percentile(gate_raw[train], 90.0))
    gate_scale = max(gate_reference, minimum_scale)
    gate = np.clip(gate_raw / gate_scale, 0.0, 2.0)
    gated_exposure = exposure_features * gate[:, None]

    state_design = _with_intercept(state_features)
    action_design = _with_intercept(
        np.concatenate([state_features, action_features], axis=1)
    )
    gated_design = _with_intercept(
        np.concatenate(
            [
                state_features,
                action_features,
                gate[:, None],
                gated_exposure,
            ],
            axis=1,
        )
    )
    permuted_action = _block_permute(
        action_features,
        [train, calibration, test],
        block_size=block_size,
        seed=permutation_seed + 1009 * horizon,
    )
    permuted_exposure = permuted_action[:, :exposure_dimension]
    permuted_gated_design = _with_intercept(
        np.concatenate(
            [
                state_features,
                permuted_action,
                gate[:, None],
                permuted_exposure * gate[:, None],
            ],
            axis=1,
        )
    )

    zero_raw = np.zeros_like(target_raw)
    last_raw = float(horizon) * previous_increment
    zero_prediction = (zero_raw - target_mean) / target_scale
    last_prediction = (last_raw - target_mean) / target_scale
    return {
        "horizon": int(horizon),
        "times": times,
        "sample_count": sample_count,
        "train_count": train_count,
        "calibration_count": calibration_count,
        "test_count": test_count,
        "train": train,
        "calibration": calibration,
        "test": test,
        "targets": targets,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "state_mean": state_mean,
        "state_scale": state_scale,
        "action_mean": action_mean,
        "action_scale": action_scale,
        "gate_scale": gate_scale,
        "gate": gate,
        "state_design": state_design,
        "action_design": action_design,
        "gated_design": gated_design,
        "permuted_gated_design": permuted_gated_design,
        "zero_prediction": zero_prediction,
        "last_prediction": last_prediction,
        "action_feature_names": action_names,
        "gated_exposure_feature_names": exposure_names,
    }


def _evaluate_model(
    dataset: Mapping[str, Any],
    design_key: str,
    *,
    diagonal: bool,
    tune_options: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    design = np.asarray(dataset[design_key])
    targets = np.asarray(dataset["targets"])
    train = dataset["train"]
    calibration = dataset["calibration"]
    test = dataset["test"]
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
    return metrics, calibration_metrics


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

    observed = root.parent.name, int(root.name.split("_")[-1])
    expected = protocol["source_object"], int(protocol["source_episode"])
    if observed != expected:
        raise ValueError(f"source episode {observed} differs from {expected}")

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
    frame_count = int(transforms.shape[0])

    sensor_names = list(protocol["tactile_preprocessing"]["sensors"])
    tactile_values: dict[str, np.ndarray] = {}
    input_hashes = {"robot/robot.npz": _sha256(robot_path)}
    for sensor in sensor_names:
        path = root / sensor / "synced_tactile.npy"
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        if values.shape[0] != frame_count:
            raise ValueError(f"robot/tactile frame mismatch for {sensor}")
        tactile_values[sensor] = np.asarray(values, dtype=np.float64)
        input_hashes[str(path.relative_to(root))] = _sha256(path)

    train_fraction = float(protocol["design"]["train_fraction"])
    nominal_train_end = int(np.floor(frame_count * train_fraction))
    baseline_fraction = float(
        protocol["tactile_preprocessing"]["baseline_prefix_fraction_of_train"]
    )
    minimum_baseline_frames = int(
        protocol["tactile_preprocessing"]["minimum_baseline_frames"]
    )
    baseline_frame_count = max(
        minimum_baseline_frames,
        int(np.floor(nominal_train_end * baseline_fraction)),
    )
    baseline_frame_count = min(baseline_frame_count, nominal_train_end - 2)
    tactile_features, total_activity, tactile_names, tactile_records = (
        _robust_dynamic_tactile_features(
            tactile_values,
            baseline_frame_count=baseline_frame_count,
            mad_threshold=float(
                protocol["tactile_preprocessing"]["mad_threshold"]
            ),
            minimum_noise_scale=float(
                protocol["tactile_preprocessing"]["minimum_noise_scale"]
            ),
            maximum_standardized_residual=float(
                protocol["tactile_preprocessing"][
                    "maximum_standardized_residual"
                ]
            ),
        )
    )

    robot_primitives, robot_primitive_names = _robot_primitives(
        transforms, openings
    )
    history_windows = [
        int(value) for value in protocol["design"]["history_windows"]
    ]
    active_outputs = _active_output_mask(
        tactile_features,
        nominal_train_end_frame=nominal_train_end,
        history_start_frame=max(history_windows),
        minimum_increment_std=float(
            protocol["tactile_preprocessing"]["minimum_increment_train_std"]
        ),
    )
    selected_names = [
        name
        for name, active in zip(tactile_names, active_outputs, strict=True)
        if active
    ]
    if len(selected_names) < int(
        protocol["qualification_gates"]["minimum_selected_output_dimension"]
    ):
        raise ValueError("too few nondegenerate dynamic tactile outputs")

    tune_options = {
        "ridge_grid": protocol["design"]["ridge_grid"],
        "covariance_scale_grid": protocol["design"]["covariance_scale_grid"],
        "eigenvalue_floor": float(
            protocol["design"]["covariance_eigenvalue_floor"]
        ),
    }
    horizon_records = []
    datasets: dict[int, dict[str, Any]] = {}
    for horizon_value in protocol["design"]["prediction_horizon_grid_frames"]:
        horizon = int(horizon_value)
        dataset = _build_horizon_dataset(
            tactile_features,
            total_activity,
            robot_primitives,
            openings,
            active_outputs,
            horizon=horizon,
            history_windows=history_windows,
            train_fraction=train_fraction,
            calibration_fraction=float(
                protocol["design"]["calibration_fraction"]
            ),
            minimum_scale=float(
                protocol["design"]["standardization_minimum_scale"]
            ),
            block_size=int(protocol["design"]["permutation_block_frames"]),
            permutation_seed=int(protocol["design"]["permutation_seed"]),
        )
        datasets[horizon] = dataset
        _model, _scale, calibration_metrics = _tune(
            dataset["gated_design"][dataset["train"]],
            dataset["targets"][dataset["train"]],
            dataset["gated_design"][dataset["calibration"]],
            dataset["targets"][dataset["calibration"]],
            diagonal=False,
            **tune_options,
        )
        horizon_records.append(
            {
                "horizon_frames": horizon,
                "sample_count": int(dataset["sample_count"]),
                "train_count": int(dataset["train_count"]),
                "calibration_count": int(dataset["calibration_count"]),
                "test_count": int(dataset["test_count"]),
                "gated_full_calibration_metrics": calibration_metrics,
            }
        )
    selected_horizon_record = min(
        horizon_records,
        key=lambda record: (
            record["gated_full_calibration_metrics"]["nll_per_dimension"],
            record["horizon_frames"],
        ),
    )
    selected_horizon = int(selected_horizon_record["horizon_frames"])
    dataset = datasets[selected_horizon]

    specifications = {
        "state_only_innovation_bayesian_full_covariance": (
            "state_design",
            False,
        ),
        "ungated_action_history_bayesian_full_covariance": (
            "action_design",
            False,
        ),
        "state_gated_action_history_bayesian_diagonal_covariance": (
            "gated_design",
            True,
        ),
        "state_gated_action_history_bayesian_full_covariance": (
            "gated_design",
            False,
        ),
        "state_gated_block_permuted_action_bayesian_full_covariance": (
            "permuted_gated_design",
            False,
        ),
    }
    methods: dict[str, Any] = {
        "zero_innovation": _point_metrics(
            dataset["targets"][dataset["test"]],
            dataset["zero_prediction"][dataset["test"]],
        ),
        "last_increment_extrapolation": _point_metrics(
            dataset["targets"][dataset["test"]],
            dataset["last_prediction"][dataset["test"]],
        ),
    }
    calibration_selection = {}
    for name, (design_key, diagonal) in specifications.items():
        metrics, calibration_metrics = _evaluate_model(
            dataset,
            design_key,
            diagonal=diagonal,
            tune_options=tune_options,
        )
        methods[name] = metrics
        calibration_selection[name] = calibration_metrics

    state_metrics = methods[
        "state_only_innovation_bayesian_full_covariance"
    ]
    ungated_metrics = methods[
        "ungated_action_history_bayesian_full_covariance"
    ]
    diagonal_metrics = methods[
        "state_gated_action_history_bayesian_diagonal_covariance"
    ]
    gated_metrics = methods[
        "state_gated_action_history_bayesian_full_covariance"
    ]
    permuted_metrics = methods[
        "state_gated_block_permuted_action_bayesian_full_covariance"
    ]
    gains = {
        "gated_action_vs_state_nll_per_dimension": float(
            state_metrics["nll_per_dimension"]
            - gated_metrics["nll_per_dimension"]
        ),
        "gated_vs_ungated_action_nll_per_dimension": float(
            ungated_metrics["nll_per_dimension"]
            - gated_metrics["nll_per_dimension"]
        ),
        "full_vs_diagonal_nll_per_dimension": float(
            diagonal_metrics["nll_per_dimension"]
            - gated_metrics["nll_per_dimension"]
        ),
        "true_vs_block_permuted_action_nll_per_dimension": float(
            permuted_metrics["nll_per_dimension"]
            - gated_metrics["nll_per_dimension"]
        ),
        "gated_action_vs_zero_innovation_rmse": float(
            methods["zero_innovation"]["rmse"] - gated_metrics["rmse"]
        ),
        "gated_action_rmse_ratio_to_zero_innovation": float(
            gated_metrics["rmse"]
            / max(methods["zero_innovation"]["rmse"], 1e-12)
        ),
    }

    gates = protocol["qualification_gates"]
    dynamic_fraction = float(np.mean(total_activity > 0.0))
    checks = {
        "minimum_selected_output_dimension": len(selected_names)
        >= int(gates["minimum_selected_output_dimension"]),
        "minimum_dynamic_frame_fraction": dynamic_fraction
        >= float(gates["minimum_dynamic_frame_fraction"]),
        "maximum_dynamic_frame_fraction": dynamic_fraction
        <= float(gates["maximum_dynamic_frame_fraction"]),
        "minimum_train_transitions": int(dataset["train_count"])
        >= int(gates["minimum_train_transitions"]),
        "minimum_calibration_transitions": int(dataset["calibration_count"])
        >= int(gates["minimum_calibration_transitions"]),
        "minimum_test_transitions": int(dataset["test_count"])
        >= int(gates["minimum_test_transitions"]),
        "gated_action_beats_state_nll": gains[
            "gated_action_vs_state_nll_per_dimension"
        ]
        >= float(
            gates["minimum_gated_action_vs_state_nll_gain_per_dimension"]
        ),
        "state_gating_beats_ungated_action_nll": gains[
            "gated_vs_ungated_action_nll_per_dimension"
        ]
        >= float(gates["minimum_gated_vs_ungated_nll_gain_per_dimension"]),
        "full_covariance_beats_diagonal_nll": gains[
            "full_vs_diagonal_nll_per_dimension"
        ]
        >= float(gates["minimum_full_vs_diagonal_nll_gain_per_dimension"]),
        "true_pairing_beats_block_permutation_nll": gains[
            "true_vs_block_permuted_action_nll_per_dimension"
        ]
        >= float(
            gates["minimum_true_vs_permuted_action_nll_gain_per_dimension"]
        ),
        "gated_mean_not_materially_worse_than_zero_innovation": gains[
            "gated_action_rmse_ratio_to_zero_innovation"
        ]
        <= float(gates["maximum_gated_rmse_ratio_to_zero_innovation"]),
        "normalized_joint_nees_in_range": float(
            gates["minimum_normalized_joint_nees"]
        )
        <= gated_metrics["normalized_joint_nees"]
        <= float(gates["maximum_normalized_joint_nees"]),
        "marginal_coverage_in_range": float(
            gates["minimum_marginal_coverage"]
        )
        <= gated_metrics["marginal_90_coverage"]
        <= float(gates["maximum_marginal_coverage"]),
    }
    qualified = all(checks.values())

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-source-response-innovation-v2",
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
            "primitive_feature_names": robot_primitive_names,
            "primitive_feature_dimension": len(robot_primitive_names),
        },
        "tactile_preprocessing": {
            "baseline_frame_count": baseline_frame_count,
            "nominal_train_end_frame": nominal_train_end,
            "configured_feature_names": tactile_names,
            "selected_feature_names": selected_names,
            "selected_output_dimension": len(selected_names),
            "dynamic_frame_count": int(
                np.count_nonzero(total_activity > 0.0)
            ),
            "dynamic_frame_fraction": dynamic_fraction,
            "total_activity_sha256": _array_sha256(total_activity),
            "sensor_records": tactile_records,
        },
        "horizon_selection": {
            "selection_rule": (
                "minimum gated-full calibration NLL per dimension; lower "
                "horizon breaks exact ties"
            ),
            "candidate_records": horizon_records,
            "selected_horizon_frames": selected_horizon,
        },
        "chronological_split": {
            "sample_count": int(dataset["sample_count"]),
            "train_count": int(dataset["train_count"]),
            "calibration_count": int(dataset["calibration_count"]),
            "test_count": int(dataset["test_count"]),
            "first_state_frame": int(dataset["times"][0]),
            "last_state_frame": int(dataset["times"][-1]),
            "first_test_state_frame": int(
                dataset["times"][dataset["test"].start]
            ),
        },
        "feature_contract": {
            "history_windows_frames": history_windows,
            "action_feature_names": dataset["action_feature_names"],
            "gated_exposure_feature_names": dataset[
                "gated_exposure_feature_names"
            ],
            "gate_scale_train_p90_log_dynamic_activity": float(
                dataset["gate_scale"]
            ),
        },
        "calibration_selection": calibration_selection,
        "test_methods": methods,
        "test_gains": gains,
        "qualification_checks": checks,
        "source_response_innovation_qualified": qualified,
        "decision": (
            "source-response-innovation-qualified"
            if qualified
            else "source-response-innovation-not-qualified"
        ),
        "information_boundary": {
            "source_robot_payload_opened": True,
            "source_tactile_payloads_opened": True,
            "source_camera_pixels_opened": False,
            "persistent_dataset_write_performed": False,
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
        "# Deform360 source tactile-innovation response v2\n\n"
        f"Decision: `{result['decision']}`\n\n"
        f"Selected horizon: `{selected_horizon}` frames\n\n"
        f"Selected output dimension: `{len(selected_names)}`\n\n"
        f"Dynamic-frame fraction after train-only baseline removal: "
        f"`{dynamic_fraction:.6f}`\n\n"
        "## Test gains (positive is favorable)\n\n"
        f"- gated action vs state NLL/dim: "
        f"`{gains['gated_action_vs_state_nll_per_dimension']:.6f}`\n"
        f"- gated vs ungated action NLL/dim: "
        f"`{gains['gated_vs_ungated_action_nll_per_dimension']:.6f}`\n"
        f"- full vs diagonal NLL/dim: "
        f"`{gains['full_vs_diagonal_nll_per_dimension']:.6f}`\n"
        f"- true vs block-permuted action NLL/dim: "
        f"`{gains['true_vs_block_permuted_action_nll_per_dimension']:.6f}`\n"
        f"- gated action vs zero-innovation RMSE: "
        f"`{gains['gated_action_vs_zero_innovation_rmse']:.6f}`\n\n"
        f"Normalized joint NEES: "
        f"`{gated_metrics['normalized_joint_nees']:.6f}`\n\n"
        f"Marginal 90% coverage: "
        f"`{gated_metrics['marginal_90_coverage']:.6f}`\n\n"
        "Frames are nested observations from one source episode. This does "
        "not establish target benefit, object-level generalization, or a "
        "paper claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
